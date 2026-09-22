from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import polars as pl

from backend.app.api.routes import BacktestReq
from backend.app.backtest.engine import (
    MULTI_FACTOR_BACKTEST_PROTOCOL,
    _weighted_sleeve_specs,
    run_multi_factor_backtest,
)


def _fake_single_result(*, expression: str, direction: int, initial_capital: float, market: str, mode: str, **kwargs):
    del kwargs
    days = 60
    daily_gain = (0.001 if expression == "alpha_a" else -0.00025) * direction
    dates = [str(date(2024, 1, 2) + timedelta(days=index)) for index in range(days)]
    equity = [(1.0 + daily_gain) ** (index + 1) for index in range(days)]
    daily = []
    previous_nlv = initial_capital
    for index, (trade_date, nav) in enumerate(zip(dates, equity)):
        close_nlv = initial_capital * nav
        daily.append({
            "trade_date": trade_date,
            "open_nlv_before_fills": previous_nlv,
            "open_nlv_after_fills": previous_nlv,
            "open_gross_exposure_before_control": 0.9,
            "open_gross_exposure": 0.9,
            "leverage_control_orders": 0,
            "leverage_control_resolved": True,
            "close_nlv": close_nlv,
            "net_nav": nav,
            "same_orders_cost_free_nav_proxy": nav + 0.001,
            "daily_return": daily_gain,
            "cash": close_nlv * 0.1,
            "long_market_value": close_nlv * (0.8 if mode == "long_only" else 0.9),
            "short_market_value": 0.0 if mode == "long_only" else -close_nlv * 0.8,
            "gross_exposure": 0.8 if mode == "long_only" else 1.7,
            "net_exposure": 0.8 if mode == "long_only" else 0.1,
            "turnover": 0.02,
            "fills": 1 if index == 1 else 0,
            "events": 1,
            "orders_created": 1 if index == 0 else 0,
            "positions": 1,
            "borrow_fee": 0.1 if mode == "long_short" else 0.0,
            "margin_interest": 0.0,
            "portfolio_drawdown": 0.0,
            "portfolio_risk_orders": 0,
            "portfolio_risk_active": False,
            "portfolio_risk_rearmed": False,
            "risk_cooldown_remaining": 0,
            "position_state_max_error": 0.0,
        })
        previous_nlv = close_nlv
    final_nlv = initial_capital * equity[-1]
    stats = {
        "protocol": "step_event_v2",
        "days": days,
        "initial_capital": initial_capital,
        "final_nlv": final_nlv,
        "final_nav": equity[-1],
        "ann_ret": equity[-1] ** (252 / days) - 1,
        "ann_vol": 0.1,
        "sharpe": 1.0 if daily_gain > 0 else -1.0,
        "sortino": 1.0,
        "calmar": 1.0,
        "max_dd": 0.02,
        "avg_daily_turnover": 0.02,
        "avg_gross_exposure": 0.8,
        "avg_net_exposure": 0.8,
        "fills": 1,
        "orders": 1,
        "orders_executed": 1,
        "rejected_orders": 0,
        "partial_orders": 0,
        "fill_rate": 1.0,
        "commission_and_tax": 1.0,
        "slippage_cost": 0.5,
        "borrow_cost": 6.0 if mode == "long_short" else 0.0,
        "margin_interest": 0.0,
        "total_execution_cost": 7.5 if mode == "long_short" else 1.5,
        "fee_profile": "ashare_wan2_no_min_v1" if market == "ashare" else "ibkr_pro_fixed_us_v1",
        "currency": "CNY" if market == "ashare" else "USD",
        "open_positions": 1,
        "same_orders_cost_free_final_nav_proxy": equity[-1] + 0.001,
        "closed_trades": 0,
        "win_rate": 0.0,
        "profit_factor": None,
        "payoff_ratio": None,
        "avg_trade_return": 0.0,
        "avg_holding_sessions": 0.0,
        "portfolio_liquidations": 0,
        "portfolio_risk_trigger_events": 0,
        "portfolio_risk_rearms": 0,
        "portfolio_risk_active_sessions": 0,
        "portfolio_risk_active_at_end": False,
        "portfolio_risk_last_trigger": None,
        "max_portfolio_risk_cycle_drawdown": 0.0,
        "terminal_flat_sessions": 0,
        "exit_reason_counts": {},
        "gross_leverage_breach_events": 0,
        "automatic_deleveraging_events": 0,
        "leverage_limited_orders": 0,
        "max_open_gross_leverage_observed": 0.9,
        "max_open_gross_leverage_after_control": 0.9,
    }
    trade = {
        "fill_id": "FILL-00000001",
        "order_id": "ORD-00000001",
        "signal_date": dates[0],
        "trade_date": dates[1],
        "symbol": "TEST",
        "name": "Test",
    }
    event = {
        "seq": 1,
        "trade_date": dates[0],
        "phase": "CLOSE_SIGNAL",
        "event_type": "ORDER_CREATED",
        "symbol": "TEST",
        "order_id": "ORD-00000001",
        "message": "test",
    }
    return {
        "protocol": "step_event_v2",
        "config": {"market": market, "mode": mode, "max_gross_leverage": 2.0},
        "fee_schedule": {"profile": stats["fee_profile"]},
        "stats": stats,
        "curve": {"dates": dates, "equity": equity, "cost_free_proxy": [value + 0.001 for value in equity], "daily_ret": [daily_gain] * days},
        "daily_steps": daily,
        "trades": [trade],
        "events": [event],
        "round_trips": [],
        "positions": [{"symbol": "TEST", "quantity": 1}],
        "integrity": {
            "statement_rows": 1,
            "cash_reconciliation_max_error": 0.0,
            "same_day_signal_fill_violations": 0,
            "gross_leverage_violation_details": [],
            "ledger_source_of_truth": True,
            "all_pass": True,
        },
        "trade_page": {"total": 1},
        "event_page": {"total": 1},
        "artifacts": None,
    }


@pytest.mark.parametrize(
    ("market", "mode"),
    [
        ("ashare", "long_only"),
        ("us", "long_only"),
        ("us", "long_short"),
    ],
)
def test_weighted_sleeves_cover_both_markets_and_us_modes(tmp_path: Path, market: str, mode: str):
    calls = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        return _fake_single_result(**kwargs)

    with patch("backend.app.backtest.engine.run_backtest", side_effect=fake_runner):
        result = run_multi_factor_backtest(
            [
                {"name": "Alpha A", "expression": "alpha_a", "weight": 3, "direction": 1},
                {"name": "Alpha B", "expression": "alpha_b", "weight": 1, "direction": -1},
            ],
            market=market,
            mode=mode,
            universe_n=500,
            start="2024-01-01",
            end="2024-12-31",
            initial_capital=1_000_000,
            artifact_dir=tmp_path,
        )

    assert result["protocol"] == MULTI_FACTOR_BACKTEST_PROTOCOL
    assert [call["initial_capital"] for call in calls] == [750_000, 250_000]
    assert all(call["market"] == market and call["mode"] == mode for call in calls)
    assert result["stats"]["factor_count"] == 2
    assert result["integrity"]["all_pass"] is True
    assert result["integrity"]["factor_sleeve_integrity_failures"] == 0
    assert result["integrity"]["attribution_final_nlv_max_error"] <= 1e-5
    assert result["integrity"]["attribution_return_contribution_max_error"] <= 1e-8
    assert sum(row["return_contribution"] for row in result["factor_attribution"]) == pytest.approx(
        result["stats"]["final_nav"] - 1.0,
        abs=2e-6,
    )
    assert {row["factor_id"] for row in result["trades"]} == {"F01", "F02"}
    assert len({row["fill_id"] for row in result["trades"]}) == 2
    assert not (tmp_path / "factor_attribution.csv").exists()
    assert (tmp_path / "factor_attribution.parquet").exists()


def test_factor_spec_validation_and_request_backward_compatibility():
    with pytest.raises(ValueError, match="权重必须为正数"):
        _weighted_sleeve_specs([{"expression": "close", "weight": 0}])
    with pytest.raises(ValueError, match="1..12"):
        _weighted_sleeve_specs([])

    legacy = BacktestReq(expression="rank(close)")
    assert legacy.factors == []
    multi = BacktestReq(factors=[
        {"name": "Value", "expression": "rank(pb)", "weight": 2, "direction": -1}
    ])
    assert multi.factors[0].weight == 2
    assert multi.factors[0].direction == -1


def _synthetic_factor_frame(expression: str, market: str) -> pl.DataFrame:
    rows = []
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    for day_index in range(65):
        trade_date = date(2024, 1, 2) + timedelta(days=day_index)
        for symbol_index, symbol in enumerate(symbols):
            price = 20.0 + symbol_index * 2.0 + day_index * (0.03 - symbol_index * 0.005)
            base_factor = 4.0 - symbol_index
            rows.append({
                "trade_date": trade_date,
                "ts_code": symbol,
                "name": symbol,
                "factor": base_factor if expression == "alpha_a" else 5.0 - base_factor,
                "univ_rank": symbol_index + 1,
                "raw_open": price,
                "raw_high": price * 1.01,
                "raw_low": price * 0.99,
                "raw_close": price * 1.002,
                "vol": 2_000_000.0,
                "amount": 40_000_000.0,
                "adjustment_factor": 1.0,
                "can_buy_open_proxy": True,
                "can_sell_open_proxy": True,
                "_adv20_prev": 2_000_000.0,
                "_atr_pct": 0.02,
                "_vol20_prev": 0.20,
                "market": market,
            })
    return pl.DataFrame(rows).sort("trade_date", "ts_code")


@pytest.mark.parametrize(
    ("market", "mode", "account_type", "max_leverage", "short_target"),
    [
        ("ashare", "long_only", "cash", 1.0, 0.0),
        ("us", "long_only", "cash", 1.0, 0.0),
        ("us", "long_short", "margin", 2.0, 0.8),
    ],
)
def test_real_event_engine_multi_factor_smoke(
    market: str,
    mode: str,
    account_type: str,
    max_leverage: float,
    short_target: float,
):
    def prepared_frame(**kwargs):
        return _synthetic_factor_frame(kwargs["expression"], market), None

    with patch("backend.app.backtest.engine._prepare_backtest_frame", side_effect=prepared_frame):
        result = run_multi_factor_backtest(
            [
                {"name": "趋势", "expression": "alpha_a", "weight": 2, "direction": 1},
                {"name": "反向值", "expression": "alpha_b", "weight": 1, "direction": -1},
            ],
            market=market,
            mode=mode,
            universe_n=4,
            start="2024-01-01",
            end="2024-12-31",
            initial_capital=1_000_000,
            top_fraction=0.25,
            rebalance_every=5,
            slippage_bps=0,
            max_volume_participation=1.0,
            account_type=account_type,
            cash_buffer_fraction=0.05,
            max_gross_leverage=max_leverage,
            long_gross_target=0.8,
            short_gross_target=short_target,
            max_position_weight=1.0,
            response_daily_limit=None,
        )

    assert result["integrity"]["all_pass"] is True
    assert result["stats"]["factor_count"] == 2
    assert result["stats"]["fills"] > 0
    assert len(result["curve"]["dates"]) == 65
    assert sum(row["final_nlv"] for row in result["factor_attribution"]) == pytest.approx(
        result["stats"]["final_nlv"], abs=1e-5
    )
    if mode == "long_short":
        assert any(position["quantity"] < 0 for position in result["positions"])
    else:
        assert all(position["quantity"] >= 0 for position in result["positions"])
