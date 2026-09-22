from datetime import date, timedelta
from unittest.mock import patch

import polars as pl
import numpy as np

from backend.app.backtest.engine import run_backtest
from scripts.run_dynamic_26_regime_hedge import _regime_state


def _switching_frame(switch_date: date) -> pl.DataFrame:
    rows = []
    symbols = ["AAA", "BBB", "CCC", "DDD"]
    for day_index in range(65):
        trade_date = date(2024, 1, 2) + timedelta(days=day_index)
        for symbol_index, symbol in enumerate(symbols):
            price = 20.0 + symbol_index * 2.0 + day_index * 0.02
            normal = float(4 - symbol_index)
            rows.append({
                "trade_date": trade_date,
                "ts_code": symbol,
                "name": symbol,
                "factor": normal if trade_date < switch_date else -normal,
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
                "market": "us",
            })
    return pl.DataFrame(rows).sort("trade_date", "ts_code")


def test_prepared_dynamic_frame_rebalances_on_declared_signal_date_at_next_open():
    switch_date = date(2024, 1, 9)  # index 7: not a normal five-session rebalance
    frame = _switching_frame(switch_date)
    with patch(
        "backend.app.backtest.engine._prepare_backtest_frame",
        side_effect=AssertionError("prepared override must bypass materialization"),
    ):
        result = run_backtest(
            "dynamic_overlay",
            market="us",
            mode="long_short",
            universe_n=4,
            start="2024-01-01",
            end="2024-12-31",
            initial_capital=50_000,
            top_fraction=0.25,
            rebalance_every=5,
            slippage_bps=0,
            max_volume_participation=1.0,
            max_position_weight=1.0,
            long_gross_target=0.8,
            short_gross_target=0.8,
            response_trade_limit=10**9,
            response_daily_limit=None,
            _prepared_frame_override=frame,
            _extra_rebalance_dates={switch_date},
        )

    assert result["integrity"]["all_pass"] is True
    assert result["config"]["extra_rebalance_signal_dates"] == 1
    switched_fills = [
        row for row in result["trades"] if row["signal_date"] == switch_date.isoformat()
    ]
    assert switched_fills
    assert all(row["trade_date"] > row["signal_date"] for row in switched_fills)


def test_prepared_dynamic_frame_requires_factor_contract():
    frame = _switching_frame(date(2024, 1, 9)).drop("factor")
    try:
        run_backtest(
            "dynamic_overlay",
            _prepared_frame_override=frame,
        )
    except ValueError as exc:
        assert "factor" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("missing factor column must fail")


def test_regime_state_uses_close_t_window_only_and_applies_it_on_next_session():
    dates = [(date(2024, 1, 1) + timedelta(days=index)).isoformat() for index in range(65)]
    returns = np.full(65, 0.001, dtype=np.float64)
    rows, extra_dates = _regime_state(dates, returns)

    assert all(not row["regime_on_at_close"] for row in rows[:59])
    assert rows[59]["regime_on_at_close"] is True
    assert rows[59]["regime_on_applied_today"] is False
    assert rows[60]["regime_on_applied_today"] is True
    assert rows[59]["window_start"] == dates[0]
    assert rows[59]["window_end"] == dates[59]
    assert date.fromisoformat(dates[59]) in extra_dates
    assert all(
        rows[index]["regime_on_applied_today"] == rows[index - 1]["regime_on_at_close"]
        for index in range(1, len(rows))
    )


def test_dynamic_gross_scale_preserves_factor_schedule_and_executes_lag_one():
    switch_date = date(2024, 1, 9)
    frame = _switching_frame(date(2099, 1, 1))
    dates = frame["trade_date"].unique(maintain_order=True).to_list()
    schedule = {
        value: (0.5 if value >= switch_date else 1.0)
        for value in dates
    }
    result = run_backtest(
        "vol_scaled",
        market="us",
        mode="long_short",
        universe_n=4,
        start="2024-01-01",
        end="2024-12-31",
        initial_capital=100_000,
        top_fraction=0.25,
        rebalance_every=5,
        slippage_bps=0,
        max_volume_participation=1.0,
        max_position_weight=1.0,
        rebalance_buffer_pct=0.0,
        long_gross_target=0.8,
        short_gross_target=0.8,
        response_trade_limit=10**9,
        response_daily_limit=None,
        _prepared_frame_override=frame,
        _target_gross_scale_by_signal_date=schedule,
    )

    assert result["integrity"]["all_pass"] is True
    assert result["config"]["dynamic_gross_scale_signal_dates"] == len(dates)
    switch_fills = [
        row for row in result["trades"] if row["signal_date"] == switch_date.isoformat()
    ]
    assert switch_fills
    assert all(row["trade_date"] > row["signal_date"] for row in switch_fills)
    daily = {row["trade_date"]: row for row in result["daily_steps"]}
    assert daily[switch_date.isoformat()]["target_gross_scale_next_open"] == 0.5
    next_date = dates[dates.index(switch_date) + 1].isoformat()
    assert daily[next_date]["gross_exposure"] < daily[switch_date.isoformat()]["gross_exposure"]
