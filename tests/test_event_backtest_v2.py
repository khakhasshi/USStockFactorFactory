from datetime import date

from backend.app.backtest.engine import EventBacktestConfig, StepEventBacktester
from backend.app.backtest.risk import ExitPolicyConfig


def _row(
    trade_date: date,
    *,
    price: float,
    high: float | None = None,
    low: float | None = None,
    market: str = "us",
) -> dict:
    return {
        "trade_date": trade_date,
        "ts_code": "AAA",
        "name": "AAA",
        "factor": 1.0,
        "univ_rank": 1,
        "raw_open": price,
        "raw_high": high if high is not None else price,
        "raw_low": low if low is not None else price,
        "raw_close": price,
        "vol": 1_000_000.0,
        "_adv20_prev": 1_000_000.0,
        "_atr_pct": 0.02,
        "_vol20_prev": 0.25,
        "amount": 100_000_000.0,
        "adjustment_factor": 1.0,
        "can_buy_open_proxy": True,
        "can_sell_open_proxy": True,
        "market": market,
    }


def _enter(runner: StepEventBacktester, *, next_open: float = 100.0, high=None, low=None):
    day0 = date(2024, 1, 2)
    day1 = date(2024, 1, 3)
    runner.step(
        trade_date=day0,
        rows=[_row(day0, price=100.0, market=runner.config.market)],
        next_trade_date=day1,
        rebalance=True,
    )
    runner.step(
        trade_date=day1,
        rows=[_row(day1, price=next_open, high=high, low=low, market=runner.config.market)],
        next_trade_date=date(2024, 1, 4),
        rebalance=False,
    )


def test_us_cash_account_caps_gap_buy_and_integrity_detects_no_negative_cash():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        account_type="cash",
        max_gross_leverage=1.25,
    ))
    _enter(runner, next_open=110.0)
    result = runner.result()
    assert runner.cash >= -1e-8
    assert runner.positions["AAA"] < 1_000
    assert result["integrity"]["cash_account_negative_cash_violations"] == 0
    assert result["integrity"]["all_pass"]


def test_intraday_fixed_stop_closes_us_position_and_records_round_trip():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        exit_policy=ExitPolicyConfig(fixed_stop_loss_pct=0.08),
    ))
    _enter(runner, next_open=100.0, high=103.0, low=90.0)
    result = runner.result()
    assert "AAA" not in runner.positions
    assert result["round_trips"][0]["exit_reason"] == "risk_stop_intraday"
    assert result["round_trips"][0]["exit_price"] == 92.0
    assert result["integrity"]["all_pass"]


def test_same_bar_stop_and_take_profit_uses_conservative_path():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        exit_policy=ExitPolicyConfig(
            fixed_stop_loss_pct=0.08,
            fixed_take_profit_pct=0.10,
            intrabar_conflict_policy="conservative",
        ),
    ))
    _enter(runner, next_open=100.0, high=112.0, low=90.0)
    result = runner.result()
    assert result["round_trips"][0]["exit_reason"] == "risk_stop_intraday"
    assert result["integrity"]["intrabar_ambiguities"] == 1
    assert result["integrity"]["all_pass"]


def test_ashare_t1_blocks_same_day_stop_and_carries_exit():
    runner = StepEventBacktester(EventBacktestConfig(
        market="ashare",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        max_order_age_sessions=3,
        exit_policy=ExitPolicyConfig(fixed_stop_loss_pct=0.08),
    ))
    _enter(runner, next_open=100.0, high=101.0, low=90.0)
    assert runner.positions["AAA"] > 0
    assert any(order["reason"] == "risk_stop_intraday" for order in runner.pending_orders)
    day2 = date(2024, 1, 4)
    runner.step(
        trade_date=day2,
        rows=[_row(day2, price=89.0, high=90.0, low=88.0, market="ashare")],
        next_trade_date=None,
        rebalance=False,
    )
    assert "AAA" not in runner.positions
    assert runner.result()["integrity"]["all_pass"]


def test_liquidity_uses_previous_adv_and_square_root_impact():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=2,
        spread_bps=4,
        impact_model="square_root",
        impact_coefficient_bps=20,
        max_volume_participation=0.10,
    ))
    _enter(runner, next_open=100.0)
    fill = runner.trades[0]
    assert fill["liquidity_basis"] == "previous_20_session_adv"
    assert fill["impact_bps"] > 4.0
    assert fill["participation"] <= 0.10


def test_stale_long_position_is_written_off_instead_of_becoming_ghost_asset():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        max_stale_sessions=2,
    ))
    _enter(runner, next_open=100.0)
    for index, day in enumerate((date(2024, 1, 4), date(2024, 1, 5))):
        runner.step(
            trade_date=day,
            rows=[],
            market_by_symbol={},
            next_trade_date=None,
            rebalance=False,
        )
    result = runner.result()
    assert "AAA" not in runner.positions
    assert result["integrity"]["stale_position_writeoffs"] == 1
    assert result["round_trips"][-1]["exit_reason"] == "stale_position_writeoff"


def test_effective_targets_reserve_operating_headroom_below_hard_limit():
    config = EventBacktestConfig(
        market="us",
        mode="long_short",
        cash_buffer_fraction=0.05,
        max_gross_leverage=2.0,
        long_gross_target=1.0,
        short_gross_target=1.0,
    )
    config.validate()
    assert config.effective_long_gross_target == 0.95
    assert config.effective_short_gross_target == 0.95
    assert abs(config.leverage_headroom - 0.1) < 1e-12


def test_margin_gap_buy_is_clipped_per_fill_to_hard_leverage_cap():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        account_type="margin",
        cash_buffer_fraction=0,
        max_gross_leverage=1,
        long_gross_target=1,
    ))
    trade_day = date(2024, 1, 3)
    runner._order_seq = 1
    runner.pending_orders.append({
        "order_id": "ORD-00000001",
        "signal_date": "2024-01-02",
        "execute_date": str(trade_day),
        "symbol": "AAA",
        "target_quantity": 1_000.0,
        "quantity_at_signal": 0.0,
        "reason": "manual_margin_order",
        "order_type": "MARKET",
        "time_in_force": "DAY",
        "age_sessions": 0,
        "atr_pct_at_signal": None,
    })
    runner.step(
        trade_date=trade_day,
        rows=[_row(trade_day, price=110.0)],
        next_trade_date=date(2024, 1, 4),
        rebalance=False,
    )
    result = runner.result()
    fill = runner.trades[0]
    assert fill["leverage_limited"] is True
    assert fill["gross_leverage_after"] <= 1.0 + 1e-8
    assert result["integrity"]["leverage_limited_orders"] == 1
    assert result["integrity"]["gross_leverage_violations"] == 0
    assert result["integrity"]["all_pass"]


def test_open_breach_is_auto_deleveraged_and_retained_as_audit_evidence():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        account_type="margin",
        cash_buffer_fraction=0.02,
        max_gross_leverage=1,
        long_gross_target=0.95,
    ))
    _enter(runner, next_open=100.0)
    # A financing/collateral shock between sessions can create a passive
    # opening breach even though the original fills respected the ceiling.
    runner.cash -= 10_000
    day2 = date(2024, 1, 4)
    row = runner.step(
        trade_date=day2,
        rows=[_row(day2, price=100.0)],
        next_trade_date=date(2024, 1, 5),
        rebalance=False,
    )
    result = runner.result()
    assert row["open_gross_exposure_before_control"] > 1
    assert row["open_gross_exposure"] <= 1 + 1e-6
    assert row["leverage_control_orders"] >= 1
    assert result["integrity"]["gross_leverage_breach_events"] == 1
    assert result["integrity"]["automatic_deleveraging_events"] == 1
    assert result["integrity"]["gross_leverage_violations"] == 0
    assert result["integrity"]["gross_leverage_violation_details"][0]["resolved"]
    assert result["integrity"]["all_pass"]


def test_unfillable_open_breach_fails_with_date_and_before_after_details():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        account_type="margin",
        cash_buffer_fraction=0.02,
        max_gross_leverage=1,
        long_gross_target=0.95,
    ))
    _enter(runner, next_open=100.0)
    runner.cash -= 10_000
    day2 = date(2024, 1, 4)
    blocked = _row(day2, price=100.0)
    blocked["vol"] = 0.0
    blocked["_adv20_prev"] = 0.0
    runner.step(
        trade_date=day2,
        rows=[blocked],
        next_trade_date=date(2024, 1, 5),
        rebalance=False,
    )
    result = runner.result()
    detail = result["integrity"]["gross_leverage_violation_details"][0]
    assert detail["trade_date"] == "2024-01-04"
    assert detail["before"] > 1
    assert detail["after"] > 1
    assert detail["resolved"] is False
    assert result["integrity"]["gross_leverage_violations"] == 1
    assert result["integrity"]["all_pass"] is False


def test_portfolio_drawdown_exit_rearms_after_flat_cooldown_instead_of_locking_cash():
    runner = StepEventBacktester(EventBacktestConfig(
        market="us",
        mode="long_only",
        universe_n=1,
        top_fraction=0.2,
        initial_capital=100_000,
        slippage_bps=0,
        max_volume_participation=1,
        account_type="margin",
        cash_buffer_fraction=0.02,
        max_gross_leverage=1,
        long_gross_target=0.95,
        portfolio_stop_drawdown_pct=0.10,
        risk_cooldown_sessions=2,
    ))
    _enter(runner, next_open=100.0)

    day2 = date(2024, 1, 4)
    trigger_row = runner.step(
        trade_date=day2,
        rows=[_row(day2, price=80.0)],
        next_trade_date=date(2024, 1, 5),
        rebalance=False,
    )
    assert trigger_row["portfolio_risk_orders"] > 0
    assert trigger_row["portfolio_risk_active"] is True

    day3 = date(2024, 1, 5)
    runner.step(
        trade_date=day3,
        rows=[_row(day3, price=80.0)],
        next_trade_date=date(2024, 1, 8),
        rebalance=True,
    )
    assert not runner.positions

    day4 = date(2024, 1, 8)
    rearm_row = runner.step(
        trade_date=day4,
        rows=[_row(day4, price=80.0)],
        next_trade_date=date(2024, 1, 9),
        rebalance=True,
    )
    assert rearm_row["portfolio_risk_rearmed"] is True
    assert rearm_row["orders_created"] == 0

    day5 = date(2024, 1, 9)
    signal_row = runner.step(
        trade_date=day5,
        rows=[_row(day5, price=80.0)],
        next_trade_date=date(2024, 1, 10),
        rebalance=True,
    )
    assert signal_row["orders_created"] > 0

    day6 = date(2024, 1, 10)
    runner.step(
        trade_date=day6,
        rows=[_row(day6, price=80.0)],
        next_trade_date=date(2024, 1, 11),
        rebalance=False,
    )
    result = runner.result()
    assert runner.positions["AAA"] > 0
    assert result["stats"]["portfolio_risk_trigger_events"] == 1
    assert result["stats"]["portfolio_risk_rearms"] == 1
    assert result["stats"]["portfolio_risk_active_at_end"] is False
    assert result["integrity"]["event_phase_order_violations"] == 0
    assert result["integrity"]["all_pass"]
