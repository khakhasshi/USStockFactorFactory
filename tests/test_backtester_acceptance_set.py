from backend.scripts.run_dual_market_backtester_acceptance import (
    _summarize,
    build_task_set,
)


def test_dual_market_acceptance_set_is_frozen_balanced_and_modular():
    tasks = build_task_set()
    assert len(tasks) == 60
    assert len({row["task_id"] for row in tasks}) == 60
    assert len({row["task_hash"] for row in tasks}) == 60
    assert sum(row["market"] == "ashare" for row in tasks) == 30
    assert sum(row["market"] == "us" for row in tasks) == 30
    assert {row["mode"] for row in tasks if row["market"] == "ashare"} == {"long_only"}
    assert {row["mode"] for row in tasks if row["market"] == "us"} == {
        "long_only",
        "long_short",
    }
    assert {row["position_sizing"] for row in tasks} == {
        "equal_weight",
        "inverse_volatility",
        "atr_risk",
    }
    assert {row["impact_model"] for row in tasks} == {
        "fixed",
        "linear",
        "square_root",
    }
    assert any(row["portfolio_stop_drawdown_pct"] for row in tasks)
    assert any(row["unfilled_order_policy"] == "carry" for row in tasks)
    assert any(row["liquidate_at_end"] for row in tasks)


def test_acceptance_set_uses_full_requested_window_and_safe_operating_targets():
    for task in build_task_set():
        assert task["start"] == "2020-01-01"
        assert task["end"] == "2026-08-21"
        assert task["long_gross_target"] + task["short_gross_target"] \
            <= task["max_gross_leverage"]
        assert task["expression_profile"]["required_history"] <= 1_000


def test_acceptance_understands_compact_120_day_detail_contract():
    task = build_task_set()[0]
    days = 1_000
    integrity = {
        "all_pass": True,
        "gross_leverage_violations": 0,
        "cash_account_negative_cash_violations": 0,
        "ashare_buy_lot_violations": 0,
        "long_only_negative_position_violations": 0,
        "event_phase_order_violations": 0,
    }
    result = {
        "stats": {
            "days": days,
            "fills": 1,
            "final_nlv": 1.0,
            "max_open_gross_leverage_after_control": 0.9,
            "portfolio_risk_trigger_events": 0,
            "portfolio_risk_rearms": 0,
            "portfolio_risk_active_at_end": False,
            "terminal_flat_sessions": 0,
        },
        "integrity": integrity,
        "curve": {
            "dates": [str(index) for index in range(days)],
            "equity": [1.0] * days,
            "cost_free_proxy": [1.0] * days,
            "daily_ret": [0.0] * days,
        },
        "daily_steps": [{"close_nlv": 1.0}] * 120,
    }
    summary = _summarize(task, result, 1.0)
    assert summary["checks"]["curve_lengths_match"]
    assert summary["checks"]["daily_detail_window_contract"]
    assert summary["status"] == "pass"
