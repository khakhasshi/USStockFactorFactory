from dataclasses import fields
from datetime import date, timedelta
import math

import pytest

from backend.app.backtest.engine import EventBacktestConfig, StepEventBacktester
from backend.app.backtest.risk import ExitPolicyConfig
from backend.scripts.run_dual_market_backtester_acceptance import build_task_set


@pytest.mark.parametrize("task", build_task_set(), ids=lambda t: t["task_id"])
def test_cached_valuation_exact_parity_all_acceptance_profiles(task):
    names = {f.name for f in fields(EventBacktestConfig)}
    cfg = {k: v for k, v in task.items() if k in names}
    cfg["exit_policy"] = ExitPolicyConfig(**task["exit_policy"])
    cfg["max_stale_sessions"] = 3
    runners = [StepEventBacktester(EventBacktestConfig(**cfg)) for _ in range(2)]
    runners[0].execution_optimizations = False
    runners[1].execution_optimizations = True
    start = date(2024, 1, 2)
    for day in range(32):
        rows = []
        for stock in range(16):
            if stock == 0 and day >= 9:
                continue  # Held security loses all quotes / stale writeoff.
            split = stock == 1 and day >= 7
            px = (25 + stock * 3) * (1 + .13 * math.sin(day * .8 + stock))
            px *= .5 if split else 1
            rows.append({
                "ts_code": f"S{stock:03}", "name": f"S{stock:03}",
                "trade_date": start + timedelta(days=day), "univ_rank": stock + 1,
                "factor": math.cos(stock + day / 3),
                "raw_open": px, "raw_close": px * 1.01,
                "raw_high": px * 1.18, "raw_low": px * .82,
                "vol": 50000., "amount": 50000. * px,
                "_adv20_prev": 50000., "_atr_pct": .025, "_vol20_prev": .3,
                "adjustment_factor": 2. if split else 1.,
                "can_buy_open_proxy": not (day % 7 == 2 and stock == 2),
                "can_sell_open_proxy": not (day % 7 == 2 and stock == 3),
            })
        # Reuse caller's market object, exercising step-local reset boundaries.
        market = {r["ts_code"]: r for r in rows}
        for runner in runners:
            runner.step(trade_date=start + timedelta(days=day), rows=rows,
                        market_by_symbol=market,
                        next_trade_date=start + timedelta(days=day + 1) if day < 31 else None,
                        rebalance=day % task["rebalance_every"] == 0)
            assert runner._valuation_market is None
        assert runners[0].daily[-1] == runners[1].daily[-1]
    assert runners[0].result() == runners[1].result()


def test_reference_outside_step_and_cash_not_cached():
    r = StepEventBacktester(EventBacktestConfig())
    m = {"A": {"raw_open": 3.}}
    r.positions["A"] = 10.
    assert r._nlv(m, "raw_open") == r._nlv_reference(m, "raw_open")
    r._valuation_market = m
    before = r._nlv(m, "raw_open")
    r.cash -= 7.
    assert r._nlv(m, "raw_open")[0] == before[0] - 7.
    r.positions["A"] = -2.
    r._invalidate_valuation("A")
    assert r._nlv(m, "raw_open") == r._nlv_reference(m, "raw_open")


def test_exception_clears_session_caches(monkeypatch):
    r = StepEventBacktester(EventBacktestConfig())
    def fail(**kwargs):
        raise RuntimeError("fixture")
    monkeypatch.setattr(r, "_step_impl", fail)
    with pytest.raises(RuntimeError):
        r.step(trade_date=date(2024, 1, 2), rows=[], next_trade_date=None, rebalance=True)
    assert r._valuation_market is None
    assert not r._valuation_values and not r._valuation_totals


@pytest.mark.parametrize("expression", ["rank(close)", "ts_mean(rank(close),20)",
    "rank(ts_mean(close,20)/ts_mean(close,60))"])
@pytest.mark.parametrize("mode", ["long_only", "long_short"])
def test_training_upper_pruning_preserves_warmup_cross_section_and_outputs(monkeypatch, expression, mode):
    from test_evaluation_v3 import _panel, _SyntheticPanel
    from backend.app.eval import harness
    from backend.app.config import evaluation_config
    from polars.testing import assert_frame_equal
    panel = _SyntheticPanel(_panel())
    monkeypatch.setattr(harness.PanelStore, "get", lambda *a, **k: panel)
    results = []
    for flag in ("0", "1"):
        monkeypatch.setenv("FF_EVALUATION_WINDOW_OPTIMIZATIONS", flag)
        results.append(harness._prepare_daily(expression, universe_n=50, horizon=5,
            portfolio_mode=mode, direction=-1, panel_glob="synthetic", market="us",
            layers=["INNER_PUBLIC", "META_TRAIN"], cfg=evaluation_config("us")))
    # Daily signals/returns are exact. Polars parallel group reduction of
    # decile means already varies by ~1e-18 between TWO reference runs.
    assert_frame_equal(results[0][0], results[1][0], check_exact=True)
    assert_frame_equal(results[0][1], results[1][1], check_exact=False,
                       rel_tol=0, abs_tol=1e-15)
