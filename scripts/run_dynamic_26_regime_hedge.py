#!/usr/bin/env python3
"""Run the preregistered 60-session, 5% #26 dynamic hedge experiment.

The investable portfolio has three real event-ledger sleeves:

* 69% #21 fixed core;
* 26% #19 fixed core;
* 5% overlay that holds #21 when the regime is off and #26 when it is on.

An independent, fee-after #26 shadow sleeve supplies the regime state.  Its
close-t 60-session return may only affect overlay orders scheduled for t+1.
The shadow sleeve is not included in portfolio P&L or execution costs.
"""

from __future__ import annotations

import json
import math
import sys
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app.backtest.engine import (  # noqa: E402
    _performance_from_nav,
    _prepare_backtest_frame,
    run_backtest,
    run_multi_factor_backtest,
)
from backend.app.backtest.stability import (  # noqa: E402
    analyze_return_stability,
    monte_carlo_analysis,
)


BASELINE_ID = 732
REPORT_DIR = ROOT / "var/reports/us-dynamic-26-regime-hedge-20260827"
EXPRESSION_21 = "-rank(ts_mean((high - low) / amount, 20))"
EXPRESSION_19 = "rank((-ts_mean((close / amount), 200)))"
EXPRESSION_26 = (
    "rank((delay(close, 21)/delay(close, 252)- 1)/"
    "(ts_std(ts_delta(close, 1), 60)+ 1e-9)*"
    "(ts_mean(amount, 20)/(ts_mean(amount, 120) + 1e-9)))"
)
WINDOW = 60
HEDGE_WEIGHT = 0.05


def _api_backtest(backtest_id: int) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"http://127.0.0.1:8765/api/backtests/{backtest_id}", timeout=30
    ) as response:
        return json.load(response)


def _common_kwargs(params: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "universe_n", "start", "end", "cost_bps", "mode", "panel_glob",
        "borrow_cost_bps_annual", "top_fraction", "rebalance_every",
        "slippage_bps", "max_volume_participation", "fee_profile",
        "account_type", "cash_buffer_fraction", "max_gross_leverage",
        "margin_interest_bps_annual", "position_sizing", "max_positions",
        "max_position_weight", "min_trade_notional", "rebalance_buffer_pct",
        "long_gross_target", "short_gross_target", "risk_per_position_fraction",
        "spread_bps", "impact_model", "impact_coefficient_bps",
        "unfilled_order_policy", "max_order_age_sessions", "max_stale_sessions",
        "liquidate_at_end", "portfolio_stop_drawdown_pct",
        "portfolio_daily_loss_pct", "risk_cooldown_sessions", "exit_policy",
    )
    output = {field: params[field] for field in fields}
    output.update({
        "market": "us",
        "execution_backend": "python",
        "capture_detail": True,
        "response_trade_limit": 10**9,
        "response_daily_limit": None,
        "monte_carlo_enabled": False,
    })
    return output


def _daily_returns(rows: list[dict[str, Any]], initial_capital: float) -> np.ndarray:
    nlv = np.asarray([float(row["close_nlv"]) for row in rows], dtype=np.float64)
    prior = np.concatenate([[float(initial_capital)], nlv[:-1]])
    return nlv / prior - 1.0


def _regime_state(
    dates: list[str], returns: np.ndarray
) -> tuple[list[dict[str, Any]], set[date]]:
    rows = []
    prior_state = False
    extra_dates: set[date] = set()
    for index, value in enumerate(dates):
        window_return = None
        state = False
        if index + 1 >= WINDOW:
            window_return = float(np.prod(1.0 + returns[index - WINDOW + 1 : index + 1]) - 1.0)
            state = window_return > 0.0
        if index > 0 and state != prior_state:
            extra_dates.add(date.fromisoformat(value))
        rows.append({
            "trade_date": value,
            "window_sessions": WINDOW,
            "window_start": dates[index - WINDOW + 1] if index + 1 >= WINDOW else None,
            "window_end": value if index + 1 >= WINDOW else None,
            "shadow_26_fee_after_return_60d_at_close": window_return,
            "regime_on_at_close": state,
            "regime_on_applied_today": rows[-1]["regime_on_at_close"] if rows else False,
            "source_max_index": index,
            "applied_source_max_index": index - 1,
        })
        prior_state = state
    return rows, extra_dates


def _dynamic_frame(
    params: dict[str, Any], regime_rows: list[dict[str, Any]]
) -> pl.DataFrame:
    prepare = dict(
        universe_n=params["universe_n"], start=params["start"], end=params["end"],
        panel_glob=params["panel_glob"], market="us",
        forward_horizon=params["rebalance_every"],
        atr_period=params["exit_policy"]["atr_period"],
    )
    core, _ = _prepare_backtest_frame(expression=EXPRESSION_21, **prepare)
    hedge, _ = _prepare_backtest_frame(expression=EXPRESSION_26, **prepare)
    if core.select("trade_date", "ts_code").equals(
        hedge.select("trade_date", "ts_code")
    ) is False:
        raise RuntimeError("#21 and #26 prepared frames are not aligned")
    regime = pl.DataFrame({
        "trade_date": [date.fromisoformat(row["trade_date"]) for row in regime_rows],
        "_regime_on": [bool(row["regime_on_at_close"]) for row in regime_rows],
    })
    return (
        core.rename({"factor": "_factor_21"})
        .join(
            hedge.select("trade_date", "ts_code", pl.col("factor").alias("_factor_26")),
            on=["trade_date", "ts_code"], how="inner", validate="1:1",
        )
        .join(regime, on="trade_date", how="left", validate="m:1")
        .with_columns(
            pl.when(pl.col("_regime_on"))
            .then(pl.col("_factor_26"))
            .otherwise(pl.col("_factor_21"))
            .alias("factor")
        )
        .drop("_factor_21", "_factor_26", "_regime_on")
        .sort("trade_date", "ts_code")
    )


def _subset_performance(returns: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = returns[mask]
    nav = np.cumprod(1.0 + selected)
    perf = _performance_from_nav(nav.tolist())
    return {
        "sessions": int(selected.size),
        "total_return": round(float(nav[-1] - 1.0), 8),
        **perf,
    }


def _combine(
    core: dict[str, Any],
    overlay: dict[str, Any],
    regime_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if core["curve"]["dates"] != overlay["curve"]["dates"]:
        raise RuntimeError("core and overlay dates differ")
    dates = core["curve"]["dates"]
    combined = []
    previous_nlv = 1_000_000.0
    peak = previous_nlv
    for index, trade_date in enumerate(dates):
        core_row = core["daily_steps"][index]
        overlay_row = overlay["daily_steps"][index]
        close_nlv = float(core_row["close_nlv"]) + float(overlay_row["close_nlv"])
        core_previous = 950_000.0 if index == 0 else float(core["daily_steps"][index - 1]["close_nlv"])
        overlay_previous = 50_000.0 if index == 0 else float(overlay["daily_steps"][index - 1]["close_nlv"])
        turnover_notional = (
            float(core_row["turnover"]) * core_previous
            + float(overlay_row["turnover"]) * overlay_previous
        )
        daily_return = close_nlv / previous_nlv - 1.0
        peak = max(peak, close_nlv)
        regime = regime_rows[index]
        combined.append({
            "trade_date": trade_date,
            "close_nlv": round(close_nlv, 6),
            "net_nav": round(close_nlv / 1_000_000.0, 8),
            "daily_return": round(daily_return, 8),
            "turnover": round(turnover_notional / previous_nlv, 8),
            "portfolio_drawdown": round(1.0 - close_nlv / peak, 8),
            "regime_on_at_close": regime["regime_on_at_close"],
            "regime_on_applied_today": regime["regime_on_applied_today"],
            "shadow_26_fee_after_return_60d_at_close": regime[
                "shadow_26_fee_after_return_60d_at_close"
            ],
        })
        previous_nlv = close_nlv
    returns = np.asarray([row["daily_return"] for row in combined], dtype=np.float64)
    nav = np.asarray([row["net_nav"] for row in combined], dtype=np.float64)
    perf = _performance_from_nav(nav.tolist())
    round_trips = [*core.get("round_trips", []), *overlay.get("round_trips", [])]
    wins = [row for row in round_trips if float(row.get("net_pnl", 0.0)) > 0]
    losses = [row for row in round_trips if float(row.get("net_pnl", 0.0)) < 0]
    gross_profit = sum(float(row["net_pnl"]) for row in wins)
    gross_loss = abs(sum(float(row["net_pnl"]) for row in losses))
    stats = {
        **perf,
        "initial_capital": 1_000_000.0,
        "final_nlv": round(float(nav[-1] * 1_000_000.0), 6),
        "avg_daily_turnover": round(float(np.mean([row["turnover"] for row in combined])), 6),
        "profit_factor": round(gross_profit / gross_loss, 6) if gross_loss > 1e-12 else None,
        "win_rate": round(len(wins) / len(round_trips), 6) if round_trips else 0.0,
        "closed_trades": len(round_trips),
        "commission_and_tax": round(
            float(core["stats"]["commission_and_tax"])
            + float(overlay["stats"]["commission_and_tax"]), 6
        ),
        "slippage_cost": round(
            float(core["stats"]["slippage_cost"])
            + float(overlay["stats"]["slippage_cost"]), 6
        ),
        "borrow_cost": round(
            float(core["stats"]["borrow_cost"])
            + float(overlay["stats"]["borrow_cost"]), 6
        ),
        "margin_interest": round(
            float(core["stats"]["margin_interest"])
            + float(overlay["stats"]["margin_interest"]), 6
        ),
        "total_execution_cost": round(
            float(core["stats"]["total_execution_cost"])
            + float(overlay["stats"]["total_execution_cost"]), 6
        ),
    }
    return combined, stats


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _api_backtest(BASELINE_ID)
    if baseline["status"] != "done" or not baseline["result"]["integrity"]["all_pass"]:
        raise RuntimeError("frozen baseline #732 is not an audited completed backtest")
    params = baseline["params"]
    common = _common_kwargs(params)

    print("[1/5] run independent fee-after #26 shadow sleeve", flush=True)
    shadow = run_backtest(
        EXPRESSION_26, direction=1, initial_capital=1_000_000.0,
        artifact_dir=REPORT_DIR / "reference_26", **common,
    )
    dates = shadow["curve"]["dates"]
    shadow_returns = _daily_returns(shadow["daily_steps"], 1_000_000.0)
    regime_rows, extra_dates = _regime_state(dates, shadow_returns)

    print("[2/5] materialize causal dynamic overlay frame", flush=True)
    overlay_frame = _dynamic_frame(params, regime_rows)

    print("[3/5] run fixed 69% #21 + 26% #19 core", flush=True)
    core = run_multi_factor_backtest(
        [
            {"name": "因子21", "expression": EXPRESSION_21, "weight": 69, "direction": 1},
            {"name": "因子19", "expression": EXPRESSION_19, "weight": 26, "direction": 1},
        ],
        initial_capital=950_000.0,
        artifact_dir=REPORT_DIR / "core_69_26",
        **common,
    )

    print("[4/5] run 5% event-ledger overlay with state-change rebalances", flush=True)
    overlay = run_backtest(
        "causal_dynamic_overlay_21_or_26", direction=1, initial_capital=50_000.0,
        artifact_dir=REPORT_DIR / "overlay_5pct",
        _prepared_frame_override=overlay_frame,
        _extra_rebalance_dates=extra_dates,
        **common,
    )

    print("[5/5] reconcile portfolio and diagnostics", flush=True)
    combined, stats = _combine(core, overlay, regime_rows)
    stability = analyze_return_stability(
        combined, total_initial_capital=1_000_000.0, sleeves=[]
    )
    bootstrap = monte_carlo_analysis(
        combined, simulations=2000, block_size_sessions=20, seed=20260824
    )
    applied = np.asarray(
        [bool(row["regime_on_applied_today"]) for row in regime_rows], dtype=bool
    )
    combined_returns = np.asarray(
        [float(row["daily_return"]) for row in combined], dtype=np.float64
    )
    annual_on = {}
    for year in sorted({value[:4] for value in dates}):
        mask = np.asarray([value.startswith(year) for value in dates], dtype=bool)
        annual_on[year] = round(float(np.mean(applied[mask])), 8)
    lag_mismatches = sum(
        bool(regime_rows[index]["regime_on_applied_today"])
        != bool(regime_rows[index - 1]["regime_on_at_close"])
        for index in range(1, len(regime_rows))
    )
    first_day_mismatch = bool(regime_rows[0]["regime_on_applied_today"])
    trade_timing_violations = sum(
        row.get("reason") == "factor_rebalance_next_open"
        and row["signal_date"] >= row["trade_date"]
        for row in overlay["trades"]
    )
    baseline_stats = baseline["result"]["stats"]
    baseline_nav = np.asarray(
        baseline["result"]["curve"]["equity"], dtype=np.float64
    )
    baseline_returns = baseline_nav / np.concatenate([[1.0], baseline_nav[:-1]]) - 1.0
    baseline_when_on = _subset_performance(baseline_returns, applied)
    baseline_when_off = _subset_performance(baseline_returns, ~applied)
    dynamic_when_on = _subset_performance(combined_returns, applied)
    dynamic_when_off = _subset_performance(combined_returns, ~applied)
    annual = stability["annual"]
    result = {
        "schema": "factorfactory.dynamic-26-regime-hedge/v1",
        "protocol": "step_event_v2_dynamic_regime_overlay_v1",
        "generated_at": "2026-08-27",
        "predeclared_parameters": {
            "window_sessions": WINDOW,
            "hedge_weight": HEDGE_WEIGHT,
            "off_weights": {"factor_21": 0.74, "factor_19": 0.26, "factor_26": 0.0},
            "on_weights": {"factor_21": 0.69, "factor_19": 0.26, "factor_26": 0.05},
            "optimization_performed": False,
        },
        "actual_period": {"start": dates[0], "end": dates[-1], "sessions": len(dates)},
        "baseline": {
            "backtest_id": BASELINE_ID,
            "stats": baseline_stats,
            "integrity_all_pass": baseline["result"]["integrity"]["all_pass"],
        },
        "dynamic": {
            "stats": stats,
            "annual": annual,
            "calendar_2022": next(row for row in annual if row["period"] == "2022"),
            "regime_on_ratio": round(float(np.mean(applied)), 8),
            "regime_on_ratio_by_year": annual_on,
            "performance_when_on": dynamic_when_on,
            "performance_when_off": dynamic_when_off,
            "static_baseline_when_on": baseline_when_on,
            "static_baseline_when_off": baseline_when_off,
            "conditional_delta_when_on": {
                "cagr": round(dynamic_when_on["ann_ret"] - baseline_when_on["ann_ret"], 8),
                "sharpe": round(dynamic_when_on["sharpe"] - baseline_when_on["sharpe"], 8),
                "max_drawdown": round(dynamic_when_on["max_dd"] - baseline_when_on["max_dd"], 8),
            },
            "conditional_delta_when_off": {
                "cagr": round(dynamic_when_off["ann_ret"] - baseline_when_off["ann_ret"], 8),
                "sharpe": round(dynamic_when_off["sharpe"] - baseline_when_off["sharpe"], 8),
                "max_drawdown": round(dynamic_when_off["max_dd"] - baseline_when_off["max_dd"], 8),
            },
            "delta_vs_baseline": {
                "cagr": round(float(stats["ann_ret"]) - float(baseline_stats["ann_ret"]), 8),
                "sharpe": round(float(stats["sharpe"]) - float(baseline_stats["sharpe"]), 8),
                "max_drawdown": round(float(stats["max_dd"]) - float(baseline_stats["max_dd"]), 8),
            },
        },
        "bootstrap": {
            "method": bootstrap["method"],
            "simulations": bootstrap["simulations"],
            "block_size_sessions": bootstrap["block_size_sessions"],
            "seed": bootstrap["seed"],
            "sharpe_median": bootstrap["sharpe"]["p50"],
            "sharpe_p05": bootstrap["sharpe"]["p05"],
            "max_drawdown_p95": bootstrap["max_drawdown"]["p95"],
            "terminal_loss_probability": bootstrap["risk_probabilities"]["terminal_loss"],
        },
        "causality_audit": {
            "signal_definition": "fee-after independent #26 sleeve cumulative return over t-59..t > 0",
            "decision_time": "t_close_after_realized_fee_after_nlv",
            "application_time": "t_plus_1_raw_open",
            "applied_state_equals_previous_close_state": lag_mismatches == 0 and not first_day_mismatch,
            "lag_mismatch_count": lag_mismatches + int(first_day_mismatch),
            "overlay_trade_signal_date_violations": trade_timing_violations,
            "state_transition_signal_dates": len(extra_dates),
            "future_data_used": False,
            "all_pass": (
                lag_mismatches == 0
                and not first_day_mismatch
                and trade_timing_violations == 0
                and core["integrity"]["all_pass"]
                and overlay["integrity"]["all_pass"]
                and shadow["integrity"]["all_pass"]
            ),
        },
        "ledger_integrity": {
            "core_all_pass": core["integrity"]["all_pass"],
            "overlay_all_pass": overlay["integrity"]["all_pass"],
            "shadow_reference_all_pass": shadow["integrity"]["all_pass"],
            "shadow_cost_included_in_portfolio": False,
            "portfolio_cost_source": "core_69_26 plus executable_5pct_overlay only",
        },
    }
    if not result["causality_audit"]["all_pass"]:
        raise RuntimeError(f"causality audit failed: {result['causality_audit']}")

    pl.DataFrame(combined, infer_schema_length=None).write_parquet(
        REPORT_DIR / "dynamic_daily_ledger.parquet", compression="zstd"
    )
    pl.DataFrame(combined, infer_schema_length=None).write_csv(
        REPORT_DIR / "dynamic_daily_ledger.csv"
    )
    pl.DataFrame(regime_rows, infer_schema_length=None).write_parquet(
        REPORT_DIR / "regime_state_ledger.parquet", compression="zstd"
    )
    pl.DataFrame(regime_rows, infer_schema_length=None).write_csv(
        REPORT_DIR / "regime_state_ledger.csv"
    )
    (REPORT_DIR / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "stats": stats,
        "delta": result["dynamic"]["delta_vs_baseline"],
        "regime_on_ratio": result["dynamic"]["regime_on_ratio"],
        "causality": result["causality_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
