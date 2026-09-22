#!/usr/bin/env python3
"""Run the preregistered 60d / 10% / [0.50, 1.00] core vol target."""

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

from backend.app.backtest.engine import _performance_from_nav, run_multi_factor_backtest  # noqa: E402


BASELINE_ID = 732
WINDOW = 60
TARGET_VOL = 0.10
MIN_SCALE = 0.50
MAX_SCALE = 1.00
REPORT_DIR = ROOT / "var/reports/us-core-74-26-vol-target-20260827"
EXPRESSION_21 = "-rank(ts_mean((high - low) / amount, 20))"
EXPRESSION_19 = "rank((-ts_mean((close / amount), 200)))"


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
        "monte_carlo_enabled": True,
        "monte_carlo_simulations": 2000,
        "monte_carlo_block_size_sessions": 20,
        "monte_carlo_seed": 20260824,
    })
    return output


def _returns_from_nav(nav: list[float]) -> np.ndarray:
    values = np.asarray(nav, dtype=np.float64)
    return values / np.concatenate([[1.0], values[:-1]]) - 1.0


def _scale_schedule(
    dates: list[str], underlying_returns: np.ndarray
) -> tuple[list[dict[str, Any]], dict[date, float]]:
    rows = []
    mapping: dict[date, float] = {}
    for index, trade_date in enumerate(dates):
        realized_vol = None
        scale = 1.0
        if index + 1 >= WINDOW:
            sample = underlying_returns[index - WINDOW + 1 : index + 1]
            realized_vol = float(np.std(sample, ddof=1) * math.sqrt(252.0))
            raw_scale = TARGET_VOL / realized_vol if realized_vol > 1e-12 else 1.0
            scale = float(np.clip(raw_scale, MIN_SCALE, MAX_SCALE))
        rows.append({
            "trade_date": trade_date,
            "window_start": dates[index - WINDOW + 1] if index + 1 >= WINDOW else None,
            "window_end": trade_date if index + 1 >= WINDOW else None,
            "underlying_fee_after_vol_60d_at_close": realized_vol,
            "target_scale_next_open": scale,
            "applied_scale_today": rows[-1]["target_scale_next_open"] if rows else 1.0,
            "source_max_index": index,
            "applied_source_max_index": index - 1,
        })
        mapping[date.fromisoformat(trade_date)] = scale
    return rows, mapping


def _subset_performance(returns: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    selected = returns[mask]
    nav = np.cumprod(1.0 + selected)
    return {
        "sessions": int(selected.size),
        "total_return": round(float(nav[-1] - 1.0), 8),
        **_performance_from_nav(nav.tolist()),
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    baseline = _api_backtest(BASELINE_ID)
    if baseline["status"] != "done" or not baseline["result"]["integrity"]["all_pass"]:
        raise RuntimeError("frozen static baseline #732 is not audited and complete")
    params = baseline["params"]
    dates = baseline["result"]["curve"]["dates"]
    baseline_returns = _returns_from_nav(baseline["result"]["curve"]["equity"])
    schedule_rows, schedule = _scale_schedule(dates, baseline_returns)

    print("run 74/26 event-ledger portfolio with causal daily gross scaling", flush=True)
    dynamic = run_multi_factor_backtest(
        [
            {"name": "因子21", "expression": EXPRESSION_21, "weight": 74, "direction": 1},
            {"name": "因子19", "expression": EXPRESSION_19, "weight": 26, "direction": 1},
        ],
        initial_capital=1_000_000.0,
        artifact_dir=REPORT_DIR / "backtest",
        _target_gross_scale_by_signal_date=schedule,
        **_common_kwargs(params),
    )
    if not dynamic["integrity"]["all_pass"]:
        raise RuntimeError("dynamic vol-target event ledger failed integrity")
    if dynamic["curve"]["dates"] != dates:
        raise RuntimeError("dynamic and static date paths differ")

    applied = np.asarray(
        [float(row["applied_scale_today"]) for row in schedule_rows], dtype=np.float64
    )
    signal_scale = np.asarray(
        [float(row["target_scale_next_open"]) for row in schedule_rows], dtype=np.float64
    )
    dynamic_returns = _returns_from_nav(dynamic["curve"]["equity"])
    reduced = applied < 1.0 - 1e-12
    full = ~reduced
    annual_scale = {}
    for year in sorted({value[:4] for value in dates}):
        mask = np.asarray([value.startswith(year) for value in dates], dtype=bool)
        annual_scale[year] = {
            "mean": round(float(np.mean(applied[mask])), 8),
            "below_1_ratio": round(float(np.mean(reduced[mask])), 8),
        }

    lag_mismatches = int(int(applied[0] != 1.0) + sum(
        int(abs(applied[index] - signal_scale[index - 1]) > 1e-12)
        for index in range(1, len(applied))
    ))
    timing_violations = int(sum(
        row["signal_date"] >= row["trade_date"]
        for row in dynamic["trades"]
        if row.get("reason") == "factor_rebalance_next_open"
    ))
    scale_trade_errors = []
    scale_by_date = {
        row["trade_date"]: float(row["target_scale_next_open"])
        for row in schedule_rows
    }
    for trade in dynamic["trades"]:
        if trade.get("reason") != "factor_rebalance_next_open":
            continue
        reported = trade.get("target_gross_scale")
        if reported is not None:
            scale_trade_errors.append(
                abs(float(reported) - scale_by_date[trade["signal_date"]])
            )

    annual = dynamic["stability_analysis"]["annual"]
    baseline_annual = baseline["result"]["stability_analysis"]["annual"]
    baseline_annual_by_period = {row["period"]: row for row in baseline_annual}
    annual_comparison = []
    for row in annual:
        frozen_row = baseline_annual_by_period[row["period"]]
        annual_comparison.append({
            "period": row["period"],
            "is_ytd": row["is_ytd"],
            "dynamic_total_return": row["total_return"],
            "dynamic_sharpe": row["sharpe"],
            "dynamic_max_drawdown": row["max_drawdown"],
            "baseline_total_return": frozen_row["total_return"],
            "baseline_sharpe": frozen_row["sharpe"],
            "baseline_max_drawdown": frozen_row["max_drawdown"],
            "delta_total_return": round(
                float(row["total_return"]) - float(frozen_row["total_return"]), 8
            ),
            "delta_sharpe": round(float(row["sharpe"]) - float(frozen_row["sharpe"]), 8),
            "delta_max_drawdown": round(
                float(row["max_drawdown"]) - float(frozen_row["max_drawdown"]), 8
            ),
        })
    stats = dynamic["stats"]
    baseline_stats = baseline["result"]["stats"]
    bootstrap = dynamic["monte_carlo"]
    result = {
        "schema": "factorfactory.core-vol-target/v1",
        "protocol": "step_event_v2_weighted_sleeves_vol_target_v1",
        "generated_at": "2026-08-27",
        "predeclared_parameters": {
            "window_sessions": WINDOW,
            "annual_target_volatility": TARGET_VOL,
            "min_scale": MIN_SCALE,
            "max_scale": MAX_SCALE,
            "factor_weights": {"factor_21": 0.74, "factor_19": 0.26},
            "leverage_allowed": False,
            "optimization_performed": False,
            "volatility_estimator_source": (
                "frozen unscaled 74/26 core fee-after return ledger; "
                "observed through t close"
            ),
        },
        "actual_period": {"start": dates[0], "end": dates[-1], "sessions": len(dates)},
        "baseline": {
            "backtest_id": BASELINE_ID,
            "stats": baseline_stats,
            "annual": baseline_annual,
            "integrity_all_pass": baseline["result"]["integrity"]["all_pass"],
        },
        "dynamic": {
            "stats": stats,
            "annual": annual,
            "annual_comparison": annual_comparison,
            "calendar_2022": next(row for row in annual if row["period"] == "2022"),
            "delta_vs_baseline": {
                "cagr": round(float(stats["ann_ret"]) - float(baseline_stats["ann_ret"]), 8),
                "sharpe": round(float(stats["sharpe"]) - float(baseline_stats["sharpe"]), 8),
                "max_drawdown": round(float(stats["max_dd"]) - float(baseline_stats["max_dd"]), 8),
            },
            "scale_distribution": {
                "mean": round(float(np.mean(applied)), 8),
                "median": round(float(np.median(applied)), 8),
                "p05": round(float(np.quantile(applied, 0.05)), 8),
                "p95": round(float(np.quantile(applied, 0.95)), 8),
                "below_1_ratio": round(float(np.mean(applied < 1.0 - 1e-12)), 8),
                "below_0_75_ratio": round(float(np.mean(applied < 0.75)), 8),
                "at_0_50_ratio": round(float(np.mean(np.isclose(applied, 0.50, atol=1e-12))), 8),
                "by_year": annual_scale,
            },
            "performance_when_reduced": _subset_performance(dynamic_returns, reduced),
            "performance_when_full": _subset_performance(dynamic_returns, full),
            "static_baseline_when_reduced": _subset_performance(baseline_returns, reduced),
            "static_baseline_when_full": _subset_performance(baseline_returns, full),
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
            "signal_definition": "std(static core fee-after r[t-59:t], ddof=1)*sqrt(252)",
            "decision_time": "t_close_after_realized_fee_after_core_return",
            "application_time": "t_plus_1_raw_open",
            "first_59_signal_dates_full_risk": bool(all(signal_scale[:59] == 1.0)),
            "lag_mismatch_count": lag_mismatches,
            "trade_signal_date_violations": timing_violations,
            "trade_target_scale_max_error": max(scale_trade_errors, default=0.0),
            "future_data_used": False,
            "all_pass": (
                lag_mismatches == 0
                and timing_violations == 0
                and max(scale_trade_errors, default=0.0) <= 1e-12
                and dynamic["integrity"]["all_pass"]
            ),
        },
        "ledger_integrity": dynamic["integrity"],
    }
    if not result["causality_audit"]["all_pass"]:
        raise RuntimeError(f"causality audit failed: {result['causality_audit']}")

    pl.DataFrame(schedule_rows, infer_schema_length=None).write_parquet(
        REPORT_DIR / "vol_target_state_ledger.parquet", compression="zstd"
    )
    pl.DataFrame(schedule_rows, infer_schema_length=None).write_csv(
        REPORT_DIR / "vol_target_state_ledger.csv"
    )
    (REPORT_DIR / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "stats": stats,
        "delta": result["dynamic"]["delta_vs_baseline"],
        "scale": result["dynamic"]["scale_distribution"],
        "causality": result["causality_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
