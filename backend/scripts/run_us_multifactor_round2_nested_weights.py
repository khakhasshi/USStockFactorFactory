#!/usr/bin/env python3
"""Run round-two nested sleeve-weight experiments for the frozen US factors.

The six return-source sleeves and all expression directions are frozen before
this run.  2020-2022 estimates four low-complexity weighting policies, 2023
selects a policy under a conservative improvement gate, and 2024 is read once
as the protocol-local final test.  Because the original factor identities came
from a 2020-2026 leaderboard audit, this remains a candidate-selection-
contaminated research diagnostic rather than a globally untouched holdout.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from app.backtest.batch import (  # noqa: E402
    BatchBacktestSpec,
    information_coefficients,
    run_cost_scenarios,
)
from app.backtest.engine import _write_artifacts  # noqa: E402
from app.backtest.vector import run_vector_cost_scenarios  # noqa: E402
from scripts.run_us_multifactor_round1_2020_2024 import (  # noqa: E402
    COMPONENTS,
    END,
    REBALANCE_EVERY,
    START,
    TOP_FRACTION,
    UNIVERSE_N,
    _materialize_components,
    _period_slice,
    _period_stats,
)


PROTOCOL = "us_frozen_multifactor_round2_nested_sleeve_weights_v1"
POLICY_LABEL = "CANDIDATE_SELECTION_CONTAMINATED_WEIGHT_OOS_DIAGNOSTIC"
TRAIN = (date(2020, 1, 1), date(2022, 12, 31))
VALIDATION = (date(2023, 1, 1), date(2023, 12, 31))
FINAL_TEST = (date(2024, 1, 1), date(2024, 12, 31))
SLIPPAGE_BPS = (0.0, 5.0, 15.0)
BORROW_BPS = 300.0
SHRINK_TO_EQUAL = 0.50
MAX_SLEEVE_WEIGHT = 0.30


SLEEVES = [
    {
        "key": "price_impact_liquidity",
        "components": {"U0002_price_impact_liquidity": 1.0},
    },
    {
        "key": "low_absolute_range",
        "components": {"U0005_low_absolute_range": 1.0},
    },
    {
        "key": "long_horizon_momentum",
        "components": {
            "U0010_momentum_x_volatility": 0.5,
            "U0013_residual_momentum": 0.5,
        },
    },
    {
        "key": "share_volume_liquidity",
        "components": {"U0001_share_volume_liquidity": 1.0},
    },
    {
        "key": "short_reversal",
        "components": {"U0006_short_reversal": 1.0},
    },
    {
        "key": "price_relationship",
        "components": {"U0012_price_relationship": 1.0},
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalise(raw: dict[str, float]) -> dict[str, float]:
    clean = {key: max(0.0, float(raw.get(key, 0.0))) for key in _sleeve_keys()}
    total = sum(clean.values())
    if total <= 1e-12:
        return _equal_weights()
    return {key: value / total for key, value in clean.items()}


def _equal_weights() -> dict[str, float]:
    return {key: 1.0 / len(SLEEVES) for key in _sleeve_keys()}


def _sleeve_keys() -> list[str]:
    return [row["key"] for row in SLEEVES]


def _cap_simplex(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Project positive normalised weights onto a capped simplex."""
    remaining = set(weights)
    output = {key: 0.0 for key in weights}
    remaining_mass = 1.0
    raw = dict(weights)
    while remaining:
        denominator = sum(raw[key] for key in remaining)
        if denominator <= 1e-12:
            equal = remaining_mass / len(remaining)
            for key in remaining:
                output[key] = equal
            break
        proposed = {
            key: remaining_mass * raw[key] / denominator for key in remaining
        }
        capped = {key for key, value in proposed.items() if value > cap + 1e-12}
        if not capped:
            output.update(proposed)
            break
        for key in capped:
            output[key] = cap
            remaining_mass -= cap
            remaining.remove(key)
    return output


def _regularise(raw: dict[str, float]) -> dict[str, float]:
    equal = _equal_weights()
    fitted = _normalise(raw)
    shrunk = {
        key: SHRINK_TO_EQUAL * equal[key] + (1.0 - SHRINK_TO_EQUAL) * fitted[key]
        for key in equal
    }
    capped = _cap_simplex(shrunk, MAX_SLEEVE_WEIGHT)
    return {key: round(value, 12) for key, value in capped.items()}


def _component_weights(sleeve_weights: dict[str, float]) -> dict[str, float]:
    output: dict[str, float] = {}
    sleeve_by_key = {row["key"]: row for row in SLEEVES}
    for sleeve_key, sleeve_weight in sleeve_weights.items():
        for component_key, within_weight in sleeve_by_key[sleeve_key]["components"].items():
            output[component_key] = float(sleeve_weight) * float(within_weight)
    if not math.isclose(sum(output.values()), 1.0, abs_tol=1e-9):
        raise ValueError("component weights do not sum to one")
    return output


def _composite_frame(frame: pl.DataFrame, sleeve_weights: dict[str, float]) -> pl.DataFrame:
    key_to_index = {component["key"]: index for index, component in enumerate(COMPONENTS)}
    expression = pl.lit(0.0)
    for key, weight in _component_weights(sleeve_weights).items():
        expression += pl.col(f"component_{key_to_index[key]}") * float(weight)
    return frame.with_columns(expression.alias("factor"))


def _spec(mode: str, start: date, end: date, slippage: tuple[float, ...]) -> BatchBacktestSpec:
    return BatchBacktestSpec(
        market="us",
        mode=mode,
        universe_n=UNIVERSE_N,
        horizon=REBALANCE_EVERY,
        top_fraction=TOP_FRACTION,
        initial_capital=1_000_000.0,
        rebalance_every=REBALANCE_EVERY,
        max_volume_participation=0.05,
        train_start=str(start),
        train_end=str(end),
        holdout_start=str(start),
        holdout_end=str(end),
        vault_start=str(end),
        vault_end=str(end),
        slippage_bps=slippage,
        borrow_cost_bps_annual=BORROW_BPS if mode == "long_short" else 0.0,
    )


def _ranking_metrics(scenario: dict, mode: str) -> dict:
    if mode == "long_only":
        return {
            "ann_return": float(scenario["active_ann_return"]),
            "sharpe": float(scenario["active_sharpe"]),
            "max_drawdown": float(scenario["active_max_drawdown"]),
            "basis": "active",
        }
    return {
        "ann_return": float(scenario["ann_return"]),
        "sharpe": float(scenario["sharpe"]),
        "max_drawdown": float(scenario["max_drawdown"]),
        "basis": "net",
    }


def _fit_policies(component_frame: pl.DataFrame, mode: str) -> tuple[dict, list[dict]]:
    diagnostics: list[dict] = []
    for sleeve in SLEEVES:
        weights = {key: 0.0 for key in _sleeve_keys()}
        weights[sleeve["key"]] = 1.0
        frame = _composite_frame(component_frame, weights)
        scenarios = run_vector_cost_scenarios(
            frame,
            direction=1,
            spec=_spec(mode, *TRAIN, (15.0,)),
            capture_periods=True,
        )
        scenario = scenarios["15"]
        periods = scenario.pop("period_rows")
        return_key = "active_return" if mode == "long_only" else "net_return"
        values = [float(row[return_key]) for row in periods]
        volatility = statistics.stdev(values) if len(values) > 1 else 0.0
        yearly = {
            str(year): _period_stats(
                _period_slice(periods, date(year, 1, 1), date(year, 12, 31)), return_key
            )
            for year in range(2020, 2023)
        }
        ic = information_coefficients(
            frame,
            start=str(TRAIN[0]),
            end=str(TRAIN[1]),
            horizon=REBALANCE_EVERY,
            universe_n=UNIVERSE_N,
            direction=1,
        )
        diagnostics.append({
            "sleeve": sleeve["key"],
            "mode": mode,
            "volatility_5d": round(volatility, 10),
            "worst_year_sharpe": round(min(value["sharpe"] for value in yearly.values()), 6),
            "median_year_sharpe": round(statistics.median(value["sharpe"] for value in yearly.values()), 6),
            "rank_ic_mean": ic["rank_ic_mean"],
            "rank_icir": ic["rank_icir"],
            "training_15bps": scenario,
            "yearly": yearly,
        })

    by_key = {row["sleeve"]: row for row in diagnostics}
    inverse_vol = {
        key: 1.0 / max(1e-8, by_key[key]["volatility_5d"])
        for key in _sleeve_keys()
    }
    positive_ic = {
        key: max(0.0, float(by_key[key]["rank_icir"]))
        for key in _sleeve_keys()
    }
    robust_quality = {
        key: math.exp(max(-1.0, min(2.0, float(by_key[key]["worst_year_sharpe"]))))
        for key in _sleeve_keys()
    }
    inv_norm = _normalise(inverse_vol)
    ic_norm = _normalise(positive_ic)
    policies = {
        "source_equal": _equal_weights(),
        "train_inverse_vol": _regularise(inverse_vol),
        "train_positive_icir": _regularise(positive_ic),
        "train_worst_year_quality": _regularise(robust_quality),
        "train_risk_ic_ensemble": _regularise({
            key: 0.5 * inv_norm[key] + 0.5 * ic_norm[key]
            for key in _sleeve_keys()
        }),
    }
    return policies, diagnostics


def _evaluate_window(
    component_frame: pl.DataFrame,
    *,
    mode: str,
    policies: dict[str, dict[str, float]],
    window_name: str,
    window: tuple[date, date],
    slippage: tuple[float, ...],
) -> tuple[list[dict], dict[str, pl.DataFrame]]:
    rows: list[dict] = []
    frames: dict[str, pl.DataFrame] = {}
    for policy, sleeve_weights in policies.items():
        frame = _composite_frame(component_frame, sleeve_weights)
        frames[policy] = frame
        scenarios = run_vector_cost_scenarios(
            frame,
            direction=1,
            spec=_spec(mode, *window, slippage),
            capture_periods=False,
        )
        for bps, scenario in scenarios.items():
            ranking = _ranking_metrics(scenario, mode)
            rows.append({
                "window": window_name,
                "mode": mode,
                "policy": policy,
                "slippage_bps": float(bps),
                "ranking_basis": ranking["basis"],
                "ranking_ann_return": ranking["ann_return"],
                "ranking_sharpe": ranking["sharpe"],
                "ranking_max_drawdown": ranking["max_drawdown"],
                "raw_ann_return": scenario["ann_return"],
                "raw_sharpe": scenario["sharpe"],
                "raw_max_drawdown": scenario["max_drawdown"],
                "avg_daily_turnover": scenario["avg_daily_turnover"],
                "total_execution_cost": scenario["total_execution_cost"],
            })
    return rows, frames


def _select_policy(validation_rows: list[dict], mode: str) -> dict:
    rows = [
        row for row in validation_rows
        if row["mode"] == mode and math.isclose(float(row["slippage_bps"]), 15.0)
    ]
    baseline = next(row for row in rows if row["policy"] == "source_equal")
    candidates = []
    for row in rows:
        if row["policy"] == "source_equal":
            continue
        rules = {
            "sharpe_margin_0_15": row["ranking_sharpe"] >= baseline["ranking_sharpe"] + 0.15,
            "return_not_lower": row["ranking_ann_return"] >= baseline["ranking_ann_return"],
            "drawdown_within_125pct": row["ranking_max_drawdown"] <= max(
                0.02, baseline["ranking_max_drawdown"] * 1.25
            ),
            "positive_sharpe": row["ranking_sharpe"] > 0.0,
        }
        candidates.append({"row": row, "rules": rules, "eligible": all(rules.values())})
    eligible = [item for item in candidates if item["eligible"]]
    if eligible:
        winner = max(
            eligible,
            key=lambda item: (
                item["row"]["ranking_sharpe"],
                item["row"]["ranking_ann_return"],
                -item["row"]["ranking_max_drawdown"],
            ),
        )["row"]["policy"]
        reason = "learned_policy_passed_all_2023_improvement_gates"
    else:
        winner = "source_equal"
        reason = "no_learned_policy_passed_all_2023_improvement_gates"
    return {
        "mode": mode,
        "selected_policy": winner,
        "fallback_used": winner == "source_equal",
        "reason": reason,
        "baseline": baseline,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, default=Path(
        "var/reports/external-joint-ashare-us-to-us-long-only-vector-2020-latest-20260821/protocol.json"
    ))
    parser.add_argument("--round1-summary", type=Path, default=Path(
        "var/reports/us-multifactor-round1-2020-2024-20260822/summary.json"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refuse to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    round1 = json.loads(args.round1_summary.read_text(encoding="utf-8"))
    if round1.get("status") != "complete":
        raise ValueError("round-one summary is not complete")
    protocol = {
        "protocol": PROTOCOL,
        "policy_label": POLICY_LABEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "factor_identity_warning": (
            "factor identities and directions were selected using leaderboards that overlap "
            "2020-2026; only weight fitting is nested here"
        ),
        "windows": {
            "weight_fit": {"start": str(TRAIN[0]), "end": str(TRAIN[1])},
            "policy_validation": {"start": str(VALIDATION[0]), "end": str(VALIDATION[1])},
            "protocol_local_final_test": {"start": str(FINAL_TEST[0]), "end": str(FINAL_TEST[1])},
        },
        "selection_gate": {
            "sharpe_improvement": ">= 0.15",
            "annualized_return": ">= source_equal",
            "max_drawdown": "<= max(2%, 125% of source_equal)",
            "sharpe": "> 0",
            "fallback": "source_equal",
        },
        "regularisation": {
            "shrink_to_equal": SHRINK_TO_EQUAL,
            "max_sleeve_weight": MAX_SLEEVE_WEIGHT,
            "all_six_sleeves_retained": True,
        },
        "universe_n": UNIVERSE_N,
        "top_fraction": TOP_FRACTION,
        "rebalance_every": REBALANCE_EVERY,
        "slippage_bps": list(SLIPPAGE_BPS),
        "long_short_borrow_cost_bps_annual": BORROW_BPS,
        "sleeves": SLEEVES,
        "source_protocol": str(args.source_protocol.resolve()),
        "source_protocol_sha256": _sha256(args.source_protocol),
        "round1_summary": str(args.round1_summary.resolve()),
        "round1_summary_sha256": _sha256(args.round1_summary),
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    _write_json(output / "protocol.json", protocol)

    print("[1/5] materialize frozen components", flush=True)
    component_frame, frame_summary = _materialize_components(
        panel_glob=str(source_protocol["panel_glob"]),
        dsl_fields=list(source_protocol["dsl_fields"]),
    )
    protocol["frame"] = frame_summary
    _write_json(output / "protocol.json", protocol)

    fitted: dict[str, dict] = {}
    training_diagnostics: list[dict] = []
    print("[2/5] fit low-complexity policies on 2020-2022", flush=True)
    for mode in ("long_only", "long_short"):
        policies, diagnostics = _fit_policies(component_frame, mode)
        fitted[mode] = policies
        training_diagnostics.extend(diagnostics)
        print(f"  {mode}: {len(policies)} policies", flush=True)
    _write_json(output / "fitted_policies.json", fitted)
    _write_json(output / "training_sleeve_diagnostics.json", training_diagnostics)

    print("[3/5] select policy on 2023 only", flush=True)
    validation_rows: list[dict] = []
    for mode in ("long_only", "long_short"):
        rows, _ = _evaluate_window(
            component_frame,
            mode=mode,
            policies=fitted[mode],
            window_name="validation_2023",
            window=VALIDATION,
            slippage=(15.0,),
        )
        validation_rows.extend(rows)
    selections = {
        mode: _select_policy(validation_rows, mode)
        for mode in ("long_only", "long_short")
    }
    _write_json(output / "selection_decisions.json", selections)
    print(
        "  selected " + json.dumps(
            {mode: row["selected_policy"] for mode, row in selections.items()},
            ensure_ascii=False,
        ),
        flush=True,
    )

    print("[4/5] read 2024 final test and replay step engine", flush=True)
    final_rows: list[dict] = []
    final_frames: dict[tuple[str, str], pl.DataFrame] = {}
    for mode in ("long_only", "long_short"):
        selected = selections[mode]["selected_policy"]
        test_policies = {
            "source_equal": fitted[mode]["source_equal"],
            selected: fitted[mode][selected],
        }
        rows, frames = _evaluate_window(
            component_frame,
            mode=mode,
            policies=test_policies,
            window_name="final_test_2024",
            window=FINAL_TEST,
            slippage=SLIPPAGE_BPS,
        )
        final_rows.extend(rows)
        for policy, frame in frames.items():
            final_frames[(mode, policy)] = frame

    step_rows: list[dict] = []
    step_jobs = []
    for mode in ("long_only", "long_short"):
        selected = selections[mode]["selected_policy"]
        for policy in dict.fromkeys(["source_equal", selected]):
            step_jobs.append((mode, policy))
    for index, (mode, policy) in enumerate(step_jobs, start=1):
        print(f"  step [{index}/{len(step_jobs)}] {mode} {policy}", flush=True)
        scenario = run_cost_scenarios(
            final_frames[(mode, policy)],
            direction=1,
            spec=_spec(mode, *FINAL_TEST, (15.0,)),
            capture_detail=True,
        )["15"]
        result = scenario.pop("result")
        artifact_dir = output / "step_ledgers" / mode / policy
        manifest = _write_artifacts(result, artifact_dir)
        step_rows.append({
            "mode": mode,
            "policy": policy,
            **scenario,
            "integrity_all_pass": scenario["integrity"]["all_pass"],
            "artifact_dir": str(artifact_dir),
            "artifact_manifest": manifest,
        })

    _write_csv(output / "validation_results.csv", validation_rows)
    _write_csv(output / "final_test_results.csv", final_rows)
    _write_json(output / "step_results.json", step_rows)

    print("[5/5] render report", flush=True)
    lines = [
        "# 美股多因子第二轮：嵌套袖套权重实验",
        "",
        f"- 协议：`{PROTOCOL}`",
        "- 2020-2022：只拟合权重政策；2023：只选择政策；2024：协议内最终测试。",
        "- 六个收益来源全部保留；学习权重向等权收缩50%，单袖套不超过30%。",
        "- 因子身份来自覆盖2020-2026的既有榜单，所以2024并非全局未见样本。",
        "",
        "## 2023选择",
        "",
        "| 模式 | 方案 | 主动/净年化 | Sharpe | 最大回撤 | 入选 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for mode in ("long_only", "long_short"):
        selected = selections[mode]["selected_policy"]
        for row in sorted(
            [value for value in validation_rows if value["mode"] == mode],
            key=lambda value: value["ranking_sharpe"],
            reverse=True,
        ):
            lines.append(
                f"| {mode} | {row['policy']} | {row['ranking_ann_return']:.2%} | "
                f"{row['ranking_sharpe']:.3f} | {row['ranking_max_drawdown']:.2%} | "
                f"{'是' if row['policy'] == selected else ''} |"
            )
    lines.extend([
        "",
        "## 2024最终测试（15 BPS）",
        "",
        "| 模式 | 方案 | 主动/净年化 | Sharpe | 最大回撤 | 日均换手 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in final_rows:
        if math.isclose(float(row["slippage_bps"]), 15.0):
            lines.append(
                f"| {row['mode']} | {row['policy']} | {row['ranking_ann_return']:.2%} | "
                f"{row['ranking_sharpe']:.3f} | {row['ranking_max_drawdown']:.2%} | "
                f"{row['avg_daily_turnover']:.3%} |"
            )
    lines.extend([
        "",
        "## 2024步进复测（组合总收益口径）",
        "",
        "| 模式 | 方案 | 总年化 | Sharpe | 最大回撤 | 成本 | 完整性 |",
        "|---|---|---:|---:|---:|---:|---|",
    ])
    for row in step_rows:
        lines.append(
            f"| {row['mode']} | {row['policy']} | {row['ann_return']:.2%} | "
            f"{row['sharpe']:.3f} | {row['max_drawdown']:.2%} | "
            f"{row['total_execution_cost']:,.0f} | {row['integrity_all_pass']} |"
        )
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {
        "protocol": PROTOCOL,
        "status": "complete",
        "policy_label": POLICY_LABEL,
        "selected": {mode: row["selected_policy"] for mode, row in selections.items()},
        "fallback_used": {mode: row["fallback_used"] for mode, row in selections.items()},
        "validation_rows": len(validation_rows),
        "final_test_rows": len(final_rows),
        "step_jobs": [{"mode": row["mode"], "policy": row["policy"]} for row in step_rows],
    }
    _write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
