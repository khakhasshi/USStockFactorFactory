#!/usr/bin/env python3
"""Optimise one frozen A-share candidate pool with purged walk-forward V3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.backtest.batch import BatchBacktestSpec, run_cost_scenarios  # noqa: E402
from app.config import ASHARE_PANEL_GLOB  # noqa: E402
from app.factors.walk_forward_optimizer import (  # noqa: E402
    DEFAULT_FOLDS,
    WalkForwardConfig,
    search_walk_forward_weights,
)
from app.factors.weight_optimizer import (  # noqa: E402
    WeightSearchConfig,
    apply_composite_weights,
    build_slices,
    composite_expression,
    materialize_component_frame,
)
from scripts.optimize_ashare_factor_weights import (  # noqa: E402
    _component_from_candidate,
)
from scripts.validate_ashare_factor_pair import _period_stats  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = (
    PROJECT_ROOT
    / "var"
    / "reports"
    / "ashare-factor-weight-search-clean-20260810"
    / "candidate_snapshot.json"
)


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_pool(path: Path) -> tuple[list, dict]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("holdout_or_vault_read") is not False:
        raise ValueError("候选快照没有声明 holdout_or_vault_read=false")
    if snapshot.get("track") != "current_protocol_clean":
        raise ValueError("walk-forward 正式路线只接受 current_protocol_clean 快照")
    selected = ((snapshot.get("selection") or {}).get("selected") or [])
    if not 2 <= len(selected) <= 8:
        raise ValueError("冻结候选池数量必须在 2–8")
    components = [_component_from_candidate(row) for row in selected]
    return components, {
        "path": str(path.resolve()),
        "sha256": _hash_file(path),
        "created_at": snapshot.get("created_at"),
        "track": snapshot.get("track"),
        "selected_component_keys": [row["component_key"] for row in selected],
        "selected_factor_ids": [row.get("factor_id") for row in selected],
        "selected_mechanisms": [row["mechanism_family"] for row in selected],
    }


def _event_replay(frame, best: dict, config: WeightSearchConfig) -> dict:
    composite = apply_composite_weights(frame, best["weights"])
    output = {"validation_folds": {}, "full_pre2023": None}
    for fold in DEFAULT_FOLDS:
        spec = BatchBacktestSpec(
            market="ashare",
            mode="long_only",
            universe_n=config.universe_n,
            horizon=config.horizon,
            top_fraction=config.top_fraction,
            rebalance_every=config.horizon,
            holdout_start=str(fold.validation_start),
            holdout_end=str(fold.validation_end),
            slippage_bps=(15.0,),
            fee_profile="ashare_wan2_no_min_v1",
        )
        output["validation_folds"][fold.name] = run_cost_scenarios(
            composite,
            direction=1,
            spec=spec,
            capture_detail=False,
        )["15"]
    full_spec = BatchBacktestSpec(
        market="ashare",
        mode="long_only",
        universe_n=config.universe_n,
        horizon=config.horizon,
        top_fraction=config.top_fraction,
        rebalance_every=config.horizon,
        holdout_start=str(DEFAULT_FOLDS[0].validation_start),
        holdout_end=str(DEFAULT_FOLDS[-1].validation_end),
        slippage_bps=(0.0, 5.0, 15.0),
        fee_profile="ashare_wan2_no_min_v1",
    )
    output["full_pre2023"] = run_cost_scenarios(
        composite,
        direction=1,
        spec=full_spec,
        capture_detail=False,
    )
    return {
        "engine": "StepEventBacktester",
        "signal": "t_close",
        "fill": "t_plus_1_raw_open",
        "fee_profile": "ashare_wan2_no_min_v1",
        "purpose": "pre2023_walk_forward_finalist_replay_not_search_objective",
        **output,
    }


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _event_calibrate(
    frame,
    result: dict,
    config: WeightSearchConfig,
    *,
    top_k: int,
) -> dict:
    """Re-rank vector finalists on one continuous pre-2023 event replay."""
    finalists = result["results"][:max(2, min(top_k, len(result["results"])))]
    rows = []
    for index, candidate in enumerate(finalists, start=1):
        print(
            f"[event calibration {index}/{len(finalists)}] "
            f"vector rank {candidate['rank']}",
            flush=True,
        )
        composite = apply_composite_weights(frame, candidate["weights"])
        spec = BatchBacktestSpec(
            market="ashare",
            mode="long_only",
            universe_n=config.universe_n,
            horizon=config.horizon,
            top_fraction=config.top_fraction,
            rebalance_every=config.horizon,
            holdout_start=str(DEFAULT_FOLDS[0].validation_start),
            holdout_end=str(DEFAULT_FOLDS[-1].validation_end),
            slippage_bps=(15.0,),
            fee_profile="ashare_wan2_no_min_v1",
        )
        scenario = run_cost_scenarios(
            composite,
            direction=1,
            spec=spec,
            capture_detail=True,
        )["15"]
        event_result = scenario.pop("result")
        fold_metrics = {
            fold.name: _period_stats(
                event_result["daily_steps"],
                start=fold.validation_start,
                end=fold.validation_end,
            )
            for fold in DEFAULT_FOLDS
        }
        sharpes = [float(row["sharpe"]) for row in fold_metrics.values()]
        drawdowns = [float(row["max_drawdown"]) for row in fold_metrics.values()]
        worst_sharpe = min(sharpes)
        median_sharpe = statistics.median(sharpes)
        positive_rate = sum(value > 0 for value in sharpes) / len(sharpes)
        max_drawdown = max(drawdowns)
        event_score = (
            0.30 * _clip(worst_sharpe)
            + 0.20 * _clip(median_sharpe)
            + 0.20 * _clip(float(scenario["sharpe"]))
            + 0.10 * (2.0 * positive_rate - 1.0)
            + 0.10 * (
                1.0 - _clip(max_drawdown / 0.35, 0.0, 2.0)
            )
            + 0.10 * (
                1.0
                - _clip(float(scenario["max_drawdown"]) / 0.35, 0.0, 2.0)
            )
        )
        combined_score = 0.75 * event_score + 0.25 * float(
            candidate["robust_score"]
        )
        rows.append({
            "vector_rank": int(candidate["rank"]),
            "weights": candidate["weights"],
            "vector_robust_score": candidate["robust_score"],
            "event_score": round(event_score, 8),
            "combined_score": round(combined_score, 8),
            "worst_fold_sharpe": round(worst_sharpe, 6),
            "median_fold_sharpe": round(median_sharpe, 6),
            "positive_fold_rate": round(positive_rate, 6),
            "max_fold_drawdown": round(max_drawdown, 6),
            "full_2015_2022": {
                key: value for key, value in scenario.items()
                if key not in {"integrity"}
            },
            "event_integrity": scenario["integrity"],
            "folds": fold_metrics,
        })
    rows.sort(
        key=lambda row: (
            row["combined_score"],
            row["worst_fold_sharpe"],
            row["full_2015_2022"]["sharpe"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(rows, start=1):
        row["event_rank"] = rank
    winner = next(
        candidate for candidate in finalists
        if candidate["weights"] == rows[0]["weights"]
    )
    return {
        "protocol": "topk_continuous_event_calibration_pre2023_v1",
        "selection_scope": "2015_to_2022_only",
        "holdout_or_vault_read": False,
        "top_k": len(finalists),
        "score_semantics": (
            "75pct_event_worst_fold_median_full_sharpe_drawdown_"
            "plus_25pct_vector_walk_forward"
        ),
        "winner": rows[0],
        "winner_result": winner,
        "results": rows,
    }


def _promotion_gate(result: dict) -> dict:
    replay = result["event_verification"]
    folds = list(replay["validation_folds"].values())
    sharpes = [float(row["sharpe"]) for row in folds]
    drawdowns = [float(row["max_drawdown"]) for row in folds]
    integrity = all(bool(row["integrity"]["all_pass"]) for row in folds)
    full = replay["full_pre2023"]["15"]
    observed = {
        "worst_fold_sharpe_15bps": round(min(sharpes), 6),
        "median_fold_sharpe_15bps": round(statistics.median(sharpes), 6),
        "positive_fold_rate_15bps": round(
            sum(value > 0 for value in sharpes) / len(sharpes), 6
        ),
        "max_fold_drawdown_15bps": round(max(drawdowns), 6),
        "full_2015_2022_sharpe_15bps": float(full["sharpe"]),
        "full_2015_2022_max_drawdown_15bps": float(full["max_drawdown"]),
        "event_integrity_all_pass": integrity,
        "return_source_independence": result["best"][
            "return_source_independence"
        ],
    }
    rules = {
        "event_integrity": integrity,
        "worst_fold_sharpe": observed["worst_fold_sharpe_15bps"] >= -0.20,
        "median_fold_sharpe": observed["median_fold_sharpe_15bps"] >= 0.50,
        "positive_fold_rate": observed["positive_fold_rate_15bps"] >= 0.75,
        "max_fold_drawdown": observed["max_fold_drawdown_15bps"] <= 0.35,
        "full_pre2023_sharpe": observed["full_2015_2022_sharpe_15bps"] >= 0.50,
        "full_pre2023_drawdown": (
            observed["full_2015_2022_max_drawdown_15bps"] <= 0.35
        ),
        "return_source_independence": (
            observed["return_source_independence"] >= 0.35
        ),
    }
    failed = sorted(key for key, passed in rules.items() if not passed)
    return {
        "decision": (
            "PRIVATE_VALIDATION_CANDIDATE"
            if not failed
            else "RESEARCH_ONLY_BLOCKED"
        ),
        "eligible_for_private_validation": not failed,
        "production_eligible": False,
        "policy": {
            "worst_fold_sharpe_15bps": ">=-0.20",
            "median_fold_sharpe_15bps": ">=0.50",
            "positive_fold_rate_15bps": ">=0.75",
            "max_fold_drawdown_15bps": "<=0.35",
            "full_2015_2022_sharpe_15bps": ">=0.50",
            "full_2015_2022_max_drawdown_15bps": "<=0.35",
            "return_source_independence": ">=0.35",
        },
        "observed": observed,
        "rules": rules,
        "failed_rules": failed,
    }


def _write_outputs(output_dir: Path, payload: dict, candidate_manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "candidate_snapshot.json").write_text(
        json.dumps(
            candidate_manifest,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    (output_dir / "search_results.json").write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    fields = [
        "rank", "robust_score", "active_factors", "weights",
        "worst_validation_core_score", "median_validation_core_score",
        "worst_validation_active_sharpe", "median_validation_active_sharpe",
        "worst_validation_long_sharpe", "median_validation_long_sharpe",
        "positive_active_folds", "positive_long_folds",
        "mean_degradation_penalty", "regime_sharpe_dispersion",
        "effective_factor_count", "active_mechanism_groups",
        "return_source_independence",
    ]
    with (output_dir / "leaderboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload["results"]:
            writer.writerow({
                key: json.dumps(row[key]) if key == "weights" else row[key]
                for key in fields
            })


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--output-dir")
    parser.add_argument("--coarse-step", type=float, default=0.10)
    parser.add_argument("--refine-step", type=float, default=0.02)
    parser.add_argument("--min-factors", type=int, default=2)
    parser.add_argument("--max-factors", type=int, default=5)
    parser.add_argument("--min-weight", type=float, default=0.10)
    parser.add_argument("--max-weight", type=float, default=0.50)
    parser.add_argument("--max-mechanism-weight", type=float, default=0.50)
    parser.add_argument("--min-mechanisms", type=int, default=3)
    parser.add_argument("--max-active-pair-similarity", type=float, default=0.85)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--event-calibration-top-k", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    snapshot_path = Path(args.candidate_snapshot).resolve()
    components, source_manifest = _load_frozen_pool(snapshot_path)
    config = WeightSearchConfig(
        min_factors=args.min_factors,
        max_factors=min(args.max_factors, len(components)),
        coarse_step=args.coarse_step,
        refine_step=args.refine_step,
        min_active_weight=args.min_weight,
        max_weight=args.max_weight,
        max_mechanism_weight=args.max_mechanism_weight,
        max_active_pair_similarity=args.max_active_pair_similarity,
        cost_bps=args.cost_bps,
        stress_cost_bps=args.stress_cost_bps,
    )
    walk_forward = WalkForwardConfig(min_mechanism_groups=args.min_mechanisms)
    print(f"[1/4] materialize frozen pool ({len(components)} factors)", flush=True)
    frame, provenance = materialize_component_frame(
        components,
        universe_n=config.universe_n,
        horizon=config.horizon,
        panel_glob=ASHARE_PANEL_GLOB,
        include_execution_columns=True,
    )
    print("[2/4] build purged non-overlapping slices", flush=True)
    slices, slice_summary = build_slices(
        frame,
        component_count=len(components),
        horizon=config.horizon,
        universe_n=config.universe_n,
    )
    print("[3/4] purged walk-forward coarse grid + refinement", flush=True)
    result = search_walk_forward_weights(
        slices,
        len(components),
        config,
        factor_groups=[row.mechanism_family for row in components],
        walk_forward=walk_forward,
    )
    result["provenance"] = provenance
    result["slice_summary"] = slice_summary
    result["components"] = [asdict(row) for row in components]
    result["vector_stage_best"] = result["best"]
    print("[3b/4] exact event calibration of vector finalists", flush=True)
    event_calibration = _event_calibrate(
        frame,
        result,
        config,
        top_k=args.event_calibration_top_k,
    )
    result["best"] = event_calibration.pop("winner_result")
    result["event_calibration"] = event_calibration
    result["best_composite_expression"] = composite_expression(
        components, result["best"]["weights"]
    )
    result["best"]["components"] = [
        {
            **asdict(component),
            "weight": result["best"]["weights"][index],
        }
        for index, component in enumerate(components)
        if result["best"]["weights"][index] > 1e-12
    ]
    print("[4/4] replay winner in all pre-2023 validation folds", flush=True)
    result["event_verification"] = _event_replay(frame, result["best"], config)
    result["promotion_gate"] = _promotion_gate(result)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "var" / "reports" / f"ashare-walk-forward-weight-search-{stamp}"
    )
    candidate_manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "market": "ashare",
        "track": "walk_forward_v3_same_frozen_pool",
        "source_snapshot": source_manifest,
        "llm_used": False,
        "holdout_or_vault_read": False,
        "folds": [asdict(row) for row in DEFAULT_FOLDS],
        "components": [asdict(row) for row in components],
    }
    _write_outputs(output_dir, result, candidate_manifest)
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "source_snapshot_sha256": source_manifest["sha256"],
        "slice_summary": slice_summary,
        "evaluated_candidates": result["evaluated_candidates"],
        "best": result["best"],
        "event_verification": result["event_verification"],
        "promotion_gate": result["promotion_gate"],
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
