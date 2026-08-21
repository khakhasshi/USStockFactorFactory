#!/usr/bin/env python3
"""Find deterministic 2-5 factor weights for frozen A-share factors.

The default candidate universe is built from current-protocol research-pass
factors in the requested experiments.  ``--include-leaderboard`` adds the
first rows of a historical full-window leaderboard, but marks the entire run
as diagnostic because that report used dates beyond META_TRAIN to pre-select
its candidates.  Neither mode reads META_HOLDOUT or FACTOR_VAULT returns while
fitting weights.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.backtest.batch import BatchBacktestSpec, run_cost_scenarios  # noqa: E402
from app.config import (  # noqa: E402
    ASHARE_PANEL_GLOB,
    EVALUATION_PROTOCOL_VERSION,
    get_dsl_fields,
)
from app.db import SessionLocal, engine as db_engine  # noqa: E402
from app.dsl.engine import validate as validate_expression  # noqa: E402
from app.factors.combination_candidates import (  # noqa: E402
    select_diverse_combination_candidates,
)
from app.factors.diversity import infer_mechanism  # noqa: E402
from app.factors.weight_optimizer import (  # noqa: E402
    FactorComponent,
    WeightSearchConfig,
    apply_composite_weights,
    build_slices,
    combination_promotion_gate,
    composite_expression,
    materialize_component_frame,
    search_optimal_weights,
)
from app.models import Factor  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEADERBOARD = (
    PROJECT_ROOT
    / "var"
    / "reports"
    / "ashare-long-only-vector-screen-2020-latest-20260807-133714"
    / "leaderboard_full.json"
)


def _json_default(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("至少需要一个整数 ID")
    if any(item < 1 for item in values):
        raise ValueError("ID 必须为正整数")
    return list(dict.fromkeys(values))


def _factor_candidate(row: Factor) -> dict[str, Any]:
    meta = dict(row.research_meta or {})
    public = dict(row.public_metrics or {})
    direction = int(meta.get("direction") or 0)
    if meta.get("market") != "ashare":
        raise ValueError(f"F{row.id} 不是 A 股因子")
    if row.evaluation_protocol != EVALUATION_PROTOCOL_VERSION:
        raise ValueError(
            f"F{row.id} 协议为 {row.evaluation_protocol}，"
            f"不是当前 {EVALUATION_PROTOCOL_VERSION}"
        )
    if row.lifecycle_stage != "research_pass":
        raise ValueError(f"F{row.id} 生命周期不是 research_pass")
    if row.provenance_status != "valid_task_config":
        raise ValueError(f"F{row.id} 研究配置来源无效")
    if not (meta.get("direction_selection") or {}).get("frozen_for_downstream"):
        raise ValueError(f"F{row.id} 的方向尚未冻结")
    if direction not in {-1, 1}:
        raise ValueError(f"F{row.id} 缺少有效冻结方向")
    expression_error = validate_expression(row.expression, get_dsl_fields("ashare"))
    if expression_error:
        raise ValueError(f"F{row.id} 当前 DSL 无法执行: {expression_error}")
    mechanism = str(meta.get("mechanism_family") or "").strip()
    if not mechanism:
        mechanism = infer_mechanism(row.expression, row.hypothesis)
    return {
        "component_key": f"db:{row.id}",
        "factor_id": row.id,
        "name": f"E{row.experiment_id}:{row.name}",
        "expression": row.expression,
        "direction": direction,
        "mechanism_family": mechanism,
        "quality_score": float(public.get("score") or 0.0),
        "source_priority": 2,
        "source_rank": row.id,
        "source": "factor_database",
        "source_ref": f"experiment:{row.experiment_id}:factor:{row.id}",
        "experiment_id": row.experiment_id,
        "evaluation_protocol": row.evaluation_protocol,
        "lifecycle_stage": row.lifecycle_stage,
        "provenance_status": row.provenance_status,
        "training_return_path_signature": meta.get(
            "training_return_path_signature"
        ),
        "candidate_selection_scope": "INNER_PUBLIC_plus_META_TRAIN_only",
    }


def _leaderboard_candidates(path: Path, top_n: int) -> tuple[list[dict], dict]:
    if not path.is_file():
        raise ValueError(f"历史榜单不存在: {path}")
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("历史榜单格式必须是 JSON 数组")
    eligible = sorted(
        (
            row for row in rows
            if row.get("status") == "ok"
            and bool(row.get("practical_pass"))
            and int(row.get("overall_rank") or 10**9) <= top_n
        ),
        key=lambda row: int(row["overall_rank"]),
    )
    candidates = []
    invalid = []
    for row in eligible:
        rank = int(row["overall_rank"])
        expression = str(row.get("expression") or "").strip()
        error = validate_expression(expression, get_dsl_fields("ashare"))
        direction = int(row.get("direction") or 0)
        if error or direction not in {-1, 1}:
            invalid.append({
                "overall_rank": rank,
                "expression": expression,
                "error": error or "invalid_direction",
            })
            continue
        digest = hashlib.sha256(
            f"{direction}|{expression}".encode("utf-8")
        ).hexdigest()[:16]
        candidates.append({
            "component_key": f"leaderboard:{rank}:{digest}",
            "factor_id": None,
            "name": f"历史榜单#{rank}",
            "expression": expression,
            "direction": direction,
            "mechanism_family": infer_mechanism(expression),
            "quality_score": float(row.get("robust_score") or 0.0),
            "source_priority": 1,
            "source_rank": rank,
            "source": "historical_leaderboard",
            "source_ref": f"{path.resolve()}#overall_rank={rank}",
            "leaderboard_factor_ids": row.get("factor_ids") or [],
            "training_return_path_signature": None,
            "candidate_selection_scope": "2020_to_latest_full_window_diagnostic",
        })
    protocol_path = path.parent / "protocol.json"
    policy = {}
    if protocol_path.is_file():
        policy = json.loads(protocol_path.read_text(encoding="utf-8"))
    return candidates, {
        "path": str(path.resolve()),
        "requested_top_n": top_n,
        "eligible_rows": len(eligible),
        "valid_candidates": len(candidates),
        "invalid_candidates": invalid,
        "policy_label": policy.get("policy_label", "unknown"),
        "selection_warning": (
            "This source was ranked on a 2020-to-latest diagnostic window. "
            "Any optimizer run containing it is not independent holdout evidence."
        ),
    }


async def _load_factor_rows(
    *,
    factor_ids: list[int] | None = None,
    experiment_ids: list[int] | None = None,
) -> list[Factor]:
    async with SessionLocal() as session:
        statement = select(Factor)
        if factor_ids is not None:
            statement = statement.where(Factor.id.in_(factor_ids))
        if experiment_ids is not None:
            statement = statement.where(Factor.experiment_id.in_(experiment_ids))
        rows = (await session.execute(statement.order_by(Factor.id))).scalars().all()
    if factor_ids is not None:
        by_id = {row.id: row for row in rows}
        missing = [factor_id for factor_id in factor_ids if factor_id not in by_id]
        if missing:
            raise ValueError(f"数据库中不存在因子: {missing}")
        rows = [by_id[factor_id] for factor_id in factor_ids]
    return rows


async def _load_candidates_and_close(
    *,
    factor_ids: list[int] | None,
    experiment_ids: list[int] | None,
) -> list[dict]:
    """Keep asyncpg checkout and engine disposal on the same event loop."""
    try:
        rows = await _load_factor_rows(
            factor_ids=factor_ids,
            experiment_ids=experiment_ids,
        )
        candidates = []
        rejected = []
        for row in rows:
            try:
                candidates.append(_factor_candidate(row))
            except ValueError as exc:
                if factor_ids is not None:
                    raise
                rejected.append({"factor_id": row.id, "reason": str(exc)})
        if not candidates:
            detail = f"；过滤详情: {rejected[:10]}" if rejected else ""
            raise ValueError(f"没有可用于组合优化的当前协议因子{detail}")
        return candidates
    finally:
        await db_engine.dispose()


def _component_from_candidate(row: dict[str, Any]) -> FactorComponent:
    return FactorComponent(
        factor_id=row.get("factor_id"),
        name=str(row["name"]),
        expression=str(row["expression"]),
        direction=int(row["direction"]),
        mechanism_family=str(row["mechanism_family"]),
        source=str(row["source"]),
        source_ref=str(row["source_ref"]),
    )


def _best_variants(result: dict) -> dict:
    rows = result["results"]
    by_factor_count = {}
    for size in range(2, 6):
        winner = next((row for row in rows if row["active_factors"] == size), None)
        if winner is not None:
            by_factor_count[str(size)] = winner
    three_sources = next(
        (row for row in rows if row["active_mechanism_groups"] >= 3),
        None,
    )
    return {
        "best_by_active_factor_count": by_factor_count,
        "best_with_at_least_three_mechanisms": three_sources,
    }


def _write_outputs(
    output_dir: Path,
    components: list[FactorComponent],
    result: dict,
    candidate_manifest: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    best = result["best"]
    payload = {
        **result,
        "components": [asdict(row) for row in components],
        "best_composite_expression": composite_expression(components, best["weights"]),
    }
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
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    fieldnames = [
        "rank", "robust_score", "active_factors", "weights",
        "worst_sharpe_lcb", "worst_icir", "worst_ann_return_lcb",
        "worst_stress_sharpe", "max_drawdown", "generalization_gap",
        "worst_profitable_year_rate", "worst_era_consistency",
        "worst_active_era_consistency", "worst_time_block_sharpe",
        "worst_time_block_ann_return", "worst_positive_time_block_rate",
        "worst_tail_sharpe", "worst_tail_monotonicity",
        "worst_ic_tail_conversion_rate", "return_source_independence",
        "weighted_return_path_similarity",
        "effective_factor_count", "active_mechanism_groups",
        "effective_mechanism_count", "max_mechanism_weight",
    ]
    with (output_dir / "leaderboard.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["results"]:
            writer.writerow({
                key: json.dumps(row[key]) if key == "weights" else row[key]
                for key in fieldnames
            })


def _event_verify(frame, best: dict, config: WeightSearchConfig) -> dict:
    composite = apply_composite_weights(frame, best["weights"])
    output = {}
    for layer, start, end in (
        ("INNER_PUBLIC", "2010-01-01", "2019-12-31"),
        ("META_TRAIN", "2020-01-01", "2022-12-31"),
    ):
        spec = BatchBacktestSpec(
            market="ashare",
            mode="long_only",
            universe_n=config.universe_n,
            horizon=config.horizon,
            top_fraction=config.top_fraction,
            rebalance_every=config.horizon,
            holdout_start=start,
            holdout_end=end,
            slippage_bps=(0.0, 5.0, 15.0),
            fee_profile="ashare_wan2_no_min_v1",
        )
        output[layer] = run_cost_scenarios(
            composite,
            direction=1,
            spec=spec,
            capture_detail=False,
        )
    return {
        "engine": "StepEventBacktester",
        "purpose": "finalist_execution_replay_not_search_objective",
        "fee_profile": "ashare_wan2_no_min_v1",
        "scenarios": output,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--factor-ids",
        help="显式冻结因子 ID；允许跨任务，且不会自动加入其他候选",
    )
    source.add_argument(
        "--source-experiments",
        default="17,18,19",
        help="自动候选池的任务 ID，默认 17,18,19",
    )
    parser.add_argument("--include-leaderboard", action="store_true")
    parser.add_argument("--leaderboard-report", default=str(DEFAULT_LEADERBOARD))
    parser.add_argument("--leaderboard-top", type=int, default=3)
    parser.add_argument("--max-per-mechanism", type=int, default=3)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--candidate-structural-threshold", type=float, default=0.64)
    parser.add_argument("--candidate-return-correlation-cap", type=float, default=0.80)
    parser.add_argument("--output-dir")
    parser.add_argument("--coarse-step", type=float, default=0.10)
    parser.add_argument("--refine-step", type=float, default=0.02)
    parser.add_argument("--min-factors", type=int, default=2)
    parser.add_argument("--max-factors", type=int, default=5)
    parser.add_argument("--min-weight", type=float, default=0.05)
    parser.add_argument("--max-weight", type=float, default=0.65)
    parser.add_argument("--max-mechanism-weight", type=float, default=0.60)
    parser.add_argument("--max-active-pair-similarity", type=float, default=0.85)
    parser.add_argument("--tail-fractions", default="0.10,0.20,0.30")
    parser.add_argument("--time-block-folds", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=20.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--skip-event-verify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    factor_ids = _parse_int_list(args.factor_ids) if args.factor_ids else None
    experiment_ids = (
        None if factor_ids is not None else _parse_int_list(args.source_experiments)
    )
    database_candidates = asyncio.run(_load_candidates_and_close(
        factor_ids=factor_ids,
        experiment_ids=experiment_ids,
    ))
    input_candidates = list(database_candidates)
    leaderboard_manifest = None
    if args.include_leaderboard:
        leaderboard_candidates, leaderboard_manifest = _leaderboard_candidates(
            Path(args.leaderboard_report),
            args.leaderboard_top,
        )
        input_candidates.extend(leaderboard_candidates)

    if factor_ids is not None:
        selection = {
            "selected": input_candidates,
            "excluded": [],
            "selection_config": {"mode": "explicit_factor_ids"},
            "input_candidates": len(input_candidates),
            "selected_candidates": len(input_candidates),
            "selected_mechanisms": sorted({
                row["mechanism_family"] for row in input_candidates
            }),
        }
    else:
        selection = select_diverse_combination_candidates(
            input_candidates,
            structural_threshold=args.candidate_structural_threshold,
            return_path_correlation_cap=args.candidate_return_correlation_cap,
            max_per_mechanism=args.max_per_mechanism,
            max_candidates=args.max_candidates,
        )
    components = [_component_from_candidate(row) for row in selection["selected"]]
    if len(components) < 2:
        raise ValueError("候选治理后不足两个因子")

    track = "augmented_historical_diagnostic" if args.include_leaderboard else (
        "explicit_current_protocol" if factor_ids is not None
        else "current_protocol_clean"
    )
    candidate_manifest = {
        "created_at": datetime.now().isoformat(),
        "track": track,
        "market": "ashare",
        "llm_used": False,
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
        "weight_fit_scope": "INNER_PUBLIC_plus_META_TRAIN_only",
        "holdout_or_vault_read": False,
        "independent_holdout_claim_allowed": not args.include_leaderboard,
        "source_experiments": experiment_ids,
        "explicit_factor_ids": factor_ids,
        "leaderboard": leaderboard_manifest,
        "input_candidates": input_candidates,
        "selection": selection,
    }

    tail_fractions = tuple(
        float(value.strip())
        for value in args.tail_fractions.split(",")
        if value.strip()
    )
    config = WeightSearchConfig(
        min_factors=args.min_factors,
        max_factors=min(args.max_factors, len(components)),
        coarse_step=args.coarse_step,
        refine_step=args.refine_step,
        min_active_weight=args.min_weight,
        max_weight=args.max_weight,
        max_mechanism_weight=args.max_mechanism_weight,
        max_active_pair_similarity=args.max_active_pair_similarity,
        tail_fractions=tail_fractions,
        time_block_folds=args.time_block_folds,
        cost_bps=args.cost_bps,
        stress_cost_bps=args.stress_cost_bps,
    )
    print(
        f"[1/4] materialize {len(components)} frozen A-share factors "
        f"({track})",
        flush=True,
    )
    frame, provenance = materialize_component_frame(
        components,
        universe_n=config.universe_n,
        horizon=config.horizon,
        panel_glob=ASHARE_PANEL_GLOB,
        include_execution_columns=not args.skip_event_verify,
    )
    print(f"[2/4] build non-overlapping {config.horizon}d slices", flush=True)
    slices, slice_summary = build_slices(
        frame,
        component_count=len(components),
        horizon=config.horizon,
        universe_n=config.universe_n,
    )
    print("[3/4] complete coarse grid + deterministic refinement", flush=True)
    result = search_optimal_weights(
        slices,
        len(components),
        config,
        factor_groups=[row.mechanism_family for row in components],
    )
    result["track"] = track
    result["independent_holdout_claim_allowed"] = not args.include_leaderboard
    result["provenance"] = provenance
    result["slice_summary"] = slice_summary
    result.update(_best_variants(result))
    result["best"]["components"] = [
        {
            "factor_id": component.factor_id,
            "name": component.name,
            "source": component.source,
            "source_ref": component.source_ref,
            "direction": component.direction,
            "mechanism_family": component.mechanism_family,
            "weight": result["best"]["weights"][index],
        }
        for index, component in enumerate(components)
        if result["best"]["weights"][index] > 1e-12
    ]
    if not args.skip_event_verify:
        print("[4/4] replay winner in the step-event engine", flush=True)
        result["event_verification"] = _event_verify(frame, result["best"], config)
    else:
        print("[4/4] event replay skipped by request", flush=True)
    result["promotion_gate"] = combination_promotion_gate(result)
    if args.include_leaderboard:
        result["promotion_gate"]["decision"] = "DIAGNOSTIC_ONLY"
        result["promotion_gate"]["production_eligible"] = False
        result["promotion_gate"]["failed_rules"] = sorted(set([
            *result["promotion_gate"].get("failed_rules", []),
            "historical_full_window_candidate_preselection",
        ]))

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "var" / "reports" / f"ashare-factor-weight-search-{track}-{stamp}"
    )
    _write_outputs(output_dir, components, result, candidate_manifest)
    print(json.dumps({
        "output_dir": str(output_dir.resolve()),
        "track": track,
        "input_candidates": len(input_candidates),
        "selected_candidates": len(components),
        "excluded_candidates": selection.get("excluded", []),
        "evaluated_weight_candidates": result["evaluated_candidates"],
        "best": result["best"],
        "promotion_gate": result["promotion_gate"],
    }, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
