#!/usr/bin/env python3
"""Build a read-only catalog of distinct training-safe factor return sources."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import EVALUATION_PROTOCOL_VERSION  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.factors.return_source_governance import (  # noqa: E402
    RETURN_SOURCE_GOVERNANCE_PROTOCOL,
    cluster_training_return_sources,
    governance_reason_counts,
)
from app.models import Experiment, Factor  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", type=int, action="append")
    parser.add_argument("--correlation-threshold", type=float, default=0.85)
    parser.add_argument("--all-protocols", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    return parser


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


async def _load_rows(args: argparse.Namespace) -> tuple[list[dict], dict[int, str]]:
    filters = []
    if args.experiment_id:
        filters.append(Factor.experiment_id.in_(args.experiment_id))
    if not args.all_protocols:
        filters.append(Factor.evaluation_protocol == EVALUATION_PROTOCOL_VERSION)
    async with SessionLocal() as session:
        factors = list((await session.scalars(select(Factor).where(*filters))).all())
        experiments = list((await session.scalars(select(Experiment))).all())
    experiment_names = {row.id: row.name for row in experiments}
    rows = []
    for factor in factors:
        research = dict(factor.research_meta or {})
        public = dict(factor.public_metrics or {})
        rows.append({
            "factor_id": factor.id,
            "experiment_id": factor.experiment_id,
            "experiment_name": experiment_names.get(factor.experiment_id, "unknown"),
            "task_name": factor.task_name,
            "expression": factor.expression,
            "mechanism_family": research.get("mechanism_family") or "other",
            "market": research.get("market") or "unknown",
            "portfolio_mode": research.get("portfolio_mode") or "unknown",
            "task_signature": research.get("task_signature"),
            "horizon": (research.get("task_snapshot") or {}).get("horizon")
            or public.get("horizon"),
            "quality_score": (public.get("discovery") or {}).get(
                "learning_score", public.get("score", public.get("icir", 0.0))
            ),
            "evaluation_protocol": factor.evaluation_protocol,
            "lifecycle_stage": factor.lifecycle_stage,
            "public_metrics": public,
            "research_meta": research,
        })
    return rows, experiment_names


def _markdown(summary: dict, assignments: list[dict], rows: list[dict]) -> str:
    by_id = {str(row["factor_id"]): row for row in rows}
    representatives = [row for row in assignments if row["is_representative"]]
    duplicate_counts = Counter(
        row["representative_identity"]
        for row in assignments
        if not row["is_representative"]
    )
    representatives.sort(
        key=lambda row: (-duplicate_counts[row["identity"]], row["identity"])
    )
    lines = [
        "# Training Return Source Catalog",
        "",
        f"- Protocol: `{summary['protocol']}`",
        f"- Correlation threshold: `{summary['correlation_threshold']}`",
        f"- Factors inspected: `{summary['items']}`",
        f"- Signature coverage: `{summary['signature_coverage']:.1%}`",
        f"- Comparable scopes: `{summary['comparable_scopes']}`",
        f"- Return-source clusters: `{summary['return_source_clusters']}`",
        f"- Redundancy rate: `{summary['return_source_redundancy_rate']:.1%}`",
        f"- Effective return sources: `{summary['effective_return_sources']:.2f}`",
        f"- Largest cluster: `{summary['largest_return_source_cluster']}`",
        "",
        "## Largest representative clusters",
        "",
        "| Representative | Cluster | Size | Experiment | Task | Mechanism | Expression |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for assignment in representatives[:20]:
        factor = by_id.get(assignment["identity"], {})
        size = duplicate_counts[assignment["identity"]] + 1
        expression = str(factor.get("expression") or "").replace("|", "\\|")[:180]
        lines.append(
            f"| {assignment['identity']} | {assignment['cluster_id']} | {size} | "
            f"{factor.get('experiment_name', 'unknown')} | {factor.get('task_name', '')} | "
            f"{assignment['mechanism_family']} | `{expression}` |"
        )
    lines.extend([
        "",
        "## Interpretation boundary",
        "",
        "This catalog uses only stored INNER_PUBLIC + META_TRAIN sketches. It is an admission and training-governance artifact, not HOLDOUT/VAULT evidence and not live-trading approval.",
        "",
    ])
    return "\n".join(lines)


async def _run(args: argparse.Namespace) -> Path:
    rows, _ = await _load_rows(args)
    snapshot = cluster_training_return_sources(
        rows,
        correlation_threshold=args.correlation_threshold,
        include_assignments=True,
    )
    assignments = list(snapshot.pop("assignments", []))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_dir or (
        PROJECT_ROOT / "var" / "reports" / f"training-return-source-catalog-{timestamp}"
    )
    output = output.resolve()
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise SystemExit(f"拒绝覆盖非空目录: {output}")
    output.mkdir(parents=True, exist_ok=True)

    by_id = {str(row["factor_id"]): row for row in rows}
    csv_rows = []
    for assignment in assignments:
        factor = by_id.get(assignment["identity"], {})
        csv_rows.append({
            **assignment,
            "factor_id": factor.get("factor_id"),
            "experiment_id": factor.get("experiment_id"),
            "experiment_name": factor.get("experiment_name"),
            "task_name": factor.get("task_name"),
            "expression": factor.get("expression"),
        })
    with (output / "assignments.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(csv_rows[0]) if csv_rows else ["factor_id"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    report = {
        **snapshot,
        "generated_at": datetime.now(timezone.utc),
        "evaluation_protocol_filter": (
            "all" if args.all_protocols else EVALUATION_PROTOCOL_VERSION
        ),
        "experiment_ids": args.experiment_id or "all",
        "duplicate_mechanism_counts": governance_reason_counts({
            "assignments": assignments
        }),
        "recommended_next_experiment_config": {
            "return_source_governance": {
                "protocol": RETURN_SOURCE_GOVERNANCE_PROTOCOL,
                "correlation_threshold": args.correlation_threshold,
                "meta_score_weight": 0.30,
                "required_sources": 4,
                "cross_experiment_admission": True,
            }
        },
        "interpretation_boundary": (
            "INNER_PUBLIC+META_TRAIN sketches only; not holdout/vault/live approval"
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output / "README.md").write_text(
        _markdown(report, assignments, rows),
        encoding="utf-8",
    )
    return output


def main() -> None:
    args = _parser().parse_args()
    output = asyncio.run(_run(args))
    print(output)


if __name__ == "__main__":
    main()
