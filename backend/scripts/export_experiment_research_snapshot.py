#!/usr/bin/env python3
"""Export a training-safe, score-ordered research-record snapshot."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import EVALUATION_PROTOCOL_VERSION  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import Experiment, Factor, Node  # noqa: E402
from app.research_records import (  # noqa: E402
    RESEARCH_RECORD_SCHEMA_VERSION,
    research_record_payload,
    task_research_summary,
)
from app.runtime_identity import runtime_identity  # noqa: E402


async def build_snapshot(experiment_id: int) -> dict:
    async with SessionLocal() as session:
        experiment = await session.get(Experiment, experiment_id)
        if experiment is None:
            raise ValueError(f"experiment {experiment_id} not found")
        nodes = list((await session.scalars(select(Node).where(
            Node.experiment_id == experiment_id,
            Node.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
        ))).all())
        factors = list((await session.scalars(select(Factor).where(
            Factor.experiment_id == experiment_id,
            Factor.node_id.is_not(None),
        ))).all())
    factor_by_node = {row.node_id: row.id for row in factors}
    grouped: dict[str, list[Node]] = {}
    for node in nodes:
        grouped.setdefault(node.task_name or "unknown", []).append(node)
    records = []
    for task_nodes in grouped.values():
        task_nodes.sort(
            key=lambda row: (row.status == "ok", float(row.public_score or 0), row.id),
            reverse=True,
        )
        records.extend(
            research_record_payload(
                node,
                task_rank=rank,
                formal_factor_id=factor_by_node.get(node.id),
            )
            for rank, node in enumerate(task_nodes, start=1)
        )
    records.sort(
        key=lambda row: (row["status"] == "ok", row["learning_score"], row["id"]),
        reverse=True,
    )
    return {
        "schema_version": RESEARCH_RECORD_SCHEMA_VERSION,
        "experiment": {
            "id": experiment.id,
            "name": experiment.name,
            "status": experiment.status,
            "research_config": experiment.research_config or {},
        },
        "runtime_identity_at_export": runtime_identity(),
        "interpretation_boundary": (
            "training research records only; not formal factor, holdout, vault, "
            "paper, live, or production approval"
        ),
        "tasks": task_research_summary(records),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = asyncio.run(build_snapshot(args.experiment_id))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
