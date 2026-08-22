#!/usr/bin/env python3
"""Rebind the four ideal continuous tasks to the unified 10010 service.

The research artifacts already live in the shared ``factor_factory`` database.
This migration therefore changes only the durable desired-state binding and
records the original runtime lineage.  It is intentionally idempotent and does
not copy, delete, or rewrite nodes, trials, factors, outer steps, or LLM audits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.models import (  # noqa: E402
    EngineEvent,
    Experiment,
    Factor,
    LLMCallAudit,
    Node,
    OuterStep,
    Setting,
    Trial,
)


SOURCE_SERVICES = {
    "factorfactory-two-layer-ideal": 10011,
    "factorfactory-three-layer-ideal": 10012,
}
TARGET_SERVICE = "factorfactory-10010"
TARGET_PORT = 10010
MIGRATION_PROTOCOL = "ideal_services_to_10010_v1"


def migrated_config(config: dict, *, migrated_at: str) -> tuple[dict, bool]:
    """Return a lineage-preserving 10010 config and whether it changed."""
    original = deepcopy(config or {})
    source_instance = str(original.get("service_instance") or "")
    if source_instance == TARGET_SERVICE:
        return original, False
    if source_instance not in SOURCE_SERVICES:
        raise ValueError(f"task is not bound to a retired ideal service: {source_instance!r}")

    source_port = int(original.get("service_port") or SOURCE_SERVICES[source_instance])
    lineage = list(original.get("runtime_lineage") or [])
    lineage.append(
        {
            "protocol": MIGRATION_PROTOCOL,
            "migrated_at": migrated_at,
            "source_service_instance": source_instance,
            "source_service_port": source_port,
            "target_service_instance": TARGET_SERVICE,
            "target_service_port": TARGET_PORT,
            "artifact_transport": "none_shared_database",
        }
    )
    original.update(
        {
            "service_instance": TARGET_SERVICE,
            "service_port": TARGET_PORT,
            "service_autostart": True,
            "runtime_lineage": lineage,
        }
    )
    return original, True


async def artifact_counts(session, experiment_id: int) -> dict[str, int]:
    models = {
        "nodes": Node,
        "trials": Trial,
        "factors": Factor,
        "outer_steps": OuterStep,
        "llm_call_audits": LLMCallAudit,
        "engine_events": EngineEvent,
    }
    return {
        name: int(
            await session.scalar(
                select(func.count(model.id)).where(model.experiment_id == experiment_id)
            )
            or 0
        )
        for name, model in models.items()
    }


async def run(*, apply: bool) -> list[dict]:
    migrated_at = datetime.now(UTC).isoformat()
    rows: list[dict] = []
    async with SessionLocal() as session:
        experiments = list(
            (
                await session.scalars(
                    select(Experiment)
                    .where(Experiment.status == "open")
                    .order_by(Experiment.id)
                )
            ).all()
        )
        for experiment in experiments:
            config = dict(experiment.research_config or {})
            service = str(config.get("service_instance") or "")
            if service not in {*SOURCE_SERVICES, TARGET_SERVICE}:
                continue
            # Only the original ideal two/three-layer continuous campaign is in scope.
            if not str(config.get("architecture_profile") or "").startswith("ideal_"):
                continue
            counts_before = await artifact_counts(session, experiment.id)
            next_config, changed = migrated_config(config, migrated_at=migrated_at)
            if apply and changed:
                experiment.research_config = next_config
            rows.append(
                {
                    "experiment_id": experiment.id,
                    "name": experiment.name,
                    "action": "migrated" if changed and apply else ("would_migrate" if changed else "unchanged"),
                    "source_service": service,
                    "target_service": TARGET_SERVICE,
                    "artifacts": counts_before,
                }
            )

        if apply:
            # Preserve the old active pointers as historical metadata; 10010's UI
            # uses the unscoped key while its supervisor starts every bound task.
            active = await session.get(Setting, "active_experiment")
            preferred_id = rows[0]["experiment_id"] if rows else 1
            if active is None:
                session.add(Setting(key="active_experiment", value={"id": preferred_id}))
            elif rows and int((active.value or {}).get("id", 0)) not in {
                row["experiment_id"] for row in rows
            }:
                active.value = {"id": preferred_id}
            await session.commit()

            # A post-commit recount makes accidental artifact mutation visible.
            for row in rows:
                after = await artifact_counts(session, row["experiment_id"])
                row["artifacts_after"] = after
                row["artifact_counts_unchanged"] = after == row["artifacts"]
                if not row["artifact_counts_unchanged"]:
                    raise RuntimeError(f"artifact count changed during migration: {row}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="commit the binding migration")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(apply=args.apply)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
