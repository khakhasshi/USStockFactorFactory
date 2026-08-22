#!/usr/bin/env python3
"""Create the four service-bound, continuous ideal research tasks.

The command is idempotent.  It never overwrites a task that already contains
research artifacts; a conflicting non-empty task is reported as an error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select  # noqa: E402

from app.config import (  # noqa: E402
    DEFAULT_ENGINE_CONFIG_V2,
    EVALUATION_PROTOCOL_VERSION,
    FROZEN_RATING_PROTOCOL_VERSION,
    FROZEN_RATING_WINDOW_END,
    FROZEN_RATING_WINDOW_START,
    default_panel_glob,
    evaluation_config,
)
from app.db import SessionLocal  # noqa: E402
from app.factors.diversity import mechanisms_for_market  # noqa: E402
from app.factors.return_source_governance import (  # noqa: E402
    RETURN_SOURCE_GOVERNANCE_PROTOCOL,
)
from app.models import Experiment, Factor, Node, OuterStep, Trial  # noqa: E402
from app.research_architecture import (  # noqa: E402
    RESEARCH_ARCHITECTURE_SCHEMA,
    resolve_research_architecture,
)


SERVICE_PROFILES = {
    "two_layer": {
        "port": 10011,
        "instance": "factorfactory-two-layer-ideal",
        "architecture_template": "algorithm_pool_researcher",
        "label": "双层理想架构",
    },
    "three_layer": {
        "port": 10012,
        "instance": "factorfactory-three-layer-ideal",
        "architecture_template": "full_three_layer",
        "label": "三层理想架构",
    },
}


def _tasks(market: str) -> list[dict]:
    mode = "long_only" if market == "ashare" else "long_short"
    costs = (20, 30, 20) if market == "ashare" else (15, 25, 15)
    return [
        {
            "name": "T1_liquid500_5d",
            "universe_n": 500,
            "horizon": 5,
            "cost_bps": costs[0],
            "mode": mode,
            "direction": 1,
            "direction_policy": "both_train_select",
        },
        {
            "name": "T2_mid1500_10d",
            "universe_n": 1500,
            "horizon": 10,
            "cost_bps": costs[1],
            "mode": mode,
            "direction": 1,
            "direction_policy": "both_train_select",
        },
        {
            "name": "T3_liquid500_20d",
            "universe_n": 500,
            "horizon": 20,
            "cost_bps": costs[2],
            "mode": mode,
            "direction": 1,
            "direction_policy": "both_train_select",
        },
    ]


def _engine_config(market: str) -> dict:
    config = deepcopy(DEFAULT_ENGINE_CONFIG_V2)
    config.update(
        {
            "tasks": _tasks(market),
            "inner_budget_per_outer_step": 6,
            "n_seeds_per_candidate": 5,
            "paired_cohorts_per_comparison": 5,
            "incumbent_remeasure_every": 1,
            "incumbent_remeasure_budget": 6,
            "baseline_warmup_budget": 6,
            "batch_candidates_per_call": 6,
            "draft_ratio": 0.5,
            "max_tree_depth": 3,
            "governor_warmup_candidates": 30,
            # These are retained as finite cohort settings for diagnostics;
            # continuous_operation makes them non-stopping controls.
            "max_outer_steps": 6,
            "max_runtime_hours": 4.0,
            "max_llm_calls": 60,
            "memory_mode": "adaptive",
        }
    )
    return config


def task_manifest() -> list[dict]:
    rows: list[dict] = []
    for architecture, profile in SERVICE_PROFILES.items():
        resolved = resolve_research_architecture(
            {"architecture_template": profile["architecture_template"]}
        )
        for market, market_label in (("us", "美股"), ("ashare", "A股")):
            mode = "long_only" if market == "ashare" else "long_short"
            research_config = {
                **resolved,
                "architecture_schema": RESEARCH_ARCHITECTURE_SCHEMA,
                "architecture_profile": f"ideal_{architecture}_continuous_v1",
                "architecture_arm": f"ideal_{architecture}",
                "service_instance": profile["instance"],
                "service_port": profile["port"],
                "service_autostart": True,
                "continuous_operation": True,
                "budget_policy": {
                    "mode": "unlimited",
                    "candidate_evaluation_budget": 0,
                    "target_factor_count": 0,
                    "restart_policy": "automatic_exponential_backoff",
                    "intended_schedule": "24x7",
                },
                "market": market,
                "portfolio_mode": mode,
                "direction": 1,
                "direction_policy": "both_train_select",
                "panel_glob": default_panel_glob(market),
                "engine_mode": "v2",
                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                "evaluation_config": evaluation_config(market),
                "frozen_rating": {
                    "protocol": FROZEN_RATING_PROTOCOL_VERSION,
                    "window_start": FROZEN_RATING_WINDOW_START,
                    "window_end": FROZEN_RATING_WINDOW_END,
                    "resolved_calendar_years": [2020, 2026],
                    "direction_frozen_before_rating": True,
                    "visible_to_layer2_researcher": False,
                    "visible_to_layer3_governor": False,
                },
                "pit_policy": {
                    "required_for_research": False,
                    "factor_admission_gate": False,
                    "disclosure_only": True,
                },
                "target_factor_count": 0,
                "candidate_evaluation_budget": 0,
                "target_mechanisms": list(mechanisms_for_market(market)),
                "return_source_governance": {
                    "protocol": RETURN_SOURCE_GOVERNANCE_PROTOCOL,
                    "correlation_threshold": 0.85,
                    "required_sources": 5,
                    "meta_score_weight": 0.15,
                    "cross_experiment_admission": False,
                },
                "engine_config": _engine_config(market),
            }
            rows.append(
                {
                    "name": f"{profile['label']}-{market_label}-7x24-v1",
                    "description": (
                        f"{profile['label']}；{market_label}；无限候选与因子预算；"
                        "服务自动恢复；冻结评级 2020-01-01 至 2026 最新可用日；"
                        "评级结果不回流研究 LLM；PIT 仅披露且不作入库门槛。"
                    ),
                    "status": "open",
                    "research_config": research_config,
                }
            )
    return rows


async def _artifact_count(session, experiment_id: int) -> int:
    total = 0
    for model in (Node, Trial, Factor, OuterStep):
        total += int(
            await session.scalar(
                select(func.count(model.id)).where(
                    model.experiment_id == experiment_id
                )
            )
            or 0
        )
    return total


async def apply_manifest(rows: list[dict]) -> list[dict]:
    output: list[dict] = []
    async with SessionLocal() as session:
        for spec in rows:
            existing = await session.scalar(
                select(Experiment).where(Experiment.name == spec["name"])
            )
            if existing is None:
                existing = Experiment(**spec)
                session.add(existing)
                await session.flush()
                output.append(
                    {"id": existing.id, "name": existing.name, "action": "created"}
                )
                continue
            if existing.research_config == spec["research_config"]:
                output.append(
                    {"id": existing.id, "name": existing.name, "action": "unchanged"}
                )
                continue
            artifacts = await _artifact_count(session, existing.id)
            if artifacts:
                raise RuntimeError(
                    f"refusing to overwrite non-empty task {existing.id} "
                    f"({existing.name}); artifacts={artifacts}"
                )
            existing.description = spec["description"]
            existing.status = "open"
            existing.research_config = spec["research_config"]
            output.append(
                {"id": existing.id, "name": existing.name, "action": "updated_empty"}
            )
        await session.commit()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the manifest to the database; default is a JSON preview",
    )
    args = parser.parse_args()
    rows = task_manifest()
    payload = asyncio.run(apply_manifest(rows)) if args.apply else rows
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
