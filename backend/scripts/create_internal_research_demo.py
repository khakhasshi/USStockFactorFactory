#!/usr/bin/env python3
"""Create the A-share and US programmatic-first internal research demo tasks."""

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
from app.db import SessionLocal, init_db  # noqa: E402
from app.factors.diversity import mechanisms_for_market  # noqa: E402
from app.models import Experiment, Factor, Node, OuterStep, Trial  # noqa: E402
from app.research_architecture import resolve_research_architecture  # noqa: E402


SERVICE_INSTANCE = "factorfactory-internal-research-demo"


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


def manifest() -> list[dict]:
    architecture = resolve_research_architecture(
        {"architecture_template": "algorithm_pool_only"}
    )
    rows = []
    for market, label in (("us", "美股"), ("ashare", "A股")):
        mode = "long_only" if market == "ashare" else "long_short"
        engine_config = deepcopy(DEFAULT_ENGINE_CONFIG_V2)
        engine_config.update(
            {
                "tasks": _tasks(market),
                "inner_budget_per_outer_step": 8,
                "n_seeds_per_candidate": 3,
                "paired_cohorts_per_comparison": 3,
                "batch_candidates_per_call": 8,
                "max_tree_depth": 3,
                "max_outer_steps": 10,
                "max_runtime_hours": 24.0,
                "max_llm_calls": 1,
            }
        )
        config = {
            **architecture,
            "architecture_profile": "internal_research_hybrid_v1",
            "architecture_arm": "programmatic_residual_overfit_governed",
            "service_instance": SERVICE_INSTANCE,
            "service_port": 20010,
            "service_autostart": True,
            "continuous_operation": True,
            "market": market,
            "portfolio_mode": mode,
            "direction": 1,
            "direction_policy": "both_train_select",
            "panel_glob": default_panel_glob(market),
            "engine_mode": "v2",
            "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
            "evaluation_config": evaluation_config(market),
            "engine_config": engine_config,
            "target_mechanisms": list(mechanisms_for_market(market)),
            "candidate_evaluation_budget": 0,
            "target_factor_count": 0,
            "budget_policy": {
                "mode": "unlimited",
                "intended_schedule": "24x7",
                "restart_policy": "automatic_exponential_backoff",
            },
            "frozen_rating": {
                "protocol": FROZEN_RATING_PROTOCOL_VERSION,
                "window_start": FROZEN_RATING_WINDOW_START,
                "window_end": FROZEN_RATING_WINDOW_END,
                "resolved_calendar_years": [2020, 2026],
                "visible_to_research": False,
                "direction_frozen_before_rating": True,
            },
            "overfit_governance": {
                "protocol": "factorfactory.dynamic-overfit-governance/v1",
                "dynamic_actual_trials": True,
                "effective_trials": True,
                "dsr": True,
                "pbo_cscv": True,
                "harvey_liu_haircut": True,
                "winner_curse": True,
                "scope": "public_plus_meta_train_only",
            },
            "residual_search": {
                "protocol": "factorfactory.residual-oof-beam/v2",
                "exact_oof_endpoint": "/api/research-intelligence/residual-beam",
                "continuous_mode": "sample_level_cross_sectional_time_ordered_oof",
                "automated_layers": ["INNER_PUBLIC", "META_TRAIN"],
                "whole_date_folds": True,
                "beam_width": 12,
                "proxy_fallback_forbidden": True,
            },
            "mechanism_lens": {
                "protocol": "factorfactory.mechanism-lens/v1",
                "base_mechanism_count": 33,
                "bulk_parameter_grid_forbidden": True,
                "document_input_endpoint": "/api/research-intelligence/document-to-dsl",
            },
            "double_blind_review": {
                "protocol": "factorfactory.double-blind-review/v1",
                "method_reviewer": "optional_llm_isolated",
                "code_reviewer": "deterministic_plus_optional_llm_isolated",
                "formal_promotion_requires_both": True,
                "llm_failure_policy": "research_continues_review_pending",
            },
            "llm_policy": {
                "mandatory_layers": 0,
                "optional_shortlist_reviewers": 2,
                "reason": "programmatic research survives provider outage; LLM is used only where marginal value is highest",
            },
            "pit_policy": {
                "required_for_research": False,
                "factor_admission_gate": False,
                "disclosure_only": True,
            },
        }
        rows.append(
            {
                "name": f"内部研究增强Demo-{label}-7x24-v1",
                "description": (
                    f"{label}；程序化发现为主，0层强制LLM、2个可选双盲审查器；"
                    "实际试验账本、动态多重检验、Residual OOF Beam、33机制地图、"
                    "文档到DSL；冻结评级2020至最新且不回流研究。"
                ),
                "status": "open",
                "research_config": config,
            }
        )
    return rows


async def apply(rows: list[dict]) -> list[dict]:
    await init_db()
    output = []
    async with SessionLocal() as session:
        for spec in rows:
            experiment = await session.scalar(
                select(Experiment).where(Experiment.name == spec["name"])
            )
            if experiment is None:
                experiment = Experiment(**spec)
                session.add(experiment)
                await session.flush()
                output.append({"id": experiment.id, "name": experiment.name, "action": "created"})
                continue
            artifact_count = 0
            for model in (Node, Trial, Factor, OuterStep):
                artifact_count += int(
                    await session.scalar(
                        select(func.count(model.id)).where(model.experiment_id == experiment.id)
                    )
                    or 0
                )
            if experiment.research_config != spec["research_config"] and artifact_count:
                raise RuntimeError(f"refusing to overwrite non-empty demo task {experiment.id}")
            experiment.description = spec["description"]
            experiment.status = "open"
            experiment.research_config = spec["research_config"]
            output.append({"id": experiment.id, "name": experiment.name, "action": "unchanged" if artifact_count else "updated_empty"})
        await session.commit()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    rows = manifest()
    payload = asyncio.run(apply(rows)) if args.apply else rows
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
