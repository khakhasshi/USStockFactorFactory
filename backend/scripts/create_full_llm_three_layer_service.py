#!/usr/bin/env python3
"""Create the two port-10013 full-LLM continuous research tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import (  # noqa: E402
    DEFAULT_ENGINE_CONFIG_V2,
    EVALUATION_PROTOCOL_VERSION,
    FROZEN_RATING_PROTOCOL_VERSION,
    FROZEN_RATING_WINDOW_END,
    FROZEN_RATING_WINDOW_START,
    default_panel_glob,
    evaluation_config,
)
from app.factors.diversity import mechanisms_for_market  # noqa: E402
from app.factors.return_source_governance import (  # noqa: E402
    RETURN_SOURCE_GOVERNANCE_PROTOCOL,
)
from app.research_architecture import resolve_research_architecture  # noqa: E402
from scripts.create_continuous_ideal_services import apply_manifest  # noqa: E402


SERVICE_PORT = 10013
SERVICE_INSTANCE = "factorfactory-full-llm-three-layer"


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
            "inner_budget_per_outer_step": 8,
            "n_seeds_per_candidate": 5,
            "paired_cohorts_per_comparison": 5,
            "incumbent_remeasure_every": 1,
            "incumbent_remeasure_budget": 8,
            "baseline_warmup_budget": 8,
            "batch_candidates_per_call": 4,
            "draft_ratio": 0.58,
            "max_tree_depth": 3,
            "governor_warmup_candidates": 0,
            "scientific_governor_interval_outer_steps": 3,
            # Optional inspiration only. L1 has direct-expression authority.
            "algorithm_inspiration_share": 0.30,
            "max_outer_steps": 6,
            "max_runtime_hours": 4.0,
            "max_llm_calls": 80,
            "memory_mode": "adaptive",
        }
    )
    return config


def task_manifest() -> list[dict]:
    architecture = resolve_research_architecture(
        {"architecture_template": "full_llm_three_layer"}
    )
    rows: list[dict] = []
    for market, label in (("us", "美股"), ("ashare", "A股")):
        mode = "long_only" if market == "ashare" else "long_short"
        config = {
            **architecture,
            "architecture_profile": "full_llm_three_layer_continuous_v1",
            "architecture_arm": "full_llm_three_layer",
            "service_instance": SERVICE_INSTANCE,
            "service_port": SERVICE_PORT,
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
            "candidate_evaluation_budget": 0,
            "target_factor_count": 0,
            "target_mechanisms": list(mechanisms_for_market(market)),
            "llm_roles": {
                "layer1": {
                    "role": "mechanism_scientist",
                    "provider_setting": "inner_provider",
                    "direct_expression_authority": True,
                    "actions": ["draft_direct", "mutate", "repair"],
                },
                "layer2": {
                    "role": "research_director",
                    "provider_setting": "outer_provider",
                    "controls": ["prompt_policy", "search_policy", "memory_policy"],
                },
                "layer3": {
                    "role": "scientific_governor",
                    "provider_setting": "scientific_governor_provider_or_outer_fallback",
                    "controls": [
                        "mechanism_portfolio",
                        "exploration_share",
                        "falsification_rule",
                        "directive_tenure",
                    ],
                },
            },
            "memory_architecture": {
                "layer1": "candidate_lineage_and_repair_memory",
                "layer2": "policy_hypothesis_and_ab_result_journal",
                "layer3": "institutional_mechanism_portfolio_memory",
                "continuous": True,
                "rating_feedback_allowed": False,
            },
            "frozen_rating": {
                "protocol": FROZEN_RATING_PROTOCOL_VERSION,
                "window_start": FROZEN_RATING_WINDOW_START,
                "window_end": FROZEN_RATING_WINDOW_END,
                "resolved_calendar_years": [2020, 2026],
                "direction_frozen_before_rating": True,
                "visible_to_layer1_mechanism_scientist": False,
                "visible_to_layer2_research_director": False,
                "visible_to_layer3_scientific_governor": False,
            },
            "pit_policy": {
                "required_for_research": False,
                "factor_admission_gate": False,
                "disclosure_only": True,
            },
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
                "name": f"全LLM三层架构-{label}-7x24-v1",
                "description": (
                    f"10013 全 LLM 三层；{label}；L1 可直接生成表达式；"
                    "无限预算；冻结评级自 2020 起至最新可用日且不回流三层 LLM。"
                ),
                "status": "open",
                "research_config": config,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    rows = task_manifest()
    result = asyncio.run(apply_manifest(rows)) if args.apply else rows
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
