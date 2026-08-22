#!/usr/bin/env python3
"""Preview or create the three stopped/open V667 ablation tasks."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import (  # noqa: E402
    DEFAULT_ENGINE_CONFIG_V2,
    EVALUATION_PROTOCOL_VERSION,
    default_panel_glob,
    evaluation_config,
)
from app.db import SessionLocal  # noqa: E402
from app.models import Experiment  # noqa: E402
from app.factors.return_source_governance import (  # noqa: E402
    RETURN_SOURCE_GOVERNANCE_PROTOCOL,
)


US_MECHANISMS = [
    "momentum",
    "reversal",
    "volatility",
    "liquidity",
    "volume_price_interaction",
    "gap_intraday",
    "price_relationship",
]


def campaign(prefix: str) -> list[dict]:
    base = {
        "market": "us",
        "portfolio_mode": "long_short",
        "direction": 1,
        "direction_policy": "both_train_select",
        "panel_glob": default_panel_glob("us"),
        "engine_mode": "v2",
        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
        "evaluation_config": evaluation_config("us"),
        "target_factor_count": 0,
        "candidate_evaluation_budget": 120,
        "target_mechanisms": US_MECHANISMS,
        "return_source_governance": {
            "protocol": RETURN_SOURCE_GOVERNANCE_PROTOCOL,
            "correlation_threshold": 0.85,
            "required_sources": 5,
            "meta_score_weight": 0.15,
        },
        "engine_config": deepcopy(DEFAULT_ENGINE_CONFIG_V2),
    }
    arms = [
        ("Random", "random", "cold"),
        ("LLM-Cold", "llm", "cold"),
        ("LLM-Memory", "llm", "adaptive"),
    ]
    return [
        {
            "name": f"{prefix}-{label}",
            "description": (
                "V667 predeclared three-arm ablation; same tasks, mechanisms, "
                "seed cohorts, costs and candidate budget"
            ),
            "status": "open",
            "research_config": {
                **deepcopy(base),
                "proposal_mode": proposal_mode,
                "memory_mode": memory_mode,
                "ablation_arm": label,
            },
        }
        for label, proposal_mode, memory_mode in arms
    ]


async def apply_campaign(rows: list[dict]) -> list[dict]:
    output = []
    async with SessionLocal() as session:
        for spec in rows:
            existing = await session.scalar(
                select(Experiment).where(Experiment.name == spec["name"])
            )
            if existing is not None:
                output.append({"id": existing.id, "name": existing.name, "created": False})
                continue
            row = Experiment(**spec)
            session.add(row)
            await session.flush()
            output.append({"id": row.id, "name": row.name, "created": True})
        await session.commit()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="美股V667")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="create the open tasks; without this flag only print the manifest",
    )
    args = parser.parse_args()
    rows = campaign(args.prefix)
    if args.apply:
        print(json.dumps(asyncio.run(apply_campaign(rows)), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
