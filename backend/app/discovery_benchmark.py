"""Bounded, isolated training-only arm comparison. No promotion side effects."""
from __future__ import annotations

import random
import time
from collections import Counter

from .dsl.engine import normalize_hash


def compare_arms(arms, budget, seed, propose, evaluate, *, progress=None, attempt_multiplier=5):
    if not arms or len(set(arms)) != len(arms) or not 1 <= budget <= 1000:
        raise ValueError("distinct nonempty arms and a budget in 1..1000 are required")
    # Separate feedback/RNG per arm; allocation never follows interim winners.
    states = {arm: {"rng": random.Random(seed), "seen": set(), "feedback": [],
                    "attempts": [], "evaluations": 0} for arm in arms}
    scheduler = random.Random(seed)
    limit = budget * max(1, attempt_multiplier)
    sequence = 0
    while True:
        active = [a for a, s in states.items() if s["evaluations"] < budget and len(s["attempts"]) < limit]
        if not active:
            break
        scheduler.shuffle(active)
        for arm in active:
            state = states[arm]
            start = time.monotonic()
            sequence += 1
            row = {"sequence": sequence, "requested": arm, "evaluated": False}
            try:
                proposal = propose(arm, state["feedback"], state["rng"])
                expression = proposal.expression
                row.update(expression=expression, proposal_meta=proposal.metadata,
                           executed=proposal.metadata.get("executed_algorithm", arm))
                key = normalize_hash(expression, direction_invariant=True)
                if key in state["seen"]:
                    row["status"] = "duplicate"
                else:
                    state["seen"].add(key)
                    # Failed evaluation consumes its budget too; don't give an
                    # error-prone arm unlimited hidden attempts.
                    state["evaluations"] += 1
                    row["evaluated"] = True
                    result = evaluate(expression)
                    row.update(status="ok", result=result)
                    discovery = result.get("discovery") or {}
                    state["feedback"].append({"id": sequence, "status": "ok", "expression": expression,
                        "public_score": discovery.get("learning_score", 0),
                        "public_metrics": {"discovery": discovery}, "proposal_meta": proposal.metadata})
            except Exception as exc:
                row.update(status="error", error=f"{type(exc).__name__}: {exc}"[:500])
            row["wall_seconds"] = time.monotonic() - start
            state["attempts"].append(row)
            if progress:
                progress({"sequence": sequence, "arm": arm, "evaluations": state["evaluations"],
                          "budget_per_arm": budget, "attempt": row})
    summary = []
    for arm, state in states.items():
        attempts = state["attempts"]
        passed = sum(bool((r.get("result", {}).get("discovery") or {}).get("passed")) for r in attempts)
        summary.append({"arm": arm, "evaluations": state["evaluations"], "attempts": len(attempts),
            "budget_complete": state["evaluations"] == budget,
            "executed": dict(Counter(r.get("executed", "proposal_error") for r in attempts)),
            "training_passed": passed, "wall_seconds": sum(r["wall_seconds"] for r in attempts),
            "errors": sum(r["status"] == "error" for r in attempts),
            "duplicates": sum(r["status"] == "duplicate" for r in attempts)})
    return {"protocol": "factorfactory.equal-budget-discovery/v1", "seed": seed,
        "budget_per_arm": budget, "summary": summary,
        "attempts": sorted([r for s in states.values() for r in s["attempts"]], key=lambda r: r["sequence"]),
        "scope": "training_only_not_independent_oos_or_net_increment_proof",
        "formal_eligible": False, "llm_cost": None}
