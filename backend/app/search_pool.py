"""Training-safe layer-1 search portfolio for three-layer research campaigns.

The pool deliberately stays below the LLM layer.  It uses only persisted
PUBLIC/META_TRAIN feedback envelopes supplied by the orchestrator and never
loads holdout, vault, or rating data.  The algorithms are intentionally small
and auditable: grammar random search, evolutionary mutation, a kernel-regression
surrogate, and a tabular Q-learning action policy.  A UCB scheduler allocates
trials among them.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .factors.similarity import expression_similarity
from .factors.return_path import return_path_correlation
from .miner.agent import mutate_expression, random_expression_for_family


SEARCH_POOL_SCHEMA_VERSION = "three_layer_l1_v1"
DEFAULT_SEARCH_ALGORITHMS = (
    "structured_random",
    "evolutionary",
    "surrogate_kernel",
    "q_learning",
    "residual_oof_beam",
)


@dataclass(frozen=True)
class SearchProposal:
    expression: str
    hypothesis: str
    metadata: dict


def _reward(node: dict) -> float:
    """Bound the authoritative training score for stable online policies."""
    try:
        value = float(node.get("public_score") or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return max(-1.0, min(1.0, value))


def _algorithm(node: dict) -> str:
    return str((node.get("proposal_meta") or {}).get("search_algorithm") or "")


def _compatible_nodes(nodes: list[dict], family: str) -> list[dict]:
    return [
        node for node in nodes
        if node.get("status") == "ok"
        and str((node.get("proposal_meta") or {}).get("target_family") or "") == family
        and str(node.get("expression") or "").strip()
    ]


def _ucb_algorithm(
    algorithms: tuple[str, ...],
    nodes: list[dict],
    rng: random.Random,
) -> tuple[str, dict]:
    stats = {}
    total = 0
    for name in algorithms:
        rewards = [_reward(node) for node in nodes if _algorithm(node) == name]
        total += len(rewards)
        stats[name] = {
            "n": len(rewards),
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        }
    unseen = [name for name in algorithms if stats[name]["n"] == 0]
    if unseen:
        chosen = unseen[0]
        reason = "deterministic_cold_start"
    else:
        scored = []
        for name in algorithms:
            row = stats[name]
            bonus = math.sqrt(2.0 * math.log(max(2, total)) / row["n"])
            scored.append((row["mean_reward"] + bonus, rng.random(), name))
        chosen = max(scored)[2]
        reason = "ucb1_training_reward"
    return chosen, {
        "policy": "ucb1",
        "reason": reason,
        "total_observations": total,
        "algorithm_stats": stats,
    }


def _best_node(nodes: list[dict]) -> dict | None:
    return max(nodes, key=_reward, default=None)


def _residual_beam_parent(nodes: list[dict]) -> tuple[dict | None, dict]:
    """Choose a high-score path that is least explained by other strong paths.

    This scheduler consumes only compressed PUBLIC+META_TRAIN return paths. It
    is a safe continuous-search proxy; exact OOF residual ranking is performed
    by ``residual_beam.residual_oof_beam_search`` when prediction matrices are
    available and is never inferred from the frozen rating layer.
    """
    viable = []
    for node in nodes:
        signature = (node.get("public_metrics") or {}).get(
            "training_return_path_signature"
        ) or {}
        if signature.get("available"):
            viable.append((node, signature))
    if not viable:
        return _best_node(nodes), {
            "residual_scope": "fallback_no_comparable_training_paths",
            "beam_candidates": 0,
        }
    scored = []
    for node, signature in viable:
        correlations = [
            abs(value)
            for other, other_signature in viable
            if other.get("id") != node.get("id")
            and (value := return_path_correlation(signature, other_signature)) is not None
        ]
        independence = 1.0 - max(correlations, default=0.0)
        score = 0.7 * _reward(node) + 0.3 * independence
        scored.append((score, independence, int(node.get("id") or 0), node))
    score, independence, _, parent = max(scored)
    return parent, {
        "residual_scope": "compressed_public_plus_meta_train_path_proxy",
        "exact_oof_required_for_promotion": True,
        "beam_candidates": len(scored),
        "parent_independence": round(independence, 6),
        "parent_residual_proxy_score": round(score, 6),
    }


def _surrogate_candidate(
    family: str,
    fields: list[str],
    nodes: list[dict],
    rng: random.Random,
) -> tuple[str, dict]:
    """Select one of eight grammar draws using kernel regression + uncertainty."""
    candidates = [random_expression_for_family(family, fields, rng) for _ in range(8)]
    if not nodes:
        return candidates[0], {"surrogate_candidates": 8, "surrogate_cold_start": True}
    scored = []
    for expression in candidates:
        neighbours = sorted(
            (
                (expression_similarity(expression, str(node["expression"])), _reward(node))
                for node in nodes
            ),
            reverse=True,
        )[:8]
        weights = [max(0.02, similarity) ** 2 for similarity, _ in neighbours]
        prediction = sum(w * reward for w, (_, reward) in zip(weights, neighbours)) / sum(weights)
        nearest = neighbours[0][0] if neighbours else 0.0
        uncertainty = 0.20 * (1.0 - nearest)
        scored.append((prediction + uncertainty, prediction, uncertainty, expression))
    score, prediction, uncertainty, expression = max(scored)
    return expression, {
        "surrogate_candidates": len(candidates),
        "surrogate_prediction": round(prediction, 6),
        "surrogate_uncertainty": round(uncertainty, 6),
        "surrogate_acquisition": round(score, 6),
        "surrogate_model": "similarity_kernel_regression_ucb",
    }


def _q_action(nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    actions = ("fresh", "mutate")
    stats = {}
    for action in actions:
        rewards = [
            _reward(node) for node in nodes
            if str((node.get("proposal_meta") or {}).get("search_action") or "") == action
        ]
        stats[action] = {
            "n": len(rewards),
            "q": sum(rewards) / len(rewards) if rewards else 0.0,
        }
    observations = sum(row["n"] for row in stats.values())
    epsilon = max(0.05, 0.35 / math.sqrt(max(1, observations + 1)))
    if observations < len(actions):
        action = actions[observations]
        reason = "tabular_cold_start"
    elif rng.random() < epsilon:
        action = rng.choice(actions)
        reason = "epsilon_exploration"
    else:
        action = max(actions, key=lambda name: (stats[name]["q"], name))
        reason = "greedy_q"
    return action, {
        "q_policy": "epsilon_greedy_sample_mean",
        "q_state": "task_x_mechanism",
        "q_action_stats": stats,
        "q_epsilon": round(epsilon, 6),
        "q_reason": reason,
    }


def propose_search_seed(
    *,
    family: str,
    fields: list[str],
    feedback_nodes: list[dict],
    algorithms: list[str] | tuple[str, ...] | None,
    rng: random.Random,
) -> SearchProposal:
    configured = tuple(algorithms or DEFAULT_SEARCH_ALGORITHMS)
    unknown = sorted(set(configured) - set(DEFAULT_SEARCH_ALGORITHMS))
    if not configured or unknown:
        raise ValueError(f"非法第一层搜索算法: {unknown or 'empty'}")
    nodes = _compatible_nodes(feedback_nodes, family)
    algorithm, policy_meta = _ucb_algorithm(configured, nodes, rng)
    parent = _best_node(nodes)
    action = "fresh"
    algorithm_meta: dict = {}

    if algorithm == "structured_random":
        expression = random_expression_for_family(family, fields, rng)
    elif algorithm == "evolutionary":
        action = "mutate" if parent else "fresh"
        expression = (
            mutate_expression(str(parent["expression"]), fields=fields, rng=rng)
            if parent else random_expression_for_family(family, fields, rng)
        )
    elif algorithm == "surrogate_kernel":
        expression, algorithm_meta = _surrogate_candidate(
            family, fields, nodes, rng
        )
    elif algorithm == "q_learning":
        action, algorithm_meta = _q_action(nodes, rng)
        expression = (
            mutate_expression(str(parent["expression"]), fields=fields, rng=rng)
            if action == "mutate" and parent
            else random_expression_for_family(family, fields, rng)
        )
    else:
        parent, algorithm_meta = _residual_beam_parent(nodes)
        action = "mutate" if parent else "fresh"
        expression = (
            mutate_expression(str(parent["expression"]), fields=fields, rng=rng)
            if parent
            else random_expression_for_family(family, fields, rng)
        )

    metadata = {
        "search_pool_schema": SEARCH_POOL_SCHEMA_VERSION,
        "search_algorithm": algorithm,
        "search_action": action,
        "search_policy": policy_meta,
        "search_parent_node_id": parent.get("id") if parent else None,
        "search_observations": len(nodes),
        "target_family": family,
        **algorithm_meta,
    }
    return SearchProposal(
        expression=expression,
        hypothesis=f"第一层 {algorithm} 在 {family} 机制中的训练安全候选",
        metadata=metadata,
    )
