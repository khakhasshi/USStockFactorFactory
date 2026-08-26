"""Training-safe, quota-controlled layer-1 discovery fabric.

The portfolio is concurrent in intent, not a sequential pipeline.  It consumes
only PUBLIC/META_TRAIN feedback and targets long-run group shares of
30/25/20/15/10 percent.  Holdout, vault and rating data are unavailable here.
"""

from __future__ import annotations

import importlib.util
import math
import random
import re
from dataclasses import dataclass

import numpy as np

from .dsl.engine import normalize_hash, validate
from .factors.similarity import expression_similarity
from .miner.agent import mutate_expression, random_expression_for_family
from .qlib_native import ALPHA158_FEATURES, QLIB_UPSTREAM_COMMIT


SEARCH_POOL_SCHEMA_VERSION = "discovery_fabric_l1_v5"
SEARCH_POLICY_SCHEMA = "factorfactory.discovery-fabric/v5"
SEARCH_HEALTH_SCHEMA = "factorfactory.search-health/v3"
DEFAULT_SEARCH_HEALTH_CONFIG = {
    "duplicate_window": 100,
    "min_duplicate_observations": 50,
    "duplicate_rate_threshold": 0.35,
    "no_gate_record_patience": 150,
    "exploration_recovery_unique_evaluations": 60,
    # Recovery must be bounded by attempts as well as successes.  Requiring
    # only unique evaluations can leave an exhausted grammar in recovery
    # forever, producing duplicate audit rows without doing any backtests.
    "recovery_attempt_patience": 180,
    "algorithm_quarantine_min_attempts": 20,
    "algorithm_quarantine_duplicate_rate": 0.85,
    "algorithm_quarantine_unique_yield": 0.10,
    "elite_archive_size": 5,
}
SEARCH_GROUP_WEIGHTS = {
    "structural_search": 0.30,
    "residual_search": 0.25,
    "ml_residual_distillation": 0.20,
    "local_optimisation": 0.15,
    "high_risk_exploration": 0.10,
}
ALGORITHM_GROUPS = {
    "grammar_enumerative": "structural_search",
    "map_elites": "structural_search",
    "mcts_puct": "structural_search",
    "residual_oof_beam": "residual_search",
    "gbdt_residual_distill": "ml_residual_distillation",
    "evolutionary": "local_optimisation",
    "tpe_smac": "local_optimisation",
    "novelty_search": "high_risk_exploration",
    "cegis_repair": "high_risk_exploration",
    # Qlib is a provenance-pinned structural prior, not a separate scoring
    # authority.  It consumes part of the structural sleeve when enabled.
    "qlib_alpha158_prior": "structural_search",
    "qlib_joint_residual_distill": "ml_residual_distillation",
    # Kept for historical experiments and explicit ablations.
    "structured_random": "structural_search",
    "surrogate_kernel": "ml_residual_distillation",
    "q_learning": "local_optimisation",
}
DEFAULT_SEARCH_ALGORITHMS = (
    "grammar_enumerative", "map_elites", "mcts_puct",
    "residual_oof_beam", "gbdt_residual_distill",
    "evolutionary", "tpe_smac", "novelty_search", "cegis_repair",
)
SUPPORTED_SEARCH_ALGORITHMS = tuple(ALGORITHM_GROUPS)
RESET_EXPLORATION_ALGORITHMS = (
    "grammar_enumerative",
    "map_elites",
    "novelty_search",
    "cegis_repair",
    "structured_random",
    "qlib_alpha158_prior",
)
STAGNATION_PRIORITY_ALGORITHMS = (
    "residual_oof_beam",
    "gbdt_residual_distill",
    "qlib_joint_residual_distill",
)


@dataclass(frozen=True)
class SearchProposal:
    expression: str
    hypothesis: str
    metadata: dict


def _finite(value, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _clip01(value) -> float:
    return max(0.0, min(1.0, _finite(value)))


def _discovery(node: dict) -> dict:
    public = node.get("public_metrics") or {}
    if not isinstance(public, dict):
        public = {}
    discovery = public.get("discovery") or {}
    if isinstance(discovery, dict) and discovery:
        return discovery
    feedback = node.get("feedback_summary") or {}
    outcome = feedback.get("outcome") if isinstance(feedback, dict) else {}
    return outcome if isinstance(outcome, dict) else {}


def _gate_score(node: dict) -> float:
    return max(0.0, _finite(_discovery(node).get("gate_score")))


def _learning_score(node: dict) -> float:
    discovery = _discovery(node)
    return max(
        0.0,
        _finite(
            discovery.get("learning_score", node.get("public_score") or 0.0)
        ),
    )


def _expression_key(expression: str) -> str:
    """Direction-invariant AST key for the mandatory two-sided evaluation."""
    try:
        return normalize_hash(expression, direction_invariant=True)
    except (SyntaxError, TypeError, ValueError):
        return re.sub(r"\s+", "", str(expression or "")).lower()


def _reward(node: dict) -> float:
    """Gate-oriented reward; learning score is deliberately only a tiebreaker."""
    discovery = _discovery(node)
    components = discovery.get("gate_components") or {}
    component_values = [
        _clip01(value)
        for value in components.values()
        if isinstance(value, (int, float))
    ] if isinstance(components, dict) else []
    component_mean = (
        sum(component_values) / len(component_values)
        if component_values else 0.0
    )
    gate_quality = _clip01(_gate_score(node))
    learning_quality = _clip01(_learning_score(node) / 1.5)
    passed = 1.0 if discovery.get("passed") else 0.0
    return (
        0.45 * gate_quality
        + 0.35 * component_mean
        + 0.15 * learning_quality
        + 0.05 * passed
    )


def _algorithm(node: dict) -> str:
    return str((node.get("proposal_meta") or {}).get("search_algorithm") or "")


def _compatible_nodes(nodes: list[dict], family: str) -> list[dict]:
    return [node for node in nodes if node.get("status") == "ok"
            and str((node.get("proposal_meta") or {}).get("target_family") or "") == family
            and str(node.get("expression") or "").strip()]


def _unique_evaluated_nodes(nodes: list[dict]) -> list[dict]:
    unique: list[dict] = []
    seen: set[str] = set()
    for node in sorted(nodes, key=lambda row: int(row.get("id") or 0)):
        if node.get("status") != "ok":
            continue
        expression = str(node.get("expression") or "").strip()
        if not expression:
            continue
        key = _expression_key(expression)
        if key in seen:
            continue
        seen.add(key)
        unique.append(node)
    return unique


def _duplicate_attempt(node: dict) -> bool:
    meta = node.get("proposal_meta") or {}
    return bool(
        node.get("status") == "rejected"
        and meta.get("pre_evaluation_rejection") == "duplicate_normalized_ast"
    ) or bool(meta.get("campaign_exact_duplicate"))


def _resolved_health_config(config: dict | None) -> dict:
    resolved = dict(DEFAULT_SEARCH_HEALTH_CONFIG)
    for key, default in DEFAULT_SEARCH_HEALTH_CONFIG.items():
        value = (config or {}).get(key, default)
        try:
            resolved[key] = float(value) if isinstance(default, float) else int(value)
        except (TypeError, ValueError):
            resolved[key] = default
    resolved["duplicate_window"] = max(20, resolved["duplicate_window"])
    resolved["min_duplicate_observations"] = max(
        10,
        min(resolved["duplicate_window"], resolved["min_duplicate_observations"]),
    )
    resolved["duplicate_rate_threshold"] = max(
        0.05, min(0.95, resolved["duplicate_rate_threshold"])
    )
    resolved["no_gate_record_patience"] = max(
        20, resolved["no_gate_record_patience"]
    )
    resolved["exploration_recovery_unique_evaluations"] = max(
        10, resolved["exploration_recovery_unique_evaluations"]
    )
    resolved["recovery_attempt_patience"] = max(
        30, resolved["recovery_attempt_patience"]
    )
    resolved["algorithm_quarantine_min_attempts"] = max(
        5, resolved["algorithm_quarantine_min_attempts"]
    )
    resolved["algorithm_quarantine_duplicate_rate"] = max(
        0.50, min(0.99, resolved["algorithm_quarantine_duplicate_rate"])
    )
    resolved["algorithm_quarantine_unique_yield"] = max(
        0.0, min(0.50, resolved["algorithm_quarantine_unique_yield"])
    )
    resolved["elite_archive_size"] = max(1, resolved["elite_archive_size"])
    return resolved


def search_health(nodes: list[dict], config: dict | None = None) -> dict:
    """Detect duplicate collapse without reading holdout, vault, or rating data."""
    cfg = _resolved_health_config(config)
    ordered = sorted(nodes, key=lambda row: int(row.get("id") or 0))
    unique = _unique_evaluated_nodes(ordered)
    # Only duplicate collapse creates a recovery epoch. Historical v2 rows
    # may carry ``search_reset_triggered`` for mere score stagnation; treating
    # those as duplicate recovery permanently suppresses residual/ML arms.
    reset_rows = []
    for row in ordered:
        meta = row.get("proposal_meta") or {}
        reset_reasons = set(meta.get("search_reset_reasons") or [])
        duplicate_reset = (
            meta.get("search_recovery_kind") == "duplicate_collapse"
            or "recent_duplicate_rate" in reset_reasons
            or "recovery_attempts_exhausted" in reset_reasons
        )
        if meta.get("search_reset_triggered") and duplicate_reset:
            reset_rows.append(row)
    last_reset_id = int(reset_rows[-1].get("id") or 0) if reset_rows else 0
    prior_epoch = max(
        (
            int((row.get("proposal_meta") or {}).get("search_epoch") or 0)
            for row in ordered
        ),
        default=0,
    )
    epoch_unique = [row for row in unique if int(row.get("id") or 0) >= last_reset_id]
    epoch_rows = [
        row for row in ordered if int(row.get("id") or 0) >= last_reset_id
    ] if last_reset_id else []
    epoch_duplicate_count = sum(_duplicate_attempt(row) for row in epoch_rows)
    epoch_duplicate_rate = (
        epoch_duplicate_count / len(epoch_rows) if epoch_rows else 0.0
    )
    in_recovery = bool(last_reset_id) and len(epoch_unique) < int(
        cfg["exploration_recovery_unique_evaluations"]
    )
    recovery_exhausted = bool(
        in_recovery
        and len(epoch_rows) >= int(cfg["recovery_attempt_patience"])
        and epoch_duplicate_rate >= float(cfg["duplicate_rate_threshold"])
    )

    recent = ordered[-int(cfg["duplicate_window"]):]
    recent_duplicate_count = sum(_duplicate_attempt(row) for row in recent)
    duplicate_rate = recent_duplicate_count / len(recent) if recent else 0.0

    best_gate = -1.0
    last_record_index = -1
    for index, row in enumerate(unique):
        value = _gate_score(row)
        if value > best_gate + 1e-12:
            best_gate = value
            last_record_index = index
    unique_since_record = (
        len(unique) - 1 - last_record_index if unique else 0
    )
    duplicate_reasons = []
    if len(recent) >= int(cfg["min_duplicate_observations"]) and (
        duplicate_rate >= float(cfg["duplicate_rate_threshold"])
    ):
        duplicate_reasons.append("recent_duplicate_rate")
    stagnation_reasons = []
    if unique_since_record >= int(cfg["no_gate_record_patience"]):
        stagnation_reasons.append("no_new_gate_record")

    state = "healthy"
    reset_triggered = False
    epoch = prior_epoch
    if recovery_exhausted:
        state = "duplicate_space_exhausted"
        reset_triggered = True
        epoch = prior_epoch + 1
        duplicate_reasons = list(dict.fromkeys([
            *duplicate_reasons,
            "recovery_attempts_exhausted",
        ]))
    elif in_recovery:
        state = "duplicate_recovery"
    elif duplicate_reasons:
        state = "duplicate_reset_triggered"
        reset_triggered = True
        epoch = prior_epoch + 1
    elif stagnation_reasons:
        # Stagnation is a scientific allocation signal, not a broken search
        # space. Keep the full discovery fabric live and rebalance toward
        # residual/ML arms rather than opening a duplicate-recovery epoch.
        state = "stagnation_rebalance"

    elite_ids = [
        int(row.get("id") or 0)
        for row in sorted(unique, key=_reward, reverse=True)[
            : int(cfg["elite_archive_size"])
        ]
        if int(row.get("id") or 0)
    ]
    return {
        "schema": SEARCH_HEALTH_SCHEMA,
        "state": state,
        "recovery_kind": (
            "duplicate_collapse"
            if state.startswith("duplicate_")
            else "scientific_stagnation"
            if state == "stagnation_rebalance"
            else None
        ),
        "search_epoch": epoch,
        "search_reset_triggered": reset_triggered,
        "reset_reasons": duplicate_reasons,
        "stagnation_reasons": stagnation_reasons,
        "last_reset_node_id": last_reset_id or None,
        "unique_evaluations": len(unique),
        "epoch_unique_evaluations": len(epoch_unique),
        "epoch_attempts": len(epoch_rows),
        "epoch_duplicate_rate": round(epoch_duplicate_rate, 6),
        "recent_proposals": len(recent),
        "recent_duplicate_count": recent_duplicate_count,
        "recent_duplicate_rate": round(duplicate_rate, 6),
        "unique_since_gate_record": unique_since_record,
        "best_gate_score": round(max(0.0, best_gate), 6),
        "elite_node_ids": elite_ids,
        "thresholds": cfg,
    }


def _quota_ucb(
    algorithms: tuple[str, ...],
    nodes: list[dict],
    rng: random.Random,
    *,
    adaptive: bool = False,
    qlib_prior_share: float | None = None,
) -> tuple[str, dict]:
    """Allocate by unique evaluations and enforce each rolling cohort mix."""
    unique_nodes = _unique_evaluated_nodes(nodes)
    cohort_size = 60
    cohort_nodes = unique_nodes[-cohort_size:]
    reward_by_id: dict[int, float] = {}
    gate_delta_by_id: dict[int, float] = {}
    running_best_gate = 0.0
    for node in unique_nodes:
        gate = _gate_score(node)
        gate_delta = max(0.0, gate - running_best_gate)
        running_best_gate = max(running_best_gate, gate)
        gate_delta_by_id[int(node.get("id") or 0)] = gate_delta
        reward_by_id[int(node.get("id") or 0)] = min(
            1.0,
            0.72 * _reward(node) + 0.28 * _clip01(gate_delta / 0.02),
        )

    stats = {}
    total = 0
    for name in algorithms:
        algorithm_nodes = [node for node in unique_nodes if _algorithm(node) == name]
        rewards = [reward_by_id.get(int(node.get("id") or 0), 0.0) for node in algorithm_nodes]
        final_attempts = [node for node in nodes if _algorithm(node) == name]
        hidden_duplicate_attempts = sum(
            1
            for node in nodes
            for retry_name in list(
                (node.get("proposal_meta") or {}).get(
                    "campaign_retry_algorithms"
                ) or []
            )
            if str(retry_name) == name
        )
        proposal_attempts = len(final_attempts) + hidden_duplicate_attempts
        duplicate_attempts = (
            sum(_duplicate_attempt(node) for node in final_attempts)
            + hidden_duplicate_attempts
        )
        total += len(rewards)
        stats[name] = {
            "n": len(rewards),
            "unique_evaluations": len(rewards),
            "proposal_attempts": proposal_attempts,
            "hidden_duplicate_resamples": hidden_duplicate_attempts,
            "duplicate_attempts": duplicate_attempts,
            "duplicate_rate": round(
                duplicate_attempts / proposal_attempts, 6
            ) if proposal_attempts else 0.0,
            "unique_yield_rate": round(
                len(rewards) / proposal_attempts, 6
            ) if proposal_attempts else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "best_gate_score": max(
                (_gate_score(node) for node in algorithm_nodes), default=0.0
            ),
            "mean_gate_delta": (
                sum(gate_delta_by_id.get(int(node.get("id") or 0), 0.0) for node in algorithm_nodes)
                / len(algorithm_nodes)
                if algorithm_nodes else 0.0
            ),
        }
    enabled_groups = {ALGORITHM_GROUPS[name] for name in algorithms}
    group_weights = {
        group: SEARCH_GROUP_WEIGHTS[group] for group in enabled_groups
    }
    # After a meaningful warm-up, shift at most half of the base allocation
    # toward groups that produce unique gate improvements per CPU second.  A
    # hard floor preserves exploration and no HOLDOUT/Vault metric is read.
    if adaptive and len(unique_nodes) >= 50:
        efficiencies = {}
        for group in enabled_groups:
            members = [node for node in unique_nodes if ALGORITHM_GROUPS.get(_algorithm(node)) == group]
            utility = sum(
                reward_by_id.get(int(node.get("id") or 0), 0.0)
                + 2.0 * gate_delta_by_id.get(int(node.get("id") or 0), 0.0)
                for node in members
            )
            seconds = sum(
                max(
                    0.05,
                    _finite(
                        ((node.get("public_metrics") or {}).get("evaluation_runtime") or {}).get("total_ms")
                    ) / 1000.0,
                )
                for node in members
            )
            efficiencies[group] = utility / max(1.0, seconds)
        total_efficiency = sum(efficiencies.values())
        if total_efficiency > 0:
            floor = 0.05
            mixed = {
                group: 0.50 * SEARCH_GROUP_WEIGHTS[group]
                + 0.50 * efficiencies[group] / total_efficiency
                for group in enabled_groups
            }
            group_weights = {
                group: max(floor, value) for group, value in mixed.items()
            }
    weight_sum = sum(group_weights.values())
    cohort_total = len(cohort_nodes)
    group_stats = {}
    for group in enabled_groups:
        members = [name for name in algorithms if ALGORITHM_GROUPS[name] == group]
        n = sum(
            1 for node in cohort_nodes
            if _algorithm(node) in members
        )
        target = group_weights[group] / weight_sum
        group_stats[group] = {
            "n": n, "target_share": round(target, 6),
            "actual_share": round(n / cohort_total, 6) if cohort_total else 0.0,
            "deficit_trials": round(target * (cohort_total + 1) - n, 6),
        }
    unseen = [name for name in algorithms if stats[name]["n"] == 0]
    if unseen:
        group = max({ALGORITHM_GROUPS[name] for name in unseen},
                    key=lambda item: (group_stats[item]["deficit_trials"], SEARCH_GROUP_WEIGHTS[item], item))
        chosen = next(name for name in unseen if ALGORITHM_GROUPS[name] == group)
        reason = "quota_aware_cold_start"
    else:
        group = max(enabled_groups, key=lambda item: (group_stats[item]["deficit_trials"], -group_stats[item]["n"], item))
        members = [name for name in algorithms if ALGORITHM_GROUPS[name] == group]
        if group == "structural_search" and "qlib_alpha158_prior" in members and qlib_prior_share is not None:
            target_share = max(0.0, min(0.30, float(qlib_prior_share)))
            qlib_n = stats["qlib_alpha158_prior"]["n"]
            qlib_deficit = target_share * (total + 1) - qlib_n
            if qlib_deficit > 0:
                members = ["qlib_alpha158_prior"]
            elif len(members) > 1:
                members = [name for name in members if name != "qlib_alpha158_prior"]
        chosen = max((stats[name]["mean_reward"] + math.sqrt(2 * math.log(max(2, total)) / stats[name]["n"]) - 0.60 * stats[name]["duplicate_rate"], rng.random(), name)
                     for name in members)[2]
        reason = "unique_quota_then_gate_delta_ucb1"
    return chosen, {
        "schema": SEARCH_POLICY_SCHEMA, "policy": "quota_deficit_plus_ucb1",
        "reason": reason, "selected_group": group,
        "selected_group_target_share": group_stats[group]["target_share"],
        "quota_unit": "unique_evaluated_normalized_ast",
        "reward_semantics": "gate_quality_plus_incremental_gate_record",
        "adaptive_efficiency_enabled": bool(adaptive),
        "allocation_cohort_size": cohort_size,
        "allocation_cohort_observations": cohort_total,
        "qlib_prior_target_share": qlib_prior_share,
        "total_observations": total, "group_stats": group_stats,
        "algorithm_stats": stats,
    }


def _best(nodes: list[dict]) -> dict | None:
    return max(nodes, key=_reward, default=None)


def _draw(family: str, fields: list[str], rng: random.Random, n: int) -> list[str]:
    return [random_expression_for_family(family, fields, rng) for _ in range(n)]


def _novelty(expression: str, nodes: list[dict]) -> float:
    return 1.0 if not nodes else max(0.0, 1.0 - max(expression_similarity(expression, str(node.get("expression") or "")) for node in nodes))


def _depth(expression: str) -> int:
    value = peak = 0
    for char in expression:
        value += (char == "(") - (char == ")")
        peak = max(peak, value)
    return peak


_QLIB_MECHANISM_FAMILIES = {
    "momentum": {"trend", "trend_geometry", "momentum", "extreme_timing"},
    "reversal": {"price_location", "direction_count", "direction_magnitude", "extreme_timing"},
    "volatility": {"volatility", "kbar", "trend_geometry"},
    "liquidity": {"volume", "volume_direction", "price_volume"},
    "volume_price_interaction": {"price_volume", "volume", "volume_direction", "kbar"},
    "gap_intraday": {"kbar", "price", "price_location"},
    "price_relationship": {"price", "price_location", "trend_geometry"},
}


def _expression_fields(expression: str) -> set[str]:
    function_names = set(re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", expression.lower()))
    tokens = set(re.findall(r"\b[a-z_][a-z0-9_]*\b", expression.lower()))
    return tokens - function_names


def _qlib_candidates(
    family: str,
    fields: list[str],
    *,
    include_low_fidelity_vwap: bool = False,
) -> list:
    allowed_families = _QLIB_MECHANISM_FAMILIES.get(
        family, {feature.family for feature in ALPHA158_FEATURES}
    )
    available_fields = set(fields)
    return [
        feature
        for feature in ALPHA158_FEATURES
        if feature.family in allowed_families
        and _expression_fields(feature.expression) <= available_fields
        and (include_low_fidelity_vwap or feature.name != "VWAP0")
    ]


def _qlib_prior(
    family: str,
    fields: list[str],
    nodes: list[dict],
    rng: random.Random,
) -> tuple[str, dict]:
    candidates = _qlib_candidates(family, fields)
    if not candidates:
        return random_expression_for_family(family, fields, rng), {
            "dependency_fallback": "no_compatible_alpha158_feature",
            "qlib_upstream_commit": QLIB_UPSTREAM_COMMIT,
        }
    seen = {normalize_hash(str(node.get("expression") or "")) for node in nodes}
    unseen = [feature for feature in candidates if normalize_hash(feature.expression) not in seen]
    pool = unseen or candidates
    scored = sorted(
        (
            _novelty(feature.expression, nodes) - 0.08 * min(1.0, len(feature.expression) / 300),
            rng.random(),
            feature,
        )
        for feature in pool
    )
    _, _, selected = scored[-1]
    return selected.expression, {
        "qlib_schema": "factorfactory.qlib-alpha158/v1",
        "qlib_upstream_commit": QLIB_UPSTREAM_COMMIT,
        "qlib_feature": selected.name,
        "qlib_feature_family": selected.family,
        "qlib_exact_seed": True,
        "qlib_candidates": len(candidates),
        "qlib_unseen_candidates": len(unseen),
        "fidelity_stage": "pinned_alpha158_prior_then_factorfactory_full_backtest",
    }


def _qlib_joint_prior(
    family: str,
    fields: list[str],
    nodes: list[dict],
    candidates: list[dict] | None,
    rng: random.Random,
    excluded_hashes: set[str] | None = None,
) -> tuple[str, dict]:
    valid = []
    for row in candidates or []:
        expression = str(row.get("expression") or "").strip()
        if expression and not validate(expression, fields):
            valid.append(row)
    if not valid:
        expression, meta = _qlib_prior(family, fields, nodes, rng)
        return expression, {
            **meta,
            "joint_model_ready": False,
            "dependency_fallback": "joint_model_artifact_not_ready",
        }
    seen = {
        _expression_key(str(node.get("expression") or "")) for node in nodes
    } | set(excluded_hashes or ())
    unseen = [row for row in valid if _expression_key(str(row["expression"])) not in seen]
    if not unseen:
        expression = random_expression_for_family(family, fields, rng)
        return expression, {
            "joint_model_ready": True,
            "joint_candidate_space_exhausted": True,
            "joint_model_candidates": len(valid),
            "joint_model_unseen_candidates": 0,
            "dependency_fallback": "joint_dsl_pool_exhausted_grammar_escape",
            "exact_oof_required_for_promotion": True,
        }
    pool = unseen
    selected = max(
        pool,
        key=lambda row: (
            _novelty(str(row["expression"]), nodes)
            - 0.05 * min(1.0, len(str(row["expression"])) / 500.0),
            rng.random(),
        ),
    )
    return str(selected["expression"]), {
        "joint_model_ready": True,
        "joint_model_protocol": "factorfactory.qlib-joint-residual-distill/v1",
        "joint_model_candidates": len(valid),
        "joint_model_unseen_candidates": len(unseen),
        "distillation_kind": selected.get("distillation_kind"),
        "distillation_components": list(selected.get("components") or []),
        "exact_oof_required_for_promotion": True,
        "fidelity_stage": "sample_level_oof_residual_model_then_dsl_then_v4_2",
    }


def _residual_oof_candidate(
    family: str,
    fields: list[str],
    nodes: list[dict],
    candidates: list[dict] | None,
    rng: random.Random,
    excluded_hashes: set[str] | None = None,
) -> tuple[str, dict]:
    seen = {
        _expression_key(str(node.get("expression") or "")) for node in nodes
    } | set(excluded_hashes or ())
    valid = [
        row for row in (candidates or [])
        if str(row.get("expression") or "").strip()
        and validate(str(row["expression"]), fields) is None
        and (
            not row.get("family")
            or str(row.get("family")) == family
        )
    ]
    # If a small beam has no candidate for this mechanism, allow the global
    # beam rather than silently reverting to compressed return-path proxies.
    if not valid:
        valid = [
            row for row in (candidates or [])
            if str(row.get("expression") or "").strip()
            and validate(str(row["expression"]), fields) is None
        ]
    unseen = [
        row for row in valid
        if _expression_key(str(row["expression"])) not in seen
    ]
    if not unseen:
        return random_expression_for_family(family, fields, rng), {
            "residual_scope": "actual_oof_artifact_unavailable_or_exhausted",
            "residual_oof_ready": False,
            "beam_candidates": len(valid),
            "dependency_fallback": "actual_oof_beam_exhausted_grammar_escape",
        }
    selected = max(
        unseen,
        key=lambda row: (
            float(row.get("score") or 0.0)
            + 0.05 * _novelty(str(row["expression"]), nodes),
            rng.random(),
        ),
    )
    return str(selected["expression"]), {
        "residual_scope": "sample_level_cross_sectional_time_ordered_oof",
        "residual_oof_ready": True,
        "beam_candidates": len(valid),
        "beam_unseen_candidates": len(unseen),
        "residual_rank_ic": selected.get("residual_rank_ic"),
        "incremental_oof_ic": selected.get("incremental_oof_ic"),
        "return_source_independence": selected.get("independence"),
        "residual_stability": selected.get("stability"),
        "residual_beam_score": selected.get("score"),
        "residual_candidate_hash": selected.get("normalized_hash"),
        "exact_oof_completed_before_full_evaluation": True,
        "fidelity_stage": "actual_training_layer_oof_residual_then_v4_2",
    }


def _features(expression: str) -> dict[str, float]:
    lowered = expression.lower()
    tokens = re.findall(r"[a-z_][a-z0-9_]*|\d+(?:\.\d+)?", lowered)
    numbers = [float(token) for token in tokens if token[0].isdigit()]
    functions = re.findall(r"\b([a-z_][a-z0-9_]*)\s*\(", lowered)
    return {
        "length": min(1.0, len(expression) / 500), "depth": min(1.0, _depth(expression) / 12),
        "function_count": min(1.0, len(functions) / 20),
        "function_diversity": len(set(functions)) / max(1, len(functions)),
        "window_mean": min(1.0, sum(numbers) / max(1, len(numbers)) / 252),
        "window_max": min(1.0, max(numbers, default=0) / 504),
        "rank": float("rank(" in lowered), "corr": float("corr(" in lowered),
        "volatility": float("std(" in lowered or "atr(" in lowered),
        "returns": float("returns(" in lowered or "delta(" in lowered),
        "fundamental": float(any(field in lowered for field in ("pe_", "pb", "ps_", "mv"))),
    }


def _grammar(family: str, fields: list[str], nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    candidates = _draw(family, fields, rng, 32)
    scored = []
    for expression in candidates:
        features = _features(expression)
        simplicity = 1 - .55 * features["depth"] - .45 * features["length"]
        scored.append((.72 * _novelty(expression, nodes) + .28 * simplicity, expression))
    score, expression = max(scored)
    return expression, {"structural_candidates": 32, "structural_prescreen": "novelty_complexity_pareto",
                        "structural_acquisition": round(score, 6), "fidelity_stage": "structure_proxy_then_full_backtest"}


def _niche(expression: str) -> tuple[int, int, int, int]:
    f = _features(expression)
    return min(3, int(f["depth"] * 4)), min(3, int(f["function_count"] * 4)), int(f["corr"]), int(f["fundamental"])


def _map_elites(family: str, fields: list[str], nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    elites = {}
    for node in nodes:
        niche = _niche(str(node["expression"]))
        if niche not in elites or _reward(node) > _reward(elites[niche]):
            elites[niche] = node
    candidates = _draw(family, fields, rng, 24)
    if elites:
        parent = rng.choice(list(elites.values()))
        candidates.extend(mutate_expression(str(parent["expression"]), fields=fields, rng=rng) for _ in range(8))
    scored = []
    for expression in candidates:
        niche = _niche(expression)
        coverage = 1.0 if niche not in elites else max(0.0, .4 - _reward(elites[niche]))
        scored.append((coverage + .35 * _novelty(expression, nodes), expression, niche))
    score, expression, niche = max(scored)
    return expression, {"map_elites_archive_size": len(elites), "map_elites_candidate_niche": list(niche),
                        "map_elites_acquisition": round(score, 6), "fidelity_stage": "quality_diversity_proxy_then_full_backtest"}


def _mcts(family: str, fields: list[str], nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    if not nodes:
        return random_expression_for_family(family, fields, rng), {"mcts_rollouts": 1, "dependency_fallback": "no_parent_nodes"}
    total = len(nodes)
    candidates = []
    for node in sorted(nodes, key=_reward, reverse=True)[:12]:
        visits = max(1, int((node.get("proposal_meta") or {}).get("mcts_visits") or 1))
        prior = .5 + .5 * _reward(node)
        candidates.append((_reward(node) + 1.25 * prior * math.sqrt(total) / (1 + visits), rng.random(), node))
    parent = max(candidates)[2]
    rollouts = [mutate_expression(str(parent["expression"]), fields=fields, rng=rng) for _ in range(16)]
    expression = max(rollouts, key=lambda value: (_novelty(value, nodes), -len(value)))
    return expression, {"mcts_policy": "PUCT_over_persisted_nodes", "mcts_rollouts": 16,
                        "mcts_parent_reward": round(_reward(parent), 6), "mcts_visits": 1,
                        "selected_parent_id": parent.get("id")}


def _targeted_residual_gate(
    *,
    branch: dict,
    fields: list[str],
    feedback_nodes: list[dict],
    rng: random.Random,
) -> tuple[str, dict, dict]:
    """Create simple training-safe gates around one declared seed node."""
    seed_node_id = int(branch.get("seed_node_id") or 0)
    parent = next(
        (row for row in feedback_nodes
         if int(row.get("id") or 0) == seed_node_id
         and row.get("status") == "ok"),
        None,
    )
    if parent is None:
        raise ValueError(f"专属残差分支找不到训练节点 {seed_node_id}")
    seed_expression = str(parent.get("expression") or "").strip()
    candidates: list[tuple[str, str]] = []
    if "close" in fields:
        candidates.extend([
            (f"rank(({seed_expression}) * (1 - rank(ts_std(returns(close, 1), 60))))", "low_volatility_regime"),
            (f"rank(({seed_expression}) * rank(ts_std(returns(close, 1), 20) / (ts_std(returns(close, 1), 120) + 1e-9)))", "relative_volatility_regime"),
            (f"rank(({seed_expression}) * (1 - rank(abs(returns(close, 20)))))", "trend_neutrality"),
        ])
    if {"close", "high", "low"}.issubset(fields):
        candidates.append((
            f"rank(({seed_expression}) * rank((close - low) / (high - low + 1e-9)))",
            "range_efficiency",
        ))
    if "amount" in fields:
        candidates.append((
            f"rank(({seed_expression}) * rank(ts_mean(amount, 20)))",
            "liquidity_support",
        ))
    if "turnover_rate" in fields:
        candidates.append((
            f"rank(({seed_expression}) * rank(ts_mean(turnover_rate, 20)))",
            "turnover_support",
        ))
    valid = [(expression, axis) for expression, axis in candidates
             if validate(expression, fields) is None]
    if not valid:
        raise ValueError("专属残差分支没有字段兼容的条件门控表达式")
    expression, axis = max(
        valid,
        key=lambda item: (_novelty(item[0], feedback_nodes), rng.random()),
    )
    return expression, parent, {
        "targeted_branch_id": str(branch.get("branch_id") or "targeted_residual_gate"),
        "targeted_seed_node_id": seed_node_id,
        "targeted_branch_mode": "residual_conditional_gate",
        "conditional_gate_axis": axis,
        "directed_research": True,
        "residual_scope": "declared_seed_training_residual",
    }


def _fit_stumps(rows: list[dict[str, float]], targets: list[float], rounds: int = 16):
    keys, base = sorted(rows[0]), sum(targets) / len(targets)
    predictions, stumps = [base] * len(rows), []
    for _ in range(rounds):
        residuals = [target - pred for target, pred in zip(targets, predictions)]
        best = None
        for key in keys:
            values = sorted(row[key] for row in rows)
            for threshold in sorted(set(values[::max(1, len(values) // 5)])):
                left = [residual for row, residual in zip(rows, residuals) if row[key] <= threshold]
                right = [residual for row, residual in zip(rows, residuals) if row[key] > threshold]
                if not left or not right:
                    continue
                lv, rv = sum(left) / len(left), sum(right) / len(right)
                loss = sum((residual - (lv if row[key] <= threshold else rv)) ** 2 for row, residual in zip(rows, residuals))
                if best is None or loss < best[0]:
                    best = loss, key, threshold, .22 * lv, .22 * rv
        if best is None:
            break
        _, key, threshold, lv, rv = best
        stumps.append((key, threshold, lv, rv))
        predictions = [pred + (lv if row[key] <= threshold else rv) for row, pred in zip(rows, predictions)]
    return lambda row: base + sum(lv if row[key] <= threshold else rv for key, threshold, lv, rv in stumps), len(stumps)


def _fit_gbdt(rows: list[dict[str, float]], targets: list[float]):
    """Prefer a native deterministic GBDT, retain an auditable local fallback."""
    keys = sorted(rows[0])
    matrix = np.asarray([[row[key] for key in keys] for row in rows], dtype=float)
    native_error = None
    if importlib.util.find_spec("lightgbm"):
        try:
            import lightgbm as lgb
            dataset = lgb.Dataset(
                matrix, label=np.asarray(targets, dtype=float), free_raw_data=False
            )
            model = lgb.train(
                {
                    "objective": "regression", "metric": "l2",
                    "learning_rate": 0.06, "max_depth": 3, "num_leaves": 7,
                    "min_data_in_leaf": 4, "lambda_l2": 1.0,
                    "feature_fraction": 1.0, "bagging_fraction": 1.0,
                    "seed": 1729, "num_threads": 1, "deterministic": True,
                    "force_col_wise": True, "verbosity": -1,
                },
                dataset,
                num_boost_round=48,
            )
            return (
                lambda row: float(model.predict(np.asarray([[row[key] for key in keys]], dtype=float))[0]),
                "lightgbm_native_deterministic_regressor", 48, None,
            )
        except Exception as exc:
            native_error = f"lightgbm:{type(exc).__name__}"
    if importlib.util.find_spec("catboost"):
        try:
            from catboost import CatBoostRegressor
            model = CatBoostRegressor(
                iterations=48, depth=3, learning_rate=0.06,
                loss_function="RMSE", random_seed=1729,
                thread_count=1, verbose=False,
            )
            model.fit(matrix, targets)
            return (
                lambda row: float(model.predict([[row[key] for key in keys]])[0]),
                "catboost_deterministic_regressor", 48, native_error,
            )
        except Exception as exc:
            native_error = ";".join(filter(None, [native_error, f"catboost:{type(exc).__name__}"]))
    predict, rounds = _fit_stumps(rows, targets)
    return predict, "internal_hist_gradient_boosted_stumps", rounds, native_error


def _gbdt(
    family: str,
    fields: list[str],
    nodes: list[dict],
    rng: random.Random,
    *,
    qlib_candidate_pool: bool = False,
    residual_oof_candidates: list[dict] | None = None,
) -> tuple[str, dict]:
    candidates = _draw(family, fields, rng, 48)
    candidates += [mutate_expression(str(parent["expression"]), fields=fields, rng=rng) for parent in sorted(nodes, key=_reward, reverse=True)[:8]]
    qlib_features = _qlib_candidates(family, fields) if qlib_candidate_pool else []
    candidates += [feature.expression for feature in qlib_features]
    residual_rows = [
        row for row in (residual_oof_candidates or [])
        if str(row.get("expression") or "").strip()
        and row.get("score") is not None
    ]
    if len(residual_rows) < 8:
        expression = max(candidates, key=lambda value: (_novelty(value, nodes), -len(value)))
        return expression, {"model_backend": "structural_novelty_fallback", "dependency_fallback": "requires_8_actual_oof_residual_labels",
                            "training_rows": len(residual_rows), "distillation_candidates": len(candidates),
                            "qlib_candidate_pool": qlib_candidate_pool,
                            "qlib_candidate_count": len(qlib_features),
                            "qlib_upstream_commit": QLIB_UPSTREAM_COMMIT if qlib_candidate_pool else None}
    rows = [_features(str(row["expression"])) for row in residual_rows]
    # These labels come from the strict time-ordered OOF residual artifact,
    # not from the candidate's ordinary training score or expression novelty.
    targets = [float(row["score"]) for row in residual_rows]
    predict, backend, rounds, native_error = _fit_gbdt(rows, targets)
    score, expression = max((predict(_features(value)) + .08 * _novelty(value, nodes), value) for value in candidates)
    native = [name for name in ("lightgbm", "catboost") if importlib.util.find_spec(name)]
    return expression, {"model_backend": backend, "optional_native_backends_detected": native,
                        "native_backend_error": native_error,
                        "training_target": "actual_oof_residual_beam_score",
                        "exact_oof_completed_before_distillation": True, "training_rows": len(rows), "boosting_rounds": rounds,
                        "distillation_candidates": len(candidates), "distillation_acquisition": round(score, 6),
                        "qlib_candidate_pool": qlib_candidate_pool,
                        "qlib_candidate_count": len(qlib_features),
                        "qlib_upstream_commit": QLIB_UPSTREAM_COMMIT if qlib_candidate_pool else None}


def _tpe(family: str, fields: list[str], nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    if len(nodes) < 8:
        parent = _best(nodes)
        expression = mutate_expression(str(parent["expression"]), fields=fields, rng=rng) if parent else random_expression_for_family(family, fields, rng)
        return expression, {"dependency_fallback": "requires_8_training_observations", "tpe_rows": len(nodes)}
    ordered = sorted(nodes, key=_reward, reverse=True)
    good, bad = ordered[:max(2, len(ordered) // 4)], ordered[max(2, len(ordered) // 4):]
    candidates = [mutate_expression(str(rng.choice(good)["expression"]), fields=fields, rng=rng) for _ in range(32)] + _draw(family, fields, rng, 12)
    def density(expression, sample):
        return sum(max(.01, expression_similarity(expression, str(node["expression"]))) ** 3 for node in sample) / max(1, len(sample))
    score, expression = max((math.log(1e-4 + density(value, good)) - math.log(1e-4 + density(value, bad)) + .12 * _novelty(value, nodes), value) for value in candidates)
    return expression, {"local_model": "TPE_density_ratio_with_SMAC_incumbent_mutation", "tpe_good_rows": len(good),
                        "tpe_bad_rows": len(bad), "local_candidates": len(candidates), "local_acquisition": round(score, 6)}


def _novel(family: str, fields: list[str], nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    candidates = _draw(family, fields, rng, 48)
    expression = max(candidates, key=lambda value: (_novelty(value, nodes), -len(value)))
    return expression, {"novelty_archive_size": len(nodes), "novelty_candidates": 48,
                        "novelty_score": round(_novelty(expression, nodes), 6), "risk_class": "high_exploration"}


def _cegis(family: str, fields: list[str], feedback: list[dict], nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    failures = [node for node in feedback if node.get("status") != "ok" and str(node.get("expression") or "").strip()]
    if not failures:
        expression, meta = _grammar(family, fields, nodes, rng)
        return expression, {**meta, "dependency_fallback": "no_counterexamples", "cegis_counterexamples": 0}
    source = failures[-1]
    original, error = str(source["expression"]), str(source.get("error") or "").lower()
    repaired = original.replace("ps(", "ps_ttm(").replace("volume", "vol")
    repair = "semantic_repair"
    if repaired == original or any(token in error for token in ("too long", "过长", "复杂", "未知", "illegal")):
        repaired, repair = mutate_expression(original, fields=fields, rng=rng), "mutation_after_counterexample"
    if repaired == original:
        repaired, repair = random_expression_for_family(family, fields, rng), "grammar_regeneration_after_counterexample"
    return repaired, {"cegis_counterexamples": len(failures), "cegis_source_node_id": source.get("id"),
                      "cegis_error_class": error[:160], "cegis_repair": repair, "risk_class": "falsification_guided"}


def _surrogate(family: str, fields: list[str], nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    candidates = _draw(family, fields, rng, 8)
    if not nodes:
        return candidates[0], {"surrogate_candidates": 8, "surrogate_cold_start": True}
    scored = []
    for expression in candidates:
        neighbours = sorted(((expression_similarity(expression, str(node["expression"])), _reward(node)) for node in nodes), reverse=True)[:8]
        weights = [max(.02, similarity) ** 2 for similarity, _ in neighbours]
        prediction = sum(weight * reward for weight, (_, reward) in zip(weights, neighbours)) / sum(weights)
        uncertainty = .2 * (1 - neighbours[0][0])
        scored.append((prediction + uncertainty, prediction, uncertainty, expression))
    score, prediction, uncertainty, expression = max(scored)
    return expression, {"surrogate_candidates": 8, "surrogate_prediction": round(prediction, 6),
                        "surrogate_uncertainty": round(uncertainty, 6), "surrogate_acquisition": round(score, 6)}


def _q_action(nodes: list[dict], rng: random.Random) -> tuple[str, dict]:
    actions, stats = ("fresh", "mutate"), {}
    for action in actions:
        rewards = [_reward(node) for node in nodes if str((node.get("proposal_meta") or {}).get("search_action") or "") == action]
        stats[action] = {"n": len(rewards), "q": sum(rewards) / len(rewards) if rewards else 0.0}
    observations = sum(row["n"] for row in stats.values())
    epsilon = max(.05, .35 / math.sqrt(max(1, observations + 1)))
    if observations < 2:
        action, reason = actions[observations], "tabular_cold_start"
    elif rng.random() < epsilon:
        action, reason = rng.choice(actions), "epsilon_exploration"
    else:
        action, reason = max(actions, key=lambda name: (stats[name]["q"], name)), "greedy_q"
    return action, {"q_policy": "epsilon_greedy_sample_mean", "q_action_stats": stats, "q_epsilon": round(epsilon, 6), "q_reason": reason}


def propose_search_seed(*, family: str, fields: list[str], feedback_nodes: list[dict],
                        algorithms: list[str] | tuple[str, ...] | None, rng: random.Random,
                        allocation_history: list[dict] | None = None,
                        health_config: dict | None = None,
                        avoid_algorithms: list[str] | tuple[str, ...] | set[str] | None = None,
                        qlib_candidate_pool: bool = False,
                        qlib_joint_candidates: list[dict] | None = None,
                        residual_oof_candidates: list[dict] | None = None,
                        excluded_expression_hashes: set[str] | None = None,
                        adaptive_allocation: bool = False,
                        qlib_prior_share: float | None = None,
                        targeted_branch: dict | None = None) -> SearchProposal:
    configured = tuple(algorithms or DEFAULT_SEARCH_ALGORITHMS)
    unknown = sorted(set(configured) - set(SUPPORTED_SEARCH_ALGORITHMS))
    if not configured or unknown:
        raise ValueError(f"非法第一层搜索算法: {unknown or 'empty'}")
    if targeted_branch and "residual_oof_beam" not in configured:
        raise ValueError("专属残差分支要求启用 residual_oof_beam")
    # Quotas are task-global. Keeping them family-local repeatedly cold-starts
    # Grammar whenever the mechanism scheduler changes family and can turn a
    # nominal 30% structural sleeve into a dominant one. Parents, surrogates
    # and residual comparisons remain family-local below.
    allocation_nodes = [
        node
        for node in (
            allocation_history
            if allocation_history is not None
            else feedback_nodes
        )
        if str(node.get("expression") or "").strip()
    ]
    nodes = _compatible_nodes(feedback_nodes, family)
    health = search_health(allocation_nodes, health_config)
    cfg = _resolved_health_config(health_config)
    recovery_algorithms = tuple(
        name for name in RESET_EXPLORATION_ALGORITHMS if name in configured
    )
    active_algorithms = (
        recovery_algorithms
        if health["state"] in {
            "duplicate_reset_triggered",
            "duplicate_recovery",
            "duplicate_space_exhausted",
        }
        and recovery_algorithms
        else configured
    )
    # Quarantine is based on a rolling attempt window.  Therefore a collapsed
    # arm is temporarily removed, but naturally becomes eligible again after
    # other arms have generated enough new evidence.  This prevents both a
    # permanent ban and the old infinite duplicate loop.
    recent_window = max(
        int(cfg["duplicate_window"]) * 2,
        int(cfg["algorithm_quarantine_min_attempts"]),
    )
    _, recent_policy = _quota_ucb(
        active_algorithms,
        allocation_nodes[-recent_window:],
        rng,
        adaptive=False,
        qlib_prior_share=qlib_prior_share,
    )
    quarantined = {
        name
        for name, stats in recent_policy["algorithm_stats"].items()
        if int(stats["proposal_attempts"]) >= int(
            cfg["algorithm_quarantine_min_attempts"]
        )
        and float(stats["duplicate_rate"]) >= float(
            cfg["algorithm_quarantine_duplicate_rate"]
        )
        and float(stats["unique_yield_rate"]) <= float(
            cfg["algorithm_quarantine_unique_yield"]
        )
    }
    if health["state"] == "stagnation_rebalance":
        # These are the arms specifically intended to explain the incumbent's
        # unexplained return. They remain available during score stagnation,
        # even when their recent finite grammar pool had poor unique yield.
        quarantined -= set(STAGNATION_PRIORITY_ALGORITHMS)
    avoided = set(avoid_algorithms or ()) & set(active_algorithms)
    eligible_algorithms = tuple(
        name for name in active_algorithms
        if name not in quarantined and name not in avoided
    )
    if not eligible_algorithms:
        # Preserve liveness when every arm has recently collapsed.  Prefer the
        # least wasteful arm and let the caller escape to another mechanism
        # family on the next bounded retry.
        fallback_pool = tuple(
            name for name in active_algorithms if name not in avoided
        ) or active_algorithms
        eligible_algorithms = (min(
            fallback_pool,
            key=lambda name: (
                recent_policy["algorithm_stats"][name]["duplicate_rate"],
                -recent_policy["algorithm_stats"][name]["unique_yield_rate"],
                name,
            ),
        ),)
    algorithm, policy = _quota_ucb(
        eligible_algorithms,
        allocation_nodes,
        rng,
        adaptive=adaptive_allocation,
        qlib_prior_share=qlib_prior_share,
    )
    if targeted_branch:
        algorithm = "residual_oof_beam"
        policy["reason"] = "declared_targeted_residual_conditional_gate"
        policy["targeted_branch_id"] = str(
            targeted_branch.get("branch_id") or "targeted_residual_gate"
        )
    policy["search_health"] = health
    policy["configured_algorithms"] = list(configured)
    policy["recovery_algorithms"] = list(active_algorithms)
    policy["health_active_algorithms"] = list(active_algorithms)
    policy["active_algorithms"] = list(eligible_algorithms)
    policy["quarantined_algorithms"] = sorted(quarantined)
    policy["retry_avoided_algorithms"] = sorted(avoided)
    policy["recent_algorithm_stats"] = recent_policy["algorithm_stats"]
    policy["stagnation_priority_algorithms"] = [
        name for name in STAGNATION_PRIORITY_ALGORITHMS
        if name in configured
    ]
    if active_algorithms != configured:
        policy["reason"] = f"{health['state']}_forced_diversity"
    if quarantined:
        policy["reason"] += "+low_unique_yield_quarantine"
    if health["state"] == "stagnation_rebalance":
        policy["reason"] = "stagnation_full_fabric_residual_rebalance"
    parent, action, meta = _best(nodes), "fresh", {}
    if targeted_branch:
        expression, parent, meta = _targeted_residual_gate(
            branch=targeted_branch,
            fields=fields,
            feedback_nodes=feedback_nodes,
            rng=rng,
        )
        action = "conditional_gate"
    elif algorithm in {"structured_random", "grammar_enumerative"}:
        expression, meta = _grammar(family, fields, nodes, rng)
    elif algorithm == "qlib_alpha158_prior":
        expression, meta = _qlib_prior(family, fields, nodes, rng)
    elif algorithm == "qlib_joint_residual_distill":
        expression, meta = _qlib_joint_prior(
            family, fields, nodes, qlib_joint_candidates, rng,
            excluded_expression_hashes,
        )
    elif algorithm == "map_elites":
        expression, meta = _map_elites(family, fields, nodes, rng)
    elif algorithm == "mcts_puct":
        expression, meta = _mcts(family, fields, nodes, rng)
    elif algorithm == "evolutionary":
        action = "mutate" if parent else "fresh"
        expression = mutate_expression(str(parent["expression"]), fields=fields, rng=rng) if parent else random_expression_for_family(family, fields, rng)
    elif algorithm == "tpe_smac":
        action, (expression, meta) = "mutate", _tpe(family, fields, nodes, rng)
    elif algorithm == "surrogate_kernel":
        expression, meta = _surrogate(family, fields, nodes, rng)
    elif algorithm == "gbdt_residual_distill":
        # Meta-learning is task-wide; the distilled candidate remains generated
        # inside the currently assigned mechanism family.
        expression, meta = _gbdt(
            family,
            fields,
            allocation_nodes,
            rng,
            qlib_candidate_pool=qlib_candidate_pool,
            residual_oof_candidates=residual_oof_candidates,
        )
    elif algorithm == "q_learning":
        action, meta = _q_action(nodes, rng)
        expression = mutate_expression(str(parent["expression"]), fields=fields, rng=rng) if action == "mutate" and parent else random_expression_for_family(family, fields, rng)
    elif algorithm == "novelty_search":
        expression, meta = _novel(family, fields, nodes, rng)
    elif algorithm == "cegis_repair":
        expression, meta = _cegis(family, fields, feedback_nodes, nodes, rng)
    elif algorithm == "residual_oof_beam":
        expression, meta = _residual_oof_candidate(
            family,
            fields,
            nodes,
            residual_oof_candidates,
            rng,
            excluded_expression_hashes,
        )
        action = "oof_beam" if meta.get("residual_oof_ready") else "fresh"
    else:  # pragma: no cover - validated algorithm registry is exhaustive
        raise ValueError(f"第一层算法未实现: {algorithm}")
    group = ALGORITHM_GROUPS[algorithm]
    metadata = {"search_pool_schema": SEARCH_POOL_SCHEMA_VERSION, "search_algorithm": algorithm,
                "search_group": group, "search_group_target_share": SEARCH_GROUP_WEIGHTS[group],
                "search_action": action, "search_policy": policy,
                "search_epoch": health["search_epoch"],
                "search_health_state": health["state"],
                "search_reset_triggered": health["search_reset_triggered"],
                "search_reset_reasons": health["reset_reasons"],
                "search_stagnation_reasons": health["stagnation_reasons"],
                "search_recovery_kind": health["recovery_kind"],
                "elite_archive_node_ids": health["elite_node_ids"],
                "search_parent_node_id": parent.get("id") if parent else None,
                "search_observations": len(nodes), "target_family": family, **meta}
    hypothesis = (
        f"专属残差 Beam 围绕节点 {targeted_branch.get('seed_node_id')} 做条件门控"
        if targeted_branch
        else f"第一层 {algorithm} 在 {family} 机制中的训练安全候选"
    )
    return SearchProposal(expression=expression,
                          hypothesis=hypothesis,
                          metadata=metadata)
