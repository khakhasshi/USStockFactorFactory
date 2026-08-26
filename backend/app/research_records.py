"""Training-safe task research records backed by persisted search nodes.

Research records are intentionally not a second Factor table.  Every evaluated
Node is already the immutable source of truth for one candidate attempt.  This
module presents those nodes as a task-scoped, score-ordered research library
without granting formal factor, holdout, vault, paper, or live eligibility.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .models import Node
from .observability import redact_value


RESEARCH_RECORD_SCHEMA_VERSION = "factorfactory.task-research-record/v1"


def research_record_payload(
    node: Node,
    *,
    task_rank: int | None = None,
    formal_factor_id: int | None = None,
    research_factor_id: int | None = None,
) -> dict[str, Any]:
    public = dict(node.public_metrics or {})
    discovery = dict(public.get("discovery") or {})
    effective = dict(discovery.get("effective_metrics") or {})
    feedback = dict(node.feedback_summary or {})
    proposal = dict(node.proposal_meta or {})
    outcome = dict(feedback.get("outcome") or {})
    admission = dict(
        proposal.get("factor_admission")
        or feedback.get("factor_admission")
        or {}
    )
    search_policy = dict(proposal.get("search_policy") or {})
    search_health = dict(search_policy.get("search_health") or {})
    pre_eval_rejection = str(
        proposal.get("pre_evaluation_rejection") or ""
    )
    return {
        "schema_version": RESEARCH_RECORD_SCHEMA_VERSION,
        "id": node.id,
        "experiment_id": node.experiment_id,
        "task_name": node.task_name,
        "task_rank": task_rank,
        "record_tier": (
            "formal_factor" if formal_factor_id is not None else "research_candidate"
        ),
        "formal_factor_id": formal_factor_id,
        "formal_factor_admitted": formal_factor_id is not None,
        "research_factor_id": research_factor_id,
        "research_candidate_registered": research_factor_id is not None,
        "evaluation_protocol": node.evaluation_protocol,
        "miner_version_id": node.miner_version_id,
        "outer_step_no": node.outer_step_no,
        "seed": node.seed,
        "parent_id": node.parent_id,
        "operation": node.op,
        "source": node.source,
        "status": node.status,
        "expression": node.expression,
        "hypothesis": node.hypothesis,
        "learning_score": float(
            discovery.get("learning_score", node.public_score or 0.0) or 0.0
        ),
        "hard_gate_score": float(discovery.get("gate_score") or 0.0),
        "discovery_passed": bool(discovery.get("passed")),
        "direction": discovery.get("selected_direction"),
        "direction_policy": discovery.get("direction_policy"),
        "mechanism_family": str(
            proposal.get("declared_family")
            or proposal.get("mechanism_family")
            or proposal.get("target_family")
            or (feedback.get("diversity") or {}).get("declared_family")
            or "other"
        ),
        "metrics": {
            "icir": effective.get("icir", public.get("icir")),
            "portfolio_sharpe": effective.get("portfolio_sharpe"),
            "return_hac_t": effective.get("return_hac_t"),
            "sharpe_lcb": effective.get("sharpe_lcb"),
            "ann_return_lcb": effective.get("ann_return_lcb"),
            "era_consistency": effective.get("era_consistency"),
            "profitable_era_rate": effective.get("profitable_era_rate"),
            "monotonicity": effective.get("monotonicity"),
            "daily_turnover": effective.get("daily_turnover"),
            "worst_stress_sharpe": effective.get("worst_stress_sharpe"),
            "cost_cushion_multiple": effective.get("cost_cushion_multiple"),
        },
        "failure_reasons": list(
            discovery.get("failure_reasons")
            or feedback.get("failure_reasons")
            or []
        ),
        "improvement_targets": list(feedback.get("improvement_targets") or []),
        "reflection": str(proposal.get("reflection") or ""),
        "targeted_failures": list(proposal.get("targeted_failures") or []),
        "search_audit": {
            "algorithm": proposal.get("search_algorithm"),
            "group": proposal.get("search_group"),
            "search_epoch": proposal.get("search_epoch", 0),
            "health_state": proposal.get("search_health_state"),
            "reset_triggered": bool(proposal.get("search_reset_triggered")),
            "reset_reasons": list(proposal.get("search_reset_reasons") or []),
            "normalized_expression_hash": proposal.get(
                "normalized_expression_hash"
            ),
            "novelty_retries": int(
                proposal.get("campaign_novelty_retries") or 0
            ),
            "pre_evaluation_rejection": pre_eval_rejection or None,
            "evaluation_performed": bool(
                proposal.get("evaluation_performed", node.status != "rejected")
            ),
            "budget_charged": bool(
                proposal.get("budget_charged", node.status != "rejected")
            ),
            "recent_duplicate_rate": search_health.get(
                "recent_duplicate_rate"
            ),
            "unique_since_gate_record": search_health.get(
                "unique_since_gate_record"
            ),
            "elite_archive_node_ids": list(
                proposal.get("elite_archive_node_ids") or []
            ),
            "qlib": {
                "upstream_commit": proposal.get("qlib_upstream_commit"),
                "alpha158_feature": proposal.get("qlib_feature"),
                "alpha158_family": proposal.get("qlib_feature_family"),
                "joint_model_ready": proposal.get("joint_model_ready"),
                "joint_model_protocol": proposal.get("joint_model_protocol"),
                "distillation_kind": proposal.get("distillation_kind"),
                "distillation_components": list(
                    proposal.get("distillation_components") or []
                ),
                "candidate_pool_count": proposal.get(
                    "qlib_candidate_count"
                ),
                "holdout_vault_consumed": False,
            },
        },
        "factor_admission": redact_value(admission),
        "created_at": str(node.created_at),
        "interpretation_boundary": (
            "training research record only; not formal factor, holdout, vault, "
            "paper, live, or production approval"
        ),
    }


def task_research_summary(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record.get("task_name") or "unknown"), []).append(record)
    summaries = []
    for task_name, rows in sorted(grouped.items()):
        valid = [row for row in rows if row.get("status") == "ok"]
        pre_eval_rejected = [
            row for row in rows
            if (row.get("search_audit") or {}).get("pre_evaluation_rejection")
        ]
        sources = Counter(str(row.get("source") or "unknown") for row in rows)
        qlib_rows = [
            row for row in rows
            if any(
                value not in (None, "", [], False)
                for value in ((row.get("search_audit") or {}).get("qlib") or {}).values()
            )
        ]
        summaries.append({
            "task_name": task_name,
            "records": len(rows),
            "valid": len(valid),
            "passed": sum(bool(row.get("discovery_passed")) for row in valid),
            "formal_factors": sum(bool(row.get("formal_factor_admitted")) for row in rows),
            "pre_eval_rejected": len(pre_eval_rejected),
            "best_learning_score": round(
                max((float(row.get("learning_score") or 0.0) for row in valid), default=0.0),
                4,
            ),
            "best_hard_gate_score": round(
                max((float(row.get("hard_gate_score") or 0.0) for row in valid), default=0.0),
                4,
            ),
            "source_counts": dict(sources.most_common()),
            "qlib_candidates": len(qlib_rows),
            "qlib_joint_distillations": sum(
                bool(((row.get("search_audit") or {}).get("qlib") or {}).get("joint_model_ready"))
                for row in qlib_rows
            ),
        })
    return summaries
