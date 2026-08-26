"""Derive a conservative, user-facing factor evidence lifecycle.

The Factor table is also the task-scoped research notebook.  A row appearing
there therefore does not imply that HOLDOUT, Vault, frozen rating, and the
double-blind reviews have all passed.  This module keeps storage history intact
while exposing that distinction as an explicit API contract.
"""

from __future__ import annotations

from typing import Any


PROTOCOL = "factorfactory.factor-evidence-lifecycle/v1"


def _value(row: Any, name: str, default=None):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def factor_evidence_state(
    factor: Any,
    *,
    current_code_review: dict | None = None,
    current_rating_protocol: str | None = None,
) -> dict:
    eligibility = dict(_value(factor, "eligibility", {}) or {})
    validation = dict(_value(factor, "validation_metrics", {}) or {})
    ranking = dict(validation.get("ranking") or {})
    research_meta = dict(_value(factor, "research_meta", {}) or {})
    blind = dict(research_meta.get("double_blind_review") or {})
    stored_code = blind.get("code_review")
    code_review = current_code_review or (
        stored_code if isinstance(stored_code, dict) else {}
    )
    method_review = blind.get("method_review")
    sealed = blind.get("sealed_reviews") or blind.get("review_seal") or {}
    method_passed = bool(
        isinstance(method_review, dict) and method_review.get("passed")
    ) or bool(isinstance(sealed, dict) and sealed.get("method_passed"))
    code_passed = bool(code_review.get("passed")) or bool(
        isinstance(sealed, dict) and sealed.get("code_passed")
    )
    rating_current = bool(
        ranking.get("available")
        and ranking.get("score") is not None
        and (
            current_rating_protocol is None
            or ranking.get("rating_protocol_version") == current_rating_protocol
        )
    )
    hard_gates = {
        key: bool(eligibility.get(key))
        for key in (
            "research_pass",
            "holdout_pass",
            "vault_pass",
            "capacity_pass",
        )
    }
    formal = bool(
        all(hard_gates.values())
        and rating_current
        and method_passed
        and code_passed
    )
    audited = bool(validation and eligibility)
    if formal:
        tier = "formal_research_factor"
        label = "正式研究因子"
    elif audited:
        tier = "audited_candidate"
        label = "已审计候选（非正式）"
    elif str(_value(factor, "lifecycle_stage", "")) == "research_pass":
        tier = "training_candidate"
        label = "训练通过候选（非正式）"
    else:
        tier = "legacy_unreviewed"
        label = "历史未完整审计"
    blockers = []
    for key, passed in hard_gates.items():
        if not passed:
            blockers.append(key)
    if not rating_current:
        blockers.append("current_frozen_rating")
    if not method_passed:
        blockers.append("double_blind_method_review")
    if not code_passed:
        blockers.append("double_blind_code_review")
    return {
        "protocol": PROTOCOL,
        "tier": tier,
        "label": label,
        "formal_factor": formal,
        "training_only": tier == "training_candidate",
        "audited": audited,
        "hard_gates": hard_gates,
        "rating_current": rating_current,
        "blind_review": {
            "method_passed": method_passed,
            "code_passed": code_passed,
        },
        "promotion_blockers": blockers,
        "raw_status": str(_value(factor, "status", "") or ""),
        "effective_status": (
            str(_value(factor, "status", "") or "library-admitted")
            if formal
            else "research-candidate"
        ),
    }
