"""Derive a conservative, user-facing factor evidence lifecycle.

The Factor table is also the task-scoped research notebook.  A row appearing
there therefore does not imply that HOLDOUT, Vault, frozen rating, and the
double-blind reviews have all passed.  This module keeps storage history intact
while exposing that distinction as an explicit API contract.
"""

from __future__ import annotations

import hashlib
from typing import Any

from .research_overfit import PROTOCOL as OVERFIT_PROTOCOL


PROTOCOL = "factorfactory.factor-evidence-lifecycle/v2"


def _value(row: Any, name: str, default=None):
    if isinstance(row, dict):
        return row.get(name, default)
    return getattr(row, name, default)


def _frozen_input_pass(provenance: dict, expression: str) -> bool:
    panel = dict(provenance.get("panel") or {})
    code = dict(panel.get("code") or {})
    hashes = [panel.get("data_sha256"), panel.get("snapshot_id"),
              panel.get("market_calendar_sha256"), code.get("code_sha256"),
              provenance.get("expression_sha256"), provenance.get("config_sha256"),
              provenance.get("run_fingerprint")]
    hashes_complete = all(isinstance(value, str) and len(value) == 64 for value in hashes)
    expression_matches = bool(expression and provenance.get("expression_sha256") ==
                              hashlib.sha256(expression.encode()).hexdigest())
    return bool(provenance.get("passed", True)
                and provenance.get("immutable_inputs_available")
                and panel.get("immutable_inputs_available")
                and panel.get("status") == "FROZEN"
                and code.get("persisted")
                and panel.get("files") and code.get("source_files")
                and provenance.get("requested_start") and provenance.get("requested_end")
                and provenance.get("actual_start") and provenance.get("actual_end")
                and int(provenance.get("actual_sessions") or 0) > 0
                and hashes_complete and expression_matches)


def _event_evidence_pass(event: dict, provenance: dict, expression: str) -> bool:
    if not (event.get("passed") and event.get("status") == "PASS"
            and event.get("protocol") == "factorfactory.event-promotion-gate/v1"):
        return False
    windows = dict(event.get("windows") or {})
    contract = dict(event.get("contract") or {})
    reference_panel = dict(provenance.get("panel") or {})
    if (set(windows) != {"holdout", "vault", "rating"}
            or contract.get("direction_frozen") not in {-1, 1}
            or not contract.get("universe_n") or not contract.get("rebalance_sessions")):
        return False
    for window in windows.values():
        window = dict(window or {})
        inputs = dict(window.get("input_provenance") or {})
        panel = dict(inputs.get("panel") or {})
        config = dict(window.get("config") or {})
        if not (window.get("passed") and window.get("status") == "PASS"
                and (window.get("integrity") or {}).get("all_pass")
                and _frozen_input_pass(inputs, expression)
                and panel.get("data_sha256") == reference_panel.get("data_sha256")
                and panel.get("snapshot_id") == reference_panel.get("snapshot_id")
                and (panel.get("code") or {}).get("code_sha256") == (reference_panel.get("code") or {}).get("code_sha256")
                and config.get("direction") == contract["direction_frozen"]
                and config.get("universe_n") == contract["universe_n"]
                and config.get("rebalance_every") == contract["rebalance_sessions"]):
            return False
    return True


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
    # A protocol string or an old successful vector screen is not enough to
    # promote a factor.  Evidence must exist, not merely an eligibility flag.
    event_audit = dict(validation.get("event_audit") or {})
    provenance = dict(validation.get("audit_provenance") or {})
    overfit = dict(validation.get("overfit_governance") or {})
    hard_gates.update({
        "event_pass": _event_evidence_pass(event_audit, provenance, str(_value(factor, "expression", "") or "")),
        "frozen_input_pass": _frozen_input_pass(provenance, str(_value(factor, "expression", "") or "")),
        "overfit_evidence_complete": bool(
            overfit.get("available")
            and overfit.get("protocol") == OVERFIT_PROTOCOL
            and overfit.get("status") in {"PASS", "FAIL"}
            and not overfit.get("missing_evidence_trials", 0)
        ),
        "overfit_pass": bool(overfit.get("passed") and overfit.get("status") == "PASS"),
        "dsr_pass": bool(overfit.get("dsr_pass")),
        "pbo_pass": bool(overfit.get("pbo_pass")),
    })
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
        "event_audit": {"passed": hard_gates["event_pass"],
                        "status": event_audit.get("status", "NOT_AUDITED")},
        "frozen_input": {"passed": hard_gates["frozen_input_pass"],
                         "status": "FROZEN" if hard_gates["frozen_input_pass"] else "INCOMPLETE"},
        "overfit_governance": {
            "status": overfit.get("status", "INSUFFICIENT_DATA"),
            "available": hard_gates["overfit_evidence_complete"],
            "dsr_pass": hard_gates["dsr_pass"],
            "pbo_pass": hard_gates["pbo_pass"],
            "missing_evidence_trials": overfit.get("missing_evidence_trials"),
            "reasons": list(overfit.get("reasons") or []),
        },
        "promotion_blockers": blockers,
        "raw_status": str(_value(factor, "status", "") or ""),
        "effective_status": (
            str(_value(factor, "status", "") or "library-admitted")
            if formal
            else "research-candidate"
        ),
    }
