"""Training-only discovery evidence, immutable combination pool and score views.

No function in this module confers formal factor/trading eligibility. Missing
evidence is unknown, not zero skill. Historical scores remain unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4

from .config import PROJECT_ROOT

PROTOCOL = "factorfactory.discovery-evidence/v1"
ROOT = PROJECT_ROOT / "var" / "reports" / "discovery-v4"


def bind_proposal_evidence(expression: str, metadata: dict, seed_expression: str | None = None) -> dict:
    """A changed DSL cannot inherit measured evidence from its seed."""
    from .dsl.engine import normalize_hash
    result = dict(metadata)
    same = lambda other: bool(other) and normalize_hash(expression, direction_invariant=True) == normalize_hash(other, direction_invariant=True)
    evidence = result.get("combination_evidence") or {}
    if evidence and not same(evidence.get("expression")):
        result["seed_combination_evidence"] = result.pop("combination_evidence")
        result["exact_oof_completed_before_full_evaluation"] = False
        result["incremental_evidence_status"] = "expression_changed_requires_remeasurement"
        for key in ("residual_rank_ic", "incremental_oof_ic", "return_source_independence", "residual_stability", "residual_beam_score"):
            result.pop(key, None)
    if seed_expression and not same(seed_expression) and result.get("proposal_authority") in {"researcher_llm", "mechanism_scientist_llm"}:
        result["seed_executed_algorithm"] = result.get("executed_algorithm")
        result["executed_algorithm"] = "llm_seed_revision"
    return result


def persist_residual_artifact(artifact: dict, experiment_id: int, task_name: str) -> dict:
    payload = {**artifact, "experiment_id": int(experiment_id), "task_name": task_name,
               "evidence_protocol": PROTOCOL, "formal_eligible": False}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    digest = hashlib.sha256(raw).hexdigest()
    directory = ROOT / f"exp-{int(experiment_id)}" / hashlib.sha256(task_name.encode()).hexdigest()[:16]
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.json"
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        if path.read_bytes() != raw:
            raise ValueError("immutable discovery artifact collision")
    pointer = directory / f"latest-{uuid4().hex}.tmp"
    pointer.write_text(json.dumps({"sha256": digest}), encoding="utf-8")
    os.replace(pointer, directory / "latest.json")
    return {**payload, "artifact_sha256": digest, "artifact_path": str(path)}


def combination_pool(experiment_id: int) -> list[dict]:
    """Read verified, task-scoped pool; corruption is displayed, never promoted."""
    result = []
    for pointer in sorted((ROOT / f"exp-{int(experiment_id)}").glob("*/latest.json")):
        try:
            digest = json.loads(pointer.read_text())["sha256"]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid digest")
            raw = (pointer.parent / f"{digest}.json").read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("artifact hash mismatch")
            artifact = json.loads(raw)
            result.append({"task_name": artifact["task_name"], "artifact_sha256": digest,
                "created_at": artifact["created_at"], "panel_identity": artifact.get("panel_identity"),
                "protocol": artifact["protocol"], "valid_oof_rows": artifact.get("valid_oof_rows"),
                "paths": artifact.get("combination_paths", []), "formal_eligible": False,
                "status": "training_predictive_pool_event_confirmation_required"})
        except (OSError, ValueError, KeyError) as exc:
            result.append({"status": "INVALID_EVIDENCE", "error": str(exc)[:180], "formal_eligible": False})
    return result


def _finite(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def score_channels(discovery: dict, proposal: dict, status: str) -> dict:
    metrics = discovery.get("effective_metrics") or {}
    required = ["portfolio_sharpe", "sharpe_lcb", "worst_stress_sharpe",
                "profitable_era_rate", "cost_cushion_multiple", "icir"]
    available = status == "ok" and all(_finite(metrics.get(k)) is not None for k in required)
    clip = lambda x: max(0., min(1., x))
    # One profitability family, not multiple votes for the same mean return.
    quality = None
    if available:
        quality = 100 * (0.45 * clip(float(metrics["sharpe_lcb"]) / 1.5)
            + 0.25 * clip(float(metrics["profitable_era_rate"]))
            + 0.20 * clip(float(metrics["cost_cushion_multiple"]) / 3)
            + 0.10 * clip(float(metrics["icir"]) / 2))
        if float(metrics["portfolio_sharpe"]) <= 0 or float(metrics["worst_stress_sharpe"]) < 0:
            quality = min(quality, 24.9)
    evidence = proposal.get("combination_evidence") or {}
    increment = _finite(evidence.get("incremental_oof_ic"))
    is_joint = bool(evidence.get("evidence_protocol") == "factorfactory.residual-oof-beam/v4-joint-refit")
    return {"protocol": PROTOCOL, "scope": "training_only_uncalibrated_not_formal_rating",
        "exploration": discovery.get("learning_score", discovery.get("score")),
        "standalone_quality": round(quality, 3) if quality is not None else None,
        "standalone_passed": bool(discovery.get("passed")),
        "incremental_quality": round(100 * clip(float(evidence.get("score") or 0)), 3) if is_joint else None,
        "incremental_rank_ic": increment if is_joint else None,
        "incremental_passed": bool(is_joint and evidence.get("eligible")),
        "net_increment_confirmed": False,
        "failure_class": ("data_or_execution_unavailable" if not available else
                          "training_passed_pending_audit" if discovery.get("passed") else
                          "economic_or_statistical_gate_failed"),
        "calibration": "not_yet_prospectively_validated"}


def algorithm_diagnostics(records: list[dict]) -> dict:
    rows = {}
    for record in records:
        audit = record.get("search_audit") or {}
        requested = audit.get("requested_algorithm") or audit.get("algorithm") or record.get("source") or "unknown"
        executed = audit.get("executed_algorithm") or "legacy_unverified"
        key = (record.get("task_name"), requested, executed)
        row = rows.setdefault(key, {"task_name": key[0], "requested": requested, "executed": executed,
            "attempts": 0, "evaluated": 0, "fallbacks": 0, "training_passed": 0,
            "known_compute_seconds": 0., "timed_evaluations": 0})
        row["attempts"] += 1
        row["fallbacks"] += bool(audit.get("fallback_reason"))
        row["evaluated"] += bool(audit.get("evaluation_performed"))
        row["training_passed"] += bool(record.get("discovery_passed"))
        seconds = _finite(audit.get("evaluation_seconds"))
        if seconds is not None and audit.get("evaluation_performed"):
            row["known_compute_seconds"] += seconds
            row["timed_evaluations"] += 1
    return {"protocol": PROTOCOL, "comparison": "observational_not_randomized_ablation",
        "llm_cost": None, "llm_cost_reason": "provider_billing_not_available_per_candidate",
        "rows": sorted(rows.values(), key=lambda row: (row["task_name"] or "", -row["attempts"]))}
