"""Training-safe governance for economically distinct factor return sources.

The full-library P0 audit demonstrated that mechanism labels and AST structure
are not sufficient proxies for independent alpha sources.  This module keeps
the online decision boundary narrower than that diagnostic audit: it consumes
only INNER_PUBLIC + META_TRAIN return-path sketches already present in V4
feedback envelopes and factor metadata.  META_HOLDOUT, FACTOR_VAULT and
full-window leaderboard returns are never accepted as inputs.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable

from .return_path import return_path_correlation


RETURN_SOURCE_GOVERNANCE_PROTOCOL = "training_return_source_governance_v2"
DEFAULT_RETURN_PATH_CORRELATION_THRESHOLD = 0.85


def resolve_return_source_governance(raw: Any) -> dict[str, Any]:
    """Validate and normalize the opt-in experiment configuration."""
    value = raw or {}
    if isinstance(value, str):
        value = {"protocol": value}
    if not isinstance(value, dict):
        raise ValueError("return_source_governance 必须是对象或协议名称")
    protocol = str(value.get("protocol") or "disabled").strip()
    enabled = protocol == RETURN_SOURCE_GOVERNANCE_PROTOCOL
    if protocol not in {"", "disabled", RETURN_SOURCE_GOVERNANCE_PROTOCOL}:
        raise ValueError(f"不支持的收益来源治理协议: {protocol}")
    try:
        score_weight = float(value.get("meta_score_weight", 0.30))
        correlation_threshold = float(
            value.get(
                "correlation_threshold",
                DEFAULT_RETURN_PATH_CORRELATION_THRESHOLD,
            )
        )
        required_sources = int(value.get("required_sources", 4))
    except (TypeError, ValueError) as exc:
        raise ValueError("收益来源治理参数格式错误") from exc
    if not 0.0 <= score_weight <= 0.5:
        raise ValueError("meta_score_weight 必须在 [0, 0.5]")
    if not 0.5 <= correlation_threshold < 1.0:
        raise ValueError("correlation_threshold 必须在 [0.5, 1)")
    if not 2 <= required_sources <= 12:
        raise ValueError("required_sources 必须在 [2, 12]")
    return {
        "enabled": enabled,
        "protocol": protocol if enabled else "disabled",
        "meta_score_weight": score_weight if enabled else 0.0,
        "correlation_threshold": correlation_threshold,
        "required_sources": required_sources,
        "cross_experiment_admission": (
            bool(value.get("cross_experiment_admission", True))
            if enabled
            else False
        ),
    }


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _signature(item: dict[str, Any]) -> dict[str, Any]:
    direct = item.get("training_return_path_signature")
    if isinstance(direct, dict):
        return direct
    public = item.get("public_metrics") or {}
    nested = public.get("training_return_path_signature")
    if isinstance(nested, dict):
        return nested
    diversity = item.get("diversity") or {}
    nested = diversity.get("return_path_signature")
    return nested if isinstance(nested, dict) else {}


def _quality(item: dict[str, Any]) -> float:
    outcome = item.get("outcome") or {}
    public = item.get("public_metrics") or {}
    discovery = public.get("discovery") or {}
    for value in (
        item.get("quality_score"),
        outcome.get("learning_score"),
        outcome.get("score"),
        discovery.get("learning_score"),
        public.get("score"),
        item.get("public_score"),
    ):
        if value is not None:
            return _finite(value)
    return 0.0


def _identity(item: dict[str, Any], index: int) -> str:
    for key in ("component_key", "factor_id", "node_id", "id"):
        value = item.get(key)
        if value not in (None, ""):
            return str(value)
    return f"row:{index:06d}"


def _scope(item: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return the smallest scope in which path correlations are comparable."""
    public = item.get("public_metrics") or {}
    research = item.get("research_meta") or {}
    market = str(item.get("market") or research.get("market") or "unknown")
    mode = str(
        item.get("portfolio_mode")
        or research.get("portfolio_mode")
        or "unknown"
    )
    horizon = item.get("horizon", public.get("horizon"))
    # Online envelopes have a stable task name even when an older factor lacks
    # universe_n metadata.  Prefer an explicit task signature when available;
    # otherwise task_name prevents cross-horizon/cross-universe comparisons.
    task_signature = (
        item.get("task_signature")
        or research.get("task_signature")
        or item.get("task_name")
        or "unknown"
    )
    return market, mode, str(horizon or "unknown"), str(task_signature)


def cluster_training_return_sources(
    items: Iterable[dict[str, Any]],
    *,
    correlation_threshold: float = DEFAULT_RETURN_PATH_CORRELATION_THRESHOLD,
    include_assignments: bool = False,
) -> dict[str, Any]:
    """Cluster comparable training return paths around quality representatives.

    Clusters are representative-based rather than transitive connected
    components.  This avoids correlation chaining, where A resembles B and B
    resembles C even though A and C are materially different sources.  Only
    positive correlation is treated as duplication; a negatively correlated
    path is a diversifier after the signal direction has been frozen.
    """
    if not 0.0 < float(correlation_threshold) < 1.0:
        raise ValueError("correlation_threshold 必须在 (0, 1) 内")
    rows = [dict(item) for item in items]
    available: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        signature = _signature(row)
        vector = list(signature.get("vector") or [])
        if not signature.get("available") or len(vector) < 4:
            continue
        available.append({
            "identity": _identity(row, index),
            "quality": _quality(row),
            "scope": _scope(row),
            "signature": signature,
            "mechanism_family": str(
                row.get("mechanism_family")
                or (row.get("diversity") or {}).get("mechanism_family")
                or "other"
            ),
        })

    by_scope: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in available:
        by_scope[row["scope"]].append(row)

    assignments: list[dict[str, Any]] = []
    cluster_sizes: list[int] = []
    representative_qualities: list[float] = []
    scope_summaries: dict[str, dict[str, Any]] = {}
    cluster_number = 0
    for scope, scope_rows in sorted(by_scope.items()):
        ordered = sorted(
            scope_rows,
            key=lambda row: (-row["quality"], row["identity"]),
        )
        representatives: list[dict[str, Any]] = []
        members: list[list[dict[str, Any]]] = []
        for row in ordered:
            best_index: int | None = None
            best_correlation = float("-inf")
            for index, representative in enumerate(representatives):
                correlation = return_path_correlation(
                    row["signature"], representative["signature"]
                )
                if correlation is not None and correlation > best_correlation:
                    best_index = index
                    best_correlation = correlation
            if best_index is None or best_correlation < correlation_threshold:
                representatives.append(row)
                members.append([])
                best_index = len(representatives) - 1
                best_correlation = 1.0
            members[best_index].append(row)

        scope_key = "|".join(scope)
        scope_summaries[scope_key] = {
            "available": len(scope_rows),
            "clusters": len(representatives),
            "redundancy_rate": round(
                1.0 - len(representatives) / max(1, len(scope_rows)), 6
            ),
        }
        for representative, cluster_members in zip(representatives, members):
            cluster_number += 1
            cluster_id = f"TRS-{cluster_number:04d}"
            cluster_sizes.append(len(cluster_members))
            representative_qualities.append(representative["quality"])
            if include_assignments:
                for member in cluster_members:
                    correlation = return_path_correlation(
                        member["signature"], representative["signature"]
                    )
                    assignments.append({
                        "identity": member["identity"],
                        "cluster_id": cluster_id,
                        "representative_identity": representative["identity"],
                        "is_representative": member is representative,
                        "correlation_to_representative": round(
                            float(correlation if correlation is not None else 0.0),
                            6,
                        ),
                        "scope": scope_key,
                        "mechanism_family": member["mechanism_family"],
                    })

    available_count = len(available)
    cluster_count = len(cluster_sizes)
    hhi = (
        sum((size / available_count) ** 2 for size in cluster_sizes)
        if available_count else 0.0
    )
    summary = {
        "protocol": RETURN_SOURCE_GOVERNANCE_PROTOCOL,
        "correlation_threshold": float(correlation_threshold),
        "items": len(rows),
        "available_signatures": available_count,
        "signature_coverage": round(available_count / max(1, len(rows)), 6),
        "comparable_scopes": len(by_scope),
        "return_source_clusters": cluster_count,
        "return_source_redundancy_rate": round(
            1.0 - cluster_count / max(1, available_count), 6
        ),
        "return_source_cluster_hhi": round(hhi, 6),
        "effective_return_sources": round(1.0 / hhi, 4) if hhi > 0 else 0.0,
        "largest_return_source_cluster": max(cluster_sizes, default=0),
        "representative_qualities": [
            round(value, 6) for value in sorted(representative_qualities, reverse=True)
        ],
        "scope_summaries": scope_summaries,
    }
    if include_assignments:
        summary["assignments"] = sorted(
            assignments,
            key=lambda row: (row["cluster_id"], not row["is_representative"], row["identity"]),
        )
    return summary


def return_source_quality(
    snapshot: dict[str, Any],
    *,
    required_sources: int = 4,
) -> float:
    """Mean quality of the best distinct sources, padded for missing sources."""
    required = max(1, int(required_sources))
    qualities = [
        _finite(value)
        for value in (snapshot.get("representative_qualities") or [])[:required]
    ]
    padded = (qualities + [0.0] * required)[:required]
    return round(sum(padded) / required, 6)


def governance_reason_counts(snapshot: dict[str, Any]) -> dict[str, int]:
    """Small diagnostic helper for reports and tests."""
    counts = Counter(
        str(row.get("mechanism_family") or "other")
        for row in (snapshot.get("assignments") or [])
        if not row.get("is_representative")
    )
    return dict(counts.most_common())
