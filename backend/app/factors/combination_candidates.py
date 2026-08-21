"""Deterministic candidate-pool governance for multi-factor optimisation.

Selection is deliberately outcome-light: it uses only each candidate's
training-safe quality score, expression structure, declared mechanism and
training return-path sketch.  META_HOLDOUT and FACTOR_VAULT are never inputs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence

from .return_path import return_path_correlation
from .similarity import build_similarity_index, expression_similarity


def _candidate_key(row: dict[str, Any]) -> tuple:
    return (
        -int(row.get("source_priority") or 0),
        -float(row.get("quality_score") or 0.0),
        int(row.get("source_rank") or 10**9),
        str(row.get("component_key") or ""),
    )


def select_diverse_combination_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    structural_threshold: float = 0.64,
    return_path_correlation_cap: float = 0.80,
    max_per_mechanism: int = 3,
    max_candidates: int = 8,
) -> dict[str, Any]:
    """Select a small auditable pool while preserving mechanism coverage.

    The sequence is: exact key validation, structural-group representative,
    within-mechanism return-path de-correlation, then round-robin truncation.
    Current-protocol factors can use a larger ``source_priority`` than legacy
    leaderboard rows so an old report cannot displace a current factor solely
    because its score uses a different scale.
    """
    if not candidates:
        raise ValueError("候选因子不能为空")
    if not 0.3 <= structural_threshold <= 1.0:
        raise ValueError("structural_threshold 必须在 [0.3, 1.0]")
    if not 0.0 <= return_path_correlation_cap <= 1.0:
        raise ValueError("return_path_correlation_cap 必须在 [0, 1]")
    if max_per_mechanism < 1 or max_candidates < 2:
        raise ValueError("候选数量上限设置无效")

    rows = [dict(row) for row in candidates]
    keys = [str(row.get("component_key") or "").strip() for row in rows]
    if any(not key for key in keys):
        raise ValueError("每个候选必须有 component_key")
    if len(set(keys)) != len(keys):
        raise ValueError("component_key 不得重复")
    for row in rows:
        if not str(row.get("expression") or "").strip():
            raise ValueError(f"{row['component_key']} 缺少表达式")
        if not str(row.get("mechanism_family") or "").strip():
            raise ValueError(f"{row['component_key']} 缺少收益机制")

    indexed = [
        {
            "id": index + 1,
            "name": row["component_key"],
            "expression": row["expression"],
            # Similarity grouping only needs stable within-pool ordering.  Do
            # not compare raw V4 scores with old leaderboard score scales.
            "score": float(row.get("source_priority") or 0) * 1_000_000.0
            + float(row.get("quality_score") or 0.0),
        }
        for index, row in enumerate(rows)
    ]
    similarity = build_similarity_index(indexed, threshold=structural_threshold)
    row_by_index = {index + 1: row for index, row in enumerate(rows)}
    # Candidate pools are intentionally small.  The main factor-library index
    # uses LSH for speed and can miss a high-similarity pair when wrappers sit
    # at different AST depths (for example rank(x) versus winsor(x)).  Close
    # that approximation gap here with an exact all-pairs pass up to a safe
    # audit-sized limit, while retaining LSH behaviour for unusually large
    # pools.
    parent = {index: index for index in row_by_index}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for group in similarity["groups"]:
        member_ids = [int(member["id"]) for member in group["members"]]
        for member_id in member_ids[1:]:
            union(member_ids[0], member_id)
    exact_pair_comparisons = 0
    if len(rows) <= 256:
        for left in range(1, len(rows) + 1):
            for right in range(left + 1, len(rows) + 1):
                exact_pair_comparisons += 1
                if expression_similarity(
                    row_by_index[left]["expression"],
                    row_by_index[right]["expression"],
                ) >= structural_threshold:
                    union(left, right)

    structural_groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, row in row_by_index.items():
        structural_groups[find(index)].append(row)
    excluded: list[dict[str, Any]] = []
    structural_survivors: list[dict[str, Any]] = []
    for root, members in sorted(structural_groups.items()):
        members.sort(key=_candidate_key)
        winner = members[0]
        group_id = f"C-{root:04d}"
        winner["structural_group_id"] = group_id
        structural_survivors.append(winner)
        for row in members[1:]:
            excluded.append({
                "component_key": row["component_key"],
                "reason": "structural_duplicate",
                "kept_component_key": winner["component_key"],
                "structural_group_id": group_id,
            })

    by_mechanism: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in structural_survivors:
        by_mechanism[str(row["mechanism_family"])].append(row)

    mechanism_survivors: dict[str, list[dict[str, Any]]] = {}
    for mechanism, mechanism_rows in by_mechanism.items():
        selected: list[dict[str, Any]] = []
        for row in sorted(mechanism_rows, key=_candidate_key):
            if len(selected) >= max_per_mechanism:
                excluded.append({
                    "component_key": row["component_key"],
                    "reason": "mechanism_candidate_cap",
                    "mechanism_family": mechanism,
                })
                continue
            conflicts = []
            for incumbent in selected:
                correlation = return_path_correlation(
                    row.get("training_return_path_signature"),
                    incumbent.get("training_return_path_signature"),
                )
                if correlation is not None and abs(correlation) > return_path_correlation_cap:
                    conflicts.append((incumbent, correlation))
            if conflicts:
                incumbent, correlation = max(
                    conflicts,
                    key=lambda pair: abs(pair[1]),
                )
                excluded.append({
                    "component_key": row["component_key"],
                    "reason": "training_return_path_too_correlated",
                    "kept_component_key": incumbent["component_key"],
                    "absolute_correlation": round(abs(correlation), 6),
                })
                continue
            selected.append(row)
        mechanism_survivors[mechanism] = selected

    # Round-robin keeps at least one representative from every mechanism
    # before a deep source with many variants can consume the pool cap.
    mechanisms = sorted(
        mechanism_survivors,
        key=lambda mechanism: (
            _candidate_key(mechanism_survivors[mechanism][0])
            if mechanism_survivors[mechanism]
            else (0, 0, 0, mechanism)
        ),
    )
    selected = []
    depth = 0
    while len(selected) < max_candidates:
        added = False
        for mechanism in mechanisms:
            values = mechanism_survivors[mechanism]
            if depth < len(values):
                selected.append(values[depth])
                added = True
                if len(selected) >= max_candidates:
                    break
        if not added:
            break
        depth += 1

    selected_keys = {row["component_key"] for row in selected}
    for mechanism_rows in mechanism_survivors.values():
        for row in mechanism_rows:
            if row["component_key"] not in selected_keys:
                excluded.append({
                    "component_key": row["component_key"],
                    "reason": "global_candidate_cap",
                })

    if len(selected) < 2:
        raise ValueError("去重后不足两个候选，无法组合优化")
    return {
        "selected": selected,
        "excluded": sorted(excluded, key=lambda row: row["component_key"]),
        "selection_config": {
            "structural_threshold": structural_threshold,
            "return_path_correlation_cap": return_path_correlation_cap,
            "max_per_mechanism": max_per_mechanism,
            "max_candidates": max_candidates,
        },
        "input_candidates": len(rows),
        "selected_candidates": len(selected),
        "selected_mechanisms": sorted({
            str(row["mechanism_family"]) for row in selected
        }),
        "structural_index_stats": {
            **similarity["stats"],
            "exact_pair_comparisons": exact_pair_comparisons,
            "exact_pair_limit": 256,
            "final_groups": len(structural_groups),
        },
    }
