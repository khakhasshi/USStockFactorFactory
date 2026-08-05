"""Fast structural similarity index for DSL factor expressions.

The index is deliberately expression-based and does not read HOLDOUT or VAULT
returns.  It catches textual duplicates, sign/rank wrappers, shared fields,
operators, and nearby windows before a researcher spends a private evaluation
budget on economically redundant ideas.
"""

from __future__ import annotations

import ast
import hashlib
from collections import Counter, defaultdict
from dataclasses import dataclass

from ..dsl.engine import expression_profile, normalize_hash

_WRAPPERS = {"rank", "zscore", "winsor"}
_ROLLING = {"ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max", "ts_rank", "ts_delta", "ts_corr", "delay"}


def _strip_equivalent_wrappers(node: ast.expr) -> ast.expr:
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    while (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _WRAPPERS
        and len(node.args) == 1
    ):
        node = node.args[0]
    return node


def _window_bucket(value: int) -> str:
    if value <= 5:
        return "xs"
    if value <= 20:
        return "short"
    if value <= 60:
        return "medium"
    if value <= 120:
        return "long"
    return "xlong"


def _family(fields: set[str], operators: set[str]) -> str:
    if fields & {"pe_ttm", "pb", "ps_ttm", "dv_ttm"}:
        return "valuation"
    if fields & {"net_mf_amount", "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"}:
        return "capital_flow"
    if "ts_std" in operators or "high" in fields and "low" in fields:
        return "volatility"
    if "amount" in fields or "vol" in fields or fields & {"turnover_rate", "volume_ratio"}:
        return "liquidity_volume"
    if "ts_corr" in operators:
        return "relationship"
    if operators & {"ts_delta", "delay", "ts_rank", "ts_mean"} and fields & {"open", "close"}:
        return "price_trend_reversal"
    if len(fields) > 1 or len(operators) > 2:
        return "composite"
    return "other"


def _features(expression: str) -> Counter[str]:
    root = _strip_equivalent_wrappers(ast.parse(expression, mode="eval").body)
    features: Counter[str] = Counter()
    for node in ast.walk(root):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            features[f"op:{fn}"] += 3 if fn in _ROLLING else 1
            for arg in node.args[1:]:
                if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                    features[f"window:{_window_bucket(arg.value)}"] += 2
                    features[f"window_exact:{arg.value}"] += 1
        elif isinstance(node, ast.Name):
            features[f"name:{node.id}"] += 4
        elif isinstance(node, ast.BinOp):
            features[f"bin:{type(node.op).__name__}"] += 2
    # Function identifiers are ast.Name nodes too; avoid double counting them
    # as market-data fields.
    for token in list(features):
        if token.startswith("op:"):
            features.pop(f"name:{token[3:]}", None)
    structure = ast.dump(root, annotate_fields=False, include_attributes=False)
    structure = structure.replace("USub()", "")
    features[f"shape:{hashlib.sha1(structure.encode()).hexdigest()[:12]}"] += 5
    return features


def weighted_jaccard(left: Counter[str], right: Counter[str]) -> float:
    keys = left.keys() | right.keys()
    numerator = sum(min(left[key], right[key]) for key in keys)
    denominator = sum(max(left[key], right[key]) for key in keys)
    return numerator / denominator if denominator else 1.0


def _simhash(features: Counter[str]) -> int:
    vector = [0] * 64
    for token, weight in features.items():
        digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            vector[bit] += weight if digest & (1 << bit) else -weight
    value = 0
    for bit, score in enumerate(vector):
        if score >= 0:
            value |= 1 << bit
    return value


def expression_fingerprint(expression: str) -> dict:
    profile = expression_profile(expression)
    features = _features(expression)
    fields = set(profile["fields"])
    operators = set(profile["operators"])
    return {
        "expr_hash": normalize_hash(expression),
        "simhash64": f"{_simhash(features):016x}",
        "family": _family(fields, operators),
        "fields": sorted(fields),
        "operators": sorted(operators),
        "windows": profile["windows"],
        "required_history": profile["required_history"],
        "complexity": profile["complexity"],
    }


@dataclass
class _Indexed:
    id: int
    name: str
    expression: str
    score: float
    features: Counter[str]
    simhash: int
    family: str


class _UnionFind:
    def __init__(self, ids: list[int]) -> None:
        self.parent = {value: value for value in ids}

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            nxt = self.parent[value]
            self.parent[value] = root
            value = nxt
        return root

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def build_similarity_index(items: list[dict], threshold: float = 0.64) -> dict:
    if not 0.3 <= threshold <= 1.0:
        raise ValueError("similarity threshold 必须在 [0.3, 1.0]")
    indexed: list[_Indexed] = []
    for item in items:
        try:
            features = _features(item["expression"])
            profile = expression_profile(item["expression"])
        except (SyntaxError, ValueError):
            continue
        indexed.append(_Indexed(
            id=int(item["id"]),
            name=str(item.get("name") or item["id"]),
            expression=item["expression"],
            score=float(item.get("score") or 0.0),
            features=features,
            simhash=_simhash(features),
            family=_family(set(profile["fields"]), set(profile["operators"])),
        ))

    uf = _UnionFind([item.id for item in indexed])
    by_id = {item.id: item for item in indexed}
    candidate_pairs: set[tuple[int, int]] = set()
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for item in indexed:
        for band in range(4):
            buckets[(band, (item.simhash >> (band * 16)) & 0xFFFF)].append(item.id)
    for ids in buckets.values():
        if len(ids) < 2:
            continue
        # Large generic buckets are narrowed by family below.
        for i, left in enumerate(ids):
            for right in ids[i + 1:]:
                if by_id[left].family == by_id[right].family or len(ids) <= 40:
                    candidate_pairs.add((min(left, right), max(left, right)))

    similarities: dict[tuple[int, int], float] = {}
    for left, right in candidate_pairs:
        similarity = weighted_jaccard(by_id[left].features, by_id[right].features)
        similarities[(left, right)] = similarity
        if similarity >= threshold:
            uf.union(left, right)

    clusters: dict[int, list[_Indexed]] = defaultdict(list)
    for item in indexed:
        clusters[uf.find(item.id)].append(item)

    groups = []
    factor_to_group: dict[int, str] = {}
    for members in clusters.values():
        members.sort(key=lambda row: (-row.score, row.id))
        stable = min(normalize_hash(row.expression) for row in members)
        group_id = f"G-{stable[:8]}"
        for member in members:
            factor_to_group[member.id] = group_id
        pair_scores = []
        for i, left in enumerate(members):
            for right in members[i + 1:]:
                key = (min(left.id, right.id), max(left.id, right.id))
                value = similarities.get(key)
                if value is None:
                    value = weighted_jaccard(left.features, right.features)
                pair_scores.append(value)
        groups.append({
            "id": group_id,
            "family": members[0].family,
            "size": len(members),
            "representative": {
                "id": members[0].id,
                "name": members[0].name,
                "expression": members[0].expression,
                "score": members[0].score,
            },
            "mean_similarity": round(sum(pair_scores) / len(pair_scores), 4) if pair_scores else 1.0,
            "factor_ids": [row.id for row in members],
            "members": [
                {
                    "id": row.id,
                    "name": row.name,
                    "expression": row.expression,
                    "score": row.score,
                    "similarity_to_representative": round(
                        weighted_jaccard(members[0].features, row.features), 4
                    ),
                }
                for row in members
            ],
        })
    groups.sort(key=lambda group: (-group["size"], group["family"], group["id"]))
    duplicate_groups = sum(group["size"] > 1 for group in groups)
    redundancy = (
        1.0 - len(groups) / len(indexed)
        if indexed else 0.0
    )
    return {
        "groups": groups,
        "factor_to_group": factor_to_group,
        "stats": {
            "factors": len(indexed),
            "groups": len(groups),
            "duplicate_groups": duplicate_groups,
            "redundancy_ratio": round(redundancy, 4),
            "threshold": threshold,
            "candidate_pairs": len(candidate_pairs),
            "algorithm": "simhash_lsh_plus_weighted_jaccard_v1",
        },
    }


def nearest_factors(items: list[dict], factor_id: int, limit: int = 20) -> list[dict]:
    prepared = []
    target = next((item for item in items if int(item["id"]) == int(factor_id)), None)
    if target is None:
        return []
    target_features = _features(target["expression"])
    for item in items:
        if int(item["id"]) == int(factor_id):
            continue
        try:
            score = weighted_jaccard(target_features, _features(item["expression"]))
        except (SyntaxError, ValueError):
            continue
        prepared.append({
            "id": int(item["id"]),
            "name": item.get("name"),
            "expression": item["expression"],
            "score": item.get("score"),
            "similarity": round(score, 4),
        })
    prepared.sort(key=lambda row: (-row["similarity"], -float(row.get("score") or 0), row["id"]))
    return prepared[:max(1, min(limit, 100))]
