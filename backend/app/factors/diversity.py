"""Mechanism-family scheduling and diversity diagnostics for factor mining."""

from __future__ import annotations

import ast
import math
import random
from collections import Counter
from typing import Any, Iterable

from ..dsl.engine import expression_profile, normalize_hash
from .return_source_governance import (
    cluster_training_return_sources,
    return_source_quality,
)


COMMON_MECHANISMS = (
    "momentum",
    "reversal",
    "volatility",
    "liquidity",
    "volume_price_interaction",
    "gap_intraday",
    "price_relationship",
)
ASHARE_EXTRA_MECHANISMS = (
    "valuation",
    "size",
    "capital_flow",
)

MECHANISM_LABELS = {
    "momentum": "趋势延续/动量",
    "reversal": "过度反应/均值回归",
    "volatility": "波动与尾部风险溢价",
    "liquidity": "流动性与交易拥挤溢价",
    "volume_price_interaction": "量价确认与价格冲击",
    "gap_intraday": "隔夜缺口与日内定价",
    "price_relationship": "价格序列关系与状态变化",
    "valuation": "估值补偿",
    "size": "规模与容量补偿",
    "capital_flow": "资金流压力与反转/延续",
    "other": "未分类",
}

_PRICE = {"open", "high", "low", "close"}
_LIQUIDITY = {"amount", "vol", "turnover_rate", "volume_ratio"}
_VALUATION = {"pe_ttm", "pb", "ps_ttm", "dv_ttm"}
_SIZE = {"total_mv", "circ_mv", "float_share"}
_FLOW = {
    "net_mf_amount", "buy_lg_amount", "sell_lg_amount",
    "buy_elg_amount", "sell_elg_amount",
}


def mechanisms_for_market(market: str) -> tuple[str, ...]:
    if market == "us":
        return COMMON_MECHANISMS
    if market == "ashare":
        return COMMON_MECHANISMS + ASHARE_EXTRA_MECHANISMS
    raise ValueError("market 必须是 ashare 或 us")


def infer_mechanism(expression: str, hypothesis: str = "") -> str:
    try:
        profile = expression_profile(expression)
    except (SyntaxError, ValueError):
        return "other"
    fields = set(profile["fields"])
    operators = set(profile["operators"])
    text = str(hypothesis or "").lower()
    if fields & _VALUATION:
        return "valuation"
    if fields & _FLOW:
        return "capital_flow"
    if fields & _SIZE:
        return "size"
    if "open" in fields and fields & {"close", "high", "low"}:
        return "gap_intraday"
    if fields & _PRICE and fields & _LIQUIDITY:
        return "volume_price_interaction"
    if "ts_corr" in operators:
        return "price_relationship"
    if "ts_std" in operators or {"high", "low"}.issubset(fields):
        return "volatility"
    if fields & _LIQUIDITY:
        return "liquidity"
    if fields & _PRICE:
        reversal_tokens = ("reversal", "mean reversion", "反转", "反轉", "均值回归", "均值回歸")
        momentum_tokens = ("momentum", "trend", "动量", "動量", "趋势", "趨勢", "延续", "延續")
        if any(token in text for token in reversal_tokens):
            return "reversal"
        if any(token in text for token in momentum_tokens):
            return "momentum"
        if isinstance(ast.parse(expression, mode="eval").body, ast.UnaryOp):
            return "reversal"
        return "momentum"
    return "other"


def mechanism_compatible(expression: str, family: str) -> bool:
    try:
        profile = expression_profile(expression)
    except (SyntaxError, ValueError):
        return False
    fields = set(profile["fields"])
    operators = set(profile["operators"])
    rules = {
        "momentum": bool(fields & _PRICE and operators & {"ts_delta", "delay", "ts_mean", "ts_rank"}),
        "reversal": bool(fields & _PRICE and operators & {"ts_delta", "delay", "ts_mean", "ts_rank"}),
        "volatility": bool("ts_std" in operators or {"high", "low"}.issubset(fields)),
        "liquidity": bool(fields & _LIQUIDITY),
        "volume_price_interaction": bool(fields & _PRICE and fields & _LIQUIDITY),
        "gap_intraday": bool("open" in fields and fields & {"close", "high", "low"}),
        "price_relationship": "ts_corr" in operators,
        "valuation": bool(fields & _VALUATION),
        "size": bool(fields & _SIZE),
        "capital_flow": bool(fields & _FLOW),
    }
    return bool(rules.get(family, False))


def mechanism_from_item(item: dict) -> str:
    proposal = item.get("proposal") or item.get("proposal_meta") or {}
    declared = str(
        proposal.get("declared_family")
        or proposal.get("mechanism_family")
        or ""
    )
    if declared in MECHANISM_LABELS:
        return declared
    return infer_mechanism(
        str(item.get("expression") or ""),
        str(item.get("hypothesis") or ""),
    )


def diversity_snapshot(items: Iterable[dict], market: str) -> dict[str, Any]:
    rows = list(items)
    families = mechanisms_for_market(market)
    attempts = Counter()
    passes = Counter()
    best: dict[str, float] = {}
    hashes: list[str] = []
    for item in rows:
        family = mechanism_from_item(item)
        attempts[family] += 1
        outcome = item.get("outcome") or {}
        if bool(outcome.get("passed")):
            passes[family] += 1
        try:
            score = float(outcome.get("learning_score", item.get("public_score", 0.0)) or 0.0)
        except (TypeError, ValueError):
            score = 0.0
        if math.isfinite(score):
            best[family] = max(best.get(family, float("-inf")), score)
        try:
            hashes.append(normalize_hash(str(item.get("expression") or "")))
        except (SyntaxError, ValueError):
            pass
    total = sum(attempts.values())
    hhi = (
        sum((count / total) ** 2 for count in attempts.values())
        if total else 0.0
    )
    covered = sum(attempts.get(family, 0) > 0 for family in families)
    passed_covered = sum(passes.get(family, 0) > 0 for family in families)
    return {
        "mechanism_counts": {family: attempts.get(family, 0) for family in families},
        "passed_mechanism_counts": {family: passes.get(family, 0) for family in families},
        "mechanism_best_scores": {
            family: round(best[family], 4) for family in families if family in best
        },
        "mechanism_coverage": round(covered / len(families), 6),
        "passed_mechanism_coverage": round(passed_covered / len(families), 6),
        "mechanism_hhi": round(hhi, 6),
        "distinct_mechanisms": covered,
        "distinct_passed_mechanisms": passed_covered,
        "exact_ast_duplicate_rate": round(1.0 - len(set(hashes)) / max(1, len(hashes)), 6),
    }


def select_target_mechanism(
    items: Iterable[dict],
    market: str,
    rng: random.Random | None = None,
) -> str:
    snapshot = diversity_snapshot(items, market)
    attempts = snapshot["mechanism_counts"]
    passes = snapshot["passed_mechanism_counts"]
    families = list(mechanisms_for_market(market))
    ranked = sorted(
        families,
        key=lambda family: (passes[family], attempts[family], families.index(family)),
    )
    best_key = (passes[ranked[0]], attempts[ranked[0]])
    tied = [family for family in ranked if (passes[family], attempts[family]) == best_key]
    return (rng or random).choice(tied)


def diversity_adjusted_score(
    task_best_scores: dict[str, float],
    envelopes: Iterable[dict],
    market: str,
    required_mechanisms: int = 4,
    *,
    return_source_weight: float = 0.0,
    required_return_sources: int = 4,
) -> tuple[float, dict]:
    rows = list(envelopes)
    if not 0.0 <= float(return_source_weight) <= 0.5:
        raise ValueError("return_source_weight 必须在 [0, 0.5]")
    snapshot = diversity_snapshot(rows, market)
    task_quality = (
        sum(float(value) for value in task_best_scores.values())
        / len(task_best_scores)
        if task_best_scores else 0.0
    )
    family_scores = sorted(
        (float(value) for value in snapshot["mechanism_best_scores"].values()),
        reverse=True,
    )
    required = max(1, min(required_mechanisms, len(mechanisms_for_market(market))))
    padded = (family_scores[:required] + [0.0] * required)[:required]
    family_quality = sum(padded) / required
    legacy_score = 0.40 * task_quality + 0.60 * family_quality
    return_sources = cluster_training_return_sources(rows)
    source_quality = return_source_quality(
        return_sources,
        required_sources=required_return_sources,
    )
    source_weight = float(return_source_weight)
    score = (1.0 - source_weight) * legacy_score + source_weight * source_quality
    detail = {
        **snapshot,
        "score_semantics": (
            "legacy_mechanism_score_plus_training_return_source_quality_v2"
            if source_weight > 0
            else "task_quality_40pct_plus_top4_mechanism_quality_60pct_v1"
        ),
        "task_quality": round(task_quality, 4),
        "family_quality": round(family_quality, 4),
        "required_mechanisms": required,
        "legacy_diversity_adjusted_score": round(legacy_score, 4),
        "return_source_weight": round(source_weight, 4),
        "required_return_sources": max(1, int(required_return_sources)),
        "return_source_quality": round(source_quality, 4),
        "return_source_clusters": return_sources["return_source_clusters"],
        "effective_return_sources": return_sources["effective_return_sources"],
        "scoped_behavior_duplicate_rate": return_sources[
            "return_source_redundancy_rate"
        ],
        "return_source_signature_coverage": return_sources["signature_coverage"],
        "diversity_adjusted_score": round(score, 4),
    }
    return round(score, 4), detail
