"""Training-safe feedback contracts shared by the inner and outer LLM loops.

The mining loop may learn from INNER_PUBLIC/META_TRAIN through the aggregate
discovery result, but it must never receive HOLDOUT/VAULT payloads.  Keeping
that rule in one module makes prompt construction, persistence and tests use
the same contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from typing import Any, Iterable

from .config import EVALUATION_PROTOCOL_VERSION
from .factors.diversity import diversity_snapshot, mechanism_from_item
from .factors.return_path import (
    combined_training_signature,
    return_path_correlation,
)
from .factors.return_source_governance import cluster_training_return_sources
from .factors.semantics import audit_expression_semantics
from .factors.similarity import expression_similarity
from .observability import redact_text, redact_value

FEEDBACK_SCHEMA_VERSION = "factorfactory.evaluation-feedback/v3"
OUTER_REPORT_SCHEMA_VERSION = "factorfactory.outer-feedback/v4"

DEFAULT_CONTEXT_POLICY = {
    "top_k": 4,
    "priority_k": 2,
    "near_miss_k": 3,
    "failure_k": 4,
    "error_k": 2,
    "max_context_chars": 12_000,
}

_FORBIDDEN_TRAINING_KEYS = {
    "holdout",
    "vault",
    "meta_holdout",
    "factor_vault",
    "validation_metrics",
    "eligibility",
    "live_rank",
}
_FORBIDDEN_TRAINING_TEXT = (
    "META_HOLDOUT",
    "FACTOR_VAULT",
)

_ACTION_RULES = (
    ("ICIR 非正", "检查双向评价证据与经济假设是否一致；不要用机械符号翻转掩盖错误机制"),
    ("HAC 显著性不足", "减少自由度和窗口搜索，优先可复现的单一机制"),
    ("组合 Sharpe 非正", "改进费后收益来源，避免只有相关性而没有可交易收益"),
    ("HAC 置信度不足", "提高跨期稳定性并减少重叠标签造成的虚假显著性"),
    ("收益下置信界非正", "提高最差情形收益，不再追逐高点估计"),
    ("era 方向一致性不足", "降低状态依赖，检查机制在不同市场阶段是否同向"),
    ("era 费后盈利比例不足", "优先改善失败时期而不是继续抬高最好时期"),
    ("单调性", "重构横截面排序，使分位收益随信号强度稳定变化"),
    ("换手", "使用更慢窗口、平滑或滞后结构，减少无效调仓"),
    ("压力成本", "扩大毛收益与交易成本之间的安全边际"),
    ("成本盈亏平衡缓冲不足", "降低换手或寻找更强、可执行的收益来源"),
    ("覆盖率", "减少字段缺失敏感结构并扩大有效横截面覆盖"),
    ("有效样本", "降低历史窗口要求或避免稀疏字段组合"),
)


def _finite_float(value: Any, default: float | None = None) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _round(value: Any, digits: int = 4) -> float | None:
    result = _finite_float(value)
    return round(result, digits) if result is not None else None


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalised_expression(expression: str) -> str:
    return re.sub(r"\s+", "", expression or "").lower()


def _cluster_duplicate_rate(
    items: list[dict],
    similarity,
    threshold: float,
) -> float:
    if len(items) < 2:
        return 0.0
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    for left in range(len(items)):
        for right in range(left + 1, len(items)):
            try:
                value = similarity(items[left], items[right])
            except (SyntaxError, ValueError, TypeError):
                value = None
            if value is not None and float(value) >= threshold:
                union(left, right)
    groups = len({find(index) for index in range(len(items))})
    return round(1.0 - groups / len(items), 6)


def _worst_stress_sharpe(metrics: dict) -> float | None:
    values = [
        _finite_float(row.get("sharpe"))
        for row in (metrics.get("cost_stress") or [])
        if isinstance(row, dict)
    ]
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


def _relevant_branch(metrics: dict, portfolio_mode: str) -> dict:
    if portfolio_mode == "long_only":
        return dict(metrics.get("active") or {})
    return dict(metrics.get("net") or {})


def _improvement_targets(reasons: Iterable[str]) -> list[str]:
    actions: list[str] = []
    for reason in reasons:
        for token, action in _ACTION_RULES:
            if token in reason and action not in actions:
                actions.append(action)
    return actions[:6]


def ensure_training_safe(value: Any, path: str = "feedback") -> None:
    """Reject accidental HOLDOUT/VAULT material before it reaches a prompt."""
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if lowered in _FORBIDDEN_TRAINING_KEYS:
                raise ValueError(f"{path}.{key} 是禁止回灌的字段")
            ensure_training_safe(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            ensure_training_safe(child, f"{path}[{index}]")
    elif isinstance(value, str):
        upper = value.upper()
        for token in _FORBIDDEN_TRAINING_TEXT:
            if token in upper:
                raise ValueError(f"{path} 包含禁止回灌的数据层标识 {token}")


def build_feedback_envelope(
    *,
    node_id: int | None,
    task_name: str,
    expression: str,
    hypothesis: str,
    source: str,
    status: str,
    error: str | None,
    public_score: float | None,
    public_metrics: dict | None,
    evaluation_protocol: str,
    market: str,
    portfolio_mode: str,
    direction: int,
    parent_id: int | None = None,
    proposal_meta: dict | None = None,
) -> dict:
    """Build the only feedback shape allowed to enter either LLM prompt."""
    public = dict(public_metrics or {})
    discovery = dict(public.get("discovery") or {})
    effective = dict(discovery.get("effective_metrics") or {})
    selection = dict(discovery.get("selection_evidence") or {})
    components = {
        str(key): _round(value)
        for key, value in (discovery.get("components") or {}).items()
        if _finite_float(value) is not None
    }
    reasons = [
        redact_text(reason, 240)
        for reason in (discovery.get("failure_reasons") or [])
        if str(reason).strip()
    ]
    if status != "ok":
        reasons = [f"候选计算失败: {redact_text(error or 'unknown error', 300)}"]

    branch = _relevant_branch(public, portfolio_mode)
    confidence = dict(public.get("return_confidence") or {})
    exposure = dict(public.get("market_exposure") or {})
    selected_direction = int(discovery.get("selected_direction", direction))
    direction_selection = dict(
        discovery.get("direction_selection") or {}
    )
    protocol = (
        str(evaluation_protocol or public.get("protocol_version") or "")
        or "legacy_unoriented"
    )
    proposal = dict(proposal_meta or {})
    mechanism_family = str(
        proposal.get("declared_family")
        or proposal.get("mechanism_family")
        or ""
    )
    if not mechanism_family:
        mechanism_family = mechanism_from_item({
            "expression": expression,
            "hypothesis": hypothesis,
            "proposal_meta": proposal,
        })
    semantic_audit = dict(
        proposal.get("semantic_audit")
        or audit_expression_semantics(expression, market)
    )
    return_signature = dict(
        public.get("training_return_path_signature")
        or combined_training_signature(
            public,
            dict(public_metrics or {}).get("gate") or {},
        )
    )
    # The caller normally passes public and gate separately.  Fall back to a
    # public-only signature here; orchestrator enriches it before persistence.
    if not return_signature.get("available"):
        public_signature = public.get("return_path_signature") or {}
        return_signature = {
            "available": bool(public_signature.get("available")),
            "protocol": "public_return_path_v1",
            "vector": list(public_signature.get("vector") or []),
            "fingerprint": public_signature.get("fingerprint") or "",
        }
    envelope = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "protocol_version": protocol,
        "node_id": node_id,
        "parent_id": parent_id,
        "task_name": task_name,
        "market": market,
        "portfolio_mode": portfolio_mode,
        "direction": selected_direction,
        "preferred_direction": int(
            discovery.get("preferred_direction", direction)
        ),
        "direction_policy": str(
            discovery.get("direction_policy") or "fixed"
        ),
        "direction_selection": redact_value(direction_selection),
        "expression": str(expression)[:1200],
        "hypothesis": str(hypothesis or "")[:500],
        "source": source,
        "proposal": redact_value(proposal),
        "diversity": {
            "target_family": proposal.get("target_family"),
            "declared_family": mechanism_family,
            "inferred_family": proposal.get("inferred_family")
            or mechanism_from_item({
                "expression": expression,
                "hypothesis": hypothesis,
            }),
            "family_match": proposal.get("family_match"),
            "semantic_audit": redact_value(semantic_audit),
            "return_path_signature": redact_value(return_signature),
        },
        "outcome": {
            "status": status,
            "score": _round(public_score, 4) or 0.0,
            "learning_score": _round(
                discovery.get("learning_score", public_score),
                4,
            )
            or 0.0,
            "gate_score": _round(discovery.get("gate_score"), 4) or 0.0,
            "score_semantics": str(
                discovery.get("score_semantics") or "legacy_clipped"
            ),
            "passed": bool(discovery.get("passed")) if status == "ok" else False,
        },
        "metrics": {
            "ic_mean": _round(public.get("ic_mean"), 6),
            "icir": _round(effective.get("icir", public.get("icir"))),
            "coverage": _round(
                effective.get("coverage", public.get("coverage"))
            ),
            "daily_turnover": _round(
                effective.get(
                    "daily_turnover",
                    public.get("daily_turnover", public.get("turnover")),
                )
            ),
            "portfolio_sharpe": _round(
                effective.get("portfolio_sharpe", branch.get("sharpe"))
            ),
            "portfolio_ann_return": _round(branch.get("ann_return"), 6),
            "hac_p_value": _round(
                effective.get("hac_p_value", public.get("hac_p_value"))
            ),
            "return_hac_t": _round(
                effective.get(
                    "return_hac_t",
                    selection.get(
                        "worst_training_return_hac_t",
                        confidence.get("hac_t_stat"),
                    ),
                )
            ),
            "selection_hurdle_t": _round(selection.get("hurdle_t")),
            "sharpe_lcb": _round(
                effective.get(
                    "sharpe_lcb",
                    selection.get(
                        "worst_training_sharpe_lcb",
                        confidence.get("sharpe_lcb"),
                    ),
                )
            ),
            "ann_return_lcb": _round(
                effective.get(
                    "ann_return_lcb",
                    selection.get(
                        "worst_training_ann_return_lcb",
                        confidence.get("ann_return_lcb"),
                    ),
                ),
                6,
            ),
            "era_consistency": _round(
                effective.get(
                    "era_consistency",
                    public.get("era_consistency"),
                )
            ),
            "profitable_era_rate": _round(
                effective.get(
                    "profitable_era_rate",
                    public.get("profitable_era_rate"),
                )
            ),
            "monotonicity": _round(
                effective.get("monotonicity", public.get("monotonicity"))
            ),
            "worst_era_sharpe": _round(public.get("worst_era_sharpe")),
            "worst_stress_sharpe": _round(
                effective.get(
                    "worst_stress_sharpe",
                    _worst_stress_sharpe(public),
                )
            ),
            "cost_cushion_multiple": _round(
                effective.get(
                    "cost_cushion_multiple",
                    public.get("cost_cushion_multiple"),
                )
            ),
            "adv_participation_p95": _round(
                public.get("adv_participation_p95")
            ),
            "market_beta": _round(exposure.get("beta")),
        },
        "components": components,
        "failure_reasons": reasons,
        "improvement_targets": _improvement_targets(reasons),
    }
    envelope["feedback_fingerprint"] = _canonical_hash(envelope)
    ensure_training_safe(envelope)
    return envelope


def enrich_feedback_with_factor_admission(
    envelope: dict,
    admission: dict | None,
) -> dict:
    """Attach post-evaluation library governance without altering V4 scores."""
    if not admission:
        return envelope
    result = {
        **envelope,
        "outcome": dict(envelope.get("outcome") or {}),
        "failure_reasons": list(envelope.get("failure_reasons") or []),
        "improvement_targets": list(envelope.get("improvement_targets") or []),
        "factor_admission": redact_value(admission),
    }
    accepted = bool(admission.get("accepted"))
    result["outcome"]["factor_library_admitted"] = accepted
    if not accepted:
        reason = str(admission.get("reason") or "factor_admission_rejected")
        if reason.startswith("return_path_duplicate"):
            failure = "训练收益路径与已入库因子高度重复"
            target = "改变经济机制和收益形成路径，不要只换窗口、符号或归一化包装"
        elif reason.startswith("structural_duplicate"):
            failure = "表达式结构与已入库因子高度重复"
            target = "切换字段关系或算子结构，提出可归因的新机制而非语法近邻"
        else:
            failure = f"因子治理拒绝: {reason}"
            target = "根据因子治理拒绝原因改写候选"
        if failure not in result["failure_reasons"]:
            result["failure_reasons"].append(failure)
        if target not in result["improvement_targets"]:
            result["improvement_targets"].append(target)
    result["feedback_fingerprint"] = _canonical_hash(result)
    ensure_training_safe(result)
    return result


def resolve_context_policy(template: dict | None) -> dict:
    raw = dict((template or {}).get("context_policy") or {})
    policy = dict(DEFAULT_CONTEXT_POLICY)
    limits = {
        "top_k": (1, 8),
        "priority_k": (0, 5),
        "near_miss_k": (0, 6),
        "failure_k": (0, 8),
        "error_k": (0, 4),
        "max_context_chars": (3000, 20_000),
    }
    for key, (lower, upper) in limits.items():
        try:
            policy[key] = max(lower, min(upper, int(raw.get(key, policy[key]))))
        except (TypeError, ValueError):
            pass
    return policy


def normalise_scoring_weights(template: dict | None) -> dict:
    raw = dict((template or {}).get("scoring_weights") or {})
    weights = {
        "icir_weight": max(
            0.05, _finite_float(raw.get("icir_weight"), 0.45) or 0.45
        ),
        "consistency_weight": max(
            0.05,
            _finite_float(raw.get("consistency_weight"), 0.25) or 0.25,
        ),
        "turnover_weight": max(
            0.05,
            _finite_float(raw.get("turnover_weight"), 0.30) or 0.30,
        ),
    }
    total = sum(weights.values())
    return {key: round(value / total, 6) for key, value in weights.items()}


def feedback_priority(envelope: dict, template: dict | None) -> float:
    """Template-controlled context priority; never the authoritative score."""
    weights = normalise_scoring_weights(template)
    metrics = envelope.get("metrics") or {}
    components = envelope.get("components") or {}
    predictive = _finite_float(components.get("predictive"))
    if predictive is None:
        predictive = max(
            0.0,
            min(1.0, (_finite_float(metrics.get("icir"), 0.0) or 0.0) / 2.0),
        )
    consistency = _finite_float(components.get("stability"))
    if consistency is None:
        consistency = max(
            0.0,
            min(
                1.0,
                _finite_float(metrics.get("era_consistency"), 0.0) or 0.0,
            ),
        )
    turnover = max(
        0.0, _finite_float(metrics.get("daily_turnover"), 1.0) or 1.0
    )
    turnover_quality = math.exp(-turnover / 0.20)
    return round(
        weights["icir_weight"] * predictive
        + weights["consistency_weight"] * consistency
        + weights["turnover_weight"] * turnover_quality,
        6,
    )


def _weakest_components(envelope: dict, limit: int = 3) -> list[str]:
    rows = [
        (key, value)
        for key, value in (envelope.get("components") or {}).items()
        if _finite_float(value) is not None
    ]
    return [
        f"{key}={float(value):.2f}"
        for key, value in sorted(rows, key=lambda item: item[1])[:limit]
    ]


def _display_metric(
    value: Any,
    *,
    digits: int = 2,
    signed: bool = False,
    percent: bool = False,
    suffix: str = "",
) -> str:
    number = _finite_float(value)
    if number is None:
        return "NA"
    if percent:
        number *= 100.0
    sign = "+" if signed else ""
    return f"{number:{sign}.{digits}f}{suffix}"


def select_feedback_examples(
    envelopes: Iterable[dict],
    template: dict | None,
) -> dict[str, list[dict]]:
    policy = resolve_context_policy(template)
    current = [
        item
        for item in envelopes
        if item.get("protocol_version") == EVALUATION_PROTOCOL_VERSION
    ]
    valid = [
        item
        for item in current
        if (item.get("outcome") or {}).get("status") == "ok"
    ]
    errors = [item for item in current if item not in valid]
    winners = sorted(
        valid,
        key=lambda item: (
            bool((item.get("outcome") or {}).get("passed")),
            _finite_float((item.get("outcome") or {}).get("score"), 0.0),
        ),
        reverse=True,
    )
    near_misses = sorted(
        [
            item
            for item in valid
            if not (item.get("outcome") or {}).get("passed")
        ],
        key=lambda item: (
            len(item.get("failure_reasons") or []),
            -(_finite_float((item.get("outcome") or {}).get("score"), 0.0) or 0.0),
        ),
    )
    failures = sorted(
        [
            item
            for item in valid
            if item.get("failure_reasons")
        ],
        key=lambda item: (
            len(item.get("failure_reasons") or []),
            item.get("node_id") or 0,
        ),
        reverse=True,
    )
    priority = sorted(
        valid,
        key=lambda item: feedback_priority(item, template),
        reverse=True,
    )
    return {
        "authoritative_best": winners[: policy["top_k"]],
        "template_priority": priority[: policy["priority_k"]],
        "near_misses": near_misses[: policy["near_miss_k"]],
        "failures": failures[: policy["failure_k"]],
        "errors": sorted(
            errors,
            key=lambda item: item.get("node_id") or 0,
            reverse=True,
        )[: policy["error_k"]],
    }


def _format_example(item: dict, template: dict | None) -> str:
    outcome = item.get("outcome") or {}
    metrics = item.get("metrics") or {}
    reasons = item.get("failure_reasons") or []
    proposal = item.get("proposal") or {}
    direction_selection = item.get("direction_selection") or {}
    diversity = item.get("diversity") or {}
    line = (
        f"node={item.get('node_id') or 'pending'} "
        f"mechanism={diversity.get('declared_family') or 'unknown'} "
        f"direction={int(item.get('direction') or 1):+d} "
        f"learning_score={_finite_float(outcome.get('learning_score', outcome.get('score')), 0.0):.3f} "
        f"hard_gate_score={_finite_float(outcome.get('gate_score'), 0.0):.3f} "
        f"passed={bool(outcome.get('passed'))} "
        f"priority={feedback_priority(item, template):.3f} "
        f"ICIR={_display_metric(metrics.get('icir'), signed=True)} "
        f"Sharpe={_display_metric(metrics.get('portfolio_sharpe'), signed=True)} "
        f"ann_ret={_display_metric(metrics.get('portfolio_ann_return'), digits=1, signed=True, percent=True, suffix='%')}"
    )
    details = [line, f"  expr: {item.get('expression', '')}"]
    orientation_candidates = direction_selection.get("candidates") or {}
    if orientation_candidates:
        details.append(
            "  direction_test: "
            + ", ".join(
                (
                    f"{label} learn="
                    f"{_finite_float(row.get('learning_score'), 0.0):.3f} "
                    f"gate={_finite_float(row.get('gate_score'), 0.0):.3f} "
                    f"passed={bool(row.get('passed'))}"
                )
                for label, row in sorted(orientation_candidates.items())
                if isinstance(row, dict)
            )
        )
    details.append(
        "  confidence: "
        f"HAC_t={_display_metric(metrics.get('return_hac_t'), signed=True)}"
        f"/hurdle={_display_metric(metrics.get('selection_hurdle_t'))}, "
        f"Sharpe_LCB={_display_metric(metrics.get('sharpe_lcb'), signed=True)}, "
        f"return_LCB={_display_metric(metrics.get('ann_return_lcb'), signed=True, percent=True, suffix='%')}"
    )
    details.append(
        "  stability/cost: "
        f"era={_display_metric(metrics.get('era_consistency'))}, "
        f"profitable_era={_display_metric(metrics.get('profitable_era_rate'))}, "
        f"monotonicity={_display_metric(metrics.get('monotonicity'))}, "
        f"worst_era_sharpe={_display_metric(metrics.get('worst_era_sharpe'), signed=True)}, "
        f"stress_sharpe={_display_metric(metrics.get('worst_stress_sharpe'), signed=True)}"
    )
    details.append(
        "  implementability: "
        f"turnover={_display_metric(metrics.get('daily_turnover'), digits=1, percent=True, suffix='%')}, "
        f"cost_cushion={_display_metric(metrics.get('cost_cushion_multiple'), suffix='x')}, "
        f"coverage={_display_metric(metrics.get('coverage'), digits=1, percent=True, suffix='%')}, "
        f"ADV_p95={_display_metric(metrics.get('adv_participation_p95'), percent=True, suffix='%')}, "
        f"beta={_display_metric(metrics.get('market_beta'), signed=True)}"
    )
    components = [
        f"{key}={float(value):.2f}"
        for key, value in sorted((item.get("components") or {}).items())
        if _finite_float(value) is not None
    ]
    if components:
        details.append(f"  components: {', '.join(components)}")
    if reasons:
        details.append(f"  failed: {'；'.join(reasons[:4])}")
    weakest = _weakest_components(item)
    if weakest:
        details.append(f"  weakest: {', '.join(weakest)}")
    if item.get("improvement_targets"):
        details.append(
            f"  next: {'；'.join((item.get('improvement_targets') or [])[:3])}"
        )
    reflection = str(proposal.get("reflection") or "").strip()
    if reflection:
        details.append(f"  prior_reflection: {reflection[:240]}")
    return "\n".join(details)


def build_inner_feedback_context(
    envelopes: Iterable[dict],
    template: dict | None,
) -> tuple[str, dict]:
    """Return a bounded rendered context plus the exact safe snapshot used."""
    current = [
        item
        for item in envelopes
        if item.get("protocol_version") == EVALUATION_PROTOCOL_VERSION
    ]
    for item in current:
        ensure_training_safe(item)
    selected = select_feedback_examples(current, template)
    summary = aggregate_feedback(current)
    weights = normalise_scoring_weights(template)
    lines = [
        f"反馈协议: {EVALUATION_PROTOCOL_VERSION} / {FEEDBACK_SCHEMA_VERSION}",
        "边界: 仅训练安全聚合反馈；最终封存层从不进入上下文。",
        (
            "方向语义: 每个候选在训练安全层同时测试 +1/-1，"
            "试验预算按双向计数；选中方向随后冻结，禁止根据隔离层翻号。"
        ),
        (
            "分数语义: learning_score 是连续失败梯度，仅用于比较、归因和搜索；"
            "hard_gate_score 与 passed 才表示硬准入，学习分绝不能覆盖失败门槛。"
        ),
        (
            "模板上下文优先权重（只影响示例选择，不改变权威评价）: "
            f"ICIR={weights['icir_weight']:.2f}, "
            f"稳定性={weights['consistency_weight']:.2f}, "
            f"低换手={weights['turnover_weight']:.2f}"
        ),
        (
            f"总体: {summary['attempts']} 次，pass={summary['pass_rate']:.1%}，"
            f"learning_mean={summary['score_mean']:.3f}，"
            f"hard_gate_mean={summary['gate_score_mean']:.3f}，"
            f"方向(+/-)={summary['direction_counts']}，"
            f"error={summary['errors']}，exact_duplicate={summary['duplicate_rate']:.1%}，"
            f"structural_duplicate={summary['structural_duplicate_rate']:.1%}，"
            f"return_path_duplicate={summary['behavior_duplicate_rate']:.1%}"
        ),
        (
            f"收益机制覆盖: distinct={summary['distinct_mechanisms']}，"
            f"passed_distinct={summary['distinct_passed_mechanisms']}，"
            f"coverage={summary['mechanism_coverage']:.1%}，"
            f"HHI={summary['mechanism_hhi']:.3f}；"
            f"attempts={summary['mechanism_counts']}；"
            f"passed={summary['passed_mechanism_counts']}"
        ),
        (
            "训练收益来源: "
            f"clusters={summary['return_source_clusters']}，"
            f"effective={summary['effective_return_sources']:.2f}，"
            f"scoped_redundancy={summary['scoped_behavior_duplicate_rate']:.1%}，"
            f"signature_coverage={summary['return_source_signature_coverage']:.1%}"
        ),
    ]
    if summary["failure_reason_counts"]:
        lines.append(
            "重复失败经验: "
            + "；".join(
                f"{key}×{value}"
                for key, value in list(
                    summary["failure_reason_counts"].items()
                )[:6]
            )
        )
    if summary["component_means"]:
        lines.append(
            "组件均值: "
            + ", ".join(
                f"{key}={value:.2f}"
                for key, value in summary["component_means"].items()
            )
        )
    if summary["improvement_target_counts"]:
        lines.append(
            "优先改进经验: "
            + "；".join(
                f"{key}×{value}"
                for key, value in summary[
                    "improvement_target_counts"
                ].items()
            )
        )
    headings = {
        "authoritative_best": "连续学习分较优样本（不代表准入）",
        "template_priority": "当前模板关注样本",
        "near_misses": "最接近通过的样本",
        "failures": "需要避免的失败样本",
        "errors": "表达式/计算错误",
    }
    for key, heading in headings.items():
        rows = selected[key]
        if not rows:
            continue
        lines.append(f"\n[{heading}]")
        lines.extend(_format_example(item, template) for item in rows)
    text = "\n".join(lines)
    max_chars = resolve_context_policy(template)["max_context_chars"]
    if len(text) > max_chars:
        text = text[: max_chars - 40] + "\n[上下文已按策略截断]"
    snapshot = {
        "schema_version": FEEDBACK_SCHEMA_VERSION,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "summary": summary,
        "selection": {
            key: [item.get("feedback_fingerprint") for item in rows]
            for key, rows in selected.items()
        },
        "scoring_weights": weights,
        "context_policy": resolve_context_policy(template),
    }
    snapshot["context_fingerprint"] = _canonical_hash(snapshot)
    ensure_training_safe(snapshot)
    return text, snapshot


def _summary_core(envelopes: list[dict]) -> dict:
    attempts = len(envelopes)
    valid = [
        item
        for item in envelopes
        if (item.get("outcome") or {}).get("status") == "ok"
    ]
    scores = [
        _finite_float(
            (item.get("outcome") or {}).get(
                "learning_score",
                (item.get("outcome") or {}).get("score"),
            ),
            0.0,
        )
        or 0.0
        for item in valid
    ]
    gate_scores = [
        _finite_float(
            (item.get("outcome") or {}).get("gate_score"),
            0.0,
        )
        or 0.0
        for item in valid
    ]
    passed = [
        item for item in valid if (item.get("outcome") or {}).get("passed")
    ]
    reasons = Counter(
        reason
        for item in valid
        for reason in (item.get("failure_reasons") or [])
    )
    targets = Counter(
        target
        for item in valid
        for target in (item.get("improvement_targets") or [])
    )
    sources = Counter(str(item.get("source") or "unknown") for item in envelopes)
    directions = Counter(
        f"{int(item.get('direction') or 1):+d}"
        for item in valid
    )
    components: dict[str, list[float]] = defaultdict(list)
    metrics: dict[str, list[float]] = defaultdict(list)
    for item in valid:
        for key, value in (item.get("components") or {}).items():
            if (number := _finite_float(value)) is not None:
                components[key].append(number)
        for key, value in (item.get("metrics") or {}).items():
            if (number := _finite_float(value)) is not None:
                metrics[key].append(number)
    expressions = [
        _normalised_expression(str(item.get("expression") or ""))
        for item in valid
        if item.get("expression")
    ]
    unique_expressions = len(set(expressions))
    market = str(next((item.get("market") for item in valid if item.get("market")), "us"))
    diversity = diversity_snapshot(valid, market)
    structural_duplicate_rate = _cluster_duplicate_rate(
        valid,
        lambda left, right: expression_similarity(
            str(left.get("expression") or ""),
            str(right.get("expression") or ""),
        ),
        0.84,
    )
    behavior_rows = [
        item
        for item in valid
        if ((item.get("diversity") or {}).get("return_path_signature") or {}).get("available")
    ]
    behavior_duplicate_rate = _cluster_duplicate_rate(
        behavior_rows,
        lambda left, right: return_path_correlation(
            (left.get("diversity") or {}).get("return_path_signature"),
            (right.get("diversity") or {}).get("return_path_signature"),
        ),
        0.85,
    )
    return_sources = cluster_training_return_sources(
        valid,
        correlation_threshold=0.85,
    )
    return {
        "score_semantics": "continuous_failure_margin_v4.2",
        "attempts": attempts,
        "valid": len(valid),
        "errors": attempts - len(valid),
        "passed": len(passed),
        "pass_rate": round(len(passed) / max(1, len(valid)), 6),
        "score_mean": round(statistics.fmean(scores), 4) if scores else 0.0,
        "score_std": (
            round(statistics.pstdev(scores), 4) if len(scores) >= 2 else 0.0
        ),
        "score_best": round(max(scores), 4) if scores else 0.0,
        "gate_score_mean": (
            round(statistics.fmean(gate_scores), 4)
            if gate_scores
            else 0.0
        ),
        "gate_score_best": (
            round(max(gate_scores), 4)
            if gate_scores
            else 0.0
        ),
        "duplicate_rate": round(
            1.0 - unique_expressions / max(1, len(expressions)),
            6,
        ),
        "structural_duplicate_rate": structural_duplicate_rate,
        "behavior_duplicate_rate": behavior_duplicate_rate,
        # V1 is retained above for historical report comparability.  V2 uses
        # quality representatives and task-scoped comparisons, avoiding both
        # cross-task false matches and transitive correlation chaining.
        "scoped_behavior_duplicate_rate": return_sources[
            "return_source_redundancy_rate"
        ],
        "return_source_clusters": return_sources["return_source_clusters"],
        "effective_return_sources": return_sources["effective_return_sources"],
        "largest_return_source_cluster": return_sources[
            "largest_return_source_cluster"
        ],
        "return_source_signature_coverage": return_sources[
            "signature_coverage"
        ],
        "return_source_governance": {
            key: value
            for key, value in return_sources.items()
            if key != "representative_qualities"
        },
        **diversity,
        "source_counts": dict(sources.most_common()),
        "direction_counts": dict(directions.most_common()),
        "failure_reason_counts": dict(reasons.most_common(12)),
        "improvement_target_counts": dict(targets.most_common(8)),
        "component_means": {
            key: round(statistics.fmean(values), 4)
            for key, values in sorted(components.items())
        },
        "metric_means": {
            key: round(statistics.fmean(values), 6)
            for key, values in sorted(metrics.items())
        },
    }


def aggregate_feedback(envelopes: Iterable[dict]) -> dict:
    rows = list(envelopes)
    for item in rows:
        ensure_training_safe(item)
    protocols = sorted(
        {
            str(item.get("protocol_version") or "unknown")
            for item in rows
        }
    )
    core = _summary_core(rows)
    by_task: dict[str, list[dict]] = defaultdict(list)
    for item in rows:
        by_task[str(item.get("task_name") or "unknown")].append(item)
    report = {
        "schema_version": OUTER_REPORT_SCHEMA_VERSION,
        "protocol_versions": protocols,
        **core,
        "by_task": {
            task: _summary_core(items)
            for task, items in sorted(by_task.items())
        },
    }
    report["feedback_fingerprint"] = _canonical_hash(report)
    ensure_training_safe(report)
    return report


def combine_seed_feedback(seed_results: Iterable[dict]) -> dict:
    seeds = list(seed_results)
    envelopes = [
        envelope
        for seed in seeds
        for envelope in (seed.get("envelopes") or [])
    ]
    report = aggregate_feedback(envelopes)
    report["seeds"] = [
        {
            "seed": int(seed.get("seed", 0)),
            "meta_score": _round(seed.get("score")) or 0.0,
            "task_best_scores": {
                key: _round(value) or 0.0
                for key, value in (seed.get("task_best_scores") or {}).items()
            },
        }
        for seed in seeds
    ]
    report["seed_score_mean"] = (
        round(
            statistics.fmean(row["meta_score"] for row in report["seeds"]),
            4,
        )
        if report["seeds"]
        else 0.0
    )
    report["seed_score_std"] = (
        round(
            statistics.pstdev(row["meta_score"] for row in report["seeds"]),
            4,
        )
        if len(report["seeds"]) >= 2
        else 0.0
    )
    report["feedback_fingerprint"] = _canonical_hash(report)
    ensure_training_safe(report)
    return report


def compare_feedback_reports(candidate: dict, incumbent: dict | None) -> dict:
    incumbent = dict(incumbent or {})
    cand_components = candidate.get("component_means") or {}
    inc_components = incumbent.get("component_means") or {}
    component_deltas = {
        key: round(
            (_finite_float(cand_components.get(key), 0.0) or 0.0)
            - (_finite_float(inc_components.get(key), 0.0) or 0.0),
            4,
        )
        for key in sorted(set(cand_components) | set(inc_components))
    }
    report = {
        "schema_version": OUTER_REPORT_SCHEMA_VERSION,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "candidate_fingerprint": candidate.get("feedback_fingerprint"),
        "incumbent_fingerprint": incumbent.get("feedback_fingerprint"),
        "deltas": {
            "seed_score_mean": round(
                (_finite_float(candidate.get("seed_score_mean"), 0.0) or 0.0)
                - (
                    _finite_float(incumbent.get("seed_score_mean"), 0.0)
                    or 0.0
                ),
                4,
            ),
            "gate_score_mean": round(
                (
                    _finite_float(
                        candidate.get("gate_score_mean"),
                        0.0,
                    )
                    or 0.0
                )
                - (
                    _finite_float(
                        incumbent.get("gate_score_mean"),
                        0.0,
                    )
                    or 0.0
                ),
                4,
            ),
            "pass_rate": round(
                (_finite_float(candidate.get("pass_rate"), 0.0) or 0.0)
                - (_finite_float(incumbent.get("pass_rate"), 0.0) or 0.0),
                4,
            ),
            "duplicate_rate": round(
                (
                    _finite_float(candidate.get("duplicate_rate"), 0.0)
                    or 0.0
                )
                - (
                    _finite_float(incumbent.get("duplicate_rate"), 0.0)
                    or 0.0
                ),
                4,
            ),
            "errors": int(candidate.get("errors") or 0)
            - int(incumbent.get("errors") or 0),
        },
        "component_deltas": component_deltas,
        "candidate_failures": dict(
            list((candidate.get("failure_reason_counts") or {}).items())[:8]
        ),
        "incumbent_failures": dict(
            list((incumbent.get("failure_reason_counts") or {}).items())[:8]
        ),
    }
    report["comparison_fingerprint"] = _canonical_hash(report)
    ensure_training_safe(report)
    return report
