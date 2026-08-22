"""Training-safe third-layer scientific governance for the full-LLM service."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

from .feedback import ensure_training_safe
from .llm import client as llm
from .observability import redact_text, redact_value


SYSTEM_PROMPT = """你是全 LLM 因子研究系统的第三层科学总督。
你不直接写因子表达式，也不能改变数据、回测、成本、评级、准入门槛或安全规则。
你的职责是依据训练安全的聚合研究记录，决定机制研究组合、探索比例、研究主任目标、
证伪规则和指令任期。没有证据时可以提出探索性研究章程，但不得声称已有收益证据。
不得引用或猜测封存评级、隔离数据、具体年份或数据分层名称。

只回复 JSON：
{
  "action": "hold|rebalance|open_frontier|falsify",
  "focus_mechanisms": ["允许的机制标识"],
  "deprioritize_mechanisms": ["允许的机制标识"],
  "exploration_share": 0.50,
  "director_objective": "交给第二层研究主任的目标",
  "falsification_rule": "在训练安全的完整 cohort 中如何证伪",
  "expiry_outer_steps": 2,
  "confidence": 0.50,
  "evidence_used": ["聚合指标或版本标识"],
  "reasoning_summary": "简短、可审计的决策依据"
}"""


def _bounded_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(low, min(high, number))


def _clean_mechanisms(value: Any, allowed: tuple[str, ...]) -> list[str]:
    rows = value if isinstance(value, list) else []
    output: list[str] = []
    for item in rows:
        mechanism = str(item).strip()
        if mechanism in allowed and mechanism not in output:
            output.append(mechanism)
    return output


def _safe_hold(
    allowed_mechanisms: tuple[str, ...],
    previous: dict | None,
    reason: str,
) -> dict:
    if previous:
        result = deepcopy(previous)
        result["decision_status"] = "semantic_rejection_hold"
        result["hold_reason"] = reason[:500]
        return result
    return {
        "directive_id": "system-hold",
        "action": "hold",
        "focus_mechanisms": list(allowed_mechanisms),
        "deprioritize_mechanisms": [],
        "exploration_share": 0.5,
        "director_objective": "保持广泛探索，等待有效的科学总督指令。",
        "falsification_rule": "无有效 LLM 指令，不据此宣称研究策略有效。",
        "expiry_outer_steps": 1,
        "confidence": 0.0,
        "evidence_used": [],
        "reasoning_summary": "科学总督响应被语义安全层拒绝。",
        "decision_status": "semantic_rejection_hold",
        "hold_reason": reason[:500],
    }


def _compact_history(history: list[dict]) -> tuple[list[dict], str]:
    # Version rows also contain an internal ``template._readonly`` block that
    # names the physical isolation layers.  It is useful to validators but is
    # deliberately excluded from every LLM context.
    payload = []
    for row in history[-12:]:
        payload.append(redact_value({
            "version_no": row.get("version_no"),
            "status": row.get("status"),
            "meta_score": row.get("meta_score"),
            "template_note": str(row.get("template_note") or "")[:400],
            "feedback_summary": row.get("feedback_summary") or {},
            "reflection": row.get("reflection") or {},
            "context_fingerprint": row.get("context_fingerprint"),
            "evaluation_protocol": row.get("evaluation_protocol"),
        }))
    ensure_training_safe(payload)
    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return payload, hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


async def propose_scientific_directive(
    history: list[dict],
    provider: dict | None,
    *,
    market: str,
    portfolio_mode: str,
    allowed_mechanisms: tuple[str, ...],
    previous_directive: dict | None = None,
    trace_context: dict | None = None,
) -> tuple[dict, str, str, dict]:
    """Return an audited L3 directive; transport failures fail closed."""
    if provider is None:
        raise RuntimeError("第三层科学总督 provider 未配置")
    compact_history, fingerprint = _compact_history(history)
    user = json.dumps(
        {
            "market": market,
            "portfolio_mode": portfolio_mode,
            "allowed_mechanisms": list(allowed_mechanisms),
            "previous_directive": redact_value(previous_directive or {}),
            "training_safe_history": compact_history,
            "instruction": (
                "形成下一任期研究指令。第一层 LLM 可以直接提出因子表达式；"
                "你只治理机制资源分配与证伪目标。"
            ),
        },
        ensure_ascii=False,
    )
    text: str | None = None
    try:
        text = await llm.chat(
            provider,
            SYSTEM_PROMPT,
            user,
            0.35,
            trace={
                **(trace_context or {}),
                "role": "scientific_governor",
                "phase": "strategic_directive",
                "feedback_fingerprint": fingerprint,
                "architecture_layer": 3,
            },
        )
        raw = llm.extract_json(text)
        action = str(raw.get("action") or "").strip()
        if action not in {"hold", "rebalance", "open_frontier", "falsify"}:
            raise llm.LLMError("action 非法")
        focus = _clean_mechanisms(raw.get("focus_mechanisms"), allowed_mechanisms)
        deprioritized = _clean_mechanisms(
            raw.get("deprioritize_mechanisms"), allowed_mechanisms
        )
        focus = [item for item in focus if item not in deprioritized]
        if not focus:
            focus = [item for item in allowed_mechanisms if item not in deprioritized]
        objective = str(raw.get("director_objective") or "").strip()[:1000]
        falsification = str(raw.get("falsification_rule") or "").strip()[:1000]
        if not objective or not falsification:
            raise llm.LLMError("缺少 director_objective 或 falsification_rule")
        evidence = raw.get("evidence_used") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]
        directive = {
            "action": action,
            "focus_mechanisms": focus,
            "deprioritize_mechanisms": deprioritized,
            "exploration_share": _bounded_float(
                raw.get("exploration_share"), 0.5, 0.2, 0.8
            ),
            "director_objective": objective,
            "falsification_rule": falsification,
            "expiry_outer_steps": int(
                _bounded_float(raw.get("expiry_outer_steps"), 2, 1, 6)
            ),
            "confidence": _bounded_float(raw.get("confidence"), 0.5, 0.0, 1.0),
            "evidence_used": [str(item).strip()[:300] for item in evidence[:10]],
            "reasoning_summary": str(raw.get("reasoning_summary") or "")[:1000],
            "history_fingerprint": fingerprint,
            "decision_status": "llm_accepted",
        }
        ensure_training_safe(directive)
        directive["directive_id"] = hashlib.sha256(
            json.dumps(directive, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()[:16]
        await llm.mark_validation(text, accepted=True)
        return directive, objective, "scientific_governor_llm", {
            "decision": action,
            "directive_id": directive["directive_id"],
            "history_context_fingerprint": fingerprint,
            "training_safe": True,
            "architecture_layer": 3,
        }
    except Exception as exc:
        error = redact_text(exc, 500)
        await llm.mark_validation(text, accepted=False, error=error)
        if text is None:
            raise RuntimeError(f"第三层科学总督 provider 不可用: {error}") from exc
        directive = _safe_hold(allowed_mechanisms, previous_directive, error)
        ensure_training_safe(directive)
        return directive, "科学总督响应被拒，保持上一有效指令", "llm_rejected", {
            "decision": "semantic_rejection_hold",
            "history_context_fingerprint": fingerprint,
            "error": error,
            "training_safe": True,
            "architecture_layer": 3,
        }
