"""统一 LLM 客户端: 支持 openai (chat/completions) 与 anthropic (messages) 两种接口格式.

Provider 配置存于 settings 表 key='llm_providers':
{"providers": [{"name","format":"openai|anthropic","base_url","api_key","model",
"thinking":{"type":"enabled|disabled"}}],
 "inner_provider": "...", "outer_provider": "..."}
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any

import httpx

REQUEST_TIMEOUT_SECONDS = max(
    30.0,
    float(os.environ.get("FF_LLM_TIMEOUT_SECONDS", "420")),
)
MAX_ATTEMPTS = max(1, int(os.environ.get("FF_LLM_MAX_ATTEMPTS", "2")))
TIMEOUT = httpx.Timeout(REQUEST_TIMEOUT_SECONDS, connect=20.0)
logger = logging.getLogger("llm.audit")


def _openai_payload(
    provider: dict,
    system: str,
    user: str,
    temperature: float,
) -> dict[str, Any]:
    """Build an OpenAI-compatible payload with bounded provider extensions."""
    payload: dict[str, Any] = {
        "model": provider.get("model", ""),
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    thinking = provider.get("thinking")
    thinking_type = (
        str(thinking.get("type") or "").strip().lower()
        if isinstance(thinking, dict)
        else ""
    )
    if thinking_type in {"enabled", "disabled"}:
        payload["thinking"] = {"type": thinking_type}
    return payload


class LLMError(Exception):
    pass


class LLMTransportError(LLMError):
    """Retryable provider-side or network failure."""


class AuditedText(str):
    """String-compatible response carrying its append-only audit row id."""

    audit_id: int | None

    def __new__(
        cls,
        value: str,
        audit_id: int | None = None,
    ) -> "AuditedText":
        instance = super().__new__(cls, value)
        instance.audit_id = audit_id
        return instance


def _prompt_hash(system: str, user: str) -> str:
    return hashlib.sha256(
        (system + "\n---USER---\n" + user).encode("utf-8")
    ).hexdigest()


async def _persist_audit(
    *,
    provider: dict,
    system: str,
    user: str,
    trace: dict | None,
    status: str,
    response: str,
    error: str,
    latency_ms: float,
) -> int | None:
    """Persist a redacted trace without ever making the LLM call fail."""
    try:
        from ..config import EVALUATION_PROTOCOL_VERSION
        from ..db import SessionLocal
        from ..models import LLMCallAudit
        from ..observability import redact_text, redact_value

        context = dict(trace or {})
        known = {
            "experiment_id",
            "role",
            "phase",
            "evaluation_protocol",
            "miner_version_id",
            "outer_step_no",
            "task_name",
            "feedback_fingerprint",
        }
        trace_meta = {
            key: value
            for key, value in context.items()
            if key not in known
        }
        async with SessionLocal() as session:
            row = LLMCallAudit(
                experiment_id=context.get("experiment_id"),
                role=str(context.get("role") or "unspecified")[:32],
                phase=str(context.get("phase") or "")[:64],
                status=status,
                provider_name=str(provider.get("name") or "")[:128],
                model=str(provider.get("model") or "")[:128],
                evaluation_protocol=str(
                    context.get("evaluation_protocol")
                    or EVALUATION_PROTOCOL_VERSION
                )[:32],
                miner_version_id=context.get("miner_version_id"),
                outer_step_no=context.get("outer_step_no"),
                task_name=str(context.get("task_name") or "")[:64],
                prompt_hash=_prompt_hash(system, user),
                feedback_fingerprint=str(
                    context.get("feedback_fingerprint") or ""
                )[:64],
                latency_ms=round(latency_ms, 3),
                system_prompt=redact_text(system, 24_000),
                user_prompt=redact_text(user, 32_000),
                response=redact_text(response, 16_000),
                error=redact_text(error, 2000),
                trace_meta=redact_value(trace_meta),
            )
            session.add(row)
            await session.flush()
            audit_id = row.id
            await session.commit()
            return audit_id
    except Exception as exc:  # noqa: BLE001 - observability cannot break research
        logger.warning("LLM 调用审计写入失败: %s", str(exc)[:300])
        return None


async def mark_validation(
    response: str | None,
    *,
    accepted: bool,
    error: object = "",
    trace_meta_updates: dict[str, Any] | None = None,
) -> None:
    """Finalize semantic parse/validation without affecting research flow."""
    audit_id = getattr(response, "audit_id", None)
    if not audit_id:
        return
    try:
        from ..db import SessionLocal
        from ..models import LLMCallAudit
        from ..observability import redact_text, redact_value

        async with SessionLocal() as session:
            row = await session.get(LLMCallAudit, int(audit_id))
            if row is None or row.status != "response_ok":
                return
            row.status = "accepted" if accepted else "rejected"
            if not accepted:
                row.error = redact_text(error, 2000)
            if trace_meta_updates:
                row.trace_meta = {
                    **(row.trace_meta or {}),
                    **redact_value(trace_meta_updates),
                }
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - audit cannot break research
        logger.warning("LLM 语义审计更新失败: %s", str(exc)[:300])


async def chat(
    provider: dict,
    system: str,
    user: str,
    temperature: float = 0.8,
    *,
    trace: dict[str, Any] | None = None,
) -> str:
    fmt = provider.get("format", "openai")
    base = provider.get("base_url", "").rstrip("/")
    key = provider.get("api_key", "")
    model = provider.get("model", "")
    started = time.perf_counter()
    if not (base and key and model):
        error = "provider 配置不完整 (base_url/api_key/model)"
        await _persist_audit(
            provider=provider,
            system=system,
            user=user,
            trace=trace,
            status="transport_error",
            response="",
            error=error,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        raise LLMError(error)

    output = ""
    attempts_used = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        attempts_used = attempt
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                if fmt == "anthropic":
                    url = (
                        f"{base}/v1/messages"
                        if not base.endswith("/v1")
                        else f"{base}/messages"
                    )
                    resp = await client.post(
                        url,
                        headers={
                            "x-api-key": key,
                            "anthropic-version": "2023-06-01",
                        },
                        json={
                            "model": model,
                            "max_tokens": 1800,
                            "temperature": temperature,
                            "system": system,
                            "messages": [{"role": "user", "content": user}],
                        },
                    )
                    if resp.status_code != 200:
                        error_type = (
                            LLMTransportError
                            if resp.status_code == 429 or resp.status_code >= 500
                            else LLMError
                        )
                        raise error_type(
                            f"anthropic {resp.status_code}: {resp.text[:300]}"
                        )
                    data = resp.json()
                    output = "".join(
                        block.get("text", "")
                        for block in data.get("content", [])
                    )
                else:
                    url = (
                        f"{base}/chat/completions"
                        if "/chat/completions" not in base
                        else base
                    )
                    resp = await client.post(
                        url,
                        headers={"Authorization": f"Bearer {key}"},
                        json=_openai_payload(
                            provider,
                            system,
                            user,
                            temperature,
                        ),
                    )
                    if resp.status_code != 200:
                        error_type = (
                            LLMTransportError
                            if resp.status_code == 429 or resp.status_code >= 500
                            else LLMError
                        )
                        raise error_type(
                            f"openai {resp.status_code}: {resp.text[:300]}"
                        )
                    output = resp.json()["choices"][0]["message"]["content"]
            audit_trace = {
                **(trace or {}),
                "transport_attempts": attempts_used,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            }
            audit_id = await _persist_audit(
                provider=provider,
                system=system,
                user=user,
                trace=audit_trace,
                status="response_ok",
                response=output,
                error="",
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            return AuditedText(output, audit_id)
        except Exception as exc:
            retryable = isinstance(
                exc,
                (httpx.TransportError, httpx.TimeoutException, LLMTransportError),
            )
            if retryable and attempt < MAX_ATTEMPTS:
                await asyncio.sleep(min(15.0, 2.0 ** (attempt - 1)))
                continue
            audit_trace = {
                **(trace or {}),
                "transport_attempts": attempts_used,
                "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
            }
            await _persist_audit(
                provider=provider,
                system=system,
                user=user,
                trace=audit_trace,
                status="transport_error",
                response=output,
                error=str(exc),
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
            raise
    raise LLMError("LLM request exhausted without a terminal result")


def extract_json(text: str) -> dict:
    """从 LLM 回复中提取第一个 JSON 对象."""
    start = text.find("{")
    if start < 0:
        raise LLMError(f"回复中无 JSON: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise LLMError("JSON 未闭合")
