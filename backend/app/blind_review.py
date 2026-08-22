"""Cryptographically separated method and code review packets."""

from __future__ import annotations

import hashlib
import json

from .dsl.engine import expression_profile, validate
from .factors.semantics import audit_expression_semantics


PROTOCOL = "factorfactory.double-blind-review/v1"


def _hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_review_packets(*, expression: str, hypothesis: str, market: str) -> dict:
    method = {
        "role": "method_reviewer",
        "expression": expression,
        "hypothesis": hypothesis,
        "questions": [
            "经济机制是否可证伪且方向一致",
            "是否只是已有机制的换窗换包装",
            "预期在哪些状态失效",
            "是否值得消耗独立验证预算",
        ],
        "forbidden_context": ["code_review", "frozen_rating", "vault"],
    }
    code = {
        "role": "code_reviewer",
        "expression": expression,
        "market": market,
        "questions": [
            "DSL字段和算子是否合法",
            "是否存在未来函数或时间错位",
            "除零、log非正数和极端值是否稳定",
            "历史窗口和复杂度是否合理",
        ],
        "forbidden_context": ["method_review", "economic_narrative", "frozen_rating", "vault"],
    }
    return {
        "protocol": PROTOCOL,
        "method_packet": {**method, "packet_hash": _hash(method)},
        "code_packet": {**code, "packet_hash": _hash(code)},
        "isolation": "reviewers_receive_only_their_own_packet",
    }


def deterministic_code_review(expression: str, market: str) -> dict:
    semantic = audit_expression_semantics(expression, market)
    dsl_error = validate(expression)
    try:
        profile = expression_profile(expression)
    except (SyntaxError, ValueError):
        profile = {}
    warnings = list(semantic.get("warnings") or [])
    compact = expression.replace(" ", "")
    if "/" in compact and "+1e-9" not in compact and "+1e-8" not in compact:
        warnings.append("division_without_explicit_epsilon")
    return {
        "role": "code_reviewer_deterministic",
        "passed": dsl_error is None and not semantic.get("errors"),
        "dsl_error": dsl_error,
        "semantic_errors": semantic.get("errors") or [],
        "warnings": warnings,
        "profile": profile,
    }


def seal_reviews(*, packets: dict, method_review: dict, code_review: dict) -> dict:
    method_hash = packets.get("method_packet", {}).get("packet_hash")
    code_hash = packets.get("code_packet", {}).get("packet_hash")
    if method_review.get("packet_hash") != method_hash or code_review.get("packet_hash") != code_hash:
        raise ValueError("review packet hash mismatch; cross-context or stale review rejected")
    method_pass = bool(method_review.get("passed"))
    code_pass = bool(code_review.get("passed"))
    return {
        "protocol": PROTOCOL,
        "method_packet_hash": method_hash,
        "code_packet_hash": code_hash,
        "method_passed": method_pass,
        "code_passed": code_pass,
        "verdict": "approved" if method_pass and code_pass else "rejected",
        "reviews": {"method": method_review, "code": code_review},
    }
