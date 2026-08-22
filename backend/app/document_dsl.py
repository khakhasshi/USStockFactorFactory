"""Traceable document/research-note to minimal DSL candidate conversion."""

from __future__ import annotations

import hashlib
import re

from .config import get_dsl_fields
from .dsl.engine import validate
from .mechanism_catalog import catalog


PROTOCOL = "factorfactory.document-to-dsl/v1"
_TEMPLATES = {
    "trend_multi_scale": "rank(returns(close, 20) + returns(close, 60))",
    "trend_efficiency": "rank(returns(close, 60) / (ts_std(returns(close, 1), 60) + 1e-9))",
    "short_reversal": "rank(-returns(close, 5))",
    "downside_volatility": "rank(-ts_std(returns(close, 1), 60))",
    "range_compression": "rank(-ts_mean((high - low) / (close + 1e-9), 20))",
    "volume_burst": "rank(vol / (ts_mean(vol, 20) + 1e-9))",
    "volume_exhaustion": "rank(-returns(close, 5) / (ts_mean(vol, 20) + 1e-9))",
    "price_volume_confirmation": "rank(ts_corr(returns(close, 1), ts_delta(log(vol), 1), 20))",
    "amihud_illiquidity": "rank(ts_mean(abs(returns(close, 1)) / (amount + 1e-9), 20))",
    "overnight_gap": "rank((open - delay(close, 1)) / (delay(close, 1) + 1e-9))",
    "intraday_reversal": "rank(-(close - open) / (open + 1e-9))",
    "close_location": "rank((close - low) / (high - low + 1e-9))",
    "valuation_earnings_yield": "rank(1 / (pe_ttm + 1e-9))",
    "valuation_book_to_price": "rank(1 / (pb + 1e-9))",
    "valuation_sales_yield": "rank(1 / (ps_ttm + 1e-9))",
    "size_capacity": "rank(-log(total_mv))",
    "capital_flow_imbalance": "rank(net_mf_amount / (amount + 1e-9))",
    "large_order_exhaustion": "rank((buy_lg_amount - sell_lg_amount) / (amount + 1e-9))",
}


def document_to_dsl(text: str, *, market: str, max_candidates: int = 3) -> dict:
    body = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(body) < 20:
        raise ValueError("document text must contain at least 20 non-whitespace characters")
    if market not in {"us", "ashare"}:
        raise ValueError("market must be us or ashare")
    fields = set(get_dsl_fields(market))
    lowered = body.lower()
    scored = []
    for row in catalog(market)["mechanisms"]:
        tokens = set(re.findall(r"[a-z_]+|[\u4e00-\u9fff]{2,}", row["mechanism"] + " " + row["hypothesis"]))
        score = sum(token in lowered for token in tokens)
        if row["mechanism"] in _TEMPLATES and set(row["required_fields"]) <= fields:
            scored.append((score, row["mechanism"], row))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    chosen = scored[: max(1, min(3, int(max_candidates)))]
    candidates = []
    for keyword_score, mechanism, row in chosen:
        expression = _TEMPLATES[mechanism]
        error = validate(expression, list(fields))
        candidates.append(
            {
                "mechanism": mechanism,
                "hypothesis": row["hypothesis"],
                "expression": expression,
                "dsl_valid": error is None,
                "validation_error": error,
                "document_keyword_score": keyword_score,
                "score_bonus": 0.0,
            }
        )
    return {
        "protocol": PROTOCOL,
        "market": market,
        "source_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
        "source_excerpt": body[:500],
        "policy": "provenance_only_same_evaluator_no_idea_bonus",
        "candidates": candidates,
    }
