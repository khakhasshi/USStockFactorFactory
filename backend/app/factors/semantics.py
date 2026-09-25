"""Field provenance and adjustment-basis checks for factor expressions.

The market panels intentionally keep adjusted OHLC prices together with raw
exchange turnover (or a raw-close-times-volume proxy).  That is useful for
capacity modelling, but blindly dividing an absolute adjusted price move by
``amount`` creates a split/adjustment-factor exposure rather than a stable
economic signal.  The miner treats those mixed-basis expressions as invalid;
historical records remain readable and can be labelled by this audit.
"""

from __future__ import annotations

import ast
import hashlib
from typing import Any


ADJUSTED_PRICE_FIELDS = frozenset({"open", "high", "low", "close"})
RAW_TURNOVER_FIELDS = frozenset({"amount"})

_COMMON_FIELDS: dict[str, dict[str, str]] = {
    "open": {
        "basis": "forward_adjusted_price",
        "unit": "price",
        "provenance": "panel_adjusted_ohlc",
    },
    "high": {
        "basis": "forward_adjusted_price",
        "unit": "price",
        "provenance": "panel_adjusted_ohlc",
    },
    "low": {
        "basis": "forward_adjusted_price",
        "unit": "price",
        "provenance": "panel_adjusted_ohlc",
    },
    "close": {
        "basis": "forward_adjusted_price",
        "unit": "price",
        "provenance": "panel_adjusted_ohlc",
    },
    "vol": {
        "basis": "raw_reported_shares",
        "unit": "shares",
        "provenance": "panel_raw_volume",
    },
    "amount": {
        "basis": "raw_currency_turnover",
        "unit": "currency",
        "provenance": "market_amount_or_raw_close_times_volume_proxy",
    },
}

_ASHARE_FIELDS: dict[str, dict[str, str]] = {
    "pe_ttm": {"basis": "provider_snapshot", "unit": "ratio", "provenance": "ashare_daily_basic"},
    "pb": {"basis": "provider_snapshot", "unit": "ratio", "provenance": "ashare_daily_basic"},
    "ps_ttm": {"basis": "provider_snapshot", "unit": "ratio", "provenance": "ashare_daily_basic"},
    "dv_ttm": {"basis": "provider_snapshot", "unit": "ratio", "provenance": "ashare_daily_basic"},
    "total_mv": {"basis": "raw_currency_market_value", "unit": "currency", "provenance": "ashare_daily_basic"},
    "circ_mv": {"basis": "raw_currency_market_value", "unit": "currency", "provenance": "ashare_daily_basic"},
    "turnover_rate": {"basis": "provider_snapshot", "unit": "ratio", "provenance": "ashare_daily_basic"},
    "volume_ratio": {"basis": "provider_snapshot", "unit": "ratio", "provenance": "ashare_daily_basic"},
    "net_mf_amount": {"basis": "raw_currency_flow", "unit": "currency", "provenance": "ashare_moneyflow"},
    "buy_lg_amount": {"basis": "raw_currency_flow", "unit": "currency", "provenance": "ashare_moneyflow"},
    "sell_lg_amount": {"basis": "raw_currency_flow", "unit": "currency", "provenance": "ashare_moneyflow"},
    "buy_elg_amount": {"basis": "raw_currency_flow", "unit": "currency", "provenance": "ashare_moneyflow"},
    "sell_elg_amount": {"basis": "raw_currency_flow", "unit": "currency", "provenance": "ashare_moneyflow"},
    "float_share": {"basis": "raw_reported_shares", "unit": "shares", "provenance": "ashare_share_structure"},
}


def field_contract(market: str) -> dict[str, dict[str, str]]:
    if market not in {"ashare", "us"}:
        raise ValueError("market 必须是 ashare 或 us")
    result = {name: dict(meta) for name, meta in _COMMON_FIELDS.items()}
    if market == "ashare":
        result.update({name: dict(meta) for name, meta in _ASHARE_FIELDS.items()})
    if market == "us":
        result["amount"]["provenance"] = "raw_close_times_volume_proxy"
    return result


def render_field_contract(market: str, fields: list[str] | None = None) -> str:
    contract = field_contract(market)
    selected = fields or list(contract)
    lines = []
    for name in selected:
        meta = contract.get(name)
        if meta is None:
            continue
        lines.append(
            f"- {name}: unit={meta['unit']}; basis={meta['basis']}; "
            f"source={meta['provenance']}"
        )
    return "\n".join(lines)


def _fields(node: ast.AST) -> set[str]:
    operators = {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
    }
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and child.id not in operators
    }


def _contains_price(node: ast.AST) -> bool:
    return bool(_fields(node) & ADJUSTED_PRICE_FIELDS)


def _price_scale_invariant(node: ast.AST) -> bool:
    """Whether positive per-security price rescaling leaves this node stable."""
    if not _contains_price(node):
        return True
    if isinstance(node, ast.Name):
        return False
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _price_scale_invariant(node.operand)
    if isinstance(node, ast.BinOp):
        left_price = _contains_price(node.left)
        right_price = _contains_price(node.right)
        if isinstance(node.op, ast.Div) and left_price and right_price:
            return True
        if left_price and not right_price:
            return _price_scale_invariant(node.left)
        if right_price and not left_price:
            return _price_scale_invariant(node.right)
        return False
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = node.func.id
        if fn in {"ts_corr", "ts_corr_v2", "ts_rank", "sign", "returns", "ts_robust_zscore", "ts_drawdown"}:
            return True
        price_args = [arg for arg in node.args if _contains_price(arg)]
        return bool(price_args) and all(
            _price_scale_invariant(arg) for arg in price_args
        )
    return False


def _mixed_basis_nodes(root: ast.AST) -> list[ast.AST]:
    rows: list[ast.AST] = []
    for node in ast.walk(root):
        if not isinstance(node, (ast.BinOp, ast.Call)):
            continue
        fields = _fields(node)
        if not fields & RAW_TURNOVER_FIELDS:
            continue
        if not fields & ADJUSTED_PRICE_FIELDS:
            continue
        if not _price_scale_invariant(node):
            rows.append(node)
    return rows


def audit_expression_semantics(expression: str, market: str) -> dict[str, Any]:
    """Return a stable, non-mutating semantic audit for one DSL expression."""
    try:
        root = ast.parse(expression, mode="eval").body
    except SyntaxError as exc:
        return {
            "status": "error",
            "errors": [f"syntax_error: {exc.msg}"],
            "warnings": [],
            "fields": [],
            "audit_fingerprint": "",
        }
    fields = sorted(_fields(root))
    errors: list[str] = []
    warnings: list[str] = []
    if _mixed_basis_nodes(root):
        errors.append(
            "adjusted_price_raw_turnover_mixed_basis: 绝对前复权价格尺度与原始成交额混用；"
            "先将价格变化归一化为收益率/振幅比例，再与 amount 组合"
        )
    if "amount" in fields:
        source = field_contract(market)["amount"]["provenance"]
        warnings.append(f"amount_provenance={source}")
    payload = "|".join([market, ast.dump(root), *errors, *warnings])
    return {
        "status": "error" if errors else "ok",
        "errors": errors,
        "warnings": warnings,
        "fields": fields,
        "price_scale_invariant": _price_scale_invariant(root),
        "audit_fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16],
    }
