"""Manual factor-correlation diagnostics and executable DSL composition.

The correlation tool compares oriented factors on one common panel, universe,
window and rebalance schedule.  It reports three distinct notions of overlap:
cross-sectional signal-rank correlation, Rank-IC path correlation and
standalone after-cost portfolio-return correlation.  The DSL builder keeps the
structured component snapshot in its response and emits a compact expression
only after every component and the final expression pass the market whitelist.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
import threading
import time
from datetime import date, datetime, timezone
from typing import Any

import numpy as np
import polars as pl

from .config import default_panel_glob, get_dsl_fields
from .data.panel import PanelStore
from .dsl.engine import (
    MAX_EXPRESSION_AST_NODES,
    MAX_EXPRESSION_LENGTH,
    expression_profile,
    parse,
    required_history,
    validate,
)
from .factors.diversity import infer_mechanism


FACTOR_CORRELATION_PROTOCOL = "manual_factor_correlation_v1"
FACTOR_EXPRESSION_PROTOCOL = "multi_factor_expression_builder_v1"
MAX_CORRELATION_COMPONENTS = 12
MAX_EXPRESSION_COMPONENTS = 20
_CACHE_LIMIT = 8
_correlation_cache: dict[str, dict] = {}
_correlation_cache_lock = threading.Lock()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any, length: int = 32) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _as_date(value: Any, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} 必须是 YYYY-MM-DD") from exc


def _finite_float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} 必须是数值") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} 必须是有限数值")
    return result


def _normalise_components(
    raw_components: list[dict],
    market: str,
    *,
    maximum: int,
) -> list[dict]:
    if not 2 <= len(raw_components) <= maximum:
        raise ValueError(f"因子数量必须在 2 到 {maximum} 之间")
    fields = get_dsl_fields(market)
    output = []
    seen_keys: set[str] = set()
    for index, raw in enumerate(raw_components):
        expression = re.sub(r"\s+", "", str(raw.get("expression") or "").strip())
        error = validate(expression, fields)
        if error:
            raise ValueError(f"因子 {index + 1} 表达式非法: {error}")
        direction = int(raw.get("direction", 1))
        if direction not in {-1, 1}:
            raise ValueError(f"因子 {index + 1} direction 必须为 1 或 -1")
        key = str(raw.get("key") or f"F{index + 1:02d}").strip()[:64]
        if not key or key in seen_keys:
            raise ValueError("因子 key 必须非空且不能重复")
        seen_keys.add(key)
        output.append({
            "key": key,
            "name": str(raw.get("name") or key).strip()[:128],
            "expression": expression,
            "direction": direction,
            "weight": _finite_float(raw.get("weight", 1.0), f"因子 {index + 1} weight"),
            "mechanism": str(raw.get("mechanism") or infer_mechanism(expression)),
            "required_history": required_history(expression),
            "expression_hash": _fingerprint({"expression": expression, "direction": direction}, 16),
        })
    return output


def validate_correlation_spec(raw: dict) -> dict:
    market = str(raw.get("market") or "ashare")
    if market not in {"ashare", "us"}:
        raise ValueError("market 必须为 ashare 或 us")
    mode = str(raw.get("portfolio_mode") or ("long_only" if market == "ashare" else "long_short"))
    if mode not in {"long_only", "long_short"}:
        raise ValueError("portfolio_mode 必须为 long_only 或 long_short")
    if market == "ashare" and mode != "long_only":
        raise ValueError("A股相关性收益路径仅支持纯多")
    components = _normalise_components(
        list(raw.get("components") or []), market, maximum=MAX_CORRELATION_COMPONENTS
    )
    start = _as_date(raw.get("start", "2020-01-01"), "start")
    end = _as_date(raw.get("end", str(date.today())), "end")
    if start >= end:
        raise ValueError("start 必须早于 end")
    horizon = int(raw.get("horizon", 5))
    if horizon not in {1, 5, 20}:
        raise ValueError("horizon 仅支持 1/5/20")
    universe_n = int(raw.get("universe_n", 500))
    if not 100 <= universe_n <= 5000:
        raise ValueError("universe_n 必须在 100 到 5000 之间")
    top_fraction = _finite_float(raw.get("top_fraction", 0.20), "top_fraction")
    if not 0.05 <= top_fraction <= 0.50:
        raise ValueError("top_fraction 必须位于 [0.05, 0.50]")
    cost_bps = _finite_float(raw.get("cost_bps", 15.0), "cost_bps")
    if not 0 <= cost_bps <= 500:
        raise ValueError("cost_bps 必须位于 [0, 500]")
    threshold = _finite_float(raw.get("threshold", 0.80), "threshold")
    if not 0.50 <= threshold < 1.0:
        raise ValueError("threshold 必须位于 [0.50, 1.0)")
    borrow = _finite_float(
        raw.get("borrow_cost_bps_annual", 300.0 if mode == "long_short" else 0.0),
        "borrow_cost_bps_annual",
    )
    if borrow < 0:
        raise ValueError("borrow_cost_bps_annual 不能为负")
    if mode == "long_only":
        borrow = 0.0
    canonical = {
        "protocol": FACTOR_CORRELATION_PROTOCOL,
        "market": market,
        "portfolio_mode": mode,
        "panel_glob": str(raw.get("panel_glob") or default_panel_glob(market)),
        "components": components,
        "start": str(start),
        "end": str(end),
        "universe_n": universe_n,
        "horizon": horizon,
        "top_fraction": top_fraction,
        "cost_bps": cost_bps,
        "borrow_cost_bps_annual": borrow,
        "threshold": threshold,
    }
    canonical["request_hash"] = _fingerprint(canonical)
    return canonical


def _compact_expression(expression: str) -> tuple[str, ast.expr]:
    tree = ast.parse(expression, mode="eval").body
    return re.sub(r"\s+", "", ast.unparse(tree)), tree


def _top_call(node: ast.expr, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _negative_top_call(node: ast.expr, name: str) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and _top_call(node.operand, name)
    )


def _oriented_normalised_term(expression: str, direction: int, normalisation: str) -> tuple[str, str]:
    compact, node = _compact_expression(expression)
    if normalisation == "rank":
        if _top_call(node, "rank"):
            return (compact if direction == 1 else f"-{compact}"), "existing_rank"
        if _negative_top_call(node, "rank"):
            return (compact if direction == 1 else compact[1:]), "existing_signed_rank"
        return (f"rank({compact})" if direction == 1 else f"-rank({compact})"), "added_rank"
    if normalisation == "zscore":
        if _top_call(node, "zscore"):
            return (compact if direction == 1 else f"-{compact}"), "existing_zscore"
        if _negative_top_call(node, "zscore"):
            return (compact if direction == 1 else compact[1:]), "existing_signed_zscore"
        return (f"zscore({compact})" if direction == 1 else f"-zscore({compact})"), "added_zscore"
    if normalisation == "none":
        return (compact if direction == 1 else f"-({compact})"), "raw_scale"
    raise ValueError("normalization 必须是 rank、zscore 或 none")


def _join_signed_terms(terms: list[str], weights: np.ndarray | None = None) -> str:
    rendered = []
    for index, term in enumerate(terms):
        negative = term.startswith("-")
        unsigned = term[1:] if negative else term
        if weights is not None:
            unsigned = f"{float(weights[index]):.10g}*({unsigned})"
        if index == 0:
            rendered.append(f"-{unsigned}" if negative else unsigned)
        else:
            rendered.append(f"-{unsigned}" if negative else f"+{unsigned}")
    return "".join(rendered)


def build_combination_expression(raw: dict) -> dict:
    market = str(raw.get("market") or "ashare")
    if market not in {"ashare", "us"}:
        raise ValueError("market 必须为 ashare 或 us")
    components = _normalise_components(
        list(raw.get("components") or []), market, maximum=MAX_EXPRESSION_COMPONENTS
    )
    normalisation = str(raw.get("normalization") or "rank")
    if normalisation not in {"rank", "zscore", "none"}:
        raise ValueError("normalization 必须是 rank、zscore 或 none")
    weights = np.asarray([row["weight"] for row in components], dtype=np.float64)
    if np.any(weights <= 0):
        raise ValueError("所有参与组合的权重必须大于0")
    weights /= weights.sum()
    terms = []
    transform_notes = []
    for component, weight in zip(components, weights):
        term, transform = _oriented_normalised_term(
            component["expression"], component["direction"], normalisation
        )
        terms.append(term)
        transform_notes.append({
            "key": component["key"],
            "direction": component["direction"],
            "normalized_weight": round(float(weight), 10),
            "transform": transform,
            "term": term,
        })
    equal_weight = bool(np.max(np.abs(weights - weights[0])) <= 1e-12)
    omit_common_scale = bool(raw.get("omit_common_scale", True))
    if equal_weight and omit_common_scale:
        expression = _join_signed_terms(terms)
        scale_note = f"省略共同系数 1/{len(terms)}；正比例缩放不改变选股排序"
    else:
        expression = _join_signed_terms(terms, weights)
        scale_note = "保留归一化权重，权重和为1"
    error = validate(expression, get_dsl_fields(market))
    if error:
        raise ValueError(f"生成后的组合表达式非法: {error}")
    profile = expression_profile(expression)
    warnings = []
    if normalisation == "none":
        warnings.append("未做截面标准化，不同量纲会形成隐性权重")
    if len(expression) > 500:
        warnings.append("表达式超过旧版500字符限制；当前服务已支持到4000字符")
    return {
        "protocol": FACTOR_EXPRESSION_PROTOCOL,
        "market": market,
        "normalization": normalisation,
        "direction": 1,
        "component_count": len(components),
        "expression": expression,
        "length": len(expression),
        "max_length": MAX_EXPRESSION_LENGTH,
        "ast_node_limit": MAX_EXPRESSION_AST_NODES,
        "required_history": required_history(expression),
        "fields": profile["fields"],
        "operators": profile["operators"],
        "weights": [round(float(value), 10) for value in weights],
        "equal_weight": equal_weight,
        "scale_note": scale_note,
        "warnings": warnings,
        "transformations": transform_notes,
        "component_snapshot": components,
        "snapshot_hash": _fingerprint({
            "market": market,
            "normalization": normalisation,
            "components": components,
            "weights": weights.tolist(),
        }),
    }


def _safe_corrcoef(matrix: np.ndarray, *, rowvar: bool = False) -> np.ndarray:
    if matrix.shape[0] < 3:
        return np.eye(matrix.shape[1] if not rowvar else matrix.shape[0])
    value = np.corrcoef(matrix, rowvar=rowvar)
    value = np.atleast_2d(np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0))
    np.fill_diagonal(value, 1.0)
    return value


def _average_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _analyse_materialized_frame(frame: pl.DataFrame, spec: dict) -> dict:
    component_count = len(spec["components"])
    score_columns = [f"component_{index}" for index in range(component_count)]
    forward = f"fwd_{spec['horizon']}"
    dates = frame["trade_date"].unique().sort().to_list()
    signal_dates = dates[:: spec["horizon"]]
    signal_set = set(signal_dates)
    grouped = frame.filter(pl.col("trade_date").is_in(signal_dates)).partition_by(
        "trade_date", maintain_order=True
    )
    signal_matrices = []
    ic_paths = []
    rank_ic_paths = []
    return_paths = []
    turnover_paths = []
    used_dates = []
    previous: list[dict[str, float]] = [dict() for _ in range(component_count)]
    for group in grouped:
        scores = group.select(score_columns).to_numpy().astype(np.float64)
        returns = group[forward].to_numpy().astype(np.float64)
        symbols = [str(value) for value in group["ts_code"].to_list()]
        finite = np.isfinite(returns) & np.all(np.isfinite(scores), axis=1)
        scores, returns = scores[finite], returns[finite]
        symbols = [value for value, keep in zip(symbols, finite) if keep]
        if len(returns) < 50:
            continue
        signal_matrices.append(_safe_corrcoef(scores, rowvar=False))
        return_ranks = _average_rank(returns)
        ic_row, rank_ic_row, period_returns, period_turnover = [], [], [], []
        n_select = max(1, int(math.floor(len(returns) * spec["top_fraction"])))
        for index in range(component_count):
            score = scores[:, index]
            ic_row.append(float(_safe_corrcoef(np.column_stack([score, returns]))[0, 1]))
            rank_ic_row.append(float(_safe_corrcoef(np.column_stack([score, return_ranks]))[0, 1]))
            top = np.argpartition(score, -n_select)[-n_select:]
            current = {symbols[item]: 1.0 / n_select for item in top}
            gross = float(returns[top].mean())
            benchmark = float(returns.mean()) if spec["portfolio_mode"] == "long_only" else 0.0
            if spec["portfolio_mode"] == "long_short":
                bottom = np.argpartition(score, n_select - 1)[:n_select]
                for item in bottom:
                    current[symbols[item]] = -1.0 / n_select
                gross -= float(returns[bottom].mean())
            keys = set(previous[index]) | set(current)
            turnover = sum(abs(current.get(key, 0.0) - previous[index].get(key, 0.0)) for key in keys)
            borrow = (
                spec["borrow_cost_bps_annual"] / 10_000.0 * spec["horizon"] / 252.0
                if spec["portfolio_mode"] == "long_short" else 0.0
            )
            net = gross - benchmark - turnover * spec["cost_bps"] / 10_000.0 - borrow
            period_returns.append(net)
            period_turnover.append(turnover)
            previous[index] = current
        ic_paths.append(ic_row)
        rank_ic_paths.append(rank_ic_row)
        return_paths.append(period_returns)
        turnover_paths.append(period_turnover)
        used_dates.append(str(group["trade_date"][0]))
    if len(used_dates) < 8:
        raise ValueError("有效共同收益期不足8个，请扩大区间或减少缺失严重的因子")
    signal = np.mean(np.stack(signal_matrices), axis=0)
    np.fill_diagonal(signal, 1.0)
    ic_array = np.asarray(ic_paths)
    rank_ic_array = np.asarray(rank_ic_paths)
    returns_array = np.asarray(return_paths)
    turnover_array = np.asarray(turnover_paths)
    ic_correlation = _safe_corrcoef(rank_ic_array, rowvar=False)
    return_correlation = _safe_corrcoef(returns_array, rowvar=False)
    periods_per_year = 252.0 / spec["horizon"]
    stats = []
    for index, component in enumerate(spec["components"]):
        path = returns_array[:, index]
        path_std = float(path.std(ddof=1))
        ic_std = float(ic_array[:, index].std(ddof=1))
        rank_std = float(rank_ic_array[:, index].std(ddof=1))
        stats.append({
            "key": component["key"],
            "name": component["name"],
            "direction": component["direction"],
            "ann_return": round(float(path.mean() * periods_per_year), 8),
            "sharpe": round(float(path.mean() / path_std * math.sqrt(periods_per_year)) if path_std > 1e-12 else 0.0, 6),
            "avg_turnover": round(float(turnover_array[:, index].mean()), 8),
            "ic_mean": round(float(ic_array[:, index].mean()), 8),
            "icir": round(float(ic_array[:, index].mean() / ic_std * math.sqrt(periods_per_year)) if ic_std > 1e-12 else 0.0, 6),
            "rank_ic_mean": round(float(rank_ic_array[:, index].mean()), 8),
            "rank_icir": round(float(rank_ic_array[:, index].mean() / rank_std * math.sqrt(periods_per_year)) if rank_std > 1e-12 else 0.0, 6),
            "periods": len(used_dates),
        })
    return {
        "dates": used_dates,
        "signal_rank_correlation": signal.round(8).tolist(),
        "rank_ic_path_correlation": ic_correlation.round(8).tolist(),
        "portfolio_return_correlation": return_correlation.round(8).tolist(),
        "factor_stats": stats,
        "signal_dates_considered": len(signal_set),
    }


def run_factor_correlation(raw: dict) -> dict:
    spec = validate_correlation_spec(raw)
    cache_key = spec["request_hash"]
    with _correlation_cache_lock:
        cached = _correlation_cache.get(cache_key)
    if cached is not None:
        return {**cached, "cache_hit": True}
    started = time.perf_counter()
    fields = get_dsl_fields(spec["market"])
    store = PanelStore.get(spec["panel_glob"], spec["market"], factor_fields=fields)
    panel, dates, identity, generation = store.read_snapshot()
    requested_start, requested_end = _as_date(spec["start"], "start"), _as_date(spec["end"], "end")
    effective_start = max(requested_start, min(dates))
    effective_end = min(requested_end, max(dates))
    if effective_start >= effective_end:
        raise ValueError("面板不覆盖所选区间")
    max_history = max(row["required_history"] for row in spec["components"])
    first_index = dates.index(next(value for value in dates if value >= effective_start))
    history_start = dates[max(0, first_index - max_history - 2)]
    lazy = panel.lazy().filter(pl.col("trade_date").is_between(history_start, effective_end))
    score_columns = []
    for index, component in enumerate(spec["components"]):
        raw_column = f"_factor_tool_raw_{index}"
        oriented = f"_factor_tool_oriented_{index}"
        score = f"component_{index}"
        lazy = parse(component["expression"], fields).apply(lazy, alias=raw_column)
        lazy = lazy.with_columns(
            pl.when(pl.col("univ_rank") <= spec["universe_n"])
            .then(pl.col(raw_column) * component["direction"])
            .otherwise(None)
            .alias(oriented)
        ).with_columns(
            (
                pl.col(oriented).rank(method="average").over("trade_date")
                / (pl.col(oriented).count().over("trade_date") + 1e-12)
            ).alias(score)
        )
        score_columns.append(score)
    forward = f"fwd_{spec['horizon']}"
    frame = (
        lazy.select("trade_date", "ts_code", "univ_rank", forward, *score_columns)
        .filter(
            pl.col("trade_date").is_between(effective_start, effective_end)
            & (pl.col("univ_rank") <= spec["universe_n"])
            & pl.col(forward).is_finite()
            & pl.all_horizontal(pl.col(column).is_finite() for column in score_columns)
        )
        .sort("trade_date", "ts_code")
        .collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False))
    )
    analysis = _analyse_materialized_frame(frame, spec)
    labels = [row["key"] for row in spec["components"]]
    high_pairs = []
    matrices = {
        "signal_rank_correlation": analysis["signal_rank_correlation"],
        "rank_ic_path_correlation": analysis["rank_ic_path_correlation"],
        "portfolio_return_correlation": analysis["portfolio_return_correlation"],
    }
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            values = {key: float(matrix[left][right]) for key, matrix in matrices.items()}
            max_abs = max(abs(value) for value in values.values())
            if max_abs < spec["threshold"]:
                continue
            high_pairs.append({
                "left": labels[left],
                "right": labels[right],
                **values,
                "max_abs_correlation": round(max_abs, 8),
                "classification": (
                    "same_return_source"
                    if abs(values["portfolio_return_correlation"]) >= spec["threshold"]
                    else "signal_overlap"
                    if abs(values["signal_rank_correlation"]) >= spec["threshold"]
                    else "ic_regime_overlap"
                ),
            })
    high_pairs.sort(key=lambda row: row["max_abs_correlation"], reverse=True)
    result = {
        "protocol": FACTOR_CORRELATION_PROTOCOL,
        "request_hash": cache_key,
        "cache_hit": False,
        "market": spec["market"],
        "portfolio_mode": spec["portfolio_mode"],
        "window": {"requested_start": spec["start"], "requested_end": spec["end"], "start": str(effective_start), "end": str(effective_end)},
        "universe_n": spec["universe_n"],
        "horizon": spec["horizon"],
        "top_fraction": spec["top_fraction"],
        "cost_bps": spec["cost_bps"],
        "threshold": spec["threshold"],
        "labels": labels,
        "components": spec["components"],
        "matrices": matrices,
        "high_correlation_pairs": high_pairs,
        "factor_stats": analysis["factor_stats"],
        "common_periods": len(analysis["dates"]),
        "return_dates": analysis["dates"],
        "panel": {
            "identity": identity,
            "generation": generation,
            "glob": spec["panel_glob"],
            "history_start": str(history_start),
            "rows": frame.height,
            "summary": store.summary(),
        },
        "interpretation": {
            "signal_rank_correlation": "共同调仓截面中，方向化因子百分位相关的时间均值",
            "rank_ic_path_correlation": "各因子Rank-IC时间序列之间的相关性",
            "portfolio_return_correlation": "统一选股比例、成本和模式下，单因子收益路径相关性",
            "primary_deduplication_matrix": "portfolio_return_correlation",
        },
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with _correlation_cache_lock:
        if len(_correlation_cache) >= _CACHE_LIMIT:
            _correlation_cache.pop(next(iter(_correlation_cache)))
        _correlation_cache[cache_key] = result
    return result


def factor_tool_capabilities(market: str) -> dict:
    if market not in {"ashare", "us"}:
        raise ValueError("market 必须为 ashare 或 us")
    return {
        "correlation_protocol": FACTOR_CORRELATION_PROTOCOL,
        "expression_protocol": FACTOR_EXPRESSION_PROTOCOL,
        "market": market,
        "dsl_fields": get_dsl_fields(market),
        "limits": {
            "correlation_components": MAX_CORRELATION_COMPONENTS,
            "expression_components": MAX_EXPRESSION_COMPONENTS,
            "expression_characters": MAX_EXPRESSION_LENGTH,
            "expression_ast_nodes": MAX_EXPRESSION_AST_NODES,
        },
        "supported_horizons": [1, 5, 20],
        "normalizations": ["rank", "zscore", "none"],
        "default_panel_glob": default_panel_glob(market),
    }
