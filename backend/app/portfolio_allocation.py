"""Robust, point-in-time portfolio allocation for screener snapshots.

The allocator deliberately avoids fitting expected returns.  A cross-sectional
factor score is only used as a conservative *risk-budget tilt* while adjusted
close history controls the capital allocation.  This keeps the result useful as
a research sizing proposal without pretending that one ranking snapshot is a
calibrated return forecast.
"""

from __future__ import annotations

import math
import time
from datetime import date
from typing import Sequence

import numpy as np
import polars as pl


ALLOCATION_PROTOCOL = "screener_robust_risk_budget_v1"
ALLOCATION_METHODS = {
    "robust_risk_budget",
    "inverse_volatility",
    "equal_weight",
}


def _normalise(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if not math.isfinite(total) or total <= 0:
        return np.full(len(values), 1.0 / len(values), dtype=np.float64)
    return values / total


def _robust_zscore(values: np.ndarray) -> np.ndarray:
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if scale <= 1e-12:
        scale = float(np.std(values))
    if scale <= 1e-12:
        return np.zeros_like(values)
    return np.clip((values - center) / scale, -2.0, 2.0)


def _risk_budget_weights(
    covariance: np.ndarray,
    budgets: np.ndarray,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 2_000,
) -> tuple[np.ndarray, int, bool]:
    """Solve long-only risk budgeting by cyclic coordinate descent.

    The unconstrained positive solution minimises
    ``0.5 * x' Σ x - sum(b_i log(x_i))``.  Normalising ``x`` then produces a
    portfolio whose volatility contributions follow ``budgets``.
    """

    diagonal = np.maximum(np.diag(covariance), 1e-12)
    x = np.sqrt(np.maximum(budgets, 1e-12) / diagonal)
    x = np.maximum(x, 1e-12)
    converged = False
    iteration = 0
    for iteration in range(1, max_iterations + 1):
        previous = x.copy()
        for index in range(len(x)):
            cross = float(covariance[index] @ x - diagonal[index] * x[index])
            discriminant = max(
                0.0,
                cross * cross + 4.0 * diagonal[index] * budgets[index],
            )
            x[index] = max(
                1e-12,
                (-cross + math.sqrt(discriminant)) / (2.0 * diagonal[index]),
            )
        relative_change = float(
            np.max(np.abs(x - previous) / np.maximum(np.abs(previous), 1e-12))
        )
        if relative_change <= tolerance:
            converged = True
            break
    return _normalise(x), iteration, converged


def _cap_weights(weights: np.ndarray, requested_cap: float) -> tuple[np.ndarray, float]:
    """Apply a feasible long-only cap while preserving relative weights."""

    count = len(weights)
    effective_cap = max(float(requested_cap), 1.0 / count)
    effective_cap = min(1.0, effective_cap)
    output = np.zeros(count, dtype=np.float64)
    remaining = np.ones(count, dtype=bool)
    remaining_mass = 1.0
    base = np.maximum(weights.astype(np.float64), 0.0)

    while np.any(remaining):
        indices = np.flatnonzero(remaining)
        relative = _normalise(base[indices])
        proposed = relative * remaining_mass
        over = proposed > effective_cap + 1e-12
        if not np.any(over):
            output[indices] = proposed
            break
        capped_indices = indices[over]
        output[capped_indices] = effective_cap
        remaining[capped_indices] = False
        remaining_mass = max(0.0, 1.0 - float(np.sum(output)))

    return _normalise(output), effective_cap


def _history_matrix(
    df: pl.DataFrame,
    symbols: Sequence[str],
    target_date: date,
    lookback: int,
) -> tuple[np.ndarray, list[date], dict[str, int], dict[str, float]]:
    available_dates = (
        df.lazy()
        .filter(pl.col("trade_date") <= target_date)
        .select("trade_date")
        .unique()
        .sort("trade_date")
        .tail(lookback + 1)
        .collect()["trade_date"]
        .to_list()
    )
    if len(available_dates) < 2:
        raise ValueError("截面日前没有足够行情用于配权")

    history = (
        df.lazy()
        .filter(
            pl.col("trade_date").is_in(available_dates)
            & pl.col("ts_code").is_in(list(symbols))
        )
        .select("trade_date", "ts_code", "close")
        .collect()
    )
    price_maps: dict[str, dict[date, float]] = {symbol: {} for symbol in symbols}
    for row in history.iter_rows(named=True):
        value = row.get("close")
        if value is None or not math.isfinite(float(value)) or float(value) <= 0:
            continue
        price_maps[str(row["ts_code"])][row["trade_date"]] = float(value)

    missing_symbols = [symbol for symbol in symbols if not price_maps[symbol]]
    if missing_symbols:
        raise ValueError(f"选中证券缺少历史收盘价: {', '.join(missing_symbols)}")

    prices = np.full((len(available_dates), len(symbols)), np.nan, dtype=np.float64)
    missing_counts: dict[str, int] = {}
    last_prices: dict[str, float] = {}
    for column, symbol in enumerate(symbols):
        last = math.nan
        observed = 0
        for row_index, trading_date in enumerate(available_dates):
            current = price_maps[symbol].get(trading_date)
            if current is not None:
                last = current
                observed += 1
            if math.isfinite(last):
                prices[row_index, column] = last
        missing_counts[symbol] = len(available_dates) - observed
        valid = prices[:, column][np.isfinite(prices[:, column])]
        last_prices[symbol] = float(valid[-1])

    previous = prices[:-1]
    current = prices[1:]
    valid_rows = np.all(np.isfinite(previous) & np.isfinite(current), axis=1)
    returns = current[valid_rows] / previous[valid_rows] - 1.0
    return_dates = [
        trading_date
        for trading_date, valid in zip(available_dates[1:], valid_rows)
        if bool(valid)
    ]
    return returns, return_dates, missing_counts, last_prices


def build_purchase_allocation(
    *,
    df: pl.DataFrame,
    target_date: date,
    selected: Sequence[dict],
    lookback: int = 120,
    method: str = "robust_risk_budget",
    max_weight: float = 0.35,
    score_tilt: float = 0.35,
) -> dict:
    """Build a deterministic long-only purchase allocation for selected rows."""

    started = time.perf_counter()
    if method not in ALLOCATION_METHODS:
        raise ValueError(f"未知配权方法: {method}")
    if not 30 <= int(lookback) <= 504:
        raise ValueError("lookback 必须在 30 到 504 个交易日之间")
    if not 0.05 <= float(max_weight) <= 1.0:
        raise ValueError("max_weight 必须在 0.05 到 1.0 之间")
    if not 0.0 <= float(score_tilt) <= 1.0:
        raise ValueError("score_tilt 必须在 0 到 1 之间")
    if not 1 <= len(selected) <= 100:
        raise ValueError("必须选择 1 到 100 只证券")

    symbols = [str(row.get("ts_code") or "").strip() for row in selected]
    if any(not symbol for symbol in symbols):
        raise ValueError("选中证券代码不能为空")
    if len(set(symbols)) != len(symbols):
        raise ValueError("选中证券不能重复")

    returns, return_dates, missing_counts, last_prices = _history_matrix(
        df,
        symbols,
        target_date,
        int(lookback),
    )
    minimum_observations = min(60, max(30, int(lookback) // 2))
    if len(return_dates) < minimum_observations:
        raise ValueError(
            f"共同有效收益样本仅 {len(return_dates)} 日，至少需要 "
            f"{minimum_observations} 日；请减少股票或缩短窗口"
        )

    lower = np.quantile(returns, 0.01, axis=0)
    upper = np.quantile(returns, 0.99, axis=0)
    clipped = np.clip(returns, lower, upper)
    half_life = min(63.0, max(20.0, int(lookback) * 0.35))
    ages = np.arange(len(clipped) - 1, -1, -1, dtype=np.float64)
    observation_weights = np.exp(math.log(0.5) * ages / half_life)
    observation_weights = _normalise(observation_weights)
    mean = np.sum(clipped * observation_weights[:, None], axis=0)
    centered = clipped - mean
    denominator = max(1e-8, 1.0 - float(np.sum(observation_weights**2)))
    sample_covariance = (
        (centered * observation_weights[:, None]).T @ centered / denominator
    )

    count = len(symbols)
    shrinkage = float(np.clip(count / (len(clipped) + count), 0.15, 0.50))
    covariance = (
        (1.0 - shrinkage) * sample_covariance
        + shrinkage * np.diag(np.diag(sample_covariance))
    )
    median_variance = max(float(np.median(np.diag(covariance))), 1e-12)
    eigen_floor = median_variance * 1e-8
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if minimum_eigenvalue < eigen_floor:
        covariance += np.eye(count) * (eigen_floor - minimum_eigenvalue)

    annualised_volatility = np.sqrt(np.maximum(np.diag(covariance), 0.0) * 252.0)
    conviction = np.asarray([
        100.0 - float(row.get("score") or 0.0)
        if row.get("side") == "bottom"
        else float(row.get("score") or 0.0)
        for row in selected
    ], dtype=np.float64)
    conviction = np.clip(conviction, 0.0, 100.0)
    if method == "equal_weight":
        target_budgets = np.full(count, 1.0 / count, dtype=np.float64)
    else:
        target_budgets = _normalise(
            np.exp(float(score_tilt) * _robust_zscore(conviction))
        )

    iterations = 0
    converged = True
    if method == "robust_risk_budget" and count > 1:
        raw_weights, iterations, converged = _risk_budget_weights(
            covariance,
            target_budgets,
        )
    elif method == "inverse_volatility":
        daily_volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        raw_weights = _normalise(target_budgets / daily_volatility)
    else:
        raw_weights = np.full(count, 1.0 / count, dtype=np.float64)

    weights, effective_cap = _cap_weights(raw_weights, float(max_weight))
    portfolio_variance = max(float(weights @ covariance @ weights), 0.0)
    portfolio_daily_volatility = math.sqrt(portfolio_variance)
    portfolio_annualised_volatility = portfolio_daily_volatility * math.sqrt(252.0)
    equal_weights = np.full(count, 1.0 / count, dtype=np.float64)
    equal_annualised_volatility = math.sqrt(
        max(float(equal_weights @ covariance @ equal_weights), 0.0) * 252.0
    )
    if portfolio_variance > 1e-18:
        risk_contribution = weights * (covariance @ weights) / portfolio_variance
    else:
        risk_contribution = np.full(count, 1.0 / count, dtype=np.float64)
    diversification_ratio = (
        float(np.sum(weights * np.sqrt(np.maximum(np.diag(covariance), 0.0))))
        / max(portfolio_daily_volatility, 1e-12)
    )
    standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
    correlation = covariance / np.outer(standard_deviation, standard_deviation)
    np.fill_diagonal(correlation, 1.0)
    if count > 1:
        upper_triangle = np.triu_indices(count, 1)
        pair_correlations = correlation[upper_triangle]
        maximum_pair_position = int(np.argmax(pair_correlations))
        pair_left = int(upper_triangle[0][maximum_pair_position])
        pair_right = int(upper_triangle[1][maximum_pair_position])
        maximum_pair = {
            "left": symbols[pair_left],
            "right": symbols[pair_right],
            "correlation": round(float(pair_correlations[maximum_pair_position]), 6),
        }
        average_correlation = float(np.mean(pair_correlations))
    else:
        maximum_pair = None
        average_correlation = 0.0

    warnings = [
        "权重是研究期的风险配置建议，不是交易批准，也不包含预期收益保证。",
        "未使用未来数据；协方差样本截止于选股截面日。",
        "尚未加入行业、账户持仓、税务、最小交易单位和实时成交容量约束。",
    ]
    if effective_cap > float(max_weight) + 1e-12:
        warnings.append(
            f"所设单股上限 {float(max_weight):.1%} 对 {count} 只股票不可行，"
            f"已自动调整为 {effective_cap:.1%}。"
        )
    missing_total = sum(missing_counts.values())
    if missing_total:
        warnings.append(
            f"窗口内共有 {missing_total} 个缺失收盘观察，按停牌/缺报期间价格不变前向填充。"
        )
    if any(row.get("side") == "bottom" for row in selected):
        warnings.append(
            "选中集合含尾部股票；本结果仍按纯多头购买配比计算，尾部信号不代表系统推荐买入。"
        )
    if not converged:
        warnings.append("风险预算迭代达到上限，结果已归一化但风险贡献可能偏离目标。")

    allocations = []
    for index, row in enumerate(selected):
        allocations.append({
            "ts_code": symbols[index],
            "name": str(row.get("name") or ""),
            "side": str(row.get("side") or "top"),
            "side_rank": int(row.get("side_rank") or row.get("rank") or index + 1),
            "score": round(float(row.get("score") or 0.0), 6),
            "conviction": round(float(conviction[index]), 6),
            "purchase_weight": round(float(weights[index]), 8),
            "target_risk_budget": round(float(target_budgets[index]), 8),
            "risk_contribution": round(float(risk_contribution[index]), 8),
            "annualised_volatility": round(float(annualised_volatility[index]), 8),
            "last_adjusted_close": round(last_prices[symbols[index]], 6),
            "missing_observations": int(missing_counts[symbols[index]]),
        })
    allocations.sort(key=lambda row: row["purchase_weight"], reverse=True)

    condition_number = float(np.linalg.cond(covariance))
    return {
        "protocol": ALLOCATION_PROTOCOL,
        "method": method,
        "target_date": str(target_date),
        "selection_count": count,
        "allocations": allocations,
        "parameters": {
            "lookback": int(lookback),
            "half_life": round(half_life, 3),
            "winsor_limits": [0.01, 0.99],
            "score_tilt": float(score_tilt),
            "requested_max_weight": float(max_weight),
            "effective_max_weight": round(effective_cap, 8),
            "covariance_shrinkage": round(shrinkage, 8),
        },
        "diagnostics": {
            "sample_start": str(return_dates[0]),
            "sample_end": str(return_dates[-1]),
            "observations": len(return_dates),
            "portfolio_annualised_volatility": round(portfolio_annualised_volatility, 8),
            "equal_weight_annualised_volatility": round(equal_annualised_volatility, 8),
            "diversification_ratio": round(diversification_ratio, 8),
            "effective_holdings": round(1.0 / float(np.sum(weights**2)), 8),
            "average_correlation": round(average_correlation, 8),
            "maximum_correlation_pair": maximum_pair,
            "covariance_condition_number": round(condition_number, 6),
            "risk_budget_max_error": round(
                float(np.max(np.abs(risk_contribution - target_budgets))),
                8,
            ),
            "iterations": iterations,
            "converged": converged,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        },
        "warnings": warnings,
    }
