"""Time-slice stability and causal signal diagnostics for event backtests.

Portfolio performance is derived exclusively from the event ledger's daily
returns.  IC diagnostics are deliberately separate: a signal observed at the
close of t is compared with the adjusted-open return from t+1 to t+1+h.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl


STABILITY_PROTOCOL = "backtest_time_slice_stability_v1"
SIGNAL_DIAGNOSTICS_PROTOCOL = "causal_forward_open_ic_v1"
MONTE_CARLO_PROTOCOL = "moving_block_bootstrap_v1"


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _as_date(value: Any) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _round(value: float | None, digits: int = 8) -> float | None:
    return None if value is None or not math.isfinite(value) else round(value, digits)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    return mean, math.sqrt(max(0.0, variance))


def _shift_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    year, month_zero = divmod(month_index, 12)
    month = month_zero + 1
    month_days = (
        date(year + (month == 12), 1 if month == 12 else month + 1, 1)
        - date(year, month, 1)
    ).days
    return date(year, month, min(value.day, month_days))


def _period_performance(
    dates: list[date],
    returns: list[float],
    turnovers: list[float],
    start_index: int,
    end_index: int,
) -> dict[str, Any]:
    period_returns = returns[start_index : end_index + 1]
    wealth = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for daily_return in period_returns:
        wealth *= 1.0 + daily_return
        peak = max(peak, wealth)
        max_drawdown = max(max_drawdown, 1.0 - wealth / peak if peak > 0 else 1.0)
    sessions = len(period_returns)
    mean_return, std_return = _mean_std(period_returns)
    total_return = wealth - 1.0
    cagr = wealth ** (252.0 / sessions) - 1.0 if sessions and wealth > 0 else -1.0
    return {
        "start": dates[start_index].isoformat(),
        "end": dates[end_index].isoformat(),
        "sessions": sessions,
        "total_return": _round(total_return, 8),
        "cagr": _round(cagr, 8),
        "sharpe": _round(
            mean_return / std_return * math.sqrt(252.0)
            if std_return > 1e-12 else 0.0,
            6,
        ),
        "ann_vol": _round(std_return * math.sqrt(252.0), 8),
        "max_drawdown": _round(max_drawdown, 8),
        "positive_day_rate": _round(
            sum(value > 0 for value in period_returns) / sessions if sessions else 0.0,
            6,
        ),
        "avg_daily_turnover": _round(
            sum(turnovers[start_index : end_index + 1]) / sessions
            if sessions else 0.0,
            8,
        ),
    }


def _attach_sleeve_contributions(
    period: dict[str, Any],
    *,
    start_index: int,
    end_index: int,
    sleeves: list[dict[str, Any]],
    total_initial_capital: float,
) -> None:
    rows: list[dict[str, Any]] = []
    for sleeve in sleeves:
        values = [float(value) for value in sleeve.get("nlv", [])]
        if len(values) <= end_index:
            continue
        start_nlv = (
            float(sleeve["initial_capital"])
            if start_index == 0
            else values[start_index - 1]
        )
        end_nlv = values[end_index]
        contribution = (end_nlv - start_nlv) / total_initial_capital
        rows.append({
            "factor_id": sleeve["factor_id"],
            "name": sleeve["name"],
            "normalized_weight": _round(float(sleeve["normalized_weight"]), 10),
            "start_nlv": _round(start_nlv, 6),
            "end_nlv": _round(end_nlv, 6),
            "net_pnl": _round(end_nlv - start_nlv, 6),
            "return_contribution": _round(contribution, 10),
            "standalone_return": _round(
                end_nlv / start_nlv - 1.0 if start_nlv > 0 else -1.0,
                10,
            ),
        })
    period["sleeves"] = rows
    period["contribution_reconciliation_error"] = _round(
        sum(float(row["return_contribution"]) for row in rows)
        - (
            float(period["total_return"])
            * sum(
                (
                    float(sleeve["initial_capital"])
                    if start_index == 0
                    else float(sleeve["nlv"][start_index - 1])
                )
                for sleeve in sleeves
            )
            / total_initial_capital
        ),
        10,
    )


def _reversal_diagnostics(annual: list[dict[str, Any]]) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    factor_ids = sorted({
        row["factor_id"] for period in annual for row in period.get("sleeves", [])
    })
    for factor_id in factor_ids:
        prior: tuple[dict[str, Any], dict[str, Any]] | None = None
        for period in annual:
            row = next(
                (item for item in period.get("sleeves", []) if item["factor_id"] == factor_id),
                None,
            )
            if row is None or int(period["sessions"]) < 60:
                continue
            current_return = float(row["standalone_return"])
            if prior is not None:
                prior_period, prior_row = prior
                prior_return = float(prior_row["standalone_return"])
                if prior_return * current_return < 0:
                    events.append({
                        "factor_id": factor_id,
                        "name": row["name"],
                        "from_period": prior_period["period"],
                        "to_period": period["period"],
                        "from_standalone_return": prior_return,
                        "to_standalone_return": current_return,
                        "from_contribution": prior_row["return_contribution"],
                        "to_contribution": row["return_contribution"],
                        "material": (
                            abs(float(prior_row["return_contribution"])) >= 0.005
                            and abs(float(row["return_contribution"])) >= 0.005
                        ),
                    })
            prior = (period, row)
    return {
        "definition": (
            "相邻且至少60个交易日的年度切片中，sleeve独立收益正负号翻转；"
            "前后对总初始资本的贡献绝对值均不低于0.5%时标记为material"
        ),
        "events": events,
        "reversal_count": len(events),
        "material_reversal_count": sum(bool(row["material"]) for row in events),
        "affected_factor_count": len({row["factor_id"] for row in events}),
    }


def analyze_return_stability(
    daily_steps: list[dict[str, Any]],
    *,
    total_initial_capital: float,
    sleeves: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Calculate annual and month-end rolling performance from ledger returns."""
    if not daily_steps:
        return {"protocol": STABILITY_PROTOCOL, "status": "INSUFFICIENT_DATA"}
    dates = [_as_date(row["trade_date"]) for row in daily_steps]
    returns = [float(row.get("daily_return") or 0.0) for row in daily_steps]
    turnovers = [float(row.get("turnover") or 0.0) for row in daily_steps]
    sleeves = sleeves or []

    annual: list[dict[str, Any]] = []
    for year in sorted({value.year for value in dates}):
        indexes = [index for index, value in enumerate(dates) if value.year == year]
        period = _period_performance(dates, returns, turnovers, indexes[0], indexes[-1])
        period.update({
            "period": str(year),
            "is_ytd": indexes[-1] == len(dates) - 1 and dates[-1].month < 12,
        })
        if sleeves:
            _attach_sleeve_contributions(
                period,
                start_index=indexes[0],
                end_index=indexes[-1],
                sleeves=sleeves,
                total_initial_capital=total_initial_capital,
            )
        annual.append(period)

    month_ends: list[int] = []
    for index, value in enumerate(dates):
        if index == len(dates) - 1 or dates[index + 1].month != value.month:
            month_ends.append(index)
    rolling: dict[str, list[dict[str, Any]]] = {"12m": [], "24m": []}
    for months, minimum_sessions in ((12, 180), (24, 360)):
        for end_index in month_ends:
            target = _shift_months(dates[end_index], -months) + timedelta(days=1)
            start_index = next(
                (index for index, value in enumerate(dates[: end_index + 1]) if value >= target),
                end_index,
            )
            if end_index - start_index + 1 < minimum_sessions:
                continue
            period = _period_performance(
                dates, returns, turnovers, start_index, end_index
            )
            period["period"] = f"{months}m@{dates[end_index].isoformat()}"
            if sleeves:
                _attach_sleeve_contributions(
                    period,
                    start_index=start_index,
                    end_index=end_index,
                    sleeves=sleeves,
                    total_initial_capital=total_initial_capital,
                )
            rolling[f"{months}m"].append(period)

    return {
        "protocol": STABILITY_PROTOCOL,
        "status": "OK",
        "return_source": "step_event_v2_daily_ledger",
        "annual": annual,
        "rolling": rolling,
        "latest_rolling": {
            key: values[-1] if values else None for key, values in rolling.items()
        },
        "regime_reversal": _reversal_diagnostics(annual),
    }


def _ic_summary(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    if len(rows) < 2:
        return {"status": "INSUFFICIENT_DATA", "n_dates": len(rows)}
    ic_values = [float(row["ic"]) for row in rows]
    rank_values = [float(row["rank_ic"]) for row in rows]
    ic_mean, ic_std = _mean_std(ic_values)
    rank_mean, rank_std = _mean_std(rank_values)
    annualizer = math.sqrt(252.0 / max(1, horizon))
    return {
        "status": "OK",
        "start": rows[0]["trade_date"],
        "end": rows[-1]["trade_date"],
        "n_dates": len(rows),
        "mean_cross_section_n": _round(
            sum(int(row["n"]) for row in rows) / len(rows), 2
        ),
        "ic_mean": _round(ic_mean, 8),
        "ic_std": _round(ic_std, 8),
        "icir": _round(ic_mean / ic_std * annualizer if ic_std > 1e-12 else 0.0, 6),
        "ic_positive_rate": _round(sum(value > 0 for value in ic_values) / len(rows), 6),
        "rank_ic_mean": _round(rank_mean, 8),
        "rank_ic_std": _round(rank_std, 8),
        "rank_icir": _round(
            rank_mean / rank_std * annualizer if rank_std > 1e-12 else 0.0,
            6,
        ),
        "rank_ic_positive_rate": _round(
            sum(value > 0 for value in rank_values) / len(rows), 6
        ),
    }


def analyze_signal_diagnostics(
    frame: pl.DataFrame,
    *,
    horizon: int,
    universe_n: int,
    direction: int,
) -> dict[str, Any]:
    """Calculate Pearson IC and Spearman RankIC on actual rebalance dates."""
    forward = f"fwd_{int(horizon)}"
    base = {
        "protocol": SIGNAL_DIAGNOSTICS_PROTOCOL,
        "status": "INSUFFICIENT_DATA",
        "horizon_sessions": int(horizon),
        "direction": int(direction),
        "label_definition": (
            f"signal at t close versus adjusted-open return t+1 to t+1+{int(horizon)}"
        ),
        "sampling": "actual non-overlapping rebalance dates",
        "portfolio_return_separation": (
            "IC is a signal diagnostic; CAGR/Sharpe/MDD remain event-ledger metrics"
        ),
    }
    if forward not in frame.columns or frame.is_empty():
        return base | {"reason": f"回测帧缺少 {forward} 或为空"}
    dates = frame["trade_date"].unique().sort().to_list()
    signal_dates = dates[:: max(1, int(horizon))]
    minimum_cross_section = min(50, max(10, int(universe_n * 0.20)))
    daily = (
        frame.lazy()
        .filter(
            pl.col("trade_date").is_in(signal_dates)
            & (pl.col("univ_rank") <= int(universe_n))
            & pl.col("factor").is_finite()
            & pl.col(forward).is_finite()
        )
        .with_columns((pl.col("factor") * int(direction)).alias("_signal"))
        .with_columns(
            pl.col("_signal").rank(method="average").over("trade_date").alias("_signal_rank"),
            pl.col(forward).rank(method="average").over("trade_date").alias("_return_rank"),
        )
        .group_by("trade_date")
        .agg(
            pl.corr("_signal", forward).alias("ic"),
            pl.corr("_signal_rank", "_return_rank").alias("rank_ic"),
            pl.len().alias("n"),
        )
        .filter(
            (pl.col("n") >= minimum_cross_section)
            & pl.col("ic").is_finite()
            & pl.col("rank_ic").is_finite()
        )
        .sort("trade_date")
        .collect()
    )
    path = [
        {
            "trade_date": _as_date(row["trade_date"]).isoformat(),
            "ic": _round(float(row["ic"]), 8),
            "rank_ic": _round(float(row["rank_ic"]), 8),
            "n": int(row["n"]),
        }
        for row in daily.to_dicts()
        if _finite(row.get("ic")) and _finite(row.get("rank_ic"))
    ]
    if len(path) < 2:
        return base | {
            "reason": "有效IC截面少于2",
            "minimum_cross_section_n": minimum_cross_section,
            "path": path,
        }
    annual = []
    for year in sorted({_as_date(row["trade_date"]).year for row in path}):
        rows = [row for row in path if _as_date(row["trade_date"]).year == year]
        annual.append({"period": str(year), **_ic_summary(rows, horizon)})

    rolling: dict[str, list[dict[str, Any]]] = {"12m": [], "24m": []}
    path_dates = [_as_date(row["trade_date"]) for row in path]
    month_end_indexes = [
        index for index, value in enumerate(path_dates)
        if index == len(path_dates) - 1 or path_dates[index + 1].month != value.month
    ]
    for months, minimum_points in ((12, max(8, int(180 / max(1, horizon)))), (24, max(16, int(360 / max(1, horizon))))):
        for end_index in month_end_indexes:
            target = _shift_months(path_dates[end_index], -months) + timedelta(days=1)
            rows = [
                row for row, value in zip(path[: end_index + 1], path_dates[: end_index + 1])
                if value >= target
            ]
            if len(rows) < minimum_points:
                continue
            rolling[f"{months}m"].append({
                "period": f"{months}m@{path_dates[end_index].isoformat()}",
                **_ic_summary(rows, horizon),
            })
    return base | {
        "status": "OK",
        "minimum_cross_section_n": minimum_cross_section,
        "overall": _ic_summary(path, horizon),
        "annual": annual,
        "rolling": rolling,
        "latest_rolling": {
            key: values[-1] if values else None for key, values in rolling.items()
        },
        "path": path,
    }


def combine_sleeve_signal_diagnostics(
    sleeves: list[tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    factors = []
    for spec, result in sleeves:
        factors.append({
            "factor_id": spec["factor_id"],
            "name": spec["name"],
            "expression": spec["expression"],
            "direction": spec["direction"],
            "normalized_weight": spec["normalized_weight"],
            "diagnostics": result.get("signal_diagnostics", {}),
        })
    available = [
        row for row in factors if row["diagnostics"].get("overall", {}).get("status") == "OK"
    ]
    weighted = {}
    if available:
        denominator = sum(float(row["normalized_weight"]) for row in available)
        for key in ("ic_mean", "icir", "rank_ic_mean", "rank_icir"):
            weighted[key] = _round(sum(
                float(row["normalized_weight"])
                * float(row["diagnostics"]["overall"].get(key) or 0.0)
                for row in available
            ) / denominator, 8)
    return {
        "protocol": SIGNAL_DIAGNOSTICS_PROTOCOL,
        "status": "OK" if available else "INSUFFICIENT_DATA",
        "combination_semantics": "independent_capital_sleeves",
        "composite_rank_ic_available": False,
        "disclosure": (
            "独立资金袖套没有单一组合排序信号；weighted_summary只是各袖套IC统计的资本权重摘要，"
            "不是把相关系数线性组合后得到的组合IC"
        ),
        "weighted_summary": weighted,
        "factors": factors,
    }


def _distribution(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {}
    return {
        "mean": _round(float(np.mean(finite)), 8),
        "std": _round(float(np.std(finite, ddof=1)), 8),
        "p01": _round(float(np.quantile(finite, 0.01)), 8),
        "p05": _round(float(np.quantile(finite, 0.05)), 8),
        "p25": _round(float(np.quantile(finite, 0.25)), 8),
        "p50": _round(float(np.quantile(finite, 0.50)), 8),
        "p75": _round(float(np.quantile(finite, 0.75)), 8),
        "p95": _round(float(np.quantile(finite, 0.95)), 8),
        "p99": _round(float(np.quantile(finite, 0.99)), 8),
    }


def monte_carlo_analysis(
    daily_steps: list[dict[str, Any]],
    *,
    simulations: int = 2000,
    block_size_sessions: int = 20,
    seed: int = 20260824,
    sleeves: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Moving-block bootstrap of fee-after ledger returns.

    Every sleeve uses the same sampled date indexes, preserving contemporaneous
    cross-sleeve dependence.  This is a historical path-resampling stress test,
    not a parametric forecast of future returns.
    """
    simulations = int(simulations)
    block_size_sessions = int(block_size_sessions)
    seed = int(seed)
    if not 100 <= simulations <= 20_000:
        raise ValueError("蒙特卡洛模拟次数必须在 100..20000")
    if not 1 <= block_size_sessions <= 252:
        raise ValueError("蒙特卡洛区块长度必须在 1..252 个交易日")
    if len(daily_steps) < 120:
        return {
            "protocol": MONTE_CARLO_PROTOCOL,
            "status": "INSUFFICIENT_DATA",
            "reason": "有效交易日少于120",
        }
    returns = np.asarray(
        [float(row.get("daily_return") or 0.0) for row in daily_steps],
        dtype=np.float64,
    )
    if not np.isfinite(returns).all() or np.any(returns <= -1.0):
        return {
            "protocol": MONTE_CARLO_PROTOCOL,
            "status": "INSUFFICIENT_DATA",
            "reason": "日收益包含非有限值或小于等于-100%的值",
        }
    n_sessions = int(returns.size)
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(n_sessions / block_size_sessions)
    offsets = np.arange(block_size_sessions, dtype=np.int64)
    terminal_return = np.empty(simulations, dtype=np.float64)
    cagr = np.empty(simulations, dtype=np.float64)
    sharpe = np.empty(simulations, dtype=np.float64)
    max_drawdown = np.empty(simulations, dtype=np.float64)

    prepared_sleeves: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    for sleeve in sleeves or []:
        values = np.asarray(sleeve.get("nlv", []), dtype=np.float64)
        if values.size != n_sessions or np.any(values <= 0) or not np.isfinite(values).all():
            continue
        initial_capital = float(sleeve["initial_capital"])
        prior = np.concatenate([[initial_capital], values[:-1]])
        prepared_sleeves.append((
            sleeve,
            values / prior - 1.0,
            np.empty(simulations, dtype=np.float64),
        ))

    # Bound peak memory even when the API permits 20,000 paths.
    chunk_size = min(256, simulations)
    for cursor in range(0, simulations, chunk_size):
        chunk = min(chunk_size, simulations - cursor)
        starts = rng.integers(0, n_sessions, size=(chunk, blocks_needed))
        indexes = ((starts[..., None] + offsets) % n_sessions).reshape(chunk, -1)
        indexes = indexes[:, :n_sessions]
        sampled = returns[indexes]
        wealth = np.cumprod(1.0 + sampled, axis=1)
        terminal_return[cursor : cursor + chunk] = wealth[:, -1] - 1.0
        cagr[cursor : cursor + chunk] = (
            np.power(np.maximum(wealth[:, -1], 1e-15), 252.0 / n_sessions) - 1.0
        )
        sample_mean = np.mean(sampled, axis=1)
        sample_std = np.std(sampled, axis=1, ddof=1)
        sharpe[cursor : cursor + chunk] = np.divide(
            sample_mean * math.sqrt(252.0),
            sample_std,
            out=np.zeros_like(sample_mean),
            where=sample_std > 1e-12,
        )
        running_peak = np.maximum.accumulate(
            np.concatenate([np.ones((chunk, 1)), wealth], axis=1), axis=1
        )[:, 1:]
        max_drawdown[cursor : cursor + chunk] = np.max(
            1.0 - wealth / running_peak, axis=1
        )
        for _, sleeve_returns, sleeve_terminal in prepared_sleeves:
            sleeve_terminal[cursor : cursor + chunk] = (
                np.prod(1.0 + sleeve_returns[indexes], axis=1) - 1.0
            )

    observed_wealth = np.cumprod(1.0 + returns)
    observed_peak = np.maximum.accumulate(np.concatenate([[1.0], observed_wealth]))[1:]
    observed_mean, observed_std = _mean_std(returns.tolist())
    observed = {
        "terminal_return": _round(float(observed_wealth[-1] - 1.0), 8),
        "cagr": _round(float(observed_wealth[-1] ** (252.0 / n_sessions) - 1.0), 8),
        "sharpe": _round(
            observed_mean / observed_std * math.sqrt(252.0)
            if observed_std > 1e-12 else 0.0,
            6,
        ),
        "max_drawdown": _round(float(np.max(1.0 - observed_wealth / observed_peak)), 8),
    }

    sleeve_rows = []
    for sleeve, _, sleeve_terminal in prepared_sleeves:
        sleeve_rows.append({
            "factor_id": sleeve["factor_id"],
            "name": sleeve["name"],
            "normalized_weight": _round(float(sleeve["normalized_weight"]), 10),
            "terminal_return": _distribution(sleeve_terminal),
            "probability_positive_terminal_return": _round(
                float(np.mean(sleeve_terminal > 0.0)), 8
            ),
        })

    return {
        "protocol": MONTE_CARLO_PROTOCOL,
        "status": "OK",
        "method": "circular_moving_block_bootstrap",
        "disclosure": (
            "对费后事件账本日收益做历史路径重采样；保留区块内序列结构，"
            "不代表未来分布，也不替代时序样本外检验"
        ),
        "simulations": simulations,
        "seed": seed,
        "sessions": n_sessions,
        "block_size_sessions": block_size_sessions,
        "observed": observed,
        "terminal_return": _distribution(terminal_return),
        "cagr": _distribution(cagr),
        "sharpe": _distribution(sharpe),
        "max_drawdown": _distribution(max_drawdown),
        "risk_probabilities": {
            "terminal_loss": _round(float(np.mean(terminal_return < 0.0)), 8),
            "negative_sharpe": _round(float(np.mean(sharpe < 0.0)), 8),
            "max_drawdown_ge_20pct": _round(float(np.mean(max_drawdown >= 0.20)), 8),
            "max_drawdown_ge_30pct": _round(float(np.mean(max_drawdown >= 0.30)), 8),
            "max_drawdown_ge_50pct": _round(float(np.mean(max_drawdown >= 0.50)), 8),
            "terminal_nav_below_0_8": _round(float(np.mean(terminal_return <= -0.20)), 8),
        },
        "sleeves": sleeve_rows,
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or len(left) < 3:
        return None
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if np.std(left_array) <= 1e-12 or np.std(right_array) <= 1e-12:
        return None
    value = float(np.corrcoef(left_array, right_array)[0, 1])
    return _round(value, 6) if math.isfinite(value) else None


def factor_performance_correlation(
    *,
    dates: list[Any],
    sleeves: list[dict[str, Any]],
    signal_diagnostics: dict[str, Any] | None = None,
    stability_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare realised sleeve paths and IC regimes under matched dates."""
    if len(sleeves) < 2:
        return {
            "protocol": "factor_performance_correlation_v1",
            "status": "NOT_APPLICABLE",
            "reason": "至少需要两个独立资金sleeve",
        }
    parsed_dates = [_as_date(value) for value in dates]
    labels = [str(row["factor_id"]) for row in sleeves]
    names = [str(row["name"]) for row in sleeves]
    daily_returns: dict[str, list[float]] = {}
    monthly_returns: dict[str, dict[str, float]] = {}
    for sleeve in sleeves:
        values = [float(value) for value in sleeve["nlv"]]
        prior = [float(sleeve["initial_capital"]), *values[:-1]]
        returns = [value / previous - 1.0 if previous > 0 else 0.0 for value, previous in zip(values, prior)]
        daily_returns[sleeve["factor_id"]] = returns
        by_month: dict[str, float] = {}
        for trade_date, daily_return in zip(parsed_dates, returns):
            key = trade_date.strftime("%Y-%m")
            by_month[key] = (1.0 + by_month.get(key, 0.0)) * (1.0 + daily_return) - 1.0
        monthly_returns[sleeve["factor_id"]] = by_month

    ic_paths: dict[str, dict[str, float]] = {}
    for factor in (signal_diagnostics or {}).get("factors", []):
        ic_paths[factor["factor_id"]] = {
            row["trade_date"]: float(row["rank_ic"])
            for row in factor.get("diagnostics", {}).get("path", [])
        }
    rolling_paths: dict[str, dict[str, float]] = {label: {} for label in labels}
    for period in (stability_analysis or {}).get("rolling", {}).get("12m", []):
        for row in period.get("sleeves", []):
            rolling_paths.setdefault(row["factor_id"], {})[period["end"]] = float(
                row["standalone_return"]
            )

    def matrix(metric: str) -> list[list[float | None]]:
        output: list[list[float | None]] = []
        for left in labels:
            row: list[float | None] = []
            for right in labels:
                if left == right:
                    row.append(1.0)
                    continue
                if metric == "daily_return":
                    row.append(_correlation(daily_returns[left], daily_returns[right]))
                elif metric == "monthly_return":
                    common = sorted(set(monthly_returns[left]) & set(monthly_returns[right]))
                    row.append(_correlation(
                        [monthly_returns[left][key] for key in common],
                        [monthly_returns[right][key] for key in common],
                    ))
                elif metric == "rank_ic_path":
                    common = sorted(set(ic_paths.get(left, {})) & set(ic_paths.get(right, {})))
                    row.append(_correlation(
                        [ic_paths[left][key] for key in common],
                        [ic_paths[right][key] for key in common],
                    ))
                else:
                    common = sorted(set(rolling_paths.get(left, {})) & set(rolling_paths.get(right, {})))
                    row.append(_correlation(
                        [rolling_paths[left][key] for key in common],
                        [rolling_paths[right][key] for key in common],
                    ))
            output.append(row)
        return output

    matrices = {
        metric: matrix(metric)
        for metric in ("daily_return", "monthly_return", "rolling_12m_return", "rank_ic_path")
    }
    pairs = []
    for left_index in range(len(labels)):
        for right_index in range(left_index + 1, len(labels)):
            daily = matrices["daily_return"][left_index][right_index]
            monthly = matrices["monthly_return"][left_index][right_index]
            rolling = matrices["rolling_12m_return"][left_index][right_index]
            rank_ic = matrices["rank_ic_path"][left_index][right_index]
            if monthly is not None and monthly >= 0.70 and rank_ic is not None and rank_ic >= 0.60:
                classification = "same_return_source_risk"
            elif monthly is not None and monthly >= 0.70:
                classification = "realised_performance_overlap"
            elif rolling is not None and rolling >= 0.70:
                classification = "regime_overlap"
            elif monthly is not None and monthly <= -0.30:
                classification = "diversifying_negative_correlation"
            else:
                classification = "distinct_or_inconclusive"
            pairs.append({
                "left": labels[left_index],
                "left_name": names[left_index],
                "right": labels[right_index],
                "right_name": names[right_index],
                "daily_return_correlation": daily,
                "monthly_return_correlation": monthly,
                "rolling_12m_return_correlation": rolling,
                "rank_ic_path_correlation": rank_ic,
                "classification": classification,
            })
    return {
        "protocol": "factor_performance_correlation_v1",
        "status": "OK",
        "labels": labels,
        "names": names,
        "matrices": matrices,
        "pairs": pairs,
        "disclosure": (
            "实际表现相关性来自各sleeve独立费后账本；RankIC路径相关性来自相同因果前向开盘口径。"
            "高相关仅提示重复收益来源风险，不自动证明经济机制相同"
        ),
    }
