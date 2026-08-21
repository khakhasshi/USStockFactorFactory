"""Fast weight-vector screening for cross-sectional equity factors.

This module is deliberately a *screening* engine.  It preserves the factor
observation date, next-open forward-return convention, rebalance cadence,
portfolio side, direction, fee schedule, slippage levels, and short-borrow
proxy.  It does not pretend to reproduce the stateful event ledger:

* target weights are filled completely;
* A-share lots, cash affordability, limit-up/down blocks and partial fills are
  omitted;
* period returns do not expose intraperiod drawdown;
* fees are estimated from target-weight changes rather than settlement fills.

Candidates promoted by this engine must therefore be replayed by
``StepEventBacktester`` before they can appear in an event-verified leaderboard
or receive a settlement statement.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import date
from typing import Any

import polars as pl

from .batch import BatchBacktestSpec
from .fees import calculate_trade_fees, fee_schedule_snapshot


VECTOR_SCREEN_PROTOCOL = "weight_vector_open_to_open_screen_v1"


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _stats(returns: list[float], periods_per_year: float) -> dict:
    clean = [float(value) for value in returns if _finite(value)]
    if not clean:
        raise ValueError("向量回测没有有效收益期")
    nav = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in clean:
        nav *= max(1e-12, 1.0 + value)
        peak = max(peak, nav)
        max_drawdown = max(max_drawdown, 1.0 - nav / peak)
    mean = sum(clean) / len(clean)
    variance = (
        sum((value - mean) ** 2 for value in clean) / (len(clean) - 1)
        if len(clean) > 1
        else 0.0
    )
    std = math.sqrt(max(0.0, variance))
    return {
        "ann_return": round(
            nav ** (periods_per_year / len(clean)) - 1.0,
            8,
        ),
        "sharpe": round(
            mean / std * math.sqrt(periods_per_year)
            if std > 1e-12
            else 0.0,
            6,
        ),
        "max_drawdown": round(max_drawdown, 8),
        "final_nav": round(nav, 8),
    }


@dataclass(frozen=True)
class _VectorPeriod:
    signal_date: date
    execute_date: date
    gross_return: float
    benchmark_return: float
    turnover: float
    buy_turnover: float
    sell_turnover: float
    fee_fraction: float
    gross_exposure: float
    net_exposure: float
    order_count_proxy: int
    holdings_fingerprint: str


def _signal_dates(
    frame: pl.DataFrame,
    spec: BatchBacktestSpec,
) -> tuple[list[date], dict[date, date]]:
    start = date.fromisoformat(spec.holdout_start)
    end = date.fromisoformat(spec.holdout_end)
    dates = (
        frame.lazy()
        .filter(pl.col("trade_date").is_between(start, end))
        .select("trade_date")
        .unique()
        .sort("trade_date")
        .collect()["trade_date"]
        .to_list()
    )
    if len(dates) < 60:
        raise ValueError("向量回测样本不足 (有效交易日 < 60)")
    signals = [
        dates[index]
        for index in range(0, len(dates) - spec.rebalance_every, spec.rebalance_every)
    ]
    execute_dates = {
        dates[index]: dates[index + 1]
        for index in range(0, len(dates) - spec.rebalance_every, spec.rebalance_every)
    }
    return signals, execute_dates


def _selected_weights(
    frame: pl.DataFrame,
    *,
    direction: int,
    spec: BatchBacktestSpec,
    signal_dates: list[date],
) -> tuple[pl.DataFrame, dict[date, float]]:
    forward = f"fwd_{spec.horizon}"
    if forward not in frame.columns:
        raise ValueError(f"回测帧缺少 {forward}")
    if spec.horizon != spec.rebalance_every:
        raise ValueError(
            "向量筛选当前要求 horizon 与 rebalance_every 相同"
        )
    eligible = (
        frame.lazy()
        .filter(
            pl.col("trade_date").is_in(signal_dates)
            & (pl.col("univ_rank") <= spec.universe_n)
            & pl.col("factor").is_finite()
            & pl.col(forward).is_finite()
            & pl.col("raw_close").is_finite()
            & (pl.col("raw_close") > 0)
        )
        .with_columns(
            (pl.col("factor") * direction).alias("_signal"),
            pl.len().over("trade_date").alias("_eligible_n"),
        )
        .with_columns(
            pl.col("_signal")
            .rank(method="ordinal", descending=True)
            .over("trade_date")
            .alias("_signal_rank"),
            (
                (pl.col("_eligible_n") * spec.top_fraction)
                .floor()
                .cast(pl.Int64)
                .clip(1, None)
            ).alias("_select_n"),
        )
        .with_columns(
            (pl.col("_signal_rank") <= pl.col("_select_n")).alias(
                "_is_long"
            ),
            (
                pl.col("_signal_rank")
                > pl.col("_eligible_n") - pl.col("_select_n")
            ).alias("_is_short"),
        )
        .with_columns(
            pl.when(pl.col("_is_long"))
            .then(1.0 / pl.col("_select_n"))
            .when(
                pl.col("_is_short")
                & pl.lit(spec.mode == "long_short")
            )
            .then(-1.0 / pl.col("_select_n"))
            .otherwise(0.0)
            .alias("_target_weight")
        )
        .collect()
    )
    benchmark = {
        row["trade_date"]: float(row["benchmark_return"])
        for row in (
            eligible.lazy()
            .group_by("trade_date")
            .agg(pl.col(forward).mean().alias("benchmark_return"))
            .collect()
            .iter_rows(named=True)
        )
    }
    selected = eligible.filter(pl.col("_target_weight") != 0).select(
        "trade_date",
        "ts_code",
        "raw_close",
        forward,
        "_target_weight",
    )
    return selected, benchmark


def _estimate_fee_fraction(
    *,
    market: str,
    trade_date: date,
    deltas: dict[str, float],
    prices: dict[str, float],
    initial_capital: float,
    fee_profile: str | None,
) -> float:
    total = 0.0
    for symbol, delta in deltas.items():
        if abs(delta) <= 1e-12:
            continue
        price = float(prices.get(symbol) or 0.0)
        if price <= 0:
            # A missing exit price is one reason this path remains screening
            # only.  Use a unit price to retain the intended notional and the
            # conservative per-order minimum instead of dropping the cost.
            price = 1.0
        notional = abs(delta) * initial_capital
        quantity = max(1e-9, notional / price)
        fee = calculate_trade_fees(
            market=market,
            side="BUY" if delta > 0 else "SELL",
            quantity=quantity,
            price=price,
            trade_date=trade_date,
            profile=fee_profile,
        )
        total += fee.total
    return total / initial_capital


def _periods(
    frame: pl.DataFrame,
    *,
    direction: int,
    spec: BatchBacktestSpec,
) -> list[_VectorPeriod]:
    if direction not in {-1, 1}:
        raise ValueError("direction 必须为 1 或 -1")
    signals, execute_dates = _signal_dates(frame, spec)
    selected, benchmark = _selected_weights(
        frame,
        direction=direction,
        spec=spec,
        signal_dates=signals,
    )
    forward = f"fwd_{spec.horizon}"
    grouped = {
        group["trade_date"][0]: group
        for group in selected.partition_by(
            "trade_date",
            maintain_order=True,
        )
    }
    previous_weights: dict[str, float] = {}
    previous_prices: dict[str, float] = {}
    periods: list[_VectorPeriod] = []
    for signal_date in signals:
        group = grouped.get(signal_date)
        if group is None or group.height == 0:
            continue
        current_weights = {
            str(row["ts_code"]): float(row["_target_weight"])
            for row in group.iter_rows(named=True)
        }
        current_prices = {
            str(row["ts_code"]): float(row["raw_close"])
            for row in group.iter_rows(named=True)
        }
        deltas = {
            symbol: current_weights.get(symbol, 0.0)
            - previous_weights.get(symbol, 0.0)
            for symbol in set(current_weights) | set(previous_weights)
        }
        turnover = sum(abs(value) for value in deltas.values())
        buy_turnover = sum(
            value for value in deltas.values() if value > 0
        )
        sell_turnover = sum(
            -value for value in deltas.values() if value < 0
        )
        prices = {**previous_prices, **current_prices}
        fee_fraction = _estimate_fee_fraction(
            market=spec.market,
            trade_date=execute_dates[signal_date],
            deltas=deltas,
            prices=prices,
            initial_capital=spec.initial_capital,
            fee_profile=spec.fee_profile,
        )
        gross_return = sum(
            float(row["_target_weight"]) * float(row[forward])
            for row in group.iter_rows(named=True)
        )
        periods.append(_VectorPeriod(
            signal_date=signal_date,
            execute_date=execute_dates[signal_date],
            gross_return=gross_return,
            benchmark_return=float(benchmark.get(signal_date, 0.0)),
            turnover=turnover,
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            fee_fraction=fee_fraction,
            gross_exposure=sum(
                abs(value) for value in current_weights.values()
            ),
            net_exposure=sum(current_weights.values()),
            order_count_proxy=sum(
                abs(value) > 1e-12 for value in deltas.values()
            ),
            holdings_fingerprint=hashlib.sha256(
                "|".join(
                    f"{symbol}:{weight:.12f}"
                    for symbol, weight in sorted(current_weights.items())
                ).encode("utf-8")
            ).hexdigest(),
        ))
        previous_weights = current_weights
        previous_prices = current_prices
    if len(periods) < 12:
        raise ValueError("向量回测有效调仓期少于 12")
    return periods


def run_vector_cost_scenarios(
    frame: pl.DataFrame,
    *,
    direction: int,
    spec: BatchBacktestSpec,
    capture_periods: bool = False,
) -> dict[str, dict]:
    """Evaluate one factor orientation using fast weight-vector periods."""
    periods = _periods(frame, direction=direction, spec=spec)
    portfolio_fingerprint = hashlib.sha256(
        "\n".join(
            f"{period.signal_date.isoformat()}:{period.holdings_fingerprint}"
            for period in periods
        ).encode("utf-8")
    ).hexdigest()[:16]
    periods_per_year = 252.0 / spec.rebalance_every
    borrow_fraction = (
        spec.borrow_cost_bps_annual
        / 10_000.0
        * spec.rebalance_every
        / 252.0
        if spec.mode == "long_short"
        else 0.0
    )
    output: dict[str, dict] = {}
    for bps in spec.slippage_bps:
        returns: list[float] = []
        active_returns: list[float] = []
        benchmark_returns: list[float] = []
        nav = 1.0
        fee_amount = 0.0
        slippage_amount = 0.0
        borrow_amount = 0.0
        period_rows: list[dict] = []
        for period in periods:
            slippage_fraction = period.turnover * float(bps) / 10_000.0
            net_return = (
                period.gross_return
                - period.fee_fraction
                - slippage_fraction
                - borrow_fraction
            )
            active_return = (
                net_return - period.benchmark_return
                if spec.mode == "long_only"
                else net_return
            )
            capital_before = spec.initial_capital * nav
            fee_amount += capital_before * period.fee_fraction
            slippage_amount += capital_before * slippage_fraction
            borrow_amount += capital_before * borrow_fraction
            nav *= max(1e-12, 1.0 + net_return)
            returns.append(net_return)
            active_returns.append(active_return)
            benchmark_returns.append(period.benchmark_return)
            if capture_periods:
                period_rows.append({
                    "signal_date": str(period.signal_date),
                    "execute_date": str(period.execute_date),
                    "gross_return": period.gross_return,
                    "benchmark_return": period.benchmark_return,
                    "turnover": period.turnover,
                    "fee_fraction": period.fee_fraction,
                    "slippage_fraction": slippage_fraction,
                    "borrow_fraction": borrow_fraction,
                    "net_return": net_return,
                    "active_return": active_return,
                    "nav": nav,
                })
        stats = _stats(returns, periods_per_year)
        active_stats = _stats(active_returns, periods_per_year)
        benchmark_stats = _stats(benchmark_returns, periods_per_year)
        finite = all(_finite(value) for value in returns)
        scenario = {
            **stats,
            "active_ann_return": active_stats["ann_return"],
            "active_sharpe": active_stats["sharpe"],
            "active_max_drawdown": active_stats["max_drawdown"],
            "active_final_nav": active_stats["final_nav"],
            "benchmark_ann_return": benchmark_stats["ann_return"],
            "benchmark_sharpe": benchmark_stats["sharpe"],
            "avg_daily_turnover": round(
                sum(row.turnover for row in periods)
                / len(periods)
                / spec.rebalance_every,
                8,
            ),
            "fills": None,
            "fill_rate": None,
            "orders_proxy": sum(
                row.order_count_proxy for row in periods
            ),
            "commission_and_tax": round(fee_amount, 6),
            "slippage_cost": round(slippage_amount, 6),
            "borrow_cost": round(borrow_amount, 6),
            "total_execution_cost": round(
                fee_amount + slippage_amount + borrow_amount,
                6,
            ),
            "avg_gross_exposure": round(
                sum(row.gross_exposure for row in periods)
                / len(periods),
                6,
            ),
            "avg_net_exposure": round(
                sum(row.net_exposure for row in periods)
                / len(periods),
                6,
            ),
            "fee_profile": fee_schedule_snapshot(
                spec.market,
                spec.fee_profile,
            )["profile"],
            "currency": "CNY" if spec.market == "ashare" else "USD",
            "integrity": {
                "all_pass": bool(finite and len(periods) >= 12),
                "protocol": VECTOR_SCREEN_PROTOCOL,
                "finite_period_returns": finite,
                "periods": len(periods),
                "settlement_statement_available": False,
                "event_replay_required": True,
            },
            "detail_capture": "vector_screen_no_statement",
            "screening_only": True,
            "periods": len(periods),
            "portfolio_fingerprint": portfolio_fingerprint,
        }
        if capture_periods:
            scenario["period_rows"] = period_rows
        output[f"{float(bps):g}"] = scenario
    return output
