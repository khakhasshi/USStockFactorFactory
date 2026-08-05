"""手动回测引擎: rank 加权多空组合, 日频调仓, 换手成本, 净值曲线.

口径: t 日收盘信号 -> t+1 开盘建仓 -> t+2 开盘平仓 (fwd_1, 前复权 open).
"""

import math

import polars as pl

from ..config import DEFAULT_PORTFOLIO_MODE, get_dsl_fields
from ..data.panel import PanelStore
from ..dsl.engine import parse


def run_backtest(
    expression: str,
    universe_n: int = 500,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    cost_bps: float = 15.0,
    direction: int = 1,
    mode: str = DEFAULT_PORTFOLIO_MODE,  # long_short / long_only
    panel_glob: str | None = None,
    market: str = "us",
    borrow_cost_bps_annual: float = 0.0,
    top_fraction: float = 0.20,
) -> dict:
    if mode not in {"long_short", "long_only"}:
        raise ValueError("mode 必须是 long_short 或 long_only")
    df = PanelStore.get(panel_glob, market).ensure_loaded()
    pipe = parse(expression, get_dsl_fields(market))

    work = (
        pipe.apply(
            df.lazy().filter(
                pl.col("trade_date").is_between(pl.lit(start).cast(pl.Date), pl.lit(end).cast(pl.Date))
            )
        )
        .filter(pl.col("univ_rank") <= universe_n)
        .filter(pl.col("factor").is_finite() & pl.col("fwd_1").is_finite())
        .with_columns(
            (pl.col("factor").rank().over("trade_date") / pl.col("factor").count().over("trade_date"))
            .alias("f_pct"),
            pl.col("trade_date").rank(method="dense").alias("_date_seq"),
        )
    )
    work = work.with_columns(
        (pl.col("f_pct") if direction >= 0 else 1.0 - pl.col("f_pct")).alias("signal_pct")
    ).with_columns(
        (pl.col("signal_pct") >= 1.0 - top_fraction).alias("is_long"),
        (pl.col("signal_pct") <= top_fraction).alias("is_short"),
    ).with_columns(
        pl.col("is_long").sum().over("trade_date").alias("n_long"),
        pl.col("is_short").sum().over("trade_date").alias("n_short"),
    ).with_columns(
        pl.when(pl.col("is_long")).then(1.0 / pl.col("n_long")).otherwise(0.0).alias("long_w"),
        pl.when(pl.col("is_short")).then(1.0 / pl.col("n_short")).otherwise(0.0).alias("short_w"),
    ).with_columns(
        (pl.col("long_w") if mode == "long_only" else pl.col("long_w") - pl.col("short_w")).alias("w")
    )

    work = work.with_columns(
        pl.col("w")
        .shift(1)
        .over("ts_code", order_by="trade_date")
        .fill_null(0.0)
        .alias("_previous_seen_weight"),
        pl.col("_date_seq")
        .shift(1)
        .over("ts_code", order_by="trade_date")
        .alias("_previous_seen_seq"),
    ).with_columns(
        pl.when(pl.col("_previous_seen_seq") == pl.col("_date_seq") - 1)
        .then(pl.col("_previous_seen_weight"))
        .otherwise(0.0)
        .alias("_previous_weight")
    ).with_columns(
        (pl.col("w") - pl.col("_previous_weight")).abs().alias("_current_weight_change"),
        pl.col("_previous_weight").abs().alias("_matched_previous_gross"),
    )
    target_gross = 1.0 if mode == "long_only" else 2.0
    daily = (
        work.group_by("trade_date")
        .agg(
            (pl.col("w") * pl.col("fwd_1")).sum().alias("gross_ret"),
            pl.col("fwd_1").mean().alias("benchmark_ret"),
            (pl.col("long_w") * pl.col("fwd_1")).sum().alias("long_ret"),
            (pl.col("short_w") * pl.col("fwd_1")).sum().alias("short_ret"),
            pl.col("_current_weight_change").sum().alias("_current_turnover"),
            pl.col("_matched_previous_gross").sum().alias("_matched_previous_gross"),
            pl.col("_date_seq").first().alias("_date_seq"),
            pl.len().alias("n"),
        )
        .with_columns(
            pl.when(pl.col("_date_seq") == 1)
            .then(0.0)
            .otherwise(
                (pl.lit(target_gross) - pl.col("_matched_previous_gross"))
                .clip(0.0, target_gross)
            )
            .alias("_exit_turnover")
        )
        .with_columns(
            (pl.col("_current_turnover") + pl.col("_exit_turnover")).alias("turnover")
        )
        .drop(
            "_current_turnover",
            "_matched_previous_gross",
            "_date_seq",
            "_exit_turnover",
        )
        .filter(pl.col("n") >= 50)
        .sort("trade_date")
        .collect()
    )
    if daily.height < 60:
        raise ValueError("回测样本不足 (有效交易日 < 60)")

    cost = cost_bps / 1e4
    borrow_daily = (
        float(borrow_cost_bps_annual) / 1e4 / 252.0 if mode == "long_short" else 0.0
    )
    daily = daily.with_columns(
        (
            pl.col("gross_ret") - pl.col("turnover") * cost - borrow_daily
        ).alias("net_ret"),
        (
            pl.col("gross_ret") - pl.col("benchmark_ret") - pl.col("turnover") * cost
            if mode == "long_only"
            else pl.col("gross_ret") - pl.col("turnover") * cost - borrow_daily
        ).alias("active_ret"),
    )
    net = daily["net_ret"]
    equity, peak, max_dd = [], 1.0, 0.0
    nav = 1.0
    for r in net:
        nav *= 1 + (r or 0)
        peak = max(peak, nav)
        max_dd = max(max_dd, 1 - nav / peak)
        equity.append(nav)

    n = daily.height
    ann_ret = nav ** (252 / n) - 1
    ann_vol = float(net.std() or 1e-9) * math.sqrt(252)
    sharpe = float(net.mean()) / float(net.std() or 1e-9) * math.sqrt(252)
    active = daily["active_ret"]
    active_sharpe = float(active.mean()) / float(active.std() or 1e-9) * math.sqrt(252)
    gross = daily["gross_ret"]
    gross_sharpe = float(gross.mean()) / float(gross.std() or 1e-9) * math.sqrt(252)
    dates = [str(d) for d in daily["trade_date"]]

    return {
        "stats": {
            "days": n,
            "ann_ret": round(ann_ret, 4),
            "ann_vol": round(ann_vol, 4),
            "sharpe": round(sharpe, 3),
            "gross_sharpe": round(gross_sharpe, 3),
            "active_sharpe": round(active_sharpe, 3),
            "max_dd": round(max_dd, 4),
            "avg_daily_turnover": round(float(daily["turnover"].mean()), 4),
            "avg_one_way_turnover": round(float(daily["turnover"].mean()) / 2.0, 4),
            "base_cost_bps": float(cost_bps),
            "borrow_cost_bps_annual": float(borrow_cost_bps_annual) if mode == "long_short" else 0.0,
            "long_leg_ann_mean": round(float(daily["long_ret"].mean()) * 252, 4),
            "short_leg_ann_mean": round(-float(daily["short_ret"].mean()) * 252, 4)
            if mode == "long_short"
            else None,
            "final_nav": round(nav, 4),
        },
        "curve": {
            "dates": dates,
            "equity": [round(v, 5) for v in equity],
            "daily_ret": [round(float(r or 0), 6) for r in net],
        },
    }
