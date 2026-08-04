"""手动回测引擎: rank 加权多空组合, 日频调仓, 换手成本, 净值曲线.

口径: t 日收盘信号 -> t+1 开盘建仓 -> t+2 开盘平仓 (fwd_1, 前复权 open).
"""

import math

import polars as pl

from ..data.panel import PanelStore
from ..dsl.engine import parse


def run_backtest(
    expression: str,
    universe_n: int = 500,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    cost_bps: float = 15.0,
    direction: int = 1,
    mode: str = "long_short",  # long_short / long_only
) -> dict:
    df = PanelStore.get().ensure_loaded()
    pipe = parse(expression)

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
            .alias("f_pct")
        )
    )
    if mode == "long_only":
        # 只做多头前 20%, 等权
        work = work.with_columns(
            pl.when(pl.col("f_pct") >= 0.8 if direction >= 0 else pl.col("f_pct") <= 0.2)
            .then(1.0)
            .otherwise(0.0)
            .alias("w_raw")
        ).with_columns((pl.col("w_raw") / pl.col("w_raw").sum().over("trade_date")).alias("w"))
    else:
        # rank 中心化多空, 双边杠杆 1
        work = work.with_columns(((pl.col("f_pct") - 0.5) * float(direction)).alias("w_raw")).with_columns(
            (pl.col("w_raw") / pl.col("w_raw").abs().sum().over("trade_date") * 2).alias("w")
        )

    work = work.with_columns(
        (pl.col("w") - pl.col("w").shift(1).over("ts_code", order_by="trade_date")).abs().alias("w_chg")
    )
    daily = (
        work.group_by("trade_date")
        .agg(
            (pl.col("w") * pl.col("fwd_1")).sum().alias("gross_ret"),
            pl.col("w_chg").sum().alias("turnover"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 50)
        .sort("trade_date")
        .collect()
    )
    if daily.height < 60:
        raise ValueError("回测样本不足 (有效交易日 < 60)")

    cost = cost_bps / 1e4
    daily = daily.with_columns((pl.col("gross_ret") - pl.col("turnover") * cost).alias("net_ret"))
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
    dates = [str(d) for d in daily["trade_date"]]

    return {
        "stats": {
            "days": n,
            "ann_ret": round(ann_ret, 4),
            "ann_vol": round(ann_vol, 4),
            "sharpe": round(sharpe, 3),
            "max_dd": round(max_dd, 4),
            "avg_daily_turnover": round(float(daily["turnover"].mean()), 4),
            "final_nav": round(nav, 4),
        },
        "curve": {
            "dates": dates,
            "equity": [round(v, 5) for v in equity],
            "daily_ret": [round(float(r or 0), 6) for r in net],
        },
    }
