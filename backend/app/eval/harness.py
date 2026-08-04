"""评估 harness: 因子 -> 分层 RankIC/ICIR/换手/era 一致性 -> public/gate 分数.

Miner 只拿得到 public (INNER_PUBLIC 层); gate (META_TRAIN) 仅存库供外层与晋级使用;
META_HOLDOUT / FACTOR_VAULT 不在常规评估路径内。
"""

import math

import polars as pl

from ..data.panel import PanelStore
from ..dsl.engine import parse

EVAL_LAYERS = ["INNER_PUBLIC", "META_TRAIN"]


def _layer_metrics(daily: pl.DataFrame) -> dict:
    if daily.height < 30:
        return {"n_days": daily.height, "ic_mean": None, "icir": None, "score": None}
    ic = daily["ic"]
    ic_mean = float(ic.mean())
    ic_std = float(ic.std() or 1e-9)
    icir = ic_mean / ic_std * math.sqrt(252)
    # era 一致性: era 均值与总体同号的比例
    era_means = daily.group_by("era").agg(pl.col("ic").mean()).sort("era")
    signs = era_means["ic"].sign()
    overall_sign = 1.0 if ic_mean >= 0 else -1.0
    consistency = float((signs == overall_sign).mean()) if era_means.height else 0.0
    turnover = float(daily["turnover"].mean() or 0.0)
    # 分数: |ICIR| × era 一致性惩罚 × 换手硬截断 (日换手 ≥50% 归零; 实验1的 exp 衰减惩罚过弱)
    score = abs(icir) * min(1.0, consistency / 0.6) * max(0.0, 1.0 - turnover / 0.5)
    return {
        "n_days": daily.height,
        "ic_mean": round(ic_mean, 5),
        "icir": round(icir, 4),
        "era_consistency": round(consistency, 3),
        "turnover": round(turnover, 4),
        "direction": overall_sign,
        "score": round(score, 4),
        "era_series": [
            {"era": int(r["era"]), "ic": round(float(r["ic"]), 5)}
            for r in era_means.iter_rows(named=True)
        ],
    }


def evaluate(expression: str, universe_n: int = 500, horizon: int = 5) -> dict:
    """返回 {public: {...}, gate: {...}}; 表达式非法/数据不足抛 ValueError."""
    df = PanelStore.get().ensure_loaded()
    pipe = parse(expression)
    fwd = f"fwd_{horizon}"
    if fwd not in df.columns:
        raise ValueError(f"不支持的 horizon: {horizon}")

    work = (
        pipe.apply(df.lazy().filter(pl.col("layer").is_in(EVAL_LAYERS)))
        .filter(pl.col("univ_rank") <= universe_n)
        .filter(pl.col("factor").is_finite() & pl.col(fwd).is_finite())
        .with_columns(
            pl.col("factor").rank().over("trade_date").alias("f_rank"),
            pl.col(fwd).rank().over("trade_date").alias("r_rank"),
        )
        .with_columns(
            # 换手代理: 因子截面分位相对前一日的平均绝对变化
            (pl.col("f_rank") / pl.col("f_rank").count().over("trade_date")).alias("f_pct")
        )
        .with_columns(
            (pl.col("f_pct") - pl.col("f_pct").shift(1).over("ts_code", order_by="trade_date"))
            .abs()
            .alias("f_chg")
        )
    )
    daily = (
        work.group_by("trade_date", "layer", "era")
        .agg(
            pl.corr("f_rank", "r_rank").alias("ic"),
            pl.col("f_chg").mean().alias("turnover"),
            pl.len().alias("n"),
        )
        .filter(pl.col("n") >= 50)
        .drop_nulls("ic")
        .filter(pl.col("ic").is_not_nan())
        .sort("trade_date")
        .collect()
    )
    if daily.height == 0:
        raise ValueError("有效评估样本为空 (表达式可能全为 null 或退化为常数)")

    out = {}
    for layer, key in [("INNER_PUBLIC", "public"), ("META_TRAIN", "gate")]:
        out[key] = _layer_metrics(daily.filter(pl.col("layer") == layer))
    return out


def era_detail(expression: str, universe_n: int = 500, horizon: int = 5) -> dict:
    """步进可视化数据: 全部四层的 era 级 IC 序列 (只读展示, 不参与优化信号)."""
    df = PanelStore.get().ensure_loaded()
    pipe = parse(expression)
    fwd = f"fwd_{horizon}"
    daily = (
        pipe.apply(df.lazy())
        .filter(pl.col("univ_rank") <= universe_n)
        .filter(pl.col("factor").is_finite() & pl.col(fwd).is_finite())
        .with_columns(
            pl.col("factor").rank().over("trade_date").alias("f_rank"),
            pl.col(fwd).rank().over("trade_date").alias("r_rank"),
        )
        .group_by("trade_date", "layer", "era")
        .agg(pl.corr("f_rank", "r_rank").alias("ic"), pl.len().alias("n"))
        .filter(pl.col("n") >= 50)
        .drop_nulls("ic")
        .filter(pl.col("ic").is_not_nan())
        .collect()
    )
    eras = (
        daily.group_by("era", "layer")
        .agg(pl.col("ic").mean().alias("ic_mean"), pl.col("ic").std().alias("ic_std"), pl.len().alias("days"))
        .sort("era")
    )
    layers: dict[str, dict] = {}
    for layer in ["INNER_PUBLIC", "META_TRAIN", "META_HOLDOUT", "FACTOR_VAULT"]:
        sub = daily.filter(pl.col("layer") == layer)
        if sub.height < 10:
            continue
        ic_mean = float(sub["ic"].mean())
        ic_std = float(sub["ic"].std() or 1e-9)
        icir = ic_mean / ic_std * math.sqrt(252)
        era_means = sub.group_by("era").agg(pl.col("ic").mean())
        sign = 1.0 if ic_mean >= 0 else -1.0
        consistency = float((era_means["ic"].sign() == sign).mean()) if era_means.height else 0.0
        layers[layer] = {
            "ic_mean": round(ic_mean, 5),
            "icir": round(icir, 4),
            "era_consistency": round(consistency, 3),
            "score": round(abs(icir) * min(1.0, consistency / 0.6), 4),
        }
    return {
        "layers": layers,
        "eras": [
            {
                "era": int(r["era"]),
                "layer": r["layer"],
                "ic_mean": round(float(r["ic_mean"]), 5),
                "ic_std": round(float(r["ic_std"] or 0), 5),
                "days": int(r["days"]),
            }
            for r in eras.iter_rows(named=True)
        ],
    }
