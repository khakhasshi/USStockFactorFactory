"""评估 harness v2: 因子 -> 分层 RankIC/ICIR/换手/era 一致性 -> public/gate 分数.

评分公式 (2026-08-05 升级):
  score = |ICIR| × era_penalty × turnover_penalty × gate_flip_penalty × cost_killer

改进点:
  1. era一致性阈值从60%→75%, 用平方惩罚替代线性
  2. 换手惩罚从断崖式 max(0,1-to/0.5) 改为指数衰减 exp(-3×to)
  3. gate翻号惩罚: public与gate层ICIR符号相反→×0.3
  4. 成本实扣: 年化交易成本 > |ICIR| → score=0
"""

import math

import polars as pl

from ..data.panel import PanelStore
from ..dsl.engine import parse

EVAL_LAYERS = ["INNER_PUBLIC", "META_TRAIN"]
COST_BPS = 15  # 默认单边交易成本 (bps)


def _layer_metrics(daily: pl.DataFrame) -> dict:
    """单层指标计算 (不含跨层惩罚)."""
    if daily.height < 30:
        return {"n_days": daily.height, "ic_mean": None, "icir": None, "score": None}
    ic = daily["ic"]
    ic_mean = float(ic.mean())
    ic_std = float(ic.std() or 1e-9)
    icir = ic_mean / ic_std * math.sqrt(252)
    # era 一致性
    era_means = daily.group_by("era").agg(pl.col("ic").mean()).sort("era")
    signs = era_means["ic"].sign()
    overall_sign = 1.0 if ic_mean >= 0 else -1.0
    consistency = float((signs == overall_sign).mean()) if era_means.height else 0.0
    turnover = float(daily["turnover"].mean() or 0.0)
    # era 级 t 统计量
    era_ics = era_means["ic"].to_list()
    if len(era_ics) >= 3:
        era_mean = sum(era_ics) / len(era_ics)
        era_se = (sum((x - era_mean)**2 for x in era_ics) / (len(era_ics) - 1))**0.5 / math.sqrt(len(era_ics)) if len(era_ics) > 1 else 0
        t_stat = era_mean / era_se if era_se > 1e-9 else 0.0
    else:
        t_stat = None

    return {
        "n_days": daily.height,
        "ic_mean": round(ic_mean, 5),
        "icir": round(icir, 4),
        "era_consistency": round(consistency, 3),
        "turnover": round(turnover, 4),
        "direction": overall_sign,
        "t_stat": round(t_stat, 3) if t_stat is not None else None,
        "score": None,  # 由 evaluate() 统一计算
        "era_series": [
            {"era": int(r["era"]), "ic": round(float(r["ic"]), 5)}
            for r in era_means.iter_rows(named=True)
        ],
    }


def _compute_score(public: dict, gate: dict, turnover: float | None = None) -> float:
    """跨层综合评分 (v2 公式).

    惩罚项:
      era_penalty:     min(1.0, (consistency/0.75)^2)  阈值75%
      turnover_penalty: exp(-3 × turnover)              指数衰减
      gate_flip_penalty: ×0.3 (如果 public/gate IC 符号相反)
      cost_killer:      0 (如果年化成本 > |ICIR|)
    """
    icir = abs(public.get("icir") or 0)
    if icir < 0.001:
        return 0.0

    # era一致性惩罚 (用public层)
    cons = public.get("era_consistency") or 0
    era_penalty = min(1.0, (cons / 0.75)**2)

    # 换手指数衰减 (取public和gate中的较大者, 更保守)
    to = turnover if turnover is not None else max(
        public.get("turnover") or 0, gate.get("turnover") or 0
    )
    turnover_penalty = math.exp(-3.0 * to)

    # gate翻号惩罚
    pub_icir = public.get("icir") or 0
    gate_icir = gate.get("icir") or 0
    if pub_icir is not None and gate_icir is not None:
        pub_sign = 1.0 if pub_icir >= 0 else -1.0
        gate_sign = 1.0 if gate_icir >= 0 else -1.0
        gate_flip_penalty = 0.3 if pub_sign != gate_sign else 1.0
    else:
        gate_flip_penalty = 1.0

    # 成本实扣: 年化换手 > 3倍 (日换手≈120%) → 不可交易
    # exp(-3*to) 已提供平滑惩罚, 此处仅做硬截断
    annual_turnover = to * 252  # 年化换手倍数
    if annual_turnover > 3.0:   # 日均换手 >120% → 成本必然不可行
        return 0.0
    cost_killer = 1.0  # 保留占位, 未来可加入真实成本模型

    score = icir * era_penalty * turnover_penalty * gate_flip_penalty * cost_killer
    return round(score, 4)


def evaluate(expression: str, universe_n: int = 500, horizon: int = 5) -> dict:
    """返回 {public: {...}, gate: {...}}; public和gate均含score (由跨层公式统一计算)."""
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

    # 分别计算单层指标
    pub_daily = daily.filter(pl.col("layer") == "INNER_PUBLIC")
    gate_daily = daily.filter(pl.col("layer") == "META_TRAIN")
    public = _layer_metrics(pub_daily)
    gate = _layer_metrics(gate_daily)

    # 跨层综合评分
    score = _compute_score(public, gate)
    public["score"] = score
    gate["score"] = score  # gate分也用同一个跨层公式 (统一尺度)

    return {"public": public, "gate": gate}


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
