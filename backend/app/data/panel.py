"""AsOfResearchView 数据层: 加载 yfinance 面板, 预计算前向收益/era/四级隔离层/universe 排名.

诚实标记: 本数据源为当前成分股回看历史 (non-PIT), production_eligible=False.
"""

import threading
from datetime import date

import polars as pl

from ..config import LAYER_BOUNDS, PANEL_GLOB

HORIZONS = [1, 5, 10, 20]

PANEL_META = {
    "pit_quality": "non_pit_current_constituents",
    "production_eligible": False,
    "universe_policy": "rolling_60d_amount_rank_v1",
}


class PanelStore:
    _instance = None
    _lock = threading.Lock()

    def __init__(self) -> None:
        self.df: pl.DataFrame | None = None
        self.load_error: str = ""

    @classmethod
    def get(cls) -> "PanelStore":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def ensure_loaded(self) -> pl.DataFrame:
        if self.df is not None:
            return self.df
        with self._lock:
            if self.df is not None:
                return self.df
            self.df = self._load()
            return self.df

    def _load(self) -> pl.DataFrame:
        lf = pl.scan_parquet(PANEL_GLOB, hive_partitioning=True)
        base_cols = [
            "trade_date", "ts_code", "open", "high", "low", "close",
            "vol", "amount", "raw_open",
        ]
        # 额外字段: A股估值/市值/流动性/资金流向 (按存在性自适应)
        extra_fields = [
            "pe_ttm", "pb", "ps_ttm", "dv_ttm",
            "total_mv", "circ_mv",
            "turnover_rate", "volume_ratio",
            "net_mf_amount", "buy_lg_amount", "sell_lg_amount",
            "buy_elg_amount", "sell_elg_amount",
            "float_share",
        ]
        # 质量过滤列: 按存在性自适应
        available = set(lf.collect_schema().names())
        quality_cols = []
        for c in ["is_tradable_observation", "is_valid_ohlc", "is_security_identity_consistent"]:
            if c in available:
                quality_cols.append(c)
        cols = base_cols + [f for f in extra_fields if f in available] + quality_cols

        lf = lf.select(cols)
        # 逐列过滤 (不存在的列跳过)
        for qc in quality_cols:
            lf = lf.filter(pl.col(qc))
        lf = lf.drop(quality_cols) if quality_cols else lf
        lf = lf.with_columns(pl.col("trade_date").cast(pl.Date)).sort("ts_code", "trade_date")

        by_code = {"partition_by": "ts_code", "order_by": "trade_date"}
        # 前向收益: t 日信号 -> t+1 开盘成交 -> t+1+h 开盘平仓 (前复权 open 口径)
        fwd_cols = [
            (pl.col("open").shift(-(1 + h)).over(**by_code) / pl.col("open").shift(-1).over(**by_code) - 1)
            .alias(f"fwd_{h}")
            for h in HORIZONS
        ]
        lf = lf.with_columns(
            *fwd_cols,
            pl.col("amount").rolling_mean(60, min_samples=20).over(**by_code).alias("amt60"),
        )
        # universe 排名 (按 60 日均成交额, 每日截面)
        lf = lf.with_columns(
            pl.col("amt60").rank(method="ordinal", descending=True).over("trade_date").alias("univ_rank")
        )
        # era (半年) 与四级隔离层
        lf = lf.with_columns(
            (pl.col("trade_date").dt.year() * 10 + ((pl.col("trade_date").dt.month() > 6).cast(pl.Int32) + 1))
            .alias("era")
        )
        layer_expr = pl.lit("NONE")
        for name, (lo, hi) in LAYER_BOUNDS.items():
            layer_expr = (
                pl.when(pl.col("trade_date").is_between(date.fromisoformat(lo), date.fromisoformat(hi)))
                .then(pl.lit(name))
                .otherwise(layer_expr)
            )
        lf = lf.with_columns(layer_expr.alias("layer"))
        return lf.collect()

    def summary(self) -> dict:
        try:
            df = self.ensure_loaded()
        except Exception as e:  # noqa: BLE001
            return {"loaded": False, "error": str(e), **PANEL_META}
        return {
            "loaded": True,
            "rows": df.height,
            "securities": df["ts_code"].n_unique(),
            "date_min": str(df["trade_date"].min()),
            "date_max": str(df["trade_date"].max()),
            "layers": {k: list(v) for k, v in LAYER_BOUNDS.items()},
            **PANEL_META,
        }
