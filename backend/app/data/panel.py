"""AsOfResearchView 数据层: 加载 yfinance 面板, 预计算前向收益/era/四级隔离层/universe 排名.

诚实标记: 本数据源为当前成分股回看历史 (non-PIT), production_eligible=False.
"""

import glob
import hashlib
import os
import threading
import time
from datetime import date, datetime, timezone

import polars as pl

from ..config import PANEL_GLOB, get_layer_bounds

HORIZONS = [1, 5, 10, 20]

PANEL_META = {
    "pit_quality": "non_pit_current_constituents",
    "production_eligible": False,
    "universe_policy": "rolling_60d_amount_rank_v1",
}


class PanelStore:
    _instances: dict[str, "PanelStore"] = {}
    _registry_lock = threading.Lock()

    def __init__(self, panel_glob: str | None = None, market: str | None = None) -> None:
        self.df: pl.DataFrame | None = None
        self.trading_dates: list[date] = []
        self.load_error: str = ""
        self.panel_glob = panel_glob or os.environ.get("FF_PANEL_GLOB")
        self.market = market or os.environ.get("FF_MARKET", "us")
        self.layer_bounds = get_layer_bounds(self.market)
        self._load_lock = threading.Lock()
        self._diagnostic_lock = threading.Lock()
        self.load_state = "cold"
        self.load_attempts = 0
        self.load_started_at: str | None = None
        self.loaded_at: str | None = None
        self.load_duration_ms: float | None = None
        self.last_accessed_at: str | None = None
        self._loaded_summary: dict = {}
        self._inventory_cache: dict = {}
        self._inventory_cached_at = 0.0

    @classmethod
    def get(cls, panel_glob: str | None = None, market: str | None = None) -> "PanelStore":
        resolved_market = market or os.environ.get("FF_MARKET", "us")
        resolved_glob = panel_glob or os.environ.get("FF_PANEL_GLOB", "__default__")
        key = f"{resolved_market}::{resolved_glob}"
        with cls._registry_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(panel_glob, resolved_market)
            return cls._instances[key]

    def ensure_loaded(self) -> pl.DataFrame:
        self.last_accessed_at = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        if self.df is not None:
            return self.df
        # Different market panels may load concurrently; only duplicate loads
        # of this exact panel instance are serialized.
        with self._load_lock:
            if self.df is not None:
                return self.df
            self.load_attempts += 1
            self.load_state = "loading"
            self.load_error = ""
            self.load_started_at = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
            started = time.perf_counter()
            try:
                frame = self._load()
            except Exception as exc:
                self.load_state = "error"
                self.load_error = str(exc)[:1200]
                self.load_duration_ms = round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                )
                raise
            self.df = frame
            self.load_state = "ready"
            self.loaded_at = datetime.now(timezone.utc).isoformat(
                timespec="milliseconds"
            )
            self.load_duration_ms = round(
                (time.perf_counter() - started) * 1000.0,
                3,
            )
            self._loaded_summary = {
                "rows": frame.height,
                "securities": frame["ts_code"].n_unique(),
                "date_min": str(frame["trade_date"].min()),
                "date_max": str(frame["trade_date"].max()),
                "columns": len(frame.columns),
                "estimated_size_bytes": frame.estimated_size("b"),
            }
            return self.df

    def _load(self) -> pl.DataFrame:
        panel_glob = self.panel_glob or PANEL_GLOB
        lf = pl.scan_parquet(panel_glob, hive_partitioning=True)
        base_cols = [
            "trade_date", "ts_code", "name", "open", "high", "low", "close",
            "vol", "amount", "raw_open", "raw_close",
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
        execution_fields = [
            "can_buy_open_proxy", "can_sell_open_proxy",
            "up_limit", "down_limit", "adjustment_factor", "adj_factor",
        ]
        cols = (
            [column for column in base_cols if column in available]
            + [f for f in extra_fields if f in available]
            + [f for f in execution_fields if f in available]
            + quality_cols
        )

        lf = lf.select(cols)
        # 逐列过滤 (不存在的列跳过)
        for qc in quality_cols:
            lf = lf.filter(pl.col(qc))
        lf = lf.drop(quality_cols) if quality_cols else lf
        if "raw_open" not in cols:
            lf = lf.with_columns(pl.col("open").alias("raw_open"))
        if "raw_close" not in cols:
            lf = lf.with_columns(pl.col("close").alias("raw_close"))
        if "adjustment_factor" not in cols:
            if "adj_factor" in cols:
                lf = lf.with_columns(pl.col("adj_factor").alias("adjustment_factor"))
            else:
                lf = lf.with_columns(pl.lit(1.0).alias("adjustment_factor"))
        if "can_buy_open_proxy" not in cols:
            lf = lf.with_columns(pl.lit(True).alias("can_buy_open_proxy"))
        if "can_sell_open_proxy" not in cols:
            lf = lf.with_columns(pl.lit(True).alias("can_sell_open_proxy"))
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
        for name, (lo, hi) in self.layer_bounds.items():
            layer_expr = (
                pl.when(pl.col("trade_date").is_between(date.fromisoformat(lo), date.fromisoformat(hi)))
                .then(pl.lit(name))
                .otherwise(layer_expr)
            )
        lf = lf.with_columns(layer_expr.alias("layer"))
        frame = lf.collect()
        self.trading_dates = frame["trade_date"].unique().sort().to_list()
        return frame

    def summary(self, ensure_loaded: bool = True) -> dict:
        if not ensure_loaded and self.df is None:
            panel_glob = self.panel_glob or PANEL_GLOB
            try:
                columns = pl.scan_parquet(panel_glob, hive_partitioning=True).collect_schema().names()
            except Exception as e:  # noqa: BLE001
                return {
                    "loaded": False,
                    "error": str(e),
                    "market": self.market,
                    "layers": {k: list(v) for k, v in self.layer_bounds.items()},
                    **PANEL_META,
                    "panel_glob": panel_glob,
                }
            return {
                "loaded": False,
                "state": "cold",
                "market": self.market,
                "layers": {k: list(v) for k, v in self.layer_bounds.items()},
                "available_columns": columns,
                **PANEL_META,
                "panel_glob": panel_glob,
            }
        try:
            df = self.ensure_loaded()
        except Exception as e:  # noqa: BLE001
            return {"loaded": False, "error": str(e), **PANEL_META}
        loaded = self._loaded_summary or {
            "rows": df.height,
            "securities": df["ts_code"].n_unique(),
            "date_min": str(df["trade_date"].min()),
            "date_max": str(df["trade_date"].max()),
        }
        return {
            "loaded": True,
            "state": self.load_state,
            **loaded,
            "market": self.market,
            "layers": {k: list(v) for k, v in self.layer_bounds.items()},
            **PANEL_META,
            "panel_glob": self.panel_glob or PANEL_GLOB,
        }

    def _source_inventory(self) -> dict:
        """Cheap cached file identity; never loads the Polars panel."""
        now = time.monotonic()
        with self._diagnostic_lock:
            if self._inventory_cache and now - self._inventory_cached_at < 30.0:
                return dict(self._inventory_cache)
            source = self.panel_glob or PANEL_GLOB
            try:
                paths = sorted(glob.glob(source))
                sampled = paths[:5000]
                total_bytes = 0
                latest_mtime = 0.0
                identity_rows = []
                stat_errors = 0
                for path in sampled:
                    try:
                        stat = os.stat(path)
                    except OSError:
                        stat_errors += 1
                        continue
                    total_bytes += stat.st_size
                    latest_mtime = max(latest_mtime, stat.st_mtime)
                    identity_rows.append(
                        f"{path}|{stat.st_size}|{stat.st_mtime_ns}"
                    )
                identity = hashlib.sha256(
                    "\n".join(identity_rows).encode("utf-8")
                ).hexdigest()[:16]
                result = {
                    "source": source,
                    "file_count": len(paths),
                    "sampled_files": len(sampled),
                    "inventory_truncated": len(paths) > len(sampled),
                    "stat_errors": stat_errors,
                    "total_bytes": total_bytes,
                    "latest_mtime": (
                        datetime.fromtimestamp(
                            latest_mtime,
                            timezone.utc,
                        ).isoformat(timespec="seconds")
                        if latest_mtime
                        else None
                    ),
                    "identity": identity,
                    "source_error": None if paths else "panel glob 未匹配任何文件",
                }
            except (OSError, ValueError) as exc:
                result = {
                    "source": source,
                    "file_count": 0,
                    "sampled_files": 0,
                    "total_bytes": 0,
                    "latest_mtime": None,
                    "identity": None,
                    "source_error": str(exc)[:1200],
                }
            self._inventory_cache = result
            self._inventory_cached_at = now
            return dict(result)

    def diagnostics(self) -> dict:
        inventory = self._source_inventory()
        loaded = dict(self._loaded_summary)
        state = self.load_state
        if state == "cold" and inventory.get("source_error"):
            state = "error"
        return {
            "id": hashlib.sha256(
                f"{self.market}::{inventory.get('source')}".encode("utf-8")
            ).hexdigest()[:12],
            "market": self.market,
            "state": state,
            "loaded": self.df is not None,
            "load_attempts": self.load_attempts,
            "load_started_at": self.load_started_at,
            "loaded_at": self.loaded_at,
            "load_duration_ms": self.load_duration_ms,
            "last_accessed_at": self.last_accessed_at,
            "load_error": self.load_error or None,
            "layer_bounds": {k: list(v) for k, v in self.layer_bounds.items()},
            **PANEL_META,
            **inventory,
            **loaded,
        }

    @classmethod
    def registry_snapshot(cls) -> dict:
        with cls._registry_lock:
            stores = list(cls._instances.values())
        panels = [store.diagnostics() for store in stores]
        return {
            "instances": len(panels),
            "loaded": sum(bool(row.get("loaded")) for row in panels),
            "loading": sum(row.get("state") == "loading" for row in panels),
            "errors": sum(row.get("state") == "error" for row in panels),
            "estimated_size_bytes": sum(
                int(row.get("estimated_size_bytes") or 0) for row in panels
            ),
            "panels": panels,
        }
