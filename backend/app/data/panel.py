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

from ..config import (
    PANEL_GLOB,
    default_panel_glob,
    get_dsl_fields,
    get_layer_bounds,
)

HORIZONS = [1, 5, 10, 20]
REQUIRED_PANEL_COLUMNS = {
    "trade_date",
    "ts_code",
    "open",
    "high",
    "low",
    "close",
    "vol",
    "amount",
}

PANEL_META = {
    "pit_quality": "non_pit_current_constituents",
    "production_eligible": False,
    "universe_policy": "rolling_60d_amount_rank_v1",
}


class PanelStore:
    _instances: dict[str, "PanelStore"] = {}
    _registry_lock = threading.Lock()

    def __init__(
        self,
        panel_glob: str | None = None,
        market: str | None = None,
        factor_fields: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.df: pl.DataFrame | None = None
        self.trading_dates: list[date] = []
        self.load_error: str = ""
        self.market = market or os.environ.get("FF_MARKET", "us")
        self.factor_fields = tuple(sorted(set(
            factor_fields or get_dsl_fields(self.market)
        )))
        # An explicitly requested market must never inherit the process-wide
        # panel.  The single-port service normally boots with FF_MARKET=us,
        # while A-share backtests are selected per task at request time.
        self.panel_glob = (
            panel_glob
            or (
                default_panel_glob(self.market)
                if market is not None
                else os.environ.get("FF_PANEL_GLOB")
                or default_panel_glob(self.market)
            )
        )
        self.layer_bounds = get_layer_bounds(self.market)
        self._load_lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self._snapshot_lock = threading.Lock()
        self._diagnostic_lock = threading.Lock()
        self.load_state = "cold"
        self.load_attempts = 0
        self.load_started_at: str | None = None
        self.initial_loaded_at: str | None = None
        self.loaded_at: str | None = None
        self.load_duration_ms: float | None = None
        self.last_accessed_at: str | None = None
        self.loaded_identity: str | None = None
        self.generation = 0
        self.reload_state = "idle"
        self.reload_attempts = 0
        self.reload_count = 0
        self.reload_started_at: str | None = None
        self.reloaded_at: str | None = None
        self.reload_duration_ms: float | None = None
        self.reload_error: str = ""
        self.last_change_detected_at: str | None = None
        self._pending_identity: str | None = None
        self._pending_identity_checks = 0
        self._loaded_summary: dict = {}
        self._inventory_cache: dict = {}
        self._inventory_cached_at = 0.0

    @classmethod
    def get(
        cls,
        panel_glob: str | None = None,
        market: str | None = None,
        factor_fields: list[str] | tuple[str, ...] | None = None,
    ) -> "PanelStore":
        resolved_market = market or os.environ.get("FF_MARKET", "us")
        resolved_fields = tuple(sorted(set(
            factor_fields or get_dsl_fields(resolved_market)
        )))
        resolved_glob = (
            panel_glob
            or (
                default_panel_glob(resolved_market)
                if market is not None
                else os.environ.get("FF_PANEL_GLOB")
                or default_panel_glob(resolved_market)
            )
        )
        fields_key = ",".join(resolved_fields)
        key = f"{resolved_market}::{resolved_glob}::{fields_key}"
        with cls._registry_lock:
            if key not in cls._instances:
                cls._instances[key] = cls(
                    resolved_glob,
                    resolved_market,
                    resolved_fields,
                )
            return cls._instances[key]

    def ensure_loaded(self) -> pl.DataFrame:
        self.last_accessed_at = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )
        with self._snapshot_lock:
            if self.df is not None:
                return self.df
        # Different market panels may load concurrently; only duplicate loads
        # of this exact panel instance are serialized.
        with self._load_lock:
            with self._snapshot_lock:
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
                frame, trading_dates = self._load()
                inventory = self._source_inventory(force=True)
            except Exception as exc:
                self.load_state = "error"
                self.load_error = str(exc)[:1200]
                self.load_duration_ms = round(
                    (time.perf_counter() - started) * 1000.0,
                    3,
                )
                raise
            duration_ms = round(
                (time.perf_counter() - started) * 1000.0,
                3,
            )
            self._commit_snapshot(
                frame,
                trading_dates,
                source_identity=inventory.get("identity"),
                duration_ms=duration_ms,
                reloaded=False,
            )
            return frame

    def read_snapshot(
        self,
    ) -> tuple[pl.DataFrame, tuple[date, ...], str | None, int]:
        """Return one internally consistent panel generation.

        A hot reload swaps ``df`` and ``trading_dates`` under one lock. Existing
        callers retain their reference to the old immutable Polars frame while
        new callers receive the new generation, so no request observes a
        half-swapped calendar/frame pair.
        """
        self.ensure_loaded()
        with self._snapshot_lock:
            assert self.df is not None
            return (
                self.df,
                tuple(self.trading_dates),
                self.loaded_identity,
                self.generation,
            )

    @staticmethod
    def _frame_summary(frame: pl.DataFrame) -> dict:
        return {
            "rows": frame.height,
            "securities": frame["ts_code"].n_unique(),
            "date_min": str(frame["trade_date"].min()),
            "date_max": str(frame["trade_date"].max()),
            "columns": len(frame.columns),
            "estimated_size_bytes": frame.estimated_size("b"),
        }

    def _commit_snapshot(
        self,
        frame: pl.DataFrame,
        trading_dates: list[date],
        *,
        source_identity: str | None,
        duration_ms: float,
        reloaded: bool,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        summary = self._frame_summary(frame)
        with self._snapshot_lock:
            self.df = frame
            self.trading_dates = trading_dates
            self.loaded_identity = source_identity
            self.generation += 1
            self._loaded_summary = summary
            self.loaded_at = now
            if self.initial_loaded_at is None:
                self.initial_loaded_at = now
            self.load_state = "ready"
            self.load_error = ""
            if reloaded:
                self.reload_count += 1
                self.reloaded_at = now
                self.reload_duration_ms = duration_ms
            else:
                self.load_duration_ms = duration_ms
            self.reload_state = "idle"
            self.reload_error = ""
            self._pending_identity = None
            self._pending_identity_checks = 0

    def _load(self) -> tuple[pl.DataFrame, list[date]]:
        panel_glob = self.panel_glob or default_panel_glob(self.market)
        lf = pl.scan_parquet(panel_glob, hive_partitioning=True)
        base_cols = [
            "trade_date", "ts_code", "name", "open", "high", "low", "close",
            "vol", "amount", "raw_open", "raw_high", "raw_low", "raw_close",
        ]
        # Normal service instances keep the native task whitelist.  External
        # frozen-library workers can request a larger, explicit field set;
        # those fields become part of the PanelStore cache identity and do not
        # silently broaden the live research grammar.
        extra_fields = list(self.factor_fields)
        # 质量过滤列: 按存在性自适应
        available = set(lf.collect_schema().names())
        expected = set(self.factor_fields) | REQUIRED_PANEL_COLUMNS
        missing = sorted(expected - available)
        if missing:
            raise ValueError(
                f"{self.market} 面板缺少系统已声明可用的字段: "
                f"{', '.join(missing)}"
            )
        quality_cols = []
        for c in ["is_tradable_observation", "is_valid_ohlc", "is_security_identity_consistent"]:
            if c in available:
                quality_cols.append(c)
        execution_fields = [
            "can_buy_open_proxy", "can_sell_open_proxy",
            "up_limit", "down_limit", "adjustment_factor", "adj_factor",
        ]
        cols = list(dict.fromkeys(
            [column for column in base_cols if column in available]
            + [f for f in extra_fields if f in available]
            + [f for f in execution_fields if f in available]
            + quality_cols
        ))

        lf = lf.select(cols)
        # 逐列过滤 (不存在的列跳过)
        for qc in quality_cols:
            lf = lf.filter(pl.col(qc))
        lf = lf.drop(quality_cols) if quality_cols else lf
        if "raw_open" not in cols:
            lf = lf.with_columns(pl.col("open").alias("raw_open"))
        if "raw_high" not in cols:
            lf = lf.with_columns(pl.col("high").alias("raw_high"))
        if "raw_low" not in cols:
            lf = lf.with_columns(pl.col("low").alias("raw_low"))
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
        trading_dates = frame["trade_date"].unique().sort().to_list()
        if not trading_dates:
            raise ValueError(f"{self.market} 面板没有有效交易日")
        return frame, trading_dates

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

    def _source_inventory(self, *, force: bool = False) -> dict:
        """Cheap cached file identity; never loads the Polars panel."""
        now = time.monotonic()
        with self._diagnostic_lock:
            if (
                not force
                and self._inventory_cache
                and now - self._inventory_cached_at < 30.0
            ):
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
                schema_columns: list[str] = []
                schema_error: str | None = None
                if paths:
                    try:
                        schema_columns = (
                            pl.scan_parquet(
                                source,
                                hive_partitioning=True,
                            )
                            .collect_schema()
                            .names()
                        )
                    except Exception as exc:  # noqa: BLE001
                        schema_error = str(exc)[:1200]
                required_columns = sorted(REQUIRED_PANEL_COLUMNS)
                expected_dsl_fields = get_dsl_fields(self.market)
                missing_required = sorted(
                    set(required_columns) - set(schema_columns)
                )
                missing_dsl = sorted(
                    set(expected_dsl_fields) - set(schema_columns)
                )
                schema_status = (
                    "error"
                    if schema_error or missing_required or missing_dsl
                    else "ok" if paths
                    else "unavailable"
                )
                result = {
                    "source": source,
                    "inventory_checked_at": datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    ),
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
                    "schema_status": schema_status,
                    "schema_error": schema_error,
                    "available_columns": schema_columns,
                    "available_column_count": len(schema_columns),
                    "required_columns": required_columns,
                    "missing_required_columns": missing_required,
                    "expected_dsl_fields": expected_dsl_fields,
                    "missing_dsl_fields": missing_dsl,
                }
            except (OSError, ValueError) as exc:
                result = {
                    "source": source,
                    "inventory_checked_at": datetime.now(timezone.utc).isoformat(
                        timespec="milliseconds"
                    ),
                    "file_count": 0,
                    "sampled_files": 0,
                    "total_bytes": 0,
                    "latest_mtime": None,
                    "identity": None,
                    "source_error": str(exc)[:1200],
                    "schema_status": "error",
                    "schema_error": str(exc)[:1200],
                    "available_columns": [],
                    "available_column_count": 0,
                    "required_columns": sorted(REQUIRED_PANEL_COLUMNS),
                    "missing_required_columns": sorted(REQUIRED_PANEL_COLUMNS),
                    "expected_dsl_fields": get_dsl_fields(self.market),
                    "missing_dsl_fields": get_dsl_fields(self.market),
                }
            self._inventory_cache = result
            self._inventory_cached_at = now
            return dict(result)

    def reload_if_changed(
        self,
        *,
        force: bool = False,
        require_stable: bool = False,
    ) -> dict:
        """Reload a changed source with a double-buffered atomic swap.

        ``require_stable`` is used by the automatic watcher: the same changed
        identity must be observed twice before a multi-gigabyte reload starts.
        Manual reloads skip that debounce. A failed or concurrently-changing
        source never replaces the currently serving generation.
        """
        inventory = self._source_inventory(force=True)
        source_identity = inventory.get("identity")
        with self._snapshot_lock:
            loaded = self.df is not None
            loaded_identity = self.loaded_identity
            generation = self.generation
        if not loaded:
            try:
                self.ensure_loaded()
            except Exception as exc:  # noqa: BLE001
                return {
                    "status": "load_failed",
                    "market": self.market,
                    "generation": generation,
                    "serving_continued": False,
                    "error": str(exc)[:1200],
                }
            return {
                "status": "loaded",
                "market": self.market,
                "generation": self.generation,
                "source_identity": self.loaded_identity,
                "serving_continued": False,
            }
        if inventory.get("source_error") or inventory.get("schema_status") == "error":
            error = (
                inventory.get("source_error")
                or inventory.get("schema_error")
                or "面板数据契约失败"
            )
            with self._snapshot_lock:
                self.reload_state = "error"
                self.reload_error = str(error)[:1200]
            return {
                "status": "reload_failed",
                "market": self.market,
                "generation": generation,
                "source_identity": source_identity,
                "loaded_identity": loaded_identity,
                "serving_continued": True,
                "error": str(error)[:1200],
            }
        if not force and source_identity == loaded_identity:
            with self._snapshot_lock:
                self._pending_identity = None
                self._pending_identity_checks = 0
                if self.reload_state != "reloading":
                    self.reload_state = "idle"
                    self.reload_error = ""
            return {
                "status": "unchanged",
                "market": self.market,
                "generation": generation,
                "source_identity": source_identity,
                "serving_continued": True,
            }
        if require_stable and not force:
            with self._snapshot_lock:
                if self._pending_identity != source_identity:
                    self._pending_identity = source_identity
                    self._pending_identity_checks = 1
                    self.last_change_detected_at = datetime.now(
                        timezone.utc
                    ).isoformat(timespec="milliseconds")
                else:
                    self._pending_identity_checks += 1
                checks = self._pending_identity_checks
            if checks < 2:
                return {
                    "status": "change_detected",
                    "market": self.market,
                    "generation": generation,
                    "source_identity": source_identity,
                    "loaded_identity": loaded_identity,
                    "stable_checks": checks,
                    "serving_continued": True,
                }
        if not self._reload_lock.acquire(blocking=False):
            return {
                "status": "already_reloading",
                "market": self.market,
                "generation": generation,
                "source_identity": source_identity,
                "loaded_identity": loaded_identity,
                "serving_continued": True,
            }
        started = time.perf_counter()
        try:
            with self._snapshot_lock:
                self.reload_attempts += 1
                self.reload_state = "reloading"
                self.reload_error = ""
                self.reload_started_at = datetime.now(timezone.utc).isoformat(
                    timespec="milliseconds"
                )
            frame, trading_dates = self._load()
            final_inventory = self._source_inventory(force=True)
            final_identity = final_inventory.get("identity")
            if (
                final_inventory.get("source_error")
                or final_inventory.get("schema_status") == "error"
            ):
                raise ValueError(
                    final_inventory.get("source_error")
                    or final_inventory.get("schema_error")
                    or "热重载后的面板数据契约失败"
                )
            if final_identity != source_identity:
                with self._snapshot_lock:
                    self.reload_state = "deferred"
                    self.reload_error = "源文件在重载过程中再次变化，已保留旧面板"
                    self._pending_identity = final_identity
                    self._pending_identity_checks = 1
                    self.last_change_detected_at = datetime.now(
                        timezone.utc
                    ).isoformat(timespec="milliseconds")
                return {
                    "status": "source_changed_during_reload",
                    "market": self.market,
                    "generation": generation,
                    "source_identity": final_identity,
                    "loaded_identity": loaded_identity,
                    "serving_continued": True,
                }
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            self._commit_snapshot(
                frame,
                trading_dates,
                source_identity=final_identity,
                duration_ms=duration_ms,
                reloaded=True,
            )
            return {
                "status": "reloaded",
                "market": self.market,
                "generation": self.generation,
                "previous_generation": generation,
                "source_identity": final_identity,
                "previous_identity": loaded_identity,
                "duration_ms": duration_ms,
                "rows": frame.height,
                "date_max": str(trading_dates[-1]),
                "serving_continued": True,
            }
        except Exception as exc:  # noqa: BLE001
            duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
            with self._snapshot_lock:
                self.reload_state = "error"
                self.reload_error = str(exc)[:1200]
                self.reload_duration_ms = duration_ms
            return {
                "status": "reload_failed",
                "market": self.market,
                "generation": generation,
                "source_identity": source_identity,
                "loaded_identity": loaded_identity,
                "duration_ms": duration_ms,
                "serving_continued": True,
                "error": str(exc)[:1200],
            }
        finally:
            self._reload_lock.release()

    def diagnostics(self) -> dict:
        inventory = self._source_inventory()
        with self._snapshot_lock:
            loaded = dict(self._loaded_summary)
            is_loaded = self.df is not None
            loaded_identity = self.loaded_identity
            generation = self.generation
            reload_state = self.reload_state
            reload_error = self.reload_error or None
            pending_identity = self._pending_identity
            pending_checks = self._pending_identity_checks
        source_identity = inventory.get("identity")
        stale = bool(
            is_loaded
            and loaded_identity
            and source_identity
            and loaded_identity != source_identity
        )
        state = self.load_state
        if state == "cold" and (
            inventory.get("source_error")
            or inventory.get("schema_status") == "error"
        ):
            state = "error"
        elif reload_state == "reloading":
            state = "reloading"
        elif stale:
            state = "stale"
        return {
            "id": hashlib.sha256(
                f"{self.market}::{inventory.get('source')}".encode("utf-8")
            ).hexdigest()[:12],
            "market": self.market,
            "state": state,
            "loaded": is_loaded,
            "load_attempts": self.load_attempts,
            "load_started_at": self.load_started_at,
            "initial_loaded_at": self.initial_loaded_at,
            "loaded_at": self.loaded_at,
            "load_duration_ms": self.load_duration_ms,
            "last_accessed_at": self.last_accessed_at,
            "load_error": self.load_error or None,
            "generation": generation,
            "loaded_identity": loaded_identity,
            "source_identity": source_identity,
            "stale": stale,
            "reload_state": reload_state,
            "reload_attempts": self.reload_attempts,
            "reload_count": self.reload_count,
            "reload_started_at": self.reload_started_at,
            "reloaded_at": self.reloaded_at,
            "reload_duration_ms": self.reload_duration_ms,
            "reload_error": reload_error,
            "last_change_detected_at": self.last_change_detected_at,
            "pending_identity": pending_identity,
            "pending_identity_checks": pending_checks,
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
            "reloading": sum(row.get("state") == "reloading" for row in panels),
            "stale": sum(bool(row.get("stale")) for row in panels),
            "reload_errors": sum(bool(row.get("reload_error")) for row in panels),
            "errors": sum(row.get("state") == "error" for row in panels),
            "estimated_size_bytes": sum(
                int(row.get("estimated_size_bytes") or 0) for row in panels
            ),
            "panels": panels,
        }

    @classmethod
    def reload_changed_instances(cls) -> list[dict]:
        """Check and reload every currently serving panel generation."""
        with cls._registry_lock:
            stores = list(cls._instances.values())
        results = []
        for store in stores:
            with store._snapshot_lock:
                loaded = store.df is not None
            if not loaded:
                continue
            result = store.reload_if_changed(require_stable=True)
            if result.get("status") != "unchanged":
                results.append(result)
        return results
