"""Bounded cache of causal DSL columns, never scores or validation outcomes.

Exact source intervals are intentionally separate: finite warmup backtests must
not consume all-history rolling statistics whose floating arithmetic may differ.
"""
from __future__ import annotations

import ast
from collections import OrderedDict, Counter
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import threading
import time

import polars as pl

from .dsl.engine import parse
from .dsl.operators_v2 import WINDOW_ARGUMENTS

PROTOCOL = "factorfactory.causal-column-cache/v1"
_ENGINE_HASH = hashlib.sha256((Path(__file__).parent / "dsl/engine.py").read_bytes()).hexdigest()
_V2_HASH = hashlib.sha256((Path(__file__).parent / "dsl/operators_v2.py").read_bytes()).hexdigest()
_MODULE_HASH = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_WINDOW_CALLS = frozenset({"delay", "returns", "ts_mean", "ts_std", "ts_sum", "ts_min", "ts_max",
    "ts_delta", "ts_rank", "ts_corr", "ts_quantile", "ts_slope", "ts_rsquare", "ts_resi",
    "ts_argmax", "ts_argmin", "rank", "zscore", "winsor", "winsor_mad"})
_WINDOW_CALLS = _WINDOW_CALLS | set(WINDOW_ARGUMENTS) | {"cs_residual", "group_rank", "group_zscore"}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


class ColumnCache:
    def __init__(self, max_bytes: int, disk_root: Path | None = None, disk_bytes: int = 0):
        self.max_bytes = max(0, max_bytes)
        self.disk_root = disk_root
        self.disk_bytes = max(0, disk_bytes)
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = threading.RLock()
        self.flights = [threading.RLock() for _ in range(64)]
        self.counters = Counter()

    def flight(self, key):
        return self.flights[int(key[:8], 16) % len(self.flights)]

    def get(self, key, rows, *, disk=False):
        with self.lock:
            value = self.entries.get(key)
            if value is not None and len(value) == rows:
                self.entries.move_to_end(key)
                self.counters["memory_hits"] += 1
                return value.clone()
        if disk and self.disk_bytes and self.disk_root:
            try:
                meta_path = self.disk_root / f"{key}.json"
                data_path = self.disk_root / f"{key}.arrow"
                meta = json.loads(meta_path.read_text())
                if meta["key"] != key or meta["rows"] != rows or meta["protocol"] != PROTOCOL:
                    raise ValueError("cache metadata mismatch")
                if data_path.stat().st_size > self.disk_bytes:
                    raise ValueError("cache file exceeds budget")
                blob = data_path.read_bytes()
                if hashlib.sha256(blob).hexdigest() != meta["sha256"]:
                    raise ValueError("cache checksum mismatch")
                value = pl.read_ipc(blob).to_series()
                if len(value) != rows or str(value.dtype) != meta["dtype"]:
                    raise ValueError("cache column mismatch")
                self.put(key, value)
                with self.lock:
                    self.counters["disk_hits"] += 1
                return value
            except FileNotFoundError:
                pass
            except Exception:
                with self.lock:
                    self.counters["disk_invalid_or_unavailable"] += 1
        with self.lock:
            self.counters["misses"] += 1
        return None

    def put(self, key, value, *, disk=False):
        size = value.estimated_size()
        with self.lock:
            if size <= self.max_bytes and self.max_bytes:
                prior = self.entries.pop(key, None)
                if prior is not None:
                    self.bytes -= prior.estimated_size()
                while self.entries and self.bytes + size > self.max_bytes:
                    _, removed = self.entries.popitem(last=False)
                    self.bytes -= removed.estimated_size()
                    self.counters["evictions"] += 1
                self.entries[key] = value.clone()
                self.bytes += size
        # Optional disposable on-disk final columns; intermediates stay in RAM.
        # A full column is written only when small enough for the disk budget.
        if disk and self.disk_bytes and self.disk_root and size <= self.disk_bytes // 2:
            try:
                import fcntl
                self.disk_root.mkdir(parents=True, exist_ok=True)
                with (self.disk_root / ".lock").open("a") as lock_file:
                    fcntl.flock(lock_file, fcntl.LOCK_EX)
                    import io
                    buffer = io.BytesIO()
                    value.rename("value").to_frame().write_ipc(buffer, compression="lz4")
                    blob = buffer.getvalue()
                    entries = sorted(self.disk_root.glob("*.arrow"), key=lambda p: p.stat().st_mtime_ns)
                    total = sum(p.stat().st_size for p in entries)
                    for old in entries:
                        if total + len(blob) <= self.disk_bytes:
                            break
                        if len(old.stem) == 64 and all(c in "0123456789abcdef" for c in old.stem):
                            total -= old.stat().st_size
                            old.unlink(missing_ok=True)
                            old.with_suffix(".json").unlink(missing_ok=True)
                    if len(blob) > self.disk_bytes:
                        return
                    tmp = self.disk_root / f"{key}.{os.getpid()}.tmp"
                    tmp.write_bytes(blob)
                    tmp.replace(self.disk_root / f"{key}.arrow")
                    meta = {"key": key, "rows": len(value), "dtype": str(value.dtype),
                            "protocol": PROTOCOL, "sha256": hashlib.sha256(blob).hexdigest()}
                    tmp.write_text(json.dumps(meta))
                    tmp.replace(self.disk_root / f"{key}.json")
                    self.counters["disk_writes"] += 1
            except Exception:
                with self.lock:
                    self.counters["disk_write_errors"] += 1

    def stats(self):
        with self.lock:
            return {"protocol": PROTOCOL, "enabled": os.getenv("FF_FACTOR_COMPUTE_CACHE", "1") != "0" and self.max_bytes > 0,
                    "entries": len(self.entries), "memory_bytes": self.bytes,
                    "memory_limit_bytes": self.max_bytes, "disk_limit_bytes": self.disk_bytes,
                    **dict(self.counters)}


_CACHE = ColumnCache(
    int(os.getenv("FF_FACTOR_CACHE_MB", "384")) * 1024**2,
    Path(__file__).resolve().parents[2] / "var/compute_cache/causal_columns_v1",
    int(os.getenv("FF_FACTOR_CACHE_DISK_MB", "0")) * 1024**2,
)
_LAYOUTS = OrderedDict()
_LAYOUT_LOCK = threading.Lock()


def cache_stats():
    return _CACHE.stats()


def _layout(df, snapshot):
    token = (id(snapshot.store), snapshot.generation, snapshot.loaded_identity,
             snapshot.manifest["snapshot_id"], df.height)
    with _LAYOUT_LOCK:
        item = _LAYOUTS.get(token)
        if item is not None and item[0] is snapshot.store:
            _LAYOUTS.move_to_end(token)
            return item[1]
        digest = hashlib.sha256(df.select("ts_code", "trade_date").hash_rows(seed=91024).to_numpy().tobytes()).hexdigest()
        _LAYOUTS[token] = (snapshot.store, digest)
        while len(_LAYOUTS) > 4:
            _LAYOUTS.popitem(last=False)
        return digest


def factor_column(df: pl.DataFrame, expression: str, fields: list[str], market: str,
                  *, snapshot=None, start=None, end=None):
    """Return a full-row-aligned column or None to use the original lazy plan.

    Caller supplies the unfiltered frozen panel; cache values outside the exact
    source interval are null. No labels, directions, scores or costs are stored.
    """
    if os.getenv("FF_FACTOR_COMPUTE_CACHE", "1") == "0" or _CACHE.max_bytes == 0:
        return None
    if snapshot is None:
        from .audit_snapshot import active_snapshot
        snapshot = active_snapshot()
    if snapshot is None or not snapshot.manifest.get("immutable_inputs_available"):
        return None
    if snapshot.market != market or df is not snapshot.frame:
        return None
    # Always validate before lookup, including on a cache hit.
    reference = parse(expression, fields)
    root = ast.parse(expression, mode="eval").body
    nodes = {ast.dump(n, include_attributes=False): n for n in ast.walk(root)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _WINDOW_CALLS}
    root_key = ast.dump(root, include_attributes=False)
    context = {"protocol": PROTOCOL, "snapshot": snapshot.manifest["snapshot_id"],
               "source_code": snapshot.manifest["code"]["code_sha256"],
               "dsl_code": _ENGINE_HASH, "dsl_v2_code": _V2_HASH, "cache_code": _MODULE_HASH, "polars": pl.__version__,
               "market": market, "fields": sorted(fields), "layout": _layout(df, snapshot),
               "start": str(start) if start is not None else None,
               "end": str(end) if end is not None else None}
    key = _digest([context, root_key])
    # Row indices are source positions, not joins on possibly duplicated keys.
    mask = pl.lit(True)
    if start is not None:
        mask &= pl.col("trade_date") >= date.fromisoformat(str(start))
    if end is not None:
        mask &= pl.col("trade_date") <= date.fromisoformat(str(end))
    indices = (pl.int_range(0, df.height, dtype=pl.UInt32, eager=True)
               if start is None and end is None
               else df.select(pl.arg_where(mask).alias("row")).to_series())
    with _CACHE.flight(key):
        result = _CACHE.get(key, len(indices), disk=True)
        if result is None:
            started = time.perf_counter()
            used_fields = sorted({n.id for n in ast.walk(root) if isinstance(n, ast.Name) and n.id in fields})
            source = df.select(list(dict.fromkeys(["ts_code", "trade_date", *used_fields])))[indices]
            overrides, attached = {}, []
            # Bound intermediate fanout for large DSL trees. Window nodes only.
            selected = sorted(nodes, key=len)[:4]
            for i, node_key in enumerate(selected):
                if node_key == root_key:
                    continue
                value = _CACHE.get(_digest([context, node_key]), len(indices))
                if value is not None:
                    name = f"__shared_{i}"
                    attached.append(value.rename(name))
                    overrides[node_key] = name
            if attached:
                source = source.with_columns(attached)
            pipeline = parse(expression, fields, _overrides=overrides) if overrides else reference
            lf = source.lazy()
            for name, expr in pipeline.stages:
                lf = lf.with_columns(expr.alias(name))
            outputs = [(k, pipeline.node_outputs[k]) for k in selected
                       if k != root_key and k in pipeline.node_outputs and k not in overrides]
            computed = lf.select(pipeline.final.alias("factor"), *[
                expr.alias(f"shared_{i}") for i, (_, expr) in enumerate(outputs)]).collect()
            result = computed["factor"]
            # Scalar expressions must broadcast to every source row.
            if len(result) == 1 and len(indices) != 1:
                result = pl.Series("factor", [result[0]] * len(indices), dtype=result.dtype)
            for i, (node_key, _) in enumerate(outputs):
                _CACHE.put(_digest([context, node_key]), computed[f"shared_{i}"])
            _CACHE.put(key, result, disk=True)
            with _CACHE.lock:
                _CACHE.counters["factor_materializations"] += 1
                _CACHE.counters["subexpression_reuses"] += len(overrides)
                _CACHE.counters["compute_ms"] += round((time.perf_counter() - started) * 1000, 3)
        if len(result) != len(indices):
            raise ValueError("Cached factor row alignment mismatch")
        if len(indices) == df.height:
            return result.rename("factor")
        return pl.repeat(None, df.height, dtype=result.dtype, eager=True).scatter(indices, result).rename("factor")
