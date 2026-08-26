"""Single-pass, cached cross-sectional screener."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
import time
from collections import OrderedDict
from datetime import date

import polars as pl

from .dsl.engine import parse, required_history


class _ScreenerCache:
    def __init__(self, maxsize: int = 96) -> None:
        self.maxsize = maxsize
        self._values: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()
        self._started_at = time.time()
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0
        self._clears = 0
        self._cleared_entries = 0
        self._last_hit_at: float | None = None
        self._last_put_at: float | None = None
        self._last_clear_at: float | None = None

    def get(self, key: str) -> dict | None:
        with self._lock:
            value = self._values.get(key)
            if value is None:
                self._misses += 1
                return None
            self._hits += 1
            self._last_hit_at = time.time()
            self._values.move_to_end(key)
            return copy.deepcopy(value)

    def put(self, key: str, value: dict) -> None:
        with self._lock:
            self._puts += 1
            self._last_put_at = time.time()
            self._values[key] = copy.deepcopy(value)
            self._values.move_to_end(key)
            while len(self._values) > self.maxsize:
                self._values.popitem(last=False)
                self._evictions += 1

    def clear(self) -> int:
        """Invalidate every cached cross-section after a panel generation swap."""
        with self._lock:
            removed = len(self._values)
            self._values.clear()
            self._clears += 1
            self._cleared_entries += removed
            self._last_clear_at = time.time()
            return removed

    def stats(self) -> dict:
        with self._lock:
            attempts = self._hits + self._misses
            return {
                "entries": len(self._values),
                "capacity": self.maxsize,
                "hits": self._hits,
                "misses": self._misses,
                "puts": self._puts,
                "evictions": self._evictions,
                "clears": self._clears,
                "cleared_entries": self._cleared_entries,
                "hit_rate": round(self._hits / max(1, attempts), 6),
                "started_at_epoch": self._started_at,
                "last_hit_at_epoch": self._last_hit_at,
                "last_put_at_epoch": self._last_put_at,
                "last_clear_at_epoch": self._last_clear_at,
            }


SCREEN_CACHE = _ScreenerCache()


def _cache_key(
    *,
    panel_identity: str,
    target_date: date,
    factors: list[dict],
    universe_n: int,
    top_n: int,
    direction: str,
) -> str:
    payload = {
        "panel": panel_identity,
        "date": str(target_date),
        "factors": [
            {
                "name": row.get("name"),
                "expression": row["expression"],
                "weight": float(row.get("weight", 1.0)),
                "direction": int(row.get("direction", 1)),
            }
            for row in factors
        ],
        "universe_n": universe_n,
        "top_n": top_n,
        "direction": direction,
        "version": "single_pass_v3_tail_rank",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def screen_cross_section(
    *,
    df: pl.DataFrame,
    trading_dates: list[date],
    panel_identity: str,
    target_date: date,
    factors: list[dict],
    fields: list[str],
    universe_n: int,
    top_n: int,
    direction: str,
) -> dict:
    started = time.perf_counter()
    key = _cache_key(
        panel_identity=panel_identity,
        target_date=target_date,
        factors=factors,
        universe_n=universe_n,
        top_n=top_n,
        direction=direction,
    )
    cached = SCREEN_CACHE.get(key)
    if cached is not None:
        cached["performance"] = {
            **cached.get("performance", {}),
            "cache_hit": True,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        }
        return cached

    if target_date not in trading_dates:
        raise ValueError("目标日期不在交易日历中")
    target_index = trading_dates.index(target_date)
    lookback = max(required_history(row["expression"]) for row in factors)
    start_index = max(0, target_index - lookback - 2)
    history_start = trading_dates[start_index]

    lf = df.lazy().filter(
        pl.col("trade_date").is_between(history_start, target_date)
    )
    value_columns = []
    for index, factor in enumerate(factors):
        alias = f"_screen_factor_{index}"
        lf = parse(factor["expression"], fields).apply(lf, alias=alias)
        value_columns.append(alias)
    lf = lf.filter(
        (pl.col("trade_date") == target_date)
        & (pl.col("univ_rank") <= universe_n)
    )
    for index, factor in enumerate(factors):
        value = pl.col(value_columns[index]) * int(factor.get("direction", 1))
        lf = lf.with_columns(
            (
                value.rank(method="average")
                / value.count()
                * 100.0
            ).alias(f"_screen_rank_{index}")
        )
    lf = lf.filter(pl.all_horizontal([pl.col(column).is_finite() for column in value_columns]))

    total_weight = sum(float(row.get("weight", 1.0)) for row in factors)
    score = pl.lit(0.0)
    for index, factor in enumerate(factors):
        score += pl.col(f"_screen_rank_{index}") * (
            float(factor.get("weight", 1.0)) / total_weight
        )
    select_columns = [
        "ts_code", "name", "univ_rank", "raw_close", "amount",
        *value_columns,
        *[f"_screen_rank_{index}" for index in range(len(factors))],
    ]
    ranked = (
        lf.with_columns(score.alias("score"))
        .select(*select_columns, "score")
        .collect()
        .with_columns(
            pl.col("score")
            .rank(method="ordinal", descending=True)
            .cast(pl.Int64)
            .alias("_head_rank"),
            pl.col("score")
            .rank(method="ordinal")
            .cast(pl.Int64)
            .alias("_tail_rank"),
        )
    )
    eligible_count = ranked.height
    if direction == "bottom":
        chosen = (
            ranked.sort("_tail_rank")
            .head(top_n)
            .with_columns(
                pl.lit("bottom").alias("_side"),
                pl.col("_tail_rank").alias("_side_rank"),
            )
        )
    elif direction == "both":
        head_count = max(1, (top_n + 1) // 2)
        tail_count = max(0, top_n - head_count)
        high = (
            ranked.sort("_head_rank")
            .head(head_count)
            .with_columns(
                pl.lit("top").alias("_side"),
                pl.col("_head_rank").alias("_side_rank"),
            )
        )
        high_symbols = high["ts_code"].to_list()
        low = (
            ranked.filter(~pl.col("ts_code").is_in(high_symbols))
            .sort("_tail_rank")
            .head(tail_count)
            .with_columns(
                pl.lit("bottom").alias("_side"),
                pl.col("_tail_rank").alias("_side_rank"),
            )
        )
        chosen = pl.concat([high, low])
    else:
        chosen = (
            ranked.sort("_head_rank")
            .head(top_n)
            .with_columns(
                pl.lit("top").alias("_side"),
                pl.col("_head_rank").alias("_side_rank"),
            )
        )

    stocks = []
    for rank_no, row in enumerate(chosen.iter_rows(named=True), start=1):
        components = []
        for index, factor in enumerate(factors):
            weight = float(factor.get("weight", 1.0)) / total_weight
            rank_score = float(row[f"_screen_rank_{index}"])
            components.append({
                "name": factor.get("name") or f"因子{index + 1}",
                "expression": factor["expression"],
                "direction": int(factor.get("direction", 1)),
                "weight": round(weight, 6),
                "value": round(float(row[value_columns[index]]), 8),
                "rank_score": round(rank_score, 4),
                "contribution": round(rank_score * weight, 4),
            })
        stocks.append({
            "rank": rank_no,
            "side": row["_side"],
            "side_rank": int(row["_side_rank"]),
            "head_rank": int(row["_head_rank"]),
            "tail_rank": int(row["_tail_rank"]),
            "ts_code": row["ts_code"],
            "name": row.get("name") or "",
            "score": round(float(row["score"]), 4),
            "universe_rank": int(row["univ_rank"]),
            "raw_close": round(float(row["raw_close"]), 4) if row.get("raw_close") is not None else None,
            "amount": round(float(row["amount"]), 2) if row.get("amount") is not None else None,
            "components": components,
        })
    elapsed = (time.perf_counter() - started) * 1000
    result = {
        "stocks": stocks,
        "eligible_count": eligible_count,
        "history_start": str(history_start),
        "required_history": lookback,
        "ranking_semantics": {
            "head_rank": "1 表示方向调整后综合分最高",
            "tail_rank": "1 表示方向调整后综合分最低",
            "side_rank": "所选头部或尾部榜内名次",
            "tail_usage": "A股纯多头任务中尾部仅表示回避/负向观察，不表示可做空",
        },
        "performance": {
            "engine": "polars_single_lazy_plan_v3_tail_rank",
            "cache_hit": False,
            "elapsed_ms": round(elapsed, 2),
            "factor_count": len(factors),
            "rows_scored": eligible_count,
        },
    }
    SCREEN_CACHE.put(key, result)
    return result
