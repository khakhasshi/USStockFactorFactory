"""Large-sample US overnight gap event study.

The study is deliberately separate from the factor engine.  It reads the
frozen US research panel, detects the return from day ``n`` open to day
``n+1`` open, and measures the subsequent open-to-open return after 5, 10 and
20 observed *market* sessions.  Two price modes are reported:

* ``raw``: the unadjusted opening price, closest to an executable screen;
* ``adjusted``: the forward-adjusted opening price, useful for corporate
  action robustness checks.

The primary answer is based on strict market-calendar spacing.  If a symbol
has no observation on an intervening market session, that event is retained
in the raw event detail but excluded from the corresponding horizon summary.
This prevents a suspension or a data gap from being silently called a
five-day return.

The output is a frozen, self-contained HTML report plus Parquet/CSV detail
files.  It is read-only with respect to the source panel and database.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.config import US_PANEL_GLOB  # noqa: E402

DEFAULT_PANEL_GLOB = US_PANEL_GLOB
DEFAULT_THRESHOLDS = (0.03, 0.05, 0.08, 0.10)
HORIZONS = (5, 10, 20)
COOLDOWN_SESSIONS = 20
REPORT_PROTOCOL = "US_OVERNIGHT_GAP_EVENT_STUDY_V1"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, Path)):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def parse_thresholds(raw: str | Sequence[float]) -> tuple[float, ...]:
    if isinstance(raw, str):
        values = tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    else:
        values = tuple(float(item) for item in raw)
    if not values or any(value <= 0 or value >= 1 for value in values):
        raise ValueError("阈值必须是 (0, 1) 之间的百分比小数，例如 0.05")
    return tuple(sorted(set(values)))


def _compact_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    universe = metadata.get("universe_completion") or {}
    return {
        "provider": metadata.get("provider"),
        "provider_id": metadata.get("provider_id"),
        "market": metadata.get("market"),
        "first_trade_date": metadata.get("first_trade_date"),
        "last_trade_date": metadata.get("last_trade_date"),
        "generated_at_utc": metadata.get("generated_at_utc"),
        "rows": metadata.get("rows"),
        "symbols": metadata.get("symbols"),
        "price_adjustment": metadata.get("price_adjustment"),
        "execution_price_adjustment": metadata.get("execution_price_adjustment"),
        "pit_quality": "non_pit_current_constituents",
        "universe_mode": metadata.get("universe_mode"),
        "universe_survivorship_safe": metadata.get("universe_survivorship_safe"),
        "available_symbols": universe.get("available_symbols"),
        "requested_symbols": universe.get("requested_symbols"),
        "coverage": universe.get("coverage"),
        "coverage_status": universe.get("status"),
        "production_blockers": metadata.get("production_blockers") or [],
    }


def _panel_root(panel_glob: str) -> Path:
    path = Path(panel_glob)
    # .../trade_year=*/data_0.parquet -> .../daily_panel
    return path.parent.parent


def _quality_filter() -> pl.Expr:
    return (
        pl.col("is_valid_ohlc")
        & pl.col("is_tradable_observation")
        & pl.col("is_adjusted_price_continuous")
        & pl.col("is_adjustment_pair_consistent")
        & (pl.col("raw_open") > 0)
        & (pl.col("open") > 0)
    )


def build_event_candidates(panel_glob: str) -> pl.DataFrame:
    """Build one row per symbol/day with next-open and forward-open returns."""
    source = pl.scan_parquet(panel_glob, hive_partitioning=True)
    needed = [
        "trade_date",
        "ts_code",
        "name",
        "open",
        "raw_open",
        "raw_close",
        "adjustment_factor",
        "is_valid_ohlc",
        "is_tradable_observation",
        "is_adjusted_price_continuous",
        "is_adjustment_pair_consistent",
    ]
    calendar = (
        source.select("trade_date")
        .unique()
        .sort("trade_date")
        .with_row_index("calendar_idx")
    )
    next_open_exprs = [
        pl.col("raw_open").shift(-1).over("ts_code").alias("event_raw_open"),
        pl.col("open").shift(-1).over("ts_code").alias("event_adj_open"),
        pl.col("trade_date")
        .shift(-1)
        .over("ts_code")
        .alias("event_trade_date"),
        pl.col("calendar_idx")
        .shift(-1)
        .over("ts_code")
        .alias("event_calendar_idx"),
    ]
    forward_open_exprs: list[pl.Expr] = []
    forward_return_exprs: list[pl.Expr] = []
    selected_forward_columns: list[str] = []
    for h in HORIZONS:
        forward_open_exprs.extend(
            [
                pl.col("raw_open")
                .shift(-(h + 1))
                .over("ts_code")
                .alias(f"raw_open_h{h}"),
                pl.col("open")
                .shift(-(h + 1))
                .over("ts_code")
                .alias(f"adj_open_h{h}"),
                pl.col("raw_close")
                .shift(-(h + 1))
                .over("ts_code")
                .alias(f"raw_close_h{h}"),
                pl.col("trade_date")
                .shift(-(h + 1))
                .over("ts_code")
                .alias(f"future_trade_date_h{h}"),
                pl.col("calendar_idx")
                .shift(-(h + 1))
                .over("ts_code")
                .alias(f"future_calendar_idx_h{h}"),
            ]
        )
        forward_return_exprs.extend(
            [
                (
                    pl.col(f"raw_open_h{h}") / pl.col("event_raw_open") - 1
                ).alias(f"raw_fwd_open_h{h}"),
                (
                    pl.col(f"adj_open_h{h}") / pl.col("event_adj_open") - 1
                ).alias(f"adjusted_fwd_open_h{h}"),
                (
                    pl.col(f"raw_close_h{h}") / pl.col("event_raw_open") - 1
                ).alias(f"raw_fwd_close_h{h}"),
                (
                    pl.col(f"future_calendar_idx_h{h}")
                    - pl.col("event_calendar_idx")
                ).alias(f"future_event_spacing_h{h}"),
            ]
        )
        selected_forward_columns.extend(
            [
                f"raw_fwd_open_h{h}",
                f"adjusted_fwd_open_h{h}",
                f"raw_fwd_close_h{h}",
                f"future_event_spacing_h{h}",
                f"future_trade_date_h{h}",
            ]
        )
    frame = (
        source.select(needed)
        .join(calendar, on="trade_date", how="left")
        .sort(["ts_code", "trade_date"])
        .with_columns(next_open_exprs + forward_open_exprs)
        .with_columns(
            [
                (pl.col("event_raw_open") / pl.col("raw_open") - 1).alias("raw_gap"),
                (pl.col("event_adj_open") / pl.col("open") - 1).alias("adj_gap"),
            ]
            + forward_return_exprs
        )
        .filter(_quality_filter())
        .select(
            [
                "trade_date",
                "event_trade_date",
                "ts_code",
                "name",
                "calendar_idx",
                "event_calendar_idx",
                "raw_open",
                "event_raw_open",
                "open",
                "event_adj_open",
                "adjustment_factor",
                "raw_gap",
                "adj_gap",
            ]
            + selected_forward_columns
        )
        .collect(engine="streaming")
    )
    return frame


def _float_values(frame: pl.DataFrame, column: str) -> np.ndarray:
    return frame.get_column(column).drop_nulls().to_numpy()


def _cluster_bootstrap_ci(
    frame: pl.DataFrame,
    value_column: str,
    *,
    reps: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if reps <= 0 or frame.is_empty():
        return None, None
    grouped = (
        frame.select(["ts_code", value_column])
        .drop_nulls()
        .group_by("ts_code")
        .agg(
            pl.col(value_column).sum().alias("cluster_sum"),
            pl.len().alias("cluster_count"),
        )
    )
    cluster_sums = grouped.get_column("cluster_sum").to_numpy()
    cluster_counts = grouped.get_column("cluster_count").to_numpy()
    if len(cluster_sums) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(cluster_sums), size=(reps, len(cluster_sums)))
    # Resample whole stocks, retaining every event belonging to a sampled
    # stock.  This preserves the event-weighted estimand used by mean_return
    # while allowing arbitrary within-stock dependence.
    boot_means = (
        cluster_sums[draws].sum(axis=1) / cluster_counts[draws].sum(axis=1)
    )
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975))


def _independent_mask(frame: pl.DataFrame, cooldown_sessions: int) -> np.ndarray:
    """Greedy per-symbol de-clustering using the last selected event."""
    symbols = frame.get_column("ts_code").to_list()
    indices = frame.get_column("event_calendar_idx").to_numpy()
    order = np.lexsort((indices, np.asarray(symbols, dtype=object)))
    selected = np.zeros(len(frame), dtype=bool)
    last_symbol: Any = None
    last_idx = -10**9
    for position in order:
        symbol = symbols[int(position)]
        idx = int(indices[int(position)]) if indices[int(position)] is not None else -10**9
        if symbol != last_symbol or idx - last_idx > cooldown_sessions:
            selected[int(position)] = True
            last_symbol = symbol
            last_idx = idx
    return selected


def summarize_group(
    frame: pl.DataFrame,
    *,
    gap_column: str,
    forward_column: str,
    threshold: float,
    sign: str,
    horizon: int,
    sample_kind: str,
    bootstrap_reps: int,
    seed: int,
    cooldown_sessions: int = COOLDOWN_SESSIONS,
) -> dict[str, Any]:
    condition = pl.col(gap_column) >= threshold if sign == "positive" else pl.col(gap_column) <= -threshold
    subset = frame.filter(condition & pl.col(forward_column).is_not_null())
    subset = subset.filter(
        pl.col(f"future_event_spacing_h{horizon}") == horizon
    )
    if subset.is_empty():
        return {
            "price_mode": "raw" if gap_column.startswith("raw") else "adjusted",
            "threshold": threshold,
            "sign": sign,
            "horizon": horizon,
            "sample": sample_kind,
            "events": 0,
            "stocks": 0,
        }
    if sample_kind == "cooldown20d":
        mask = _independent_mask(subset, cooldown_sessions)
        subset = subset.filter(pl.Series("independent", mask))
    values = _float_values(subset, forward_column).astype(float)
    gaps = _float_values(subset, gap_column).astype(float)
    if values.size == 0:
        return {
            "price_mode": "raw" if gap_column.startswith("raw") else "adjusted",
            "threshold": threshold,
            "sign": sign,
            "horizon": horizon,
            "sample": sample_kind,
            "events": 0,
            "stocks": 0,
        }
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    se = std / math.sqrt(len(values)) if len(values) > 1 else None
    ci_low, ci_high = _cluster_bootstrap_ci(
        subset,
        forward_column,
        reps=bootstrap_reps,
        seed=seed,
    )
    return {
        "price_mode": "raw" if gap_column.startswith("raw") else "adjusted",
        "threshold": threshold,
        "threshold_pct": threshold * 100,
        "sign": sign,
        "horizon": horizon,
        "sample": sample_kind,
        "events": int(len(values)),
        "stocks": int(subset.get_column("ts_code").n_unique()),
        "gap_mean": float(gaps.mean()),
        "gap_median": float(np.median(gaps)),
        "mean_return": mean,
        "median_return": float(np.median(values)),
        "win_rate": float((values > 0).mean()),
        "loss_rate": float((values < 0).mean()),
        "positive_minus_negative": float((values > 0).sum() - (values < 0).sum()) / len(values),
        "std_return": std,
        "p05": float(np.quantile(values, 0.05)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p95": float(np.quantile(values, 0.95)),
        "naive_t_stat": (mean / se) if se else None,
        "cluster_bootstrap_ci_low": ci_low,
        "cluster_bootstrap_ci_high": ci_high,
        "forward_column": forward_column,
    }


def build_statistics(
    events: pl.DataFrame,
    thresholds: Sequence[float],
    *,
    bootstrap_reps: int,
    cooldown_sessions: int = COOLDOWN_SESSIONS,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    seed = 20260820
    for mode in ("raw", "adjusted"):
        gap_column = "raw_gap" if mode == "raw" else "adj_gap"
        for threshold_index, threshold in enumerate(thresholds):
            for sign_index, sign in enumerate(("positive", "negative")):
                for horizon in HORIZONS:
                    forward_column = f"{mode}_fwd_open_h{horizon}"
                    for sample_index, sample_kind in enumerate(("all", "cooldown20d")):
                        rows.append(
                            summarize_group(
                                events,
                                gap_column=gap_column,
                                forward_column=forward_column,
                                threshold=threshold,
                                sign=sign,
                                horizon=horizon,
                                sample_kind=sample_kind,
                                bootstrap_reps=bootstrap_reps,
                                seed=seed
                                + threshold_index * 1000
                                + sign_index * 100
                                + horizon
                                + sample_index,
                                cooldown_sessions=cooldown_sessions,
                            )
                        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["price_mode", "threshold", "sign", "horizon", "sample"]
    )


def build_year_statistics(
    events: pl.DataFrame,
    *,
    threshold: float = 0.05,
    sign: str = "positive",
    mode: str = "raw",
) -> pl.DataFrame:
    gap_column = "raw_gap" if mode == "raw" else "adj_gap"
    rows: list[dict[str, Any]] = []
    base = events.filter(
        (pl.col(gap_column) >= threshold if sign == "positive" else pl.col(gap_column) <= -threshold)
    )
    for row in base.select(pl.col("event_trade_date").dt.year().alias("year")).unique().sort("year").iter_rows():
        year = int(row[0])
        subset = base.filter(pl.col("event_trade_date").dt.year() == year)
        for horizon in HORIZONS:
            value_column = f"{mode}_fwd_open_h{horizon}"
            subset_h = subset.filter(
                pl.col(value_column).is_not_null()
                & (pl.col(f"future_event_spacing_h{horizon}") == horizon)
            )
            values = _float_values(subset_h, value_column).astype(float)
            if not len(values):
                continue
            rows.append(
                {
                    "year": year,
                    "horizon": horizon,
                    "events": int(len(values)),
                    "stocks": int(subset_h.get_column("ts_code").n_unique()),
                    "mean_return": float(values.mean()),
                    "median_return": float(np.median(values)),
                    "win_rate": float((values > 0).mean()),
                    "p25": float(np.quantile(values, 0.25)),
                    "p75": float(np.quantile(values, 0.75)),
                }
            )
    return pl.DataFrame(rows, infer_schema_length=None).sort(["year", "horizon"])


def build_gap_buckets(
    events: pl.DataFrame,
    *,
    mode: str = "raw",
    sign: str = "positive",
) -> pl.DataFrame:
    gap_column = "raw_gap" if mode == "raw" else "adj_gap"
    forward_column = f"{mode}_fwd_open_h10"
    base = events.filter(
        pl.col(gap_column).is_not_null()
        & pl.col(forward_column).is_not_null()
        & (pl.col("future_event_spacing_h10") == 10)
    )
    if sign == "positive":
        base = base.filter(pl.col(gap_column) >= 0.03)
        bucket_expr = (
            pl.when(pl.col(gap_column) < 0.05)
            .then(pl.lit("3%-5%"))
            .when(pl.col(gap_column) < 0.08)
            .then(pl.lit("5%-8%"))
            .when(pl.col(gap_column) < 0.12)
            .then(pl.lit("8%-12%"))
            .otherwise(pl.lit("12%+"))
        )
    else:
        base = base.filter(pl.col(gap_column) <= -0.03)
        bucket_expr = (
            pl.when(pl.col(gap_column) > -0.05)
            .then(pl.lit("-3%--5%"))
            .when(pl.col(gap_column) > -0.08)
            .then(pl.lit("-5%--8%"))
            .when(pl.col(gap_column) > -0.12)
            .then(pl.lit("-8%--12%"))
            .otherwise(pl.lit("-12%-"))
        )
    return (
        base.with_columns(bucket_expr.alias("gap_bucket"))
        .group_by("gap_bucket")
        .agg(
            pl.len().alias("events"),
            pl.col("ts_code").n_unique().alias("stocks"),
            pl.col(gap_column).mean().alias("mean_gap"),
            pl.col(forward_column).mean().alias("mean_return_h10"),
            pl.col(forward_column).median().alias("median_return_h10"),
            (pl.col(forward_column) > 0).mean().alias("win_rate_h10"),
        )
        .sort("gap_bucket")
    )


def _fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "—"


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _headline_rows(stats: pl.DataFrame, *, mode: str, threshold: float, sign: str) -> list[dict[str, Any]]:
    subset = stats.filter(
        (pl.col("price_mode") == mode)
        & (pl.col("threshold") == threshold)
        & (pl.col("sign") == sign)
        & (pl.col("sample") == "all")
    ).sort("horizon")
    return subset.to_dicts()


def _conclusion(stats: pl.DataFrame, *, mode: str, threshold: float, sign: str) -> str:
    rows = _headline_rows(stats, mode=mode, threshold=threshold, sign=sign)
    if not rows:
        return "没有满足严格日历间隔和完整持有期的事件，不能下结论。"
    positives = [row for row in rows if row.get("mean_return") is not None and row.get("win_rate") is not None and row["mean_return"] > 0 and row["win_rate"] > 0.5]
    negatives = [row for row in rows if row.get("mean_return") is not None and row.get("win_rate") is not None and row["mean_return"] < 0 and row["win_rate"] < 0.5]
    if len(positives) >= 2:
        return "大幅跳空后在多数持有期呈现后续偏涨，但这只是条件统计，仍需看年份稳定性、去簇样本和执行成本。"
    if len(negatives) >= 2:
        return "大幅跳空后在多数持有期呈现后续偏跌，说明该事件更像短期过度反应/回落信号，但仍需通过冻结样本验证。"
    return "5/10/20 日方向不一致，不能把隔夜跳空简单概括为持续上涨或持续下跌。"


def render_report(
    output_dir: Path,
    *,
    metadata: dict[str, Any],
    panel_glob: str,
    events: pl.DataFrame,
    stats: pl.DataFrame,
    year_stats: pl.DataFrame,
    gap_buckets: pl.DataFrame,
    started_at: str,
    elapsed_seconds: float,
    bootstrap_reps: int,
    cooldown_sessions: int,
    minimum_threshold: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_details = events.filter(
        (pl.col("raw_gap").abs() >= minimum_threshold)
        | (pl.col("adj_gap").abs() >= minimum_threshold)
    )
    event_details.write_parquet(output_dir / "event_details.parquet", compression="zstd")
    stats.write_csv(output_dir / "summary_statistics.csv")
    year_stats.write_csv(output_dir / "raw_positive_5pct_by_year.csv")
    gap_buckets.write_csv(output_dir / "raw_positive_gap_buckets.csv")
    _write_json(output_dir / "panel_metadata_compact.json", metadata)

    headline = _headline_rows(stats, mode="raw", threshold=0.05, sign="positive")
    robust = stats.filter(
        (pl.col("price_mode") == "raw")
        & (pl.col("threshold") == 0.05)
        & (pl.col("sign") == "positive")
        & (pl.col("sample") == "cooldown20d")
    ).sort("horizon").to_dicts()
    positive_events = events.filter(pl.col("raw_gap") >= 0.05)
    negative_events = events.filter(pl.col("raw_gap") <= -0.05)
    quality = {
        "panel_observation_rows": events.height,
        "event_detail_rows": event_details.height,
        "unique_symbols": events.get_column("ts_code").n_unique() if events.height else 0,
        "positive_raw_5pct_candidates": positive_events.height,
        "negative_raw_5pct_candidates": negative_events.height,
        "positive_raw_5pct_strict_h20": int(
            positive_events.filter(pl.col("future_event_spacing_h20") == 20).height
        ),
        "nonconsecutive_next_observation": int(
            events.filter(pl.col("event_calendar_idx") - pl.col("calendar_idx") != 1).height
        ),
    }
    manifest = {
        "protocol": REPORT_PROTOCOL,
        "started_at_utc": started_at,
        "rendered_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "polars_version": pl.__version__,
        "polars_thread_pool_size": pl.thread_pool_size(),
        "panel_glob": panel_glob,
        "price_modes": {"raw": "unadjusted open", "adjusted": "forward-adjusted open"},
        "horizons": list(HORIZONS),
        "thresholds": sorted(set(float(row["threshold"]) for row in stats.to_dicts())),
        "bootstrap_reps": bootstrap_reps,
        "cooldown_sessions": cooldown_sessions,
        "event_definition": "open[n+1] / open[n] - 1, with n+1 the next observed symbol session",
        "forward_definition": "open[n+1+h] / open[n+1] - 1, with strict global market-calendar spacing h",
        "quality": quality,
        "panel": metadata,
    }
    _write_json(output_dir / "manifest.json", manifest)

    def stat_table(rows: list[dict[str, Any]], include_ci: bool = True) -> str:
        values = []
        for row in rows:
            ci = (
                f"{_fmt_pct(row.get('cluster_bootstrap_ci_low'))} ~ {_fmt_pct(row.get('cluster_bootstrap_ci_high'))}"
                if include_ci
                else "—"
            )
            values.append(
                [
                    f"H{row.get('horizon')}",
                    f"{row.get('events', 0):,}",
                    f"{row.get('stocks', 0):,}",
                    _fmt_pct(row.get("mean_return")),
                    _fmt_pct(row.get("median_return")),
                    _fmt_pct(row.get("win_rate")),
                    ci,
                    _fmt_num(row.get("naive_t_stat")),
                ]
            )
        return _table(
            ["持有期", "事件数", "股票数", "平均后续收益", "中位数", "上涨概率", "按股票聚类 bootstrap 95% CI", "朴素 t 值"],
            values,
        )

    year_rows = []
    for row in year_stats.to_dicts():
        year_rows.append(
            [
                row["year"],
                f"H{row['horizon']}",
                f"{row['events']:,}",
                f"{row['stocks']:,}",
                _fmt_pct(row["mean_return"]),
                _fmt_pct(row["median_return"]),
                _fmt_pct(row["win_rate"]),
                _fmt_pct(row["p25"]),
                _fmt_pct(row["p75"]),
            ]
        )
    bucket_rows = []
    for row in gap_buckets.to_dicts():
        bucket_rows.append(
            [
                row["gap_bucket"],
                f"{row['events']:,}",
                f"{row['stocks']:,}",
                _fmt_pct(row["mean_gap"]),
                _fmt_pct(row["mean_return_h10"]),
                _fmt_pct(row["median_return_h10"]),
                _fmt_pct(row["win_rate_h10"]),
            ]
        )

    report_html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>美股隔夜跳空有效性事件研究</title>
<style>
:root {{ color-scheme: dark; --bg:#0b1017; --panel:#141b24; --panel2:#0f151d; --line:#2b3745; --text:#e8eef6; --muted:#99a8b8; --blue:#55a6ff; --green:#47d17a; --yellow:#f3b63f; --red:#ff6978; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
main {{ max-width:1500px; margin:0 auto; padding:30px 28px 70px; }} h1 {{ margin:0 0 6px; font-size:30px; }} h2 {{ margin:28px 0 12px; font-size:19px; }} h3 {{ margin:20px 0 8px; font-size:16px; }}
.eyebrow {{ color:var(--blue); font:600 12px/1.2 ui-monospace,SFMono-Regular,Menlo,monospace; letter-spacing:.14em; text-transform:uppercase; }} .sub {{ color:var(--muted); }}
.grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:22px 0; }} .card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:16px; }} .card strong {{ display:block; font-size:25px; margin-top:3px; }}
.callout {{ background:#102131; border:1px solid #23577e; border-left:4px solid var(--blue); padding:16px 18px; border-radius:8px; }} .warn {{ border-left-color:var(--yellow); background:#211b10; border-color:#5e4d27; }}
.section {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:18px; margin-top:16px; overflow:auto; }} table {{ width:100%; border-collapse:collapse; min-width:800px; }} th,td {{ border-bottom:1px solid var(--line); padding:9px 10px; text-align:right; white-space:nowrap; }} th:first-child,td:first-child {{ text-align:left; }} th {{ color:var(--muted); font-weight:600; }} tr:hover td {{ background:var(--panel2); }}
code,pre {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }} code {{ color:#b6d7ff; }} a {{ color:var(--blue); }} .small {{ color:var(--muted); font-size:12px; }} .two {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
@media(max-width:900px) {{ .grid,.two {{ grid-template-columns:1fr 1fr; }} }} @media(max-width:620px) {{ main {{ padding:20px 12px; }} .grid,.two {{ grid-template-columns:1fr; }} h1 {{ font-size:24px; }} }}
</style></head><body><main>
<div class="eyebrow">US EQUITY / EVENT STUDY / FROZEN REPORT</div>
<h1>隔夜跳空的有效性：美股大样本事件研究</h1>
<div class="sub">研究协议 {REPORT_PROTOCOL} · 生成于 {html.escape(datetime.now(timezone.utc).isoformat())}</div>
<div class="callout" style="margin-top:20px"><strong>先说结论（原始开盘、正向跳空 ≥5%）</strong><br>{html.escape(_conclusion(stats, mode="raw", threshold=0.05, sign="positive"))}<br><span class="small">下面的 H5/H10/H20 都从跳空发生日 n+1 的开盘开始计算，不把跳空本身重复计入后续收益。</span></div>
<div class="grid">
<div class="card"><span class="sub">面板行数</span><strong>{int(metadata.get('rows') or 0):,}</strong><span class="small">{html.escape(str(metadata.get('first_trade_date')))} 至 {html.escape(str(metadata.get('last_trade_date')))}</span></div>
<div class="card"><span class="sub">覆盖股票</span><strong>{int(metadata.get('symbols') or 0):,}</strong><span class="small">yfinance 非 PIT 研究投影</span></div>
<div class="card"><span class="sub">原始 ≥5% 跳空候选</span><strong>{quality['positive_raw_5pct_candidates']:,}</strong><span class="small">严格 H20 可用 {quality['positive_raw_5pct_strict_h20']:,}</span></div>
<div class="card"><span class="sub">原始 ≤−5% 对照候选</span><strong>{quality['negative_raw_5pct_candidates']:,}</strong><span class="small">用于判断方向是否对称</span></div>
</div>
<div class="section"><h2>研究口径</h2><ul>
<li>事件：同一股票的 <code>open[n+1] / open[n] - 1</code>，n+1 是该股票下一条观察；事件统计同时要求它是全市场下一个交易日。</li>
<li>后续收益：<code>open[n+1+h] / open[n+1] - 1</code>，h = 5、10、20；统计只纳入严格相隔 h 个全市场交易日的观测。</li>
<li>主结果使用 <code>raw_open</code>，更接近实际开盘价；<code>open</code> 前复权结果用于公司行动鲁棒性对照。</li>
<li>每个表都给原始事件与 20 个交易日冷却去簇样本；bootstrap 按股票聚类，避免同一股票的连续事件把置信度夸大。</li>
</ul></div>
<div class="two">
<div class="section"><h2>正向跳空 ≥5% · 原始开盘</h2>{stat_table(headline)}</div>
<div class="section"><h2>正向跳空 ≥5% · 去簇稳健性</h2>{stat_table(robust)}</div>
</div>
<div class="section"><h2>阈值与方向完整统计</h2><p class="small">可下载 <a href="summary_statistics.csv">summary_statistics.csv</a>。正向跳空、反向跳空以及 3/5/8/10% 阈值均已计算；“上涨概率”是后续收益大于 0 的事件比例。</p>
{_table(["口径","阈值","方向","样本","H","事件数","股票数","平均收益","中位数","上涨概率","CI 下限","CI 上限"], [[row.get("price_mode"), _fmt_pct(row.get("threshold")), row.get("sign"), row.get("sample"), f"H{row.get('horizon')}", f"{row.get('events',0):,}", f"{row.get('stocks',0):,}", _fmt_pct(row.get('mean_return')), _fmt_pct(row.get('median_return')), _fmt_pct(row.get('win_rate')), _fmt_pct(row.get('cluster_bootstrap_ci_low')), _fmt_pct(row.get('cluster_bootstrap_ci_high'))] for row in stats.to_dicts() if row.get('events',0) > 0])}
</div>
<div class="section"><h2>原始正向 ≥5% 按年份</h2><p class="small">用来检查结论是不是被少数年份驱动；完整表见 <a href="raw_positive_5pct_by_year.csv">raw_positive_5pct_by_year.csv</a>。</p>{_table(["年份","持有期","事件数","股票数","平均收益","中位数","上涨概率","P25","P75"], year_rows)}</div>
<div class="section"><h2>跳空幅度分桶（原始正向，H10）</h2><p class="small">完整表见 <a href="raw_positive_gap_buckets.csv">raw_positive_gap_buckets.csv</a>。</p>{_table(["跳空区间","事件数","股票数","平均跳空","H10 平均收益","H10 中位数","H10 上涨概率"], bucket_rows)}</div>
<div class="section"><h2>数据边界与工程细节</h2><ul>
<li>源面板：<code>{html.escape(panel_glob)}</code>；提供方：{html.escape(str(metadata.get('provider')))}；生成时间：{html.escape(str(metadata.get('generated_at_utc')))}。</li>
<li>当前面板是非 PIT、当前成分/可交易观察投影，不是生存者偏差安全的历史全市场证券主表；无法据此声称策略可直接实盘。</li>
<li>提供方覆盖状态：{html.escape(str(metadata.get('coverage_status')))}，可用代码 {metadata.get('available_symbols') or '—'} / 请求代码 {metadata.get('requested_symbols') or '—'}；缺失与身份隔离会影响小盘和退市样本代表性。</li>
        <li>面板观察行 {quality['panel_observation_rows']:,}，达到最低 {minimum_threshold:.0%} 绝对跳空阈值并写入明细的事件行 {quality['event_detail_rows']:,}；其中下一条观察不是全市场下一交易日的行 {quality['nonconsecutive_next_observation']:,}，这类行没有进入严格 H 统计。</li>
<li>计算：Polars 向量化，线程池 {pl.thread_pool_size()}；bootstrap 重复 {bootstrap_reps} 次；报告生成耗时 {elapsed_seconds:.1f}s。</li>
</ul><div class="callout warn"><strong>不要把“上涨概率高”直接当成可交易结论。</strong><br>本研究未加入买卖价差、滑点、借券、盘前/开盘成交可得性、停牌/熔断执行、公司分红总回报和点时点成分变更。下一步若要接近实盘，应在冻结的 PIT 股票池上做事件入样、可执行开盘成交和成本压力测试。</div></div>
<div class="section"><h2>可下载产物</h2><ul><li><a href="event_details.parquet">event_details.parquet</a>：全部候选事件与 H5/H10/H20 明细。</li><li><a href="summary_statistics.csv">summary_statistics.csv</a>：全部阈值、方向、口径、去簇统计。</li><li><a href="raw_positive_5pct_by_year.csv">raw_positive_5pct_by_year.csv</a>：年份稳定性。</li><li><a href="raw_positive_gap_buckets.csv">raw_positive_gap_buckets.csv</a>：跳空幅度分桶。</li><li><a href="manifest.json">manifest.json</a>：输入面板、协议和运行环境。</li></ul><p class="small">报告启动时间（UTC）：{html.escape(started_at)}</p></div>
</main></body></html>"""
    report_path = output_dir / "report.html"
    report_path.write_text(report_html, encoding="utf-8")
    return report_path


def run_study(
    *,
    panel_glob: str = DEFAULT_PANEL_GLOB,
    output_dir: Path | None = None,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    bootstrap_reps: int = 200,
    cooldown_sessions: int = COOLDOWN_SESSIONS,
) -> Path:
    started = datetime.now(timezone.utc)
    started_at = started.isoformat()
    panel_meta_path = _panel_root(panel_glob) / "_metadata.json"
    if not panel_meta_path.exists():
        raise FileNotFoundError(f"面板元数据不存在: {panel_meta_path}")
    metadata_raw = json.loads(panel_meta_path.read_text(encoding="utf-8"))
    metadata = _compact_metadata(metadata_raw)
    events = build_event_candidates(panel_glob)
    thresholds = parse_thresholds(thresholds)
    stats = build_statistics(
        events,
        thresholds,
        bootstrap_reps=bootstrap_reps,
        cooldown_sessions=cooldown_sessions,
    )
    year_stats = build_year_statistics(events)
    gap_buckets = build_gap_buckets(events)
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = PROJECT_ROOT / "var" / "reports" / f"us-overnight-gap-event-study-{stamp}"
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    return render_report(
        output_dir,
        metadata=metadata,
        panel_glob=panel_glob,
        events=events,
        stats=stats,
        year_stats=year_stats,
        gap_buckets=gap_buckets,
        started_at=started_at,
        elapsed_seconds=elapsed,
        bootstrap_reps=bootstrap_reps,
        cooldown_sessions=cooldown_sessions,
        minimum_threshold=min(thresholds),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="美股隔夜跳空大样本事件研究")
    parser.add_argument("--panel-glob", default=os.environ.get("US_PANEL_GLOB", DEFAULT_PANEL_GLOB))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--thresholds", default=','.join(str(x) for x in DEFAULT_THRESHOLDS))
    parser.add_argument("--bootstrap-reps", type=int, default=200)
    parser.add_argument("--cooldown-sessions", type=int, default=COOLDOWN_SESSIONS)
    args = parser.parse_args(argv)
    report = run_study(
        panel_glob=args.panel_glob,
        output_dir=args.output_dir,
        thresholds=parse_thresholds(args.thresholds),
        bootstrap_reps=args.bootstrap_reps,
        cooldown_sessions=args.cooldown_sessions,
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
