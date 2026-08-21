"""Run one frozen factor universe in both directions over a full date window.

This runner intentionally does not select a sign on an earlier training
sample.  Every valid source expression is evaluated twice (direction +1 and
direction -1) on the exact same event-engine protocol.  The output is a
diagnostic full-window leaderboard, not an out-of-sample or production claim.

The input universe comes from an already archived leaderboard ``snapshot.json``
so a long-running job cannot silently absorb later database writes.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import shutil
import socket
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import date, datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from app.backtest.batch import (  # noqa: E402
    BatchBacktestSpec,
    flatten_factor_result,
    information_coefficients,
    oriented_expression_hash,
    rank_factor_results,
    run_cost_scenarios,
)
from app.backtest.engine import (  # noqa: E402
    _prepare_backtest_frame,
    _write_artifacts,
)
from app.backtest.vector import (  # noqa: E402
    VECTOR_SCREEN_PROTOCOL,
    run_vector_cost_scenarios,
)
from app.config import ASHARE_PANEL_GLOB, US_PANEL_GLOB  # noqa: E402
from scripts.factor_leaderboard import (  # noqa: E402
    REPORT_DIMENSIONS,
    _append_jsonl,
    _json_default,
    _load_jsonl,
    _sha256,
    _write_csv,
    _write_json,
    _write_json_atomic,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_PROTOCOL = "dual_direction_full_window_event_leaderboard_v1"
VECTOR_PROTOCOL = "dual_direction_full_window_vector_screen_leaderboard_v1"
EVENT_POLICY_LABEL = "NON_PIT_FULL_WINDOW_DIAGNOSTIC"
VECTOR_POLICY_LABEL = "NON_PIT_VECTOR_SCREEN_DIAGNOSTIC"
DIRECTIONS = (1, -1)

_WORKER_SPEC: BatchBacktestSpec | None = None
_WORKER_PANEL_GLOB = ""
_WORKER_SCENARIO_ENGINE = "event"
_ACTIVE_PROTOCOL = EVENT_PROTOCOL
_ACTIVE_POLICY_LABEL = EVENT_POLICY_LABEL


def _protocol_for_engine(engine: str) -> tuple[str, str]:
    if engine == "event":
        return EVENT_PROTOCOL, EVENT_POLICY_LABEL
    if engine == "vector_screen":
        return VECTOR_PROTOCOL, VECTOR_POLICY_LABEL
    raise ValueError(f"未知回测引擎: {engine}")


def _market_label(market: str) -> str:
    return "A股" if market == "ashare" else "美股"


def _portfolio_label(mode: str) -> str:
    return "纯多头" if mode == "long_only" else "多空"


def _orientation_id(source_expression_hash: str, direction: int) -> str:
    suffix = "plus" if direction > 0 else "minus"
    return f"{source_expression_hash}-{suffix}"


def _resolve_panel_bounds(panel_glob: str) -> tuple[str, str]:
    bounds = (
        pl.scan_parquet(panel_glob, hive_partitioning=True)
        .select(pl.col("trade_date").cast(pl.Date))
        .select(
            pl.min("trade_date").alias("date_min"),
            pl.max("trade_date").alias("date_max"),
        )
        .collect()
        .row(0, named=True)
    )
    return str(bounds["date_min"]), str(bounds["date_max"])


def _worker_init(
    spec: dict,
    panel_glob: str,
    scenario_engine: str,
) -> None:
    global _WORKER_SPEC, _WORKER_PANEL_GLOB
    global _WORKER_SCENARIO_ENGINE
    global _ACTIVE_PROTOCOL, _ACTIVE_POLICY_LABEL
    _WORKER_SPEC = BatchBacktestSpec(**{
        **spec,
        "slippage_bps": tuple(spec["slippage_bps"]),
    })
    _WORKER_PANEL_GLOB = panel_glob
    _WORKER_SCENARIO_ENGINE = scenario_engine
    _ACTIVE_PROTOCOL, _ACTIVE_POLICY_LABEL = _protocol_for_engine(
        scenario_engine
    )


def _record_base(record: dict) -> dict:
    profile = record.get("profile") or {}
    return {
        "protocol": _ACTIVE_PROTOCOL,
        "policy_label": _ACTIVE_POLICY_LABEL,
        "production_eligible": False,
        "market": _WORKER_SPEC.market if _WORKER_SPEC else "unknown",
        "portfolio_mode": _WORKER_SPEC.mode if _WORKER_SPEC else "unknown",
        "sample_is_out_of_sample": False,
        "evaluation_window_kind": "full_window_2020_to_latest",
        "direction_policy": "both_directions_forced_no_sign_selection",
        "source_expression_hash": record["expression_hash"],
        "expression": record["expression"],
        "origin_scope": record.get("origin_scope", "unknown"),
        "origin_markets": record.get("origin_markets", []),
        "selection_reason": record.get("selection_reason"),
        "experiment_ids": record.get("experiment_ids", []),
        "factor_ids": record.get("factor_ids", []),
        "node_ids": record.get("node_ids", []),
        "source_record_count": record.get("source_record_count", 0),
        "complexity": profile.get("complexity"),
        "required_history": profile.get("required_history"),
        "fields": profile.get("fields", []),
        "operators": profile.get("operators", []),
        "worker_pid": os.getpid(),
    }


def _frame_for_expression(expression: str) -> pl.DataFrame:
    if _WORKER_SPEC is None:
        raise RuntimeError("worker 未初始化")
    frame, _ = _prepare_backtest_frame(
        expression=expression,
        universe_n=_WORKER_SPEC.universe_n,
        start=_WORKER_SPEC.holdout_start,
        end=_WORKER_SPEC.holdout_end,
        panel_glob=_WORKER_PANEL_GLOB,
        market=_WORKER_SPEC.market,
        forward_horizon=_WORKER_SPEC.horizon,
    )
    return frame


def _direction_failure(
    record: dict,
    *,
    direction: int,
    exc: BaseException,
    runtime_seconds: float,
) -> dict:
    return {
        "status": "error",
        "source_expression_hash": record["expression_hash"],
        "expression_hash": _orientation_id(
            record["expression_hash"],
            direction,
        ),
        "expression": record["expression"],
        "direction": direction,
        "error_type": type(exc).__name__,
        "error": str(exc)[:4000],
        "traceback_tail": traceback.format_exc(limit=5)[-6000:],
        "runtime_seconds": round(runtime_seconds, 4),
    }


def _worker_factor(record: dict) -> dict:
    if _WORKER_SPEC is None:
        raise RuntimeError("worker 未初始化")
    started = time.perf_counter()
    source_hash = record["expression_hash"]
    try:
        frame = _frame_for_expression(record["expression"])
    except Exception as exc:  # noqa: BLE001 - preserve every failed candidate
        elapsed = time.perf_counter() - started
        return {
            "status": "error",
            "source_expression_hash": source_hash,
            "expression": record["expression"],
            "worker_pid": os.getpid(),
            "orientation_results": [],
            "orientation_failures": [
                _direction_failure(
                    record,
                    direction=direction,
                    exc=exc,
                    runtime_seconds=elapsed,
                )
                for direction in DIRECTIONS
            ],
            "runtime_seconds": round(elapsed, 4),
        }

    orientation_results: list[dict] = []
    orientation_failures: list[dict] = []
    base = _record_base(record)
    for direction in DIRECTIONS:
        direction_started = time.perf_counter()
        try:
            window_ic = information_coefficients(
                frame,
                start=_WORKER_SPEC.holdout_start,
                end=_WORKER_SPEC.holdout_end,
                horizon=_WORKER_SPEC.horizon,
                universe_n=_WORKER_SPEC.universe_n,
                direction=direction,
            )
            if _WORKER_SCENARIO_ENGINE == "event":
                scenarios = run_cost_scenarios(
                    frame,
                    direction=direction,
                    spec=_WORKER_SPEC,
                    capture_detail=False,
                )
            else:
                scenarios = run_vector_cost_scenarios(
                    frame,
                    direction=direction,
                    spec=_WORKER_SPEC,
                    capture_periods=False,
                )
            orientation_results.append({
                **base,
                "status": "ok",
                "expression_hash": _orientation_id(
                    source_hash,
                    direction,
                ),
                "direction": direction,
                "oriented_expression_hash": oriented_expression_hash(
                    record["expression"],
                    direction,
                ),
                # Keep the established flatten/ranking schema while making
                # the full-window semantics explicit in separate fields.
                "train_ic": {
                    "sample_role": "not_applicable_full_window",
                },
                "holdout_ic": window_ic,
                "scenarios": scenarios,
                "score_window_start": _WORKER_SPEC.holdout_start,
                "score_window_end": _WORKER_SPEC.holdout_end,
                "runtime_seconds": round(
                    time.perf_counter() - direction_started,
                    4,
                ),
            })
        except Exception as exc:  # noqa: BLE001
            orientation_failures.append(_direction_failure(
                record,
                direction=direction,
                exc=exc,
                runtime_seconds=time.perf_counter() - direction_started,
            ))

    if len(orientation_results) == len(DIRECTIONS):
        status = "ok"
    elif orientation_results:
        status = "partial"
    else:
        status = "error"
    return {
        "status": status,
        "source_expression_hash": source_hash,
        "expression": record["expression"],
        "worker_pid": os.getpid(),
        "orientation_results": orientation_results,
        "orientation_failures": orientation_failures,
        "runtime_seconds": round(time.perf_counter() - started, 4),
    }


def _worker_ledger(
    record: dict,
    direction: int,
    orientation_id: str,
    artifact_dir: str,
) -> dict:
    if _WORKER_SPEC is None:
        raise RuntimeError("worker 未初始化")
    started = time.perf_counter()
    try:
        frame = _frame_for_expression(record["expression"])
        scenario = run_cost_scenarios(
            frame,
            direction=direction,
            spec=replace(_WORKER_SPEC, slippage_bps=(15.0,)),
            capture_detail=True,
        )["15"]
        result = scenario.pop("result")
        manifest = _write_artifacts(result, Path(artifact_dir))
        return {
            "status": "ok",
            "expression_hash": orientation_id,
            "source_expression_hash": record["expression_hash"],
            "direction": direction,
            "summary": scenario,
            "artifact_dir": artifact_dir,
            "manifest": manifest,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "expression_hash": orientation_id,
            "source_expression_hash": record["expression_hash"],
            "direction": direction,
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "traceback_tail": traceback.format_exc(limit=5)[-6000:],
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }


def _covered_directions(envelope: dict) -> set[int]:
    return {
        int(row["direction"])
        for key in ("orientation_results", "orientation_failures")
        for row in envelope.get(key, [])
        if row.get("direction") in DIRECTIONS
    }


def _portfolio_path_key(row: dict) -> str:
    fingerprint = row.get("portfolio_fingerprint_bps_15")
    if fingerprint:
        return f"portfolio:{fingerprint}"
    # Compatibility fallback for completed reports created before the vector
    # engine emitted an exact holdings-path fingerprint.  The deliberately
    # rich scenario signature collapses aliases that produce the same target
    # portfolio while leaving IC values out of the key: two expressions can
    # have slightly different cross-sectional IC yet place identical trades.
    scenario_fields = (
        "ann_return",
        "sharpe",
        "max_drawdown",
        "avg_daily_turnover",
        "final_nav",
        "orders_proxy",
        "commission_and_tax",
        "slippage_cost",
        "borrow_cost",
        "total_execution_cost",
        "avg_gross_exposure",
        "avg_net_exposure",
    )
    payload = {
        f"{field}_bps_{bps}": row.get(f"{field}_bps_{bps}")
        for bps in (0, 5, 15)
        for field in scenario_fields
    }
    return "scenario:" + hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()[:16]


def _select_independent_ledger_rows(
    rows: list[dict],
    limit: int,
) -> list[dict]:
    selected: list[dict] = []
    seen_paths: set[str] = set()
    for row in rows:
        path_key = _portfolio_path_key(row)
        if path_key in seen_paths:
            continue
        seen_paths.add(path_key)
        selected.append(row)
        if len(selected) >= limit:
            break
    return selected


def _latest_envelopes(path: Path) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for row in _load_jsonl(path):
        source_hash = str(
            row.get("source_expression_hash")
            or row.get("expression_hash")
            or ""
        )
        if source_hash:
            latest[source_hash] = row
    return latest


def _orientation_counts(envelopes: dict[str, dict]) -> tuple[int, int]:
    ok = sum(
        len(row.get("orientation_results", []))
        for row in envelopes.values()
    )
    errors = sum(
        len(row.get("orientation_failures", []))
        for row in envelopes.values()
    )
    return ok, errors


def _progress_payload(
    *,
    started_at: float,
    completed: int,
    total: int,
    orientation_ok: int,
    orientation_errors: int,
    workers: int,
    threads_per_worker: int,
    phase: str,
    window_start: str,
    window_end: str,
    latest_source_hash: str | None = None,
    latest_runtime_seconds: float | None = None,
) -> dict:
    elapsed = max(0.0, time.time() - started_at)
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - completed)
    return {
        "protocol": _ACTIVE_PROTOCOL,
        "policy_label": _ACTIVE_POLICY_LABEL,
        "phase": phase,
        "coordinator_pid": os.getpid(),
        "completed": completed,
        "total": total,
        "percent": round(100.0 * completed / max(1, total), 3),
        "completed_orientations": orientation_ok + orientation_errors,
        "total_orientations": total * len(DIRECTIONS),
        "successful_orientations": orientation_ok,
        "failed_orientations": orientation_errors,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "allocated_compute_threads": workers * threads_per_worker,
        "window_start": window_start,
        "window_end": window_end,
        "elapsed_seconds": round(elapsed, 1),
        "factors_per_second": round(rate, 6),
        "eta_seconds": round(remaining / rate, 1) if rate > 0 else None,
        "latest_source_expression_hash": latest_source_hash,
        "latest_runtime_seconds": latest_runtime_seconds,
        "updated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def _normalise_snapshot(
    source: dict,
    *,
    market: str,
    mode: str,
    source_report: Path,
    source_sha256: str,
) -> dict:
    snapshot = dict(source)
    snapshot["target_market"] = market
    snapshot["portfolio_mode"] = mode
    snapshot["valid_target_expressions"] = len(
        snapshot.get("expressions", [])
    )
    snapshot.setdefault("invalid_target_expressions", len(
        snapshot.get("invalid", [])
    ))
    snapshot["source_report_archive"] = str(source_report)
    snapshot["source_snapshot_sha256"] = source_sha256
    snapshot["frozen_for_protocol"] = _ACTIVE_PROTOCOL
    return snapshot


def _add_window_aliases(rows: list[dict]) -> None:
    aliases = (
        "n_days",
        "mean_cross_section_n",
        "ic_mean",
        "ic_std",
        "icir",
        "ic_positive_rate",
        "rank_ic_mean",
        "rank_ic_std",
        "rank_icir",
        "rank_ic_positive_rate",
        "ic_t",
        "rank_ic_t",
        "ic_p_normal",
        "rank_ic_p_normal",
        "ic_bh_q",
        "rank_ic_bh_q",
    )
    for row in rows:
        for suffix in aliases:
            key = f"oos_{suffix}"
            if key in row:
                row[f"window_{suffix}"] = row[key]


def _write_markdown(
    path: Path,
    *,
    ranked: list[dict],
    spec: BatchBacktestSpec,
    source_report: Path,
) -> None:
    representatives = [
        row for row in ranked if row.get("economic_representative")
    ]
    title = (
        f"{_market_label(spec.market)}{_portfolio_label(spec.mode)}"
        "：2020 至最新交易日双方向"
        + (
            "事件回测榜单"
            if _WORKER_SCENARIO_ENGINE == "event"
            else "向量筛选榜单"
        )
    )
    lines = [
        f"# {title}",
        "",
        f"- 协议：`{_ACTIVE_PROTOCOL}`",
        f"- 区间：{spec.holdout_start} 至 {spec.holdout_end}",
        f"- 冻结输入：`{source_report}`",
        "- 每个表达式强制测试 `+1` 与 `-1`，没有先验择向。",
        (
            "- 排名样本是完整窗口，不是独立 OOS；标签为 "
            f"`{_ACTIVE_POLICY_LABEL}`，不可解释为实盘批准。"
        ),
        (
            (
                "- 成本：事件引擎内含市场佣金/税费；另测 "
                "0/5/15 BPS 双向滑点。"
            )
            if _WORKER_SCENARIO_ENGINE == "event"
            else (
                "- 成本：向量阶段按目标权重变化估算市场费率与 "
                "0/5/15 BPS 双向滑点；最终排名须经事件账本复核。"
            )
        ),
        "",
        (
            "|排名|方向|表达式 Hash|年化 0bp|年化 5bp|年化 15bp|"
            "夏普 15bp|IC|ICIR|RankIC|RankICIR|稳健分|表达式|"
        ),
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in representatives[:50]:
        lines.append(
            "|"
            + "|".join([
                str(row.get("overall_rank", "—")),
                f"{int(row['direction']):+d}",
                f"`{row['expression_hash']}`",
                f"{float(row.get('ann_return_bps_0', 0)) * 100:.2f}%",
                f"{float(row.get('ann_return_bps_5', 0)) * 100:.2f}%",
                f"{float(row.get('ann_return_bps_15', 0)) * 100:.2f}%",
                f"{float(row.get('sharpe_bps_15', 0)):.3f}",
                f"{float(row.get('window_ic_mean', 0)):.4f}",
                f"{float(row.get('window_icir', 0)):.3f}",
                f"{float(row.get('window_rank_ic_mean', 0)):.4f}",
                f"{float(row.get('window_rank_icir', 0)):.3f}",
                f"{float(row.get('robust_score', 0)):.2f}",
                f"`{row['expression']}`",
            ])
            + "|"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_source_diagnostics(source_report: Path, output_dir: Path) -> None:
    for name in ("invalid_expressions.csv", "excluded_expressions.csv"):
        source = source_report / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=("ashare", "us"), required=True)
    parser.add_argument(
        "--portfolio-mode",
        choices=("long_only", "long_short"),
        required=True,
    )
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--panel-glob")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="latest")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument(
        "--scenario-engine",
        choices=("event", "vector_screen"),
        default="event",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ledger-top", type=int, default=3)
    parser.add_argument("--borrow-cost-bps-annual", type=float)
    parser.add_argument("--initial-capital", type=float)
    return parser.parse_args()


def main() -> None:
    global _ACTIVE_PROTOCOL, _ACTIVE_POLICY_LABEL
    global _WORKER_SCENARIO_ENGINE
    args = _parse_args()
    _ACTIVE_PROTOCOL, _ACTIVE_POLICY_LABEL = _protocol_for_engine(
        args.scenario_engine
    )
    _WORKER_SCENARIO_ENGINE = args.scenario_engine
    if args.market == "ashare" and args.portfolio_mode != "long_only":
        raise SystemExit("A股事件引擎只允许 long_only")
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers 必须在 1..4")
    if not 1 <= args.threads_per_worker <= 10:
        raise SystemExit("--threads-per-worker 必须在 1..10")
    if args.limit < 0 or args.ledger_top < 0:
        raise SystemExit("--limit 与 --ledger-top 不能为负数")

    source_report = args.source_report.resolve()
    source_snapshot_path = source_report / "snapshot.json"
    source_protocol_path = source_report / "protocol.json"
    if not source_snapshot_path.exists():
        raise SystemExit(f"冻结来源缺少 snapshot.json: {source_report}")
    if not source_protocol_path.exists():
        raise SystemExit(f"冻结来源缺少 protocol.json: {source_report}")

    source_snapshot = json.loads(
        source_snapshot_path.read_text(encoding="utf-8")
    )
    source_protocol = json.loads(
        source_protocol_path.read_text(encoding="utf-8")
    )
    protocol_spec = source_protocol.get("spec") or {}
    source_market = (
        source_protocol.get("target_market")
        or protocol_spec.get("market")
        or source_snapshot.get("target_market")
    )
    source_mode = (
        protocol_spec.get("mode")
        or source_snapshot.get("portfolio_mode")
    )
    if source_market and source_market != args.market:
        raise SystemExit(
            f"冻结来源市场为 {source_market}，运行参数为 {args.market}"
        )
    if source_mode and source_mode != args.portfolio_mode:
        raise SystemExit(
            f"冻结来源模式为 {source_mode}，运行参数为 "
            f"{args.portfolio_mode}"
        )

    panel_glob = (
        args.panel_glob
        or (ASHARE_PANEL_GLOB if args.market == "ashare" else US_PANEL_GLOB)
    )
    panel_files = [Path(path) for path in sorted(glob.glob(panel_glob))]
    if not panel_files:
        raise SystemExit(f"面板路径未匹配任何文件: {panel_glob}")
    panel_date_min, panel_date_max = _resolve_panel_bounds(panel_glob)
    window_start = str(date.fromisoformat(args.start))
    window_end = (
        panel_date_max
        if args.end == "latest"
        else str(date.fromisoformat(args.end))
    )
    if date.fromisoformat(window_start) < date.fromisoformat(panel_date_min):
        raise SystemExit(
            f"开始日期 {window_start} 早于面板起点 {panel_date_min}"
        )
    if date.fromisoformat(window_end) > date.fromisoformat(panel_date_max):
        raise SystemExit(
            f"结束日期 {window_end} 晚于面板终点 {panel_date_max}"
        )
    if date.fromisoformat(window_start) >= date.fromisoformat(window_end):
        raise SystemExit("回测开始日期必须早于结束日期")

    borrow_cost = (
        float(args.borrow_cost_bps_annual)
        if args.borrow_cost_bps_annual is not None
        else (
            300.0
            if args.market == "us"
            and args.portfolio_mode == "long_short"
            else 0.0
        )
    )
    initial_capital = (
        float(args.initial_capital)
        if args.initial_capital is not None
        else (10_000_000.0 if args.market == "ashare" else 1_000_000.0)
    )
    spec = BatchBacktestSpec(
        market=args.market,
        mode=args.portfolio_mode,
        universe_n=int(protocol_spec.get("universe_n", 500)),
        horizon=int(protocol_spec.get("horizon", 5)),
        top_fraction=float(protocol_spec.get("top_fraction", 0.20)),
        initial_capital=initial_capital,
        rebalance_every=int(protocol_spec.get("rebalance_every", 5)),
        max_volume_participation=float(
            protocol_spec.get("max_volume_participation", 0.05)
        ),
        train_start=window_start,
        train_end=window_end,
        holdout_start=window_start,
        holdout_end=window_end,
        vault_start=window_end,
        vault_end=window_end,
        slippage_bps=(0.0, 5.0, 15.0),
        borrow_cost_bps_annual=borrow_cost,
        fee_profile=protocol_spec.get("fee_profile"),
    )
    if borrow_cost < 0 or initial_capital <= 0:
        raise SystemExit("资金必须为正，借券费不能为负")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_snapshot_sha = _sha256(source_snapshot_path)
    snapshot = _normalise_snapshot(
        source_snapshot,
        market=args.market,
        mode=args.portfolio_mode,
        source_report=source_report,
        source_sha256=source_snapshot_sha,
    )
    snapshot_path = output_dir / "snapshot.json"
    if snapshot_path.exists():
        existing_snapshot = json.loads(
            snapshot_path.read_text(encoding="utf-8")
        )
        if (
            existing_snapshot.get("source_snapshot_sha256")
            != source_snapshot_sha
        ):
            raise SystemExit(
                "输出目录已有不同冻结快照，拒绝混合续算"
            )
    else:
        _write_json(snapshot_path, snapshot)
    _copy_source_diagnostics(source_report, output_dir)

    expressions = list(snapshot.get("expressions", []))
    if args.limit > 0:
        expressions = expressions[: args.limit]
    by_source_hash = {
        str(row["expression_hash"]): row for row in expressions
    }

    panel_inventory = [
        {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in panel_files
    ]
    panel_identity = hashlib.sha256(
        "\n".join(
            f"{row['path']}|{row['bytes']}|{row['mtime_ns']}"
            for row in panel_inventory
        ).encode("utf-8")
    ).hexdigest()
    execution_sources = [
        PROJECT_ROOT / "backend/app/backtest/engine.py",
        PROJECT_ROOT / "backend/app/backtest/fees.py",
        PROJECT_ROOT / "backend/app/backtest/batch.py",
        PROJECT_ROOT / "backend/app/backtest/vector.py",
        PROJECT_ROOT / "backend/app/data/panel.py",
        PROJECT_ROOT / "backend/app/dsl/engine.py",
        Path(__file__).resolve(),
    ]
    execution_source_sha256 = {
        str(path.relative_to(PROJECT_ROOT)): _sha256(path)
        for path in execution_sources
    }
    identity_payload = {
        "protocol": _ACTIVE_PROTOCOL,
        "source_snapshot_sha256": source_snapshot_sha,
        "panel_identity_path_size_mtime_sha256": panel_identity,
        "spec": spec.as_dict(),
        "directions": list(DIRECTIONS),
        "limit": args.limit,
        "execution_source_sha256": execution_source_sha256,
    }
    run_identity = hashlib.sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()
    protocol_path = output_dir / "protocol.json"
    protocol_payload = {
        **identity_payload,
        "run_identity_sha256": run_identity,
        "policy_label": _ACTIVE_POLICY_LABEL,
        "production_eligible": False,
        "sample_is_out_of_sample": False,
        "target_market": args.market,
        "portfolio_mode": args.portfolio_mode,
        "source_report_archive": str(source_report),
        "source_protocol_sha256": _sha256(source_protocol_path),
        "panel_glob": panel_glob,
        "panel_date_min": panel_date_min,
        "panel_date_max": panel_date_max,
        "panel_inventory": panel_inventory,
        "execution_source_sha256": execution_source_sha256,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "allocated_compute_threads": (
            args.workers * args.threads_per_worker
        ),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "direction_selection": "disabled_both_directions_forced",
        "scenario_engine": args.scenario_engine,
        "screening_only": args.scenario_engine == "vector_screen",
        "vector_screen_protocol": (
            VECTOR_SCREEN_PROTOCOL
            if args.scenario_engine == "vector_screen"
            else None
        ),
        "ranking_window": {
            "start": window_start,
            "end": window_end,
            "semantics": "full_window_not_independent_oos",
        },
        "ranking_uses_vault": False,
        "short_borrow_proxy": (
            {
                "annual_bps": borrow_cost,
                "semantics": (
                    "constant_pressure_proxy_not_historical_locate_data"
                ),
            }
            if args.portfolio_mode == "long_short"
            else None
        ),
    }
    if protocol_path.exists():
        existing_protocol = json.loads(
            protocol_path.read_text(encoding="utf-8")
        )
        if existing_protocol.get("run_identity_sha256") != run_identity:
            raise SystemExit(
                "输出目录已有不同面板、日期或参数的结果，拒绝混合续算"
            )
    else:
        _write_json(protocol_path, protocol_payload)
        _write_json(
            output_dir / "result_generation_protocol.json",
            protocol_payload,
        )

    results_path = output_dir / "results.jsonl"
    envelopes = _latest_envelopes(results_path)
    completed_hashes = {
        source_hash
        for source_hash, row in envelopes.items()
        if _covered_directions(row) == set(DIRECTIONS)
    }
    pending = [
        row for row in expressions
        if row["expression_hash"] not in completed_hashes
    ]
    total = len(expressions)
    started_at = time.time()
    orientation_ok, orientation_errors = _orientation_counts(envelopes)
    initial_progress = _progress_payload(
        started_at=started_at,
        completed=len(completed_hashes),
        total=total,
        orientation_ok=orientation_ok,
        orientation_errors=orientation_errors,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        phase="factor_backtest",
        window_start=window_start,
        window_end=window_end,
    )
    _write_json_atomic(output_dir / "progress.json", initial_progress)

    os.environ["POLARS_MAX_THREADS"] = str(args.threads_per_worker)
    print(
        f"{_market_label(args.market)}"
        f"{_portfolio_label(args.portfolio_mode)} · "
        f"{window_start}..{window_end} · "
        f"{args.scenario_engine} · "
        f"双方向 {len(pending) * 2}/{total * 2} 待测 · "
        f"{args.workers} worker × "
        f"{args.threads_per_worker} Polars threads",
        flush=True,
    )

    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(
            spec.as_dict(),
            panel_glob,
            args.scenario_engine,
        ),
    ) as executor:
        futures = {
            executor.submit(_worker_factor, row): row["expression_hash"]
            for row in pending
        }
        for future in as_completed(futures):
            source_hash = futures[future]
            try:
                envelope = future.result()
            except Exception as exc:  # noqa: BLE001
                record = by_source_hash[source_hash]
                envelope = {
                    "status": "error",
                    "source_expression_hash": source_hash,
                    "expression": record["expression"],
                    "orientation_results": [],
                    "orientation_failures": [
                        _direction_failure(
                            record,
                            direction=direction,
                            exc=exc,
                            runtime_seconds=0.0,
                        )
                        for direction in DIRECTIONS
                    ],
                    "runtime_seconds": 0.0,
                }
            _append_jsonl(results_path, envelope)
            envelopes[source_hash] = envelope
            completed_hashes.add(source_hash)
            orientation_ok, orientation_errors = _orientation_counts(
                envelopes
            )
            progress = _progress_payload(
                started_at=started_at,
                completed=len(completed_hashes),
                total=total,
                orientation_ok=orientation_ok,
                orientation_errors=orientation_errors,
                workers=args.workers,
                threads_per_worker=args.threads_per_worker,
                phase="factor_backtest",
                window_start=window_start,
                window_end=window_end,
                latest_source_hash=source_hash,
                latest_runtime_seconds=envelope.get("runtime_seconds"),
            )
            _write_json_atomic(output_dir / "progress.json", progress)
            done = len(completed_hashes)
            if done % 10 == 0 or done == total:
                print(
                    f"[{done}/{total} {progress['percent']:.1f}%] "
                    f"方向成功={orientation_ok} "
                    f"失败={orientation_errors} "
                    f"eta={progress['eta_seconds'] or '—'}s",
                    flush=True,
                )

        orientation_results = [
            result
            for envelope in envelopes.values()
            for result in envelope.get("orientation_results", [])
        ]
        orientation_failures = [
            failure
            for envelope in envelopes.values()
            for failure in envelope.get("orientation_failures", [])
        ]
        flat = [
            flatten_factor_result(result)
            for result in orientation_results
        ]
        ranked = rank_factor_results(flat)
        _add_window_aliases(ranked)

        finalising_progress = _progress_payload(
            started_at=started_at,
            completed=total,
            total=total,
            orientation_ok=len(orientation_results),
            orientation_errors=len(orientation_failures),
            workers=args.workers,
            threads_per_worker=args.threads_per_worker,
            phase="finalist_ledger",
            window_start=window_start,
            window_end=window_end,
        )
        _write_json_atomic(
            output_dir / "progress.json",
            finalising_progress,
        )
        representatives = [
            row for row in ranked if row["economic_representative"]
        ]
        ledger_candidates = (
            [row for row in representatives if row["practical_pass"]]
            or representatives
        )
        ledger_rows = _select_independent_ledger_rows(
            ledger_candidates,
            args.ledger_top,
        )
        _write_json(
            output_dir / "finalists_frozen_before_ledger.json",
            {
                "ranking_uses_post_ranking_ledger": False,
                "frozen_at": datetime.now(timezone.utc).isoformat(
                    timespec="seconds"
                ),
                "orientation_ids": [
                    row["expression_hash"] for row in ledger_rows
                ],
            },
        )
        ledger_root = output_dir / "finalist_ledgers"
        ledger_root.mkdir(exist_ok=True)
        ledger_futures = {}
        for row in ledger_rows:
            source_hash = row["source_expression_hash"]
            ledger_futures[executor.submit(
                _worker_ledger,
                by_source_hash[source_hash],
                int(row["direction"]),
                str(row["expression_hash"]),
                str(ledger_root / str(row["expression_hash"])),
            )] = str(row["expression_hash"])
        ledger_results: dict[str, dict] = {}
        for future in as_completed(ledger_futures):
            result = future.result()
            ledger_results[str(result["expression_hash"])] = result
        _write_json(
            output_dir / "finalist_ledger_audits.json",
            ledger_results,
        )

    _write_csv(output_dir / "leaderboard_full.csv", ranked)
    if ranked:
        pl.DataFrame(ranked, strict=False).write_parquet(
            output_dir / "leaderboard_full.parquet",
            compression="zstd",
        )
    _write_json(output_dir / "leaderboard_full.json", ranked)
    _write_csv(output_dir / "failures.csv", orientation_failures)

    representatives = [
        row for row in ranked if row["economic_representative"]
    ]
    top_by_dimension = {
        dimension: sorted(
            representatives,
            key=lambda row: float(
                row.get(dimension, float("-inf"))
            ),
            reverse=True,
        )[:20]
        for dimension in REPORT_DIMENSIONS
    }
    _write_json(output_dir / "top_by_dimension.json", top_by_dimension)
    dimension_rows = [
        {
            "dimension": dimension,
            "dimension_rank": dimension_rank,
            "dimension_value": row.get(dimension),
            **row,
        }
        for dimension, rows in top_by_dimension.items()
        for dimension_rank, row in enumerate(rows, start=1)
    ]
    _write_csv(output_dir / "top_by_dimension.csv", dimension_rows)
    _write_json(output_dir / "vault_finalists.json", {})
    _write_markdown(
        output_dir / "leaderboard.md",
        ranked=ranked,
        spec=spec,
        source_report=source_report,
    )

    complete_progress = _progress_payload(
        started_at=started_at,
        completed=total,
        total=total,
        orientation_ok=len(orientation_results),
        orientation_errors=len(orientation_failures),
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
        phase="complete",
        window_start=window_start,
        window_end=window_end,
    )
    _write_json_atomic(output_dir / "progress.json", complete_progress)
    artifacts = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != output_dir / "manifest.json":
            artifacts[str(path.relative_to(output_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "protocol": _ACTIVE_PROTOCOL,
        "policy_label": _ACTIVE_POLICY_LABEL,
        "target_market": args.market,
        "portfolio_mode": args.portfolio_mode,
        "scenario_engine": args.scenario_engine,
        "screening_only": args.scenario_engine == "vector_screen",
        "event_verified_orientations": len(ledger_results),
        "completed_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "duration_seconds": round(time.time() - started_at, 2),
        "window_start": window_start,
        "window_end": window_end,
        "source_report_archive": str(source_report),
        "source_snapshot_sha256": source_snapshot_sha,
        "scheduled_source_expressions": total,
        "scheduled_orientations": total * len(DIRECTIONS),
        "successful_orientations": len(ranked),
        "failed_orientations": len(orientation_failures),
        "economic_equivalence_groups": sum(
            bool(row["economic_representative"]) for row in ranked
        ),
        "direction_counts": dict(sorted(Counter(
            f"{int(row['direction']):+d}"
            for row in ranked
            if row["economic_representative"]
        ).items())),
        "practical_pass": sum(
            bool(row["practical_pass"])
            and bool(row["economic_representative"])
            for row in ranked
        ),
        "finalist_ledgers": len(ledger_results),
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "artifacts": artifacts,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(
        f"完成 · 双方向成功 {len(ranked)} · "
        f"失败 {len(orientation_failures)} · "
        f"经济等价组 {manifest['economic_equivalence_groups']} · "
        f"{output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
