#!/usr/bin/env python3
"""Replay and cluster the frozen top-100 rows from four US leaderboards.

Inputs:
* internal US long-short top 100;
* internal US long-only top 100;
* external joint-library US long-short top 100;
* external joint-library US long-only top 100.

Directions are copied from each leaderboard and never re-selected.  All rows
are replayed on one common 2020-to-latest panel/window at 15 BPS.  Long-only
uses benchmark-active period returns and long-short uses fee/borrow-adjusted
net period returns, making the two modes comparable as directed active-return
streams.  Positive-correlation and absolute-correlation clusters are both
reported so sign inversions cannot inflate the economic-source count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.backtest.batch import (  # noqa: E402
    BatchBacktestSpec,
    canonical_expression,
    oriented_expression_hash,
)
from app.backtest.engine import _prepare_backtest_frame  # noqa: E402
from app.backtest.vector import run_vector_cost_scenarios  # noqa: E402
from app.factors.diversity import infer_mechanism  # noqa: E402


PROTOCOL = "us_four_leaderboard_top100_return_source_audit_v1"
POLICY_LABEL = "FULL_WINDOW_FIXED_DIRECTION_DIVERSITY_DIAGNOSTIC"
REPORT_TITLE = "美股四榜前100独立收益来源审计"
MARKET = "us"
DEFAULT_REPORTS = {
    "内部多空": "us-long-short-all-factor-leaderboard-snapshot-2208-1522",
    "内部纯多": "us-long-only-all-factor-leaderboard-snapshot-2208-1522",
    "外部多空": "external-joint-ashare-us-to-us-long-short-vector-2020-latest-20260821",
    "外部纯多": "external-joint-ashare-us-to-us-long-only-vector-2020-latest-20260821",
}
MODE_BY_LIST = {
    "内部多空": "long_short",
    "内部纯多": "long_only",
    "外部多空": "long_short",
    "外部纯多": "long_only",
}
LIST_FILTERS: dict[str, dict[str, str]] = {}
PANEL_PROTOCOL_LIST_IDS = ("外部多空", "外部纯多")
COMMON_START = "2020-01-01"
COMMON_END = "2026-08-19"
TOP_N = 100
RUN_SCRIPT_PATH = Path(__file__).resolve()


def _json_default(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value).__name__)


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_top100(list_id: str, report: Path) -> list[dict[str, Any]]:
    with (report / "leaderboard_full.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    representatives = [row for row in rows if _truth(row.get("economic_representative"))]
    filters = LIST_FILTERS.get(list_id, {})
    representatives = [
        row for row in representatives
        if all(str(row.get(field) or "") == expected for field, expected in filters.items())
    ]
    representatives.sort(key=lambda row: int(row["overall_rank"]))
    selected: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for row in representatives:
        rank = int(row["overall_rank"])
        if rank in seen_ranks:
            continue
        seen_ranks.add(rank)
        expression = str(row["expression"])
        direction = int(row["direction"])
        raw_hash, canonical = canonical_expression(expression)
        oriented_hash = oriented_expression_hash(expression, direction)
        selected.append({
            "list_id": list_id,
            "report": str(report.resolve()),
            "rank": rank,
            "portfolio_mode": MODE_BY_LIST[list_id],
            "direction": direction,
            "expression": expression,
            "canonical_expression": canonical,
            "canonical_expression_hash": raw_hash,
            "canonical_oriented_hash": oriented_hash,
            "leaderboard_expression_hash": str(row.get("expression_hash") or ""),
            "leaderboard_source_expression_hash": str(row.get("source_expression_hash") or ""),
            "leaderboard_oriented_expression_hash": str(row.get("oriented_expression_hash") or ""),
            "origin_scope": str(row.get("origin_scope") or ""),
            "leaderboard_ann_return_15bps": row.get("ranking_ann_return_bps_15") or row.get("ann_return_bps_15"),
            "leaderboard_sharpe_15bps": row.get("ranking_sharpe_bps_15") or row.get("sharpe_bps_15"),
        })
        if len(selected) == TOP_N:
            break
    if len(selected) != TOP_N:
        raise RuntimeError(f"{list_id}: 经济代表不足 {TOP_N}，实际 {len(selected)}")
    return selected


def _task_id(mode: str, oriented_hash: str) -> str:
    return f"{mode}:{oriented_hash}"


def _build_tasks(
    memberships: list[dict],
    panel_glob: str,
    dsl_fields: list[str],
) -> list[dict]:
    tasks: dict[str, dict] = {}
    for member in memberships:
        mode = member["portfolio_mode"]
        task_id = _task_id(mode, member["canonical_oriented_hash"])
        member["task_id"] = task_id
        spec = BatchBacktestSpec(
            market=MARKET,
            mode=mode,
            universe_n=500,
            horizon=5,
            top_fraction=0.20,
            initial_capital=1_000_000.0,
            rebalance_every=5,
            max_volume_participation=0.05,
            train_start=COMMON_START,
            train_end=COMMON_END,
            holdout_start=COMMON_START,
            holdout_end=COMMON_END,
            vault_start=COMMON_END,
            vault_end=COMMON_END,
            slippage_bps=(15.0,),
            borrow_cost_bps_annual=300.0 if mode == "long_short" else 0.0,
        )
        candidate = {
            "task_id": task_id,
            "portfolio_mode": mode,
            "canonical_oriented_hash": member["canonical_oriented_hash"],
            "expression": member["expression"],
            "direction": member["direction"],
            "spec": spec.as_dict(),
            "panel_glob": panel_glob,
            "dsl_fields": dsl_fields,
            "market": MARKET,
            "protocol": PROTOCOL,
            "policy_label": POLICY_LABEL,
        }
        if task_id in tasks:
            existing = tasks[task_id]
            if oriented_expression_hash(existing["expression"], existing["direction"]) != member["canonical_oriented_hash"]:
                raise RuntimeError(f"任务哈希冲突: {task_id}")
        else:
            tasks[task_id] = candidate
    return sorted(tasks.values(), key=lambda row: row["task_id"])


def _quality(row: dict) -> tuple[float, float, float]:
    scenario = row["scenario"]
    if row["portfolio_mode"] == "long_only":
        return (
            float(scenario.get("active_sharpe") or 0.0),
            float(scenario.get("active_ann_return") or 0.0),
            -float(scenario.get("active_max_drawdown") or 1.0),
        )
    return (
        float(scenario.get("sharpe") or 0.0),
        float(scenario.get("ann_return") or 0.0),
        -float(scenario.get("max_drawdown") or 1.0),
    )


def _worker(task: dict) -> dict:
    started = time.perf_counter()
    try:
        spec = BatchBacktestSpec(**{**task["spec"], "slippage_bps": (15.0,)})
        frame, _ = _prepare_backtest_frame(
            expression=task["expression"],
            universe_n=spec.universe_n,
            start=spec.holdout_start,
            end=spec.holdout_end,
            panel_glob=task["panel_glob"],
            market=task["market"],
            forward_horizon=spec.horizon,
            dsl_fields=task["dsl_fields"],
        )
        scenario = run_vector_cost_scenarios(
            frame,
            direction=int(task["direction"]),
            spec=replace(spec, slippage_bps=(15.0,)),
            capture_periods=True,
        )["15"]
        periods = list(scenario.pop("period_rows"))
        return_key = "active_return" if spec.mode == "long_only" else "net_return"
        returns = [float(period[return_key]) for period in periods]
        return {
            "protocol": task["protocol"],
            "policy_label": task["policy_label"],
            "task_id": task["task_id"],
            "portfolio_mode": spec.mode,
            "canonical_oriented_hash": task["canonical_oriented_hash"],
            "expression": task["expression"],
            "direction": int(task["direction"]),
            "direction_selection": "frozen_from_source_leaderboard",
            "return_basis": "active" if spec.mode == "long_only" else "net",
            "return_dates": [str(period["signal_date"]) for period in periods],
            "returns": returns,
            "scenario": scenario,
            "mechanism_family": infer_mechanism(task["expression"]),
            "status": "ok",
            "worker_pid": os.getpid(),
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "protocol": task["protocol"],
            "policy_label": task["policy_label"],
            "task_id": task["task_id"],
            "portfolio_mode": task["portfolio_mode"],
            "canonical_oriented_hash": task["canonical_oriented_hash"],
            "expression": task["expression"],
            "direction": int(task["direction"]),
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "traceback_tail": traceback.format_exc(limit=5)[-6000:],
            "worker_pid": os.getpid(),
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }


def _load_results(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[str(row["task_id"])] = row
    return rows


def _common_matrix(rows: list[dict]) -> tuple[list[str], np.ndarray]:
    date_sets = [set(row["return_dates"]) for row in rows]
    common_dates = sorted(set.intersection(*date_sets))
    if len(common_dates) < 60:
        raise RuntimeError(f"共同收益路径不足60期: {len(common_dates)}")
    matrix = np.asarray([
        [dict(zip(row["return_dates"], row["returns"]))[value] for value in common_dates]
        for row in rows
    ], dtype=np.float64)
    return common_dates, matrix


def _cluster(rows: list[dict], threshold: float, absolute: bool) -> tuple[list[dict], list[int], np.ndarray, int]:
    common_dates, matrix = _common_matrix(rows)
    correlations = np.nan_to_num(np.corrcoef(matrix), nan=0.0)
    order = sorted(range(len(rows)), key=lambda index: _quality(rows[index]), reverse=True)
    clusters: list[dict] = []
    assignments = [-1] * len(rows)
    for index in order:
        matched = None
        for cluster_index, cluster in enumerate(clusters):
            representative = int(cluster["representative_index"])
            correlation = float(correlations[index, representative])
            comparable = abs(correlation) if absolute else correlation
            if comparable >= threshold:
                matched = cluster_index
                break
        if matched is None:
            matched = len(clusters)
            clusters.append({"representative_index": index, "members": []})
        clusters[matched]["members"].append(index)
        assignments[index] = matched
    return clusters, assignments, correlations, len(common_dates)


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _progress(results: dict[str, dict], tasks: list[dict], started: float, phase: str) -> dict:
    completed_by_mode = Counter(row["portfolio_mode"] for row in results.values())
    total_by_mode = Counter(row["portfolio_mode"] for row in tasks)
    return {
        "protocol": PROTOCOL,
        "phase": phase,
        "completed": len(results),
        "total": len(tasks),
        "percent": round(100.0 * len(results) / max(1, len(tasks)), 3),
        "by_mode": {
            mode: {"completed": completed_by_mode[mode], "total": total}
            for mode, total in total_by_mode.items()
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _cluster_artifacts(
    name: str,
    rows: list[dict],
    memberships_by_task: dict[str, list[dict]],
    output: Path,
    threshold: float,
    absolute: bool,
) -> dict:
    clusters, assignments, correlations, common_periods = _cluster(rows, threshold, absolute)
    prefix = "U" if absolute else "D"
    assignment_rows: list[dict] = []
    for index, row in enumerate(rows):
        cluster_index = assignments[index]
        cluster = clusters[cluster_index]
        representative_index = int(cluster["representative_index"])
        memberships = memberships_by_task.get(row["task_id"], [])
        scenario = row["scenario"]
        assignment_rows.append({
            "scope": name,
            "cluster": f"{prefix}{cluster_index + 1:04d}",
            "is_representative": index == representative_index,
            "correlation_to_representative": round(float(correlations[index, representative_index]), 6),
            "absolute_correlation_to_representative": round(abs(float(correlations[index, representative_index])), 6),
            "task_id": row["task_id"],
            "portfolio_mode": row["portfolio_mode"],
            "direction": row["direction"],
            "mechanism_family": row["mechanism_family"],
            "source_lists": "|".join(sorted({member["list_id"] for member in memberships})),
            "source_ranks": "|".join(f"{member['list_id']}:{member['rank']}" for member in memberships),
            "ann_return": scenario.get("active_ann_return") if row["portfolio_mode"] == "long_only" else scenario.get("ann_return"),
            "sharpe": scenario.get("active_sharpe") if row["portfolio_mode"] == "long_only" else scenario.get("sharpe"),
            "max_drawdown": scenario.get("active_max_drawdown") if row["portfolio_mode"] == "long_only" else scenario.get("max_drawdown"),
            "expression": row["expression"],
        })
    cluster_rows: list[dict] = []
    for cluster_index, cluster in enumerate(clusters):
        member_indices = list(cluster["members"])
        representative_index = int(cluster["representative_index"])
        representative = rows[representative_index]
        source_lists = sorted({
            member["list_id"]
            for index in member_indices
            for member in memberships_by_task.get(rows[index]["task_id"], [])
        })
        cluster_rows.append({
            "scope": name,
            "cluster": f"{prefix}{cluster_index + 1:04d}",
            "size": len(member_indices),
            "representative_task_id": representative["task_id"],
            "representative_mode": representative["portfolio_mode"],
            "representative_direction": representative["direction"],
            "representative_mechanism": representative["mechanism_family"],
            "source_lists": "|".join(source_lists),
            "crosses_internal_external": any(value.startswith("内部") for value in source_lists) and any(value.startswith("外部") for value in source_lists),
            "crosses_portfolio_modes": len({rows[index]["portfolio_mode"] for index in member_indices}) > 1,
            "mean_abs_correlation_to_representative": round(float(np.mean(np.abs(correlations[member_indices, representative_index]))), 6),
            "representative_expression": representative["expression"],
        })
    assignment_rows.sort(key=lambda row: (row["cluster"], not row["is_representative"], -float(row["sharpe"] or 0.0)))
    cluster_rows.sort(key=lambda row: (-int(row["size"]), row["cluster"]))
    _write_csv(output / f"{name}_assignments.csv", assignment_rows)
    _write_csv(output / f"{name}_clusters.csv", cluster_rows)
    sizes = [len(cluster["members"]) for cluster in clusters]
    cluster_hhi = sum((size / len(rows)) ** 2 for size in sizes)
    return {
        "scope": name,
        "absolute_correlation": absolute,
        "rows": len(rows),
        "clusters": len(clusters),
        "redundancy_ratio": round(1.0 - len(clusters) / len(rows), 6),
        "largest_cluster": max(sizes),
        "cluster_hhi": round(cluster_hhi, 6),
        "effective_return_sources": round(1.0 / cluster_hhi, 6),
        "singleton_clusters": sum(size == 1 for size in sizes),
        "recurring_clusters_ge3": sum(size >= 3 for size in sizes),
        "members_in_clusters_ge3": sum(size for size in sizes if size >= 3),
        "common_periods": common_periods,
        "threshold": threshold,
        "cross_internal_external_clusters": sum(bool(row["crosses_internal_external"]) for row in cluster_rows),
        "cross_portfolio_mode_clusters": sum(bool(row["crosses_portfolio_modes"]) for row in cluster_rows),
        "mechanism_counts": dict(Counter(row["mechanism_family"] for row in rows)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("var/reports"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--correlation-threshold", type=float, default=0.80)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.threads_per_worker <= 8:
        raise SystemExit("workers必须在1..4，threads-per-worker必须在1..8")
    if not 0.50 <= args.correlation_threshold < 1.0:
        raise SystemExit("correlation-threshold必须在[0.50, 1.0)")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    reports = {list_id: (args.reports_root / name).resolve() for list_id, name in DEFAULT_REPORTS.items()}
    memberships = [member for list_id, report in reports.items() for member in _load_top100(list_id, report)]
    panel_protocols = [
        json.loads((reports[list_id] / "protocol.json").read_text(encoding="utf-8"))
        for list_id in PANEL_PROTOCOL_LIST_IDS
    ]
    panel_globs = {str(protocol["panel_glob"]) for protocol in panel_protocols}
    panel_identities = {str(protocol["panel_identity_path_size_mtime_sha256"]) for protocol in panel_protocols}
    dsl_field_sets = {tuple(protocol["dsl_fields"]) for protocol in panel_protocols}
    if len(panel_globs) != 1 or len(panel_identities) != 1 or len(dsl_field_sets) != 1:
        raise RuntimeError("用于统一回放的榜单数据面板身份不一致")
    panel_glob = next(iter(panel_globs))
    panel_identity = next(iter(panel_identities))
    dsl_fields = list(next(iter(dsl_field_sets)))
    tasks = _build_tasks(memberships, panel_glob, dsl_fields)

    source_identity = {
        list_id: {
            "report": str(report),
            "leaderboard_sha256": _sha256(report / "leaderboard_full.csv"),
            "manifest_sha256": _sha256(report / "manifest.json"),
            "selected_rows": TOP_N,
            "portfolio_mode": MODE_BY_LIST[list_id],
            "membership_filter": LIST_FILTERS.get(list_id, {}),
        }
        for list_id, report in reports.items()
    }
    identity = {
        "protocol": PROTOCOL,
        "policy_label": POLICY_LABEL,
        "common_window": {"start": COMMON_START, "end": COMMON_END},
        "top_n_per_list": TOP_N,
        "sources": source_identity,
        "memberships": len(memberships),
        "replay_tasks": len(tasks),
        "panel_glob": panel_glob,
        "panel_identity": panel_identity,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "correlation_threshold": args.correlation_threshold,
        "script_sha256": _sha256(RUN_SCRIPT_PATH),
    }
    identity["run_identity_sha256"] = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing.get("run_identity_sha256") != identity["run_identity_sha256"]:
            raise SystemExit("输出目录已有不同参数，拒绝混合续算")
    else:
        _write_json_atomic(protocol_path, identity)
        _write_csv(output / "frozen_top100_memberships.csv", memberships)

    results_path = output / "return_paths.jsonl"
    results = _load_results(results_path)
    pending = [
        task for task in tasks
        if task["task_id"] not in results
        or results[task["task_id"]].get("status") != "ok"
    ]
    started = time.time()
    _write_json_atomic(output / "progress.json", _progress(results, tasks, started, "return_path_replay"))
    os.environ["POLARS_MAX_THREADS"] = str(args.threads_per_worker)
    print(
        f"{len(pending)}/{len(tasks)} fixed-direction mode paths pending · "
        f"{args.workers} workers × {args.threads_per_worker} Polars threads",
        flush=True,
    )
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(_worker, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            _append_jsonl(results_path, row)
            results[row["task_id"]] = row
            progress = _progress(results, tasks, started, "return_path_replay")
            _write_json_atomic(output / "progress.json", progress)
            if index == 1 or index % 10 == 0 or index == len(pending):
                print(
                    f"[{progress['completed']}/{progress['total']} {progress['percent']:.1f}%] "
                    f"{row['task_id']} {row['status']} {row.get('runtime_seconds', 0):.2f}s",
                    flush=True,
                )

    _write_json_atomic(output / "progress.json", _progress(results, tasks, started, "clustering"))
    valid = [results[task["task_id"]] for task in tasks if results[task["task_id"]].get("status") == "ok"]
    failures = [results[task["task_id"]] for task in tasks if results[task["task_id"]].get("status") != "ok"]
    memberships_by_task: dict[str, list[dict]] = defaultdict(list)
    for member in memberships:
        memberships_by_task[member["task_id"]].append(member)
    scopes = {
        name: rows
        for name, rows in {
            "long_short_directed": [row for row in valid if row["portfolio_mode"] == "long_short"],
            "long_only_directed": [row for row in valid if row["portfolio_mode"] == "long_only"],
            "global_directed": valid,
        }.items()
        if rows
    }
    summaries = {
        name: _cluster_artifacts(
            name,
            rows,
            memberships_by_task,
            output,
            args.correlation_threshold,
            absolute=False,
        )
        for name, rows in scopes.items()
    }
    summaries["global_unsigned"] = _cluster_artifacts(
        "global_unsigned",
        valid,
        memberships_by_task,
        output,
        args.correlation_threshold,
        absolute=True,
    )
    sensitivity_rows: list[dict[str, Any]] = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        for scope, scope_rows in {
            "long_short": [row for row in valid if row["portfolio_mode"] == "long_short"],
            "long_only": [row for row in valid if row["portfolio_mode"] == "long_only"],
            "global": valid,
        }.items():
            if not scope_rows:
                continue
            clusters, _, _, common_periods = _cluster(
                scope_rows,
                threshold,
                absolute=True,
            )
            sizes = [len(cluster["members"]) for cluster in clusters]
            hhi = sum((size / len(scope_rows)) ** 2 for size in sizes)
            sensitivity_rows.append({
                "threshold": threshold,
                "scope": scope,
                "rows": len(scope_rows),
                "clusters": len(clusters),
                "effective_return_sources": round(1.0 / hhi, 6),
                "singleton_clusters": sum(size == 1 for size in sizes),
                "recurring_clusters_ge3": sum(size >= 3 for size in sizes),
                "largest_cluster": max(sizes),
                "common_periods": common_periods,
            })
    _write_csv(output / "threshold_sensitivity.csv", sensitivity_rows)

    raw_formula_directions: dict[str, set[int]] = defaultdict(set)
    oriented_formulas: set[str] = set()
    for member in memberships:
        raw_formula_directions[member["canonical_expression_hash"]].add(member["direction"])
        oriented_formulas.add(member["canonical_oriented_hash"])
    direction_conflicts = {
        key: sorted(values) for key, values in raw_formula_directions.items() if len(values) > 1
    }
    list_counts = {}
    unsigned_assignments = list(csv.DictReader((output / "global_unsigned_assignments.csv").open(encoding="utf-8")))
    task_to_unsigned = {row["task_id"]: row["cluster"] for row in unsigned_assignments}
    directed_assignments = list(csv.DictReader((output / "global_directed_assignments.csv").open(encoding="utf-8")))
    task_to_directed = {row["task_id"]: row["cluster"] for row in directed_assignments}
    for list_id in DEFAULT_REPORTS:
        list_tasks = {member["task_id"] for member in memberships if member["list_id"] == list_id}
        list_counts[list_id] = {
            "ranked_slots": TOP_N,
            "directed_return_clusters_touched": len({task_to_directed[task] for task in list_tasks if task in task_to_directed}),
            "unsigned_economic_clusters_touched": len({task_to_unsigned[task] for task in list_tasks if task in task_to_unsigned}),
            "positive_directions": sum(member["direction"] > 0 for member in memberships if member["list_id"] == list_id),
            "negative_directions": sum(member["direction"] < 0 for member in memberships if member["list_id"] == list_id),
        }
    summary = {
        "protocol": PROTOCOL,
        "policy_label": POLICY_LABEL,
        "status": "complete" if not failures else "complete_with_failures",
        "ranked_slots": len(memberships),
        "unique_raw_formulas": len(raw_formula_directions),
        "unique_oriented_formulas": len(oriented_formulas),
        "mode_specific_replay_tasks": len(tasks),
        "successful_replay_tasks": len(valid),
        "failures": len(failures),
        "direction_conflicts": len(direction_conflicts),
        "direction_conflict_detail": direction_conflicts,
        "list_counts": list_counts,
        "cluster_summaries": summaries,
        "threshold_sensitivity": sensitivity_rows,
        "interpretation": {
            "directed": "正相关>=阈值才合并，方向冻结后的负相关流视为不同可交易收益流。",
            "unsigned": "绝对相关>=阈值合并，避免同一经济来源的符号翻转被重复计数。",
            "answer_field": "cluster_summaries.global_unsigned.clusters",
        },
    }
    _write_json_atomic(output / "summary.json", summary)
    _write_json_atomic(output / "failures.json", failures)
    _write_json_atomic(output / "progress.json", _progress(results, tasks, started, "complete"))

    directed = summaries["global_directed"]
    unsigned = summaries["global_unsigned"]
    lines = [
        f"# {REPORT_TITLE}",
        "",
        f"- 四榜榜位：{len(memberships)}",
        f"- 唯一原始公式：{len(raw_formula_directions)}",
        f"- 唯一公式方向：{len(oriented_formulas)}",
        f"- 模式化收益路径：{len(tasks)}",
        f"- 有向收益簇（相关性≥{args.correlation_threshold:.2f}）：{directed['clusters']}",
        f"- 无符号经济收益簇（|相关性|≥{args.correlation_threshold:.2f}）：**{unsigned['clusters']}**",
        f"- 集中度折算的有效独立来源：**{unsigned['effective_return_sources']:.2f}**",
        f"- 至少出现3次的重复来源簇：{unsigned['recurring_clusters_ge3']}（覆盖{unsigned['members_in_clusters_ge3']}/{len(memberships)}）",
        f"- 内部与外部共同出现的来源簇：{unsigned['cross_internal_external_clusters']}",
        "",
        "> 主答案采用无符号经济收益簇，防止同一公式或机制仅因方向翻转而被算成两个独立来源。",
        "> 方向已经严格冻结自各自榜单，没有在统一窗口重新择向。纯多使用基准主动收益；如存在多空榜，则使用扣除15 BPS滑点、佣金和借券费后的净收益。",
        "",
        "## 分榜结果",
        "",
        "| 榜单 | +方向 | -方向 | 涉及有向簇 | 涉及无符号经济簇 |",
        "|---|---:|---:|---:|---:|",
    ]
    for list_id, counts in list_counts.items():
        lines.append(
            f"| {list_id} | {counts['positive_directions']} | {counts['negative_directions']} | "
            f"{counts['directed_return_clusters_touched']} | {counts['unsigned_economic_clusters_touched']} |"
        )
    lines.extend([
        "",
        "## 口径边界",
        "",
        f"- 所有400个榜位统一重放于 {COMMON_START}～{COMMON_END}，每5个交易日一个收益期。",
        "- 这是向量化收益路径审计，适合去重；不是400个因子的逐笔步进正式复测。",
        "- 统一窗口已经参与部分榜单筛选，因此只回答冗余度和来源结构，不新增正式入库证据。",
        "- `global_unsigned_clusters.csv` 给出最终经济来源代表；`global_unsigned_assignments.csv` 给出全部400个榜位对应关系。",
        "",
        "## 产物",
        "",
        "- `frozen_top100_memberships.csv`：四榜各100名及冻结方向。",
        "- `return_paths.jsonl`：统一窗口、统一成本下的模式化收益路径。",
        "- `*_clusters.csv`：多空、纯多、全局有向、全局无符号聚类。",
        "- `*_assignments.csv`：每条模式化路径到代表簇的映射及相关系数。",
        "- `threshold_sensitivity.csv`：0.70～0.95阈值下的簇数与有效来源数。",
        "- `protocol.json`：四榜及数据面板哈希。",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
