"""Resumable full-library return-source audit.

Each frozen expression is materialised once, both orientations are screened at
15 BPS, and the better orientation is selected on the explicitly diagnostic
2020-to-latest window.  Long-only factors are compared by active return;
long-short factors are compared by fee/borrow-adjusted net return.  The output
clusters return-path signatures and never overwrites an existing leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
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

from app.backtest.batch import BatchBacktestSpec  # noqa: E402
from app.backtest.engine import _prepare_backtest_frame  # noqa: E402
from app.backtest.vector import run_vector_cost_scenarios  # noqa: E402
from app.factors.diversity import infer_mechanism  # noqa: E402
from app.factors.return_path import build_return_path_signature  # noqa: E402
from app.factors.semantics import audit_expression_semantics  # noqa: E402


PROTOCOL = "full_library_return_source_audit_v1"
POLICY_LABEL = "NON_PIT_FULL_WINDOW_DIVERSITY_DIAGNOSTIC"


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
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _load_latest(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["task_id"])] = row
    return rows


def _spec(report: Path) -> tuple[BatchBacktestSpec, str, dict]:
    protocol = json.loads((report / "protocol.json").read_text(encoding="utf-8"))
    raw = dict(protocol.get("spec") or {})
    market = str(protocol.get("target_market") or raw.get("market"))
    mode = str(protocol.get("portfolio_mode") or raw.get("mode"))
    spec = BatchBacktestSpec(**{
        **raw,
        "market": market,
        "mode": mode,
        "slippage_bps": (15.0,),
    })
    return spec, str(protocol["panel_glob"]), protocol


def _quality(scenario: dict, mode: str) -> tuple[float, float, float]:
    if mode == "long_only":
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
    task_id = str(task["task_id"])
    try:
        spec = BatchBacktestSpec(**{
            **task["spec"],
            "slippage_bps": (15.0,),
        })
        expression = str(task["expression"])
        frame, _ = _prepare_backtest_frame(
            expression=expression,
            universe_n=spec.universe_n,
            start=spec.holdout_start,
            end=spec.holdout_end,
            panel_glob=str(task["panel_glob"]),
            market=spec.market,
            forward_horizon=spec.horizon,
        )
        orientations = []
        for direction in (1, -1):
            scenario = run_vector_cost_scenarios(
                frame,
                direction=direction,
                spec=replace(spec, slippage_bps=(15.0,)),
                capture_periods=True,
            )["15"]
            periods = list(scenario.pop("period_rows"))
            return_key = "active_return" if spec.mode == "long_only" else "net_return"
            returns = [float(row[return_key]) for row in periods]
            orientations.append({
                "direction": direction,
                "quality": _quality(scenario, spec.mode),
                "scenario": scenario,
                "return_dates": [str(row["signal_date"]) for row in periods],
                "returns": returns,
                "return_path_signature": build_return_path_signature(returns),
            })
        selected = max(orientations, key=lambda row: row["quality"])
        semantic = audit_expression_semantics(expression, spec.market)
        return {
            "protocol": PROTOCOL,
            "policy_label": POLICY_LABEL,
            "task_id": task_id,
            "report_id": task["report_id"],
            "source_expression_hash": task["source_expression_hash"],
            "expression": expression,
            "status": "ok",
            "market": spec.market,
            "portfolio_mode": spec.mode,
            "direction": selected["direction"],
            "direction_selection": "best_full_window_15bps_diagnostic",
            "return_basis": "active" if spec.mode == "long_only" else "net",
            "scenario": selected["scenario"],
            "return_dates": selected["return_dates"],
            "returns": selected["returns"],
            "return_path_signature": selected["return_path_signature"],
            "orientation_summaries": [
                {
                    "direction": row["direction"],
                    "quality": row["quality"],
                    "scenario": row["scenario"],
                    "return_path_fingerprint": row["return_path_signature"]["fingerprint"],
                }
                for row in orientations
            ],
            "mechanism_family": infer_mechanism(expression),
            "semantic_audit": semantic,
            "worker_pid": os.getpid(),
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "protocol": PROTOCOL,
            "policy_label": POLICY_LABEL,
            "task_id": task_id,
            "report_id": task["report_id"],
            "source_expression_hash": task["source_expression_hash"],
            "expression": task["expression"],
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "traceback_tail": traceback.format_exc(limit=5)[-6000:],
            "worker_pid": os.getpid(),
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }


def _cluster_report(rows: list[dict], output: Path, threshold: float) -> dict:
    valid = [row for row in rows if row.get("status") == "ok"]
    if not valid:
        return {"status": "no_valid_rows", "rows": 0}
    vectors = np.asarray(
        [row["return_path_signature"]["vector"] for row in valid],
        dtype=np.float64,
    )
    correlations = (
        np.ones((1, 1), dtype=np.float64)
        if len(valid) == 1
        else np.nan_to_num(np.corrcoef(vectors), nan=0.0)
    )
    quality_order = sorted(
        range(len(valid)),
        key=lambda index: _quality(valid[index]["scenario"], valid[index]["portfolio_mode"]),
        reverse=True,
    )
    clusters: list[dict] = []
    assignment: dict[int, int] = {}
    for index in quality_order:
        matched = None
        for cluster_index, cluster in enumerate(clusters):
            representative = int(cluster["representative_index"])
            if float(correlations[index, representative]) >= threshold:
                matched = cluster_index
                break
        if matched is None:
            matched = len(clusters)
            clusters.append({"representative_index": index, "members": []})
        clusters[matched]["members"].append(index)
        assignment[index] = matched

    audit_rows = []
    for index, row in enumerate(valid):
        cluster_index = assignment[index]
        cluster = clusters[cluster_index]
        representative_index = int(cluster["representative_index"])
        scenario = row["scenario"]
        audit_rows.append({
            "return_cluster": f"R{cluster_index + 1:04d}",
            "is_cluster_representative": index == representative_index,
            "correlation_to_representative": round(float(correlations[index, representative_index]), 6),
            "source_expression_hash": row["source_expression_hash"],
            "direction": row["direction"],
            "mechanism_family": row["mechanism_family"],
            "semantic_status": row["semantic_audit"]["status"],
            "return_basis": row["return_basis"],
            "ann_return": scenario.get("active_ann_return") if row["portfolio_mode"] == "long_only" else scenario.get("ann_return"),
            "sharpe": scenario.get("active_sharpe") if row["portfolio_mode"] == "long_only" else scenario.get("sharpe"),
            "max_drawdown": scenario.get("active_max_drawdown") if row["portfolio_mode"] == "long_only" else scenario.get("max_drawdown"),
            "expression": row["expression"],
        })
    audit_rows.sort(
        key=lambda row: (
            not row["is_cluster_representative"],
            -float(row["sharpe"] or 0.0),
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    columns = list(audit_rows[0])
    with (output / "return_source_clusters.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(audit_rows)
    cluster_rows = []
    for cluster_index, cluster in enumerate(clusters):
        members = list(cluster["members"])
        representative = valid[int(cluster["representative_index"])]
        cluster_rows.append({
            "cluster": f"R{cluster_index + 1:04d}",
            "size": len(members),
            "representative_hash": representative["source_expression_hash"],
            "representative_expression": representative["expression"],
            "representative_mechanism": representative["mechanism_family"],
            "mean_correlation_to_representative": round(
                float(np.mean(correlations[members, int(cluster["representative_index"])])),
                6,
            ),
            "mechanism_counts": dict(Counter(valid[index]["mechanism_family"] for index in members)),
        })
    cluster_rows.sort(key=lambda row: (-row["size"], row["cluster"]))
    _write_json_atomic(output / "clusters.json", cluster_rows)
    family_counts = Counter(row["mechanism_family"] for row in valid)
    family_cluster_counts = defaultdict(set)
    for row in audit_rows:
        family_cluster_counts[row["mechanism_family"]].add(row["return_cluster"])
    sizes = [len(cluster["members"]) for cluster in clusters]
    total = len(valid)
    summary = {
        "protocol": PROTOCOL,
        "policy_label": POLICY_LABEL,
        "status": "complete",
        "rows": total,
        "return_clusters": len(clusters),
        "return_path_redundancy_ratio": round(1.0 - len(clusters) / total, 6),
        "return_cluster_hhi": round(sum((size / total) ** 2 for size in sizes), 6),
        "largest_cluster": max(sizes),
        "correlation_threshold": threshold,
        "return_basis": valid[0]["return_basis"],
        "mechanism_counts": dict(family_counts),
        "mechanism_return_cluster_counts": {
            family: len(groups) for family, groups in sorted(family_cluster_counts.items())
        },
        "semantic_error_count": sum(row["semantic_audit"]["status"] == "error" for row in valid),
        "failures": len(rows) - len(valid),
    }
    _write_json_atomic(output / "summary.json", summary)
    return summary


def _progress(rows: dict[str, dict], tasks: list[dict], started: float, phase: str) -> dict:
    totals = Counter(task["report_id"] for task in tasks)
    completed = Counter(row["report_id"] for row in rows.values())
    return {
        "protocol": PROTOCOL,
        "phase": phase,
        "completed": len(rows),
        "total": len(tasks),
        "percent": round(100.0 * len(rows) / max(1, len(tasks)), 3),
        "reports": {
            report_id: {
                "completed": completed[report_id],
                "total": total,
                "percent": round(100.0 * completed[report_id] / max(1, total), 3),
            }
            for report_id, total in totals.items()
        },
        "elapsed_seconds": round(time.time() - started, 1),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads-per-worker", type=int, default=2)
    parser.add_argument("--correlation-threshold", type=float, default=0.80)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4 or not 1 <= args.threads_per_worker <= 8:
        raise SystemExit("workers 必须在 1..4，threads-per-worker 必须在 1..8")
    if not 0.5 <= args.correlation_threshold < 1.0:
        raise SystemExit("correlation-threshold 必须在 [0.5, 1.0)")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    tasks = []
    source_identity = []
    for report_arg in args.source_report:
        report = report_arg.resolve()
        snapshot_path = report / "snapshot.json"
        if not snapshot_path.exists():
            raise SystemExit(f"缺少冻结 snapshot.json: {report}")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        spec, panel_glob, protocol = _spec(report)
        report_id = report.name
        expressions = list(snapshot.get("expressions") or [])
        if args.limit > 0:
            expressions = expressions[: args.limit]
        for record in expressions:
            source_hash = str(record["expression_hash"])
            tasks.append({
                "task_id": f"{report_id}:{source_hash}",
                "report_id": report_id,
                "source_expression_hash": source_hash,
                "expression": record["expression"],
                "spec": spec.as_dict(),
                "panel_glob": panel_glob,
            })
        source_identity.append({
            "report": str(report),
            "snapshot_sha256": _sha256(snapshot_path),
            "market": spec.market,
            "portfolio_mode": spec.mode,
            "panel_identity": protocol.get("panel_identity_path_size_mtime_sha256"),
            "expressions": len(expressions),
        })
    identity = {
        "protocol": PROTOCOL,
        "policy_label": POLICY_LABEL,
        "sources": source_identity,
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "allocated_compute_threads": args.workers * args.threads_per_worker,
        "correlation_threshold": args.correlation_threshold,
        "script_sha256": _sha256(Path(__file__).resolve()),
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

    results_path = output / "return_paths.jsonl"
    rows = _load_latest(results_path)
    pending = [task for task in tasks if task["task_id"] not in rows]
    started = time.time()
    _write_json_atomic(output / "progress.json", _progress(rows, tasks, started, "return_path_replay"))
    os.environ["POLARS_MAX_THREADS"] = str(args.threads_per_worker)
    print(
        f"{len(pending)}/{len(tasks)} factors pending · "
        f"{args.workers} workers × {args.threads_per_worker} Polars threads",
        flush=True,
    )
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        futures = {pool.submit(_worker, task): task for task in pending}
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            _append_jsonl(results_path, row)
            rows[row["task_id"]] = row
            progress = _progress(rows, tasks, started, "return_path_replay")
            _write_json_atomic(output / "progress.json", progress)
            if index == 1 or index % 10 == 0 or index == len(pending):
                print(
                    f"[{progress['completed']}/{progress['total']} "
                    f"{progress['percent']:.1f}%] {row['task_id']} "
                    f"{row['status']} {row.get('runtime_seconds', 0):.2f}s",
                    flush=True,
                )
    _write_json_atomic(output / "progress.json", _progress(rows, tasks, started, "clustering"))
    summaries = {}
    for source in source_identity:
        report_id = Path(source["report"]).name
        report_rows = [row for row in rows.values() if row["report_id"] == report_id]
        summaries[report_id] = _cluster_report(
            report_rows,
            output / report_id,
            args.correlation_threshold,
        )
    _write_json_atomic(output / "summary.json", {
        "protocol": PROTOCOL,
        "policy_label": POLICY_LABEL,
        "reports": summaries,
    })
    _write_json_atomic(output / "progress.json", _progress(rows, tasks, started, "complete"))
    print(json.dumps(summaries, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
