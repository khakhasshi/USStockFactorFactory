#!/usr/bin/env python3
"""Evaluate the complete Qlib Alpha158 catalog on FactorFactory panels.

Stage 1 evaluates every feature with training-safe two-sided direction
selection. Stage 2 performs the expensive HOLDOUT/Vault/frozen-rating audit on
the training-ranked finalists only. Results are checkpointed after every
factor, so interruption never destroys completed evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from app.eval.harness import evaluate, evaluate_full  # noqa: E402
from app.qlib_native import (  # noqa: E402
    ALPHA158_ARTIFACT_ROOT,
    ALPHA158_FEATURES,
    QlibDatasetSpec,
    QlibNativeRecorder,
    alpha158_catalog,
)


def _safe(value, default=None):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _mode_key(market: str, portfolio_mode: str) -> str:
    return f"{market}-{portfolio_mode}"


def _reject_low_fidelity_us_vwap(row: dict) -> dict:
    if row.get("name") != "VWAP0" or row.get("quality_rejection"):
        return row
    original = {
        key: row.get(key)
        for key in ("direction", "learning_score", "gate_score", "research_pass")
    }
    row.update({
        "direction": None,
        "learning_score": None,
        "gate_score": None,
        "research_pass": False,
        "effective_metrics": {},
        "public": {},
        "gate": {},
        "failure_reasons": ["美股面板缺少独立真实 VWAP"],
        "full_audit": None,
        "error": (
            "INSUFFICIENT_DATA: US amount is predominantly a "
            "close_times_volume_proxy; VWAP0 is near-constant and non-identifiable"
        ),
        "quality_rejection": {
            "code": "LOW_FIDELITY_VWAP_PROXY",
            "original_evaluation_preserved": original,
            "decision": "fail_closed_before_ranking",
        },
    })
    return row


def _discovery_row(feature, result: dict, elapsed: float) -> dict:
    discovery = result.get("discovery") or {}
    effective = discovery.get("effective_metrics") or {}
    gate = result.get("gate") or {}
    public = result.get("public") or {}
    return {
        "name": feature.name,
        "family": feature.family,
        "window": feature.window,
        "expression": feature.expression,
        "direction": result.get("direction"),
        "learning_score": discovery.get("learning_score"),
        "gate_score": discovery.get("gate_score"),
        "research_pass": bool(discovery.get("passed")),
        "effective_metrics": effective,
        "public": {
            key: public.get(key)
            for key in (
                "rank_ic_mean", "icir", "daily_turnover", "era_consistency",
                "profitable_era_rate", "monotonicity", "cost_cushion_multiple",
            )
        },
        "gate": {
            key: gate.get(key)
            for key in (
                "rank_ic_mean", "icir", "daily_turnover", "era_consistency",
                "profitable_era_rate", "monotonicity", "cost_cushion_multiple",
            )
        },
        "failure_reasons": list(discovery.get("failure_reasons") or []),
        "runtime_seconds": round(elapsed, 3),
        "full_audit": None,
        "error": None,
    }


def _full_audit_summary(result: dict) -> dict:
    rating = result.get("rating") or {}
    eligibility = result.get("eligibility") or {}
    ranking = result.get("ranking") or {}
    relevant = (
        rating.get("active")
        if result.get("portfolio_mode") == "long_only"
        else rating.get("net")
    ) or {}
    return {
        "grade": eligibility.get("grade"),
        "stage": eligibility.get("stage"),
        "research_pass": eligibility.get("research_pass"),
        "holdout_pass": eligibility.get("holdout_pass"),
        "vault_pass": eligibility.get("vault_pass"),
        "capacity_pass": eligibility.get("capacity_pass"),
        "rating_score": ranking.get("score"),
        "rating_rank_ic": rating.get("rank_ic_mean"),
        "rating_icir": rating.get("icir"),
        "rating_ann_return": relevant.get("ann_return"),
        "rating_sharpe": relevant.get("sharpe"),
        "rating_max_drawdown": relevant.get("max_drawdown"),
        "rating_daily_turnover": rating.get("daily_turnover"),
        "rating_era_consistency": rating.get("era_consistency"),
        "rating_monotonicity": rating.get("monotonicity"),
        "failure_reasons": list(eligibility.get("failure_reasons") or []),
        "runtime": result.get("runtime"),
    }


def _family_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("error"):
            grouped[row["family"]].append(row)
    result = []
    for family, members in sorted(grouped.items()):
        gate_scores = [_safe(row.get("gate_score"), 0.0) or 0.0 for row in members]
        learning_scores = [_safe(row.get("learning_score"), 0.0) or 0.0 for row in members]
        result.append({
            "family": family,
            "count": len(members),
            "mean_gate_score": round(sum(gate_scores) / len(gate_scores), 6),
            "best_gate_score": round(max(gate_scores), 6),
            "mean_learning_score": round(sum(learning_scores) / len(learning_scores), 6),
            "research_pass_count": sum(bool(row.get("research_pass")) for row in members),
        })
    return sorted(result, key=lambda row: (-row["best_gate_score"], row["family"]))


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = [
        "name", "family", "window", "direction", "learning_score", "gate_score",
        "research_pass", "grade", "rating_score", "rating_rank_ic", "rating_icir",
        "rating_ann_return", "rating_sharpe", "rating_max_drawdown",
        "rating_daily_turnover", "error", "expression",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            full = row.get("full_audit") or {}
            writer.writerow({
                "name": row.get("name"),
                "family": row.get("family"),
                "window": row.get("window"),
                "direction": row.get("direction"),
                "learning_score": row.get("learning_score"),
                "gate_score": row.get("gate_score"),
                "research_pass": row.get("research_pass"),
                "grade": full.get("grade"),
                "rating_score": full.get("rating_score"),
                "rating_rank_ic": full.get("rating_rank_ic"),
                "rating_icir": full.get("rating_icir"),
                "rating_ann_return": full.get("rating_ann_return"),
                "rating_sharpe": full.get("rating_sharpe"),
                "rating_max_drawdown": full.get("rating_max_drawdown"),
                "rating_daily_turnover": full.get("rating_daily_turnover"),
                "error": row.get("error"),
                "expression": row.get("expression"),
            })


def _html(summary: dict) -> str:
    def optional_number(value: Any, digits: int) -> str:
        if value is None or value == "":
            return "—"
        number = _safe(value, None)
        return f"{number:.{digits}f}" if number is not None else "—"

    rows = summary.get("rows") or []
    body_rows = []
    for index, row in enumerate(rows, 1):
        full = row.get("full_audit") or {}
        body_rows.append(
            "<tr>"
            f"<td>{index}</td><td>{row['name']}</td><td>{row['family']}</td>"
            f"<td>{row.get('direction') or '—'}</td>"
            f"<td>{_safe(row.get('gate_score'), 0):.4f}</td>"
            f"<td>{_safe(row.get('learning_score'), 0):.4f}</td>"
            f"<td>{full.get('grade') or '未做完整审计'}</td>"
            f"<td>{optional_number(full.get('rating_rank_ic'), 4)}</td>"
            f"<td>{optional_number(full.get('rating_sharpe'), 2)}</td>"
            f"<td><code>{row['expression']}</code></td></tr>"
        )
    return f"""<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<title>Alpha158 {summary['market']} {summary['portfolio_mode']}</title>
<style>body{{font:14px system-ui;background:#10151d;color:#d9e2ee;margin:32px}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #2a3442;padding:8px;text-align:left}}th{{position:sticky;top:0;background:#18202b}}code{{font-size:11px;color:#9fd3ff}}.warn{{padding:14px;background:#3b2b12;border:1px solid #805d1b}}</style></head><body>
<h1>Qlib Alpha158 · {summary['market']} · {summary['portfolio_mode']}</h1>
<p class=\"warn\">NON_PIT_RESEARCH。方向仅由训练安全层选择；完整审计仅覆盖训练排名前 {summary['protocol']['full_audit_top']}。未做完整审计不等于通过。</p>
<p>特征 {summary['completed']}/158 · 失败 {summary['failed']} · 正式通过 {summary['full_passed']} · 耗时 {summary['elapsed_seconds']} 秒</p>
<table><thead><tr><th>#</th><th>因子</th><th>家族</th><th>方向</th><th>Gate</th><th>学习分</th><th>等级</th><th>评级RankIC</th><th>评级Sharpe</th><th>DSL</th></tr></thead><tbody>{''.join(body_rows)}</tbody></table>
</body></html>"""


def run_one(
    market: str,
    portfolio_mode: str,
    *,
    universe_n: int,
    horizon: int,
    full_audit_top: int,
    run_dir: Path,
    global_progress: Path,
) -> dict:
    started = time.time()
    mode_key = _mode_key(market, portfolio_mode)
    checkpoint = run_dir / f"{mode_key}-checkpoint.json"
    rows: list[dict] = []
    if checkpoint.exists():
        rows = json.loads(checkpoint.read_text(encoding="utf-8")).get("rows") or []
    if market == "us":
        rows = [_reject_low_fidelity_us_vwap(row) for row in rows]
    completed_names = {row["name"] for row in rows}
    by_name = {feature.name: feature for feature in ALPHA158_FEATURES}
    for feature in ALPHA158_FEATURES:
        if feature.name in completed_names:
            continue
        factor_started = time.time()
        if market == "us" and feature.name == "VWAP0":
            rows.append(_reject_low_fidelity_us_vwap({
                "name": feature.name,
                "family": feature.family,
                "window": feature.window,
                "expression": feature.expression,
                "direction": None,
                "learning_score": None,
                "gate_score": None,
                "research_pass": False,
                "effective_metrics": {},
                "public": {},
                "gate": {},
                "failure_reasons": [],
                "runtime_seconds": 0.0,
                "full_audit": None,
                "error": None,
            }))
            _atomic_json(checkpoint, {"rows": rows})
            continue
        try:
            metrics = evaluate(
                feature.expression,
                universe_n,
                horizon,
                portfolio_mode,
                1,
                None,
                None,
                market,
                None,
                "both_train_select",
            )
            row = _discovery_row(feature, metrics, time.time() - factor_started)
        except Exception as exc:  # noqa: BLE001
            row = {
                "name": feature.name,
                "family": feature.family,
                "window": feature.window,
                "expression": feature.expression,
                "direction": None,
                "learning_score": None,
                "gate_score": None,
                "research_pass": False,
                "effective_metrics": {},
                "public": {},
                "gate": {},
                "failure_reasons": [],
                "runtime_seconds": round(time.time() - factor_started, 3),
                "full_audit": None,
                "error": f"{type(exc).__name__}: {exc}"[:1200],
            }
        rows.append(row)
        progress = {
            "schema": "factorfactory.qlib-alpha158-progress/v1",
            "run_id": run_dir.name,
            "state": "discovery",
            "market": market,
            "portfolio_mode": portfolio_mode,
            "completed": len(rows),
            "total": 158,
            "current": feature.name,
            "failed": sum(bool(item.get("error")) for item in rows),
            "elapsed_seconds": round(time.time() - started, 1),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(checkpoint, {**progress, "rows": rows})
        _atomic_json(global_progress, progress)
        print(
            f"[{mode_key}] {len(rows):03d}/158 {feature.name:<8} "
            f"gate={_safe(row.get('gate_score'), 0):.4f} "
            f"learning={_safe(row.get('learning_score'), 0):.4f} "
            f"{row.get('runtime_seconds')}s",
            flush=True,
        )

    ranked = sorted(
        [row for row in rows if not row.get("error")],
        key=lambda row: (
            not bool(row.get("research_pass")),
            -(_safe(row.get("gate_score"), 0.0) or 0.0),
            -(_safe(row.get("learning_score"), 0.0) or 0.0),
            row["name"],
        ),
    )
    finalists = ranked[: max(0, min(int(full_audit_top), len(ranked)))]
    for finalist_index, row in enumerate(finalists, 1):
        if row.get("full_audit"):
            continue
        feature = by_name[row["name"]]
        progress = {
            "schema": "factorfactory.qlib-alpha158-progress/v1",
            "run_id": run_dir.name,
            "state": "full_audit",
            "market": market,
            "portfolio_mode": portfolio_mode,
            "completed": finalist_index - 1,
            "total": len(finalists),
            "current": feature.name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        _atomic_json(global_progress, progress)
        try:
            full = evaluate_full(
                feature.expression,
                universe_n,
                horizon,
                portfolio_mode,
                int(row.get("direction") or 1),
                None,
                None,
                market,
                None,
                "both_train_select",
            )
            row["full_audit"] = _full_audit_summary(full)
        except Exception as exc:  # noqa: BLE001
            row["full_audit"] = {"error": f"{type(exc).__name__}: {exc}"[:1200]}
        _atomic_json(checkpoint, {**progress, "rows": rows})
        print(
            f"[{mode_key}] FULL {finalist_index:02d}/{len(finalists):02d} "
            f"{feature.name} {row['full_audit'].get('grade', 'ERROR')}",
            flush=True,
        )

    ordered = sorted(
        rows,
        key=lambda row: (
            not bool((row.get("full_audit") or {}).get("research_pass")),
            -(_safe((row.get("full_audit") or {}).get("rating_score"), -1.0) or -1.0),
            -(_safe(row.get("gate_score"), -1.0) or -1.0),
            -(_safe(row.get("learning_score"), -1.0) or -1.0),
            row["name"],
        ),
    )
    summary = {
        "schema": "factorfactory.qlib-alpha158-evaluation/v1",
        "run_id": run_dir.name,
        "market": market,
        "portfolio_mode": portfolio_mode,
        "dataset": QlibDatasetSpec(market, universe_n, horizon).payload(),
        "protocol": {
            "stage_1": "all_158_training_safe_both_direction_discovery",
            "stage_2": "training_ranked_finalists_full_holdout_vault_rating_audit",
            "evaluation": "FactorFactory V4.2",
            "rating": "FactorFactory V4.3 2020-to-latest",
            "predeclared_trials": 1000,
            "actual_directional_hypotheses": 316,
            "full_audit_top": full_audit_top,
            "policy_label": "NON_PIT_RESEARCH",
        },
        "alpha158": alpha158_catalog(market),
        "completed": len(rows),
        "failed": sum(bool(row.get("error")) for row in rows),
        "research_passed": sum(bool(row.get("research_pass")) for row in rows),
        "full_audited": sum(bool(row.get("full_audit")) for row in rows),
        "full_passed": sum(
            bool((row.get("full_audit") or {}).get("vault_pass"))
            for row in rows
        ),
        "family_summary": _family_summary(rows),
        "elapsed_seconds": round(time.time() - started, 1),
        "rows": ordered,
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", nargs="+", choices=["ashare", "us"], default=["ashare", "us"])
    parser.add_argument("--us-modes", nargs="+", choices=["long_only", "long_short"], default=["long_only", "long_short"])
    parser.add_argument("--universe-n", type=int, default=500)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--full-audit-top", type=int, default=10)
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--refresh-html",
        action="store_true",
        help="只从既有不可变 summary 重新生成派生 HTML，不重新评价",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = ALPHA158_ARTIFACT_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_html:
        refreshed = []
        for summary_path in sorted(run_dir.glob("*-summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            report_path = summary_path.with_name(
                summary_path.name.replace("-summary.json", "-report.html")
            )
            report_path.write_text(_html(summary), encoding="utf-8")
            refreshed.append(str(report_path.resolve()))
        if not refreshed:
            raise RuntimeError(f"没有找到可渲染的 summary: {run_dir}")
        print(json.dumps({"run_id": run_id, "refreshed_html": refreshed}, ensure_ascii=False))
        return 0
    global_progress = ALPHA158_ARTIFACT_ROOT / "latest-progress.json"
    recorder = QlibNativeRecorder(run_id)
    modes = []
    if "ashare" in args.markets:
        modes.append(("ashare", "long_only"))
    if "us" in args.markets:
        modes.extend(("us", mode) for mode in args.us_modes)
    artifacts = []
    summaries = []
    for market, portfolio_mode in modes:
        summary = run_one(
            market,
            portfolio_mode,
            universe_n=args.universe_n,
            horizon=args.horizon,
            full_audit_top=args.full_audit_top,
            run_dir=run_dir,
            global_progress=global_progress,
        )
        key = _mode_key(market, portfolio_mode)
        json_path = run_dir / f"{key}-summary.json"
        if json_path.exists():
            existing = json.loads(json_path.read_text(encoding="utf-8"))
            if existing.get("completed") != 158:
                raise RuntimeError(f"已有不可变结果不完整，拒绝覆盖: {json_path}")
            summary = existing
        else:
            json_path = recorder.write_json(f"{key}-summary", summary)
        csv_path = run_dir / f"{key}-leaderboard.csv"
        if not csv_path.exists():
            _write_csv(csv_path, summary["rows"])
        html_path = run_dir / f"{key}-report.html"
        if not html_path.exists():
            html_path.write_text(_html(summary), encoding="utf-8")
        artifacts.extend([json_path, csv_path, html_path])
        summaries.append(summary)

    manifest = recorder.manifest(
        artifacts,
        {
            "modes": [_mode_key(market, mode) for market, mode in modes],
            "universe_n": args.universe_n,
            "horizon": args.horizon,
            "full_audit_top": args.full_audit_top,
            "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
        },
    )
    manifest_path = recorder.write_json("manifest", manifest)
    _atomic_json(
        global_progress,
        {
            "schema": "factorfactory.qlib-alpha158-progress/v1",
            "run_id": run_id,
            "state": "complete",
            "modes": [_mode_key(market, mode) for market, mode in modes],
            "manifest": str(manifest_path.resolve()),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps({"run_id": run_id, "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
