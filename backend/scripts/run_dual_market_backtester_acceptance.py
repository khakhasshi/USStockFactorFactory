#!/usr/bin/env python3
"""Run a frozen 30 A-share + 30 US StepEvent backtester acceptance set.

This suite tests execution and accounting invariants, not factor profitability.
Every task uses a real panel and the full 2020-to-latest research window while
varying DSL structure, direction, sizing, exits, liquidity and account rules.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.backtest.engine import run_backtest  # noqa: E402
from backend.app.config import (  # noqa: E402
    ASHARE_PANEL_GLOB,
    US_PANEL_GLOB,
    get_dsl_fields,
)
from backend.app.data.panel import PanelStore  # noqa: E402
from backend.app.dsl.engine import expression_profile, validate  # noqa: E402


PROTOCOL = "dual_market_step_event_v2_acceptance_v1"
START = "2020-01-01"
END = "2026-08-21"

COMMON_EXPRESSIONS = [
    "rank(-returns(close,20))",
    "rank(returns(close,60))",
    "rank(ts_std(returns(close,1),20))",
    "rank((close-ts_mean(close,20))/ts_std(close,20))",
    "rank(ts_delta(ts_mean(close,5),20))",
    "rank(ts_rank(close,20))",
    "rank(ts_corr(returns(close,1),returns(vol,1),20))",
    "rank(ts_mean(amount,20))",
    "rank(ts_mean(vol,20))",
    "rank((high-low)/close)",
    "rank(ts_mean((high-low)/close,20))",
    "rank((close-open)/(high-low))",
    "rank(ts_corr(high,low,20))",
    "rank(ts_max(close,20)/close)",
    "rank(close/ts_min(close,20))",
    "zscore(winsor_mad(returns(close,20),5))",
    "rank(ts_std(log(vol),60))",
    "rank(ts_corr(ts_delta(log(vol),1),ts_delta(log(close),1),60))",
    "rank(ts_sum(sign(returns(close,1)),20))",
    "rank(ts_mean(abs(returns(close,1)),20))",
]

ASHARE_EXPRESSIONS = COMMON_EXPRESSIONS + [
    "rank(pb)",
    "rank(pe_ttm)",
    "rank(ps_ttm)",
    "rank(dv_ttm)",
    "rank(log(total_mv))",
    "rank(turnover_rate)",
    "rank(volume_ratio)",
    "rank(net_mf_amount/(amount+1))",
    "rank((buy_lg_amount-sell_lg_amount)/(amount+1))",
    "rank(float_share/(total_mv+1))",
]

US_EXPRESSIONS = COMMON_EXPRESSIONS + [
    "rank(returns(close,252))",
    "rank((-returns(close,252))/ts_std(returns(close,1),251))",
    "rank(ts_mean(returns(close,1),5)/ts_std(returns(close,1),20))",
    "rank(ts_corr(returns(close,1),returns(close,5),20))",
    "rank(ts_delta(close,60)/ts_std(close,60))",
    "rank(ts_max(high,20)-close)",
    "rank(close-ts_min(low,20))",
    "rank(ts_std(amount,20)/ts_mean(amount,20))",
    "rank(ts_rank(vol,60)*ts_rank(close,60))",
    "rank(ts_corr(winsor(ts_delta(log(vol),1)),winsor(ts_delta(log(close),1)),60)*ts_std(winsor(ts_delta(log(vol),1)),60)*ts_std(winsor(ts_delta(log(close),1)),60))",
]


def _exit_policy(index: int) -> tuple[dict[str, Any], float | None, int]:
    profile = index % 6
    base: dict[str, Any] = {
        "atr_period": 14,
        "intrabar_conflict_policy": "conservative" if index % 2 == 0 else "optimistic",
    }
    if profile == 1:
        base |= {"fixed_stop_loss_pct": 0.08, "fixed_take_profit_pct": 0.20}
    elif profile == 2:
        base |= {"trailing_stop_pct": 0.10, "break_even_activation_pct": 0.10}
    elif profile == 3:
        base |= {"atr_stop_multiple": 2.5, "atr_take_profit_multiple": 4.0}
    elif profile == 4:
        base |= {"time_stop_sessions": 60}
    elif profile == 5:
        base |= {
            "fixed_stop_loss_pct": 0.08,
            "atr_stop_multiple": 2.5,
            "atr_trailing_multiple": 3.0,
            "time_stop_sessions": 60,
        }
    portfolio_stop = 0.20 if profile == 5 else None
    cooldown = 5 if portfolio_stop is not None else 0
    return base, portfolio_stop, cooldown


def _task(market: str, index: int, expression: str) -> dict[str, Any]:
    policy, portfolio_stop, cooldown = _exit_policy(index)
    long_short = market == "us" and index >= 15
    mode = "long_short" if long_short else "long_only"
    max_gross = 2.0 if long_short else 1.0
    sizing = ("equal_weight", "inverse_volatility", "atr_risk")[index % 3]
    task = {
        "task_id": f"{'ASH' if market == 'ashare' else 'US'}-{index + 1:02d}",
        "market": market,
        "mode": mode,
        "direction": 1 if index % 2 == 0 else -1,
        "expression": expression,
        "panel_glob": ASHARE_PANEL_GLOB if market == "ashare" else US_PANEL_GLOB,
        "start": START,
        "end": END,
        "universe_n": (100, 300, 500)[index % 3],
        "top_fraction": 0.10 if index % 2 == 0 else 0.20,
        "initial_capital": 10_000_000.0 if market == "ashare" else 1_000_000.0,
        "rebalance_every": (5, 10, 20)[index % 3],
        "slippage_bps": 5.0 if market == "ashare" else 2.0,
        "max_volume_participation": 0.05 if index % 2 == 0 else 0.10,
        "account_type": "margin" if long_short else "cash",
        "cash_buffer_fraction": 0.02,
        "max_gross_leverage": max_gross,
        "margin_interest_bps_annual": 500.0 if market == "us" else 600.0,
        "position_sizing": sizing,
        "max_positions": (30, 60, 100)[index % 3],
        "max_position_weight": 0.10,
        "min_trade_notional": 1_000.0 if market == "ashare" else 100.0,
        "rebalance_buffer_pct": 0.02,
        "long_gross_target": 0.95,
        "short_gross_target": 0.95 if long_short else 0.0,
        "risk_per_position_fraction": 0.01,
        "spread_bps": 2.0 if market == "ashare" else 1.0,
        "impact_model": ("fixed", "linear", "square_root")[index % 3],
        "impact_coefficient_bps": 10.0,
        "unfilled_order_policy": "carry" if index % 4 == 0 else "cancel",
        "max_order_age_sessions": 3,
        "max_stale_sessions": 20,
        "liquidate_at_end": index % 5 == 0,
        "portfolio_stop_drawdown_pct": portfolio_stop,
        "portfolio_daily_loss_pct": 0.08 if index % 10 == 7 else None,
        "risk_cooldown_sessions": cooldown,
        "borrow_cost_bps_annual": 300.0 if long_short else 0.0,
        "exit_policy": policy,
    }
    profile = expression_profile(expression)
    task["expression_profile"] = {
        key: profile[key]
        for key in ("canonical", "operators", "fields", "windows", "required_history", "complexity")
    }
    frozen = json.dumps(task, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    task["task_hash"] = hashlib.sha256(frozen.encode("utf-8")).hexdigest()[:16]
    return task


def build_task_set() -> list[dict[str, Any]]:
    tasks = [
        _task("ashare", index, expression)
        for index, expression in enumerate(ASHARE_EXPRESSIONS)
    ] + [
        _task("us", index, expression)
        for index, expression in enumerate(US_EXPRESSIONS)
    ]
    if len(tasks) != 60:
        raise AssertionError(f"任务数错误: {len(tasks)}")
    if len({row["task_hash"] for row in tasks}) != len(tasks):
        raise AssertionError("冻结任务存在重复 hash")
    for row in tasks:
        error = validate(row["expression"], get_dsl_fields(row["market"]))
        if error:
            raise ValueError(f"{row['task_id']} DSL 非法: {error}")
    return tasks


def _finite_series(values: list[Any]) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _summarize(task: dict[str, Any], result: dict[str, Any], seconds: float) -> dict[str, Any]:
    stats = result["stats"]
    integrity = result["integrity"]
    curve = result["curve"]
    daily = result["daily_steps"]
    days = int(stats.get("days", 0))
    limit = float(task["max_gross_leverage"])
    max_after = float(stats.get("max_open_gross_leverage_after_control", 0.0))
    triggers = int(stats.get("portfolio_risk_trigger_events", 0))
    rearms = int(stats.get("portfolio_risk_rearms", 0))
    active_at_end = bool(stats.get("portfolio_risk_active_at_end", False))
    expected_rearm_gap = 1 if active_at_end else 0
    flat_limit = max(
        30,
        int(task["risk_cooldown_sessions"]) + int(task["rebalance_every"]) + 5,
    )
    checks = {
        "integrity_all_pass": bool(integrity.get("all_pass")),
        "sufficient_sessions": days >= 1_000,
        "nonempty_execution": int(stats.get("fills", 0)) > 0,
        "curve_lengths_match": all(
            len(curve.get(key, [])) == days
            for key in ("dates", "equity", "cost_free_proxy", "daily_ret")
        ),
        "daily_detail_window_contract": len(daily) == min(120, days),
        "finite_curve": _finite_series(curve.get("equity", []))
        and _finite_series(curve.get("daily_ret", [])),
        "positive_nlv": float(stats.get("final_nlv", 0.0)) > 0
        and all(float(value) > 0 for value in curve.get("equity", []))
        and all(float(row.get("close_nlv", 0.0)) > 0 for row in daily),
        "hard_leverage_respected": max_after <= limit + max(1e-6, limit * 1e-6)
        and int(integrity.get("gross_leverage_violations", 0)) == 0,
        "portfolio_state_consistent": rearms <= triggers
        and triggers - rearms == expected_rearm_gap,
        "no_permanent_flatline": int(stats.get("terminal_flat_sessions", 0)) <= flat_limit,
        "market_constraints_pass": int(integrity.get("cash_account_negative_cash_violations", 0)) == 0
        and int(integrity.get("ashare_buy_lot_violations", 0)) == 0
        and int(integrity.get("long_only_negative_position_violations", 0)) == 0,
        "event_order_pass": int(integrity.get("event_phase_order_violations", 0)) == 0,
    }
    return {
        "task_id": task["task_id"],
        "task_hash": task["task_hash"],
        "market": task["market"],
        "mode": task["mode"],
        "direction": task["direction"],
        "expression": task["expression"],
        "duration_seconds": round(seconds, 3),
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "stats": {
            key: stats.get(key)
            for key in (
                "days", "fills", "orders", "final_nav", "ann_ret", "sharpe",
                "max_dd", "avg_daily_turnover", "total_execution_cost",
                "gross_leverage_breach_events", "automatic_deleveraging_events",
                "leverage_limited_orders", "max_open_gross_leverage_after_control",
                "portfolio_risk_trigger_events", "portfolio_risk_rearms",
                "portfolio_risk_active_sessions", "portfolio_risk_active_at_end",
                "max_portfolio_risk_cycle_drawdown", "terminal_flat_sessions",
            )
        },
        "integrity": integrity,
        "error": "",
    }


def run_one(task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        kwargs = {
            key: value
            for key, value in task.items()
            if key not in {"task_id", "task_hash", "expression_profile"}
        }
        expression = kwargs.pop("expression")
        result = run_backtest(
            expression=expression,
            capture_detail=False,
            response_trade_limit=0,
            **kwargs,
        )
        return _summarize(task, result, time.perf_counter() - started)
    except Exception as exc:  # noqa: BLE001 - this is an acceptance recorder
        return {
            "task_id": task["task_id"],
            "task_hash": task["task_hash"],
            "market": task["market"],
            "mode": task["mode"],
            "direction": task["direction"],
            "expression": task["expression"],
            "duration_seconds": round(time.perf_counter() - started, 3),
            "status": "error",
            "checks": {},
            "stats": {},
            "integrity": {},
            "error": f"{type(exc).__name__}: {exc}"[:4000],
        }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_html(path: Path, summary: dict[str, Any]) -> None:
    rows = []
    for row in summary["results"]:
        stats = row.get("stats", {})
        failed_checks = [key for key, passed in row.get("checks", {}).items() if not passed]
        rows.append(
            "<tr>"
            f"<td>{html.escape(row['task_id'])}</td>"
            f"<td>{html.escape(row['market'])}</td>"
            f"<td>{html.escape(row['mode'])}</td>"
            f"<td>{row['direction']:+d}</td>"
            f"<td class='{row['status']}'>{html.escape(row['status'].upper())}</td>"
            f"<td>{row['duration_seconds']:.2f}</td>"
            f"<td>{stats.get('fills', '—')}</td>"
            f"<td>{stats.get('sharpe', '—')}</td>"
            f"<td>{stats.get('max_dd', '—')}</td>"
            f"<td>{stats.get('terminal_flat_sessions', '—')}</td>"
            f"<td><code>{html.escape(row['expression'])}</code></td>"
            f"<td>{html.escape(', '.join(failed_checks) or row.get('error', ''))}</td>"
            "</tr>"
        )
    document = f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<title>双市场回测器验收</title><style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;margin:28px}}
h1,h2{{color:#f0f6fc}} .cards{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px 18px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border:1px solid #30363d;padding:7px;vertical-align:top}}
th{{position:sticky;top:0;background:#21262d}}code{{white-space:pre-wrap;word-break:break-word}}.pass{{color:#3fb950;font-weight:700}}.fail,.error{{color:#f85149;font-weight:700}}
</style></head><body><h1>双市场 StepEvent 回测器验收</h1>
<p>协议 {PROTOCOL}；窗口 {START}～{END}。收益不是通过门槛，账本、执行、状态机和曲线完整性才是。</p>
<div class='cards'><div class='card'>总任务<br><b>{summary['total']}</b></div><div class='card'>PASS<br><b>{summary['passed']}</b></div><div class='card'>FAIL/ERROR<br><b>{summary['failed'] + summary['errors']}</b></div><div class='card'>总耗时<br><b>{summary['duration_seconds']:.1f}s</b></div></div>
<table><thead><tr><th>任务</th><th>市场</th><th>组合</th><th>方向</th><th>状态</th><th>秒</th><th>成交</th><th>Sharpe</th><th>回撤</th><th>末端空仓日</th><th>表达式</th><th>失败项</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def _release_market_panel(market: str) -> None:
    with PanelStore._registry_lock:
        keys = [key for key in PanelStore._instances if key.startswith(f"{market}::")]
        for key in keys:
            PanelStore._instances.pop(key, None)
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= max(1, min(8, os.cpu_count() or 1)):
        raise SystemExit("workers 必须在 1..min(8, CPU数)")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise SystemExit(f"输出目录非空，请换新目录或使用 --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    tasks = build_task_set()
    manifest = {
        "protocol": PROTOCOL,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": START,
        "end": END,
        "workers": args.workers,
        "task_count": len(tasks),
        "market_counts": {"ashare": 30, "us": 30},
        "acceptance_semantics": "engine_and_audit_not_profitability",
        "tasks": tasks,
    }
    _atomic_json(output / "task_set.json", manifest)
    progress_path = output / "progress.jsonl"
    completed: dict[str, dict[str, Any]] = {}
    if args.resume and progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                completed[row["task_id"]] = row
    started = time.perf_counter()
    total_done = len(completed)
    print(f"START protocol={PROTOCOL} tasks=60 resumed={total_done} workers={args.workers}", flush=True)
    for market in ("ashare", "us"):
        pending = [row for row in tasks if row["market"] == market and row["task_id"] not in completed]
        if not pending:
            continue
        print(f"MARKET_START market={market} pending={len(pending)}", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix=f"accept-{market}") as executor:
            futures = {executor.submit(run_one, task): task for task in pending}
            for future in as_completed(futures):
                row = future.result()
                completed[row["task_id"]] = row
                total_done += 1
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                print(
                    f"PROGRESS {total_done}/60 task={row['task_id']} status={row['status']} "
                    f"seconds={row['duration_seconds']:.2f} fills={row.get('stats', {}).get('fills', 0)} "
                    f"error={row.get('error', '')[:160]}",
                    flush=True,
                )
        print(f"MARKET_DONE market={market}", flush=True)
        _release_market_panel(market)

    results = [completed[row["task_id"]] for row in tasks if row["task_id"] in completed]
    durations = [float(row["duration_seconds"]) for row in results]
    summary = {
        "protocol": PROTOCOL,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total": len(results),
        "passed": sum(row["status"] == "pass" for row in results),
        "failed": sum(row["status"] == "fail" for row in results),
        "errors": sum(row["status"] == "error" for row in results),
        "duration_seconds": round(time.perf_counter() - started, 3),
        "task_duration_median_seconds": round(statistics.median(durations), 3) if durations else 0.0,
        "task_duration_p95_seconds": round(sorted(durations)[max(0, math.ceil(len(durations) * 0.95) - 1)], 3) if durations else 0.0,
        "results": results,
    }
    _atomic_json(output / "summary.json", summary)
    _atomic_json(
        output / "failures.json",
        [row for row in results if row["status"] != "pass"],
    )
    _write_html(output / "report.html", summary)
    print(
        f"COMPLETE total={summary['total']} pass={summary['passed']} fail={summary['failed']} "
        f"error={summary['errors']} seconds={summary['duration_seconds']:.2f}",
        flush=True,
    )
    return 0 if summary["passed"] == 60 else 2


if __name__ == "__main__":
    raise SystemExit(main())
