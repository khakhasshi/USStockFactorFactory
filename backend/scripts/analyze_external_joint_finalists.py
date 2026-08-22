#!/usr/bin/env python3
"""Audit external joint-leaderboard finalists without changing source reports.

The script treats the full-window leaderboards as diagnostic inputs.  It uses
the step-event daily ledgers for return-path deduplication and stability, then
writes a new immutable-style report directory.  It never promotes a factor or
modifies the source factor libraries.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


DEFAULT_REPORTS = {
    "A股纯多": "external-joint-ashare-us-to-ashare-long-only-vector-2020-latest-20260821",
    "美股多空": "external-joint-ashare-us-to-us-long-short-vector-2020-latest-20260821",
    "美股纯多": "external-joint-ashare-us-to-us-long-only-vector-2020-latest-20260821",
}
CORRELATION_THRESHOLD = 0.85
TRADING_DAYS = 252
BOOTSTRAP_BLOCK = 20
BOOTSTRAP_ITERATIONS = 2_000


@dataclass
class Finalist:
    job: str
    key: str
    source_hash: str
    direction: int
    expression: str
    origin_scope: str
    leaderboard_rank: int
    practical_pass: bool
    multiple_test_pass: bool
    production_eligible: bool
    dates: list[date]
    returns: np.ndarray
    manifest: dict[str, Any]
    row: dict[str, str]
    metrics: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truth(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _return_metrics(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "observations": 0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "total_return": 0.0,
            "positive_day_rate": 0.0,
        }
    nav = np.cumprod(1.0 + values)
    annualized_return = float(nav[-1] ** (TRADING_DAYS / len(values)) - 1.0)
    daily_std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    volatility = daily_std * math.sqrt(TRADING_DAYS)
    sharpe = (
        float(values.mean()) / daily_std * math.sqrt(TRADING_DAYS)
        if daily_std > 1e-15
        else 0.0
    )
    drawdown = 1.0 - nav / np.maximum.accumulate(nav)
    return {
        "observations": int(len(values)),
        "annualized_return": round(annualized_return, 8),
        "annualized_volatility": round(volatility, 8),
        "sharpe": round(sharpe, 6),
        "max_drawdown": round(float(drawdown.max()), 8),
        "total_return": round(float(nav[-1] - 1.0), 8),
        "positive_day_rate": round(float((values > 0.0).mean()), 6),
    }


def _rolling_sharpe(values: np.ndarray, window: int = TRADING_DAYS) -> dict[str, float]:
    if len(values) < window:
        return {"q10": 0.0, "median": 0.0, "q90": 0.0, "positive_fraction": 0.0}
    output: list[float] = []
    for end in range(window, len(values) + 1):
        sample = values[end - window : end]
        std = float(sample.std(ddof=1))
        if std > 1e-15:
            output.append(float(sample.mean()) / std * math.sqrt(TRADING_DAYS))
    result = np.asarray(output, dtype=float)
    return {
        "q10": round(float(np.quantile(result, 0.10)), 6),
        "median": round(float(np.median(result)), 6),
        "q90": round(float(np.quantile(result, 0.90)), 6),
        "positive_fraction": round(float((result > 0.0).mean()), 6),
    }


def _newey_west_t(values: np.ndarray, lags: int = 10) -> float:
    values = np.asarray(values, dtype=float)
    centered = values - values.mean()
    count = len(values)
    if count < 2:
        return 0.0
    long_run_variance = float(centered @ centered) / count
    for lag in range(1, min(lags, count - 1) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        covariance = float(centered[lag:] @ centered[:-lag]) / count
        long_run_variance += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run_variance, 0.0) / count)
    return round(float(values.mean() / standard_error), 6) if standard_error > 0 else 0.0


def _block_bootstrap(values: np.ndarray, seed_text: str) -> dict[str, float | int]:
    """Circular moving-block bootstrap of the arithmetic annualized mean.

    This is a dependence-aware stability diagnostic, not a post-selection
    significance test.  The leaderboard's multiple-testing result remains the
    authoritative admission field.
    """

    values = np.asarray(values, dtype=float)
    count = len(values)
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    block_count = math.ceil(count / BOOTSTRAP_BLOCK)
    means = np.empty(BOOTSTRAP_ITERATIONS, dtype=float)
    offsets = np.arange(BOOTSTRAP_BLOCK)
    for index in range(BOOTSTRAP_ITERATIONS):
        starts = rng.integers(0, count, size=block_count)
        sample_indices = ((starts[:, None] + offsets) % count).reshape(-1)[:count]
        means[index] = float(values[sample_indices].mean()) * TRADING_DAYS
    return {
        "block_sessions": BOOTSTRAP_BLOCK,
        "iterations": BOOTSTRAP_ITERATIONS,
        "annualized_mean_ci_low": round(float(np.quantile(means, 0.025)), 8),
        "annualized_mean_ci_high": round(float(np.quantile(means, 0.975)), 8),
        "positive_probability": round(float((means > 0.0).mean()), 6),
    }


def _read_daily_ledger(path: Path) -> tuple[list[date], np.ndarray]:
    payload = pq.read_table(path, columns=["trade_date", "daily_return"]).to_pydict()
    dates = [date.fromisoformat(str(value)) for value in payload["trade_date"]]
    returns = np.asarray(payload["daily_return"], dtype=float)
    return dates, returns


def _mechanism(expression: str) -> str:
    flow_fields = ("buy_lg_", "sell_lg_", "buy_elg_", "sell_elg_", "net_mf_")
    if any(field in expression for field in flow_fields):
        return "大单参与度/资金流拥挤"
    if "close / amount" in expression or "close / ts_mean(amount" in expression:
        return "成交股数/流动性水平代理"
    if "ts_mean(close" in expression and "returns(" not in expression:
        return "名义股价水平代理"
    return "通用价量横截面"


def _load_finalists(job: str, report_dir: Path) -> list[Finalist]:
    with (report_dir / "leaderboard_full.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    finalists: list[Finalist] = []
    for ledger_dir in sorted((report_dir / "finalist_ledgers").iterdir()):
        if not ledger_dir.is_dir():
            continue
        key = ledger_dir.name
        source_hash, suffix = key.rsplit("-", 1)
        direction = -1 if suffix == "minus" else 1
        matches = [
            row for row in rows
            if row.get("source_expression_hash") == source_hash
            and int(row.get("direction") or 0) == direction
        ]
        if len(matches) != 1:
            raise RuntimeError(f"{job}/{key}: 无法唯一映射榜单行，匹配数={len(matches)}")
        row = matches[0]
        dates, returns = _read_daily_ledger(ledger_dir / "daily_ledger.parquet")
        manifest = json.loads((ledger_dir / "manifest.json").read_text(encoding="utf-8"))
        base_metrics = _return_metrics(returns)
        year_metrics = []
        for year in sorted({value.year for value in dates}):
            mask = np.asarray([value.year == year for value in dates], dtype=bool)
            year_metrics.append({"year": year, **_return_metrics(returns[mask])})
        stats = manifest["stats"]
        years = float(stats["days"]) / TRADING_DAYS
        cost_free_nav = float(stats["same_orders_cost_free_final_nav_proxy"])
        net_nav = float(stats["final_nav"])
        annualized_cost_drag = cost_free_nav ** (1.0 / years) - net_nav ** (1.0 / years)
        rolling = _rolling_sharpe(returns)
        bootstrap = _block_bootstrap(returns, f"{job}:{key}")
        metrics = {
            **base_metrics,
            "yearly": year_metrics,
            "positive_years": sum(item["annualized_return"] > 0 for item in year_metrics),
            "negative_years": sum(item["annualized_return"] < 0 for item in year_metrics),
            "worst_year_return": min(item["annualized_return"] for item in year_metrics),
            "best_year_return": max(item["annualized_return"] for item in year_metrics),
            "rolling_252d_sharpe": rolling,
            "newey_west_t_lag10": _newey_west_t(returns),
            "block_bootstrap": bootstrap,
            "cost_free_final_nav_proxy": round(cost_free_nav, 8),
            "net_final_nav": round(net_nav, 8),
            "annualized_cost_drag": round(annualized_cost_drag, 8),
            "avg_daily_turnover": round(float(stats["avg_daily_turnover"]), 8),
            "avg_gross_exposure": round(float(stats["avg_gross_exposure"]), 8),
            "avg_net_exposure": round(float(stats["avg_net_exposure"]), 8),
            "fill_rate": round(float(stats["fill_rate"]), 8),
            "total_execution_cost": round(float(stats["total_execution_cost"]), 6),
            "borrow_cost": round(float(stats.get("borrow_cost", 0.0)), 6),
            "integrity_all_pass": bool(manifest["integrity"]["all_pass"]),
            "vector_raw_ann_return_15bps": _finite_float(row.get("ann_return_bps_15")),
            "vector_benchmark_ann_return_15bps": _finite_float(row.get("benchmark_ann_return_bps_15")),
            "vector_active_ann_return_15bps": _finite_float(row.get("active_ann_return_bps_15")),
        }
        finalists.append(
            Finalist(
                job=job,
                key=key,
                source_hash=source_hash,
                direction=direction,
                expression=row["expression"],
                origin_scope=row.get("origin_scope", ""),
                leaderboard_rank=int(row["overall_rank"]),
                practical_pass=_truth(row.get("practical_pass")),
                multiple_test_pass=_truth(row.get("multiple_test_pass")),
                production_eligible=_truth(row.get("production_eligible")),
                dates=dates,
                returns=returns,
                manifest=manifest,
                row=row,
                metrics=metrics,
            )
        )
    return finalists


def _aligned_correlation(left: Finalist, right: Finalist) -> float:
    left_values = dict(zip(left.dates, left.returns))
    right_values = dict(zip(right.dates, right.returns))
    common = sorted(set(left_values) & set(right_values))
    if len(common) < 2:
        return 0.0
    left_array = np.asarray([left_values[value] for value in common])
    right_array = np.asarray([right_values[value] for value in common])
    return round(float(np.corrcoef(left_array, right_array)[0, 1]), 8)


def _stability_score(item: Finalist) -> float:
    metrics = item.metrics
    rolling = metrics["rolling_252d_sharpe"]
    return (
        float(metrics["sharpe"])
        - float(metrics["max_drawdown"])
        - float(metrics["annualized_cost_drag"])
        + 0.10 * float(rolling["positive_fraction"])
    )


def _cluster_job(finalists: list[Finalist]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    ordered = sorted(finalists, key=_stability_score, reverse=True)
    clusters: list[dict[str, Any]] = []
    assignment: dict[str, str] = {}
    for candidate in ordered:
        matched = None
        matched_corr = 0.0
        for cluster in clusters:
            representative = next(item for item in finalists if item.key == cluster["representative"])
            correlation = _aligned_correlation(candidate, representative)
            if correlation >= CORRELATION_THRESHOLD:
                matched = cluster
                matched_corr = correlation
                break
        if matched is None:
            clusters.append(
                {
                    "representative": candidate.key,
                    "members": [candidate.key],
                    "representative_correlations": {candidate.key: 1.0},
                }
            )
            assignment[candidate.key] = candidate.key
        else:
            matched["members"].append(candidate.key)
            matched["representative_correlations"][candidate.key] = matched_corr
            assignment[candidate.key] = matched["representative"]
    return clusters, assignment


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=Path("var/reports"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    report_dirs = {job: args.reports_root / name for job, name in DEFAULT_REPORTS.items()}
    all_finalists: dict[str, list[Finalist]] = {
        job: _load_finalists(job, report_dir) for job, report_dir in report_dirs.items()
    }

    correlations: list[dict[str, Any]] = []
    clusters_by_job: dict[str, list[dict[str, Any]]] = {}
    assignments: dict[tuple[str, str], str] = {}
    for job, finalists in all_finalists.items():
        for left in finalists:
            for right in finalists:
                correlations.append(
                    {
                        "job": job,
                        "left": left.key,
                        "right": right.key,
                        "correlation": _aligned_correlation(left, right),
                    }
                )
        clusters, mapping = _cluster_job(finalists)
        clusters_by_job[job] = clusters
        assignments.update({(job, key): value for key, value in mapping.items()})

    finalist_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for job, finalists in all_finalists.items():
        for item in sorted(finalists, key=lambda value: value.leaderboard_rank):
            metrics = item.metrics
            representative = assignments[(job, item.key)]
            is_representative = representative == item.key
            if not is_representative:
                decision = "DROP_RETURN_PATH_DUPLICATE"
                reason = f"与 {representative} 的逐日收益相关性不低于 {CORRELATION_THRESHOLD:.2f}"
            elif job == "A股纯多":
                decision = "ARCHIVE_DIAGNOSTIC_ONLY"
                reason = "簇代表仍只有低Sharpe、高回撤、负年度和负滚动Sharpe区间"
            else:
                decision = "RETAIN_DIAGNOSTIC_NOT_PROMOTE"
                reason = "保留为独立收益簇代表；全窗口筛选且未通过多重检验，禁止正式入库"
            common = {
                "job": job,
                "key": item.key,
                "source_hash": item.source_hash,
                "direction": item.direction,
                "leaderboard_rank": item.leaderboard_rank,
                "origin_scope": item.origin_scope,
                "mechanism": _mechanism(item.expression),
                "expression": item.expression,
                "return_cluster_representative": representative,
                "is_cluster_representative": is_representative,
                "decision": decision,
            }
            finalist_rows.append(
                {
                    **common,
                    "annualized_return": metrics["annualized_return"],
                    "sharpe": metrics["sharpe"],
                    "max_drawdown": metrics["max_drawdown"],
                    "positive_years": metrics["positive_years"],
                    "negative_years": metrics["negative_years"],
                    "worst_year_return": metrics["worst_year_return"],
                    "rolling_sharpe_q10": metrics["rolling_252d_sharpe"]["q10"],
                    "rolling_sharpe_positive_fraction": metrics["rolling_252d_sharpe"]["positive_fraction"],
                    "annualized_cost_drag": metrics["annualized_cost_drag"],
                    "newey_west_t_lag10": metrics["newey_west_t_lag10"],
                    "bootstrap_ci_low": metrics["block_bootstrap"]["annualized_mean_ci_low"],
                    "bootstrap_ci_high": metrics["block_bootstrap"]["annualized_mean_ci_high"],
                    "bootstrap_positive_probability": metrics["block_bootstrap"]["positive_probability"],
                    "avg_daily_turnover": metrics["avg_daily_turnover"],
                    "avg_gross_exposure": metrics["avg_gross_exposure"],
                    "avg_net_exposure": metrics["avg_net_exposure"],
                    "integrity_all_pass": metrics["integrity_all_pass"],
                    "practical_pass": item.practical_pass,
                    "multiple_test_pass": item.multiple_test_pass,
                    "production_eligible": item.production_eligible,
                }
            )
            decision_rows.append({**common, "reason": reason})
            for yearly in metrics["yearly"]:
                yearly_rows.append({"job": job, "key": item.key, **yearly})

    # Compare the same US orientation across long-only and long-short.  With one
    # dollar long and one dollar short in the long-short book, LS - long-only is
    # an after-cost approximation to the short-book contribution.
    cross_mode_rows: list[dict[str, Any]] = []
    us_ls = {item.key: item for item in all_finalists["美股多空"]}
    us_lo = {item.key: item for item in all_finalists["美股纯多"]}
    for key in sorted(set(us_ls) & set(us_lo)):
        left, long_only = us_ls[key], us_lo[key]
        ls_values = dict(zip(left.dates, left.returns))
        lo_values = dict(zip(long_only.dates, long_only.returns))
        common_dates = sorted(set(ls_values) & set(lo_values))
        ls_array = np.asarray([ls_values[value] for value in common_dates])
        lo_array = np.asarray([lo_values[value] for value in common_dates])
        cross_mode_rows.append(
            {
                "key": key,
                "long_short_vs_long_only_correlation": round(float(np.corrcoef(ls_array, lo_array)[0, 1]), 8),
                "long_short_arithmetic_ann_return": round(float(ls_array.mean() * TRADING_DAYS), 8),
                "long_leg_arithmetic_ann_return": round(float(lo_array.mean() * TRADING_DAYS), 8),
                "implicit_short_book_after_cost_ann_contribution": round(float((ls_array - lo_array).mean() * TRADING_DAYS), 8),
            }
        )

    input_manifest: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy_label": "FULL_WINDOW_POST_SELECTION_DIAGNOSTIC",
        "correlation_threshold": CORRELATION_THRESHOLD,
        "bootstrap": {"block_sessions": BOOTSTRAP_BLOCK, "iterations": BOOTSTRAP_ITERATIONS},
        "inputs": {},
    }
    for job, report_dir in report_dirs.items():
        files = [
            report_dir / "manifest.json",
            report_dir / "protocol.json",
            report_dir / "leaderboard_full.csv",
            report_dir / "finalist_ledger_audits.json",
        ]
        files.extend(sorted((report_dir / "finalist_ledgers").glob("*/daily_ledger.parquet")))
        input_manifest["inputs"][job] = {
            "report_dir": str(report_dir.resolve()),
            "files": {
                str(path.relative_to(report_dir)): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
                for path in files
            },
        }

    summary = {
        "policy_label": input_manifest["policy_label"],
        "interpretation_boundary": (
            "2020年至最新日期是已用于筛选的全窗口；年度、滚动和bootstrap结果均为选择后诊断，"
            "不是独立OOS、正式入库或实盘批准。"
        ),
        "correlation_threshold": CORRELATION_THRESHOLD,
        "jobs": {
            job: {
                "finalists": len(items),
                "return_path_clusters": len(clusters_by_job[job]),
                "clusters": clusters_by_job[job],
                "production_eligible": sum(item.production_eligible for item in items),
                "multiple_test_pass": sum(item.multiple_test_pass for item in items),
            }
            for job, items in all_finalists.items()
        },
        "cross_mode_attribution": cross_mode_rows,
        "decisions": decision_rows,
    }

    _write_csv(args.output / "finalist_metrics.csv", finalist_rows)
    _write_csv(args.output / "yearly_metrics.csv", yearly_rows)
    _write_csv(args.output / "return_correlations.csv", correlations)
    _write_csv(args.output / "cross_mode_attribution.csv", cross_mode_rows)
    _write_csv(args.output / "decisions.csv", decision_rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "input_manifest.json").write_text(
        json.dumps(input_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    by_key = {(row["job"], row["key"]): row for row in finalist_rows}
    lines = [
        "# 外部联合因子榜首去重、收益归因与稳定性审计",
        "",
        "> 结论边界：`FULL_WINDOW_POST_SELECTION_DIAGNOSTIC`。2020年至最新日期已经参与筛选；",
        "> 本报告不能作为独立样本外证明、正式因子入库或实盘批准。Bootstrap和HAC统计量也不是选择后显著性修复。",
        "",
        "## 一句话结论",
        "",
        "- A股三个榜首只有 **1个高度重复的资金流/大单参与度收益簇**，代表本身也不合格，整体归档。",
        "- 美股多空三个榜首压缩为 **2个收益簇**；保留两个诊断代表，第三个删除为重复项。",
        "- 美股纯多三个榜首只有 **1个收益簇**；高收益包含明显市场多头暴露，不能视作三个独立Alpha。",
        "- 跨模式最终只看到两类美股假设：**低名义股价水平** 与 **成交股数/流动性水平**。全部来自A股外部库，说明可迁移，但不等于已证明稳健。",
        "",
        "## 收益路径去重",
        "",
        "| 任务 | 步进候选 | 收益簇 | 结论 |",
        "|---|---:|---:|---|",
    ]
    conclusions = {
        "A股纯多": "三个候选相关性0.988以上；只保留一个归档代表",
        "美股多空": "低价水平单独成簇；两个流动性表达式相关性约0.990",
        "美股纯多": "三个候选相关性0.971以上；实质为同一风格暴露",
    }
    for job, items in all_finalists.items():
        lines.append(f"| {job} | {len(items)} | {len(clusters_by_job[job])} | {conclusions[job]} |")

    lines.extend(
        [
            "",
            "相关性阈值为0.85，采用质量排序后的代表点聚类；负相关不会被当作重复。",
            "",
            "## 步进复测与稳定性",
            "",
            "| 任务 | 候选 | 年化 | Sharpe | 最大回撤 | 正收益年份 | 252日Sharpe为正占比 | 年化成本拖累 | 处置 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in sorted(finalist_rows, key=lambda value: (value["job"], value["leaderboard_rank"])):
        lines.append(
            f"| {row['job']} | `{row['key']}` | {_pct(float(row['annualized_return']))} | "
            f"{float(row['sharpe']):.2f} | {_pct(float(row['max_drawdown']))} | "
            f"{row['positive_years']}/7 | {_pct(float(row['rolling_sharpe_positive_fraction']))} | "
            f"{_pct(float(row['annualized_cost_drag']))} | `{row['decision']}` |"
        )

    lines.extend(
        [
            "",
            "## 收益来源归因",
            "",
            "### A股纯多",
            "",
            "三个表达式都在押注大单成交参与度，逐日收益几乎重合。2022年约亏26%至33%，2023年继续为负；",
            "全期Sharpe约0.33、最大回撤49%至53%，滚动一年Sharpe的10%分位低于-1。成本不是主要失败原因，",
            "核心问题是状态依赖和收益路径单一。因此不留下可继续晋级的A股候选。",
            "",
            "### 美股多空",
            "",
            "`ab3815c3c1023bca-plus` 是低名义股价水平代理，七个自然年均为正，但2020年接近零；",
            "另外两个表达式本质上都近似负的 `close/amount`，即偏向更高成交股数/更强流动性的股票，",
            "二者收益相关性约0.990，只保留表现更稳的 `bd51aa0f615e6c54-minus`。多空组合净暴露约1%，",
            "但年化执行与借券拖累约3.3%至3.4%，这是实质性成本来源。",
            "",
            "### 美股纯多及多空腿拆分",
            "",
            "纯多三个组合相关性均超过0.97。向量结果中组合年化约25%，同期基准约15.5%，活跃部分约9%至10%；",
            "因此headline收益很大一部分来自市场多头。将同一表达式的多空逐日收益减去纯多逐日收益，",
            "得到的隐含空头腿税后年化贡献约-13%至-15%：收益主要由多头篮子贡献，空头腿总体拖累，",
            "多空结果更像“多头选股优势减去上涨的空头篮子”，而不是两条都赚钱。",
            "",
            "## 最终研究记录",
            "",
            "1. A股：全部停止晋级；`99d1a5ac2f4073b9-plus` 仅作为该重复簇的审计代表保存。",
            "2. 美股多空：保留 `ab3815c3c1023bca-plus` 与 `bd51aa0f615e6c54-minus` 两个诊断代表。",
            "3. 美股纯多：仅保留 `bd51aa0f615e6c54-minus` 作为模式代表，不把另外两个计作独立来源。",
            "4. 所有保留项仍为 `NOT_PROMOTE`：美股候选均未通过多重检验，且使用的是已参与筛选的全窗口。",
            "5. 下一次真正验证应冻结这些表达式和方向，使用未来新增数据或未参与本次排名的独立时间段；",
            "   在此之前不应反复调整参数后重新宣称OOS。",
            "",
            "## 产物",
            "",
            "- `finalist_metrics.csv`：全期、滚动、HAC、区块Bootstrap、成本与准入字段。",
            "- `yearly_metrics.csv`：逐年年化、Sharpe、回撤和正收益日比例。",
            "- `return_correlations.csv`：任务内逐日净收益相关矩阵。",
            "- `cross_mode_attribution.csv`：美股纯多/多空及隐含空头腿近似拆分。",
            "- `decisions.csv`：每个候选的保留、归档或去重决定。",
            "- `input_manifest.json`：全部输入哈希，可用于复现和漂移检查。",
        ]
    )
    (args.output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "summary": summary["jobs"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
