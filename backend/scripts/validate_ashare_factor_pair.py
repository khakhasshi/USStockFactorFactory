#!/usr/bin/env python3
"""Frozen 2-5 factor A-share event replay and private-layer audit.

The weight-search artifact is authoritative: component identities, directions,
and weights must already be frozen using INNER_PUBLIC + META_TRAIN only.  This
script then performs one read of later layers and never retunes the weights.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from app.backtest.batch import (  # noqa: E402
    BatchBacktestSpec,
    information_coefficients,
    run_cost_scenarios,
)
from app.backtest.engine import (  # noqa: E402
    _prepare_backtest_frame,
    _write_artifacts,
)
from app.config import ASHARE_PANEL_GLOB, get_dsl_fields  # noqa: E402
from app.data.panel import PanelStore  # noqa: E402
from app.dsl.engine import parse, required_history  # noqa: E402


PROTOCOL = "frozen_ashare_multifactor_private_validation_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _prepare_frozen_composite_frame(
    *,
    components: list[dict],
    weights: list[float],
    universe_n: int,
    start: date,
    end: date,
    forward_horizon: int,
) -> tuple[pl.DataFrame, PanelStore]:
    """Reproduce the optimizer's per-component Top-N percentile semantics.

    A plain composite DSL expression is not sufficient: ``rank(x)`` is first
    evaluated on the full market, while the optimizer then orients and
    re-ranks every component *inside the Top-N liquidity universe* before
    applying weights.  This materializer keeps that second ranking step exact
    on all validation dates without exposing them to the weight search.
    """
    store = PanelStore.get(ASHARE_PANEL_GLOB, "ashare")
    panel, dates, _, _ = store.read_snapshot()
    active = [
        (component, float(weight))
        for component, weight in zip(components, weights)
        if float(weight) > 1e-12
    ]
    if len(active) < 2:
        raise ValueError("冻结组合至少需要两个正权重组件")
    in_range = [value for value in dates if start <= value <= end]
    if len(in_range) < 60:
        raise ValueError("回测样本不足 (有效交易日 < 60)")
    first_index = dates.index(in_range[0])
    history = max(required_history(str(row[0]["expression"])) for row in active)
    history_start = dates[max(0, first_index - history - 2)]
    symbols = (
        panel.lazy()
        .filter(
            pl.col("trade_date").is_between(start, end)
            & (pl.col("univ_rank") <= universe_n)
        )
        .select("ts_code")
        .unique()
        .collect()["ts_code"]
        .to_list()
    )
    lazy = panel.lazy().filter(
        pl.col("trade_date").is_between(history_start, end)
    )
    fields = get_dsl_fields("ashare")
    score_columns = []
    for index, (component, _) in enumerate(active):
        raw = f"_frozen_raw_{index}"
        oriented = f"_frozen_oriented_{index}"
        score = f"_frozen_score_{index}"
        lazy = parse(str(component["expression"]), fields).apply(lazy, alias=raw)
        lazy = lazy.with_columns(
            pl.when(pl.col("univ_rank") <= universe_n)
            .then(pl.col(raw) * int(component["direction"]))
            .otherwise(None)
            .alias(oriented)
        )
        lazy = lazy.with_columns(
            (
                pl.col(oriented).rank(method="average").over("trade_date")
                / pl.col(oriented).count().over("trade_date")
            ).alias(score)
        )
        score_columns.append(score)
    composite = pl.lit(0.0)
    for score, (_, weight) in zip(score_columns, active):
        composite += pl.col(score) * weight
    forward = f"fwd_{forward_horizon}"
    if forward not in panel.columns:
        raise ValueError(f"不支持的 forward_horizon: {forward_horizon}")
    select_columns = [
        "trade_date", "ts_code", "name", "univ_rank", "raw_open",
        "raw_close", "vol", "amount", "adjustment_factor",
        "can_buy_open_proxy", "can_sell_open_proxy", forward,
    ]
    full_cross_section = (
        lazy.with_columns(composite.alias("factor"))
        .select(*select_columns, "factor")
        .cache()
    )
    frame = (
        full_cross_section
        .filter(pl.col("trade_date").is_between(start, end))
        .filter(pl.col("ts_code").is_in(symbols))
        .sort("trade_date", "ts_code")
        .collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False))
    )
    return frame, store


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _period_stats(
    rows: list[dict],
    *,
    start: date,
    end: date,
) -> dict:
    selected = [
        row
        for row in rows
        if start <= date.fromisoformat(str(row["trade_date"])) <= end
    ]
    if len(selected) < 2:
        raise ValueError(f"{start}..{end} 的回放交易日不足")
    returns = [float(row["daily_return"]) for row in selected]
    nav = []
    value = 1.0
    for daily_return in returns:
        value *= max(1e-12, 1.0 + daily_return)
        nav.append(value)
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / max(
        1,
        len(returns) - 1,
    )
    volatility = math.sqrt(max(0.0, variance))
    peak = 1.0
    max_drawdown = 0.0
    for value in nav:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, 1.0 - value / peak)
    return {
        "start": str(start),
        "end": str(end),
        "sessions": len(selected),
        "total_return": round(nav[-1] - 1.0, 8),
        "annual_return": round(
            nav[-1] ** (252.0 / len(selected)) - 1.0,
            8,
        ),
        "annual_volatility": round(volatility * math.sqrt(252.0), 8),
        "sharpe": round(
            mean / volatility * math.sqrt(252.0)
            if volatility > 1e-12
            else 0.0,
            6,
        ),
        "max_drawdown": round(max_drawdown, 8),
        "mean_daily_turnover": round(
            sum(float(row["turnover"]) for row in selected) / len(selected),
            8,
        ),
        "fills": sum(int(row["fills"]) for row in selected),
    }


def _write_html(path: Path, report: dict) -> None:
    rows = []
    for strategy in report["strategies"]:
        scenario = strategy["execution_scenarios"]["15"]
        private = strategy["period_metrics_15bps"]["2023_latest"]
        ic = strategy["ic_by_period"]["2023_latest"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(strategy['label'])}</td>"
            f"<td>{strategy['direction']:+d}</td>"
            f"<td>{scenario['ann_return']:.2%}</td>"
            f"<td>{scenario['sharpe']:.3f}</td>"
            f"<td>{scenario['max_drawdown']:.2%}</td>"
            f"<td>{private['annual_return']:.2%}</td>"
            f"<td>{private['sharpe']:.3f}</td>"
            f"<td>{ic['rank_ic_mean']:.4f}</td>"
            f"<td>{ic['rank_icir']:.3f}</td>"
            "</tr>"
        )
    yearly_rows = []
    for strategy in report["strategies"]:
        for year, metrics in strategy["yearly_metrics_15bps"].items():
            yearly_rows.append(
                "<tr>"
                f"<td>{html.escape(strategy['label'])}</td>"
                f"<td>{year}</td>"
                f"<td>{metrics['total_return']:.2%}</td>"
                f"<td>{metrics['annual_return']:.2%}</td>"
                f"<td>{metrics['sharpe']:.3f}</td>"
                f"<td>{metrics['max_drawdown']:.2%}</td>"
                f"<td>{metrics['mean_daily_turnover']:.3f}</td>"
                "</tr>"
            )
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>A股冻结多因子验证</title>
<style>body{{font:14px -apple-system,BlinkMacSystemFont,sans-serif;background:#0b1118;color:#dce6f2;margin:32px}}table{{border-collapse:collapse;width:100%}}th,td{{padding:10px;border-bottom:1px solid #2b3745;text-align:right}}th:first-child,td:first-child{{text-align:left}}code{{color:#8fc7ff}}.warn{{color:#ffc46b}}</style></head>
<body><h1>A股冻结多因子验证</h1>
<p class="warn">NON-PIT 研究诊断；权重先冻结，随后一次性读取 2023–最新日，不代表实盘批准。</p>
<p>区间：{report['requested_start']} – {report['latest_panel_date']}；调仓：{report['rebalance_every']} 个交易日；费用：万2免5 + 15bps 滑点列。</p>
<table><thead><tr><th>策略</th><th>方向</th><th>全区间年化</th><th>全区间 Sharpe</th><th>最大回撤</th><th>2023+年化</th><th>2023+ Sharpe</th><th>2023+ RankIC</th><th>RankICIR</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<h2>15bps 年度表现</h2>
<table><thead><tr><th>策略</th><th>年份</th><th>区间收益</th><th>年化收益</th><th>Sharpe</th><th>最大回撤</th><th>日均换手</th></tr></thead>
<tbody>{''.join(yearly_rows)}</tbody></table>
<p>权重：<code>{html.escape(json.dumps(report['frozen_weights'], ensure_ascii=False))}</code></p>
</body></html>"""
    path.write_text(document, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-search", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end")
    parser.add_argument("--rebalance-every", type=int, default=20)
    parser.add_argument("--universe-n", type=int, default=500)
    parser.add_argument("--top-fraction", type=float, default=0.20)
    parser.add_argument("--slippage-bps", default="0,5,15")
    parser.add_argument(
        "--composite-only",
        action="store_true",
        help="只回放冻结组合，不重复回放每个单因子",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    weight_path = Path(args.weight_search).resolve()
    frozen = json.loads(weight_path.read_text(encoding="utf-8"))
    if frozen.get("holdout_or_vault_read") is not False:
        raise ValueError("权重搜索产物没有声明 holdout_or_vault_read=false")
    components = frozen.get("components") or []
    weights = (frozen.get("best") or {}).get("weights") or []
    if not 2 <= len(components) <= 5 or len(weights) != len(components):
        raise ValueError("本验证器要求 2–5 个冻结组件，且权重数量必须一致")
    if not math.isclose(sum(float(value) for value in weights), 1.0, abs_tol=1e-9):
        raise ValueError("冻结权重之和必须为 1")

    store = PanelStore.get(ASHARE_PANEL_GLOB, "ashare")
    store.ensure_loaded()
    latest = date.fromisoformat(
        args.end or str(store.summary()["date_max"])
    )
    start = date.fromisoformat(args.start)
    slippage = tuple(
        float(value.strip())
        for value in args.slippage_bps.split(",")
        if value.strip()
    )
    if 15.0 not in slippage:
        raise ValueError("验证场景必须包含 15bps")

    component_strategies = [
        {
            "key": f"factor_{int(component['factor_id'])}",
            "label": f"{component['name']} ({int(component['factor_id'])})",
            "expression": str(component["expression"]),
            "direction": int(component["direction"]),
            "weight": float(weights[index]),
            "kind": "single_component",
        }
        for index, component in enumerate(components)
        if float(weights[index]) > 1e-12
    ]
    composite_strategy = {
        "key": "frozen_composite",
        "label": "冻结多因子组合",
        "expression": str(frozen["best_composite_expression"]),
        "direction": 1,
        "weight": 1.0,
        "kind": "frozen_composite",
    }
    strategies = (
        [composite_strategy]
        if args.composite_only
        else [*component_strategies, composite_strategy]
    )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    period_bounds = {
        "2020_2022_training_overlap": (date(2020, 1, 1), date(2022, 12, 31)),
        "2023_2024_holdout": (date(2023, 1, 1), date(2024, 12, 31)),
        "2025_latest_vault": (date(2025, 1, 1), latest),
        "2023_latest": (date(2023, 1, 1), latest),
        "full": (start, latest),
    }
    report_rows = []
    csv_rows = []
    for index, strategy in enumerate(strategies, start=1):
        print(
            f"[{index}/{len(strategies)}] prepare + replay {strategy['label']}",
            flush=True,
        )
        if strategy["kind"] == "frozen_composite":
            frame, _ = _prepare_frozen_composite_frame(
                components=components,
                weights=[float(value) for value in weights],
                universe_n=args.universe_n,
                start=start,
                end=latest,
                forward_horizon=args.rebalance_every,
            )
        else:
            frame, _ = _prepare_backtest_frame(
                expression=strategy["expression"],
                universe_n=args.universe_n,
                start=str(start),
                end=str(latest),
                panel_glob=ASHARE_PANEL_GLOB,
                market="ashare",
                forward_horizon=args.rebalance_every,
            )
        spec = BatchBacktestSpec(
            market="ashare",
            mode="long_only",
            universe_n=args.universe_n,
            horizon=args.rebalance_every,
            top_fraction=args.top_fraction,
            initial_capital=10_000_000.0,
            rebalance_every=args.rebalance_every,
            holdout_start=str(start),
            holdout_end=str(latest),
            slippage_bps=slippage,
            fee_profile="ashare_wan2_no_min_v1",
        )
        scenarios = run_cost_scenarios(
            frame,
            direction=strategy["direction"],
            spec=spec,
            capture_detail=True,
        )
        execution_summaries = {}
        period_metrics = {}
        yearly_metrics = {}
        manifests = {}
        for bps, scenario in scenarios.items():
            result = scenario["result"]
            manifests[bps] = _write_artifacts(
                result,
                output_dir / strategy["key"] / f"bps_{bps.replace('.', '_')}",
            )
            execution_summaries[bps] = {
                key: value
                for key, value in scenario.items()
                if key != "result"
            }
            if bps == "15":
                period_metrics = {
                    name: _period_stats(
                        result["daily_steps"],
                        start=period_start,
                        end=period_end,
                    )
                    for name, (period_start, period_end) in period_bounds.items()
                }
                yearly_metrics = {
                    str(year): _period_stats(
                        result["daily_steps"],
                        start=max(start, date(year, 1, 1)),
                        end=min(latest, date(year, 12, 31)),
                    )
                    for year in range(start.year, latest.year + 1)
                    if max(start, date(year, 1, 1))
                    < min(latest, date(year, 12, 31))
                }

        ic_by_period = {
            name: information_coefficients(
                frame,
                start=str(period_start),
                end=str(period_end),
                horizon=args.rebalance_every,
                universe_n=args.universe_n,
                direction=strategy["direction"],
            )
            for name, (period_start, period_end) in period_bounds.items()
        }
        row = {
            **strategy,
            "execution_scenarios": execution_summaries,
            "period_metrics_15bps": period_metrics,
            "yearly_metrics_15bps": yearly_metrics,
            "ic_by_period": ic_by_period,
            "manifests": manifests,
        }
        report_rows.append(row)
        full_15 = execution_summaries["15"]
        private_15 = period_metrics["2023_latest"]
        private_ic = ic_by_period["2023_latest"]
        csv_rows.append({
            "strategy": strategy["label"],
            "direction": strategy["direction"],
            "weight": strategy["weight"],
            "full_ann_return_15bps": full_15["ann_return"],
            "full_sharpe_15bps": full_15["sharpe"],
            "full_max_drawdown_15bps": full_15["max_drawdown"],
            "private_2023_latest_ann_return_15bps": private_15["annual_return"],
            "private_2023_latest_sharpe_15bps": private_15["sharpe"],
            "private_2023_latest_max_drawdown_15bps": private_15["max_drawdown"],
            "private_2023_latest_rank_ic": private_ic["rank_ic_mean"],
            "private_2023_latest_rank_icir": private_ic["rank_icir"],
        })
        del frame, scenarios

    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol": PROTOCOL,
        "policy_label": "FULL_SAMPLE_NON_PIT_DIAGNOSTIC",
        "production_eligible": False,
        "weights_frozen_before_private_validation": True,
        "private_layers_used_for_retuning": False,
        "weight_search_path": str(weight_path),
        "weight_search_sha256": _hash_file(weight_path),
        "weight_search_protocol": frozen.get("protocol"),
        "frozen_weights": [float(value) for value in weights],
        "requested_start": str(start),
        "latest_panel_date": str(latest),
        "rebalance_every": args.rebalance_every,
        "universe_n": args.universe_n,
        "top_fraction": args.top_fraction,
        "slippage_bps": list(slippage),
        "fee_profile": "ashare_wan2_no_min_v1",
        "execution": {
            "signal": "t_close",
            "fill": "t_plus_1_raw_open",
            "valuation": "session_raw_close",
            "max_volume_participation": 0.05,
        },
        "composite_semantics": (
            "direction_adjust_each_component_then_rerank_inside_TopN_"
            "then_apply_frozen_weights"
        ),
        "panel": store.summary(),
        "strategies": report_rows,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    _write_html(output_dir / "report.html", report)

    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_index.json":
            files.append({
                "path": str(path.relative_to(output_dir)),
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            })
    (output_dir / "artifact_index.json").write_text(
        json.dumps(
            {
                "protocol": PROTOCOL,
                "all_integrity_checks_pass": all(
                    scenario["integrity"]["all_pass"]
                    for strategy in report_rows
                    for scenario in strategy["execution_scenarios"].values()
                ),
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "latest_panel_date": str(latest),
        "frozen_weights": weights,
        "summary": csv_rows,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
