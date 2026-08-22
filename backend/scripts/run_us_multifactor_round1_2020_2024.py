#!/usr/bin/env python3
"""Run the frozen first-round US multi-factor ablation campaign, 2020-2024.

Each component is evaluated on the full cross-section, oriented using the
already-frozen leaderboard direction, re-ranked inside the rolling top-500
liquidity universe, and only then combined.  Seven fixed variants are tested
in both long-only and long-short modes.  No direction or weight is fitted on
the requested test interval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from app.backtest.batch import (  # noqa: E402
    BatchBacktestSpec,
    information_coefficients,
    run_cost_scenarios,
)
from app.backtest.engine import _write_artifacts  # noqa: E402
from app.backtest.vector import run_vector_cost_scenarios  # noqa: E402
from app.data.panel import PanelStore  # noqa: E402
from app.dsl.engine import parse, required_history  # noqa: E402


PROTOCOL = "us_frozen_multifactor_round1_2020_2024_v1"
POLICY_LABEL = "FIXED_COMPONENT_FIXED_WEIGHT_TEST_DIAGNOSTIC"
START = date(2020, 1, 1)
END = date(2024, 12, 31)
REBALANCE_EVERY = 5
UNIVERSE_N = 500
TOP_FRACTION = 0.20


COMPONENTS = [
    {
        "key": "U0002_price_impact_liquidity",
        "cluster": "U0002",
        "direction": -1,
        "sleeve": "price_impact_liquidity",
        "expression": "rank(winsor_mad(ts_mean(((high - low) / amount), 120), 5))",
    },
    {
        "key": "U0005_low_absolute_range",
        "cluster": "U0005",
        "direction": 1,
        "sleeve": "low_absolute_range",
        "expression": "rank((-ts_mean((high - low), 60)))",
    },
    {
        "key": "U0010_momentum_x_volatility",
        "cluster": "U0010",
        "direction": 1,
        "sleeve": "long_horizon_momentum",
        "expression": "zscore(winsor_mad(((-returns(close, 252)) * (-ts_std(returns(close, 1), 60))), 5))",
    },
    {
        "key": "U0013_residual_momentum",
        "cluster": "U0013",
        "direction": 1,
        "sleeve": "long_horizon_momentum",
        "expression": "winsor((ts_delta(log(close), 250) - ts_delta(log(close), 20)) / ts_std(ts_delta(log(close), 1), 250))",
    },
    {
        "key": "U0001_share_volume_liquidity",
        "cluster": "U0001",
        "direction": -1,
        "sleeve": "share_volume_liquidity",
        "expression": "rank(ts_mean((close / ts_mean(amount, 50)), 200))",
    },
    {
        "key": "U0006_short_reversal",
        "cluster": "U0006",
        "direction": 1,
        "sleeve": "short_reversal",
        "expression": "zscore(winsor_mad((-ts_mean((returns(close, 1) * delay(returns(close, 1), 1)), 120)), 5))",
    },
    {
        "key": "U0012_price_relationship",
        "cluster": "U0012",
        "direction": 1,
        "sleeve": "price_relationship",
        "expression": "zscore(ts_corr(close, ts_mean(close, 60), 60))",
    },
    {
        "key": "U0003_low_nominal_price_challenger",
        "cluster": "U0003",
        "direction": 1,
        "sleeve": "share_volume_liquidity",
        "expression": "rank((-ts_mean(close, 252)))",
    },
]


def _source_equal_weights(
    *,
    exclude: set[str] | None = None,
    replace: dict[str, str] | None = None,
    momentum: str = "both",
) -> dict[str, float]:
    exclude = set(exclude or set())
    replace = dict(replace or {})
    sleeve_members = {
        "price_impact_liquidity": ["U0002_price_impact_liquidity"],
        "low_absolute_range": ["U0005_low_absolute_range"],
        "long_horizon_momentum": (
            ["U0010_momentum_x_volatility", "U0013_residual_momentum"]
            if momentum == "both"
            else [
                "U0010_momentum_x_volatility"
                if momentum == "U0010"
                else "U0013_residual_momentum"
            ]
        ),
        "share_volume_liquidity": ["U0001_share_volume_liquidity"],
        "short_reversal": ["U0006_short_reversal"],
        "price_relationship": ["U0012_price_relationship"],
    }
    for sleeve, members in sleeve_members.items():
        sleeve_members[sleeve] = [replace.get(member, member) for member in members]
    sleeve_members = {
        sleeve: [member for member in members if member not in exclude]
        for sleeve, members in sleeve_members.items()
    }
    sleeve_members = {sleeve: members for sleeve, members in sleeve_members.items() if members}
    sleeve_weight = 1.0 / len(sleeve_members)
    weights: dict[str, float] = {}
    for members in sleeve_members.values():
        for member in members:
            weights[member] = sleeve_weight / len(members)
    return weights


VARIANTS = [
    {
        "key": "baseline_source_equal",
        "label": "基准：7因子、6收益袖套等权",
        "weights": _source_equal_weights(),
    },
    {
        "key": "drop_short_reversal",
        "label": "消融：删除短期反转U0006",
        "weights": _source_equal_weights(exclude={"U0006_short_reversal"}),
    },
    {
        "key": "drop_price_relationship",
        "label": "消融：删除价格关系U0012",
        "weights": _source_equal_weights(exclude={"U0012_price_relationship"}),
    },
    {
        "key": "replace_liquidity_with_low_price",
        "label": "替换：U0001换成低名义股价U0003",
        "weights": _source_equal_weights(
            replace={"U0001_share_volume_liquidity": "U0003_low_nominal_price_challenger"}
        ),
    },
    {
        "key": "momentum_u0010_only",
        "label": "动量袖套：仅U0010",
        "weights": _source_equal_weights(momentum="U0010"),
    },
    {
        "key": "momentum_u0013_only",
        "label": "动量袖套：仅U0013",
        "weights": _source_equal_weights(momentum="U0013"),
    },
    {
        "key": "naive_equal_7_factors",
        "label": "对照：7因子简单等权",
        "weights": {
            component["key"]: 1.0 / 7.0
            for component in COMPONENTS
            if component["cluster"] != "U0003"
        },
    },
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _period_stats(period_rows: list[dict], return_key: str) -> dict:
    returns = [float(row[return_key]) for row in period_rows]
    if len(returns) < 2:
        return {
            "periods": len(returns), "annualized_return": 0.0, "sharpe": 0.0,
            "max_drawdown": 0.0, "final_nav": 1.0,
        }
    nav = 1.0
    peak = 1.0
    max_drawdown = 0.0
    for value in returns:
        nav *= max(1e-12, 1.0 + value)
        peak = max(peak, nav)
        max_drawdown = max(max_drawdown, 1.0 - nav / peak)
    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / (len(returns) - 1)
    std = math.sqrt(max(0.0, variance))
    periods_per_year = 252.0 / REBALANCE_EVERY
    return {
        "periods": len(returns),
        "annualized_return": round(nav ** (periods_per_year / len(returns)) - 1.0, 8),
        "sharpe": round(mean / std * math.sqrt(periods_per_year) if std > 1e-12 else 0.0, 6),
        "max_drawdown": round(max_drawdown, 8),
        "final_nav": round(nav, 8),
    }


def _period_slice(period_rows: list[dict], start: date, end: date) -> list[dict]:
    return [
        row for row in period_rows
        if start <= date.fromisoformat(str(row["signal_date"])) <= end
    ]


def _materialize_components(
    *, panel_glob: str, dsl_fields: list[str]
) -> tuple[pl.DataFrame, dict]:
    store = PanelStore.get(panel_glob, "us", factor_fields=dsl_fields)
    panel, dates, _, _ = store.read_snapshot()
    in_range = [value for value in dates if START <= value <= END]
    if len(in_range) < 60:
        raise ValueError("2020-2024有效交易日不足")
    history = max(required_history(component["expression"]) for component in COMPONENTS)
    first_index = dates.index(in_range[0])
    history_start = dates[max(0, first_index - history - 2)]
    symbols = (
        panel.lazy()
        .filter(pl.col("trade_date").is_between(START, END) & (pl.col("univ_rank") <= UNIVERSE_N))
        .select("ts_code").unique().collect()["ts_code"].to_list()
    )
    lazy = panel.lazy().filter(pl.col("trade_date").is_between(history_start, END))
    score_columns: list[str] = []
    for index, component in enumerate(COMPONENTS):
        raw = f"_round1_raw_{index}"
        oriented = f"_round1_oriented_{index}"
        score = f"component_{index}"
        lazy = parse(component["expression"], dsl_fields).apply(lazy, alias=raw)
        lazy = lazy.with_columns(
            pl.when(pl.col("univ_rank") <= UNIVERSE_N)
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
    forward = f"fwd_{REBALANCE_EVERY}"
    columns = [
        "trade_date", "ts_code", "name", "univ_rank", "raw_open", "raw_close",
        "vol", "amount", "adjustment_factor", "can_buy_open_proxy",
        "can_sell_open_proxy", forward, *score_columns,
    ]
    frame = (
        lazy.select(columns).cache()
        .filter(pl.col("trade_date").is_between(START, END))
        .filter(pl.col("ts_code").is_in(symbols))
        .sort("trade_date", "ts_code")
        .collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False))
    )
    return frame, {
        "panel_summary": store.summary(),
        "history_start": str(history_start),
        "rows": frame.height,
        "sessions": frame["trade_date"].n_unique(),
        "symbols": frame["ts_code"].n_unique(),
    }


def _composite_frame(frame: pl.DataFrame, weights: dict[str, float]) -> pl.DataFrame:
    key_to_index = {component["key"]: index for index, component in enumerate(COMPONENTS)}
    if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-9):
        raise ValueError("组合权重和必须为1")
    expression = pl.lit(0.0)
    for key, weight in weights.items():
        expression += pl.col(f"component_{key_to_index[key]}") * float(weight)
    return frame.with_columns(expression.alias("factor"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-protocol", type=Path, default=Path(
        "var/reports/external-joint-ashare-us-to-us-long-only-vector-2020-latest-20260821/protocol.json"
    ))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_protocol = json.loads(args.source_protocol.read_text(encoding="utf-8"))
    panel_glob = str(source_protocol["panel_glob"])
    dsl_fields = list(source_protocol["dsl_fields"])
    protocol = {
        "protocol": PROTOCOL,
        "policy_label": POLICY_LABEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_window": {"start": str(START), "end": str(END)},
        "direction_selection": "frozen_before_round1_from_four_leaderboard_audit",
        "weight_selection": "seven_predeclared_fixed_ablation_variants_no_fit",
        "universe_n": UNIVERSE_N,
        "top_fraction": TOP_FRACTION,
        "rebalance_every": REBALANCE_EVERY,
        "slippage_bps": [0.0, 5.0, 15.0],
        "long_short_borrow_cost_bps_annual": 300.0,
        "panel_glob": panel_glob,
        "panel_identity": source_protocol["panel_identity_path_size_mtime_sha256"],
        "source_protocol": str(args.source_protocol.resolve()),
        "source_protocol_sha256": _sha256(args.source_protocol),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "components": COMPONENTS,
        "variants": VARIANTS,
    }
    _write_json(output / "protocol.json", protocol)
    print("[1/4] materialize 8 frozen component scores", flush=True)
    component_frame, frame_summary = _materialize_components(
        panel_glob=panel_glob, dsl_fields=dsl_fields
    )
    protocol["frame"] = frame_summary
    _write_json(output / "protocol.json", protocol)

    vector_rows: list[dict] = []
    vector_detail: list[dict] = []
    yearly_rows: list[dict] = []
    frames: dict[str, pl.DataFrame] = {}
    total = len(VARIANTS) * 2
    completed = 0
    print(f"[2/4] run {total} vector ablation paths", flush=True)
    for variant in VARIANTS:
        composite = _composite_frame(component_frame, variant["weights"])
        frames[variant["key"]] = composite
        for mode in ("long_only", "long_short"):
            completed += 1
            spec = BatchBacktestSpec(
                market="us", mode=mode, universe_n=UNIVERSE_N,
                horizon=REBALANCE_EVERY, top_fraction=TOP_FRACTION,
                initial_capital=1_000_000.0, rebalance_every=REBALANCE_EVERY,
                max_volume_participation=0.05,
                train_start=str(START), train_end=str(END),
                holdout_start=str(START), holdout_end=str(END),
                vault_start=str(END), vault_end=str(END),
                slippage_bps=(0.0, 5.0, 15.0),
                borrow_cost_bps_annual=300.0 if mode == "long_short" else 0.0,
            )
            scenarios = run_vector_cost_scenarios(
                composite, direction=1, spec=spec, capture_periods=True
            )
            periods_15 = scenarios["15"].pop("period_rows")
            for scenario in scenarios.values():
                scenario.pop("period_rows", None)
            return_key = "active_return" if mode == "long_only" else "net_return"
            period_metrics = {
                "full": _period_stats(periods_15, return_key),
                "2020_2022": _period_stats(
                    _period_slice(periods_15, date(2020, 1, 1), date(2022, 12, 31)), return_key
                ),
                "2023_2024": _period_stats(
                    _period_slice(periods_15, date(2023, 1, 1), date(2024, 12, 31)), return_key
                ),
            }
            for year in range(2020, 2025):
                metrics = _period_stats(
                    _period_slice(periods_15, date(year, 1, 1), date(year, 12, 31)), return_key
                )
                yearly_rows.append({
                    "variant": variant["key"], "mode": mode, "year": year, **metrics
                })
            ic = information_coefficients(
                composite, start=str(START), end=str(END),
                horizon=REBALANCE_EVERY, universe_n=UNIVERSE_N, direction=1,
            )
            scenario_15 = scenarios["15"]
            ranking_ann = scenario_15["active_ann_return"] if mode == "long_only" else scenario_15["ann_return"]
            ranking_sharpe = scenario_15["active_sharpe"] if mode == "long_only" else scenario_15["sharpe"]
            vector_rows.append({
                "variant": variant["key"], "label": variant["label"], "mode": mode,
                "ranking_return_basis": "active" if mode == "long_only" else "net",
                "ranking_ann_return_15bps": ranking_ann,
                "ranking_sharpe_15bps": ranking_sharpe,
                "raw_ann_return_15bps": scenario_15["ann_return"],
                "raw_sharpe_15bps": scenario_15["sharpe"],
                "max_drawdown_15bps": scenario_15["active_max_drawdown"] if mode == "long_only" else scenario_15["max_drawdown"],
                "avg_daily_turnover_15bps": scenario_15["avg_daily_turnover"],
                "total_execution_cost_15bps": scenario_15["total_execution_cost"],
                "rank_ic_mean": ic["rank_ic_mean"], "rank_icir": ic["rank_icir"],
                "subperiod_2020_2022_sharpe": period_metrics["2020_2022"]["sharpe"],
                "subperiod_2023_2024_sharpe": period_metrics["2023_2024"]["sharpe"],
                "worst_subperiod_sharpe": min(
                    period_metrics["2020_2022"]["sharpe"], period_metrics["2023_2024"]["sharpe"]
                ),
            })
            vector_detail.append({
                "variant": variant, "mode": mode, "scenarios": scenarios,
                "period_metrics_15bps": period_metrics, "ic": ic,
            })
            print(f"  [{completed}/{total}] {mode} {variant['key']} done", flush=True)

    _write_csv(output / "vector_results.csv", vector_rows)
    _write_csv(output / "yearly_results.csv", yearly_rows)
    _write_json(output / "vector_results.json", vector_detail)
    best_by_mode = {}
    for mode in ("long_only", "long_short"):
        candidates = [row for row in vector_rows if row["mode"] == mode]
        best_by_mode[mode] = max(
            candidates,
            key=lambda row: (
                float(row["worst_subperiod_sharpe"]),
                float(row["ranking_sharpe_15bps"]),
                float(row["ranking_ann_return_15bps"]),
            ),
        )["variant"]
    print(f"[3/4] robust vector winners {best_by_mode}", flush=True)

    step_jobs = []
    for mode in ("long_only", "long_short"):
        for variant_key in dict.fromkeys(["baseline_source_equal", best_by_mode[mode]]):
            step_jobs.append((mode, variant_key))
    step_rows: list[dict] = []
    for index, (mode, variant_key) in enumerate(step_jobs, start=1):
        print(f"  step [{index}/{len(step_jobs)}] {mode} {variant_key}", flush=True)
        spec = BatchBacktestSpec(
            market="us", mode=mode, universe_n=UNIVERSE_N,
            horizon=REBALANCE_EVERY, top_fraction=TOP_FRACTION,
            initial_capital=1_000_000.0, rebalance_every=REBALANCE_EVERY,
            max_volume_participation=0.05,
            holdout_start=str(START), holdout_end=str(END),
            slippage_bps=(15.0,),
            borrow_cost_bps_annual=300.0 if mode == "long_short" else 0.0,
        )
        scenario = run_cost_scenarios(
            frames[variant_key], direction=1, spec=spec, capture_detail=True
        )["15"]
        result = scenario.pop("result")
        artifact_dir = output / "step_ledgers" / mode / variant_key
        manifest = _write_artifacts(result, artifact_dir)
        step_rows.append({
            "mode": mode, "variant": variant_key, **scenario,
            "integrity_all_pass": scenario["integrity"]["all_pass"],
            "artifact_dir": str(artifact_dir),
            "artifact_manifest": manifest,
        })
    _write_json(output / "step_results.json", step_rows)

    print("[4/4] render research summary", flush=True)
    baseline = {
        row["mode"]: row for row in vector_rows if row["variant"] == "baseline_source_equal"
    }
    lines = [
        "# 美股多因子第一轮实验：2020-2024",
        "",
        f"- 协议：`{PROTOCOL}`",
        f"- 区间：{START} 至 {END}",
        "- 方向：四榜收益来源审计前冻结；本轮不重新择向。",
        "- 权重：7个预先声明的固定方案；本轮不使用测试结果拟合权重。",
        "- 排名：纯多使用基准主动收益，多空使用扣除佣金、15 BPS滑点和300 BPS借券代理后的净收益。",
        "",
        "## 向量消融结果",
        "",
        "| 模式 | 方案 | 主动/净年化 | 主动/净Sharpe | 最大回撤 | 2020-22 Sharpe | 2023-24 Sharpe | Rank ICIR |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in ("long_only", "long_short"):
        for row in sorted(
            [value for value in vector_rows if value["mode"] == mode],
            key=lambda value: float(value["worst_subperiod_sharpe"]),
            reverse=True,
        ):
            marker = " ★" if row["variant"] == best_by_mode[mode] else ""
            lines.append(
                f"| {mode} | {row['variant']}{marker} | {float(row['ranking_ann_return_15bps']):.2%} | "
                f"{float(row['ranking_sharpe_15bps']):.3f} | {float(row['max_drawdown_15bps']):.2%} | "
                f"{float(row['subperiod_2020_2022_sharpe']):.3f} | "
                f"{float(row['subperiod_2023_2024_sharpe']):.3f} | {float(row['rank_icir']):.3f} |"
            )
    lines.extend([
        "",
        "## 步进复测",
        "",
        "| 模式 | 方案 | 年化 | Sharpe | 最大回撤 | 日均换手 | 成本 | 完整性 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ])
    for row in step_rows:
        lines.append(
            f"| {row['mode']} | {row['variant']} | {float(row['ann_return']):.2%} | "
            f"{float(row['sharpe']):.3f} | {float(row['max_drawdown']):.2%} | "
            f"{float(row['avg_daily_turnover']):.3%} | {float(row['total_execution_cost']):,.0f} | "
            f"{row['integrity_all_pass']} |"
        )
    lines.extend([
        "",
        "## 冻结选择",
        "",
        f"- 纯多稳健优胜方案：`{best_by_mode['long_only']}`。",
        f"- 多空稳健优胜方案：`{best_by_mode['long_short']}`。",
        "- `★`依据2020-22与2023-24两段中较差Sharpe优先，不按全区间单一最高收益选择。",
        "- 本报告是第一轮固定方案比较；非负权重优化必须另开训练/验证协议，不能反过来改写本轮结果。",
    ])
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_json(output / "summary.json", {
        "protocol": PROTOCOL,
        "status": "complete",
        "test_window": {"start": str(START), "end": str(END)},
        "variants": len(VARIANTS),
        "modes": ["long_only", "long_short"],
        "best_by_mode": best_by_mode,
        "baseline_vector": baseline,
        "step_jobs": [{"mode": row["mode"], "variant": row["variant"]} for row in step_rows],
    })
    print(json.dumps({"output": str(output), "best_by_mode": best_by_mode}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
