#!/usr/bin/env python3
"""Find low-complexity A-share and US factor-weight candidates for the next period.

The component pools and directions are frozen from the audited independent
return-source reports.  Weight search sees 2020-2022, selection sees 2023-2024,
and 2025-latest is opened once as a recent rating window.  The recent window is
never fed back into the optimiser, so the report distinguishes a deployable
hypothesis from a retrospectively best recent fit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.combination_lab import run_combination_search, validate_lab_spec  # noqa: E402
from app.factor_tools import build_combination_expression  # noqa: E402


PROTOCOL = "recent_factor_weight_candidates_nested_v1"


ASHARE_COMPONENTS = [
    {
        "key": "A0001_large_order_participation",
        "name": "大单参与度",
        "expression": "zscore(winsor_mad(ts_mean(((buy_lg_amount + sell_lg_amount) / amount), 120), 5))",
        "direction": 1,
        "mechanism": "capital_flow",
    },
    {
        "key": "A0003_volume_ratio_persistence",
        "name": "量比持续性",
        "expression": "rank((-(ts_mean(volume_ratio, 120) / ts_max(volume_ratio, 120))))",
        "direction": -1,
        "mechanism": "liquidity",
    },
    {
        "key": "A0004_float_market_cap_variation",
        "name": "流通市值尺度与变化",
        "expression": "rank((-winsor_mad(ts_std(circ_mv, 60), 5)))",
        "direction": -1,
        "mechanism": "size",
    },
    {
        "key": "A0012_low_sales_valuation",
        "name": "低市销率",
        # The source audit used the upstream alias `ps`; the local A-share DSL
        # whitelist and canonical panel use the equivalent field `ps_ttm`.
        "expression": "rank((-ps_ttm))",
        "direction": 1,
        "mechanism": "valuation",
    },
    {
        "key": "A0015_intraday_volume_reversal",
        "name": "日内价量反转",
        "expression": "-rank(ts_sum((close - open) / close * vol, 120))",
        "direction": 1,
        "mechanism": "gap_intraday",
    },
    {
        "key": "A0021_medium_term_reversal",
        "name": "中期反转",
        "expression": "zscore(winsor_mad((ts_mean(close, 20) / ts_mean(close, 60)), 5))",
        "direction": -1,
        "mechanism": "momentum_reversal",
    },
    {
        "key": "A0026_normalized_main_flow",
        "name": "标准化净资金流",
        "expression": "zscore(winsor_mad(ts_mean((net_mf_amount / ts_std(net_mf_amount, 120)), 120), 5))",
        "direction": 1,
        "mechanism": "capital_flow",
    },
]


US_COMPONENTS = [
    {
        "key": "U0002_price_impact_liquidity",
        "name": "价格冲击流动性",
        "expression": "rank(winsor_mad(ts_mean(((high - low) / amount), 120), 5))",
        "direction": -1,
        "mechanism": "price_impact_liquidity",
    },
    {
        "key": "U0005_low_absolute_range",
        "name": "低绝对振幅",
        "expression": "rank((-ts_mean((high - low), 60)))",
        "direction": 1,
        "mechanism": "low_absolute_range",
    },
    {
        "key": "U0010_momentum_x_volatility",
        "name": "长周期动量乘波动",
        "expression": "zscore(winsor_mad(((-returns(close, 252)) * (-ts_std(returns(close, 1), 60))), 5))",
        "direction": 1,
        "mechanism": "long_horizon_momentum",
    },
    {
        "key": "U0013_residual_momentum",
        "name": "残差动量",
        "expression": "winsor((ts_delta(log(close), 250) - ts_delta(log(close), 20)) / ts_std(ts_delta(log(close), 1), 250))",
        "direction": 1,
        "mechanism": "long_horizon_momentum",
    },
    {
        "key": "U0001_share_volume_liquidity",
        "name": "股数口径流动性",
        "expression": "rank(ts_mean((close / ts_mean(amount, 50)), 200))",
        "direction": -1,
        "mechanism": "share_volume_liquidity",
    },
    {
        "key": "U0006_short_reversal",
        "name": "短期反转",
        "expression": "zscore(winsor_mad((-ts_mean((returns(close, 1) * delay(returns(close, 1), 1)), 120)), 5))",
        "direction": 1,
        "mechanism": "short_reversal",
    },
    {
        "key": "U0012_price_relationship",
        "name": "价格关系",
        "expression": "zscore(ts_corr(close, ts_mean(close, 60), 60))",
        "direction": 1,
        "mechanism": "price_relationship",
    },
]


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _primary_metrics(replay: dict | None, mode: str, cost_bps: float) -> dict | None:
    if not replay:
        return None
    key = str(int(cost_bps)) if float(cost_bps).is_integer() else str(cost_bps)
    vector = replay["vector"][key]
    if mode == "long_only":
        ranking = {
            "basis": "active_vs_equal_weight_universe",
            "annualized_return": vector["active_ann_return"],
            "sharpe": vector["active_sharpe"],
            "max_drawdown": vector["active_max_drawdown"],
        }
    else:
        ranking = {
            "basis": "net_long_short",
            "annualized_return": vector["ann_return"],
            "sharpe": vector["sharpe"],
            "max_drawdown": vector["max_drawdown"],
        }
    return {
        **ranking,
        "raw_long_annualized_return": vector.get("ann_return"),
        "raw_long_sharpe": vector.get("sharpe"),
        "rank_ic_mean": replay["ic"].get("rank_ic_mean"),
        "rank_icir": replay["ic"].get("rank_icir"),
        "event_sharpe": replay["event"].get("sharpe"),
        "event_max_drawdown": replay["event"].get("max_drawdown"),
        "event_integrity": (replay["event"].get("integrity") or {}).get("all_pass"),
    }


def _case(name: str, market: str, mode: str, components: list[dict], output: Path) -> dict:
    cost = 20.0 if market == "ashare" else 15.0
    spec = validate_lab_spec({
        "name": name,
        "experiment_id": 1,
        "search_mode": "programmatic",
        "market": market,
        "portfolio_mode": mode,
        "components": components,
        "min_factors": 3,
        "max_factors": 7,
        "min_mechanisms": 3,
        "coarse_step": 0.10,
        "min_weight": 0.10,
        "max_weight": 0.40,
        "max_mechanism_weight": 0.45,
        "max_pair_correlation": 0.80,
        "path_budget": 5000,
        "validation_budget": 350,
        "top_k": 15,
        "universe_n": 500,
        "top_fraction": 0.20,
        "horizon": 5,
        "cost_bps": cost,
        "stress_cost_bps": 50.0 if market == "ashare" else 40.0,
        "borrow_cost_bps_annual": 300.0 if mode == "long_short" else 0.0,
        "train_start": "2020-01-01",
        "train_end": "2022-12-31",
        "validation_start": "2023-01-01",
        "validation_end": "2024-12-31",
        "rating_start": "2025-01-01",
        "rating_end": "2026-12-31",
    })

    def progress(row: dict) -> None:
        completed, total = row.get("completed", 0), row.get("total", 0)
        print(f"[{name}] {row.get('stage')}: {completed}/{total} {row.get('message', '')}", flush=True)

    result = run_combination_search(
        spec,
        progress_callback=progress,
        artifact_root=output / "artifacts" / name,
    )
    _write_json(output / f"{name}.json", {"spec": spec, "result": result})
    winner = result.get("winner")
    expression_artifact = None
    if winner:
        active_components = [
            {**component, "weight": weight}
            for component, weight in zip(components, winner["weights"])
            if float(weight) > 1e-12
        ]
        expression_artifact = build_combination_expression({
            "market": market,
            "normalization": "rank",
            "omit_common_scale": False,
            "components": active_components,
        })
    return {
        "case": name,
        "market": market,
        "portfolio_mode": mode,
        "decision": result["decision"],
        "effective_windows": result["effective_windows"],
        "weights": dict(zip((row["key"] for row in components), winner["weights"])) if winner else None,
        "dsl_expression": expression_artifact["expression"] if expression_artifact else None,
        "dsl_expression_length": expression_artifact["length"] if expression_artifact else None,
        "required_history": expression_artifact["required_history"] if expression_artifact else None,
        "component_snapshot_hash": expression_artifact["snapshot_hash"] if expression_artifact else None,
        "validation_score": winner.get("validation_score") if winner else None,
        "validation": winner.get("validation") if winner else None,
        "rating_rules": winner.get("rating_rules") if winner else None,
        "recent_rating": _primary_metrics(result.get("rating"), mode, cost),
        "equal_recent_rating": _primary_metrics(result.get("equal_all_components_benchmark"), mode, cost),
        "elapsed_seconds": result["elapsed_seconds"],
        "result_file": f"{name}.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="var/reports/recent-factor-weight-candidates-20260822",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=("ashare_long_only", "us_long_only", "us_long_short"),
        default=("ashare_long_only", "us_long_only", "us_long_short"),
    )
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = {
        "ashare_long_only": ("ashare", "long_only", ASHARE_COMPONENTS),
        "us_long_only": ("us", "long_only", US_COMPONENTS),
        "us_long_short": ("us", "long_short", US_COMPONENTS),
    }
    progress_path = output / "progress.json"
    rows: list[dict] = []
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        rows = [
            row for row in list(previous.get("completed") or [])
            if row.get("case") not in set(args.cases)
        ]
    for name in args.cases:
        market, mode, components = cases[name]
        rows.append(_case(name, market, mode, components, output))
        _write_json(progress_path, {"protocol": PROTOCOL, "completed": rows})

    order = {name: index for index, name in enumerate(cases)}
    rows.sort(key=lambda row: order.get(str(row.get("case")), len(order)))

    summary = {
        "protocol": PROTOCOL,
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_policy": {
            "component_and_direction_source": "audited_independent_return_source_reports",
            "train": "2020-01-01~2022-12-31",
            "validation": "2023-01-01~2024-12-31",
            "recent_rating": "2025-01-01~latest_available",
            "recent_rating_used_for_weight_fit": False,
            "interpretation": "candidate-selection-contaminated research diagnostic; next-period hypothesis, not live approval",
        },
        "cases": rows,
    }
    _write_json(output / "summary.json", summary)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case", "market", "portfolio_mode", "decision", "weights",
                "validation_score", "recent_rating", "equal_recent_rating",
                "elapsed_seconds", "result_file",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in writer.fieldnames})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
