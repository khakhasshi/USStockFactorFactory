#!/usr/bin/env python3
"""Create stability and attribution artifacts from the A-share Top-100 audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


WINDOWS = (
    ("2020_2022", "2020-01-01", "2022-12-31"),
    ("2023_2024", "2023-01-01", "2024-12-31"),
    ("2025_latest", "2025-01-01", "2026-12-31"),
)
SELECTED_SOURCES = {
    "U0001": ("大单参与度", "大单买卖总额占成交额的长期比例，代理机构参与和成交结构"),
    "U0003": ("量比持续性", "量比均值相对峰值的持续程度，代理成交活跃度稳定性"),
    "U0004": ("流通市值尺度/变化", "流通市值时间波动，需继续拆分规模与市值变化暴露"),
    "U0012": ("低市销率", "低PS估值，四榜中最清晰的基本面估值来源"),
    "U0015": ("日内价量反转", "长窗开收盘价差乘成交量的反向暴露"),
    "U0021": ("中期反转", "20日均价相对60日均价的反向暴露"),
    "U0026": ("标准化净资金流", "净主力资金相对自身长期波动的持续性"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _metrics(row: dict, start: str, end: str) -> dict[str, float | int]:
    values = np.asarray([
        value for date, value in zip(row["return_dates"], row["returns"])
        if start <= date <= end
    ], dtype=np.float64)
    periods_per_year = 252.0 / 5.0
    ann_return = float(values.mean() * periods_per_year)
    volatility = float(values.std(ddof=1))
    sharpe = float(values.mean() / volatility * math.sqrt(periods_per_year)) if volatility > 0 else 0.0
    return {"periods": len(values), "ann_return": ann_return, "sharpe": sharpe}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_dir", type=Path)
    args = parser.parse_args()
    root = args.audit_dir.resolve()
    paths = {
        row["task_id"]: row
        for row in map(json.loads, (root / "return_paths.jsonl").open(encoding="utf-8"))
    }
    assignments = _read_csv(root / "global_unsigned_assignments.csv")
    clusters = _read_csv(root / "global_unsigned_clusters.csv")
    memberships = _read_csv(root / "frozen_top100_memberships.csv")
    assignments_by_cluster: dict[str, list[dict]] = defaultdict(list)
    for row in assignments:
        assignments_by_cluster[row["cluster"]].append(row)

    stability_rows: list[dict] = []
    for cluster in clusters:
        members = assignments_by_cluster[cluster["cluster"]]
        representative = next(row for row in members if row["is_representative"] == "True")
        path = paths[representative["task_id"]]
        segments = {name: _metrics(path, start, end) for name, start, end in WINDOWS}
        yearly = {
            year: _metrics(path, f"{year}-01-01", f"{year}-12-31")
            for year in range(2020, 2027)
        }
        positive_regimes = sum(value["ann_return"] > 0 for value in segments.values())
        positive_years = sum(value["ann_return"] > 0 for value in yearly.values())
        full_ann_return = float(representative["ann_return"])
        full_sharpe = float(representative["sharpe"])
        size = int(cluster["size"])
        if size >= 2 and full_sharpe >= 0.2 and positive_regimes == len(WINDOWS):
            stability_class = "recurrent_regime_stable"
        elif full_ann_return > 0:
            stability_class = "positive_but_regime_fragile"
        else:
            stability_class = "negative_or_failed_transfer"
        row = {
            "cluster": cluster["cluster"],
            "size": size,
            "source_lists": cluster["source_lists"],
            "source_list_count": len(cluster["source_lists"].split("|")),
            "crosses_internal_external": cluster["crosses_internal_external"],
            "direction": representative["direction"],
            "mechanism_family": representative["mechanism_family"],
            "full_ann_return": full_ann_return,
            "full_sharpe": full_sharpe,
            "full_max_drawdown": float(representative["max_drawdown"]),
            "positive_regimes": positive_regimes,
            "positive_years_2020_2026": positive_years,
            "stability_class": stability_class,
            "expression": representative["expression"],
        }
        for name, _, _ in WINDOWS:
            row[f"{name}_ann_return"] = segments[name]["ann_return"]
            row[f"{name}_sharpe"] = segments[name]["sharpe"]
        stability_rows.append(row)
    stability_rows.sort(key=lambda row: (-float(row["full_sharpe"]), row["cluster"]))
    _write_csv(root / "cluster_stability.csv", stability_rows)

    list_rows: list[dict] = []
    list_order = list(dict.fromkeys(row["list_id"] for row in memberships))
    for list_id in list_order:
        selected = [row for row in memberships if row["list_id"] == list_id]
        output = {"list_id": list_id, "ranked_slots": len(selected)}
        full_sharpes = []
        full_returns = []
        for member in selected:
            scenario = paths[member["task_id"]]["scenario"]
            full_sharpes.append(float(scenario["active_sharpe"]))
            full_returns.append(float(scenario["active_ann_return"]))
        output.update({
            "full_positive_ann_return": sum(value > 0 for value in full_returns),
            "full_positive_sharpe": sum(value > 0 for value in full_sharpes),
            "full_median_ann_return": float(np.median(full_returns)),
            "full_median_sharpe": float(np.median(full_sharpes)),
        })
        for name, start, end in WINDOWS:
            values = [_metrics(paths[member["task_id"]], start, end)["sharpe"] for member in selected]
            output[f"{name}_positive_sharpe"] = sum(value > 0 for value in values)
            output[f"{name}_median_sharpe"] = float(np.median(values))
        list_rows.append(output)
    _write_csv(root / "list_regime_stability.csv", list_rows)

    selected_rows = [row for row in stability_rows if row["cluster"] in SELECTED_SOURCES]
    selected_rows.sort(key=lambda row: list(SELECTED_SOURCES).index(row["cluster"]))
    selected_paths = []
    selected_dates = []
    for row in selected_rows:
        representative = next(
            item for item in assignments_by_cluster[row["cluster"]]
            if item["is_representative"] == "True"
        )
        path = paths[representative["task_id"]]
        selected_paths.append(dict(zip(path["return_dates"], path["returns"])))
        selected_dates.append(set(path["return_dates"]))
        row["research_sleeve"], row["economic_attribution"] = SELECTED_SOURCES[row["cluster"]]
    common_dates = sorted(set.intersection(*selected_dates))
    matrix = np.asarray([[series[date] for date in common_dates] for series in selected_paths])
    correlations = np.corrcoef(matrix)
    correlation_rows = []
    for left_index, left in enumerate(selected_rows):
        for right_index, right in enumerate(selected_rows):
            correlation_rows.append({
                "left_cluster": left["cluster"],
                "right_cluster": right["cluster"],
                "correlation": float(correlations[left_index, right_index]),
                "common_periods": len(common_dates),
            })
    _write_csv(root / "selected_source_correlations.csv", correlation_rows)
    _write_csv(root / "selected_research_sources.csv", selected_rows)

    conclusion = {
        "protocol": "ashare_four_leaderboard_top100_return_source_stability_v1",
        "source_audit": str(root),
        "classification_rule": {
            "recurrent_regime_stable": "cluster_size>=2, full_sharpe>=0.2, and positive annualized active return in all three broad regimes",
            "regimes": [name for name, _, _ in WINDOWS],
        },
        "cluster_count": len(stability_rows),
        "recurrent_regime_stable_clusters": sum(
            row["stability_class"] == "recurrent_regime_stable" for row in stability_rows
        ),
        "selected_research_sleeves": list(SELECTED_SOURCES),
        "selected_max_pair_correlation": float(np.max(correlations - np.eye(len(correlations)) * 2)),
        "policy": "RESEARCH_ONLY_FULL_WINDOW_SELECTED_NOT_UNTOUCHED_HOLDOUT",
    }
    (root / "stability_conclusion.json").write_text(
        json.dumps(conclusion, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(conclusion, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
