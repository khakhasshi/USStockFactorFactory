"""Batch research helpers for cross-task equity-factor backtests.

The batch protocol deliberately separates three decisions:

* the factor sign is selected on META_TRAIN only;
* the leaderboard is ranked on META_HOLDOUT only;
* FACTOR_VAULT is never an input to the leaderboard score.

All portfolio returns come from :class:`StepEventBacktester`.  The compact
batch path disables statement retention but still reconciles every transient
fill online; finalists can then be replayed with full statements.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Iterable

import polars as pl

from .engine import EventBacktestConfig, StepEventBacktester


BATCH_PROTOCOL = "ashare_cross_task_factor_leaderboard_v1"
US_LONG_SHORT_BATCH_PROTOCOL = (
    "us_cross_task_long_short_factor_leaderboard_v1"
)
US_LONG_ONLY_BATCH_PROTOCOL = (
    "us_cross_task_long_only_factor_leaderboard_v1"
)
# Backward-compatible alias for callers that previously treated every US
# batch as long-short.
US_BATCH_PROTOCOL = US_LONG_SHORT_BATCH_PROTOCOL


def batch_protocol_for_market(
    market: str,
    mode: str | None = None,
) -> str:
    """Return the immutable batch protocol for a market/portfolio pair."""
    if market == "ashare":
        if mode not in {None, "long_only"}:
            raise ValueError("A股批量协议只支持 long_only")
        return BATCH_PROTOCOL
    if market == "us":
        resolved_mode = mode or "long_short"
        if resolved_mode == "long_short":
            return US_LONG_SHORT_BATCH_PROTOCOL
        if resolved_mode == "long_only":
            return US_LONG_ONLY_BATCH_PROTOCOL
        raise ValueError("美股 mode 必须是 long_only 或 long_short")
    raise ValueError("market 必须是 ashare 或 us")


@dataclass(frozen=True)
class BatchBacktestSpec:
    market: str = "ashare"
    mode: str = "long_only"
    universe_n: int = 500
    horizon: int = 5
    top_fraction: float = 0.20
    initial_capital: float = 10_000_000.0
    rebalance_every: int = 5
    max_volume_participation: float = 0.05
    train_start: str = "2020-01-01"
    train_end: str = "2022-12-31"
    holdout_start: str = "2023-01-01"
    holdout_end: str = "2024-12-31"
    vault_start: str = "2025-01-01"
    vault_end: str = "2026-08-04"
    slippage_bps: tuple[float, ...] = (0.0, 5.0, 15.0)
    borrow_cost_bps_annual: float = 0.0
    fee_profile: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def canonical_expression(expression: str) -> tuple[str, str]:
    """Return a stable AST hash and canonical text for one DSL expression."""
    tree = ast.parse(expression.strip(), mode="eval")
    dumped = ast.dump(tree)
    return (
        hashlib.sha256(dumped.encode("utf-8")).hexdigest()[:16],
        ast.unparse(tree.body),
    )


def oriented_expression_hash(expression: str, direction: int) -> str:
    """Collapse exact root-level sign aliases after direction is frozen."""
    if direction not in {-1, 1}:
        raise ValueError("direction 必须为 1 或 -1")
    node = ast.parse(expression.strip(), mode="eval").body
    sign = direction
    while isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        sign *= -1
        node = node.operand
    if sign < 0:
        node = ast.UnaryOp(op=ast.USub(), operand=node)
    tree = ast.Expression(body=node)
    ast.fix_missing_locations(tree)
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()[:16]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = (
        sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        if len(values) > 1
        else 0.0
    )
    return mean, math.sqrt(max(0.0, variance))


def information_coefficients(
    frame: pl.DataFrame,
    *,
    start: str,
    end: str,
    horizon: int,
    universe_n: int,
    direction: int = 1,
) -> dict:
    """Calculate non-overlapping daily Pearson IC and Spearman Rank IC.

    The final ``horizon + 1`` sessions are excluded so a signal inside one
    chronological layer can never consume the next layer's returns.
    """
    if direction not in {-1, 1}:
        raise ValueError("direction 必须为 1 或 -1")
    forward = f"fwd_{horizon}"
    if forward not in frame.columns:
        raise ValueError(f"回测帧缺少 {forward}")
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    period_dates = (
        frame.lazy()
        .filter(pl.col("trade_date").is_between(start_date, end_date))
        .select("trade_date")
        .unique()
        .sort("trade_date")
        .collect()["trade_date"]
        .to_list()
    )
    usable_count = max(0, len(period_dates) - horizon - 1)
    signal_dates = period_dates[:usable_count:horizon]
    if len(signal_dates) < 2:
        raise ValueError("IC 样本不足：有效非重叠日期少于 2")
    daily = (
        frame.lazy()
        .filter(
            pl.col("trade_date").is_in(signal_dates)
            & (pl.col("univ_rank") <= universe_n)
            & pl.col("factor").is_finite()
            & pl.col(forward).is_finite()
        )
        .with_columns(
            (pl.col("factor") * direction).alias("_signal"),
        )
        .with_columns(
            pl.col("_signal")
            .rank(method="average")
            .over("trade_date")
            .alias("_signal_rank"),
            pl.col(forward)
            .rank(method="average")
            .over("trade_date")
            .alias("_return_rank"),
        )
        .group_by("trade_date")
        .agg(
            pl.corr("_signal", forward).alias("ic"),
            pl.corr("_signal_rank", "_return_rank").alias("rank_ic"),
            pl.len().alias("n"),
        )
        .filter(
            (pl.col("n") >= 50)
            & pl.col("ic").is_finite()
            & pl.col("rank_ic").is_finite()
        )
        .sort("trade_date")
        .collect()
    )
    if daily.height < 2:
        raise ValueError("IC 样本不足：有效截面少于 2")
    ic_values = [_safe_float(value) for value in daily["ic"].to_list()]
    rank_values = [
        _safe_float(value) for value in daily["rank_ic"].to_list()
    ]
    ic_mean, ic_std = _mean_std(ic_values)
    rank_mean, rank_std = _mean_std(rank_values)
    annualizer = math.sqrt(252.0 / horizon)
    return {
        "n_days": daily.height,
        "mean_cross_section_n": round(
            sum(int(value) for value in daily["n"].to_list()) / daily.height,
            2,
        ),
        "ic_mean": round(ic_mean, 8),
        "ic_std": round(ic_std, 8),
        "icir": round(
            ic_mean / ic_std * annualizer if ic_std > 1e-12 else 0.0,
            6,
        ),
        "ic_positive_rate": round(
            sum(value > 0 for value in ic_values) / len(ic_values),
            6,
        ),
        "rank_ic_mean": round(rank_mean, 8),
        "rank_ic_std": round(rank_std, 8),
        "rank_icir": round(
            rank_mean / rank_std * annualizer
            if rank_std > 1e-12
            else 0.0,
            6,
        ),
        "rank_ic_positive_rate": round(
            sum(value > 0 for value in rank_values) / len(rank_values),
            6,
        ),
    }


def select_training_direction(
    frame: pl.DataFrame,
    spec: BatchBacktestSpec,
) -> tuple[int, dict]:
    """Select and freeze one orientation using META_TRAIN only."""
    raw = information_coefficients(
        frame,
        start=spec.train_start,
        end=spec.train_end,
        horizon=spec.horizon,
        universe_n=spec.universe_n,
        direction=1,
    )
    selector = _safe_float(raw["rank_ic_mean"])
    if abs(selector) <= 1e-12:
        selector = _safe_float(raw["ic_mean"])
    direction = -1 if selector < 0 else 1
    oriented = {
        key: (
            round(-_safe_float(value), 8)
            if direction < 0
            and key
            in {
                "ic_mean",
                "icir",
                "rank_ic_mean",
                "rank_icir",
            }
            else value
        )
        for key, value in raw.items()
    }
    if direction < 0:
        oriented["ic_positive_rate"] = round(
            1.0 - _safe_float(raw["ic_positive_rate"]),
            6,
        )
        oriented["rank_ic_positive_rate"] = round(
            1.0 - _safe_float(raw["rank_ic_positive_rate"]),
            6,
        )
    oriented["raw_direction_plus_one_rank_ic"] = raw["rank_ic_mean"]
    oriented["selection_metric"] = (
        "train_rank_ic_then_train_pearson_ic_tiebreak"
    )
    oriented["selected_direction"] = direction
    oriented["frozen_before_holdout"] = True
    return direction, oriented


def run_cost_scenarios(
    frame: pl.DataFrame,
    *,
    direction: int,
    spec: BatchBacktestSpec,
    capture_detail: bool = False,
) -> dict[str, dict]:
    """Replay one prepared signal frame under every requested slippage level."""
    start_date = date.fromisoformat(spec.holdout_start)
    end_date = date.fromisoformat(spec.holdout_end)
    period = frame.filter(
        pl.col("trade_date").is_between(start_date, end_date)
    ).drop(f"fwd_{spec.horizon}", strict=False)
    sessions = period.partition_by("trade_date", maintain_order=True)
    if len(sessions) < 60:
        raise ValueError("事件回测样本不足 (有效交易日 < 60)")
    session_dates = [session["trade_date"][0] for session in sessions]
    session_rows = [session.to_dicts() for session in sessions]
    runners = {
        float(bps): StepEventBacktester(
            EventBacktestConfig(
                market=spec.market,
                mode=spec.mode,
                direction=direction,
                universe_n=spec.universe_n,
                top_fraction=spec.top_fraction,
                initial_capital=spec.initial_capital,
                rebalance_every=spec.rebalance_every,
                slippage_bps=float(bps),
                max_volume_participation=spec.max_volume_participation,
                borrow_cost_bps_annual=spec.borrow_cost_bps_annual,
                fee_profile=spec.fee_profile,
            ),
            capture_detail=capture_detail,
        )
        for bps in spec.slippage_bps
    }
    for index, rows in enumerate(session_rows):
        market = {str(row["ts_code"]): row for row in rows}
        next_date = (
            session_dates[index + 1]
            if index + 1 < len(session_dates)
            else None
        )
        for runner in runners.values():
            runner.step(
                trade_date=session_dates[index],
                rows=rows,
                next_trade_date=next_date,
                rebalance=index % spec.rebalance_every == 0,
                market_by_symbol=market,
            )
    output: dict[str, dict] = {}
    for bps, runner in runners.items():
        result = runner.result()
        stats = result["stats"]
        output[f"{bps:g}"] = {
            "ann_return": stats["ann_ret"],
            "sharpe": stats["sharpe"],
            "max_drawdown": stats["max_dd"],
            "avg_daily_turnover": stats["avg_daily_turnover"],
            "final_nav": stats["final_nav"],
            "fills": stats["fills"],
            "fill_rate": stats["fill_rate"],
            "commission_and_tax": stats["commission_and_tax"],
            "slippage_cost": stats["slippage_cost"],
            "borrow_cost": stats["borrow_cost"],
            "total_execution_cost": stats["total_execution_cost"],
            "avg_gross_exposure": stats["avg_gross_exposure"],
            "avg_net_exposure": stats["avg_net_exposure"],
            "fee_profile": stats["fee_profile"],
            "currency": stats["currency"],
            "integrity": result["integrity"],
            "detail_capture": result["detail_capture"],
        }
        if capture_detail:
            output[f"{bps:g}"]["result"] = result
    return output


def flatten_factor_result(result: dict) -> dict:
    """Flatten nested metrics into a CSV-friendly leaderboard row."""
    row = {
        key: value
        for key, value in result.items()
        if key not in {"train_ic", "holdout_ic", "scenarios", "provenance"}
    }
    row.setdefault("policy_label", "NON_PIT_RESEARCH")
    row.setdefault("production_eligible", False)
    row["oriented_expression_hash"] = str(
        result.get("oriented_expression_hash")
        or oriented_expression_hash(
            str(result["expression"]),
            int(result["direction"]),
        )
    )
    ic_fingerprint_keys = {
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
    }
    fingerprint_payload = {
        "train_ic": {
            key: value
            for key, value in result["train_ic"].items()
            if key in ic_fingerprint_keys
        },
        "holdout_ic": {
            key: value
            for key, value in result["holdout_ic"].items()
            if key in ic_fingerprint_keys
        },
        "scenarios": {
            bps: {
                key: value
                for key, value in scenario.items()
                if key
                in {
                    "ann_return",
                    "sharpe",
                    "max_drawdown",
                    "avg_daily_turnover",
                    "final_nav",
                    "fills",
                    "fill_rate",
                    "commission_and_tax",
                    "slippage_cost",
                    "borrow_cost",
                    "total_execution_cost",
                    "avg_gross_exposure",
                    "avg_net_exposure",
                    "fee_profile",
                    "currency",
                }
            }
            for bps, scenario in result["scenarios"].items()
        },
    }
    row["evaluation_fingerprint"] = hashlib.sha256(
        json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:16]
    for prefix, metrics in (
        ("train", result["train_ic"]),
        ("oos", result["holdout_ic"]),
    ):
        for key, value in metrics.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[f"{prefix}_{key}"] = value
    for bps, scenario in result["scenarios"].items():
        suffix = bps.replace(".", "_")
        for key, value in scenario.items():
            if key in {"integrity", "result"}:
                continue
            row[f"{key}_bps_{suffix}"] = value
        row[f"total_return_bps_{suffix}"] = round(
            _safe_float(scenario.get("final_nav"), 1.0) - 1.0,
            8,
        )
        row[f"integrity_bps_{suffix}"] = bool(
            scenario["integrity"]["all_pass"]
        )
    return row


def _percentile_ranks(
    rows: list[dict],
    key: str,
) -> dict[str, float]:
    ordered = sorted(
        (
            (_safe_float(row.get(key), float("-inf")), row["expression_hash"])
            for row in rows
        ),
        key=lambda item: item[0],
    )
    n = max(1, len(ordered))
    output: dict[str, float] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2.0
        percentile = average_rank / n
        for _, expression_hash in ordered[index:end]:
            output[expression_hash] = percentile
        index = end
    return output


def _benjamini_hochberg(
    rows: list[dict],
    *,
    p_key: str,
    q_key: str,
) -> None:
    ordered = sorted(
        rows,
        key=lambda row: _safe_float(row.get(p_key), 1.0),
    )
    total = max(1, len(ordered))
    running = 1.0
    for reverse_index in range(len(ordered) - 1, -1, -1):
        rank = reverse_index + 1
        raw = _safe_float(ordered[reverse_index].get(p_key), 1.0)
        running = min(running, raw * total / rank)
        ordered[reverse_index][q_key] = round(min(1.0, running), 8)


def rank_factor_results(rows: Iterable[dict]) -> list[dict]:
    """Add absolute quality gates plus cross-sectional diagnostic ranks.

    ``robust_score`` remains a within-run percentile and is never interpreted
    as qualification. ``absolute_quality_score`` uses fixed economic targets;
    a run where every candidate fails therefore has no synthetic champion.
    """
    ranked = [dict(row) for row in rows if row.get("status") == "ok"]
    groups: dict[str, list[dict]] = {}
    for row in ranked:
        groups.setdefault(
            str(
                row.get("portfolio_fingerprint_bps_15")
                or row.get("evaluation_fingerprint")
                or row.get("oriented_expression_hash")
                or row["expression_hash"]
            ),
            [],
        ).append(row)
    representatives: list[dict] = []
    for members in groups.values():
        members.sort(
            key=lambda row: (
                -int(row.get("source_record_count") or 0),
                row["expression_hash"],
            )
        )
        representative = members[0]
        representative["economic_representative"] = True
        representative["economic_duplicate_of"] = ""
        representative["equivalence_group_size"] = len(members)
        representatives.append(representative)
        for duplicate in members[1:]:
            duplicate["economic_representative"] = False
            duplicate["economic_duplicate_of"] = representative[
                "expression_hash"
            ]
            duplicate["equivalence_group_size"] = len(members)
    for row in representatives:
        long_only = str(row.get("portfolio_mode") or "") == "long_only"
        has_active = row.get("active_sharpe_bps_15") not in {None, ""}
        use_active = long_only or has_active
        row["portfolio_metric_basis"] = "active" if use_active else "net"
        for bps in (0, 5, 15):
            row[f"ranking_ann_return_bps_{bps}"] = _safe_float(
                row.get(
                    f"active_ann_return_bps_{bps}"
                    if use_active
                    else f"ann_return_bps_{bps}"
                )
            )
            row[f"ranking_sharpe_bps_{bps}"] = _safe_float(
                row.get(
                    f"active_sharpe_bps_{bps}"
                    if use_active
                    else f"sharpe_bps_{bps}"
                )
            )
            row[f"ranking_max_drawdown_bps_{bps}"] = _safe_float(
                row.get(
                    f"active_max_drawdown_bps_{bps}"
                    if use_active
                    else f"max_drawdown_bps_{bps}"
                ),
                1.0,
            )
    dimensions = {
        "ranking_ann_return_bps_15": 0.20,
        "ranking_sharpe_bps_15": 0.25,
        "oos_ic_mean": 0.10,
        "oos_icir": 0.15,
        "oos_rank_ic_mean": 0.10,
        "oos_rank_icir": 0.15,
        "cost_resilience": 0.05,
    }
    for row in representatives:
        n_days = max(0, int(row.get("oos_n_days") or 0))
        ic_std = _safe_float(row.get("oos_ic_std"))
        rank_std = _safe_float(row.get("oos_rank_ic_std"))
        row["oos_ic_t"] = round(
            (
                _safe_float(row.get("oos_ic_mean"))
                / ic_std
                * math.sqrt(n_days)
            )
            if n_days > 1 and ic_std > 1e-12
            else 0.0,
            6,
        )
        row["oos_rank_ic_t"] = round(
            (
                _safe_float(row.get("oos_rank_ic_mean"))
                / rank_std
                * math.sqrt(n_days)
            )
            if n_days > 1 and rank_std > 1e-12
            else 0.0,
            6,
        )
        row["oos_ic_p_normal"] = round(
            math.erfc(abs(row["oos_ic_t"]) / math.sqrt(2.0)),
            8,
        )
        row["oos_rank_ic_p_normal"] = round(
            math.erfc(abs(row["oos_rank_ic_t"]) / math.sqrt(2.0)),
            8,
        )
        row["cost_resilience"] = round(
            _safe_float(row.get("ranking_ann_return_bps_15"))
            - _safe_float(row.get("ranking_ann_return_bps_0")),
            8,
        )
    _benjamini_hochberg(
        representatives,
        p_key="oos_ic_p_normal",
        q_key="oos_ic_bh_q",
    )
    _benjamini_hochberg(
        representatives,
        p_key="oos_rank_ic_p_normal",
        q_key="oos_rank_ic_bh_q",
    )
    percentiles = {
        key: _percentile_ranks(representatives, key) for key in dimensions
    }
    for row in representatives:
        expression_hash = row["expression_hash"]
        row["robust_score"] = round(
            100.0
            * sum(
                weight * percentiles[key][expression_hash]
                for key, weight in dimensions.items()
            ),
            4,
        )
        for key in dimensions:
            row[f"pct_{key}"] = round(
                percentiles[key][expression_hash],
                6,
            )
        return_0 = _safe_float(row.get("ranking_ann_return_bps_0"))
        return_5 = _safe_float(row.get("ranking_ann_return_bps_5"))
        return_15 = _safe_float(row.get("ranking_ann_return_bps_15"))
        sharpe_15 = _safe_float(row.get("ranking_sharpe_bps_15"))
        drawdown_15 = _safe_float(
            row.get("ranking_max_drawdown_bps_15"),
            1.0,
        )
        integrity = all(
            bool(row.get(f"integrity_bps_{bps}"))
            for bps in (0, 5, 15)
        )
        row["cost_monotonic"] = (
            return_0 + 1e-9 >= return_5
            and return_5 + 1e-9 >= return_15
        )
        signal_evidence_pass = bool(
            (
                _safe_float(row.get("oos_rank_ic_mean")) > 0
                and _safe_float(row.get("oos_rank_ic_bh_q"), 1.0) <= 0.10
            )
            or (
                _safe_float(row.get("oos_ic_mean")) > 0
                and _safe_float(row.get("oos_ic_bh_q"), 1.0) <= 0.10
            )
        )
        row["practical_pass"] = bool(
            integrity
            and row["cost_monotonic"]
            and return_15 > 0
            and sharpe_15 >= 0.50
            and drawdown_15 <= 0.35
            and signal_evidence_pass
        )
        row["multiple_test_pass"] = signal_evidence_pass
        profitability = 0.50 * max(0.0, min(1.0, return_15 / 0.10)) + 0.50 * max(
            0.0, min(1.0, sharpe_15 / 1.50)
        )
        drawdown_quality = max(0.0, min(1.0, (0.50 - drawdown_15) / 0.40))
        predictive = 0.50 * max(
            0.0, min(1.0, _safe_float(row.get("oos_icir")) / 1.0)
        ) + 0.50 * max(
            0.0, min(1.0, _safe_float(row.get("oos_rank_icir")) / 1.0)
        )
        q_value = min(
            _safe_float(row.get("oos_ic_bh_q"), 1.0),
            _safe_float(row.get("oos_rank_ic_bh_q"), 1.0),
        )
        confidence = max(0.0, min(1.0, (0.25 - q_value) / 0.25))
        cost_survival = (
            0.50 * float(row["cost_monotonic"])
            + 0.50 * max(0.0, min(1.0, return_15 / max(0.01, return_0)))
        )
        row["absolute_quality_score"] = round(
            100.0
            * (
                0.35 * profitability
                + 0.15 * drawdown_quality
                + 0.20 * predictive
                + 0.15 * confidence
                + 0.10 * cost_survival
                + 0.05 * float(integrity)
            ),
            4,
        )
        row["qualification_status"] = (
            "qualified"
            if row["practical_pass"]
            else "diagnostic_only_failed_gate"
        )
    representatives.sort(
        key=lambda row: (
            bool(row["practical_pass"]),
            _safe_float(row["absolute_quality_score"]),
            _safe_float(row["robust_score"]),
        ),
        reverse=True,
    )
    for index, row in enumerate(representatives, start=1):
        row["overall_rank"] = index
    for dimension in dimensions:
        ordered = sorted(
            representatives,
            key=lambda row: _safe_float(row.get(dimension), float("-inf")),
            reverse=True,
        )
        for index, row in enumerate(ordered, start=1):
            row[f"rank_{dimension}"] = index
    representative_by_hash = {
        row["expression_hash"]: row for row in representatives
    }
    for members in groups.values():
        representative = next(
            row for row in members if row["economic_representative"]
        )
        source = representative_by_hash[representative["expression_hash"]]
        for duplicate in members:
            if duplicate["economic_representative"]:
                continue
            for key in (
                "cost_resilience",
                "robust_score",
                "absolute_quality_score",
                "qualification_status",
                "portfolio_metric_basis",
                "cost_monotonic",
                "practical_pass",
                "multiple_test_pass",
                "oos_ic_t",
                "oos_rank_ic_t",
                "oos_ic_p_normal",
                "oos_rank_ic_p_normal",
                "oos_ic_bh_q",
                "oos_rank_ic_bh_q",
                "overall_rank",
                *[f"ranking_ann_return_bps_{bps}" for bps in (0, 5, 15)],
                *[f"ranking_sharpe_bps_{bps}" for bps in (0, 5, 15)],
                *[f"ranking_max_drawdown_bps_{bps}" for bps in (0, 5, 15)],
                *[f"pct_{dimension}" for dimension in dimensions],
                *[f"rank_{dimension}" for dimension in dimensions],
            ):
                duplicate[key] = source[key]
    ranked.sort(
        key=lambda row: (
            int(row["overall_rank"]),
            not bool(row["economic_representative"]),
            row["expression_hash"],
        )
    )
    return ranked
