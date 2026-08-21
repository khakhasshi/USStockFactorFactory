"""Purged walk-forward weight optimisation for frozen A-share factors.

This protocol is deliberately separate from the static V2 optimiser.  It
selects one fixed 2-5 factor portfolio using only pre-2023 data, but requires
that the same weights survive four expanding-train / forward-validation
folds.  Validation scores combine benchmark-relative alpha and absolute
long-only returns so a factor cannot look low-drawdown merely because market
beta was subtracted from the search objective.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Sequence

import numpy as np

from .weight_optimizer import (
    CrossSectionSlice,
    WeightSearchConfig,
    _component_return_path_similarity,
    _hac_lcb_matrix,
    _refine_neighbors,
    _return_path_pair_ok,
    _series_stats_matrix,
    _tail_widths,
    _weight_key,
    enumerate_coarse_weights,
)


WALK_FORWARD_PROTOCOL = "purged_walk_forward_absolute_active_v3"


@dataclass(frozen=True)
class WalkForwardFold:
    name: str
    train_start: date
    train_end: date
    validation_start: date
    validation_end: date


DEFAULT_FOLDS = (
    WalkForwardFold(
        "WF1",
        date(2010, 1, 1),
        date(2014, 12, 31),
        date(2015, 1, 1),
        date(2016, 12, 31),
    ),
    WalkForwardFold(
        "WF2",
        date(2010, 1, 1),
        date(2016, 12, 31),
        date(2017, 1, 1),
        date(2018, 12, 31),
    ),
    WalkForwardFold(
        "WF3",
        date(2010, 1, 1),
        date(2018, 12, 31),
        date(2019, 1, 1),
        date(2020, 12, 31),
    ),
    WalkForwardFold(
        "WF4",
        date(2010, 1, 1),
        date(2020, 12, 31),
        date(2021, 1, 1),
        date(2022, 12, 31),
    ),
)


@dataclass(frozen=True)
class WalkForwardConfig:
    min_mechanism_groups: int = 3
    boundary_purge_slices: int = 1
    near_optimal_tolerance: float = 0.01

    def validate(self, mechanism_count: int) -> None:
        if not 2 <= self.min_mechanism_groups <= mechanism_count:
            raise ValueError("min_mechanism_groups 超出候选收益机制数量")
        if self.boundary_purge_slices < 1:
            raise ValueError("walk-forward 边界至少 purge 一个非重叠切片")
        if not 0 < self.near_optimal_tolerance <= 0.25:
            raise ValueError("near_optimal_tolerance 必须在 (0, 0.25]")


def _window_slices(
    slices: Sequence[CrossSectionSlice],
    start: date,
    end: date,
    *,
    purge_tail: int,
) -> list[CrossSectionSlice]:
    rows = sorted(
        (row for row in slices if start <= row.trade_date <= end),
        key=lambda row: row.trade_date,
    )
    if len(rows) <= purge_tail:
        return []
    return rows[:-purge_tail]


def _positive_year_rate(
    returns: np.ndarray,
    years: np.ndarray,
) -> np.ndarray:
    output = np.zeros(returns.shape[1], dtype=np.float64)
    unique_years = sorted(set(int(value) for value in years))
    for year in unique_years:
        mask = years == year
        output += np.prod(np.maximum(1e-12, 1.0 + returns[mask]), axis=0) > 1.0
    return output / max(1, len(unique_years))


def _evaluate_window(
    slices: Sequence[CrossSectionSlice],
    weights: np.ndarray,
    config: WeightSearchConfig,
) -> dict[str, np.ndarray]:
    """Vectorised non-overlapping proxy for one chronological window."""
    if len(slices) < 8:
        raise ValueError("walk-forward 窗口有效非重叠截面不足 8 个")
    candidate_count = len(weights)
    symbol_count = 1 + max(int(row.symbols.max()) for row in slices)
    tail_widths = _tail_widths(config)
    previous = {
        width: np.zeros((symbol_count, candidate_count), dtype=np.bool_)
        for width in tail_widths
    }
    previous_n = {
        width: np.ones(candidate_count, dtype=np.float64)
        for width in tail_widths
    }
    gross_rows = {width: [] for width in tail_widths}
    turnover_rows = {width: [] for width in tail_widths}
    benchmark_rows = []
    ic_rows = []
    years = []
    for row in slices:
        scores = row.components @ weights.T
        benchmark_rows.append(
            np.full(candidate_count, row.forward_returns.mean(), dtype=np.float64)
        )
        for width in tail_widths:
            select_n = max(1, int(math.floor(len(row.forward_returns) * width)))
            selected = np.argpartition(scores, -select_n, axis=0)[-select_n:, :]
            gross_rows[width].append(row.forward_returns[selected].mean(axis=0))
            current = np.zeros_like(previous[width])
            selected_symbols = row.symbols[selected]
            columns = np.broadcast_to(
                np.arange(candidate_count), selected_symbols.shape
            )
            current[selected_symbols, columns] = True
            current_weight = current.astype(np.float64) / select_n
            prior_weight = previous[width].astype(np.float64) / previous_n[width]
            turnover_rows[width].append(
                np.abs(current_weight - prior_weight).sum(axis=0)
            )
            previous[width] = current
            previous_n[width].fill(select_n)

        score_ranks = np.argsort(
            np.argsort(scores, axis=0, kind="mergesort"),
            axis=0,
            kind="mergesort",
        ).astype(np.float64)
        score_centered = score_ranks - score_ranks.mean(axis=0)
        return_centered = row.return_ranks - row.return_ranks.mean()
        numerator = (score_centered * return_centered[:, None]).sum(axis=0)
        denominator = np.sqrt(
            (score_centered * score_centered).sum(axis=0)
            * np.sum(return_centered * return_centered)
        )
        ic_rows.append(np.divide(
            numerator,
            denominator,
            out=np.zeros(candidate_count),
            where=denominator > 1e-12,
        ))
        years.append(row.trade_date.year)

    periods_per_year = 252.0 / config.horizon
    benchmark = np.vstack(benchmark_rows)
    tail_active: dict[float, np.ndarray] = {}
    tail_long: dict[float, np.ndarray] = {}
    tail_active_stats = {}
    tail_long_stats = {}
    for width in tail_widths:
        gross = np.vstack(gross_rows[width])
        turnover = np.vstack(turnover_rows[width])
        net_long = gross - turnover * config.cost_bps / 10_000.0
        net_active = net_long - benchmark
        tail_long[width] = net_long
        tail_active[width] = net_active
        tail_long_stats[width] = _series_stats_matrix(net_long, periods_per_year)
        tail_active_stats[width] = _series_stats_matrix(
            net_active, periods_per_year
        )

    gross = np.vstack(gross_rows[config.top_fraction])
    turnover = np.vstack(turnover_rows[config.top_fraction])
    long_returns = tail_long[config.top_fraction]
    active_returns = tail_active[config.top_fraction]
    stress_long = gross - turnover * config.stress_cost_bps / 10_000.0
    stress_active = stress_long - benchmark
    long_stats = _series_stats_matrix(long_returns, periods_per_year)
    active_stats = _series_stats_matrix(active_returns, periods_per_year)
    stress_long_stats = _series_stats_matrix(stress_long, periods_per_year)
    stress_active_stats = _series_stats_matrix(stress_active, periods_per_year)
    long_ann_lcb, long_sharpe_lcb = _hac_lcb_matrix(
        long_returns, periods_per_year, 0.90
    )
    active_ann_lcb, active_sharpe_lcb = _hac_lcb_matrix(
        active_returns, periods_per_year, 0.90
    )
    ic = np.vstack(ic_rows)
    ic_mean = ic.mean(axis=0)
    ic_std = ic.std(axis=0, ddof=1)
    icir = np.divide(
        ic_mean * math.sqrt(periods_per_year),
        ic_std,
        out=np.zeros_like(ic_mean),
        where=ic_std > 1e-12,
    )
    tail_active_sharpe = np.vstack([
        tail_active_stats[width]["sharpe"] for width in tail_widths
    ])
    tail_long_sharpe = np.vstack([
        tail_long_stats[width]["sharpe"] for width in tail_widths
    ])
    years_array = np.asarray(years, dtype=np.int32)
    return {
        "active_ann_return": active_stats["ann_return"],
        "active_sharpe": active_stats["sharpe"],
        "active_max_drawdown": active_stats["max_drawdown"],
        "active_ann_return_lcb": active_ann_lcb,
        "active_sharpe_lcb": active_sharpe_lcb,
        "active_stress_sharpe": stress_active_stats["sharpe"],
        "active_positive_year_rate": _positive_year_rate(
            active_returns, years_array
        ),
        "long_ann_return": long_stats["ann_return"],
        "long_sharpe": long_stats["sharpe"],
        "long_max_drawdown": long_stats["max_drawdown"],
        "long_ann_return_lcb": long_ann_lcb,
        "long_sharpe_lcb": long_sharpe_lcb,
        "long_stress_sharpe": stress_long_stats["sharpe"],
        "long_positive_year_rate": _positive_year_rate(long_returns, years_array),
        "ic_mean": ic_mean,
        "icir": icir,
        "worst_tail_active_sharpe": tail_active_sharpe.min(axis=0),
        "worst_tail_long_sharpe": tail_long_sharpe.min(axis=0),
        "avg_turnover": turnover.mean(axis=0),
    }


def _fold_core(metrics: dict[str, np.ndarray]) -> np.ndarray:
    """One validation-fold score with active and absolute risk at parity."""
    return (
        0.18 * np.clip(metrics["active_sharpe_lcb"] / 1.0, -1.0, 1.0)
        + 0.18 * np.clip(metrics["long_sharpe_lcb"] / 1.0, -1.0, 1.0)
        + 0.10 * np.clip(metrics["active_ann_return_lcb"] / 0.08, -1.0, 1.0)
        + 0.10 * np.clip(metrics["long_ann_return_lcb"] / 0.08, -1.0, 1.0)
        + 0.09 * np.clip(metrics["active_stress_sharpe"] / 0.75, -1.0, 1.0)
        + 0.09 * np.clip(metrics["long_stress_sharpe"] / 0.75, -1.0, 1.0)
        + 0.05 * np.clip(metrics["worst_tail_active_sharpe"], -1.0, 1.0)
        + 0.05 * np.clip(metrics["worst_tail_long_sharpe"], -1.0, 1.0)
        + 0.04 * np.clip(metrics["icir"] / 2.0, -1.0, 1.0)
        + 0.04 * (2.0 * metrics["active_positive_year_rate"] - 1.0)
        + 0.04 * (2.0 * metrics["long_positive_year_rate"] - 1.0)
        + 0.04 * (
            1.0 - np.clip(metrics["long_max_drawdown"] / 0.35, 0.0, 2.0)
        )
    )


def _row_metric(metrics: dict[str, np.ndarray], index: int) -> dict[str, float]:
    return {
        key: round(float(value[index]), 8)
        for key, value in metrics.items()
    }


def evaluate_walk_forward_weights(
    slices: Sequence[CrossSectionSlice],
    candidate_weights: Sequence[Sequence[float]],
    config: WeightSearchConfig,
    *,
    factor_groups: Sequence[str],
    return_path_similarity: np.ndarray,
    folds: Sequence[WalkForwardFold] = DEFAULT_FOLDS,
    walk_forward: WalkForwardConfig = WalkForwardConfig(),
) -> list[dict]:
    if not candidate_weights:
        return []
    weights = np.asarray(candidate_weights, dtype=np.float64)
    fold_payload = []
    for fold in folds:
        train_slices = _window_slices(
            slices,
            fold.train_start,
            fold.train_end,
            purge_tail=walk_forward.boundary_purge_slices,
        )
        validation_slices = _window_slices(
            slices,
            fold.validation_start,
            fold.validation_end,
            purge_tail=walk_forward.boundary_purge_slices,
        )
        train = _evaluate_window(train_slices, weights, config)
        validation = _evaluate_window(validation_slices, weights, config)
        fold_payload.append((fold, train_slices, validation_slices, train, validation))

    validation_scores = np.vstack([
        _fold_core(validation)
        for _, _, _, _, validation in fold_payload
    ])
    active_degradation = np.vstack([
        np.maximum(0.0, train["active_sharpe"] - validation["active_sharpe"])
        / np.maximum(1.0, np.abs(train["active_sharpe"]))
        for _, _, _, train, validation in fold_payload
    ])
    long_degradation = np.vstack([
        np.maximum(0.0, train["long_sharpe"] - validation["long_sharpe"])
        / np.maximum(1.0, np.abs(train["long_sharpe"]))
        for _, _, _, train, validation in fold_payload
    ])
    validation_active_sharpes = np.vstack([
        validation["active_sharpe"]
        for _, _, _, _, validation in fold_payload
    ])
    validation_long_sharpes = np.vstack([
        validation["long_sharpe"]
        for _, _, _, _, validation in fold_payload
    ])
    hhi = np.sum(weights * weights, axis=1)
    active_count = np.sum(weights > 1e-12, axis=1)
    equal_hhi = 1.0 / active_count
    diversification = np.divide(
        1.0 - hhi,
        1.0 - equal_hhi,
        out=np.ones_like(hhi),
        where=active_count > 1,
    )
    unique_groups = tuple(dict.fromkeys(factor_groups))
    group_weights = np.column_stack([
        weights[:, [
            index for index, group in enumerate(factor_groups) if group == target
        ]].sum(axis=1)
        for target in unique_groups
    ])
    active_group_count = np.sum(group_weights > 1e-12, axis=1)
    group_hhi = np.sum(group_weights * group_weights, axis=1)
    off_diagonal = return_path_similarity.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    pair_weight = np.maximum(1e-12, 1.0 - hhi)
    weighted_similarity = np.einsum(
        "bi,ij,bj->b", weights, off_diagonal, weights
    ) / pair_weight
    independence = 1.0 - np.clip(weighted_similarity, 0.0, 1.0)
    degradation = 0.5 * active_degradation.mean(axis=0) + 0.5 * long_degradation.mean(axis=0)
    regime_dispersion = (
        0.5 * validation_active_sharpes.std(axis=0)
        + 0.5 * validation_long_sharpes.std(axis=0)
    )
    robust_score = (
        0.55 * validation_scores.min(axis=0)
        + 0.30 * np.median(validation_scores, axis=0)
        + 0.15 * validation_scores.mean(axis=0)
        - 0.10 * np.clip(degradation, 0.0, 1.5)
        - 0.07 * np.clip(regime_dispersion / 1.5, 0.0, 1.5)
        + 0.04 * np.clip(independence, 0.0, 1.0)
        + 0.02 * np.clip(diversification, 0.0, 1.0)
        + 0.02 * np.clip(1.0 - group_hhi, 0.0, 1.0)
    )

    output = []
    for index, row in enumerate(weights):
        fold_metrics = {}
        for fold, train_rows, validation_rows, train, validation in fold_payload:
            fold_metrics[fold.name] = {
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "validation_start": str(fold.validation_start),
                "validation_end": str(fold.validation_end),
                "purged_tail_slices": walk_forward.boundary_purge_slices,
                "train_slices": len(train_rows),
                "validation_slices": len(validation_rows),
                "train": _row_metric(train, index),
                "validation": _row_metric(validation, index),
                "validation_core_score": round(
                    float(_fold_core(validation)[index]), 8
                ),
                "active_sharpe_degradation": round(
                    float(active_degradation[len(fold_metrics), index]), 8
                ),
                "long_sharpe_degradation": round(
                    float(long_degradation[len(fold_metrics), index]), 8
                ),
            }
        output.append({
            "weights": row.round(10).tolist(),
            "active_factors": int(active_count[index]),
            "active_mechanism_groups": int(active_group_count[index]),
            "robust_score": round(float(robust_score[index]), 8),
            "worst_validation_core_score": round(
                float(validation_scores[:, index].min()), 8
            ),
            "median_validation_core_score": round(
                float(np.median(validation_scores[:, index])), 8
            ),
            "worst_validation_active_sharpe": round(
                float(validation_active_sharpes[:, index].min()), 8
            ),
            "median_validation_active_sharpe": round(
                float(np.median(validation_active_sharpes[:, index])), 8
            ),
            "worst_validation_long_sharpe": round(
                float(validation_long_sharpes[:, index].min()), 8
            ),
            "median_validation_long_sharpe": round(
                float(np.median(validation_long_sharpes[:, index])), 8
            ),
            "positive_active_folds": round(
                float((validation_active_sharpes[:, index] > 0).mean()), 8
            ),
            "positive_long_folds": round(
                float((validation_long_sharpes[:, index] > 0).mean()), 8
            ),
            "mean_degradation_penalty": round(float(degradation[index]), 8),
            "regime_sharpe_dispersion": round(
                float(regime_dispersion[index]), 8
            ),
            "weight_hhi": round(float(hhi[index]), 8),
            "effective_factor_count": round(float(1.0 / hhi[index]), 8),
            "mechanism_weight_hhi": round(float(group_hhi[index]), 8),
            "return_source_independence": round(float(independence[index]), 8),
            "weighted_return_path_similarity": round(
                float(weighted_similarity[index]), 8
            ),
            "folds": fold_metrics,
        })
    return output


def _minimum_group_count_ok(
    weights: Sequence[float],
    groups: Sequence[str],
    minimum: int,
) -> bool:
    return len({group for weight, group in zip(weights, groups) if weight > 1e-12}) >= minimum


def search_walk_forward_weights(
    slices: Sequence[CrossSectionSlice],
    factor_count: int,
    config: WeightSearchConfig,
    *,
    factor_groups: Sequence[str],
    folds: Sequence[WalkForwardFold] = DEFAULT_FOLDS,
    walk_forward: WalkForwardConfig = WalkForwardConfig(),
) -> dict:
    config.validate(factor_count)
    groups = tuple(factor_groups)
    if len(groups) != factor_count:
        raise ValueError("factor_groups 长度必须等于 factor_count")
    walk_forward.validate(len(set(groups)))
    return_path_similarity = _component_return_path_similarity(slices, config)
    coarse_all = enumerate_coarse_weights(factor_count, config, groups)
    coarse_groups = [
        row for row in coarse_all
        if _minimum_group_count_ok(
            row, groups, walk_forward.min_mechanism_groups
        )
    ]
    coarse = [
        row for row in coarse_groups
        if _return_path_pair_ok(
            row, return_path_similarity, config.max_active_pair_similarity
        )
    ]
    if not coarse:
        raise ValueError("walk-forward 结构约束过滤了全部候选权重")
    coarse_results = evaluate_walk_forward_weights(
        slices,
        coarse,
        config,
        factor_groups=groups,
        return_path_similarity=return_path_similarity,
        folds=folds,
        walk_forward=walk_forward,
    )
    all_results = {_weight_key(row["weights"]): row for row in coarse_results}
    ranked = sorted(
        coarse_results, key=lambda row: row["robust_score"], reverse=True
    )
    starts = ranked[:config.refine_starts]
    for size in range(config.min_factors, config.max_factors + 1):
        best = next((row for row in ranked if row["active_factors"] == size), None)
        if best is not None:
            starts.append(best)
    for start in starts:
        incumbent = start
        for _ in range(config.refine_iterations):
            neighbors = [
                row for row in _refine_neighbors(
                    incumbent["weights"], config, groups
                )
                if _minimum_group_count_ok(
                    row, groups, walk_forward.min_mechanism_groups
                )
                and _return_path_pair_ok(
                    row,
                    return_path_similarity,
                    config.max_active_pair_similarity,
                )
            ]
            unseen = [
                row for row in neighbors if _weight_key(row) not in all_results
            ]
            if unseen:
                for result in evaluate_walk_forward_weights(
                    slices,
                    unseen,
                    config,
                    factor_groups=groups,
                    return_path_similarity=return_path_similarity,
                    folds=folds,
                    walk_forward=walk_forward,
                ):
                    all_results[_weight_key(result["weights"])] = result
            neighborhood = [
                all_results[_weight_key(row)]
                for row in neighbors
                if _weight_key(row) in all_results
            ]
            candidate = max(
                [incumbent, *neighborhood],
                key=lambda row: row["robust_score"],
            )
            if candidate["robust_score"] <= incumbent["robust_score"] + 1e-12:
                break
            incumbent = candidate

    final = sorted(
        all_results.values(),
        key=lambda row: (
            row["robust_score"],
            row["worst_validation_long_sharpe"],
            row["worst_validation_active_sharpe"],
            row["return_source_independence"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(final, start=1):
        row["rank"] = rank
    tolerance = walk_forward.near_optimal_tolerance
    near = [
        row for row in final
        if row["robust_score"] >= final[0]["robust_score"] - tolerance
    ]
    near_weights = np.asarray([row["weights"] for row in near])
    fold_winners = {}
    for fold in folds:
        winner = max(
            final,
            key=lambda row: row["folds"][fold.name]["validation_core_score"],
        )
        fold_winners[fold.name] = {
            "weights": winner["weights"],
            "validation_core_score": winner["folds"][fold.name][
                "validation_core_score"
            ],
        }
    return {
        "protocol": WALK_FORWARD_PROTOCOL,
        "llm_used": False,
        "search_scope": "2010_to_2022_purged_walk_forward_only",
        "holdout_or_vault_read": False,
        "config": asdict(config),
        "walk_forward_config": asdict(walk_forward),
        "folds": [asdict(fold) for fold in folds],
        "factor_groups": list(groups),
        "objective_semantics": (
            "worst_fold_active_and_absolute_long_only_LCB_plus_cost_stress_"
            "plus_degradation_and_regime_dispersion_penalties"
        ),
        "return_path_similarity_matrix": return_path_similarity.round(8).tolist(),
        "coarse_candidates_before_filters": len(coarse_all),
        "coarse_candidates_after_mechanism_filter": len(coarse_groups),
        "coarse_candidates": len(coarse),
        "evaluated_candidates": len(final),
        "fold_winners": fold_winners,
        "search_stability": {
            "score_tolerance": tolerance,
            "near_optimal_candidates": len(near),
            "best_runner_up_margin": round(
                final[0]["robust_score"] - final[1]["robust_score"], 8
            ) if len(final) > 1 else None,
            "weight_min": near_weights.min(axis=0).round(8).tolist(),
            "weight_max": near_weights.max(axis=0).round(8).tolist(),
            "weight_std": near_weights.std(axis=0).round(8).tolist(),
        },
        "best": final[0],
        "results": final,
    }
