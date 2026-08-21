"""Deterministic factor-combination weight search.

The optimizer works on direction-frozen, cross-sectional percentile scores.
It never calls an LLM and never reads META_HOLDOUT or FACTOR_VAULT. Candidate
portfolios are fitted jointly on INNER_PUBLIC and META_TRAIN, with the weaker
layer driving the robust objective. V2 additionally requires conversion into
executable top-tail returns across multiple portfolio widths and contiguous
time blocks, while penalising correlated standalone factor return paths. The
result remains NON_PIT research and requires an explicit event-engine replay
before use outside research.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import asdict, dataclass
from datetime import date
from statistics import NormalDist
from typing import Iterable, Sequence

import numpy as np
import polars as pl

from ..config import ASHARE_PANEL_GLOB, evaluation_config, get_dsl_fields
from ..data.panel import PanelStore
from ..dsl.engine import parse


COMBINATION_PROTOCOL = "deterministic_rank_weight_search_v2"
TRAINING_LAYERS = ("INNER_PUBLIC", "META_TRAIN")
PROMOTION_POLICY = {
    "min_event_sharpe_15bps": 0.50,
    "max_event_drawdown_15bps": 0.35,
    "max_active_pairwise_return_path_similarity": 0.85,
    "min_return_source_independence": 0.35,
    "min_worst_tail_sharpe": 0.50,
}


@dataclass(frozen=True)
class FactorComponent:
    factor_id: int | None
    name: str
    expression: str
    direction: int
    mechanism_family: str = "unknown"
    source: str = "factor_database"
    source_ref: str = ""

    def validate(self) -> None:
        if self.factor_id is not None and self.factor_id < 1:
            raise ValueError("factor_id 必须为正整数或 None")
        if self.factor_id is None and not self.source_ref.strip():
            raise ValueError("非数据库因子必须提供 source_ref")
        if not self.expression.strip():
            raise ValueError("因子表达式不能为空")
        if self.direction not in {-1, 1}:
            raise ValueError("因子方向必须为 1 或 -1")


@dataclass(frozen=True)
class WeightSearchConfig:
    min_factors: int = 2
    max_factors: int = 5
    coarse_step: float = 0.10
    refine_step: float = 0.02
    min_active_weight: float = 0.05
    max_weight: float = 0.65
    max_mechanism_weight: float = 0.60
    max_active_pair_similarity: float = 0.85
    refine_starts: int = 16
    refine_iterations: int = 12
    universe_n: int = 500
    horizon: int = 20
    top_fraction: float = 0.20
    tail_fractions: tuple[float, ...] = (0.10, 0.20, 0.30)
    time_block_folds: int = 3
    cost_bps: float = 20.0
    stress_cost_bps: float = 50.0
    batch_size: int = 256

    def validate(self, factor_count: int) -> None:
        if not 2 <= self.min_factors <= self.max_factors <= factor_count:
            raise ValueError("因子数量约束必须满足 2 <= min <= max <= 输入因子数")
        for key, value in (
            ("coarse_step", self.coarse_step),
            ("refine_step", self.refine_step),
            ("min_active_weight", self.min_active_weight),
            ("max_weight", self.max_weight),
            ("max_mechanism_weight", self.max_mechanism_weight),
            ("max_active_pair_similarity", self.max_active_pair_similarity),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{key} 必须在 (0, 1] 内")
        if self.min_factors * self.min_active_weight > 1 + 1e-12:
            raise ValueError("最小权重与最小因子数约束不可行")
        if self.max_factors * self.max_weight < 1 - 1e-12:
            raise ValueError("最大权重与最大因子数约束不可行")
        if not 0 < self.top_fraction <= 0.5:
            raise ValueError("top_fraction 必须在 (0, 0.5] 内")
        if len(self.tail_fractions) < 2:
            raise ValueError("tail_fractions 至少需要两个头部宽度")
        if any(not 0 < value <= 0.5 for value in self.tail_fractions):
            raise ValueError("tail_fractions 必须全部位于 (0, 0.5] 内")
        if len(set(self.tail_fractions)) != len(self.tail_fractions):
            raise ValueError("tail_fractions 不得重复")
        if self.time_block_folds < 2:
            raise ValueError("time_block_folds 至少为 2")
        if self.horizon < 1 or self.universe_n < 1:
            raise ValueError("horizon/universe_n 必须为正整数")
        for step_name, step in (
            ("coarse_step", self.coarse_step),
            ("refine_step", self.refine_step),
        ):
            units = round(1.0 / step)
            if not math.isclose(units * step, 1.0, abs_tol=1e-9):
                raise ValueError(f"{step_name} 必须能整除 1")


@dataclass(frozen=True)
class CrossSectionSlice:
    trade_date: date
    layer: str
    era: int
    symbols: np.ndarray
    components: np.ndarray
    forward_returns: np.ndarray
    return_ranks: np.ndarray


def _rank_average(values: np.ndarray) -> np.ndarray:
    """Average ranks, zero based; ties are uncommon but handled exactly."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def materialize_component_frame(
    components: Sequence[FactorComponent],
    *,
    universe_n: int = 500,
    horizon: int = 20,
    panel_glob: str | None = None,
    layers: Sequence[str] = TRAINING_LAYERS,
    include_execution_columns: bool = False,
) -> tuple[pl.DataFrame, dict]:
    """Materialize all direction-adjusted component ranks in one Polars plan."""
    if len(components) < 2:
        raise ValueError("组合优化至少需要两个因子")
    for component in components:
        component.validate()
    if any(layer not in TRAINING_LAYERS for layer in layers):
        raise ValueError("权重搜索只能读取 INNER_PUBLIC 和 META_TRAIN")

    store = PanelStore.get(panel_glob or ASHARE_PANEL_GLOB, "ashare")
    panel = store.ensure_loaded()
    forward = f"fwd_{horizon}"
    if forward not in panel.columns:
        raise ValueError(f"A股面板不支持 horizon={horizon}")

    lazy = panel.lazy().filter(pl.col("layer").is_in(list(layers)))
    raw_columns: list[str] = []
    score_columns: list[str] = []
    fields = get_dsl_fields("ashare")
    for index, component in enumerate(components):
        raw = f"_combo_raw_{index}"
        score = f"component_{index}"
        lazy = parse(component.expression, fields).apply(lazy, alias=raw)
        raw_columns.append(raw)
        score_columns.append(score)

    # Match the live screener: each component is independently oriented and
    # re-ranked inside the requested liquidity universe before weights apply.
    lazy = lazy.with_columns(
        *[
            pl.when(pl.col("univ_rank") <= universe_n)
            .then(pl.col(raw) * component.direction)
            .otherwise(None)
            .alias(f"_combo_oriented_{index}")
            for index, (raw, component) in enumerate(zip(raw_columns, components))
        ]
    )
    lazy = lazy.with_columns(
        *[
            (
                pl.col(f"_combo_oriented_{index}")
                .rank(method="average")
                .over("trade_date")
                / pl.col(f"_combo_oriented_{index}")
                .count()
                .over("trade_date")
            ).alias(score)
            for index, score in enumerate(score_columns)
        ]
    )
    select_columns = [
        "trade_date", "layer", "era", "ts_code", "univ_rank", forward,
        *score_columns,
    ]
    if include_execution_columns:
        select_columns.extend([
            "name", "raw_open", "raw_close", "vol", "amount",
            "adjustment_factor", "can_buy_open_proxy", "can_sell_open_proxy",
        ])
    frame = (
        lazy.select(*select_columns)
        .sort("trade_date", "ts_code")
        .collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False))
    )
    identity = hashlib.sha256(
        "\n".join(
            f"{row.factor_id}|{row.source_ref}|{row.direction}|{row.expression}"
            for row in components
        ).encode("utf-8")
    ).hexdigest()[:16]
    return frame, {
        "protocol": COMBINATION_PROTOCOL,
        "llm_used": False,
        "market": "ashare",
        "layers": list(layers),
        "panel": store.summary(),
        "component_identity": identity,
        "score_semantics": "weighted_direction_adjusted_cross_sectional_percentiles",
    }


def build_slices(
    frame: pl.DataFrame,
    *,
    component_count: int,
    horizon: int,
    universe_n: int,
) -> tuple[list[CrossSectionSlice], dict[str, int]]:
    """Convert one materialized frame into non-overlapping NumPy slices."""
    forward = f"fwd_{horizon}"
    component_columns = [f"component_{index}" for index in range(component_count)]
    required = {
        "trade_date", "layer", "era", "ts_code", "univ_rank", forward,
        *component_columns,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"组合帧缺少字段: {', '.join(missing)}")

    eligible = frame.filter(
        (pl.col("univ_rank") <= universe_n)
        & pl.col(forward).is_finite()
        & pl.all_horizontal(pl.col(column).is_finite() for column in component_columns)
    )
    symbols = sorted(str(value) for value in eligible["ts_code"].unique().to_list())
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    slices: list[CrossSectionSlice] = []
    layer_counts: dict[str, int] = {}
    for layer in TRAINING_LAYERS:
        layer_frame = eligible.filter(pl.col("layer") == layer)
        dates = layer_frame["trade_date"].unique().sort().to_list()
        # Purge labels at every immutable layer boundary.  ``fwd_horizon`` on
        # the last sessions of one layer would otherwise consume prices from
        # the next layer even though the signal date itself still belongs to
        # the earlier layer.
        usable_dates = dates[:max(0, len(dates) - horizon - 1)]
        selected_dates = set(usable_dates[::horizon])
        layer_counts[layer] = len(selected_dates)
        for group in layer_frame.filter(
            pl.col("trade_date").is_in(list(selected_dates))
        ).partition_by("trade_date", maintain_order=True):
            values = group.select(component_columns).to_numpy().astype(np.float64)
            returns = group[forward].to_numpy().astype(np.float64)
            if len(returns) < 50:
                continue
            slices.append(CrossSectionSlice(
                trade_date=group["trade_date"][0],
                layer=layer,
                era=int(group["era"][0]),
                symbols=np.asarray(
                    [symbol_index[str(value)] for value in group["ts_code"].to_list()],
                    dtype=np.int32,
                ),
                components=values,
                forward_returns=returns,
                return_ranks=_rank_average(returns),
            ))
    slices.sort(key=lambda row: (TRAINING_LAYERS.index(row.layer), row.trade_date))
    if any(sum(row.layer == layer for row in slices) < 12 for layer in TRAINING_LAYERS):
        raise ValueError("训练安全层的有效非重叠截面不足")
    return slices, {**layer_counts, "symbols": len(symbols)}


def _integer_compositions(total: int, count: int, minimum: int, maximum: int):
    if count == 1:
        if minimum <= total <= maximum:
            yield (total,)
        return
    low = max(minimum, total - maximum * (count - 1))
    high = min(maximum, total - minimum * (count - 1))
    for head in range(low, high + 1):
        for tail in _integer_compositions(
            total - head, count - 1, minimum, maximum
        ):
            yield (head, *tail)


def enumerate_coarse_weights(
    factor_count: int,
    config: WeightSearchConfig,
    factor_groups: Sequence[str] | None = None,
) -> list[tuple[float, ...]]:
    """Enumerate the complete constrained coarse simplex, deterministically."""
    config.validate(factor_count)
    total_units = round(1.0 / config.coarse_step)
    min_units = max(1, math.ceil(config.min_active_weight / config.coarse_step - 1e-12))
    max_units = math.floor(config.max_weight / config.coarse_step + 1e-12)
    groups = tuple(factor_groups or (f"factor_{index}" for index in range(factor_count)))
    if len(groups) != factor_count:
        raise ValueError("factor_groups 长度必须等于因子数")
    output: list[tuple[float, ...]] = []
    for support_size in range(config.min_factors, config.max_factors + 1):
        for support in itertools.combinations(range(factor_count), support_size):
            for allocation in _integer_compositions(
                total_units, support_size, min_units, max_units
            ):
                weights = [0.0] * factor_count
                for index, units in zip(support, allocation):
                    weights[index] = units / total_units
                if _mechanism_weight_ok(weights, groups, config.max_mechanism_weight):
                    output.append(tuple(weights))
    return output


def _series_stats_matrix(returns: np.ndarray, periods_per_year: float) -> dict[str, np.ndarray]:
    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=1)
    sharpe = np.divide(
        mean * math.sqrt(periods_per_year),
        std,
        out=np.zeros_like(mean),
        where=std > 1e-12,
    )
    nav = np.cumprod(np.maximum(1e-12, 1.0 + returns), axis=0)
    peaks = np.maximum.accumulate(np.vstack([np.ones((1, returns.shape[1])), nav]), axis=0)[1:]
    max_drawdown = np.max(1.0 - nav / peaks, axis=0)
    ann_return = np.power(nav[-1], periods_per_year / len(returns)) - 1.0
    return {
        "ann_return": ann_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def _hac_lcb_matrix(
    returns: np.ndarray,
    periods_per_year: float,
    confidence: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = returns.shape[0]
    mean = returns.mean(axis=0)
    residual = returns - mean
    lag = max(1, min(8, round(n ** (1.0 / 3.0))))
    long_run = np.mean(residual * residual, axis=0)
    for offset in range(1, min(lag, n - 1) + 1):
        covariance = np.mean(residual[offset:] * residual[:-offset], axis=0)
        long_run += 2.0 * (1.0 - offset / (lag + 1.0)) * covariance
    standard_error = np.sqrt(np.maximum(0.0, long_run) / n)
    mean_lcb = mean - NormalDist().inv_cdf(confidence) * standard_error
    std = returns.std(axis=0, ddof=1)
    sharpe_lcb = np.divide(
        mean_lcb * math.sqrt(periods_per_year),
        std,
        out=np.zeros_like(mean_lcb),
        where=std > 1e-12,
    )
    return mean_lcb * periods_per_year, sharpe_lcb


def _tail_widths(config: WeightSearchConfig) -> tuple[float, ...]:
    return tuple(sorted(set((*config.tail_fractions, config.top_fraction))))


def _component_return_path_similarity(
    slices: Sequence[CrossSectionSlice],
    config: WeightSearchConfig,
) -> np.ndarray:
    """Conservative absolute correlation of standalone component alpha paths."""
    factor_count = slices[0].components.shape[1]
    symbol_count = 1 + max(int(row.symbols.max()) for row in slices)
    layer_correlations: list[np.ndarray] = []
    for layer in TRAINING_LAYERS:
        previous = np.zeros((symbol_count, factor_count), dtype=np.bool_)
        previous_n = np.ones(factor_count, dtype=np.float64)
        active_rows: list[np.ndarray] = []
        for row in (item for item in slices if item.layer == layer):
            select_n = max(
                1,
                int(math.floor(len(row.forward_returns) * config.top_fraction)),
            )
            selected = np.argpartition(
                row.components, -select_n, axis=0
            )[-select_n:, :]
            gross = row.forward_returns[selected].mean(axis=0)
            current = np.zeros_like(previous)
            selected_symbols = row.symbols[selected]
            columns = np.broadcast_to(
                np.arange(factor_count), selected_symbols.shape
            )
            current[selected_symbols, columns] = True
            turnover = np.abs(
                current.astype(np.float64) / select_n
                - previous.astype(np.float64) / previous_n
            ).sum(axis=0)
            active_rows.append(
                gross
                - row.forward_returns.mean()
                - turnover * config.cost_bps / 10_000.0
            )
            previous = current
            previous_n.fill(select_n)
        paths = np.vstack(active_rows)
        correlation = np.corrcoef(paths, rowvar=False)
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(correlation, 1.0)
        layer_correlations.append(np.abs(correlation))
    conservative = np.max(np.stack(layer_correlations), axis=0)
    np.fill_diagonal(conservative, 1.0)
    return conservative


def _evaluate_weight_batch(
    slices: Sequence[CrossSectionSlice],
    weights: np.ndarray,
    config: WeightSearchConfig,
    factor_groups: Sequence[str],
    return_path_similarity: np.ndarray,
) -> list[dict]:
    candidate_count = len(weights)
    symbol_count = 1 + max(int(row.symbols.max()) for row in slices)
    layer_outputs: dict[str, dict[str, list[np.ndarray] | list[int]]] = {}
    tail_widths = _tail_widths(config)
    for layer in TRAINING_LAYERS:
        layer_slices = [row for row in slices if row.layer == layer]
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
        benchmark_rows: list[np.ndarray] = []
        ic_rows: list[np.ndarray] = []
        years: list[int] = []
        eras: list[int] = []
        for row in layer_slices:
            scores = row.components @ weights.T
            benchmark_rows.append(np.full(candidate_count, row.forward_returns.mean()))
            for width in tail_widths:
                select_n = max(
                    1,
                    int(math.floor(len(row.forward_returns) * width)),
                )
                selected = np.argpartition(
                    scores, -select_n, axis=0
                )[-select_n:, :]
                gross_rows[width].append(
                    row.forward_returns[selected].mean(axis=0)
                )
                current = np.zeros_like(previous[width])
                selected_symbols = row.symbols[selected]
                columns = np.broadcast_to(
                    np.arange(candidate_count), selected_symbols.shape
                )
                current[selected_symbols, columns] = True
                current_weight = current.astype(np.float64) / select_n
                prior_weight = (
                    previous[width].astype(np.float64) / previous_n[width]
                )
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
            eras.append(row.era)
        layer_outputs[layer] = {
            "gross": gross_rows,
            "benchmark": benchmark_rows,
            "turnover": turnover_rows,
            "ic": ic_rows,
            "years": years,
            "eras": eras,
        }

    metrics: dict[str, dict[str, np.ndarray]] = {}
    periods_per_year = 252.0 / config.horizon
    eval_cfg = evaluation_config("ashare")
    confidence = float(eval_cfg["return_lcb_confidence"])
    for layer, raw in layer_outputs.items():
        benchmark = np.vstack(raw["benchmark"])
        ic = np.vstack(raw["ic"])
        tail_active: dict[float, np.ndarray] = {}
        tail_stats: dict[float, dict[str, np.ndarray]] = {}
        for width in tail_widths:
            gross = np.vstack(raw["gross"][width])
            turnover = np.vstack(raw["turnover"][width])
            tail_active[width] = (
                gross
                - benchmark
                - turnover * config.cost_bps / 10_000.0
            )
            tail_stats[width] = _series_stats_matrix(
                tail_active[width], periods_per_year
            )
        active = tail_active[config.top_fraction]
        turnover = np.vstack(raw["turnover"][config.top_fraction])
        stressed = (
            np.vstack(raw["gross"][config.top_fraction])
            - benchmark
            - turnover * config.stress_cost_bps / 10_000.0
        )
        stats = _series_stats_matrix(active, periods_per_year)
        stress_stats = _series_stats_matrix(stressed, periods_per_year)
        ann_lcb, sharpe_lcb = _hac_lcb_matrix(active, periods_per_year, confidence)
        ic_mean = ic.mean(axis=0)
        ic_std = ic.std(axis=0, ddof=1)
        icir = np.divide(
            ic_mean * math.sqrt(periods_per_year),
            ic_std,
            out=np.zeros_like(ic_mean),
            where=ic_std > 1e-12,
        )
        profitable_year_rate = np.zeros(candidate_count)
        for year in sorted(set(raw["years"])):
            mask = np.asarray(raw["years"]) == year
            profitable_year_rate += (
                np.prod(np.maximum(1e-12, 1.0 + active[mask]), axis=0) > 1.0
            )
        profitable_year_rate /= max(1, len(set(raw["years"])))
        era_consistency = np.zeros(candidate_count)
        active_era_consistency = np.zeros(candidate_count)
        for era in sorted(set(raw["eras"])):
            mask = np.asarray(raw["eras"]) == era
            era_consistency += ic[mask].mean(axis=0) > 0
            active_era_consistency += (
                np.prod(np.maximum(1e-12, 1.0 + active[mask]), axis=0) > 1.0
            )
        era_consistency /= max(1, len(set(raw["eras"])))
        active_era_consistency /= max(1, len(set(raw["eras"])))

        fold_sharpes = []
        fold_ann_returns = []
        for indices in np.array_split(
            np.arange(len(active)), min(config.time_block_folds, len(active) // 2)
        ):
            if len(indices) < 2:
                continue
            fold_stats = _series_stats_matrix(active[indices], periods_per_year)
            fold_sharpes.append(fold_stats["sharpe"])
            fold_ann_returns.append(fold_stats["ann_return"])
        block_sharpe = np.vstack(fold_sharpes)
        block_ann_return = np.vstack(fold_ann_returns)
        worst_time_block_sharpe = block_sharpe.min(axis=0)
        worst_time_block_ann_return = block_ann_return.min(axis=0)
        positive_time_block_rate = (block_ann_return > 0).mean(axis=0)

        tail_ann_returns = np.vstack([
            tail_stats[width]["ann_return"] for width in tail_widths
        ])
        tail_sharpes = np.vstack([
            tail_stats[width]["sharpe"] for width in tail_widths
        ])
        tail_relations = [
            tail_ann_returns[index] >= tail_ann_returns[index + 1]
            for index in range(len(tail_widths) - 1)
        ]
        tail_relations.append(tail_ann_returns[-1] > 0)
        tail_monotonicity = np.vstack(tail_relations).mean(axis=0)
        positive_ic = ic > 0
        positive_ic_count = positive_ic.sum(axis=0)
        ic_tail_conversion_rate = np.divide(
            (positive_ic & (active > 0)).sum(axis=0),
            positive_ic_count,
            out=np.zeros(candidate_count, dtype=np.float64),
            where=positive_ic_count > 0,
        )
        metrics[layer] = {
            **stats,
            "ann_return_lcb": ann_lcb,
            "sharpe_lcb": sharpe_lcb,
            "ic_mean": ic_mean,
            "icir": icir,
            "stress_sharpe": stress_stats["sharpe"],
            "profitable_year_rate": profitable_year_rate,
            "era_consistency": era_consistency,
            "active_era_consistency": active_era_consistency,
            "worst_time_block_sharpe": worst_time_block_sharpe,
            "worst_time_block_ann_return": worst_time_block_ann_return,
            "positive_time_block_rate": positive_time_block_rate,
            "worst_tail_sharpe": tail_sharpes.min(axis=0),
            "tail_monotonicity": tail_monotonicity,
            "ic_tail_conversion_rate": ic_tail_conversion_rate,
            "avg_turnover": turnover.mean(axis=0),
        }

    public = metrics["INNER_PUBLIC"]
    gate = metrics["META_TRAIN"]
    worst_sharpe_lcb = np.minimum(public["sharpe_lcb"], gate["sharpe_lcb"])
    worst_icir = np.minimum(public["icir"], gate["icir"])
    worst_ann_lcb = np.minimum(public["ann_return_lcb"], gate["ann_return_lcb"])
    worst_stress = np.minimum(public["stress_sharpe"], gate["stress_sharpe"])
    worst_years = np.minimum(
        public["profitable_year_rate"], gate["profitable_year_rate"]
    )
    worst_eras = np.minimum(public["era_consistency"], gate["era_consistency"])
    worst_active_eras = np.minimum(
        public["active_era_consistency"], gate["active_era_consistency"]
    )
    worst_time_block_sharpe = np.minimum(
        public["worst_time_block_sharpe"], gate["worst_time_block_sharpe"]
    )
    worst_time_block_ann_return = np.minimum(
        public["worst_time_block_ann_return"],
        gate["worst_time_block_ann_return"],
    )
    worst_positive_time_block_rate = np.minimum(
        public["positive_time_block_rate"], gate["positive_time_block_rate"]
    )
    worst_tail_sharpe = np.minimum(
        public["worst_tail_sharpe"], gate["worst_tail_sharpe"]
    )
    worst_tail_monotonicity = np.minimum(
        public["tail_monotonicity"], gate["tail_monotonicity"]
    )
    worst_ic_tail_conversion = np.minimum(
        public["ic_tail_conversion_rate"], gate["ic_tail_conversion_rate"]
    )
    max_drawdown = np.maximum(public["max_drawdown"], gate["max_drawdown"])
    generalization_gap = np.abs(public["sharpe"] - gate["sharpe"])
    generalization_relative_gap = np.divide(
        generalization_gap,
        np.maximum(
            1.0,
            np.maximum(np.abs(public["sharpe"]), np.abs(gate["sharpe"])),
        ),
    )
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
        weights[:, [index for index, group in enumerate(factor_groups) if group == target]].sum(axis=1)
        for target in unique_groups
    ])
    active_group_count = np.sum(group_weights > 1e-12, axis=1)
    group_hhi = np.sum(group_weights * group_weights, axis=1)
    equal_group_hhi = 1.0 / active_group_count
    group_diversification = np.divide(
        1.0 - group_hhi,
        1.0 - equal_group_hhi,
        out=np.ones_like(group_hhi),
        where=active_group_count > 1,
    )
    maximum_group_weight = np.max(group_weights, axis=1)
    off_diagonal_similarity = return_path_similarity.copy()
    np.fill_diagonal(off_diagonal_similarity, 0.0)
    pair_weight = np.maximum(1e-12, 1.0 - hhi)
    weighted_return_path_similarity = np.einsum(
        "bi,ij,bj->b", weights, off_diagonal_similarity, weights
    ) / pair_weight
    return_source_independence = 1.0 - np.clip(
        weighted_return_path_similarity, 0.0, 1.0
    )
    robust_score = (
        0.22 * np.clip(worst_sharpe_lcb / 1.5, -1.0, 1.0)
        + 0.10 * np.clip(worst_ann_lcb / 0.10, -1.0, 1.0)
        + 0.12 * np.clip(worst_stress / 1.0, -1.0, 1.0)
        + 0.12 * np.clip(worst_time_block_sharpe / 1.0, -1.0, 1.0)
        + 0.10 * np.clip(worst_tail_sharpe / 1.0, -1.0, 1.0)
        + 0.07 * (2.0 * worst_active_eras - 1.0)
        + 0.05 * (2.0 * worst_years - 1.0)
        + 0.05 * (2.0 * worst_tail_monotonicity - 1.0)
        + 0.05 * (2.0 * worst_ic_tail_conversion - 1.0)
        + 0.03 * np.clip(worst_icir / 2.0, -1.0, 1.0)
        + 0.02 * np.clip(diversification, 0.0, 1.0)
        + 0.02 * np.clip(group_diversification, 0.0, 1.0)
        + 0.05 * np.clip(return_source_independence, 0.0, 1.0)
        + 0.05 * (1.0 - np.clip(max_drawdown / 0.35, 0.0, 2.0))
        - 0.10 * np.clip(generalization_relative_gap, 0.0, 1.0)
    )

    output: list[dict] = []
    for index, candidate in enumerate(weights):
        layer_rows = {}
        for layer in TRAINING_LAYERS:
            row = metrics[layer]
            layer_rows[layer] = {
                key: round(float(value[index]), 8)
                for key, value in row.items()
            }
        output.append({
            "weights": tuple(round(float(value), 8) for value in candidate),
            "active_factors": int(active_count[index]),
            "robust_score": round(float(robust_score[index]), 8),
            "worst_sharpe_lcb": round(float(worst_sharpe_lcb[index]), 8),
            "worst_icir": round(float(worst_icir[index]), 8),
            "worst_ann_return_lcb": round(float(worst_ann_lcb[index]), 8),
            "worst_stress_sharpe": round(float(worst_stress[index]), 8),
            "worst_profitable_year_rate": round(float(worst_years[index]), 8),
            "worst_era_consistency": round(float(worst_eras[index]), 8),
            "worst_active_era_consistency": round(
                float(worst_active_eras[index]), 8
            ),
            "worst_time_block_sharpe": round(
                float(worst_time_block_sharpe[index]), 8
            ),
            "worst_time_block_ann_return": round(
                float(worst_time_block_ann_return[index]), 8
            ),
            "worst_positive_time_block_rate": round(
                float(worst_positive_time_block_rate[index]), 8
            ),
            "worst_tail_sharpe": round(float(worst_tail_sharpe[index]), 8),
            "worst_tail_monotonicity": round(
                float(worst_tail_monotonicity[index]), 8
            ),
            "worst_ic_tail_conversion_rate": round(
                float(worst_ic_tail_conversion[index]), 8
            ),
            "max_drawdown": round(float(max_drawdown[index]), 8),
            "generalization_gap": round(float(generalization_gap[index]), 8),
            "generalization_relative_gap": round(
                float(generalization_relative_gap[index]), 8
            ),
            "weight_hhi": round(float(hhi[index]), 8),
            "effective_factor_count": round(float(1.0 / hhi[index]), 8),
            "diversification": round(float(diversification[index]), 8),
            "active_mechanism_groups": int(active_group_count[index]),
            "mechanism_weight_hhi": round(float(group_hhi[index]), 8),
            "effective_mechanism_count": round(float(1.0 / group_hhi[index]), 8),
            "mechanism_diversification": round(
                float(group_diversification[index]), 8
            ),
            "max_mechanism_weight": round(float(maximum_group_weight[index]), 8),
            "weighted_return_path_similarity": round(
                float(weighted_return_path_similarity[index]), 8
            ),
            "return_source_independence": round(
                float(return_source_independence[index]), 8
            ),
            "layers": layer_rows,
        })
    return output


def evaluate_weights(
    slices: Sequence[CrossSectionSlice],
    weights: Sequence[Sequence[float]],
    config: WeightSearchConfig,
    factor_groups: Sequence[str] | None = None,
    return_path_similarity: np.ndarray | None = None,
) -> list[dict]:
    """Evaluate arbitrary feasible weight vectors in bounded NumPy batches."""
    if not weights:
        return []
    matrix = np.asarray(weights, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("weights 必须是二维矩阵")
    if np.any(matrix < -1e-12) or not np.allclose(matrix.sum(axis=1), 1.0):
        raise ValueError("每组权重必须非负且合计为 1")
    groups = tuple(factor_groups or (f"factor_{index}" for index in range(matrix.shape[1])))
    if len(groups) != matrix.shape[1]:
        raise ValueError("factor_groups 长度必须等于因子数")
    if any(
        not _mechanism_weight_ok(row, groups, config.max_mechanism_weight)
        for row in matrix
    ):
        raise ValueError("候选权重超过收益机制族集中度上限")
    similarity = (
        np.asarray(return_path_similarity, dtype=np.float64)
        if return_path_similarity is not None
        else _component_return_path_similarity(slices, config)
    )
    if similarity.shape != (matrix.shape[1], matrix.shape[1]):
        raise ValueError("收益路径相似度矩阵维度与因子数量不一致")
    if any(
        not _return_path_pair_ok(
            row, similarity, config.max_active_pair_similarity
        )
        for row in matrix
    ):
        raise ValueError("候选同时启用了训练期收益路径过度相似的因子")
    output: list[dict] = []
    for start in range(0, len(matrix), config.batch_size):
        output.extend(_evaluate_weight_batch(
            slices,
            matrix[start:start + config.batch_size],
            config,
            groups,
            similarity,
        ))
    return output


def _weight_key(weights: Iterable[float]) -> tuple[int, ...]:
    return tuple(round(float(value) * 100_000_000) for value in weights)


def _mechanism_weight_ok(
    weights: Sequence[float],
    factor_groups: Sequence[str],
    maximum: float,
) -> bool:
    totals: dict[str, float] = {}
    for group, weight in zip(factor_groups, weights):
        totals[str(group)] = totals.get(str(group), 0.0) + float(weight)
    return max(totals.values(), default=0.0) <= maximum + 1e-12


def _return_path_pair_ok(
    weights: Sequence[float],
    similarity: np.ndarray,
    maximum: float,
) -> bool:
    active = np.flatnonzero(np.asarray(weights, dtype=np.float64) > 1e-12)
    return all(
        float(similarity[left, right]) <= maximum + 1e-12
        for left, right in itertools.combinations(active, 2)
    )


def _refine_neighbors(
    weights: Sequence[float],
    config: WeightSearchConfig,
    factor_groups: Sequence[str],
) -> list[tuple[float, ...]]:
    current = np.asarray(weights, dtype=np.float64)
    support = np.flatnonzero(current > 1e-12)
    output = []
    for donor in support:
        for receiver in support:
            if donor == receiver:
                continue
            candidate = current.copy()
            candidate[donor] -= config.refine_step
            candidate[receiver] += config.refine_step
            if (
                candidate[donor] + 1e-12 < config.min_active_weight
                or candidate[receiver] > config.max_weight + 1e-12
            ):
                continue
            candidate = np.round(candidate, 10)
            row = tuple(float(value) for value in candidate)
            if _mechanism_weight_ok(
                row, factor_groups, config.max_mechanism_weight
            ):
                output.append(row)
    return sorted(set(output))


def search_optimal_weights(
    slices: Sequence[CrossSectionSlice],
    factor_count: int,
    config: WeightSearchConfig,
    factor_groups: Sequence[str] | None = None,
) -> dict:
    """Complete coarse enumeration followed by deterministic local refinement."""
    config.validate(factor_count)
    groups = tuple(factor_groups or (f"factor_{index}" for index in range(factor_count)))
    return_path_similarity = _component_return_path_similarity(slices, config)
    coarse_unfiltered = enumerate_coarse_weights(factor_count, config, groups)
    coarse = [
        row
        for row in coarse_unfiltered
        if _return_path_pair_ok(
            row, return_path_similarity, config.max_active_pair_similarity
        )
    ]
    if not coarse:
        raise ValueError("收益路径相似度硬约束过滤了全部候选组合")
    coarse_results = evaluate_weights(
        slices, coarse, config, groups, return_path_similarity
    )
    all_results = {_weight_key(row["weights"]): row for row in coarse_results}
    ranked = sorted(coarse_results, key=lambda row: row["robust_score"], reverse=True)
    starts = ranked[:config.refine_starts]
    for size in range(config.min_factors, config.max_factors + 1):
        best = next((row for row in ranked if row["active_factors"] == size), None)
        if best is not None:
            starts.append(best)

    for start in starts:
        incumbent = start
        for _ in range(config.refine_iterations):
            unseen = [
                candidate for candidate in _refine_neighbors(
                    incumbent["weights"], config, groups
                )
                if _weight_key(candidate) not in all_results
            ]
            if unseen:
                for row in evaluate_weights(
                    slices, unseen, config, groups, return_path_similarity
                ):
                    all_results[_weight_key(row["weights"])] = row
            neighborhood = [
                all_results[_weight_key(candidate)]
                for candidate in _refine_neighbors(
                    incumbent["weights"], config, groups
                )
                if _weight_key(candidate) in all_results
            ]
            candidate_best = max(
                [incumbent, *neighborhood],
                key=lambda row: row["robust_score"],
            )
            if candidate_best["robust_score"] <= incumbent["robust_score"] + 1e-12:
                break
            incumbent = candidate_best

    final = sorted(
        all_results.values(),
        key=lambda row: (
            row["robust_score"],
            row["worst_sharpe_lcb"],
            row["worst_icir"],
            row["effective_factor_count"],
        ),
        reverse=True,
    )
    for rank, row in enumerate(final, start=1):
        row["rank"] = rank
    score_tolerance = 0.01
    near_optimal = [
        row
        for row in final
        if row["robust_score"] >= final[0]["robust_score"] - score_tolerance
    ]
    near_weights = np.asarray(
        [row["weights"] for row in near_optimal], dtype=np.float64
    )
    return {
        "protocol": COMBINATION_PROTOCOL,
        "llm_used": False,
        "search_scope": "INNER_PUBLIC_plus_META_TRAIN_only",
        "holdout_or_vault_read": False,
        "config": asdict(config),
        "factor_groups": list(groups),
        "objective_semantics": (
            "executable_active_return_lcb_plus_cost_stress_plus_contiguous_"
            "time_blocks_plus_tail_width_robustness_plus_return_source_"
            "independence_plus_correlated_pair_exclusion_v2"
        ),
        "tail_widths": list(_tail_widths(config)),
        "return_path_similarity_matrix": return_path_similarity.round(8).tolist(),
        "coarse_candidates_before_structural_filter": len(coarse_unfiltered),
        "coarse_candidates": len(coarse_results),
        "structurally_rejected_coarse_candidates": (
            len(coarse_unfiltered) - len(coarse)
        ),
        "evaluated_candidates": len(final),
        "search_stability": {
            "score_tolerance": score_tolerance,
            "near_optimal_candidates": len(near_optimal),
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


def composite_expression(
    components: Sequence[FactorComponent],
    weights: Sequence[float],
) -> str:
    """Human/audit DSL representation; the component list remains authoritative."""
    terms = []
    for component, weight in zip(components, weights):
        if weight <= 1e-12:
            continue
        signed_weight = float(weight) * component.direction
        terms.append(f"({signed_weight:.8f})*({component.expression})")
    expression = "+".join(terms)
    if len(expression) > 500:
        raise ValueError("组合后的审计表达式超过 DSL 500 字符上限")
    return expression


def apply_composite_weights(
    frame: pl.DataFrame,
    weights: Sequence[float],
) -> pl.DataFrame:
    score = pl.lit(0.0)
    for index, weight in enumerate(weights):
        score += pl.col(f"component_{index}") * float(weight)
    return frame.with_columns(score.alias("factor"))


def combination_promotion_gate(search_result: dict) -> dict:
    """Decide whether a training winner may advance to private validation.

    This gate intentionally does not inspect META_HOLDOUT or FACTOR_VAULT. It
    only checks training-safe event replays and structural diversification.
    Passing still means *private-validation candidate*, never production-ready.
    """
    event = search_result.get("event_verification") or {}
    scenarios = event.get("scenarios") or {}
    if any(layer not in scenarios for layer in TRAINING_LAYERS):
        return {
            "decision": "NOT_EVALUATED",
            "eligible_for_private_validation": False,
            "production_eligible": False,
            "reason": "training-safe event replay is missing",
            "policy": dict(PROMOTION_POLICY),
            "rules": {},
        }
    event_15 = [scenarios[layer].get("15") or {} for layer in TRAINING_LAYERS]
    matrix = np.asarray(
        search_result.get("return_path_similarity_matrix") or [],
        dtype=np.float64,
    )
    best = search_result.get("best") or {}
    weights = np.asarray(best.get("weights") or [], dtype=np.float64)
    active = np.flatnonzero(weights > 1e-12)
    if matrix.ndim == 2 and len(active) > 1:
        off_diagonal = matrix.copy()
        np.fill_diagonal(off_diagonal, 0.0)
        max_pairwise_similarity = max(
            float(off_diagonal[left, right])
            for left, right in itertools.combinations(active, 2)
        )
    else:
        max_pairwise_similarity = 0.0
    observed = {
        "worst_event_sharpe_15bps": min(
            float(row.get("sharpe") or 0.0) for row in event_15
        ),
        "max_event_drawdown_15bps": max(
            float(row.get("max_drawdown") or 1.0) for row in event_15
        ),
        "max_active_pairwise_return_path_similarity": max_pairwise_similarity,
        "return_source_independence": float(
            best.get("return_source_independence") or 0.0
        ),
        "worst_tail_sharpe": float(best.get("worst_tail_sharpe") or 0.0),
        "event_integrity_all_pass": all(
            bool((row.get("integrity") or {}).get("all_pass")) for row in event_15
        ),
    }
    rules = {
        "event_integrity": observed["event_integrity_all_pass"],
        "event_sharpe": (
            observed["worst_event_sharpe_15bps"]
            >= PROMOTION_POLICY["min_event_sharpe_15bps"]
        ),
        "event_drawdown": (
            observed["max_event_drawdown_15bps"]
            <= PROMOTION_POLICY["max_event_drawdown_15bps"]
        ),
        "active_pairwise_return_path_similarity": (
            observed["max_active_pairwise_return_path_similarity"]
            <= PROMOTION_POLICY["max_active_pairwise_return_path_similarity"]
        ),
        "return_source_independence": (
            observed["return_source_independence"]
            >= PROMOTION_POLICY["min_return_source_independence"]
        ),
        "tail_robustness": (
            observed["worst_tail_sharpe"]
            >= PROMOTION_POLICY["min_worst_tail_sharpe"]
        ),
    }
    eligible = all(rules.values())
    return {
        "decision": (
            "PRIVATE_VALIDATION_CANDIDATE_NON_PIT"
            if eligible
            else "RESEARCH_ONLY_BLOCKED"
        ),
        "eligible_for_private_validation": eligible,
        "production_eligible": False,
        "policy": dict(PROMOTION_POLICY),
        "observed": {
            key: round(value, 8) if isinstance(value, float) else value
            for key, value in observed.items()
        },
        "rules": rules,
        "failed_rules": [key for key, passed in rules.items() if not passed],
    }
