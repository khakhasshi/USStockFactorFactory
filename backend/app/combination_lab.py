"""Auditable factor-combination laboratory for programmatic and LLM searches.

The component list is authoritative.  A combination is never flattened into
one DSL string, so long portfolios retain per-component direction, provenance,
mechanism and cross-sectional ranking semantics.  LLMs may propose bounded
subsets/weight priors but every proposal is validated and scored by the same
deterministic engine as the programmatic baseline.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import polars as pl

from .backtest.batch import (
    BatchBacktestSpec,
    information_coefficients,
    run_cost_scenarios,
)
from .backtest.engine import _write_artifacts
from .backtest.vector import run_vector_cost_scenarios
from .config import BACKTEST_ARTIFACT_ROOT, get_dsl_fields
from .data.panel import PanelStore
from .dsl.engine import parse, required_history, validate
from .factors.diversity import infer_mechanism
from .llm.client import chat, extract_json, mark_validation


COMBINATION_LAB_PROTOCOL = "combination_lab_nested_v1"
MAX_COMPONENTS = 12
MAX_PATH_BUDGET = 20_000
TERMINAL_STATUSES = {"done", "error", "stopped", "interrupted"}


class CombinationCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class LabSlice:
    trade_date: date
    symbols: np.ndarray
    components: np.ndarray
    forward_returns: np.ndarray
    return_ranks: np.ndarray


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: Any, length: int = 16) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()[:length]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _rank_average(values: np.ndarray) -> np.ndarray:
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


def _as_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{label} 必须是 YYYY-MM-DD") from exc


def validate_lab_spec(spec: dict) -> dict:
    """Validate and canonicalise one immutable experiment request."""
    market = str(spec.get("market") or "us")
    if market not in {"us", "ashare"}:
        raise ValueError("market 必须是 us 或 ashare")
    portfolio_mode = str(
        spec.get("portfolio_mode")
        or ("long_only" if market == "ashare" else "long_short")
    )
    if portfolio_mode not in {"long_only", "long_short"}:
        raise ValueError("portfolio_mode 必须是 long_only 或 long_short")
    if market == "ashare" and portfolio_mode != "long_only":
        raise ValueError("A股组合优化仅支持 long_only")
    search_mode = str(spec.get("search_mode") or "programmatic")
    if search_mode not in {"programmatic", "llm"}:
        raise ValueError("search_mode 必须是 programmatic 或 llm")

    raw_components = list(spec.get("components") or [])
    if not 2 <= len(raw_components) <= MAX_COMPONENTS:
        raise ValueError(f"候选因子数量必须在 2 到 {MAX_COMPONENTS} 之间")
    fields = get_dsl_fields(market)
    components: list[dict] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_components):
        expression = str(raw.get("expression") or "").strip()
        error = validate(expression, fields)
        if error:
            raise ValueError(f"因子 {index + 1} 表达式非法: {error}")
        direction = int(raw.get("direction", 1))
        if direction not in {-1, 1}:
            raise ValueError(f"因子 {index + 1} direction 必须是 1 或 -1")
        key = str(raw.get("key") or f"C{index + 1:02d}").strip()[:64]
        if not key or key in seen:
            raise ValueError("因子 key 必须非空且不重复")
        seen.add(key)
        mechanism = str(raw.get("mechanism") or infer_mechanism(expression)).strip()
        components.append({
            "key": key,
            "name": str(raw.get("name") or key).strip()[:128],
            "expression": expression,
            "direction": direction,
            "mechanism": mechanism or "unknown",
            "source": str(raw.get("source") or "manual").strip()[:64],
            "source_ref": str(raw.get("source_ref") or "").strip()[:256],
            "expression_hash": _fingerprint({"expression": expression, "direction": direction}),
            "required_history": required_history(expression),
        })

    min_factors = int(spec.get("min_factors", 2))
    max_factors = int(spec.get("max_factors", min(5, len(components))))
    if not 2 <= min_factors <= max_factors <= len(components):
        raise ValueError("组合数量约束必须满足 2 <= 最小组合数 <= 最大组合数 <= N")
    coarse_step = float(spec.get("coarse_step", 0.10))
    if coarse_step not in {0.05, 0.10, 0.20, 0.25}:
        raise ValueError("coarse_step 仅支持 0.05/0.10/0.20/0.25")
    units = round(1.0 / coarse_step)
    if not math.isclose(units * coarse_step, 1.0, abs_tol=1e-9):
        raise ValueError("coarse_step 必须整除1")
    min_weight = float(spec.get("min_weight", 0.05))
    max_weight = float(spec.get("max_weight", 0.65))
    max_mechanism_weight = float(spec.get("max_mechanism_weight", 0.70))
    if not 0 < min_weight <= max_weight <= 1:
        raise ValueError("权重约束必须满足 0 < min_weight <= max_weight <= 1")
    if not 0 < max_mechanism_weight <= 1:
        raise ValueError("max_mechanism_weight 必须位于 (0,1]")
    if min_factors * max_weight < 1 - 1e-12 and max_factors * max_weight < 1 - 1e-12:
        raise ValueError("max_weight 太低，没有可行组合")
    if min_factors * min_weight > 1 + 1e-12:
        raise ValueError("min_weight 太高，没有可行组合")
    min_mechanisms = int(spec.get("min_mechanisms", min(2, len({r['mechanism'] for r in components}))))
    if not 1 <= min_mechanisms <= len({r["mechanism"] for r in components}):
        raise ValueError("min_mechanisms 超出候选收益机制数量")

    train_start = _as_date(spec.get("train_start", "2010-01-01"), "train_start")
    train_end = _as_date(spec.get("train_end", "2018-12-31"), "train_end")
    validation_start = _as_date(spec.get("validation_start", "2019-01-01"), "validation_start")
    validation_end = _as_date(spec.get("validation_end", "2022-12-31"), "validation_end")
    rating_start = _as_date(spec.get("rating_start", "2023-01-01"), "rating_start")
    rating_end = _as_date(spec.get("rating_end", "2026-12-31"), "rating_end")
    if not train_start <= train_end < validation_start <= validation_end < rating_start <= rating_end:
        raise ValueError("时间必须严格满足 训练 < 验证 < 冻结评级，且三个区间不得重叠")

    horizon = int(spec.get("horizon", 5))
    if horizon not in {1, 5, 20}:
        raise ValueError("horizon 仅支持 1/5/20")
    universe_n = int(spec.get("universe_n", 500))
    if not 100 <= universe_n <= 5000:
        raise ValueError("universe_n 必须在100到5000之间")
    top_fraction = float(spec.get("top_fraction", 0.20))
    if not 0.05 <= top_fraction <= 0.50:
        raise ValueError("top_fraction 必须位于[0.05,0.50]")
    path_budget = int(spec.get("path_budget", 3000))
    if not 10 <= path_budget <= MAX_PATH_BUDGET:
        raise ValueError(f"path_budget 必须在10到{MAX_PATH_BUDGET}之间")
    cost_bps = float(spec.get("cost_bps", 15.0))
    stress_cost_bps = float(spec.get("stress_cost_bps", 50.0))
    if not 0 <= cost_bps <= stress_cost_bps <= 500:
        raise ValueError("成本必须满足 0 <= cost_bps <= stress_cost_bps <= 500")
    borrow_cost = float(
        spec.get("borrow_cost_bps_annual", 300.0 if portfolio_mode == "long_short" else 0.0)
    )
    if borrow_cost < 0:
        raise ValueError("borrow_cost_bps_annual 不能为负")
    if portfolio_mode == "long_only":
        borrow_cost = 0.0

    canonical = {
        "protocol": COMBINATION_LAB_PROTOCOL,
        "name": str(spec.get("name") or "组合优化实验").strip()[:128],
        "experiment_id": int(spec.get("experiment_id") or 1),
        "search_mode": search_mode,
        "market": market,
        "portfolio_mode": portfolio_mode,
        "panel_glob": str(spec.get("panel_glob") or "").strip() or None,
        "components": components,
        "min_factors": min_factors,
        "max_factors": max_factors,
        "min_mechanisms": min_mechanisms,
        "coarse_step": coarse_step,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "max_mechanism_weight": max_mechanism_weight,
        "max_pair_correlation": float(spec.get("max_pair_correlation", 0.85)),
        "path_budget": path_budget,
        "validation_budget": min(500, max(50, int(spec.get("validation_budget", 250)))),
        "top_k": min(25, max(3, int(spec.get("top_k", 10)))),
        "universe_n": universe_n,
        "top_fraction": top_fraction,
        "horizon": horizon,
        "rebalance_every": horizon,
        "cost_bps": cost_bps,
        "stress_cost_bps": stress_cost_bps,
        "borrow_cost_bps_annual": borrow_cost,
        "train_start": str(train_start),
        "train_end": str(train_end),
        "validation_start": str(validation_start),
        "validation_end": str(validation_end),
        "rating_start": str(rating_start),
        "rating_end": str(rating_end),
        "llm_max_proposals": min(30, max(1, int(spec.get("llm_max_proposals", 8)))),
        "initial_capital": float(spec.get("initial_capital", 1_000_000.0)),
        "max_volume_participation": float(spec.get("max_volume_participation", 0.05)),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if not 0 <= canonical["max_pair_correlation"] <= 1:
        raise ValueError("max_pair_correlation 必须位于[0,1]")
    canonical["snapshot_hash"] = _fingerprint({k: v for k, v in canonical.items() if k != "created_at"}, 32)
    return canonical


def _integer_compositions(total: int, count: int, minimum: int, maximum: int):
    if count == 1:
        if minimum <= total <= maximum:
            yield (total,)
        return
    low = max(minimum, total - maximum * (count - 1))
    high = min(maximum, total - minimum * (count - 1))
    for head in range(low, high + 1):
        for tail in _integer_compositions(total - head, count - 1, minimum, maximum):
            yield (head, *tail)


def _weight_constraints_ok(weights: Sequence[float], spec: dict) -> bool:
    active = [index for index, weight in enumerate(weights) if weight > 1e-12]
    if not spec["min_factors"] <= len(active) <= spec["max_factors"]:
        return False
    if any(
        weights[index] < spec["min_weight"] - 1e-12
        or weights[index] > spec["max_weight"] + 1e-12
        for index in active
    ):
        return False
    mechanisms: dict[str, float] = {}
    for index in active:
        mechanism = spec["components"][index]["mechanism"]
        mechanisms[mechanism] = mechanisms.get(mechanism, 0.0) + float(weights[index])
    return (
        len(mechanisms) >= spec["min_mechanisms"]
        and max(mechanisms.values(), default=0.0) <= spec["max_mechanism_weight"] + 1e-12
    )


def parse_llm_weight_proposals(payload: dict | None, spec: dict) -> list[dict]:
    """Validate bounded LLM proposals; expressions and directions cannot change."""
    if not payload:
        return []
    by_key = {row["key"]: index for index, row in enumerate(spec["components"])}
    output: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for raw in list(payload.get("proposals") or [])[: spec["llm_max_proposals"]]:
        raw_weights = raw.get("weights") or {}
        if not isinstance(raw_weights, dict):
            continue
        weights = np.zeros(len(by_key), dtype=np.float64)
        valid = True
        for key, value in raw_weights.items():
            if key not in by_key:
                valid = False
                break
            weights[by_key[key]] = max(0.0, _safe_float(value))
        total = float(weights.sum())
        if not valid or total <= 1e-12:
            continue
        weights /= total
        row = tuple(float(round(value, 10)) for value in weights)
        fingerprint = tuple(round(value * 100_000) for value in row)
        if fingerprint in seen or not _weight_constraints_ok(row, spec):
            continue
        seen.add(fingerprint)
        output.append({
            "weights": row,
            "source": "llm_proposal",
            "rationale": str(raw.get("rationale") or "")[:1000],
            "hypothesis": str(raw.get("hypothesis") or "")[:1000],
        })
    return output


def enumerate_weight_candidates(spec: dict, llm_payload: dict | None = None) -> list[dict]:
    """Deterministic, budget-bounded subset and simplex enumeration."""
    count = len(spec["components"])
    candidates: list[dict] = []
    seen: set[tuple[int, ...]] = set()

    def add(weights: Sequence[float], source: str, **meta: Any) -> bool:
        row = tuple(float(round(value, 10)) for value in weights)
        key = tuple(round(value * 100_000) for value in row)
        if key in seen or not math.isclose(sum(row), 1.0, abs_tol=1e-8):
            return False
        if not _weight_constraints_ok(row, spec):
            return False
        seen.add(key)
        candidates.append({"weights": row, "source": source, **meta})
        return True

    for proposal in parse_llm_weight_proposals(llm_payload, spec):
        add(proposal["weights"], proposal["source"], rationale=proposal["rationale"], hypothesis=proposal["hypothesis"])

    subsets = [
        subset
        for size in range(spec["min_factors"], spec["max_factors"] + 1)
        for subset in itertools.combinations(range(count), size)
    ]
    for subset in subsets:
        weights = [0.0] * count
        for index in subset:
            weights[index] = 1.0 / len(subset)
        add(weights, "subset_equal")
    if len(candidates) > spec["path_budget"]:
        raise ValueError(
            f"仅等权子集就有 {len(candidates)} 条，超过预算 {spec['path_budget']}；请缩小N或组合范围"
        )

    step = spec["coarse_step"]
    total_units = round(1.0 / step)
    min_units = max(1, math.ceil(spec["min_weight"] / step - 1e-12))
    max_units = math.floor(spec["max_weight"] / step + 1e-12)
    generators: list[Iterable[tuple[int, ...]]] = [
        iter(_integer_compositions(total_units, len(subset), min_units, max_units))
        for subset in subsets
    ]
    active = list(range(len(subsets)))
    while active and len(candidates) < spec["path_budget"]:
        next_active = []
        for generator_index in active:
            try:
                allocation = next(generators[generator_index])
            except StopIteration:
                continue
            subset = subsets[generator_index]
            weights = [0.0] * count
            for index, units in zip(subset, allocation):
                weights[index] = units / total_units
            add(weights, "coarse_simplex")
            next_active.append(generator_index)
            if len(candidates) >= spec["path_budget"]:
                break
        active = next_active
    if not candidates:
        raise ValueError("约束过滤后没有可行组合")
    return candidates


def _progress(callback: Callable[[dict], None] | None, **payload: Any) -> None:
    if callback:
        callback({"updated_at": datetime.now(timezone.utc).isoformat(), **payload})


def _check_cancel(event: threading.Event | None) -> None:
    if event is not None and event.is_set():
        raise CombinationCancelled("用户停止了组合优化实验")


def _materialize_component_frame(spec: dict) -> tuple[pl.DataFrame, dict, dict]:
    fields = get_dsl_fields(spec["market"])
    store = PanelStore.get(spec.get("panel_glob"), spec["market"], factor_fields=fields)
    panel, dates, identity, generation = store.read_snapshot()
    requested_start = _as_date(spec["train_start"], "train_start")
    requested_end = _as_date(spec["rating_end"], "rating_end")
    available_start, available_end = min(dates), max(dates)
    effective_start = max(requested_start, available_start)
    effective_end = min(requested_end, available_end)
    if effective_start > _as_date(spec["train_end"], "train_end"):
        raise ValueError("面板不覆盖训练区间")
    if effective_end < _as_date(spec["rating_start"], "rating_start"):
        raise ValueError("面板不覆盖冻结评级区间")
    max_history = max(row["required_history"] for row in spec["components"])
    first_index = dates.index(next(value for value in dates if value >= effective_start))
    history_start = dates[max(0, first_index - max_history - 2)]

    lazy = panel.lazy().filter(pl.col("trade_date").is_between(history_start, effective_end))
    score_columns = []
    for index, component in enumerate(spec["components"]):
        raw = f"_combination_raw_{index}"
        oriented = f"_combination_oriented_{index}"
        score = f"component_{index}"
        lazy = parse(component["expression"], fields).apply(lazy, alias=raw)
        lazy = lazy.with_columns(
            pl.when(pl.col("univ_rank") <= spec["universe_n"])
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
    symbols = (
        panel.lazy()
        .filter(
            pl.col("trade_date").is_between(effective_start, effective_end)
            & (pl.col("univ_rank") <= spec["universe_n"])
        )
        .select("ts_code").unique().collect()["ts_code"].to_list()
    )
    forward = f"fwd_{spec['horizon']}"
    columns = [
        "trade_date", "ts_code", "name", "univ_rank", "raw_open", "raw_close",
        "vol", "amount", "adjustment_factor", "can_buy_open_proxy",
        "can_sell_open_proxy", forward, *score_columns,
    ]
    frame = (
        lazy.select(columns).cache()
        .filter(pl.col("trade_date").is_between(effective_start, effective_end))
        .filter(pl.col("ts_code").is_in(symbols))
        .sort("trade_date", "ts_code")
        .collect(optimizations=pl.QueryOptFlags(predicate_pushdown=False))
    )
    effective = {
        "train_start": str(effective_start),
        "train_end": str(min(_as_date(spec["train_end"], "train_end"), effective_end)),
        "validation_start": spec["validation_start"],
        "validation_end": str(min(_as_date(spec["validation_end"], "validation_end"), effective_end)),
        "rating_start": spec["rating_start"],
        "rating_end": str(effective_end),
    }
    return frame, {
        "identity": identity,
        "generation": generation,
        "summary": store.summary(),
        "history_start": str(history_start),
        "rows": frame.height,
        "sessions": frame["trade_date"].n_unique(),
        "symbols": frame["ts_code"].n_unique(),
    }, effective


def _build_slices(
    frame: pl.DataFrame,
    *,
    start: str,
    end: str,
    spec: dict,
) -> list[LabSlice]:
    start_date, end_date = _as_date(start, "window_start"), _as_date(end, "window_end")
    component_columns = [f"component_{index}" for index in range(len(spec["components"]))]
    dates = (
        frame.lazy().filter(pl.col("trade_date").is_between(start_date, end_date))
        .select("trade_date").unique().sort("trade_date").collect()["trade_date"].to_list()
    )
    usable = dates[: max(0, len(dates) - spec["horizon"] - 1)]
    signal_dates = usable[:: spec["horizon"]]
    if len(signal_dates) < 8:
        raise ValueError(f"{start}~{end} 有效非重叠截面不足8个")
    eligible = frame.filter(
        pl.col("trade_date").is_in(signal_dates)
        & (pl.col("univ_rank") <= spec["universe_n"])
        & pl.col(f"fwd_{spec['horizon']}").is_finite()
        & pl.col("raw_close").is_finite()
        & (pl.col("raw_close") > 0)
        & pl.all_horizontal(pl.col(column).is_finite() for column in component_columns)
    )
    symbols = sorted(str(value) for value in eligible["ts_code"].unique().to_list())
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    output = []
    for group in eligible.partition_by("trade_date", maintain_order=True):
        if group.height < 50:
            continue
        returns = group[f"fwd_{spec['horizon']}"] .to_numpy().astype(np.float64)
        output.append(LabSlice(
            trade_date=group["trade_date"][0],
            symbols=np.asarray([symbol_index[str(value)] for value in group["ts_code"].to_list()], dtype=np.int32),
            components=group.select(component_columns).to_numpy().astype(np.float64),
            forward_returns=returns,
            return_ranks=_rank_average(returns),
        ))
    if len(output) < 8:
        raise ValueError(f"{start}~{end} 有效截面不足8个")
    return output


def _series_stats_matrix(returns: np.ndarray, periods_per_year: float) -> dict[str, np.ndarray]:
    mean = returns.mean(axis=0)
    std = returns.std(axis=0, ddof=1) if len(returns) > 1 else np.zeros(returns.shape[1])
    sharpe = np.divide(mean * math.sqrt(periods_per_year), std, out=np.zeros_like(mean), where=std > 1e-12)
    nav = np.cumprod(np.maximum(1e-12, 1.0 + returns), axis=0)
    peaks = np.maximum.accumulate(np.vstack([np.ones((1, returns.shape[1])), nav]), axis=0)[1:]
    max_drawdown = np.max(1.0 - nav / peaks, axis=0)
    ann_return = np.power(nav[-1], periods_per_year / len(returns)) - 1.0
    return {"ann_return": ann_return, "sharpe": sharpe, "max_drawdown": max_drawdown}


def _normal_p_values(means: np.ndarray, stds: np.ndarray, n: int) -> np.ndarray:
    t_values = np.divide(means, stds / math.sqrt(max(1, n)), out=np.zeros_like(means), where=stds > 1e-12)
    return np.asarray([math.erfc(abs(float(value)) / math.sqrt(2.0)) for value in t_values])


def _evaluate_candidates(
    slices: Sequence[LabSlice],
    candidates: Sequence[dict],
    spec: dict,
    *,
    progress_callback: Callable[[dict], None] | None,
    cancel_event: threading.Event | None,
    stage: str,
) -> list[dict]:
    weights = np.asarray([row["weights"] for row in candidates], dtype=np.float64)
    periods_per_year = 252.0 / spec["horizon"]
    symbol_count = 1 + max(int(row.symbols.max()) for row in slices)
    batch_size = 128
    output: list[dict] = []
    for batch_start in range(0, len(weights), batch_size):
        _check_cancel(cancel_event)
        batch = weights[batch_start:batch_start + batch_size]
        candidate_count = len(batch)
        previous = np.zeros((symbol_count, candidate_count), dtype=np.float64)
        return_rows, turnover_rows, ic_rows, rank_ic_rows = [], [], [], []
        years = []
        for row in slices:
            scores = row.components @ batch.T
            n_select = max(1, int(math.floor(len(row.forward_returns) * spec["top_fraction"])))
            long_idx = np.argpartition(scores, -n_select, axis=0)[-n_select:, :]
            columns = np.broadcast_to(np.arange(candidate_count), long_idx.shape)
            long_returns = row.forward_returns[long_idx].mean(axis=0)
            current = np.zeros_like(previous)
            current[row.symbols[long_idx], columns] = 1.0 / n_select
            if spec["portfolio_mode"] == "long_short":
                short_idx = np.argpartition(scores, n_select - 1, axis=0)[:n_select, :]
                short_columns = np.broadcast_to(np.arange(candidate_count), short_idx.shape)
                short_returns = row.forward_returns[short_idx].mean(axis=0)
                current[row.symbols[short_idx], short_columns] = -1.0 / n_select
                gross = long_returns - short_returns
                benchmark = 0.0
            else:
                gross = long_returns
                benchmark = float(row.forward_returns.mean())
            turnover = np.abs(current - previous).sum(axis=0)
            borrow = (
                spec["borrow_cost_bps_annual"] / 10_000.0 * spec["horizon"] / 252.0
                if spec["portfolio_mode"] == "long_short" else 0.0
            )
            net = gross - turnover * spec["cost_bps"] / 10_000.0 - borrow - benchmark
            return_rows.append(net)
            turnover_rows.append(turnover)
            previous = current

            score_centered = scores - scores.mean(axis=0)
            return_centered = row.forward_returns - row.forward_returns.mean()
            denominator = np.sqrt(
                (score_centered * score_centered).sum(axis=0)
                * np.sum(return_centered * return_centered)
            )
            ic_rows.append(np.divide(
                (score_centered * return_centered[:, None]).sum(axis=0),
                denominator,
                out=np.zeros(candidate_count),
                where=denominator > 1e-12,
            ))
            score_ranks = np.argsort(np.argsort(scores, axis=0, kind="mergesort"), axis=0, kind="mergesort").astype(np.float64)
            score_rank_centered = score_ranks - score_ranks.mean(axis=0)
            return_rank_centered = row.return_ranks - row.return_ranks.mean()
            rank_denominator = np.sqrt(
                (score_rank_centered * score_rank_centered).sum(axis=0)
                * np.sum(return_rank_centered * return_rank_centered)
            )
            rank_ic_rows.append(np.divide(
                (score_rank_centered * return_rank_centered[:, None]).sum(axis=0),
                rank_denominator,
                out=np.zeros(candidate_count),
                where=rank_denominator > 1e-12,
            ))
            years.append(row.trade_date.year)

        returns = np.vstack(return_rows)
        turnover = np.vstack(turnover_rows)
        ic = np.vstack(ic_rows)
        rank_ic = np.vstack(rank_ic_rows)
        stats = _series_stats_matrix(returns, periods_per_year)
        stressed_stats = _series_stats_matrix(
            returns - turnover * (spec["stress_cost_bps"] - spec["cost_bps"]) / 10_000.0,
            periods_per_year,
        )
        ic_mean, ic_std = ic.mean(axis=0), ic.std(axis=0, ddof=1)
        rank_mean, rank_std = rank_ic.mean(axis=0), rank_ic.std(axis=0, ddof=1)
        annualizer = math.sqrt(periods_per_year)
        icir = np.divide(ic_mean * annualizer, ic_std, out=np.zeros_like(ic_mean), where=ic_std > 1e-12)
        rank_icir = np.divide(rank_mean * annualizer, rank_std, out=np.zeros_like(rank_mean), where=rank_std > 1e-12)
        p_values = _normal_p_values(rank_mean, rank_std, len(slices))
        unique_years = sorted(set(years))
        year_sharpes = []
        for year in unique_years:
            mask = np.asarray([value == year for value in years])
            year_sharpes.append(_series_stats_matrix(returns[mask], periods_per_year)["sharpe"])
        year_matrix = np.vstack(year_sharpes)
        hhi = (batch * batch).sum(axis=1)
        for local_index in range(candidate_count):
            source = candidates[batch_start + local_index]
            active_mechanisms = {
                spec["components"][index]["mechanism"]
                for index, value in enumerate(batch[local_index]) if value > 1e-12
            }
            worst_year = float(year_matrix[:, local_index].min())
            positive_year_rate = float((year_matrix[:, local_index] > 0).mean())
            robust_score = (
                0.30 * max(-1.0, min(1.0, worst_year / 2.0))
                + 0.20 * max(-1.0, min(1.0, float(stats["sharpe"][local_index]) / 2.0))
                + 0.20 * max(-1.0, min(1.0, float(rank_icir[local_index]) / 2.0))
                + 0.10 * (2.0 * positive_year_rate - 1.0)
                + 0.10 * (2.0 * float((rank_ic[:, local_index] > 0).mean()) - 1.0)
                + 0.10 * min(1.0, len(active_mechanisms) / max(1, spec["min_mechanisms"] + 1))
            )
            output.append({
                "candidate_index": batch_start + local_index,
                "weights": [round(float(value), 8) for value in batch[local_index]],
                "source": source["source"],
                "rationale": source.get("rationale", ""),
                "hypothesis": source.get("hypothesis", ""),
                "ann_return": round(float(stats["ann_return"][local_index]), 8),
                "sharpe": round(float(stats["sharpe"][local_index]), 6),
                "max_drawdown": round(float(stats["max_drawdown"][local_index]), 8),
                "stress_ann_return": round(float(stressed_stats["ann_return"][local_index]), 8),
                "stress_sharpe": round(float(stressed_stats["sharpe"][local_index]), 6),
                "avg_turnover": round(float(turnover[:, local_index].mean()), 8),
                "ic_mean": round(float(ic_mean[local_index]), 8),
                "icir": round(float(icir[local_index]), 6),
                "ic_positive_rate": round(float((ic[:, local_index] > 0).mean()), 6),
                "rank_ic_mean": round(float(rank_mean[local_index]), 8),
                "rank_icir": round(float(rank_icir[local_index]), 6),
                "rank_ic_positive_rate": round(float((rank_ic[:, local_index] > 0).mean()), 6),
                "rank_ic_p_value": round(float(p_values[local_index]), 8),
                "worst_year_sharpe": round(worst_year, 6),
                "positive_year_rate": round(positive_year_rate, 6),
                "weight_hhi": round(float(hhi[local_index]), 8),
                "effective_factor_count": round(float(1.0 / hhi[local_index]), 6),
                "active_factors": int((batch[local_index] > 1e-12).sum()),
                "active_mechanisms": len(active_mechanisms),
                "robust_score": round(robust_score, 8),
                "periods": len(slices),
            })
        _progress(
            progress_callback,
            stage=stage,
            completed=min(batch_start + candidate_count, len(candidates)),
            total=len(candidates),
            message=f"{stage}: {min(batch_start + candidate_count, len(candidates))}/{len(candidates)}",
        )
    return output


def _benjamini_hochberg(rows: list[dict], p_key: str = "rank_ic_p_value") -> None:
    ordered = sorted(enumerate(rows), key=lambda item: _safe_float(item[1].get(p_key), 1.0))
    running = 1.0
    q_values = [1.0] * len(rows)
    for reverse_index in range(len(ordered) - 1, -1, -1):
        original_index, row = ordered[reverse_index]
        rank = reverse_index + 1
        running = min(running, _safe_float(row.get(p_key), 1.0) * len(rows) / rank)
        q_values[original_index] = min(1.0, running)
    for row, value in zip(rows, q_values):
        row["rank_ic_fdr_q"] = round(value, 8)


def _component_return_correlation(frame: pl.DataFrame, spec: dict) -> list[list[float]]:
    """Training-only standalone component Rank-IC path correlation proxy."""
    columns = [f"component_{index}" for index in range(len(spec["components"]))]
    start, end = _as_date(spec["train_start"], "train_start"), _as_date(spec["train_end"], "train_end")
    dates = (
        frame.lazy().filter(pl.col("trade_date").is_between(start, end))
        .select("trade_date").unique().sort("trade_date").collect()["trade_date"].to_list()
    )
    signal_dates = dates[: max(0, len(dates) - spec["horizon"] - 1): spec["horizon"]]
    paths = []
    for column in columns:
        daily = (
            frame.lazy().filter(
                pl.col("trade_date").is_in(signal_dates)
                & (pl.col("univ_rank") <= spec["universe_n"])
                & pl.col(column).is_finite()
                & pl.col(f"fwd_{spec['horizon']}").is_finite()
            )
            .group_by("trade_date")
            .agg(pl.corr(column, f"fwd_{spec['horizon']}", method="spearman").alias("value"))
            .sort("trade_date").collect()["value"].fill_null(0.0).to_numpy()
        )
        paths.append(daily)
    minimum = min(len(row) for row in paths)
    matrix = np.corrcoef(np.vstack([row[:minimum] for row in paths])) if minimum >= 3 else np.eye(len(paths))
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(matrix, 1.0)
    return matrix.round(8).tolist()


def _pair_correlation_ok(weights: Sequence[float], matrix: np.ndarray, cap: float) -> bool:
    active = np.flatnonzero(np.asarray(weights) > 1e-12)
    return all(abs(float(matrix[left, right])) <= cap + 1e-12 for left, right in itertools.combinations(active, 2))


def _composite_frame(frame: pl.DataFrame, weights: Sequence[float]) -> pl.DataFrame:
    score = pl.lit(0.0)
    for index, weight in enumerate(weights):
        score += pl.col(f"component_{index}") * float(weight)
    return frame.with_columns(score.alias("factor"))


def _batch_spec(spec: dict, start: str, end: str, slippage: tuple[float, ...]) -> BatchBacktestSpec:
    return BatchBacktestSpec(
        market=spec["market"],
        mode=spec["portfolio_mode"],
        universe_n=spec["universe_n"],
        horizon=spec["horizon"],
        top_fraction=spec["top_fraction"],
        initial_capital=spec["initial_capital"],
        rebalance_every=spec["horizon"],
        max_volume_participation=spec["max_volume_participation"],
        train_start=start,
        train_end=end,
        holdout_start=start,
        holdout_end=end,
        vault_start=end,
        vault_end=end,
        slippage_bps=slippage,
        borrow_cost_bps_annual=spec["borrow_cost_bps_annual"],
    )


def _exact_replay(
    frame: pl.DataFrame,
    weights: Sequence[float],
    spec: dict,
    *,
    start: str,
    end: str,
    artifact_dir: Path | None,
) -> dict:
    composite = _composite_frame(frame, weights)
    costs = tuple(sorted({0.0, 5.0, spec["cost_bps"], spec["stress_cost_bps"]}))
    batch_spec = _batch_spec(spec, start, end, costs)
    vector = run_vector_cost_scenarios(composite, direction=1, spec=batch_spec)
    ic = information_coefficients(
        composite,
        start=start,
        end=end,
        horizon=spec["horizon"],
        universe_n=spec["universe_n"],
        direction=1,
    )
    event_spec = _batch_spec(spec, start, end, (spec["cost_bps"],))
    event = run_cost_scenarios(
        composite,
        direction=1,
        spec=event_spec,
        capture_detail=artifact_dir is not None,
    )[str(int(spec["cost_bps"])) if float(spec["cost_bps"]).is_integer() else str(spec["cost_bps"])]
    result = event.pop("result", None)
    manifest = _write_artifacts(result, artifact_dir) if result is not None and artifact_dir else None
    return {"vector": vector, "ic": ic, "event": event, "artifact_manifest": manifest, "artifact_dir": str(artifact_dir) if artifact_dir else None}


def run_combination_search(
    spec: dict,
    *,
    llm_payload: dict | None = None,
    progress_callback: Callable[[dict], None] | None = None,
    cancel_event: threading.Event | None = None,
    artifact_root: Path | None = None,
) -> dict:
    """Execute the full deterministic evidence pipeline."""
    started = time.perf_counter()
    _check_cancel(cancel_event)
    _progress(progress_callback, stage="materialize", completed=0, total=1, message="加载面板并计算组件")
    frame, panel, effective = _materialize_component_frame(spec)
    _progress(progress_callback, stage="materialize", completed=1, total=1, message="组件计算完成")
    _check_cancel(cancel_event)

    candidates = enumerate_weight_candidates(spec, llm_payload)
    correlation = _component_return_correlation(frame, spec)
    correlation_matrix = np.asarray(correlation)
    filtered = [
        row for row in candidates
        if _pair_correlation_ok(row["weights"], correlation_matrix, spec["max_pair_correlation"])
    ]
    if not filtered:
        raise ValueError("训练期相关性约束过滤了全部组合")
    _progress(progress_callback, stage="search", completed=0, total=len(filtered), message=f"评估 {len(filtered)} 条训练路径")
    train_slices = _build_slices(
        frame, start=effective["train_start"], end=effective["train_end"], spec=spec
    )
    train_rows = _evaluate_candidates(
        train_slices,
        filtered,
        spec,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        stage="training_search",
    )
    train_rows.sort(key=lambda row: (row["robust_score"], row["worst_year_sharpe"], row["rank_icir"]), reverse=True)
    validation_candidates = [filtered[row["candidate_index"]] for row in train_rows[: spec["validation_budget"]]]
    validation_slices = _build_slices(
        frame,
        start=effective["validation_start"],
        end=effective["validation_end"],
        spec=spec,
    )
    validation_rows = _evaluate_candidates(
        validation_slices,
        validation_candidates,
        spec,
        progress_callback=progress_callback,
        cancel_event=cancel_event,
        stage="walk_forward_validation",
    )
    _benjamini_hochberg(validation_rows)
    finalists = []
    for validation in validation_rows:
        training = train_rows[validation["candidate_index"]]
        rules = {
            "rank_ic_direction": validation["rank_ic_mean"] > 0,
            "rank_ic_fdr": validation["rank_ic_fdr_q"] <= 0.25,
            "worst_year_sharpe": validation["worst_year_sharpe"] >= -0.20,
            "positive_year_rate": validation["positive_year_rate"] >= 0.50,
            "stress_return": validation["stress_ann_return"] > 0,
            "minimum_mechanisms": validation["active_mechanisms"] >= spec["min_mechanisms"],
        }
        validation_score = (
            0.40 * max(-1.0, min(1.0, validation["worst_year_sharpe"] / 2.0))
            + 0.25 * max(-1.0, min(1.0, validation["sharpe"] / 2.0))
            + 0.20 * max(-1.0, min(1.0, validation["rank_icir"] / 2.0))
            + 0.10 * (2.0 * validation["positive_year_rate"] - 1.0)
            - 0.05 * min(1.0, abs(training["sharpe"] - validation["sharpe"]) / 2.0)
        )
        finalists.append({
            "weights": validation["weights"],
            "source": validation["source"],
            "rationale": validation.get("rationale", ""),
            "hypothesis": validation.get("hypothesis", ""),
            "training": training,
            "validation": validation,
            "generalization_gap": round(abs(training["sharpe"] - validation["sharpe"]), 6),
            "rules": rules,
            "all_rules_pass": all(rules.values()),
            "validation_score": round(validation_score, 8),
        })
    finalists.sort(
        key=lambda row: (
            row["all_rules_pass"],
            row["validation_score"],
            row["validation"]["worst_year_sharpe"],
            row["validation"]["rank_icir"],
        ),
        reverse=True,
    )
    finalists = finalists[: spec["top_k"]]
    winner = next((row for row in finalists if row["all_rules_pass"]), None)
    decision = "NO_COMBINATION"
    rating = None
    equal_benchmark = None
    if winner is not None:
        _check_cancel(cancel_event)
        _progress(progress_callback, stage="rating", completed=0, total=2, message="冻结评级与步进复测")
        artifact_root = artifact_root or BACKTEST_ARTIFACT_ROOT / "combination_lab" / spec["snapshot_hash"]
        rating = _exact_replay(
            frame,
            winner["weights"],
            spec,
            start=effective["rating_start"],
            end=effective["rating_end"],
            artifact_dir=artifact_root / "winner",
        )
        _progress(progress_callback, stage="rating", completed=1, total=2, message="复测等权基准")
        equal_weights = [1.0 / len(spec["components"])] * len(spec["components"])
        equal_benchmark = _exact_replay(
            frame,
            equal_weights,
            spec,
            start=effective["rating_start"],
            end=effective["rating_end"],
            artifact_dir=artifact_root / "equal_all_components",
        )
        event = rating["event"]
        primary_key = str(int(spec["cost_bps"])) if float(spec["cost_bps"]).is_integer() else str(spec["cost_bps"])
        vector_primary = rating["vector"][primary_key]
        ranking_sharpe = (
            vector_primary["active_sharpe"]
            if spec["portfolio_mode"] == "long_only"
            else vector_primary["sharpe"]
        )
        rating_rules = {
            "event_integrity": bool((event.get("integrity") or {}).get("all_pass")),
            "positive_vector_sharpe": _safe_float(ranking_sharpe) > 0,
            "positive_event_sharpe": _safe_float(event.get("sharpe")) > 0,
            "event_drawdown": _safe_float(event.get("max_drawdown"), 1.0) <= 0.40,
            "rank_ic_direction": _safe_float(rating["ic"].get("rank_ic_mean")) > 0,
        }
        winner["rating_rules"] = rating_rules
        decision = "PASS" if all(rating_rules.values()) else "RESEARCH_ONLY"
        _progress(progress_callback, stage="rating", completed=2, total=2, message=f"冻结评级完成: {decision}")

    result = {
        "protocol": COMBINATION_LAB_PROTOCOL,
        "policy_label": "NESTED_COMPONENT_ARRAY_RESEARCH",
        "decision": decision,
        "snapshot_hash": spec["snapshot_hash"],
        "llm_used": spec["search_mode"] == "llm",
        "llm_proposals_received": len(list((llm_payload or {}).get("proposals") or [])),
        "llm_proposals_accepted": sum(row["source"] == "llm_proposal" for row in filtered),
        "panel": panel,
        "effective_windows": effective,
        "search": {
            "generated_candidates": len(candidates),
            "correlation_filtered_candidates": len(filtered),
            "validated_candidates": len(validation_rows),
            "component_return_correlation": correlation,
            "training_top": train_rows[:25],
        },
        "finalists": finalists,
        "winner": winner,
        "rating": rating,
        "equal_all_components_benchmark": equal_benchmark,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    result["result_hash"] = _fingerprint(result, 32)
    return result


async def request_llm_proposals(spec: dict, provider: dict, combination_id: int) -> tuple[dict, dict]:
    components = [
        {
            "key": row["key"],
            "name": row["name"],
            "expression": row["expression"],
            "direction": row["direction"],
            "mechanism": row["mechanism"],
            "required_history": row["required_history"],
        }
        for row in spec["components"]
    ]
    system = (
        "你是受约束的多因子组合搜索策略师。只能从给定组件中提出子集和非负权重；"
        "不得修改表达式、方向、时间协议、成本或准入门槛；不得假设或请求冻结评级结果。"
        "输出一个JSON对象，键proposals是数组。每项包含weights对象(key到0..1权重)、"
        "hypothesis和rationale。权重由程序归一化且必须满足用户约束。"
    )
    user = _canonical_json({
        "market": spec["market"],
        "portfolio_mode": spec["portfolio_mode"],
        "components": components,
        "constraints": {
            "min_factors": spec["min_factors"],
            "max_factors": spec["max_factors"],
            "min_weight": spec["min_weight"],
            "max_weight": spec["max_weight"],
            "min_mechanisms": spec["min_mechanisms"],
            "max_mechanism_weight": spec["max_mechanism_weight"],
            "max_proposals": spec["llm_max_proposals"],
        },
        "visible_windows": {
            "train": [spec["train_start"], spec["train_end"]],
            "validation": [spec["validation_start"], spec["validation_end"]],
        },
        "rating_data_visible": False,
    })
    response = await chat(
        provider,
        system,
        user,
        temperature=0.35,
        trace={
            "experiment_id": spec["experiment_id"],
            "role": "combination_strategist",
            "phase": "bounded_weight_proposal",
            "evaluation_protocol": COMBINATION_LAB_PROTOCOL,
            "task_name": f"combination:{combination_id}",
            "combination_experiment_id": combination_id,
            "rating_visible": False,
        },
    )
    try:
        payload = extract_json(str(response))
        accepted = parse_llm_weight_proposals(payload, spec)
        if not accepted:
            raise ValueError("LLM未返回满足约束的组合")
        await mark_validation(
            response,
            accepted=True,
            trace_meta_updates={"accepted_proposals": len(accepted)},
        )
        return payload, {
            "audit_id": getattr(response, "audit_id", None),
            "provider": provider.get("name"),
            "model": provider.get("model"),
            "accepted_proposals": len(accepted),
        }
    except Exception as exc:
        await mark_validation(response, accepted=False, error=exc)
        raise


class CombinationLabManager:
    """One memory-bounded background runner shared by all API requests."""

    _instance: "CombinationLabManager | None" = None

    def __init__(self) -> None:
        self._tasks: dict[int, asyncio.Task] = {}
        self._cancel: dict[int, threading.Event] = {}
        self._progress: dict[int, dict] = {}
        self._semaphore = asyncio.Semaphore(1)

    @classmethod
    def get(cls) -> "CombinationLabManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def snapshot(self, combination_id: int) -> dict:
        return dict(self._progress.get(combination_id) or {})

    async def start(self, combination_id: int) -> dict:
        task = self._tasks.get(combination_id)
        if task is not None and not task.done():
            return {"id": combination_id, "status": "running", "already_running": True}
        cancel_event = threading.Event()
        self._cancel[combination_id] = cancel_event
        self._progress[combination_id] = {
            "stage": "queued", "completed": 0, "total": 1,
            "message": "等待组合优化执行槽", "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._tasks[combination_id] = asyncio.create_task(
            self._execute(combination_id, cancel_event),
            name=f"combination-lab-{combination_id}",
        )
        return {"id": combination_id, "status": "queued", "already_running": False}

    async def stop(self, combination_id: int) -> dict:
        event = self._cancel.get(combination_id)
        if event is None:
            return {"id": combination_id, "stop_requested": False, "reason": "not_running"}
        event.set()
        self._progress[combination_id] = {
            **self._progress.get(combination_id, {}),
            "message": "已请求停止，等待当前批次安全退出",
            "stop_requested": True,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        return {"id": combination_id, "stop_requested": True}

    async def _provider(self) -> dict | None:
        from .db import SessionLocal
        from .models import Setting

        async with SessionLocal() as session:
            row = await session.get(Setting, "llm_providers")
        value = dict(row.value or {}) if row else {}
        name = value.get("inner_provider") or value.get("outer_provider")
        return next(
            (
                dict(provider) for provider in value.get("providers", [])
                if provider.get("name") == name and provider.get("api_key")
            ),
            None,
        )

    async def _set_status(
        self,
        combination_id: int,
        status: str,
        *,
        result: dict | None = None,
        error: str = "",
        llm_trace: dict | None = None,
    ) -> None:
        from .db import SessionLocal
        from .models import CombinationExperiment

        async with SessionLocal() as session:
            row = await session.get(CombinationExperiment, combination_id)
            if row is None:
                return
            row.status = status
            row.error = error[:4000]
            if result is not None:
                row.result = result
            if llm_trace is not None:
                row.llm_trace = llm_trace
            # Project timestamp columns are PostgreSQL TIMESTAMP WITHOUT TIME
            # ZONE.  Persist UTC as a deliberately naive value; asyncpg rejects
            # mixing an aware datetime with that column type.
            db_now = datetime.now(timezone.utc).replace(tzinfo=None)
            if status == "running" and row.started_at is None:
                row.started_at = db_now
            if status in TERMINAL_STATUSES:
                row.completed_at = db_now
            row.progress = dict(self._progress.get(combination_id) or {})
            await session.commit()

    async def _execute(self, combination_id: int, cancel_event: threading.Event) -> None:
        from .db import SessionLocal
        from .models import CombinationExperiment

        try:
            async with self._semaphore:
                if cancel_event.is_set():
                    raise CombinationCancelled("任务在启动前被停止")
                async with SessionLocal() as session:
                    row = await session.get(CombinationExperiment, combination_id)
                    if row is None:
                        return
                    spec = dict(row.request_spec or {})
                self._progress[combination_id] = {
                    "stage": "starting", "completed": 0, "total": 1,
                    "message": "冻结协议并启动", "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                await self._set_status(combination_id, "running")
                llm_payload, llm_trace = None, {}
                if spec["search_mode"] == "llm":
                    provider = await self._provider()
                    if provider is None:
                        raise ValueError("LLM协作模式要求设置中已配置inner_provider或outer_provider")
                    self._progress[combination_id] = {
                        "stage": "llm_proposal", "completed": 0, "total": 1,
                        "message": "LLM正在提出受约束组合", "updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    llm_payload, llm_trace = await request_llm_proposals(spec, provider, combination_id)
                    await self._set_status(combination_id, "running", llm_trace=llm_trace)

                loop = asyncio.get_running_loop()

                def update(payload: dict) -> None:
                    loop.call_soon_threadsafe(self._progress.__setitem__, combination_id, payload)

                result = await asyncio.to_thread(
                    run_combination_search,
                    spec,
                    llm_payload=llm_payload,
                    progress_callback=update,
                    cancel_event=cancel_event,
                    artifact_root=BACKTEST_ARTIFACT_ROOT / "combination_lab" / f"{combination_id:08d}",
                )
                self._progress[combination_id] = {
                    "stage": "complete", "completed": 1, "total": 1,
                    "message": f"完成: {result['decision']}",
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                await self._set_status(combination_id, "done", result=result, llm_trace=llm_trace)
        except CombinationCancelled as exc:
            await self._set_status(combination_id, "stopped", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - persist research failure
            self._progress[combination_id] = {
                **self._progress.get(combination_id, {}),
                "stage": "error", "message": str(exc)[:1000],
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            await self._set_status(combination_id, "error", error=str(exc))
