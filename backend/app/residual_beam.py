"""Time-ordered OOF residual search and diversity-aware beam selection."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import polars as pl

from .config import get_dsl_fields
from .data.panel import PanelStore
from .data.causal import purged_fold_masks, purged_layer_predicate
from .dsl.engine import normalize_hash, parse, validate

PROTOCOL = "factorfactory.residual-oof-beam/v3-purged"
AUTOMATED_LAYERS = ("INNER_PUBLIC", "META_TRAIN")


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    return ranks


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(left) != len(right):
        return 0.0
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return 0.0
    value = float(np.corrcoef(_rank(left[mask]), _rank(right[mask]))[0, 1])
    return value if np.isfinite(value) else 0.0


def _ridge_predict(train_x, train_y, test_x, alpha: float) -> np.ndarray:
    design = np.column_stack([np.ones(len(train_x)), train_x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ train_y)
    return np.column_stack([np.ones(len(test_x)), test_x]) @ coefficients


def time_ordered_oof_residuals(
    target: list[float] | np.ndarray,
    incumbent_predictions: list[list[float]] | np.ndarray,
    *,
    folds: int = 5,
    ridge_alpha: float = 1e-3,
    groups: list | np.ndarray | None = None,
    label_exit_dates: list | np.ndarray | None = None,
    embargo_sessions: int = 0,
) -> np.ndarray:
    """Return expanding-window residuals without training on future folds.

    The first block is an explicit warm-up and receives the zero-prediction
    baseline.  Every later block is predicted using strictly earlier rows.
    This is intentionally more conservative than ordinary K-fold OOF.
    """
    y = np.asarray(target, dtype=float)
    x = np.asarray(incumbent_predictions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(y) != len(x) or len(y) < max(10, folds * 2):
        raise ValueError("target/incumbent rows must match and cover at least two rows per fold")
    if groups is not None:
        group_values = np.asarray(groups)
        if len(group_values) != len(y):
            raise ValueError("groups must have one value per target row")
        unique_groups = np.unique(group_values)
        if len(unique_groups) < 2:
            raise ValueError("time-ordered OOF requires at least two date groups")
        group_blocks = np.array_split(
            unique_groups,
            min(int(folds), len(unique_groups)),
        )
        blocks = [
            np.flatnonzero(np.isin(group_values, block))
            for block in group_blocks
            if len(block)
        ]
    else:
        blocks = list(
            np.array_split(
                np.arange(len(y)), min(int(folds), len(y) // 2)
            )
        )
    prediction = np.zeros(len(y), dtype=float)
    for index, test_rows in enumerate(blocks):
        if index == 0:
            continue
        train_rows = np.concatenate(blocks[:index])
        if groups is not None:
            train_mask, test_mask = purged_fold_masks(group_values, np.unique(group_values[test_rows]),
                label_exit_dates=label_exit_dates, embargo_sessions=embargo_sessions)
            train_rows, test_rows = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
        if not len(train_rows) or not len(test_rows):
            continue
        prediction[test_rows] = _ridge_predict(x[train_rows], y[train_rows], x[test_rows], ridge_alpha)
    return y - prediction


@dataclass(frozen=True)
class BeamCandidate:
    name: str
    residual_rank_ic: float
    incremental_oof_ic: float
    independence: float
    stability: float
    turnover: float
    complexity: float
    score: float


def residual_oof_beam_search(
    *,
    target: list[float] | np.ndarray,
    incumbent_predictions: list[list[float]] | np.ndarray,
    candidates: dict[str, list[float] | np.ndarray],
    folds: int = 5,
    beam_width: int = 5,
    turnover: dict[str, float] | None = None,
    complexity: dict[str, float] | None = None,
    groups: list | np.ndarray | None = None,
    label_exit_dates: list | np.ndarray | None = None,
    embargo_sessions: int = 0,
) -> dict:
    residual = time_ordered_oof_residuals(
        target,
        incumbent_predictions,
        folds=folds,
        groups=groups,
        label_exit_dates=label_exit_dates,
        embargo_sessions=embargo_sessions,
    )
    incumbent = np.asarray(incumbent_predictions, dtype=float)
    if incumbent.ndim == 1:
        incumbent = incumbent[:, None]
    rows: list[BeamCandidate] = []
    for name, raw in candidates.items():
        prediction = np.asarray(raw, dtype=float)
        if len(prediction) != len(residual):
            raise ValueError(f"candidate {name!r} length mismatch")
        residual_ic = rank_correlation(prediction, residual)
        base_ic = rank_correlation(np.mean(incumbent, axis=1), np.asarray(target, dtype=float))
        joint_ic = rank_correlation(
            np.mean(np.column_stack([incumbent, prediction]), axis=1),
            np.asarray(target, dtype=float),
        )
        incremental = joint_ic - base_ic
        correlations = [abs(rank_correlation(prediction, incumbent[:, index])) for index in range(incumbent.shape[1])]
        independence = 1.0 - max(correlations, default=0.0)
        if groups is not None:
            group_values = np.asarray(groups)
            group_blocks = np.array_split(
                np.unique(group_values), min(folds, len(np.unique(group_values)))
            )
            blocks = [
                np.flatnonzero(np.isin(group_values, block))
                for block in group_blocks
                if len(block)
            ]
        else:
            blocks = np.array_split(
                np.arange(len(prediction)), min(folds, len(prediction))
            )
        block_ics = [rank_correlation(prediction[index], residual[index]) for index in blocks]
        stability = max(0.0, 1.0 - float(np.std(block_ics)))
        turn = float((turnover or {}).get(name, 0.0))
        comp = float((complexity or {}).get(name, 0.0))
        score = 0.35 * residual_ic + 0.25 * incremental + 0.20 * independence + 0.20 * stability - 0.10 * turn - 0.05 * comp
        rows.append(BeamCandidate(name, residual_ic, incremental, independence, stability, turn, comp, score))
    rows.sort(key=lambda row: (row.score, row.independence, row.name), reverse=True)
    return {
        "protocol": PROTOCOL,
        "folds": folds,
        "beam_width": min(beam_width, len(rows)),
        "residual_std": round(float(np.std(residual)), 8),
        "date_grouped_folds": groups is not None,
        "candidates": [
            {
                key: (round(value, 8) if isinstance(value, float) else value)
                for key, value in row.__dict__.items()
            }
            for row in rows
        ],
        "beam": [
            {
                key: (round(value, 8) if isinstance(value, float) else value)
                for key, value in row.__dict__.items()
            }
            for row in rows[:beam_width]
        ],
    }


def _even_sample(values: list, keep: int) -> list:
    if keep >= len(values):
        return list(values)
    if keep <= 0:
        return []
    indexes = np.linspace(0, len(values) - 1, keep, dtype=int)
    return [values[index] for index in sorted(set(indexes.tolist()))]


def _sample_complete_dates(frame: pl.DataFrame, max_rows: int) -> tuple[pl.DataFrame, dict]:
    """Bound memory without splitting a daily cross-section or a layer."""
    if frame.height <= max_rows:
        return frame, {
            "policy": "all_training_rows",
            "source_rows": frame.height,
            "selected_rows": frame.height,
        }
    counts = frame.group_by("layer", "trade_date").len().sort("layer", "trade_date")
    selected: list = []
    # Preserve both automated layers. META_TRAIN gets at least 25% of the row
    # budget so incremental evidence cannot silently become INNER_PUBLIC-only.
    for layer, share in (("INNER_PUBLIC", 0.75), ("META_TRAIN", 0.25)):
        rows = counts.filter(pl.col("layer") == layer)
        dates = rows["trade_date"].to_list()
        average = float(rows["len"].mean() or 1.0) if rows.height else 1.0
        keep = min(len(dates), max(2, int(max_rows * share / max(1.0, average))))
        selected.extend(_even_sample(dates, keep))
    sampled = frame.filter(pl.col("trade_date").is_in(sorted(set(selected))))
    # Complete dates can make the estimate slightly exceed max_rows. Remove
    # evenly from INNER_PUBLIC first, never slice rows within a date.
    while sampled.height > max_rows and len(selected) > 4:
        selected = _even_sample(sorted(set(selected)), len(set(selected)) - 1)
        sampled = frame.filter(pl.col("trade_date").is_in(selected))
    return sampled, {
        "policy": "layer_stratified_even_complete_dates",
        "source_rows": frame.height,
        "selected_rows": sampled.height,
        "selected_dates": sampled["trade_date"].n_unique(),
    }


def build_dsl_residual_oof_artifact(
    *,
    market: str,
    panel_glob: str | None,
    universe_n: int,
    horizon: int,
    incumbent_expressions: list[str],
    candidate_rows: list[dict],
    folds: int = 5,
    beam_width: int = 8,
    max_rows: int = 150_000,
    security_sample_modulus: int = 4,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict:
    """Score DSL candidates against strict past-only, sample-level residuals.

    This is the actual L1 residual data path. It compiles candidate and
    incumbent DSL against INNER_PUBLIC/META_TRAIN observations, ranks signals
    cross-sectionally by date, and fits expanding-window incumbent models with
    whole-date folds. HOLDOUT, Vault, and frozen rating never enter the matrix.
    """
    fields = get_dsl_fields(market)
    label = f"fwd_{int(horizon)}"
    store = PanelStore.get(panel_glob, market, fields)
    panel, _, identity, generation = store.read_snapshot()
    if label not in panel.columns:
        raise ValueError(f"不支持的 horizon: {horizon}")
    incumbents = []
    seen = set()
    for expression in incumbent_expressions:
        expression = str(expression or "").strip()
        if not expression or validate(expression, fields):
            continue
        key = normalize_hash(expression, direction_invariant=True)
        if key not in seen:
            seen.add(key)
            incumbents.append(expression)
    candidates = []
    for row in candidate_rows:
        expression = str(row.get("expression") or "").strip()
        if not expression or validate(expression, fields):
            continue
        key = normalize_hash(expression, direction_invariant=True)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({**row, "expression": expression, "normalized_hash": key})
    if not candidates:
        raise ValueError("Residual OOF 没有字段合法且未与在位者重复的候选")
    if progress_callback:
        progress_callback({"phase": "feature_graph", "completed": 0, "total": len(incumbents) + len(candidates)})
    safe_layers = purged_layer_predicate(panel, market, horizon, AUTOMATED_LAYERS)
    lf = panel.lazy()
    modulus = max(1, int(security_sample_modulus))
    if modulus > 1:
        lf = lf.filter((pl.col("ts_code").hash(seed=1729) % modulus) == 0)
    aliases: dict[str, dict] = {}
    incumbent_aliases = []
    for index, expression in enumerate(incumbents):
        alias = f"_res_inc_{index:02d}"
        lf = parse(expression, fields).apply(lf, alias=alias)
        incumbent_aliases.append(alias)
        if progress_callback:
            progress_callback({"phase": "feature_graph", "completed": index + 1, "total": len(incumbents) + len(candidates)})
    for index, row in enumerate(candidates):
        alias = f"_res_cand_{index:03d}"
        lf = parse(row["expression"], fields).apply(lf, alias=alias)
        aliases[alias] = row
        if progress_callback:
            progress_callback({"phase": "feature_graph", "completed": len(incumbents) + index + 1, "total": len(incumbents) + len(candidates)})
    columns = ["trade_date", "ts_code", "layer", "univ_rank", label, f"label_exit_date_{horizon}", *incumbent_aliases, *aliases]
    matrix = (
        lf.select(columns).cache().filter(safe_layers)
        .filter(pl.col("univ_rank") <= int(universe_n))
        .collect()
    )
    matrix, sampling = _sample_complete_dates(matrix, max(5_000, int(max_rows)))
    if matrix.height < 1_000 or matrix["trade_date"].n_unique() < max(10, folds * 2):
        raise ValueError("Residual OOF 训练层样本或完整日期不足")
    rank_columns = [label, *incumbent_aliases, *aliases]
    expressions = []
    for column in rank_columns:
        clean = pl.when(pl.col(column).is_finite()).then(pl.col(column)).otherwise(None)
        expressions.append(
            (
                clean.rank(method="average").over("trade_date")
                / (clean.count().over("trade_date") + 1.0)
                - 0.5
            ).fill_null(0.0).alias(f"{column}_rank")
        )
    ranked = matrix.with_columns(*expressions).filter(pl.col(label).is_finite()).sort("trade_date", "ts_code")
    target = ranked[f"{label}_rank"].to_numpy().astype(float)
    groups = ranked["trade_date"].to_numpy()
    if incumbent_aliases:
        incumbent_predictions = ranked.select(
            [f"{name}_rank" for name in incumbent_aliases]
        ).to_numpy().astype(float)
    else:
        incumbent_predictions = np.zeros((len(target), 1), dtype=float)
    candidate_predictions = {
        alias: ranked[f"{alias}_rank"].to_numpy().astype(float)
        for alias in aliases
    }
    complexity = {
        alias: min(1.0, len(row["expression"]) / 500.0)
        for alias, row in aliases.items()
    }
    result = residual_oof_beam_search(
        target=target,
        incumbent_predictions=incumbent_predictions,
        candidates=candidate_predictions,
        folds=folds,
        beam_width=beam_width,
        complexity=complexity,
        groups=groups,
        label_exit_dates=ranked[f"label_exit_date_{horizon}"].to_numpy(),
        embargo_sessions=horizon,
    )
    def enrich(row: dict) -> dict:
        source = aliases[str(row["name"])]
        return {
            **row,
            "expression": source["expression"],
            "family": source.get("family"),
            "normalized_hash": source["normalized_hash"],
        }
    return {
        **result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "market": market,
        "panel_identity": identity,
        "panel_generation": generation,
        "automated_feedback_layers": list(AUTOMATED_LAYERS),
        "holdout_vault_consumed": False,
        "label_boundary_policy": "purged_layer_and_fold_exit_dates_with_h_session_embargo",
        "rows": ranked.height,
        "dates": ranked["trade_date"].n_unique(),
        "date_min": str(ranked["trade_date"].min()),
        "date_max": str(ranked["trade_date"].max()),
        "incumbents": len(incumbent_aliases),
        "candidate_count": len(aliases),
        "sampling": {**sampling, "security_hash_modulus": modulus},
        "candidates": [enrich(row) for row in result["candidates"]],
        "beam": [enrich(row) for row in result["beam"]],
    }
