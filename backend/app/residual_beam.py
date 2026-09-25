"""Time-ordered OOF residual search and diversity-aware beam selection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import polars as pl

from .config import get_dsl_fields
from .data.panel import PanelStore
from .data.causal import purged_fold_masks, purged_layer_predicate
from .dsl.engine import normalize_hash, parse, validate

PROTOCOL = "factorfactory.residual-oof-beam/v4-joint-refit"
AUTOMATED_LAYERS = ("INNER_PUBLIC", "META_TRAIN")


def _rank(values: np.ndarray) -> np.ndarray:
    """Average ranks: ties must never encode input row/security order."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks[order] = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    return ranks


def rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 3 or len(left) != len(right):
        return 0.0
    mask = np.isfinite(left) & np.isfinite(right)
    if int(mask.sum()) < 3:
        return 0.0
    if np.ptp(left[mask]) <= 1e-12 or np.ptp(right[mask]) <= 1e-12:
        return 0.0
    value = float(np.corrcoef(_rank(left[mask]), _rank(right[mask]))[0, 1])
    return value if np.isfinite(value) else 0.0


def _ridge_predict(train_x, train_y, test_x, alpha: float) -> np.ndarray:
    # Fit scaling on this training fold only. No imputation from future data.
    center = np.mean(train_x, axis=0)
    scale = np.std(train_x, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    train_x, test_x = (train_x - center) / scale, (test_x - center) / scale
    design = np.column_stack([np.ones(len(train_x)), train_x])
    penalty = np.eye(design.shape[1]) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.lstsq(design.T @ design + penalty, design.T @ train_y, rcond=None)[0]
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

    Warm-up, embargo and invalid rows remain NaN, never zero-prediction
    pseudo-OOF observations. Every fitted fold uses strictly prior labels.
    """
    y = np.asarray(target, dtype=float)
    x = np.asarray(incumbent_predictions, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if int(folds) < 2 or len(y) != len(x) or len(y) < max(10, folds * 2):
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
    prediction = np.full(len(y), np.nan, dtype=float)
    finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
    for index, test_rows in enumerate(blocks):
        if index == 0:
            continue
        train_rows = np.concatenate(blocks[:index])
        if groups is not None:
            train_mask, test_mask = purged_fold_masks(group_values, np.unique(group_values[test_rows]),
                label_exit_dates=label_exit_dates, embargo_sessions=embargo_sessions)
            train_rows, test_rows = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
        train_rows = train_rows[finite[train_rows]]
        test_rows = test_rows[finite[test_rows]]
        if len(train_rows) < max(3, x.shape[1] + 2) or not len(test_rows):
            continue
        prediction[test_rows] = _ridge_predict(x[train_rows], y[train_rows], x[test_rows], ridge_alpha)
    return y - prediction


def _date_rows(groups):
    order = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(groups[order][1:] != groups[order][:-1]) + 1
    return np.split(order, boundaries)


def _ic_path(left, right, groups=None, grouped_rows=None):
    mask = np.isfinite(left) & np.isfinite(right)
    if groups is None:
        return [rank_correlation(left[mask], right[mask])] if mask.sum() >= 3 else []
    result = []
    for rows in grouped_rows if grouped_rows is not None else _date_rows(groups):
        selected = rows[mask[rows]]
        if len(selected) >= 3:
            result.append(rank_correlation(left[selected], right[selected]))
    return result


def signal_quality(matrix: pl.DataFrame, column: str) -> dict:
    finite = pl.col(column).is_finite().fill_null(False)
    daily = matrix.group_by("trade_date").agg(
        finite.sum().alias("finite"),
        pl.col(column).filter(finite).n_unique().alias("unique"),
    )
    coverage = float(matrix.select(finite.mean()).item() or 0)
    variable_days = float(daily.select(((pl.col("finite") >= 3) & (pl.col("unique") >= 2)).mean()).item() or 0)
    return {"finite_coverage": coverage, "variable_date_fraction": variable_days,
            "accepted": coverage >= .2 and variable_days >= .2,
            "thresholds": {"finite_coverage": .2, "variable_date_fraction": .2}}


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
    beam_depth: int = 2,
    max_joint_fits: int = 64,
) -> dict:
    """Bounded joint-refit beam; predictive evidence, not executable P&L.

    All alternatives use identical finite observations and fold boundaries.
    Each expansion refits baseline+selected features using only past labels.
    The beam is adaptively selected training evidence, never an untouched OOS.
    """
    if not 1 <= beam_width <= 50 or not 1 <= beam_depth <= 3:
        raise ValueError("beam_width must be 1..50; beam_depth must be 1..3")
    y = np.asarray(target, dtype=float)
    incumbent = np.asarray(incumbent_predictions, dtype=float)
    if incumbent.ndim == 1:
        incumbent = incumbent[:, None]
    group_values = np.asarray(groups) if groups is not None else None
    grouped_rows = _date_rows(group_values) if group_values is not None else None
    def ic_path(left, right):
        return _ic_path(left, right, group_values, grouped_rows)
    clean = {}
    rejected = []
    for name, raw in candidates.items():
        prediction = np.asarray(raw, dtype=float)
        if prediction.ndim != 1 or len(prediction) != len(y):
            raise ValueError(f"candidate {name!r} length mismatch")
        if not np.isfinite(prediction).all() or np.ptp(prediction) <= 1e-12:
            rejected.append({"name": name, "reason": "nonfinite_or_constant_signal"})
        else:
            clean[name] = prediction
    common = np.isfinite(y) & np.isfinite(incumbent).all(axis=1)
    y = np.where(common, y, np.nan)
    def predict(names):
        x = np.column_stack([incumbent, *[clean[name] for name in names]])
        return y - time_ordered_oof_residuals(y, x, folds=folds, groups=group_values,
            label_exit_dates=label_exit_dates, embargo_sessions=embargo_sessions)
    baseline = predict(())
    valid = np.isfinite(baseline) & np.isfinite(y)
    residual = y - baseline
    base_path = ic_path(baseline, y)
    base_ic = float(np.mean(base_path)) if base_path else 0.0
    cache = {(): baseline}
    states = [{"names": (), "ic": base_ic}]
    rows, paths = [], []
    fit_limit = max(1, int(max_joint_fits))
    for depth in range(1, beam_depth + 1):
        expansions = []
        visited = set()
        for parent in states:
            for name in sorted(clean):
                names = tuple(sorted((*parent["names"], name)))
                if name in parent["names"] or names in visited:
                    continue
                visited.add(names)
                if names not in cache:
                    if len(cache) - 1 >= fit_limit:
                        continue
                    cache[names] = predict(names)
                joint = cache[names]
                pair_mask = valid & np.isfinite(joint)
                path = ic_path(np.where(pair_mask, joint, np.nan), y)
                paired_base = ic_path(np.where(pair_mask, baseline, np.nan), y)
                joint_ic = float(np.mean(path)) if path else 0.0
                incremental = joint_ic - (float(np.mean(paired_base)) if paired_base else 0.0)
                parent_path = ic_path(np.where(pair_mask, cache[parent["names"]], np.nan), y)
                marginal = joint_ic - (float(np.mean(parent_path)) if parent_path else 0.0)
                # The fitted model determines the incremental prediction direction.
                residual_path = ic_path(joint - baseline, residual)
                residual_ic = float(np.mean(residual_path)) if residual_path else 0.0
                differences = np.asarray(path) - np.asarray(paired_base)
                stability = float(np.mean(differences > 0)) if len(differences) else 0.0
                independence = 1 - max((abs(rank_correlation(clean[name][valid], incumbent[valid, j]))
                    for j in range(incumbent.shape[1])), default=0.0)
                turn = sum(float((turnover or {}).get(n, 0)) for n in names)
                comp = sum(float((complexity or {}).get(n, 0)) for n in names)
                eligible = bool(pair_mask.sum() >= 10 and residual_ic > 1e-6 and incremental > 1e-6 and marginal > 1e-6)
                score = max(0.0, incremental) * (0.5 + 0.5 * stability) + .1 * max(0.0, residual_ic) * max(0.0, independence)
                score = score / (1 + .1 * turn + .05 * comp) if eligible else 0.0
                row = {"name": name, "names": list(names), "residual_rank_ic": residual_ic,
                    "incremental_oof_ic": incremental, "marginal_oof_ic": marginal,
                    "independence": independence, "stability": stability, "turnover": turn,
                    "complexity": comp, "score": score, "eligible": eligible,
                    "valid_oof_rows": int(pair_mask.sum()), "joint_oof_ic": joint_ic,
                    "turnover_available": turnover is not None,
                    "net_sharpe_available": False, "event_confirmation_required": True}
                if depth == 1:
                    rows.append(row)
                if eligible:
                    expansions.append({"names": names, "ic": joint_ic, "score": score})
                    paths.append(row)
        states = sorted(expansions, key=lambda r: (-r["score"], r["names"]))[:beam_width]
        if not states:
            break
    rows.sort(key=lambda row: (-row["score"], row["name"]))
    accepted = [row for row in rows if row["eligible"]]
    return {
        "protocol": PROTOCOL,
        "folds": folds,
        "beam_width": min(beam_width, len(accepted)),
        "residual_std": round(float(np.nanstd(residual)), 8) if valid.any() else None,
        "date_grouped_folds": groups is not None,
        "valid_oof_rows": int(valid.sum()), "excluded_oof_rows": int((~valid).sum()),
        "baseline_oof_ic": base_ic, "joint_refits": len(cache) - 1,
        "fit_budget_exhausted": len(cache) - 1 >= fit_limit,
        "candidates": rows, "beam": accepted[:beam_width],
        "combination_paths": sorted(paths, key=lambda r: -r["score"])[:beam_width],
        "rejected": rejected,
        "score_scope": "adaptive_training_oof_predictive_increment_not_net_pnl",
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
    screening = {name: signal_quality(matrix, name) for name in [*incumbent_aliases, *aliases]}
    if any(not screening[name]["accepted"] for name in incumbent_aliases):
        raise ValueError("Residual OOF 在位组合存在常数/低覆盖信号，拒绝伪造原组合残差")
    aliases = {name: row for name, row in aliases.items() if screening[name]["accepted"]}
    if not aliases:
        raise ValueError("Residual OOF 所有候选均未通过常数/覆盖率预筛")
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
        "signal_screening": screening,
        "missing_feature_policy": "training_model_only_neutral_rank_zero_after_coverage_screen",
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
        "combination_paths": [{**row, "expressions": [aliases[name]["expression"] for name in row["names"]]}
                              for row in result["combination_paths"]],
    }
