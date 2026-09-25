"""Split-safe Alpha158 joint learning and residual-to-DSL distillation.

This module is deliberately independent from pyqlib.  It uses FactorFactory's
causal panel and DSL, never exposes HOLDOUT/Vault results to the automated
search loop, and returns only DSL candidates which must be re-evaluated by the
normal V4.2/V4.3 harness.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import polars as pl

from .config import PROJECT_ROOT, get_dsl_fields
from . import feature_cache
from .data.panel import PanelStore
from .data.causal import purged_fold_masks, purged_layer_predicate
from .dsl.engine import normalize_hash, parse, validate
from .qlib_native import ALPHA158_FEATURES, QLIB_UPSTREAM_COMMIT, Alpha158Feature
from .dsl.grammar_v2 import candidates as v2_candidates
from .dsl.operators_v2 import DSL_REVISION


PROTOCOL = "factorfactory.qlib-joint-residual-distill/v6-dsl2"
ARTIFACT_ROOT = PROJECT_ROOT / "var" / "reports" / "qlib-joint"
AUTOMATED_LAYERS = ("INNER_PUBLIC", "META_TRAIN")
ProgressCallback = Callable[[dict[str, Any]], None]


def _progress(
    callback: ProgressCallback | None,
    *,
    phase: str,
    message: str,
    completed: int | float | None = None,
    total: int | float | None = None,
) -> None:
    if callback is not None:
        callback({
            "phase": phase,
            "message": message,
            "completed": completed,
            "total": total,
        })


@dataclass(frozen=True)
class JointModelSpec:
    market: str
    panel_glob: str | None = None
    universe_n: int = 500
    horizon: int = 5
    max_rows: int = 250_000
    max_incumbents: int = 5
    folds: int = 5
    min_meta_dates: int = 60
    seed: int = 1729
    include_low_fidelity_vwap: bool = False
    allow_isolation_audit: bool = False

    def validate(self) -> None:
        if self.market not in {"ashare", "us"}:
            raise ValueError("market 必须是 ashare 或 us")
        if self.horizon not in {1, 5, 10, 20}:
            raise ValueError("horizon 必须是 1/5/10/20")
        if self.universe_n < 50:
            raise ValueError("universe_n 至少为 50")
        if self.max_rows < 5_000:
            raise ValueError("max_rows 至少为 5000")
        if self.folds < 3:
            raise ValueError("folds 至少为 3")
        if self.min_meta_dates < 20:
            raise ValueError("min_meta_dates 至少为 20")


def _safe_slug(value: str) -> str:
    rendered = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    return rendered[:96] or "task"


def _extension_rows(fields):
    return [Alpha158Feature(f"DSL2_{family}_{i}", expression, family, 20)
            for family in ("momentum", "reversal", "volatility", "liquidity", "volume_price_interaction", "gap_intraday", "price_relationship")
            for i, expression in enumerate(v2_candidates(family, fields, 20))]


def _feature_rows(spec: JointModelSpec):
    rows = list(ALPHA158_FEATURES) + _extension_rows(get_dsl_fields(spec.market))
    if not spec.include_low_fidelity_vwap:
        rows = [row for row in rows if row.name != "VWAP0"]
    return rows


def _cache_key(spec: JointModelSpec, identity: str, incumbent_expressions: Iterable[str]) -> str:
    payload = {
        "protocol": PROTOCOL,
        "market": spec.market,
        "identity": identity,
        "universe_n": spec.universe_n,
        "horizon": spec.horizon,
        "max_rows": spec.max_rows,
        "min_meta_dates": spec.min_meta_dates,
        "features": [(row.name, row.expression) for row in _feature_rows(spec)],
        "dsl_revision": DSL_REVISION,
        "incumbents": [normalize_hash(value, direction_invariant=True) for value in incumbent_expressions],
        "upstream": QLIB_UPSTREAM_COMMIT,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _even_date_sample(values: list[date], keep: int) -> list[date]:
    if keep >= len(values):
        return list(values)
    if keep <= 0 or not values:
        return []
    indices = np.linspace(0, len(values) - 1, keep, dtype=int)
    return [values[index] for index in sorted(set(indices.tolist()))]


def _sample_dates_by_layer(
    frame: pl.DataFrame,
    max_rows: int,
    *,
    min_meta_dates: int = 60,
) -> tuple[list[date], dict]:
    """Preserve complete daily cross-sections and a real META_TRAIN sample."""
    counts = frame.group_by("layer", "trade_date").len().sort(
        "layer", "trade_date"
    )
    layer_rows: dict[str, list[tuple[date, int]]] = {}
    for layer, day, rows in counts.iter_rows():
        layer_rows.setdefault(str(layer), []).append((day, int(rows)))
    meta_rows = layer_rows.get("META_TRAIN", [])
    train_rows = layer_rows.get("INNER_PUBLIC", [])
    requested_meta = min(int(min_meta_dates), len(meta_rows))
    selected_meta = _even_date_sample(
        [day for day, _ in meta_rows], requested_meta
    )
    count_lookup = {
        (layer, day): rows
        for layer, rows_by_day in layer_rows.items()
        for day, rows in rows_by_day
    }
    reserved_meta_rows = sum(
        count_lookup[("META_TRAIN", day)] for day in selected_meta
    )
    minimum_train_rows = min(5_000, sum(rows for _, rows in train_rows))
    if reserved_meta_rows + minimum_train_rows > max_rows:
        raise ValueError(
            "max_rows 无法同时容纳完整 META_TRAIN 最低日期数和最低训练样本；"
            f"需要至少 {reserved_meta_rows + minimum_train_rows:,} 行"
        )

    if frame.height <= max_rows:
        selected_train = [day for day, _ in train_rows]
        selected_meta = [day for day, _ in meta_rows]
    else:
        remaining = max_rows - reserved_meta_rows
        average_train_rows = (
            sum(rows for _, rows in train_rows) / len(train_rows)
            if train_rows else 1.0
        )
        train_keep = min(
            len(train_rows),
            max(2, int(remaining / max(1.0, average_train_rows))),
        ) if train_rows else 0
        selected_train = _even_date_sample(
            [day for day, _ in train_rows], train_keep
        )
        while selected_train and (
            reserved_meta_rows
            + sum(
                count_lookup[("INNER_PUBLIC", day)]
                for day in selected_train
            )
            > max_rows
        ):
            selected_train = _even_date_sample(
                selected_train, len(selected_train) - 1
            )

    selected = sorted(set([*selected_train, *selected_meta]))
    selected_rows = sum(
        count_lookup[(layer, day)]
        for layer, days in (
            ("INNER_PUBLIC", selected_train),
            ("META_TRAIN", selected_meta),
        )
        for day in days
    )
    return selected, {
        "policy": "layer_stratified_even_dates_preserving_cross_sections",
        "max_rows": int(max_rows),
        "selected_rows": int(selected_rows),
        "min_meta_dates": int(min_meta_dates),
        "source_dates_by_layer": {
            "INNER_PUBLIC": len(train_rows),
            "META_TRAIN": len(meta_rows),
        },
        "selected_dates_by_layer": {
            "INNER_PUBLIC": len(selected_train),
            "META_TRAIN": len(selected_meta),
        },
    }


def _materialize_matrix(
    spec: JointModelSpec,
    incumbent_expressions: list[str],
    progress_callback: ProgressCallback | None = None,
) -> tuple[pl.DataFrame, dict]:
    _progress(
        progress_callback,
        phase="panel_load",
        message="读取因果面板快照",
    )
    fields = get_dsl_fields(spec.market)
    store = PanelStore.get(spec.panel_glob, spec.market, fields)
    panel, _, identity, generation = store.read_snapshot()
    identity = identity or f"generation-{generation}"
    features = _feature_rows(spec)
    incumbent_expressions = [
        expression for expression in incumbent_expressions[: spec.max_incumbents]
        if not validate(expression, fields)
    ]
    key = _cache_key(spec, identity, incumbent_expressions)
    cache_dir = ARTIFACT_ROOT / "feature-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{key}.parquet"
    meta_path = cache_dir / f"{key}.json"
    if feature_cache.exists(cache_path) and meta_path.exists():
        _progress(
            progress_callback,
            phase="feature_cache",
            message="命中特征矩阵缓存",
            completed=1,
            total=1,
        )
        return feature_cache.read(cache_path), {
            **json.loads(meta_path.read_text(encoding="utf-8")),
            "cache_hit": True,
            "cache_path": str(cache_path.resolve()),
        }

    label = f"fwd_{spec.horizon}"
    base_columns = [
        "trade_date", "ts_code", "layer", "univ_rank", label, f"label_exit_date_{spec.horizon}",
        *fields,
    ]
    available = [column for column in dict.fromkeys(base_columns) if column in panel.columns]
    lf = panel.select(available).sort("ts_code", "trade_date").lazy()
    output_columns = ["trade_date", "ts_code", "layer", "univ_rank", label, f"label_exit_date_{spec.horizon}"]
    feature_total = len(features) + len(incumbent_expressions)
    for feature_index, feature in enumerate(features, start=1):
        lf = parse(feature.expression, fields).apply(lf, alias=feature.name)
        output_columns.append(feature.name)
        if feature_index == 1 or feature_index == len(features) or feature_index % 16 == 0:
            _progress(
                progress_callback,
                phase="feature_graph",
                message=f"编译 Alpha158 特征图 {feature_index}/{feature_total}",
                completed=feature_index,
                total=feature_total,
            )
    incumbent_names = []
    for index, expression in enumerate(incumbent_expressions):
        name = f"INCUMBENT_{index:02d}"
        lf = parse(expression, fields).apply(lf, alias=name)
        incumbent_names.append(name)
        output_columns.append(name)
        _progress(
            progress_callback,
            phase="feature_graph",
            message=f"编译特征图 {len(features) + index + 1}/{feature_total}",
            completed=len(features) + index + 1,
            total=feature_total,
        )
    # Filtering happens after time-series feature construction.  Pushing the
    # universe filter ahead of rolling operators would change signal semantics.
    _progress(
        progress_callback,
        phase="feature_materialize",
        message="物化特征矩阵；该阶段无法可靠预估耗时",
    )
    automated = lf.select(output_columns).cache().filter(
        purged_layer_predicate(panel, spec.market, spec.horizon, AUTOMATED_LAYERS)
        & (pl.col("univ_rank") <= spec.universe_n)
        & pl.col(label).is_not_null()
    ).collect()
    sampled_dates, sampling_metadata = _sample_dates_by_layer(
        automated,
        spec.max_rows,
        min_meta_dates=spec.min_meta_dates,
    )
    matrix = automated.filter(pl.col("trade_date").is_in(sampled_dates))
    matrix = matrix.with_columns(
        (
            pl.col(label).rank(method="average").over("trade_date")
            / (pl.len().over("trade_date") + 1.0)
            - 0.5
        ).alias("label")
    ).drop(label)
    feature_cache.write(cache_path, matrix)
    metadata = {
        "schema": PROTOCOL,
        "cache_key": key,
        "market": spec.market,
        "panel_identity": identity,
        "panel_generation": generation,
        "rows": matrix.height,
        "dates": matrix["trade_date"].n_unique(),
        "date_min": str(matrix["trade_date"].min()),
        "date_max": str(matrix["trade_date"].max()),
        "alpha158_features": len(features),
        "incumbent_features": len(incumbent_names),
        "sampling": sampling_metadata,
        "layers": list(AUTOMATED_LAYERS),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _progress(
        progress_callback,
        phase="feature_cache",
        message=f"特征矩阵就绪：{matrix.height:,} 行",
        completed=1,
        total=1,
    )
    return matrix, {**metadata, "cache_hit": False, "cache_path": str(cache_path.resolve())}


def _fit_processor(train: np.ndarray) -> dict[str, np.ndarray]:
    finite = np.where(np.isfinite(train), train, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(finite, axis=0)
        mad = np.nanmedian(np.abs(finite - median), axis=0)
    scale = np.where(mad > 1e-9, 1.4826 * mad, 1.0)
    median = np.nan_to_num(median, nan=0.0)
    scale = np.nan_to_num(scale, nan=1.0, posinf=1.0, neginf=1.0)
    return {"median": median, "scale": scale}


def _transform(values: np.ndarray, processor: dict[str, np.ndarray]) -> np.ndarray:
    result = (values - processor["median"]) / processor["scale"]
    return np.clip(np.nan_to_num(result, nan=0.0, posinf=3.0, neginf=-3.0), -3.0, 3.0).astype(np.float32)


def _ridge_fit_predict(train_x, train_y, test_x, alpha: float = 1.0):
    if train_x.shape[1] == 0:
        return np.zeros(len(test_x), dtype=float), np.zeros(0, dtype=float)
    gram = train_x.T @ train_x + np.eye(train_x.shape[1]) * alpha
    coef = np.linalg.solve(gram, train_x.T @ train_y)
    return test_x @ coef, coef


def _train_lgbm(train_x, train_y, valid_x, valid_y, seed: int):
    try:
        import lightgbm as lgb
    except ImportError:  # pragma: no cover - service requirements include it
        class RidgeFallback:
            n_estimators = 1
            best_iteration_ = 1
            _factorfactory_backend = "deterministic_ridge_dependency_fallback"

            def fit(self, values, target):
                _, self.coef_ = _ridge_fit_predict(values, target, values, alpha=2.0)
                self.feature_importances_ = np.abs(self.coef_)
                return self

            def predict(self, values):
                return values @ self.coef_

        return RidgeFallback().fit(train_x, train_y)
    train_set = lgb.Dataset(train_x, label=train_y, free_raw_data=False)
    valid_sets = []
    callbacks = []
    if len(valid_y):
        valid_sets = [lgb.Dataset(valid_x, label=valid_y, reference=train_set)]
        callbacks = [lgb.early_stopping(30, verbose=False)]
    booster = lgb.train(
        {
            "objective": "regression",
            "metric": "l2",
            "learning_rate": 0.035,
            "num_leaves": 31,
            "max_depth": 6,
            "min_data_in_leaf": 80,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "feature_fraction": 0.80,
            "lambda_l1": 0.2,
            "lambda_l2": 1.5,
            "seed": seed,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "num_threads": max(1, min(8, (os.cpu_count() or 2) // 2)),
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
        },
        train_set,
        num_boost_round=240,
        valid_sets=valid_sets or None,
        callbacks=callbacks or None,
    )

    class NativeBooster:
        _factorfactory_backend = "lightgbm_native_no_sklearn"
        n_estimators = 240

        def __init__(self, value):
            self.value = value
            self.best_iteration_ = int(value.best_iteration or value.current_iteration())
            self.feature_importances_ = value.feature_importance(
                importance_type="gain"
            )

        def predict(self, values):
            return self.value.predict(
                values,
                num_iteration=self.best_iteration_ or None,
            )

    return NativeBooster(booster)


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    from .residual_beam import rank_correlation
    mask = np.isfinite(left) & np.isfinite(right)
    if mask.sum() < 5:
        return 0.0
    return rank_correlation(left[mask], right[mask])


def _daily_ic(dates: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> dict:
    values = []
    for day in np.unique(dates):
        mask = (dates == day) & np.isfinite(prediction) & np.isfinite(target)
        if mask.sum() < 5 or np.ptp(prediction[mask]) <= 1e-12 or np.ptp(target[mask]) <= 1e-12:
            continue
        values.append(_rank_correlation(prediction[mask], target[mask]))
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array)) if len(array) else 0.0
    std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    return {
        "days": len(values),
        "mean_rank_ic": round(mean, 6),
        "rank_ic_std": round(std, 6),
        "rank_icir": round(mean / (std + 1e-12), 6),
        "positive_rate": round(float(np.mean(array > 0)) if len(array) else 0.0, 6),
    }


def _search_eligible(
    *,
    candidates: list[dict],
    joint_metrics: dict,
    incremental_mean_rank_ic: float,
    min_meta_dates: int,
) -> bool:
    """Keep model diagnostics separate from authority to enter DSL research.

    Absolute joint skill is insufficient: a joint model that is worse than the
    incumbent cannot seed the residual-distillation arm even when its raw IC
    looks respectable.  This is deliberately a strict positive comparison;
    equality and non-finite increments fail closed.
    """
    try:
        incremental = float(incremental_mean_rank_ic)
        mean_rank_ic = float(joint_metrics.get("mean_rank_ic"))
        rank_icir = float(joint_metrics.get("rank_icir"))
        days = int(joint_metrics.get("days") or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        candidates
        and math.isfinite(incremental)
        and incremental > 0.0
        and days >= int(min_meta_dates)
        and abs(mean_rank_ic) >= 0.01
        and abs(rank_icir) >= 0.10
    )


def _walk_forward_oof(
    x,
    y,
    dates,
    folds: int,
    seed: int,
    progress_callback: ProgressCallback | None = None,
    *, label_exit_dates=None, embargo_sessions=0, incumbent_raw=None,
):
    unique_dates = np.unique(dates)
    blocks = [block for block in np.array_split(unique_dates, folds + 1) if len(block)]
    predictions = np.full(len(y), np.nan, dtype=float)
    importances = []
    fold_rows = []
    # First block is warm-up. Every prediction is produced strictly from prior dates.
    for index in range(1, len(blocks)):
        train_dates = np.concatenate(blocks[:index])
        test_dates = blocks[index]
        train_mask, test_mask = purged_fold_masks(dates, test_dates,
            label_exit_dates=label_exit_dates, embargo_sessions=embargo_sessions)
        if train_mask.sum() < 1000 or test_mask.sum() < 100:
            _progress(
                progress_callback,
                phase="walk_forward_oof",
                message=f"OOF 折 {index}/{len(blocks) - 1} 样本不足，已跳过",
                completed=index,
                total=len(blocks) - 1,
            )
            continue
        processor = _fit_processor(x[train_mask])
        train_x = _transform(x[train_mask], processor)
        test_x = _transform(x[test_mask], processor)
        target = y[train_mask]
        incumbent_test_prediction = 0.0
        if incumbent_raw is not None and incumbent_raw.shape[1]:
            inc_processor = _fit_processor(incumbent_raw[train_mask])
            inc_train = _transform(incumbent_raw[train_mask], inc_processor)
            inc_test = _transform(incumbent_raw[test_mask], inc_processor)
            train_prediction, _ = _ridge_fit_predict(inc_train, target, inc_train)
            incumbent_test_prediction, _ = _ridge_fit_predict(inc_train, target, inc_test)
            target = target - train_prediction
        # Fixed rounds. The OOF fold's labels must not choose early-stopping
        # rounds, preprocessing, or incumbent residualization coefficients.
        model = _train_lgbm(train_x, target, np.empty((0, x.shape[1])), np.empty(0), seed + index)
        predictions[test_mask] = incumbent_test_prediction + model.predict(test_x)
        importances.append(np.asarray(model.feature_importances_, dtype=float))
        fold_rows.append({
            "fold": index,
            "train_end": str(np.max(dates[train_mask])),
            "train_label_exit_max": str(np.max(np.asarray(label_exit_dates)[train_mask])) if label_exit_dates is not None else None,
            "test_start": str(np.min(dates[test_mask])),
            "test_end": str(test_dates[-1]),
            "train_rows": int(train_mask.sum()),
            "test_rows": int(test_mask.sum()),
            "embargo_sessions": embargo_sessions,
            "validation_labels_used_for_fitting": False,
        })
        _progress(
            progress_callback,
            phase="walk_forward_oof",
            message=f"严格过去样本 OOF 折 {index}/{len(blocks) - 1}",
            completed=index,
            total=len(blocks) - 1,
        )
    return predictions, importances, fold_rows


def _distilled_candidates(feature_names, importances, *, fields: list[str]) -> list[dict]:
    if not importances:
        return []
    matrix = np.vstack(importances)
    mean = matrix.mean(axis=0)
    stability = 1.0 / (1.0 + matrix.std(axis=0) / (mean + 1e-9))
    score = mean * stability
    order = np.argsort(score)[::-1]
    lookup = {row.name: row for row in [*ALPHA158_FEATURES, *_extension_rows(fields)]}
    top = [feature_names[index] for index in order if feature_names[index] in lookup][:8]
    candidates = []
    seen = set()

    def add(expression: str, components: list[str], kind: str):
        error = validate(expression, fields)
        key = normalize_hash(expression, direction_invariant=True) if not error else ""
        if error or key in seen:
            return
        seen.add(key)
        candidates.append({
            "expression": expression,
            "components": components,
            "distillation_kind": kind,
            "normalized_hash": key,
            "complexity": len(expression),
        })

    for name in top[:3]:
        add(lookup[name].expression, [name], "stable_feature")
    for left, right in zip(top[:3], top[1:4]):
        a, b = lookup[left].expression, lookup[right].expression
        add(f"rank({a})+rank({b})", [left, right], "equal_rank_blend")
        add(f"rank({a})-rank({b})", [left, right], "signed_rank_contrast")
        add(f"(rank({a})-0.5)*gt(rank({b}),0.5)", [left, right], "conditional_high_rank_gate")
    return candidates[:12]


def _distillation_fidelity(candidates, valid, teacher):
    """Measured sampled-universe fidelity, not a full DSL execution proof.

    Uses unfilled raw feature values. All validation evidence is META_TRAIN;
    DSL must still pass the authoritative evaluator on the full panel.
    """
    dates = valid["trade_date"].to_numpy()
    output = []
    for row in candidates:
        names = row["components"]
        ranks = valid.select([(pl.col(n).rank(method="average").over("trade_date") /
                 (pl.col(n).count().over("trade_date") + 1e-12)).alias(n) for n in names])
        a = ranks[names[0]].to_numpy()
        kind = row["distillation_kind"]
        if kind == "stable_feature":
            prediction = valid[names[0]].to_numpy()
        else:
            b = ranks[names[1]].to_numpy()
            prediction = a + b if kind == "equal_rank_blend" else a - b if kind == "signed_rank_contrast" else (a - .5) * (b > .5)
        metrics = _daily_ic(dates, prediction, np.asarray(teacher))
        output.append({**row, "teacher_fidelity": metrics,
            "fidelity_scope": "META_TRAIN_sampled_universe_proxy_requires_full_DSL_validation",
            "teacher_fidelity_passed": bool(metrics["days"] >= 30 and abs(metrics["mean_rank_ic"]) >= .1)})
    return output


def run_joint_alpha158(
    spec: JointModelSpec,
    *,
    task_key: str,
    incumbent_expressions: list[str] | None = None,
    source_unique_evaluations: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict:
    """Run one training-safe joint model and persist an auditable artifact."""
    spec.validate()
    started = time.perf_counter()
    incumbents = list(incumbent_expressions or [])[: spec.max_incumbents]
    matrix, cache = _materialize_matrix(spec, incumbents, progress_callback)
    feature_rows = _feature_rows(spec)
    feature_names = [row.name for row in feature_rows]
    incumbent_names = [name for name in matrix.columns if name.startswith("INCUMBENT_")]
    train = matrix.filter(pl.col("layer") == "INNER_PUBLIC")
    valid = matrix.filter(pl.col("layer") == "META_TRAIN")
    if train.height < 5_000 or valid.height < 1_000:
        raise ValueError("联合模型训练/验证样本不足")
    _progress(
        progress_callback,
        phase="split_prepare",
        message=f"训练/验证切分：{train.height:,}/{valid.height:,} 行",
        completed=1,
        total=1,
    )
    train_raw = train.select(feature_names).to_numpy().astype(np.float32)
    valid_raw = valid.select(feature_names).to_numpy().astype(np.float32)
    train_y = train["label"].to_numpy().astype(float)
    valid_y = valid["label"].to_numpy().astype(float)
    residual_meta = {"enabled": bool(incumbent_names), "incumbents": len(incumbent_names)}
    if incumbent_names:
        inc_train_raw = train.select(incumbent_names).to_numpy().astype(np.float32)
        inc_valid_raw = valid.select(incumbent_names).to_numpy().astype(np.float32)
        inc_processor = _fit_processor(inc_train_raw)
        inc_train = _transform(inc_train_raw, inc_processor)
        inc_valid = _transform(inc_valid_raw, inc_processor)
        incumbent_valid_prediction, coefficient = _ridge_fit_predict(inc_train, train_y, inc_valid)
        incumbent_train_prediction, _ = _ridge_fit_predict(inc_train, train_y, inc_train)
        train_target = train_y - incumbent_train_prediction
        valid_target = valid_y - incumbent_valid_prediction
        residual_meta["ridge_coefficients"] = [round(float(value), 8) for value in coefficient]
    else:
        incumbent_valid_prediction = np.zeros(len(valid_y), dtype=float)
        train_target, valid_target = train_y, valid_y

    train_dates = train["trade_date"].to_numpy()
    oof_prediction, fold_importances, fold_rows = _walk_forward_oof(
        train_raw,
        train_y,
        train_dates,
        spec.folds,
        spec.seed,
        progress_callback,
        label_exit_dates=train[f"label_exit_date_{spec.horizon}"].to_numpy(),
        embargo_sessions=spec.horizon,
        incumbent_raw=inc_train_raw if incumbent_names else None,
    )
    _progress(
        progress_callback,
        phase="final_model",
        message="使用全部 INNER_PUBLIC 拟合最终残差模型",
    )
    processor = _fit_processor(train_raw)
    train_x = _transform(train_raw, processor)
    valid_x = _transform(valid_raw, processor)
    model = _train_lgbm(train_x, train_target, np.empty((0, train_x.shape[1])), np.empty(0), spec.seed)
    residual_prediction = model.predict(valid_x)
    total_prediction = incumbent_valid_prediction + residual_prediction
    equal_weight_prediction = np.mean(valid_x, axis=1)
    single_train_scores = np.asarray([
        abs(_rank_correlation(train_x[:, index], train_y))
        for index in range(train_x.shape[1])
    ])
    best_single_index = int(np.argmax(single_train_scores))
    best_single_prediction = valid_x[:, best_single_index]
    importances = [*fold_importances, np.asarray(model.feature_importances_, dtype=float)]
    candidates = _distilled_candidates(feature_names, importances, fields=get_dsl_fields(spec.market))
    candidates = _distillation_fidelity(candidates, valid, residual_prediction)
    _progress(
        progress_callback,
        phase="dsl_distillation",
        message=f"稳定重要性蒸馏得到 {len(candidates)} 个 DSL 候选",
        completed=1,
        total=1,
    )
    oof_mask = np.isfinite(oof_prediction)
    valid_dates = valid["trade_date"].to_numpy()
    residual_metrics = _daily_ic(valid_dates, residual_prediction, valid_target)
    joint_metrics = _daily_ic(valid_dates, total_prediction, valid_y)
    incumbent_metrics = _daily_ic(
        valid_dates, incumbent_valid_prediction, valid_y
    ) if incumbent_names else None
    equal_weight_metrics = _daily_ic(
        valid_dates, equal_weight_prediction, valid_y
    )
    single_metrics = _daily_ic(valid_dates, best_single_prediction, valid_y)
    incremental_ic = float(joint_metrics["mean_rank_ic"]) - float(
        (incumbent_metrics or {"mean_rank_ic": 0.0})["mean_rank_ic"]
    )
    search_candidates = [row for row in candidates if row["teacher_fidelity_passed"]]
    search_eligible = _search_eligible(
        candidates=search_candidates,
        joint_metrics=joint_metrics,
        incremental_mean_rank_ic=incremental_ic,
        min_meta_dates=spec.min_meta_dates,
    )
    result = {
        "schema": PROTOCOL,
        "task_key": task_key,
        "source_unique_evaluations": (
            int(source_unique_evaluations)
            if source_unique_evaluations is not None
            else None
        ),
        "market": spec.market,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "automated_feedback_layers": list(AUTOMATED_LAYERS),
        "holdout_vault_consumed": False,
        "panel": cache,
        "spec": {
            "universe_n": spec.universe_n,
            "horizon": spec.horizon,
            "max_rows": spec.max_rows,
            "min_meta_dates": spec.min_meta_dates,
            "folds": spec.folds,
            "features": len(feature_names),
            "excluded_low_fidelity_vwap": not spec.include_low_fidelity_vwap,
        },
        "processor": {
            "fit_scope": "INNER_PUBLIC_only",
            "policy": "median_mad_clip3_then_fill0",
            "statistics_hash": hashlib.sha256(
                processor["median"].tobytes() + processor["scale"].tobytes()
            ).hexdigest(),
        },
        "model": {
            "backend": str(
                getattr(model, "_factorfactory_backend", "lightgbm")
            ),
            "objective": "cross_section_rank_label_residual",
            "best_iteration": int(getattr(model, "best_iteration_", 0) or model.n_estimators),
            "residual": residual_meta,
        },
        "inner_public_oof": {
            **_daily_ic(
                train_dates[oof_mask],
                oof_prediction[oof_mask],
                train_y[oof_mask],
            ),
            "strictly_past_only": True,
            "metric_target": "raw_forward_rank_label_joint_prediction",
            "label_boundary_policy": "purged_layer_and_fold_exit_dates_with_h_session_embargo",
            "folds": fold_rows,
        },
        "meta_train": {
            "residual": residual_metrics,
            "joint": joint_metrics,
            "incumbent": incumbent_metrics,
            "incremental_mean_rank_ic": round(incremental_ic, 6),
        },
        "baselines": {
            "best_single": {
                "feature": feature_names[best_single_index],
                "selection_scope": "INNER_PUBLIC_only",
                "meta_train": single_metrics,
            },
            "alpha_equal_weight": {
                "meta_train": equal_weight_metrics,
            },
            "incumbent": incumbent_metrics,
        },
        "distilled_candidates": search_candidates,
        "distillation_diagnostics": candidates,
        "search_eligible": search_eligible,
        "search_eligibility_rule": (
            f"META_TRAIN abs RankIC>=0.01, abs ICIR>=0.10, "
            f">={spec.min_meta_dates} days, incremental mean RankIC>0; "
            "candidate DSL still starts V4.2 from zero"
        ),
        "candidate_evaluation_authority": "FactorFactory V4.2 from zero; model metrics never grant admission",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "upstream_commit": QLIB_UPSTREAM_COMMIT,
    }
    task_dir = ARTIFACT_ROOT / "tasks" / _safe_slug(task_key)
    task_dir.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    artifact = task_dir / f"{version}.json"
    artifact.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    latest = task_dir / "latest.json"
    temp = latest.with_suffix(".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, latest)
    result["artifact_path"] = str(artifact.resolve())
    _progress(
        progress_callback,
        phase="complete",
        message="联合模型产物已写入不可变审计文件",
        completed=1,
        total=1,
    )
    return result


def latest_joint_result(task_key: str) -> dict | None:
    path = ARTIFACT_ROOT / "tasks" / _safe_slug(task_key) / "latest.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact_path"] = str(path.resolve())
    return payload
