from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from backend.app.qlib_joint import (
    _fit_processor,
    _sample_dates_by_layer,
    _search_eligible,
    _transform,
    _walk_forward_oof,
)


def test_joint_search_requires_strictly_positive_incremental_rank_ic():
    metrics = {"days": 80, "mean_rank_ic": 0.03, "rank_icir": 0.40}
    assert _search_eligible(
        candidates=[{"expression": "rank(close)"}],
        joint_metrics=metrics,
        incremental_mean_rank_ic=0.001,
        min_meta_dates=60,
    ) is True
    assert _search_eligible(
        candidates=[{"expression": "rank(close)"}],
        joint_metrics=metrics,
        incremental_mean_rank_ic=0.0,
        min_meta_dates=60,
    ) is False
    assert _search_eligible(
        candidates=[{"expression": "rank(close)"}],
        joint_metrics=metrics,
        incremental_mean_rank_ic=-0.001,
        min_meta_dates=60,
    ) is False


def _layer_frame(train_days=100, meta_days=80, rows_per_day=100):
    rows = []
    start = date(2018, 1, 1)
    for offset in range(train_days + meta_days):
        layer = "INNER_PUBLIC" if offset < train_days else "META_TRAIN"
        day = start + timedelta(days=offset)
        rows.extend((layer, day, index) for index in range(rows_per_day))
    return pl.DataFrame(rows, schema=["layer", "trade_date", "row"], orient="row")


def test_layer_stratified_sampling_preserves_minimum_meta_dates():
    frame = _layer_frame()
    dates, metadata = _sample_dates_by_layer(
        frame, 12_000, min_meta_dates=60
    )
    sampled = frame.filter(pl.col("trade_date").is_in(dates))
    assert sampled.height <= 12_000
    assert metadata["selected_dates_by_layer"]["META_TRAIN"] >= 60
    assert metadata["selected_dates_by_layer"]["INNER_PUBLIC"] >= 2
    assert sampled.filter(pl.col("layer") == "META_TRAIN")[
        "trade_date"
    ].n_unique() >= 60


def test_layer_stratified_sampling_rejects_impossible_row_budget():
    frame = _layer_frame(rows_per_day=200)
    with pytest.raises(ValueError, match="max_rows"):
        _sample_dates_by_layer(frame, 10_000, min_meta_dates=60)


def test_processor_statistics_are_fitted_only_from_supplied_training_rows():
    train = np.asarray([[1.0, 10.0], [2.0, 11.0], [3.0, 12.0]])
    processor = _fit_processor(train)
    transformed = _transform(np.asarray([[1000.0, -1000.0]]), processor)
    assert processor["median"].tolist() == [2.0, 11.0]
    assert transformed.tolist() == [[3.0, -3.0]]


def test_joint_walk_forward_predictions_are_strictly_past_only():
    rng = np.random.default_rng(17)
    rows_per_day = 60
    days = 36
    dates = np.asarray([
        date(2019, 1, 1) + timedelta(days=day)
        for day in range(days)
        for _ in range(rows_per_day)
    ])
    x = rng.normal(size=(days * rows_per_day, 5)).astype(np.float32)
    y = x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.1, size=len(x))
    first, _, folds = _walk_forward_oof(x, y, dates, folds=3, seed=19)
    changed = y.copy()
    changed[dates >= date(2019, 1, 28)] += 10_000
    second, _, _ = _walk_forward_oof(x, changed, dates, folds=3, seed=19)
    earlier = dates < date(2019, 1, 28)
    mask = earlier & np.isfinite(first) & np.isfinite(second)
    assert folds
    assert np.allclose(first[mask], second[mask])
