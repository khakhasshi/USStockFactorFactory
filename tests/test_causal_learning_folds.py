from datetime import date, timedelta

import numpy as np
import polars as pl

from backend.app.data.causal import purged_fold_masks, purged_layer_predicate
from backend.app.qlib_joint import _walk_forward_oof
from backend.app.residual_beam import time_ordered_oof_residuals


def test_shared_layer_filter_purges_boundary_labels_and_embargoes_validation():
    days = [date(2022, 12, 28) + timedelta(days=i) for i in range(10)]
    panel = pl.DataFrame({"trade_date": days,
        "layer": ["META_TRAIN" if d.year == 2022 else "META_HOLDOUT" for d in days],
        "label_exit_date_1": [d + timedelta(days=2) for d in days]})
    result = panel.filter(purged_layer_predicate(panel, "us", 1, ["META_TRAIN", "META_HOLDOUT"]))
    assert date(2022, 12, 30) not in result["trade_date"].to_list()
    assert date(2023, 1, 1) not in result["trade_date"].to_list()
    assert date(2023, 1, 2) in result["trade_date"].to_list()


def test_fold_masks_require_settled_labels_and_embargo_test_start():
    groups = np.arange(20)
    train, test = purged_fold_masks(groups, np.arange(10, 20), label_exit_dates=groups + 3, embargo_sessions=2)
    assert np.flatnonzero(train).tolist() == list(range(7))
    assert np.flatnonzero(test).tolist() == list(range(12, 20))


def test_residual_oof_cannot_use_unsettled_training_labels():
    rng = np.random.default_rng(47)
    groups = np.arange(60)
    x = rng.normal(size=(60, 2))
    y = x[:, 0] + rng.normal(size=60)
    before = time_ordered_oof_residuals(y, x, folds=6, groups=groups,
        label_exit_dates=groups + 3, embargo_sessions=2)
    changed = y.copy()
    changed[47:50] += 1000
    after = time_ordered_oof_residuals(changed, x, folds=6, groups=groups,
        label_exit_dates=groups + 3, embargo_sessions=2)
    assert np.allclose((y - before)[52:], (changed - after)[52:])


def test_qlib_oof_test_labels_do_not_affect_own_predictions_or_incumbent_fit(monkeypatch):
    rng = np.random.default_rng(71)
    dates = np.repeat(np.arange(80), 60)
    x = rng.normal(size=(len(dates), 3)).astype(np.float32)
    incumbent = rng.normal(size=(len(dates), 2)).astype(np.float32)
    y = x[:, 0] + incumbent[:, 0] + rng.normal(size=len(dates))
    calls = []

    def train(train_x, train_y, valid_x, valid_y, seed):
        assert len(valid_y) == len(valid_x) == 0
        calls.append(len(train_y))
        coef = np.linalg.lstsq(train_x, train_y, rcond=None)[0]
        class Model:
            feature_importances_ = np.abs(coef)
            def predict(self, values):
                return values @ coef
        return Model()

    monkeypatch.setattr("backend.app.qlib_joint._train_lgbm", train)
    first, _, fold_rows = _walk_forward_oof(x, y, dates, folds=3, seed=9,
        label_exit_dates=dates + 6, embargo_sessions=5, incumbent_raw=incumbent)
    changed = y.copy()
    changed[dates >= 60] += 10000
    second, _, _ = _walk_forward_oof(x, changed, dates, folds=3, seed=9,
        label_exit_dates=dates + 6, embargo_sessions=5, incumbent_raw=incumbent)
    last_fold = (dates >= 65) & np.isfinite(first)
    assert calls and last_fold.any()
    assert np.allclose(first[last_fold], second[last_fold])
    assert all(int(row["train_label_exit_max"]) < int(row["test_start"]) - 4 for row in fold_rows)
