import numpy as np
import polars as pl
from datetime import date, timedelta

from backend.app.research_overfit import (
    cscv_pbo,
    deflated_sharpe_ratio,
    effective_trial_count,
)
from backend.app.residual_beam import residual_oof_beam_search
from backend.app.residual_beam import time_ordered_oof_residuals
from backend.app.residual_beam import build_dsl_residual_oof_artifact


def test_effective_trials_shrinks_duplicate_strategies():
    base = np.linspace(-1, 1, 40)
    matrix = np.column_stack([base, base, -base, np.sin(base * 4)])
    assert 1.0 <= effective_trial_count(matrix) < 4.0


def test_dsr_penalises_more_trials():
    few = deflated_sharpe_ratio(1.2, observations=252, effective_trials=2)
    many = deflated_sharpe_ratio(1.2, observations=252, effective_trials=500)
    assert few["dsr_probability"] > many["dsr_probability"]


def test_cscv_returns_a_bounded_pbo():
    rng = np.random.default_rng(7)
    result = cscv_pbo(rng.normal(size=(80, 8)), blocks=8)
    assert result["available"] is True
    assert 0.0 <= result["pbo"] <= 1.0


def test_residual_beam_prefers_the_unexplained_signal():
    rng = np.random.default_rng(11)
    n = 120
    incumbent = rng.normal(size=n)
    omitted = rng.normal(size=n)
    target = 0.8 * incumbent + 0.7 * omitted + rng.normal(scale=0.1, size=n)
    result = residual_oof_beam_search(
        target=target,
        incumbent_predictions=incumbent[:, None],
        candidates={"omitted": omitted, "duplicate": incumbent},
        folds=6,
        beam_width=2,
    )
    assert result["beam"][0]["name"] == "omitted"
    assert result["beam"][0]["residual_rank_ic"] > 0.7


def test_residual_oof_never_uses_future_blocks():
    target = np.arange(30, dtype=float)
    incumbent = np.arange(30, dtype=float)[:, None]
    baseline = time_ordered_oof_residuals(target, incumbent, folds=5)
    changed = target.copy()
    changed[24:] += 10_000
    after_future_change = time_ordered_oof_residuals(changed, incumbent, folds=5)
    # A mutation confined to the final fold cannot alter earlier residuals.
    assert np.allclose(baseline[:24], after_future_change[:24])


def test_residual_oof_keeps_whole_dates_in_the_same_fold():
    groups = np.repeat(np.arange(12), 8)
    incumbent = np.tile(np.linspace(-1, 1, 8), 12)[:, None]
    target = incumbent[:, 0] + np.repeat(np.arange(12) * 0.01, 8)
    baseline = time_ordered_oof_residuals(
        target, incumbent, folds=6, groups=groups
    )
    changed = target.copy()
    changed[groups == 11] += 10_000
    after = time_ordered_oof_residuals(
        changed, incumbent, folds=6, groups=groups
    )
    assert np.allclose(baseline[groups < 11], after[groups < 11])


def test_dsl_residual_artifact_uses_real_training_rows(monkeypatch):
    rng = np.random.default_rng(19)
    rows = []
    start = date(2014, 1, 1)
    for day_index in range(30):
        day = start + timedelta(days=day_index)
        layer = "INNER_PUBLIC" if day_index < 20 else "META_TRAIN"
        omitted = rng.normal(size=50)
        incumbent = rng.normal(size=50)
        target = 0.8 * omitted + 0.2 * incumbent + rng.normal(scale=0.05, size=50)
        for code_index in range(50):
            rows.append({
                "trade_date": day,
                "ts_code": f"S{code_index:03d}",
                "layer": layer,
                "univ_rank": code_index + 1,
                "open": float(incumbent[code_index]),
                "close": float(omitted[code_index]),
                "fwd_1": float(target[code_index]),
            })
    frame = pl.DataFrame(rows)

    class Store:
        def read_snapshot(self):
            return frame, tuple(), "synthetic-panel", 1

    monkeypatch.setattr(
        "backend.app.residual_beam.PanelStore.get",
        lambda *args, **kwargs: Store(),
    )
    artifact = build_dsl_residual_oof_artifact(
        market="us",
        panel_glob=None,
        universe_n=50,
        horizon=1,
        incumbent_expressions=["rank(open)"],
        candidate_rows=[{
            "expression": "rank(close)",
            "family": "momentum",
        }],
        folds=5,
        beam_width=1,
        max_rows=5_000,
        security_sample_modulus=1,
    )
    assert artifact["date_grouped_folds"] is True
    assert artifact["holdout_vault_consumed"] is False
    assert artifact["rows"] == 1_500
    assert artifact["beam"][0]["expression"] == "rank(close)"
    assert artifact["beam"][0]["residual_rank_ic"] > 0.6
