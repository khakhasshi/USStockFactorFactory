import numpy as np

from backend.app.research_overfit import (
    cscv_pbo,
    deflated_sharpe_ratio,
    effective_trial_count,
)
from backend.app.residual_beam import residual_oof_beam_search


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
