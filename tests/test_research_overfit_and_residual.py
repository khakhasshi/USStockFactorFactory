import numpy as np
import polars as pl
import pytest
from copy import deepcopy
from datetime import date, timedelta

from backend.app.research_overfit import (
    cscv_pbo,
    deflated_sharpe_ratio,
    effective_trial_count,
    build_trial_return_evidence,
    formal_overfit_governance,
    return_matrix_diagnostics,
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
    assert result["splits"] == 70
    assert result["complementary_roles_included"] is True


def test_dsr_annualized_inputs_are_converted_to_observation_units():
    daily = deflated_sharpe_ratio(.10, observations=252, effective_trials=10, sharpe_std=.05)
    annual = deflated_sharpe_ratio(.10 * np.sqrt(252), observations=252,
                                   effective_trials=10, sharpe_std=.05 * np.sqrt(252),
                                   annualization_factor=252)
    assert daily["dsr_probability"] == annual["dsr_probability"]
    assert annual["observed_sharpe"] == .10
    assert annual["sharpe_units"] == "per_observation_not_annualized"


def test_pbo_selects_sharpe_not_raw_mean_and_uses_both_roles():
    rng = np.random.default_rng(21)
    values = rng.normal(loc=.04, scale=.7, size=(80, 5))
    before = cscv_pbo(values)
    after = cscv_pbo(values * np.array([1, 10, .1, 100, .01]))
    reversed_roles = cscv_pbo(values[::-1])
    assert before["pbo"] == after["pbo"] == reversed_roles["pbo"]
    assert before["selection_metric"] == "nonannualized_sample_sharpe"


def test_pbo_ties_do_not_depend_on_column_order_and_refuses_nan():
    rng = np.random.default_rng(22)
    base = rng.normal(size=80)
    matrix = np.column_stack([base, base, rng.normal(size=80)])
    assert cscv_pbo(matrix)["pbo"] == cscv_pbo(matrix[:, ::-1])["pbo"]
    matrix[1, 0] = np.nan
    assert cscv_pbo(matrix)["available"] is False


def _raw_evidence(seed=5, context=None):
    rng = np.random.default_rng(seed)
    dates = [(date(2018, 1, 1) + timedelta(days=i)).isoformat() for i in range(80)]
    values = rng.normal(.001, .01, 80)
    return build_trial_return_evidence(
        paths=[{"direction": direction, "dates": dates,
                "returns": list(direction * values - .0001),
                "layers": ["INNER_PUBLIC"] * len(dates)} for direction in [1, -1]],
        context=context or {"market": "us", "portfolio_mode": "long_short", "horizon": 1,
                            "task_name": "T1", "panel_snapshot_id": "snapshot1",
                            "code_version": "code1", "config_hash": "config1"},
    )


def test_raw_trial_evidence_refuses_holdout_and_incomplete_directions():
    evidence = _raw_evidence()
    paths = deepcopy(evidence["paths"])
    paths[0]["layers"][-1] = "META_HOLDOUT"
    with pytest.raises(ValueError, match="prohibited"):
        build_trial_return_evidence(paths=paths, context=evidence["context"])
    with pytest.raises(ValueError, match="attempted direction"):
        build_trial_return_evidence(paths=evidence["paths"][:1], context=evidence["context"])


def test_formal_governance_missing_historical_returns_fails_closed():
    evidence = _raw_evidence()
    trials = [{"id": 1, "statistic": {"raw_return_evidence": evidence}},
              {"id": 2, "selected": False, "statistic": {
                  "training_return_path_signature": {"vector": [1, 2, 3]}}},
              {"id": 3, "statistic": {"evaluation_performed": False}}]
    result = formal_overfit_governance(trials, evidence, predeclared_trials=1)
    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["passed"] is False
    assert result["missing_evidence_trials"] == 1
    assert result["evaluated_trials"] == 2
    assert result["attempted_direction_trials"] == 4
    assert result["pre_evaluation_rejections"] == 1
    assert "incomplete_historical_trial_returns" in result["reasons"]


def test_formal_governance_uses_actual_raw_paths_and_counts_losing_trials():
    evidence = _raw_evidence()
    trials = [{"id": 1, "selected": True, "statistic": {"raw_return_evidence": evidence}},
              {"id": 2, "selected": False, "statistic": {"raw_return_evidence": _raw_evidence(7)}}]
    result = formal_overfit_governance(trials, evidence, predeclared_trials=1)
    assert result["available"] is True
    assert result["status"] in {"PASS", "FAIL"}
    assert result["effective_trials"] == 4
    assert result["pbo"]["trials"] == 4
    assert result["dsr"]["observed_sharpe"] == round(
        float(np.mean(evidence["paths"][1]["returns"]) / np.std(evidence["paths"][1]["returns"], ddof=1)), 6)
    assert result["holdout_vault_consumed"] is False


def test_formal_governance_rejects_tampered_paths_and_mixed_calendar():
    evidence = _raw_evidence()
    tampered = deepcopy(evidence)
    tampered["paths"][0]["returns"][0] += .1
    result = formal_overfit_governance([{"statistic": {"raw_return_evidence": tampered}}], evidence)
    assert result["available"] is False
    shifted = deepcopy(evidence)
    for path in shifted["paths"]:
        path["dates"] = [(date.fromisoformat(value) + timedelta(days=1)).isoformat() for value in path["dates"]]
    shifted = build_trial_return_evidence(paths=shifted["paths"], context=shifted["context"])
    result = formal_overfit_governance([
        {"statistic": {"raw_return_evidence": evidence}}, {"statistic": {"raw_return_evidence": shifted}}], evidence)
    assert "nonidentical_observation_calendar_no_fill_or_intersection_allowed" in result["reasons"]


def test_formal_governance_cannot_use_unregistered_candidate_or_subset_for_budget():
    evidence = _raw_evidence()
    result = formal_overfit_governance([{"statistic": {"raw_return_evidence": _raw_evidence(7)}}], evidence)
    assert "candidate_not_registered_in_campaign" in result["reasons"]
    result = formal_overfit_governance([{"statistic": {"raw_return_evidence": evidence}}], evidence,
                                     max_matrix_cells=1)
    assert "complete_matrix_exceeds_audit_resource_limit_no_subset_substitution" in result["reasons"]


def test_matrix_diagnostics_derives_sharpe_dispersion_from_trial_scores():
    rng = np.random.default_rng(133)
    matrix = rng.normal(.001, .01, size=(200, 5))
    result = return_matrix_diagnostics(matrix, candidate_index=1)
    scores = matrix.mean(axis=0) / matrix.std(axis=0, ddof=1)
    assert result["observed_sharpe"] == pytest.approx(scores[1])
    assert result["sharpe_dispersion"] == pytest.approx(max(scores.std(ddof=1), 1 / np.sqrt(199)))
    assert result["formal_admission_evidence"] is False


def test_overfit_diagnostic_api_ignores_legacy_sharpe_override():
    import asyncio
    from backend.app.api.routes import OverfitDiagnosticsRequest, research_overfit_diagnostics

    matrix = np.random.default_rng(21).normal(.001, .01, size=(80, 4)).tolist()
    normal = asyncio.run(research_overfit_diagnostics(OverfitDiagnosticsRequest(period_return_matrix=matrix)))
    manipulated = asyncio.run(research_overfit_diagnostics(OverfitDiagnosticsRequest(
        period_return_matrix=matrix, observed_sharpe=1_000_000, observations=1_000_000,
        skewness=-1000, kurtosis=10000)))
    assert manipulated["dsr"] == normal["dsr"]
    assert manipulated["observations"] == 80


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
    assert np.isnan(baseline[:6]).all()  # warm-up is not a zero-prediction OOF fold
    assert np.allclose(baseline[:24], after_future_change[:24], equal_nan=True)


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
    assert np.allclose(baseline[groups < 11], after[groups < 11], equal_nan=True)


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
                "label_exit_date_1": day + timedelta(days=2),
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
    assert artifact["rows"] == 1_450  # One META_TRAIN session is embargoed.
    assert artifact["beam"][0]["expression"] == "rank(close)"
    assert artifact["beam"][0]["residual_rank_ic"] > 0.6
