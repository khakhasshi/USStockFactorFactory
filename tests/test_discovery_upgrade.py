import asyncio
import json
import random
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from backend.app.residual_beam import rank_correlation, residual_oof_beam_search, time_ordered_oof_residuals
from backend.app.qlib_joint import _rank_correlation, _daily_ic, _distillation_fidelity
from backend.app.discovery_evidence import score_channels, algorithm_diagnostics, persist_residual_artifact, combination_pool
from backend.app.search_pool import propose_search_seed, _quota_ucb
from backend.app.config import get_dsl_fields


def test_average_ties_constant_and_row_permutation():
    assert rank_correlation(np.ones(6), np.arange(6)) == 0
    assert _rank_correlation(np.ones(6), np.arange(6)) == 0
    x, y = np.array([1, 1, 1, 2, 2, 2]), np.arange(6)
    expected = np.corrcoef([1, 1, 1, 4, 4, 4], y)[0, 1]
    assert rank_correlation(x, y) == pytest.approx(expected)
    assert rank_correlation(x[::-1], y[::-1]) == pytest.approx(expected)


def test_no_fake_qlib_days_for_missing_or_constant_predictions():
    dates = np.repeat(np.arange(4), 10)
    assert _daily_ic(dates, np.ones(40), np.arange(40))["days"] == 0
    assert _daily_ic(dates, np.full(40, np.nan), np.arange(40))["days"] == 0


def test_oof_masks_warmup_embargo_and_bad_rows():
    x = np.arange(60.)[:, None]
    y = x[:, 0] + 2
    x[55] = np.nan
    r = time_ordered_oof_residuals(y, x, folds=6, groups=np.arange(60),
                                 label_exit_dates=np.arange(60) + 2, embargo_sessions=2)
    assert np.isnan(r[:12]).all()
    assert all(np.isnan(r[i]) for i in [20, 21, 30, 31, 40, 41, 50, 51, 55])
    assert np.isfinite(r[52:55]).all()


def test_joint_refit_handles_negative_loading_not_equal_weight_average():
    rng = np.random.default_rng(88)
    x, omitted = rng.normal(size=(2, 600))
    y = 2 * x - 4 * omitted + rng.normal(scale=.2, size=600)
    result = residual_oof_beam_search(target=y, incumbent_predictions=x,
        candidates={"negative": omitted, "duplicate": x, "constant": np.ones(600)}, folds=5)
    row = next(r for r in result["candidates"] if r["name"] == "negative")
    assert row["incremental_oof_ic"] > .5 and row["eligible"]
    assert not next(r for r in result["candidates"] if r["name"] == "duplicate")["eligible"]
    assert result["excluded_oof_rows"] == 120
    assert result["rejected"][0]["name"] == "constant"
    assert all(not r["net_sharpe_available"] for r in result["candidates"])


def test_beam_joint_refits_two_complementary_features_and_respects_budget():
    rng = np.random.default_rng(12)
    x, a, b = rng.normal(size=(3, 1000))
    y = x + a + b + rng.normal(scale=.1, size=1000)
    result = residual_oof_beam_search(target=y, incumbent_predictions=x,
        candidates={"a": a, "b": b}, beam_width=2, beam_depth=2, max_joint_fits=3)
    assert result["joint_refits"] == 3
    assert any(len(r["names"]) == 2 for r in result["combination_paths"])


@pytest.mark.parametrize("market", ["us", "ashare"])
def test_fallback_attributed_to_actual_arm(market):
    p = propose_search_seed(family="momentum", fields=get_dsl_fields(market), feedback_nodes=[],
        algorithms=["residual_oof_beam"], rng=random.Random(1), residual_oof_candidates=[])
    assert p.metadata["requested_algorithm"] == "residual_oof_beam"
    assert p.metadata["executed_algorithm"] == "structured_random"
    rows = [{"id": i, "status": "ok", "expression": f"returns(close,{i+1})", "proposal_meta": p.metadata} for i in range(5)]
    _, policy = _quota_ucb(("residual_oof_beam", "structured_random"), rows, random.Random(3))
    assert policy["algorithm_stats"]["residual_oof_beam"]["fallback_attempts"] == 5
    assert policy["algorithm_stats"]["residual_oof_beam"]["n"] == 0
    assert policy["algorithm_stats"]["structured_random"]["n"] == 5


def test_separate_scores_never_invent_missing_increment():
    result = score_channels({"score": 1.8, "passed": False}, {}, "ok")
    assert result["exploration"] == 1.8
    assert result["standalone_quality"] is None and result["incremental_quality"] is None
    assert not result["net_increment_confirmed"]


def test_immutable_pool_verifies_hash_and_does_not_grant_eligibility(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app.discovery_evidence.ROOT", tmp_path)
    artifact = persist_residual_artifact({"created_at": "now", "protocol": "v4", "combination_paths": []}, 7, "task")
    assert combination_pool(7)[0]["artifact_sha256"] == artifact["artifact_sha256"]
    assert not combination_pool(7)[0]["formal_eligible"]
    from pathlib import Path
    Path(artifact["artifact_path"]).write_text("{}")
    assert combination_pool(7)[0]["status"] == "INVALID_EVIDENCE"


def test_model_fidelity_constant_teacher_cannot_qualify():
    dates = np.repeat(np.arange(40), 10)
    valid = pl.DataFrame({"trade_date": dates, "x": np.tile(np.arange(10), 40)})
    row = {"components": ["x"], "distillation_kind": "stable_feature"}
    result = _distillation_fidelity([row], valid, np.ones(400))
    assert not result[0]["teacher_fidelity_passed"]


def test_refresh_progress_does_not_stop_at_archive_size(monkeypatch):
    from backend.app.orchestrator import Engine
    batches = [[SimpleNamespace(id=i, expression=f"expr{i}") for i in range(1, 301)],
               [SimpleNamespace(id=i, expression=f"expr{i}") for i in range(301, 501)]]
    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def execute(self, query): return SimpleNamespace(all=lambda: batches.pop(0))
    monkeypatch.setattr("backend.app.orchestrator.SessionLocal", Session)
    worker = SimpleNamespace(exp_id=1, _residual_campaign_progress={}, _normalized_expression_hash=lambda x: x)
    assert asyncio.run(Engine._discovery_unique_count(worker, "T1")) == 300
    assert asyncio.run(Engine._discovery_unique_count(worker, "T1")) == 500


def test_signal_screen_rejects_date_constant_and_sparse_signals():
    from backend.app.residual_beam import signal_quality
    frame = pl.DataFrame({"trade_date": np.repeat(np.arange(20), 10),
                          "constant_by_date": np.repeat(np.arange(20), 10),
                          "sparse": [None] * 190 + list(range(10)),
                          "valid": np.tile(np.arange(10), 20)})
    assert not signal_quality(frame, "constant_by_date")["accepted"]
    assert not signal_quality(frame, "sparse")["accepted"]
    assert signal_quality(frame, "valid")["accepted"]


def test_code_snapshot_of_deployment_copy_without_git(tmp_path, monkeypatch):
    from backend.app import audit_snapshot
    source = tmp_path / "copy"
    (source / "backend" / "app").mkdir(parents=True)
    (source / "backend" / "app" / "main.py").write_text("x = 1")
    (source / ".env").write_text("SECRET=not_for_snapshot")
    monkeypatch.setattr(audit_snapshot, "REPO_ROOT", source)
    result = audit_snapshot.code_snapshot(tmp_path / "snapshots")
    assert result["persisted"] and result["git_commit"] is None
    assert [row["path"] for row in result["source_files"]] == ["backend/app/main.py"]


def test_llm_revision_cannot_inherit_seed_increment():
    from backend.app.discovery_evidence import bind_proposal_evidence
    evidence = {"expression": "rank(close)", "incremental_oof_ic": .2}
    meta = {"combination_evidence": evidence, "executed_algorithm": "residual_oof_beam",
            "proposal_authority": "researcher_llm", "incremental_oof_ic": .2}
    result = bind_proposal_evidence("rank(vol)", meta, "rank(close)")
    assert "combination_evidence" not in result and "incremental_oof_ic" not in result
    assert result["executed_algorithm"] == "llm_seed_revision"
    assert bind_proposal_evidence("rank(close)", meta, "rank(close)")["combination_evidence"] == evidence


def test_equal_budget_caps_duplicates_and_isolates_feedback():
    from backend.app.discovery_benchmark import compare_arms
    def propose(arm, feedback, rng):
        assert all(n["proposal_meta"]["owner"] == arm for n in feedback)
        return SimpleNamespace(expression="rank(close)" if arm == "stuck" else f"returns(close,{len(feedback)+1})",
                               metadata={"owner": arm, "executed_algorithm": arm})
    report = compare_arms(["stuck", "fresh"], 3, 1, propose,
                          lambda expression: {"discovery": {"passed": False}}, attempt_multiplier=2)
    rows = {r["arm"]: r for r in report["summary"]}
    assert rows["stuck"]["evaluations"] == 1 and rows["stuck"]["attempts"] == 6
    assert rows["fresh"]["evaluations"] == 3 and rows["fresh"]["budget_complete"]
    assert not report["formal_eligible"]


def test_historical_requests_are_not_proof_of_execution():
    report = algorithm_diagnostics([{"task_name": "T1", "search_audit": {"algorithm": "residual_oof_beam"}}])
    assert report["rows"][0]["executed"] == "legacy_unverified"


def test_grouped_ic_fast_path_matches_explicit_date_masks():
    from backend.app.residual_beam import _ic_path
    rng = np.random.default_rng(98)
    groups = rng.integers(0, 15, 300)
    a, b = rng.normal(size=(2, 300))
    a[::7] = np.nan
    expected = [rank_correlation(a[groups == day], b[groups == day]) for day in np.unique(groups)]
    assert _ic_path(a, b, groups) == pytest.approx(expected)
