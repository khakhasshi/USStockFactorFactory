from types import SimpleNamespace
import hashlib
from copy import deepcopy

from backend.app.blind_review import deterministic_code_review
from backend.app.factor_lifecycle import factor_evidence_state
from backend.app.research_overfit import PROTOCOL as OVERFIT_PROTOCOL


def test_ashare_blind_review_uses_ashare_field_whitelist():
    expression = "rank(ts_corr(buy_elg_amount/(amount+1e-9), returns(close, 1), 20))"
    review = deterministic_code_review(expression, "ashare")
    assert review["dsl_error"] is None
    assert review["passed"] is True
    assert review["market"] == "ashare"


def test_training_pass_is_explicitly_not_a_formal_factor():
    factor = SimpleNamespace(
        status="public-leading",
        lifecycle_stage="research_pass",
        validation_metrics={},
        eligibility={},
        research_meta={
            "double_blind_review": {
                "method_review": "pending_optional_llm_shortlist_review",
                "code_review": {"passed": True},
            }
        },
    )
    state = factor_evidence_state(
        factor,
        current_code_review={"passed": True},
        current_rating_protocol="v4.3",
    )
    assert state["tier"] == "training_candidate"
    assert state["formal_factor"] is False
    assert state["effective_status"] == "research-candidate"
    assert "holdout_pass" in state["promotion_blockers"]


def _fully_audited_factor():
    expression = "rank(close)"
    factor = SimpleNamespace(
        expression=expression,
        status="library-admitted",
        lifecycle_stage="live_candidate",
        validation_metrics={
            "ranking": {
                "available": True,
                "score": 82.0,
                "rating_protocol_version": "v4.3",
            },
            "event_audit": {"passed": True, "status": "PASS"},
            "audit_provenance": {
                "immutable_inputs_available": True,
                "panel": {"immutable_inputs_available": True, "status": "FROZEN",
                          "data_sha256": "a" * 64, "snapshot_id": "b" * 64,
                          "market_calendar_sha256": "c" * 64,
                          "files": [{"snapshot_path": "immutable-panel"}],
                          "code": {"persisted": True, "code_sha256": "d" * 64,
                                   "source_files": [{"snapshot_path": "immutable-source"}]}},
                "expression_sha256": hashlib.sha256(expression.encode()).hexdigest(),
                "config_sha256": "e" * 64, "run_fingerprint": "f" * 64,
                "requested_start": "2020-01-01", "requested_end": "2026-12-31",
                "actual_start": "2020-01-02", "actual_end": "2026-09-11", "actual_sessions": 1600,
            },
            "overfit_governance": {"protocol": OVERFIT_PROTOCOL, "status": "PASS", "available": True, "passed": True,
                                   "missing_evidence_trials": 0, "dsr_pass": True, "pbo_pass": True},
        },
        eligibility={
            "research_pass": True,
            "holdout_pass": True,
            "vault_pass": True,
            "capacity_pass": True,
        },
        research_meta={
            "double_blind_review": {
                "method_review": {"passed": True},
                "code_review": {"passed": True},
            }
        },
    )
    factor.validation_metrics["event_audit"].update(
        protocol="factorfactory.event-promotion-gate/v1",
        contract={"direction_frozen": 1, "universe_n": 500, "rebalance_sessions": 5},
        windows={name: {"passed": True, "status": "PASS", "integrity": {"all_pass": True},
                        "input_provenance": deepcopy(factor.validation_metrics["audit_provenance"]),
                        "config": {"direction": 1, "universe_n": 500, "rebalance_every": 5}}
                 for name in ("holdout", "vault", "rating")},
    )
    return factor


def test_only_fully_audited_and_double_blind_passed_is_formal():
    factor = _fully_audited_factor()
    state = factor_evidence_state(
        factor,
        current_code_review={"passed": True},
        current_rating_protocol="v4.3",
    )
    assert state["formal_factor"] is True
    assert state["tier"] == "formal_research_factor"


def test_legacy_vector_pass_does_not_imply_formal_admission():
    factor = _fully_audited_factor()
    factor.validation_metrics = {"ranking": factor.validation_metrics["ranking"]}
    state = factor_evidence_state(factor, current_rating_protocol="v4.3")
    assert state["formal_factor"] is False
    assert {"event_pass", "frozen_input_pass", "overfit_evidence_complete", "dsr_pass", "pbo_pass"}.issubset(
        state["promotion_blockers"])


def test_missing_historical_overfit_paths_block_formal_promotion_even_with_flags():
    factor = _fully_audited_factor()
    factor.validation_metrics["overfit_governance"].update(status="INSUFFICIENT_DATA", missing_evidence_trials=10)
    state = factor_evidence_state(factor)
    assert state["formal_factor"] is False
    assert "overfit_evidence_complete" in state["promotion_blockers"]


def test_snapshot_must_match_factor_expression_and_be_persisted():
    factor = _fully_audited_factor()
    changed = deepcopy(factor)
    changed.expression = "rank(open)"
    assert "frozen_input_pass" in factor_evidence_state(changed)["promotion_blockers"]
    factor.validation_metrics["audit_provenance"]["panel"]["code"]["persisted"] = False
    assert "frozen_input_pass" in factor_evidence_state(factor)["promotion_blockers"]


def test_event_flag_alone_or_different_event_inputs_cannot_promote():
    factor = _fully_audited_factor()
    old = deepcopy(factor)
    old.validation_metrics["event_audit"] = {"passed": True}
    assert "event_pass" in factor_evidence_state(old)["promotion_blockers"]
    factor.validation_metrics["event_audit"]["windows"]["vault"]["input_provenance"]["panel"]["data_sha256"] = "1" * 64
    assert "event_pass" in factor_evidence_state(factor)["promotion_blockers"]
