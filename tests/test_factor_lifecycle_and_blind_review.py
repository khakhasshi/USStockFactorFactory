from types import SimpleNamespace

from backend.app.blind_review import deterministic_code_review
from backend.app.factor_lifecycle import factor_evidence_state


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


def test_only_fully_audited_and_double_blind_passed_is_formal():
    factor = SimpleNamespace(
        status="library-admitted",
        lifecycle_stage="live_candidate",
        validation_metrics={
            "ranking": {
                "available": True,
                "score": 82.0,
                "rating_protocol_version": "v4.3",
            }
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
    state = factor_evidence_state(
        factor,
        current_code_review={"passed": True},
        current_rating_protocol="v4.3",
    )
    assert state["formal_factor"] is True
    assert state["tier"] == "formal_research_factor"
