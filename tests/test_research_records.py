from datetime import datetime
from types import SimpleNamespace

from backend.app.research_records import (
    RESEARCH_RECORD_SCHEMA_VERSION,
    research_record_payload,
    task_research_summary,
)


def _node(**overrides):
    values = {
        "id": 41,
        "experiment_id": 20,
        "task_name": "us_long_short_top500_h5",
        "evaluation_protocol": "walk_forward_public_v4",
        "miner_version_id": 9,
        "outer_step_no": 2,
        "seed": 3,
        "parent_id": 40,
        "op": "improve",
        "source": "llm",
        "status": "ok",
        "expression": "rank(ts_delta(close, 20))",
        "hypothesis": "medium-term price continuation",
        "public_score": 0.62,
        "public_metrics": {
            "icir": 0.8,
            "discovery": {
                "learning_score": 0.73,
                "gate_score": 0.51,
                "passed": False,
                "selected_direction": 1,
                "direction_policy": "fixed",
                "failure_reasons": ["cost_cushion"],
                "effective_metrics": {
                    "icir": 0.81,
                    "portfolio_sharpe": 0.7,
                    "daily_turnover": 0.22,
                },
            },
        },
        "gate_metrics": {"sealed_holdout_sharpe": 9.9},
        "proposal_meta": {
            "declared_family": "momentum",
            "reflection": "reduce turnover",
            "targeted_failures": ["cost_cushion"],
        },
        "feedback_summary": {
            "improvement_targets": ["lower turnover"],
        },
        "created_at": datetime(2026, 8, 21, 12, 0, 0),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_research_record_is_training_safe_and_explicitly_non_formal():
    payload = research_record_payload(_node(), task_rank=2)

    assert payload["schema_version"] == RESEARCH_RECORD_SCHEMA_VERSION
    assert payload["task_rank"] == 2
    assert payload["record_tier"] == "research_candidate"
    assert payload["formal_factor_admitted"] is False
    assert payload["learning_score"] == 0.73
    assert payload["metrics"]["icir"] == 0.81
    assert payload["search_audit"]["evaluation_performed"] is True
    assert "gate_metrics" not in payload
    assert "sealed_holdout_sharpe" not in str(payload)
    assert "not formal factor" in payload["interpretation_boundary"]


def test_research_record_can_link_to_formal_factor_without_changing_boundary():
    payload = research_record_payload(
        _node(), task_rank=1, formal_factor_id=17, research_factor_id=17
    )

    assert payload["record_tier"] == "formal_factor"
    assert payload["formal_factor_admitted"] is True
    assert payload["formal_factor_id"] == 17
    assert payload["research_candidate_registered"] is True
    assert "production approval" in payload["interpretation_boundary"]


def test_task_summary_aggregates_counts_and_best_scores():
    rows = [
        research_record_payload(_node(id=1, source="random"), task_rank=1),
        research_record_payload(
            _node(
                id=2,
                status="rejected",
                source="search_pool",
                public_score=0.0,
                public_metrics={"discovery": {"learning_score": 0.0}},
                proposal_meta={
                    "pre_evaluation_rejection": "duplicate_normalized_ast",
                    "evaluation_performed": False,
                    "budget_charged": False,
                },
            ),
            task_rank=2,
        ),
    ]

    summary = task_research_summary(rows)[0]
    assert summary["records"] == 2
    assert summary["valid"] == 1
    assert summary["passed"] == 0
    assert summary["pre_eval_rejected"] == 1
    assert summary["best_learning_score"] == 0.73
    assert summary["source_counts"] == {"random": 1, "search_pool": 1}
