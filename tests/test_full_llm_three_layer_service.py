import asyncio
import json
from unittest.mock import AsyncMock, patch

from backend.app.scientific_governor import propose_scientific_directive
from backend.scripts.create_full_llm_three_layer_service import task_manifest


def test_full_llm_manifest_has_two_direct_expression_tasks():
    rows = task_manifest()
    assert len(rows) == 2
    assert {row["research_config"]["market"] for row in rows} == {"us", "ashare"}
    for row in rows:
        config = row["research_config"]
        assert config["service_port"] == 10013
        assert config["service_instance"] == "factorfactory-full-llm-three-layer"
        assert config["full_llm_architecture"] is True
        assert config["scientific_governor_enabled"] is True
        assert config["llm_roles"]["layer1"]["direct_expression_authority"] is True
        assert config["continuous_operation"] is True
        assert config["candidate_evaluation_budget"] == 0
        assert config["target_factor_count"] == 0
        assert config["frozen_rating"]["window_start"] == "2020-01-01"
        assert config["frozen_rating"]["window_end"] == "latest_available"
        assert not any(
            value
            for key, value in config["frozen_rating"].items()
            if key.startswith("visible_to_")
        )


def test_algorithm_seed_is_optional_in_full_llm_manifest():
    for row in task_manifest():
        engine = row["research_config"]["engine_config"]
        assert 0.0 < engine["algorithm_inspiration_share"] < 1.0
        assert engine["governor_warmup_candidates"] == 0


def test_scientific_governor_returns_audited_bounded_directive():
    response = json.dumps({
        "action": "rebalance",
        "focus_mechanisms": ["momentum", "invalid"],
        "deprioritize_mechanisms": ["reversal"],
        "exploration_share": 0.95,
        "director_objective": "检验动量机制是否能降低失败率",
        "falsification_rule": "完整配对 cohort 未改善则证伪",
        "expiry_outer_steps": 20,
        "confidence": 0.7,
        "evidence_used": ["v1 aggregate"],
        "reasoning_summary": "训练反馈显示需要重新分配",
    })
    with (
        patch(
            "backend.app.scientific_governor.llm.chat",
            new=AsyncMock(return_value=response),
        ) as chat,
        patch(
            "backend.app.scientific_governor.llm.mark_validation",
            new=AsyncMock(),
        ) as validation,
    ):
        directive, _, source, reflection = asyncio.run(
            propose_scientific_directive(
                [{
                    "version_no": 1,
                    "template": {
                        "_readonly": {
                            "isolation_layers": ["META_HOLDOUT"]
                        }
                    },
                    "feedback_summary": {},
                    "evaluation_protocol": "v4.2",
                }],
                {"name": "test", "api_key": "x"},
                market="us",
                portfolio_mode="long_short",
                allowed_mechanisms=("momentum", "reversal"),
                trace_context={"experiment_id": 1},
            )
        )
    assert source == "scientific_governor_llm"
    assert directive["focus_mechanisms"] == ["momentum"]
    assert directive["exploration_share"] == 0.8
    assert directive["expiry_outer_steps"] == 6
    assert directive["decision_status"] == "llm_accepted"
    assert reflection["architecture_layer"] == 3
    assert chat.await_args.kwargs["trace"]["role"] == "scientific_governor"
    validation.assert_awaited_once_with(response, accepted=True)
