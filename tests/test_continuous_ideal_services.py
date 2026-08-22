import asyncio
from unittest.mock import AsyncMock

from backend.app.orchestrator import Engine
from backend.scripts.create_continuous_ideal_services import task_manifest


def test_continuous_manifest_has_four_bound_unlimited_tasks():
    rows = task_manifest()
    assert len(rows) == 4
    assert {row["research_config"]["service_port"] for row in rows} == {
        10011,
        10012,
    }
    assert {row["research_config"]["market"] for row in rows} == {
        "us",
        "ashare",
    }
    for row in rows:
        config = row["research_config"]
        assert config["continuous_operation"] is True
        assert config["service_autostart"] is True
        assert config["candidate_evaluation_budget"] == 0
        assert config["target_factor_count"] == 0
        assert config["budget_policy"]["mode"] == "unlimited"
        assert config["layer1_enabled"] is True
        assert config["layer2_enabled"] is True
        assert config["layer3_enabled"] is (
            config["service_port"] == 10012
        )
        assert config["frozen_rating"]["window_start"] == "2020-01-01"
        assert config["frozen_rating"]["window_end"] == "latest_available"
        assert config["frozen_rating"]["resolved_calendar_years"] == [
            2020,
            2026,
        ]
        assert config["frozen_rating"]["visible_to_layer2_researcher"] is False
        assert config["frozen_rating"]["visible_to_layer3_governor"] is False
        assert config["pit_policy"]["factor_admission_gate"] is False


def test_continuous_mode_bypasses_all_scientific_stop_budgets():
    engine = Engine()
    engine.task_config = {
        "continuous_operation": True,
        "candidate_evaluation_budget": 0,
    }
    engine.status["candidate_evaluations"] = 10_000
    engine._llm_call_count = AsyncMock(return_value=10_000)
    reached = asyncio.run(
        engine._stop_if_scientific_budget_reached(
            {
                "max_outer_steps": 1,
                "max_runtime_hours": 0.0001,
                "max_llm_calls": 1,
            },
            next_step_no=10_001,
        )
    )
    assert reached is False
    assert engine.status["budget_mode"] == "unlimited"
    assert engine.status["max_outer_steps"] is None
    assert engine.status["max_runtime_hours"] is None
    assert engine.status["max_llm_calls"] is None
    assert engine.status["candidate_evaluation_progress"] is None
