import random
import asyncio

import pytest

from backend.app.api.routes import _three_layer_arm_configs
from backend.app.config import DEFAULT_MINER_TEMPLATE, get_dsl_fields
from backend.app.dsl.engine import validate
from backend.app.llm import client as llm_client
from backend.app.meta import agent as meta_agent
from backend.app.miner import agent as miner_agent
from backend.app.orchestrator import Engine
from backend.app.search_pool import (
    DEFAULT_SEARCH_ALGORITHMS,
    SEARCH_POOL_SCHEMA_VERSION,
    propose_search_seed,
)


def test_abcde_arms_isolate_one_architecture_increment_at_a_time():
    arms = _three_layer_arm_configs()
    assert [row["arm"] for row in arms] == list("ABCDE")
    assert [row["layer2_enabled"] for row in arms] == [False, False, True, True, True]
    assert [row["layer3_enabled"] for row in arms] == [False, False, False, False, True]
    assert [row["memory_mode"] for row in arms] == ["cold", "cold", "cold", "adaptive", "adaptive"]
    assert arms[0]["search_algorithms"] == ["structured_random"]
    assert tuple(arms[1]["search_algorithms"]) == DEFAULT_SEARCH_ALGORITHMS


def test_search_pool_cold_start_allocates_each_algorithm_and_emits_valid_dsl():
    fields = get_dsl_fields("us")
    nodes = []
    seen = []
    for index in range(len(DEFAULT_SEARCH_ALGORITHMS)):
        proposal = propose_search_seed(
            family="momentum",
            fields=fields,
            feedback_nodes=nodes,
            algorithms=DEFAULT_SEARCH_ALGORITHMS,
            rng=random.Random(100 + index),
        )
        algorithm = proposal.metadata["search_algorithm"]
        seen.append(algorithm)
        assert validate(proposal.expression, fields) is None
        assert proposal.metadata["search_pool_schema"] == SEARCH_POOL_SCHEMA_VERSION
        nodes.append({
            "id": index + 1,
            "status": "ok",
            "expression": proposal.expression,
            "public_score": 0.1 * index,
            "proposal_meta": proposal.metadata,
        })
    assert tuple(seen) == DEFAULT_SEARCH_ALGORITHMS


def test_q_learning_policy_uses_only_training_safe_node_fields():
    fields = get_dsl_fields("us")
    nodes = [{
        "id": 1,
        "status": "ok",
        "expression": "rank(ts_delta(close, 60)/(delay(close, 60)+1e-9))",
        "public_score": 0.4,
        "proposal_meta": {
            "target_family": "momentum",
            "search_algorithm": "q_learning",
            "search_action": "fresh",
        },
    }]
    proposal = propose_search_seed(
        family="momentum",
        fields=fields,
        feedback_nodes=nodes,
        algorithms=["q_learning"],
        rng=random.Random(8),
    )
    rendered = repr(proposal.metadata).lower()
    assert "holdout" not in rendered
    assert "vault" not in rendered
    assert "rating" not in rendered
    assert validate(proposal.expression, fields) is None


def test_engine_reads_explicit_three_layer_switches():
    engine = Engine()
    engine.task_config = {
        "proposal_mode": "llm",
        "memory_mode": "adaptive",
        "layer1_enabled": True,
        "layer2_enabled": True,
        "layer3_enabled": False,
        "search_algorithms": ["structured_random", "evolutionary"],
    }
    assert engine._layer1_enabled() is True
    assert engine._layer2_enabled() is True
    assert engine._layer3_enabled() is False
    assert engine._search_algorithms() == ("structured_random", "evolutionary")


def test_layer2_transport_failure_circuit_breaks_without_fake_candidates(monkeypatch):
    async def fail_chat(*args, **kwargs):
        raise llm_client.LLMError("openai 402: Insufficient Balance")

    monkeypatch.setattr(miner_agent.llm, "chat", fail_chat)
    request = {
        "request_id": "r1",
        "slot": 0,
        "op": "draft",
        "task": {
            "name": "T1",
            "market": "us",
            "mode": "long_short",
            "direction": 1,
            "direction_policy": "both_train_select",
            "universe_n": 500,
            "horizon": 5,
        },
        "feedback_nodes": [],
        "target_family": "momentum",
        "base_node": None,
        "rng": random.Random(1),
    }
    with pytest.raises(RuntimeError, match="provider 不可用"):
        asyncio.run(miner_agent.propose_batch(
            DEFAULT_MINER_TEMPLATE,
            [request],
            {"name": "test", "api_key": "x", "model": "m", "base_url": "x"},
            fields=get_dsl_fields("us"),
        ))


def test_layer3_transport_failure_circuit_breaks_without_random_template(monkeypatch):
    async def fail_chat(*args, **kwargs):
        raise llm_client.LLMError("openai 402: Insufficient Balance")

    monkeypatch.setattr(meta_agent.llm, "chat", fail_chat)
    with pytest.raises(RuntimeError, match="provider 不可用"):
        asyncio.run(meta_agent.propose_template(
            DEFAULT_MINER_TEMPLATE,
            [],
            {"name": "test", "api_key": "x", "model": "m", "base_url": "x"},
        ))
