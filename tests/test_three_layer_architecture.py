import random
import asyncio

import pytest

from backend.app.api.routes import _three_layer_arm_configs
from backend.app.config import DEFAULT_MINER_TEMPLATE, get_dsl_fields
from backend.app.dsl.engine import validate
from backend.app.llm import client as llm_client
from backend.app.meta import agent as meta_agent
from backend.app.miner import agent as miner_agent
from backend.app import search_pool as search_pool_module
from backend.app.orchestrator import Engine
from backend.app.research_architecture import resolve_research_architecture
from backend.app.search_pool import (
    ALGORITHM_GROUPS,
    DEFAULT_SEARCH_ALGORITHMS,
    RESET_EXPLORATION_ALGORITHMS,
    SEARCH_GROUP_WEIGHTS,
    SEARCH_POOL_SCHEMA_VERSION,
    _quota_ucb,
    propose_search_seed,
    search_health,
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
            # Allocation is intentionally based on distinct evaluated ASTs.
            "expression": f"rank(ts_mean(close, {index + 2}))",
            "public_score": 0.1 * index,
            "proposal_meta": proposal.metadata,
        })
    # V2 cold-start is quota-aware, so initialisation order follows group
    # deficits rather than the declaration order. Every configured arm must
    # still receive one auditable observation.
    assert set(seen) == set(DEFAULT_SEARCH_ALGORITHMS)


def test_qlib_task_integration_is_persisted_and_changes_l1_candidate_source():
    architecture = resolve_research_architecture({
        "architecture_template": "algorithm_pool_only",
        "qlib_integration": {
            "enabled": True,
            "alpha158_prior_enabled": True,
            "gbdt_candidate_pool_enabled": True,
        },
    })
    assert "qlib_alpha158_prior" in architecture["search_algorithms"]
    assert "qlib_joint_residual_distill" in architecture["search_algorithms"]
    assert architecture["qlib_integration"]["effective"] is True
    assert architecture["qlib_integration"]["evaluation_authority"] == (
        "factorfactory_v4_2_v4_3"
    )

    proposal = propose_search_seed(
        family="momentum",
        fields=get_dsl_fields("us"),
        feedback_nodes=[],
        algorithms=["qlib_alpha158_prior"],
        rng=random.Random(20260823),
    )
    assert validate(proposal.expression, get_dsl_fields("us")) is None
    assert proposal.metadata["qlib_exact_seed"] is True
    assert proposal.metadata["qlib_feature"]
    assert proposal.metadata["qlib_upstream_commit"]


def test_qlib_gbdt_pool_is_real_input_not_ui_only():
    proposal = propose_search_seed(
        family="price_relationship",
        fields=get_dsl_fields("us"),
        feedback_nodes=[],
        algorithms=["gbdt_residual_distill"],
        rng=random.Random(77),
        qlib_candidate_pool=True,
    )
    assert validate(proposal.expression, get_dsl_fields("us")) is None
    assert proposal.metadata["qlib_candidate_pool"] is True
    assert proposal.metadata["qlib_candidate_count"] > 0


def test_qlib_joint_distillation_arm_uses_oof_artifact_candidates():
    proposal = propose_search_seed(
        family="momentum",
        fields=get_dsl_fields("us"),
        feedback_nodes=[],
        algorithms=["qlib_joint_residual_distill"],
        rng=random.Random(91),
        qlib_joint_candidates=[{
            "expression": "rank(returns(close,20))+rank(ts_std(close,20)/close)",
            "components": ["ROC20", "STD20"],
            "distillation_kind": "equal_rank_blend",
        }],
    )
    assert validate(proposal.expression, get_dsl_fields("us")) is None
    assert proposal.metadata["joint_model_ready"] is True
    assert proposal.metadata["distillation_components"] == ["ROC20", "STD20"]


def test_adaptive_budget_preserves_exploration_floor_after_warmup():
    nodes = [
        _search_node(
            index + 1,
            f"rank(ts_mean(close,{index + 2}))",
            DEFAULT_SEARCH_ALGORITHMS[index % len(DEFAULT_SEARCH_ALGORITHMS)],
            gate_score=(index % 9) / 100,
        )
        for index in range(80)
    ]
    _, policy = _quota_ucb(
        DEFAULT_SEARCH_ALGORITHMS,
        nodes,
        random.Random(99),
        adaptive=True,
    )
    assert policy["adaptive_efficiency_enabled"] is True
    assert all(
        row["target_share"] >= 0.05
        for row in policy["group_stats"].values()
    )


def test_discovery_fabric_converges_to_declared_group_quotas():
    nodes = []
    counts = {group: 0 for group in SEARCH_GROUP_WEIGHTS}
    for index in range(200):
        algorithm, _ = _quota_ucb(
            DEFAULT_SEARCH_ALGORITHMS, nodes, random.Random(1000 + index)
        )
        counts[ALGORITHM_GROUPS[algorithm]] += 1
        nodes.append({
            "id": index + 1,
            "status": "ok",
            "expression": f"rank(ts_mean(close, {index + 2}))",
            "public_score": (index % 7) / 10,
            "proposal_meta": {"search_algorithm": algorithm},
        })
    for group, target in SEARCH_GROUP_WEIGHTS.items():
        assert abs(counts[group] / 200 - target) <= 0.015


def _search_node(
    node_id,
    expression,
    algorithm,
    *,
    gate_score=0.0,
    learning_score=0.5,
    status="ok",
    duplicate=False,
    extra_meta=None,
):
    return {
        "id": node_id,
        "status": status,
        "expression": expression,
        "public_score": learning_score,
        "public_metrics": {
            "discovery": {
                "learning_score": learning_score,
                "gate_score": gate_score,
                "gate_components": {
                    "predictive": min(1.0, gate_score),
                    "portfolio_lcb": min(1.0, gate_score),
                },
                "passed": False,
            }
        },
        "proposal_meta": {
            "search_algorithm": algorithm,
            "campaign_exact_duplicate": duplicate,
            **(extra_meta or {}),
        },
    }


def test_quota_counts_unique_evaluated_ast_and_penalizes_duplicate_attempts():
    nodes = [
        _search_node(1, "rank(close)", "evolutionary", gate_score=0.01),
        *[
            _search_node(
                node_id,
                "(rank(close))",
                "evolutionary",
                gate_score=0.01,
                duplicate=True,
            )
            for node_id in range(2, 12)
        ],
        _search_node(12, "rank(open)", "tpe_smac", gate_score=0.02),
    ]
    _, policy = _quota_ucb(
        ("evolutionary", "tpe_smac"), nodes, random.Random(7)
    )
    assert policy["quota_unit"] == "unique_evaluated_normalized_ast"
    assert policy["algorithm_stats"]["evolutionary"]["unique_evaluations"] == 1
    assert policy["algorithm_stats"]["evolutionary"]["proposal_attempts"] == 11
    assert policy["algorithm_stats"]["evolutionary"]["duplicate_attempts"] == 10


def test_quota_accounts_for_hidden_duplicate_resamples():
    node = _search_node(
        1,
        "rank(close)",
        "grammar_enumerative",
        extra_meta={
            "campaign_retry_algorithms": [
                "residual_oof_beam",
                "residual_oof_beam",
                "residual_oof_beam",
            ]
        },
    )
    _, policy = _quota_ucb(
        ("grammar_enumerative", "residual_oof_beam"),
        [node],
        random.Random(12),
    )
    stats = policy["algorithm_stats"]["residual_oof_beam"]
    assert stats["proposal_attempts"] == 3
    assert stats["hidden_duplicate_resamples"] == 3
    assert stats["duplicate_rate"] == 1.0


def test_gate_progress_dominates_learning_score_in_algorithm_reward():
    nodes = []
    for index in range(20):
        nodes.append(_search_node(
            index + 1,
            f"rank(ts_mean(close, {index + 2}))",
            "evolutionary",
            gate_score=0.001,
            learning_score=1.4,
        ))
        nodes.append(_search_node(
            index + 101,
            f"rank(ts_mean(open, {index + 2}))",
            "tpe_smac",
            gate_score=0.4 + index / 1000,
            learning_score=0.6,
        ))
    selected, policy = _quota_ucb(
        ("evolutionary", "tpe_smac"), nodes, random.Random(3)
    )
    assert selected == "tpe_smac"
    assert policy["reward_semantics"] == (
        "gate_quality_plus_incremental_gate_record"
    )


def test_search_health_triggers_reset_and_forces_diversity_algorithms():
    nodes = [
        _search_node(
            index,
            f"rank(ts_mean(close, {index + 2}))",
            "mcts_puct",
            gate_score=0.01,
        )
        for index in range(1, 61)
    ]
    nodes.extend(
        _search_node(
            index,
            "rank(close)",
            "mcts_puct",
            status="rejected",
            duplicate=True,
            extra_meta={
                "pre_evaluation_rejection": "duplicate_normalized_ast"
            },
        )
        for index in range(61, 101)
    )
    health = search_health(nodes)
    assert health["state"] == "duplicate_reset_triggered"
    assert health["recent_duplicate_rate"] == 0.4
    proposal = propose_search_seed(
        family="momentum",
        fields=get_dsl_fields("us"),
        feedback_nodes=nodes,
        allocation_history=nodes,
        algorithms=DEFAULT_SEARCH_ALGORITHMS,
        rng=random.Random(12),
    )
    assert proposal.metadata["search_reset_triggered"] is True
    assert proposal.metadata["search_algorithm"] in RESET_EXPLORATION_ALGORITHMS
    assert set(proposal.metadata["search_policy"]["active_algorithms"]) == (
        set(DEFAULT_SEARCH_ALGORITHMS) & set(RESET_EXPLORATION_ALGORITHMS)
    )


def test_search_health_keeps_recovery_epoch_until_unique_quota_is_rebuilt():
    nodes = [
        _search_node(
            1,
            "rank(close)",
            "grammar_enumerative",
            gate_score=0.02,
            extra_meta={
                "search_reset_triggered": True,
                "search_epoch": 1,
                "search_recovery_kind": "duplicate_collapse",
                "search_reset_reasons": ["recent_duplicate_rate"],
            },
        )
    ]
    nodes.extend(
        _search_node(
            index,
            f"rank(ts_mean(open, {index + 2}))",
            "novelty_search",
            gate_score=0.02,
            extra_meta={"search_epoch": 1},
        )
        for index in range(2, 20)
    )
    health = search_health(nodes)
    assert health["state"] == "duplicate_recovery"
    assert health["search_epoch"] == 1
    assert health["epoch_unique_evaluations"] == 19


def test_search_health_bounds_failed_recovery_by_attempt_count():
    nodes = [
        _search_node(
            1,
            "rank(close)",
            "grammar_enumerative",
            extra_meta={
                "search_reset_triggered": True,
                "search_epoch": 1,
                "search_recovery_kind": "duplicate_collapse",
                "search_reset_reasons": ["recent_duplicate_rate"],
            },
        )
    ]
    nodes.extend(
        _search_node(
            index,
            "rank(close)",
            "grammar_enumerative",
            status="rejected",
            duplicate=True,
            extra_meta={
                "search_epoch": 1,
                "pre_evaluation_rejection": "duplicate_normalized_ast",
            },
        )
        for index in range(2, 33)
    )
    health = search_health(nodes, {"recovery_attempt_patience": 30})
    assert health["state"] == "duplicate_space_exhausted"
    assert health["search_reset_triggered"] is True
    assert "recovery_attempts_exhausted" in health["reset_reasons"]
    assert health["epoch_attempts"] == 32


def test_score_stagnation_keeps_residual_ml_and_qlib_arms_live():
    algorithms = (*DEFAULT_SEARCH_ALGORITHMS, "qlib_joint_residual_distill")
    nodes = [
        _search_node(
            index,
            f"rank(ts_mean(close, {index + 2}))",
            algorithms[index % len(algorithms)],
            gate_score=0.02,
        )
        for index in range(1, 153)
    ]
    health = search_health(nodes)
    assert health["state"] == "stagnation_rebalance"
    assert health["search_reset_triggered"] is False
    assert health["reset_reasons"] == []
    assert health["stagnation_reasons"] == ["no_new_gate_record"]
    proposal = propose_search_seed(
        family="momentum",
        fields=get_dsl_fields("us"),
        feedback_nodes=nodes,
        allocation_history=nodes,
        algorithms=algorithms,
        rng=random.Random(312),
        avoid_algorithms={
            "residual_oof_beam",
            "gbdt_residual_distill",
            "qlib_joint_residual_distill",
        },
    )
    policy = proposal.metadata["search_policy"]
    # Stagnation never disables these arms globally. A bounded duplicate retry
    # may still defer them for this one slot so an exhausted finite pool cannot
    # consume another 96 hidden proposals.
    active = set(policy["health_active_algorithms"])
    assert {
        "residual_oof_beam",
        "gbdt_residual_distill",
        "qlib_joint_residual_distill",
    } <= active
    assert not set(policy["active_algorithms"]).intersection({
        "residual_oof_beam",
        "gbdt_residual_distill",
        "qlib_joint_residual_distill",
    })


def test_residual_arm_consumes_actual_oof_candidate_metadata():
    proposal = propose_search_seed(
        family="momentum",
        fields=get_dsl_fields("us"),
        feedback_nodes=[],
        algorithms=["residual_oof_beam"],
        rng=random.Random(91),
        residual_oof_candidates=[{
            "expression": "rank(ts_mean(close, 20))",
            "family": "momentum",
            "score": 0.42,
            "residual_rank_ic": 0.08,
            "incremental_oof_ic": 0.015,
            "independence": 0.74,
            "stability": 0.81,
            "normalized_hash": "candidate-1",
        }],
    )
    assert proposal.expression == "rank(ts_mean(close, 20))"
    assert proposal.metadata["residual_oof_ready"] is True
    assert proposal.metadata["residual_scope"] == (
        "sample_level_cross_sectional_time_ordered_oof"
    )


def test_gbdt_distillation_trains_on_actual_oof_scores(monkeypatch):
    monkeypatch.setattr(
        search_pool_module,
        "_fit_gbdt",
        lambda rows, targets: (
            lambda row: sum(row.values()) / max(1, len(row)),
            "test_gbdt",
            3,
            None,
        ),
    )
    residual_rows = [
        {
            "expression": f"rank(ts_mean(close, {index + 2}))",
            "score": index / 20,
        }
        for index in range(8)
    ]
    proposal = propose_search_seed(
        family="momentum",
        fields=get_dsl_fields("us"),
        feedback_nodes=[],
        algorithms=["gbdt_residual_distill"],
        rng=random.Random(92),
        residual_oof_candidates=residual_rows,
    )
    assert proposal.metadata["training_target"] == (
        "actual_oof_residual_beam_score"
    )
    assert proposal.metadata["training_rows"] == 8
    assert proposal.metadata["exact_oof_completed_before_distillation"] is True


def test_rolling_cohort_restores_missing_structural_share():
    algorithms = DEFAULT_SEARCH_ALGORITHMS
    historical = [
        _search_node(
            index + 1,
            f"rank(ts_mean(close, {index + 2}))",
            algorithm,
            gate_score=0.01,
        )
        for index, algorithm in enumerate(algorithms)
    ]
    recent = [
        _search_node(
            100 + index,
            f"rank(ts_std(close, {index + 3}))",
            "residual_oof_beam",
            gate_score=0.01,
        )
        for index in range(60)
    ]
    proposal = propose_search_seed(
        family="momentum",
        fields=get_dsl_fields("us"),
        feedback_nodes=[*historical, *recent],
        allocation_history=[*historical, *recent],
        algorithms=algorithms,
        rng=random.Random(93),
    )
    policy = proposal.metadata["search_policy"]
    assert policy["allocation_cohort_observations"] == 60
    assert policy["selected_group"] == "structural_search"


def test_declared_targeted_residual_branch_uses_seed_node_and_valid_gate():
    seed_node = _search_node(
        95206,
        "rank(ts_std(((close-low)/(high-low+1e-9)), 210))",
        "residual_oof_beam",
        gate_score=1.03,
    )
    proposal = propose_search_seed(
        family="volatility",
        fields=get_dsl_fields("ashare"),
        feedback_nodes=[seed_node],
        allocation_history=[seed_node],
        algorithms=["residual_oof_beam"],
        rng=random.Random(95206),
        targeted_branch={
            "branch_id": "ashare-node-95206-residual-gate-v1",
            "seed_node_id": 95206,
        },
    )
    assert validate(proposal.expression, get_dsl_fields("ashare")) is None
    assert proposal.metadata["search_algorithm"] == "residual_oof_beam"
    assert proposal.metadata["search_parent_node_id"] == 95206
    assert proposal.metadata["targeted_seed_node_id"] == 95206
    assert proposal.metadata["directed_research"] is True


def test_pure_algorithm_architecture_is_distinct_from_outer_ab():
    engine = Engine()
    engine.task_config = {
        "proposal_mode": "search_pool",
        "layer1_enabled": True,
        "layer2_enabled": False,
        "layer3_enabled": False,
    }
    assert engine._pure_algorithm_architecture() is True
    engine.task_config["layer2_enabled"] = True
    assert engine._pure_algorithm_architecture() is False


def test_collapsed_algorithm_is_temporarily_quarantined():
    nodes = [
        _search_node(
            index,
            "rank(close)",
            "grammar_enumerative",
            status="rejected",
            duplicate=True,
            extra_meta={
                "pre_evaluation_rejection": "duplicate_normalized_ast"
            },
        )
        for index in range(1, 26)
    ]
    proposal = propose_search_seed(
        family="momentum",
        fields=get_dsl_fields("us"),
        feedback_nodes=nodes,
        allocation_history=nodes,
        algorithms=["grammar_enumerative", "map_elites"],
        rng=random.Random(88),
    )
    policy = proposal.metadata["search_policy"]
    assert "grammar_enumerative" in policy["quarantined_algorithms"]
    assert proposal.metadata["search_algorithm"] == "map_elites"
    assert validate(proposal.expression, get_dsl_fields("us")) is None


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
