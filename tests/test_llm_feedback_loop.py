import asyncio
import random
import unittest

from backend.app.api.routes import EngineStartReq
from backend.app.config import (
    DEFAULT_MINER_TEMPLATE,
    EVALUATION_PROTOCOL_VERSION,
)
from backend.app.feedback import (
    aggregate_feedback,
    build_feedback_envelope,
    build_inner_feedback_context,
    combine_seed_feedback,
    compare_feedback_reports,
    ensure_training_safe,
    feedback_priority,
)
from backend.app.meta import agent as meta_agent
from backend.app.miner import agent as miner_agent
from backend.app.orchestrator import Engine, _one_sided_score_test


def _public_metrics(
    *,
    score: float = 1.4,
    passed: bool = False,
    failure: str = "训练层 HAC 显著性不足",
) -> dict:
    return {
        "ic_mean": 0.03,
        "icir": 1.8,
        "coverage": 0.92,
        "daily_turnover": 0.12,
        "net": {"sharpe": 1.3, "ann_return": 0.11},
        "active": {"sharpe": 1.1, "ann_return": 0.09},
        "return_confidence": {
            "hac_t_stat": 2.1,
            "sharpe_lcb": 0.6,
            "ann_return_lcb": 0.04,
        },
        "era_consistency": 0.72,
        "profitable_era_rate": 0.66,
        "monotonicity": 0.55,
        "cost_cushion_multiple": 2.4,
        "cost_stress": [{"sharpe": 0.4}],
        "discovery": {
            "score": score,
            "learning_score": score,
            "gate_score": 0.3,
            "score_semantics": "continuous_failure_margin_v4.2",
            "passed": passed,
            "selected_direction": -1,
            "preferred_direction": 1,
            "direction_policy": "both_train_select",
            "direction_selection": {
                "policy": "both_train_select",
                "selection_scope": "training_safe_discovery_only",
                "preferred_direction": 1,
                "selected_direction": -1,
                "trials_multiplier": 2,
                "candidates": {
                    "+1": {
                        "direction": 1,
                        "learning_score": 0.4,
                        "gate_score": 0.0,
                        "passed": False,
                    },
                    "-1": {
                        "direction": -1,
                        "learning_score": score,
                        "gate_score": 0.3,
                        "passed": passed,
                    },
                },
            },
            "components": {
                "predictive": 0.70,
                "portfolio_lcb": 0.40,
                "selection_confidence": 0.35,
                "stability": 0.68,
                "generalization": 0.75,
                "monotonicity": 0.55,
                "cost_survival": 0.60,
                "implementability": 0.73,
            },
            # Conservative values differ deliberately from the optimistic
            # public layer so the contract test can prove which values win.
            "effective_metrics": {
                "coverage": 0.81,
                "icir": 0.74,
                "portfolio_sharpe": 0.52,
                "era_consistency": 0.61,
                "profitable_era_rate": 0.58,
                "monotonicity": 0.31,
                "daily_turnover": 0.19,
                "worst_stress_sharpe": 0.08,
                "return_hac_t": 1.18,
                "sharpe_lcb": 0.12,
                "ann_return_lcb": 0.01,
                "cost_cushion_multiple": 1.3,
                "hac_p_value": 0.12,
            },
            "selection_evidence": {
                "hurdle_t": 3.72,
                "worst_training_return_hac_t": 1.18,
            },
            "failure_reasons": [] if passed else [failure],
        },
    }


def _envelope(
    node_id: int,
    *,
    expression: str | None = None,
    score: float = 1.4,
    passed: bool = False,
    failure: str = "训练层 HAC 显著性不足",
    protocol: str = EVALUATION_PROTOCOL_VERSION,
    source: str = "llm",
) -> dict:
    return build_feedback_envelope(
        node_id=node_id,
        parent_id=None,
        task_name="T1_liquid500_5d",
        expression=expression or f"-rank(ts_delta(close, {20 + node_id}))",
        hypothesis="中期反转在交易成本后仍有横截面收益",
        source=source,
        status="ok",
        error=None,
        public_score=score,
        public_metrics=_public_metrics(
            score=score,
            passed=passed,
            failure=failure,
        ),
        evaluation_protocol=protocol,
        market="us",
        portfolio_mode="long_short",
        direction=1,
        proposal_meta={
            "reflection": "上一轮过度追逐 ICIR，本轮降低换手",
            "targeted_failures": [failure],
        },
    )


class FeedbackContractTests(unittest.TestCase):
    def test_envelope_uses_conservative_effective_metrics(self):
        row = _envelope(1)
        self.assertEqual(row["protocol_version"], EVALUATION_PROTOCOL_VERSION)
        self.assertEqual(row["metrics"]["icir"], 0.74)
        self.assertEqual(row["metrics"]["portfolio_sharpe"], 0.52)
        self.assertEqual(row["metrics"]["daily_turnover"], 0.19)
        self.assertEqual(row["metrics"]["selection_hurdle_t"], 3.72)
        self.assertEqual(row["direction"], -1)
        self.assertEqual(row["outcome"]["learning_score"], 1.4)
        self.assertEqual(row["outcome"]["gate_score"], 0.3)
        self.assertEqual(
            set(row["direction_selection"]["candidates"]),
            {"+1", "-1"},
        )
        self.assertIn("HAC 显著性不足", row["failure_reasons"][0])
        self.assertTrue(row["improvement_targets"])
        self.assertEqual(len(row["feedback_fingerprint"]), 16)

    def test_sealed_layers_are_rejected_by_key_and_text(self):
        with self.assertRaises(ValueError):
            ensure_training_safe({"holdout": {"sharpe": 9.9}})
        with self.assertRaises(ValueError):
            ensure_training_safe({"note": "read FACTOR_VAULT next"})

    def test_inner_context_excludes_legacy_protocol_and_keeps_failures(self):
        current = _envelope(2, failure="实际持仓换手超过任务上限")
        legacy = {
            **_envelope(3),
            "protocol_version": "legacy_unoriented",
            "expression": "LEGACY_SENTINEL",
        }
        context, snapshot = build_inner_feedback_context(
            [legacy, current],
            DEFAULT_MINER_TEMPLATE,
        )
        self.assertIn("实际持仓换手超过任务上限", context)
        self.assertIn(current["expression"], context)
        self.assertIn("HAC_t=", context)
        self.assertIn("cost_cushion=", context)
        self.assertIn("components:", context)
        self.assertIn("同时测试 +1/-1", context)
        self.assertIn("direction=-1", context)
        self.assertIn("learning_score=", context)
        self.assertIn("hard_gate_score=", context)
        self.assertNotIn("LEGACY_SENTINEL", context)
        self.assertEqual(
            snapshot["protocol_version"],
            EVALUATION_PROTOCOL_VERSION,
        )

    def test_context_scoring_weights_are_live_but_not_authoritative(self):
        high_predictive = _envelope(4)
        high_predictive["components"].update({
            "predictive": 0.95,
            "stability": 0.20,
        })
        high_predictive["metrics"]["daily_turnover"] = 0.50
        low_turnover = _envelope(5)
        low_turnover["components"].update({
            "predictive": 0.20,
            "stability": 0.20,
        })
        low_turnover["metrics"]["daily_turnover"] = 0.01

        predictive_template = {
            "scoring_weights": {
                "icir_weight": 0.90,
                "consistency_weight": 0.05,
                "turnover_weight": 0.05,
            }
        }
        turnover_template = {
            "scoring_weights": {
                "icir_weight": 0.05,
                "consistency_weight": 0.05,
                "turnover_weight": 0.90,
            }
        }
        self.assertGreater(
            feedback_priority(high_predictive, predictive_template),
            feedback_priority(low_turnover, predictive_template),
        )
        self.assertGreater(
            feedback_priority(low_turnover, turnover_template),
            feedback_priority(high_predictive, turnover_template),
        )
        # Context priority must not mutate the evaluator's persisted score.
        self.assertEqual(high_predictive["outcome"]["score"], 1.4)

    def test_cross_seed_report_and_comparison_are_auditable(self):
        seed_a = {
            "seed": 0,
            "score": 1.2,
            "task_best_scores": {"T1_liquid500_5d": 1.2},
            "envelopes": [_envelope(6, score=1.2)],
        }
        seed_b = {
            "seed": 1,
            "score": 1.6,
            "task_best_scores": {"T1_liquid500_5d": 1.6},
            "envelopes": [_envelope(7, score=1.6, passed=True)],
        }
        candidate = combine_seed_feedback([seed_a, seed_b])
        incumbent = aggregate_feedback([_envelope(8, score=0.8)])
        incumbent["seed_score_mean"] = 0.8
        comparison = compare_feedback_reports(candidate, incumbent)
        self.assertEqual(candidate["attempts"], 2)
        self.assertAlmostEqual(candidate["seed_score_mean"], 1.4)
        self.assertEqual(len(candidate["seeds"]), 2)
        self.assertAlmostEqual(
            comparison["deltas"]["seed_score_mean"],
            0.6,
        )
        self.assertTrue(comparison["comparison_fingerprint"])

    def test_seed_context_merge_deduplicates_without_cross_seed_state(self):
        baseline = [{"id": 10, "seed": None}, {"id": 11, "seed": 3}]
        current_seed = [{"id": 12, "seed": 7}, {"id": 10, "seed": None}]
        merged = Engine._merge_feedback_nodes(baseline, current_seed)
        self.assertEqual([row["id"] for row in merged], [12, 10, 11])


class InnerOuterAgentTests(unittest.TestCase):
    def test_inner_llm_receives_failures_templates_and_market_mode(self):
        captured = {}
        original = miner_agent.llm.chat

        async def fake_chat(
            provider,
            system,
            user,
            temperature,
            **kwargs,
        ):
            captured.update({
                "system": system,
                "user": user,
                "trace": kwargs.get("trace"),
            })
            return (
                '{"expression":"-rank(ts_delta(close, 20))",'
                '"hypothesis":"slow reversal",'
                '"reflection":"reduce turnover after failed stress cost",'
                '"targeted_failures":["实际持仓换手超过任务上限"],'
                '"expected_effect":"improve implementability"}'
            )

        miner_agent.llm.chat = fake_chat
        try:
            result = asyncio.run(miner_agent.propose(
                DEFAULT_MINER_TEMPLATE,
                "draft",
                {
                    "name": "T1_liquid500_5d",
                    "market": "ashare",
                    "mode": "long_only",
                    "direction": 1,
                    "universe_n": 500,
                    "horizon": 5,
                },
                [{
                    "id": 20,
                    "status": "ok",
                    "public_score": 1.1,
                    "expression": "-rank(ts_delta(close, 5))",
                    "public_metrics": _public_metrics(
                        failure="实际持仓换手超过任务上限"
                    ),
                    "feedback_summary": _envelope(
                        20,
                        failure="实际持仓换手超过任务上限",
                    ),
                }],
                {"name": "mock", "model": "mock-model"},
                fields=["open", "close", "vol", "amount"],
                trace_context={"experiment_id": 5},
            ))
        finally:
            miner_agent.llm.chat = original

        self.assertEqual(result[2], "llm")
        self.assertEqual(
            result[3]["feedback_protocol"],
            EVALUATION_PROTOCOL_VERSION,
        )
        self.assertIn("只能做多", captured["system"])
        self.assertIn("同时评价 +1", captured["system"])
        self.assertIn("实际持仓换手超过任务上限", captured["user"])
        self.assertIn("DSL 结构样例", captured["user"])
        self.assertEqual(captured["trace"]["role"], "inner")
        self.assertNotIn("META_HOLDOUT", captured["system"] + captured["user"])
        self.assertNotIn("FACTOR_VAULT", captured["system"] + captured["user"])

    def test_outer_llm_filters_legacy_history_and_carries_reflection(self):
        captured = {}
        original = meta_agent.llm.chat

        async def fake_chat(
            provider,
            system,
            user,
            temperature,
            **kwargs,
        ):
            captured["user"] = user
            captured["trace"] = kwargs.get("trace")
            return (
                '{"reflection":{"diagnosis":"pass rate is weak",'
                '"lessons_applied":"change one field",'
                '"evidence_used":"v2 failure counts",'
                '"hypothesis":"lower turnover examples should help"},'
                '"changed_fields":[],"template":{},'
                '"note":"evidence insufficient; keep incumbent"}'
            )

        history = [
            {
                "version_no": 1,
                "evaluation_protocol": "legacy_unoriented",
                "template_note": "LEGACY_SENTINEL",
            },
            {
                "version_no": 2,
                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                "status": "rejected",
                "meta_score": 0.8,
                "template_note": "single-variable turnover test",
                "feedback_summary": {
                    "attempts": 10,
                    "pass_rate": 0.1,
                    "failure_reason_counts": {
                        "实际持仓换手超过任务上限": 7
                    },
                    "feedback_fingerprint": "feedback-v2",
                },
                "reflection": {
                    "outcome": {"hypothesis_result": "refuted"}
                },
            },
        ]
        meta_agent.llm.chat = fake_chat
        try:
            template, note, source, reflection = asyncio.run(
                meta_agent.propose_template(
                    DEFAULT_MINER_TEMPLATE,
                    history,
                    {"name": "mock", "model": "mock-model"},
                    market="us",
                    portfolio_mode="long_short",
                    direction=1,
                    trace_context={"experiment_id": 6},
                )
            )
        finally:
            meta_agent.llm.chat = original

        self.assertEqual(template, DEFAULT_MINER_TEMPLATE)
        self.assertEqual(source, "llm")
        self.assertIn("evidence insufficient", note)
        self.assertEqual(reflection["diagnosis"], ["pass rate is weak"])
        self.assertNotIn("LEGACY_SENTINEL", captured["user"])
        self.assertIn("实际持仓换手超过任务上限", captured["user"])
        self.assertEqual(captured["trace"]["role"], "outer")

    def test_post_decision_reflection_is_structured(self):
        original = meta_agent.llm.chat

        async def fake_chat(*args, **kwargs):
            return (
                '{"hypothesis_result":"refuted",'
                '"lessons":{"observation":"score fell",'
                '"interpretation":"template regressed",'
                '"action":"revert"},'
                '"avoid_patterns":"broad multi-field edits",'
                '"next_experiment":"change context_policy only",'
                '"stop_condition":"two repeated failures"}'
            )

        meta_agent.llm.chat = fake_chat
        try:
            result, source = asyncio.run(meta_agent.reflect_on_outcome(
                proposal_reflection={"hypothesis": "more context helps"},
                candidate_report={
                    "attempts": 2,
                    "feedback_fingerprint": "candidate",
                },
                incumbent_report={
                    "attempts": 2,
                    "feedback_fingerprint": "incumbent",
                },
                comparison={
                    "deltas": {"seed_score_mean": -0.2},
                    "candidate_failures": {"high turnover": 2},
                    "comparison_fingerprint": "compare",
                },
                accepted=False,
                p_value=0.8,
                provider={"name": "mock"},
            ))
        finally:
            meta_agent.llm.chat = original

        self.assertEqual(source, "llm")
        self.assertEqual(result["hypothesis_result"], "refuted")
        self.assertEqual(result["lessons"][0]["action"], "revert")
        self.assertEqual(
            result["avoid_patterns"],
            ["broad multi-field edits"],
        )

    def test_random_fallback_is_reproducible_with_explicit_rng(self):
        first = miner_agent.random_expression(
            DEFAULT_MINER_TEMPLATE["dsl_exploration_templates"],
            ["open", "high", "low", "close", "vol", "amount"],
            random.Random(42),
        )
        second = miner_agent.random_expression(
            DEFAULT_MINER_TEMPLATE["dsl_exploration_templates"],
            ["open", "high", "low", "close", "vol", "amount"],
            random.Random(42),
        )
        self.assertEqual(first, second)

    def test_incomplete_llm_reflection_is_rejected_to_random_fallback(self):
        original = miner_agent.llm.chat

        async def incomplete(*args, **kwargs):
            return (
                '{"expression":"rank(ts_delta(close, 20))",'
                '"hypothesis":"momentum"}'
            )

        miner_agent.llm.chat = incomplete
        try:
            result = asyncio.run(miner_agent.propose(
                DEFAULT_MINER_TEMPLATE,
                "draft",
                {
                    "name": "T1",
                    "market": "us",
                    "mode": "long_short",
                    "direction": 1,
                    "universe_n": 500,
                    "horizon": 5,
                },
                [],
                {"name": "mock"},
                fields=["open", "high", "low", "close", "vol", "amount"],
                rng=random.Random(7),
            ))
        finally:
            miner_agent.llm.chat = original
        self.assertEqual(result[2], "random")
        self.assertIn("reflection", result[3]["fallback_reason"])

    def test_missing_expected_effect_is_normalized_not_randomized(self):
        original_chat = miner_agent.llm.chat
        original_mark = miner_agent.llm.mark_validation
        validation = {}

        async def missing_noncritical_field(*args, **kwargs):
            return (
                '{"expression":"rank(ts_delta(close, 20))",'
                '"hypothesis":"medium-term momentum",'
                '"reflection":"retain a simple mechanism after noisy failures",'
                '"targeted_failures":["cost stress is weak"]}'
            )

        async def capture_validation(*args, **kwargs):
            validation.update(kwargs)

        miner_agent.llm.chat = missing_noncritical_field
        miner_agent.llm.mark_validation = capture_validation
        try:
            result = asyncio.run(miner_agent.propose(
                DEFAULT_MINER_TEMPLATE,
                "draft",
                {
                    "name": "T1",
                    "market": "us",
                    "mode": "long_short",
                    "direction": 1,
                    "universe_n": 500,
                    "horizon": 5,
                },
                [{
                    "id": 21,
                    "status": "ok",
                    "public_score": 0.5,
                    "expression": "-rank(ts_delta(close, 5))",
                    "public_metrics": _public_metrics(),
                    "feedback_summary": _envelope(21),
                }],
                {"name": "mock"},
                fields=["open", "high", "low", "close", "vol", "amount"],
                rng=random.Random(7),
            ))
        finally:
            miner_agent.llm.chat = original_chat
            miner_agent.llm.mark_validation = original_mark

        self.assertEqual(result[2], "llm")
        self.assertIn("cost stress is weak", result[3]["expected_effect"])
        self.assertEqual(
            result[3]["semantic_normalizations"],
            ["expected_effect_derived_from_targeted_failures"],
        )
        self.assertTrue(validation["accepted"])
        self.assertEqual(
            validation["trace_meta_updates"]["semantic_normalizations"],
            ["expected_effect_derived_from_targeted_failures"],
        )

    def test_outer_test_and_default_start_mode_are_explicit(self):
        self.assertEqual(EngineStartReq().mode, "v2")
        legacy_start = asyncio.run(Engine().start(mode="v1"))
        self.assertFalse(legacy_start["ok"])
        self.assertIn("历史只读", legacy_start["msg"])
        strong = _one_sided_score_test(
            [1.5, 1.6, 1.7],
            [0.7, 0.8, 0.9],
            reference_mean=0.8,
        )
        weak = _one_sided_score_test(
            [0.7],
            [],
            reference_mean=0.8,
        )
        self.assertEqual(strong["type"], "welch_one_sided_t")
        self.assertLess(strong["p_value"], 0.10)
        known = _one_sided_score_test(
            [0.0, 2.0],
            [],
            reference_mean=0.0,
        )
        self.assertEqual(
            known["type"],
            "candidate_vs_frozen_incumbent_one_sided_t",
        )
        self.assertAlmostEqual(known["t_stat"], 1.0, places=6)
        self.assertAlmostEqual(known["degrees_of_freedom"], 1.0, places=6)
        self.assertAlmostEqual(known["p_value"], 0.25, places=6)
        self.assertEqual(weak["type"], "insufficient_seeds")
        self.assertEqual(weak["p_value"], 1.0)


if __name__ == "__main__":
    unittest.main()
