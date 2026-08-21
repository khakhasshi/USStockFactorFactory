import random
import unittest
from unittest.mock import AsyncMock

from backend.app.config import get_dsl_fields
from backend.app.dsl.engine import validate
from backend.app.factors.diversity import (
    mechanism_compatible,
    mechanisms_for_market,
)
from backend.app.factors.semantics import audit_expression_semantics
from backend.app.miner.agent import random_expression_for_family
from backend.app.miner.agent import propose
from backend.app.orchestrator import Engine


class RandomTaskControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_random_mode_never_reads_global_provider(self):
        engine = Engine()
        engine.task_config = {"proposal_mode": "random"}

        self.assertIsNone(await engine._provider("inner_provider"))
        self.assertIsNone(await engine._provider("outer_provider"))

    async def test_deliberate_random_provenance_is_not_a_provider_fallback(self):
        expression, _, source, meta = await propose(
            {},
            "draft",
            {
                "market": "ashare",
                "mode": "long_only",
                "direction": 1,
                "universe_n": 500,
                "horizon": 20,
            },
            [],
            None,
            fields=get_dsl_fields("ashare"),
            rng=random.Random(7),
            target_family="size",
            deliberate_random=True,
        )

        self.assertEqual(source, "random")
        self.assertIsNone(validate(expression, get_dsl_fields("ashare")))
        self.assertEqual(meta["random_reason"], "configured_random_mode")
        self.assertEqual(meta["proposal_mode"], "random")
        self.assertNotIn("fallback_reason", meta)
        self.assertNotIn("LLM 不可用", meta["reflection"])

    def test_target_factor_count_is_explicit_and_validated(self):
        engine = Engine()
        engine.task_config = {"target_factor_count": 5}
        self.assertEqual(engine._target_factor_count(), 5)

        engine.task_config = {"target_factor_count": -1}
        with self.assertRaisesRegex(ValueError, "非负整数"):
            engine._target_factor_count()

    def test_proposal_mode_rejects_unknown_values(self):
        engine = Engine()
        engine.task_config = {"proposal_mode": "mixed"}
        with self.assertRaisesRegex(ValueError, "llm 或 random"):
            engine._proposal_mode()

    def test_candidate_evaluation_budget_is_explicit_and_validated(self):
        engine = Engine()
        engine.task_config = {"candidate_evaluation_budget": 250}
        self.assertEqual(engine._candidate_evaluation_budget(), 250)

        engine.task_config = {"candidate_evaluation_budget": -1}
        with self.assertRaisesRegex(ValueError, "非负整数"):
            engine._candidate_evaluation_budget()

    def test_target_mechanisms_are_market_scoped_and_balanced(self):
        engine = Engine()
        engine.task_config = {
            "market": "ashare",
            "target_mechanisms": ["momentum", "reversal"],
        }
        rng = random.Random(7)
        selected = []
        for completed in range(6):
            engine.status["candidate_evaluations"] = completed
            selected.append(
                engine._target_family_for_attempt([], "ashare", rng)
            )
        self.assertEqual(
            selected,
            ["momentum", "reversal", "momentum", "reversal", "momentum", "reversal"],
        )

        engine.task_config["target_mechanisms"] = ["not_a_mechanism"]
        with self.assertRaisesRegex(ValueError, "不可用于当前市场"):
            engine._target_mechanisms()

    async def test_persisted_evaluation_budget_stops_worker_exactly(self):
        engine = Engine()
        engine.task_config = {"candidate_evaluation_budget": 250}
        engine.running = True
        engine._candidate_evaluation_count = AsyncMock(return_value=250)
        engine.log = AsyncMock()

        self.assertTrue(
            await engine._stop_if_evaluation_budget_reached(refresh=True)
        )
        self.assertFalse(engine.running)
        self.assertEqual(engine.status["candidate_evaluations"], 250)
        self.assertEqual(
            engine.status["stop_reason"],
            "candidate_evaluation_budget_reached",
        )
        engine.log.assert_awaited_once()

    def test_random_grammar_is_valid_semantic_and_mechanism_targeted(self):
        for market in ("ashare", "us"):
            fields = get_dsl_fields(market)
            for family in mechanisms_for_market(market):
                expressions = {
                    random_expression_for_family(
                        family,
                        fields,
                        random.Random(seed),
                    )
                    for seed in range(80)
                }
                self.assertGreaterEqual(
                    len(expressions),
                    16,
                    f"{market}/{family} 随机空间过窄",
                )
                for expression in expressions:
                    self.assertIsNone(validate(expression, fields), expression)
                    self.assertFalse(
                        audit_expression_semantics(expression, market)["errors"],
                        expression,
                    )
                    self.assertTrue(
                        mechanism_compatible(expression, family),
                        expression,
                    )

    def test_large_ashare_campaign_dedupe_can_fill_each_target_family(self):
        fields = get_dsl_fields("ashare")
        for family in (
            "momentum",
            "reversal",
            "volatility",
            "size",
            "volume_price_interaction",
        ):
            rng = random.Random(20260810)
            expressions: set[str] = set()
            for _ in range(1_000):
                expression = random_expression_for_family(family, fields, rng)
                retries = 0
                while expression in expressions and retries < 32:
                    retries += 1
                    expression = random_expression_for_family(family, fields, rng)
                self.assertNotIn(
                    expression,
                    expressions,
                    f"{family} 在 1,000 个候选前已耗尽唯一语法空间",
                )
                expressions.add(expression)


if __name__ == "__main__":
    unittest.main()
