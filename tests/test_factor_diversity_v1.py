import random
import unittest

from backend.app.factors.diversity import (
    diversity_adjusted_score,
    diversity_snapshot,
    mechanism_compatible,
    select_target_mechanism,
)
from backend.app.factors.return_path import (
    build_return_path_signature,
    return_path_correlation,
)
from backend.app.factors.semantics import (
    audit_expression_semantics,
    field_contract,
)


def _row(family: str, expression: str, score: float = 1.0) -> dict:
    return {
        "expression": expression,
        "hypothesis": family,
        "proposal": {"declared_family": family},
        "outcome": {
            "status": "ok",
            "learning_score": score,
            "passed": score >= 1.0,
        },
    }


class FactorDiversityV1Tests(unittest.TestCase):
    def test_adjusted_price_and_raw_amount_requires_normalisation(self):
        unsafe = audit_expression_semantics(
            "rank((high-low)/(amount+1e-9))",
            "us",
        )
        safe = audit_expression_semantics(
            "rank(((high-low)/(close+1e-9))/(amount+1e-9))",
            "us",
        )
        self.assertEqual(unsafe["status"], "error")
        self.assertIn("mixed_basis", unsafe["errors"][0])
        self.assertEqual(safe["status"], "ok")
        self.assertEqual(
            field_contract("us")["amount"]["provenance"],
            "raw_close_times_volume_proxy",
        )

    def test_target_scheduler_prefers_uncovered_mechanisms(self):
        rows = [
            _row("momentum", "rank(ts_delta(close, 20))"),
            _row("momentum", "rank(ts_delta(close, 60))"),
        ]
        target = select_target_mechanism(rows, "us", random.Random(3))
        self.assertNotEqual(target, "momentum")
        self.assertTrue(
            mechanism_compatible(
                "rank(ts_std(ts_delta(close,1)/(delay(close,1)+1e-9),40))",
                "volatility",
            )
        )

    def test_outer_score_requires_multiple_mechanism_representatives(self):
        single = [_row("momentum", "rank(ts_delta(close,20))", 2.0)]
        diversified = [
            _row("momentum", "rank(ts_delta(close,20))", 2.0),
            _row("reversal", "-rank(ts_delta(close,5))", 2.0),
            _row("volatility", "-rank(ts_std(close,40))", 2.0),
            _row("liquidity", "-rank(ts_mean(amount,40))", 2.0),
        ]
        single_score, _ = diversity_adjusted_score(
            {"T1": 2.0}, single, "us"
        )
        diverse_score, detail = diversity_adjusted_score(
            {"T1": 2.0}, diversified, "us"
        )
        self.assertGreater(diverse_score, single_score)
        self.assertEqual(detail["distinct_mechanisms"], 4)
        self.assertEqual(diversity_snapshot(diversified, "us")["mechanism_hhi"], 0.25)

    def test_return_path_signature_detects_economic_duplicate(self):
        left = build_return_path_signature([index / 10_000 for index in range(200)])
        same = build_return_path_signature([index / 5_000 for index in range(200)])
        opposite = build_return_path_signature([-index / 10_000 for index in range(200)])
        self.assertGreater(return_path_correlation(left, same), 0.99)
        self.assertLess(return_path_correlation(left, opposite), -0.99)


if __name__ == "__main__":
    unittest.main()
