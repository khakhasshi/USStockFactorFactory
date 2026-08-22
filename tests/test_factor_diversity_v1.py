import math
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
from backend.app.factors.return_source_governance import (
    cluster_training_return_sources,
    return_source_quality,
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

    def test_return_source_governance_is_scope_aware_and_representative_based(self):
        base = build_return_path_signature(
            [math.sin(index / 7) + index / 500 for index in range(240)]
        )
        duplicate = build_return_path_signature(
            [2 * (math.sin(index / 7) + index / 500) for index in range(240)]
        )
        diversifier = build_return_path_signature(
            [-math.sin(index / 7) + index / 500 for index in range(240)]
        )
        rows = [
            {
                "node_id": 1,
                "task_name": "T1",
                "market": "us",
                "portfolio_mode": "long_short",
                "horizon": 5,
                "public_score": 2.0,
                "training_return_path_signature": base,
            },
            {
                "node_id": 2,
                "task_name": "T1",
                "market": "us",
                "portfolio_mode": "long_short",
                "horizon": 5,
                "public_score": 1.0,
                "training_return_path_signature": duplicate,
            },
            {
                "node_id": 3,
                "task_name": "T1",
                "market": "us",
                "portfolio_mode": "long_short",
                "horizon": 5,
                "public_score": 1.5,
                "training_return_path_signature": diversifier,
            },
            {
                "node_id": 4,
                "task_name": "T2",
                "market": "us",
                "portfolio_mode": "long_short",
                "horizon": 20,
                "public_score": 1.0,
                "training_return_path_signature": duplicate,
            },
        ]

        snapshot = cluster_training_return_sources(rows, include_assignments=True)

        self.assertEqual(snapshot["available_signatures"], 4)
        self.assertEqual(snapshot["comparable_scopes"], 2)
        self.assertEqual(snapshot["return_source_clusters"], 3)
        self.assertEqual(snapshot["largest_return_source_cluster"], 2)
        self.assertEqual(snapshot["return_source_redundancy_rate"], 0.25)
        first_scope = next(
            value
            for key, value in snapshot["scope_summaries"].items()
            if key.endswith("|T1")
        )
        self.assertEqual(first_scope["clusters"], 2)
        self.assertEqual(return_source_quality(snapshot, required_sources=4), 1.125)

    def test_opt_in_meta_score_rewards_distinct_training_return_sources(self):
        base = build_return_path_signature(
            [math.sin(index / 5) + index / 1000 for index in range(240)]
        )
        same = build_return_path_signature(
            [3 * (math.sin(index / 5) + index / 1000) for index in range(240)]
        )
        different = build_return_path_signature(
            [math.cos(index / 11) - index / 2000 for index in range(240)]
        )

        def envelope(family, expression, signature):
            row = _row(family, expression, 2.0)
            row.update({
                "node_id": expression,
                "task_name": "T1",
                "market": "us",
                "portfolio_mode": "long_short",
                "diversity": {
                    "declared_family": family,
                    "return_path_signature": signature,
                },
            })
            return row

        repeated = [
            envelope("momentum", "rank(ts_delta(close,20))", base),
            envelope("reversal", "-rank(ts_delta(close,5))", same),
        ]
        independent = [
            repeated[0],
            envelope("reversal", "-rank(ts_delta(close,80))", different),
        ]

        repeated_score, repeated_detail = diversity_adjusted_score(
            {"T1": 2.0}, repeated, "us", return_source_weight=0.30
        )
        independent_score, independent_detail = diversity_adjusted_score(
            {"T1": 2.0}, independent, "us", return_source_weight=0.30
        )

        self.assertGreater(independent_score, repeated_score)
        self.assertEqual(repeated_detail["return_source_clusters"], 1)
        self.assertEqual(independent_detail["return_source_clusters"], 2)
        self.assertEqual(
            independent_detail["score_semantics"],
            "legacy_mechanism_score_plus_training_return_source_quality_v2",
        )


if __name__ == "__main__":
    unittest.main()
