import unittest

from backend.app.dsl.engine import (
    expression_profile,
    expression_to_latex,
    required_history,
)
from backend.app.factors.similarity import (
    build_similarity_index,
    nearest_factors,
)


class FactorLibraryV4Tests(unittest.TestCase):
    def setUp(self):
        self.items = [
            {
                "id": 1,
                "name": "delta-rank",
                "expression": "rank(ts_delta(close, 20))",
                "score": 2.0,
            },
            {
                "id": 2,
                "name": "delta-inverted",
                "expression": "-rank(ts_delta(close, 20))",
                "score": 1.8,
            },
            {
                "id": 3,
                "name": "delta-z",
                "expression": "zscore(ts_delta(close, 20))",
                "score": 1.5,
            },
            {
                "id": 4,
                "name": "value",
                "expression": "-rank(pb)",
                "score": 1.2,
            },
        ]

    def test_sign_and_rank_transforms_group_together(self):
        index = build_similarity_index(self.items, threshold=0.64)
        group = index["factor_to_group"]
        self.assertEqual(group[1], group[2])
        self.assertEqual(group[1], group[3])
        self.assertNotEqual(group[1], group[4])
        nearest = nearest_factors(self.items, 1)
        self.assertEqual(nearest[0]["id"], 2)
        self.assertGreaterEqual(nearest[0]["similarity"], 0.95)

    def test_latex_is_safe_and_web_renderable(self):
        latex = expression_to_latex(
            "rank((close-ts_mean(close,20))/(ts_std(close,20)+0.001))"
        )
        self.assertIn(r"\operatorname{Rank}_{cs}", latex)
        self.assertIn(r"\operatorname{Mean}_{20}", latex)
        self.assertIn(r"\operatorname{Std}_{20}", latex)
        self.assertNotIn("<script", latex)
        profile = expression_profile("rank(ts_delta(close, 20))")
        self.assertEqual(profile["fields"], ["close"])
        self.assertIn("ts_delta", profile["operators"])

    def test_nested_history_is_conservative(self):
        self.assertGreaterEqual(
            required_history("ts_mean(ts_delta(close, 20), 60)"),
            81,
        )

    def test_numeric_window_variants_do_not_escape_lsh_grouping(self):
        expressions = [
            (
                "rank(ts_mean((ts_delta(close, 1)/(delay(close, 1)+1e-9))"
                "*vol/(ts_mean(vol, 40)+1e-9), 30))"
            ),
            (
                "-(rank(ts_mean((ts_delta(close, 1)/(delay(close, 1)+1e-9))"
                "*vol/(ts_mean(vol, 3)+1e-9), 30)))"
            ),
            (
                "-(rank(ts_mean((ts_delta(close, 1)/(delay(close, 1)+1e-9))"
                "*vol/(ts_mean(vol, 120)+1e-9), 30)))"
            ),
        ]
        items = [
            {
                "id": index,
                "name": f"window-{index}",
                "expression": expression,
                "score": 4 - index,
            }
            for index, expression in enumerate(expressions, start=10)
        ]
        index = build_similarity_index(items, threshold=0.64)
        groups = index["factor_to_group"]
        self.assertEqual(groups[10], groups[11])
        self.assertEqual(groups[10], groups[12])
        self.assertEqual(index["stats"]["template_candidate_pairs"], 3)
        self.assertIn("ast_template", index["stats"]["algorithm"])


if __name__ == "__main__":
    unittest.main()
