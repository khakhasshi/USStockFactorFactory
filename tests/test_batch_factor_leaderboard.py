import unittest
from datetime import date, timedelta

import polars as pl

from backend.app.backtest.batch import (
    BatchBacktestSpec,
    canonical_expression,
    information_coefficients,
    oriented_expression_hash,
    rank_factor_results,
    select_training_direction,
)


def _ic_frame(start: date, days: int = 12, symbols: int = 60) -> pl.DataFrame:
    rows = []
    for day_index in range(days):
        trade_date = start + timedelta(days=day_index)
        for symbol_index in range(symbols):
            signal = float(symbol_index + day_index / 100)
            rows.append({
                "trade_date": trade_date,
                "ts_code": f"S{symbol_index:03d}",
                "factor": signal,
                "fwd_1": -signal,
                "univ_rank": symbol_index + 1,
            })
    return pl.DataFrame(rows)


class BatchFactorMetricTests(unittest.TestCase):
    def test_canonical_hash_ignores_formatting(self):
        left = canonical_expression("rank(close-delay(close, 5))")
        right = canonical_expression("rank( close - delay( close , 5 ) )")
        self.assertEqual(left, right)

    def test_oriented_hash_collapses_explicit_negative_alias(self):
        self.assertEqual(
            oriented_expression_hash("rank(close)", -1),
            oriented_expression_hash("-rank(close)", 1),
        )

    def test_training_direction_is_frozen_from_negative_rank_ic(self):
        frame = _ic_frame(date(2020, 1, 2))
        spec = BatchBacktestSpec(
            horizon=1,
            train_start="2020-01-01",
            train_end="2020-12-31",
        )
        direction, metrics = select_training_direction(frame, spec)
        self.assertEqual(direction, -1)
        self.assertTrue(metrics["frozen_before_holdout"])
        self.assertGreater(metrics["rank_ic_mean"], 0.99)

    def test_information_coefficients_respect_direction(self):
        frame = _ic_frame(date(2023, 1, 2))
        direct = information_coefficients(
            frame,
            start="2023-01-01",
            end="2023-12-31",
            horizon=1,
            universe_n=60,
            direction=1,
        )
        reverse = information_coefficients(
            frame,
            start="2023-01-01",
            end="2023-12-31",
            horizon=1,
            universe_n=60,
            direction=-1,
        )
        self.assertLess(direct["rank_ic_mean"], -0.99)
        self.assertGreater(reverse["rank_ic_mean"], 0.99)
        self.assertEqual(direct["n_days"], 10)

    def test_multidimensional_rank_requires_positive_stressed_results(self):
        base = {
            "status": "ok",
            "origin_scope": "us",
            "direction": 1,
            "ann_return_bps_0": 0.20,
            "ann_return_bps_5": 0.18,
            "sharpe_bps_0": 1.5,
            "sharpe_bps_5": 1.3,
            "oos_ic_mean": 0.03,
            "oos_ic_std": 0.02,
            "oos_icir": 1.2,
            "oos_rank_ic_mean": 0.04,
            "oos_rank_ic_std": 0.02,
            "oos_rank_icir": 1.4,
            "oos_n_days": 96,
            "integrity_bps_0": True,
            "integrity_bps_5": True,
            "integrity_bps_15": True,
        }
        rows = [
            {
                **base,
                "expression_hash": "winner",
                "oriented_expression_hash": "winner",
                "ann_return_bps_15": 0.12,
                "sharpe_bps_15": 1.0,
            },
            {
                **base,
                "expression_hash": "cost_failure",
                "oriented_expression_hash": "cost_failure",
                "ann_return_bps_15": -0.02,
                "sharpe_bps_15": -0.2,
            },
        ]
        ranked = rank_factor_results(rows)
        self.assertEqual(ranked[0]["expression_hash"], "winner")
        self.assertTrue(ranked[0]["practical_pass"])
        self.assertFalse(ranked[1]["practical_pass"])

    def test_identical_evaluation_fingerprints_share_one_rank(self):
        common = {
            "status": "ok",
            "evaluation_fingerprint": "same-signal",
            "source_record_count": 1,
            "ann_return_bps_0": 0.20,
            "ann_return_bps_5": 0.18,
            "ann_return_bps_15": 0.12,
            "sharpe_bps_15": 1.0,
            "oos_ic_mean": 0.03,
            "oos_ic_std": 0.02,
            "oos_icir": 1.2,
            "oos_rank_ic_mean": 0.04,
            "oos_rank_ic_std": 0.02,
            "oos_rank_icir": 1.4,
            "oos_n_days": 96,
            "integrity_bps_0": True,
            "integrity_bps_5": True,
            "integrity_bps_15": True,
        }
        ranked = rank_factor_results([
            {**common, "expression_hash": "a"},
            {**common, "expression_hash": "b"},
        ])
        self.assertEqual(len(ranked), 2)
        self.assertEqual(
            sum(row["economic_representative"] for row in ranked),
            1,
        )
        self.assertEqual(
            {row["overall_rank"] for row in ranked},
            {1},
        )


if __name__ == "__main__":
    unittest.main()
