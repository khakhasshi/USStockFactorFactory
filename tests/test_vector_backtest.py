import unittest
from datetime import date, timedelta

import polars as pl

from backend.app.backtest.batch import BatchBacktestSpec
from backend.app.backtest.vector import (
    VECTOR_SCREEN_PROTOCOL,
    run_vector_cost_scenarios,
)
from backend.scripts.window_dual_direction_leaderboard import (
    _select_independent_ledger_rows,
)


def _frame(days: int = 70) -> pl.DataFrame:
    rows = []
    start = date(2020, 1, 1)
    symbols = [
        ("A", 1.0, -0.02),
        ("B", 2.0, -0.01),
        ("C", 3.0, 0.01),
        ("D", 4.0, 0.02),
    ]
    for offset in range(days):
        trade_date = start + timedelta(days=offset)
        for rank, (symbol, factor, forward) in enumerate(
            symbols,
            start=1,
        ):
            rows.append({
                "trade_date": trade_date,
                "ts_code": symbol,
                "factor": factor,
                "univ_rank": rank,
                "raw_close": 100.0,
                "fwd_5": forward,
            })
    return pl.DataFrame(rows)


class VectorBacktestTests(unittest.TestCase):
    def test_long_only_tests_both_sides_and_costs_are_monotonic(self):
        frame = _frame()
        spec = BatchBacktestSpec(
            market="ashare",
            mode="long_only",
            universe_n=4,
            horizon=5,
            top_fraction=0.25,
            initial_capital=10_000_000.0,
            rebalance_every=5,
            holdout_start="2020-01-01",
            holdout_end="2020-03-10",
            slippage_bps=(0.0, 5.0, 15.0),
        )
        positive = run_vector_cost_scenarios(
            frame,
            direction=1,
            spec=spec,
            capture_periods=True,
        )
        negative = run_vector_cost_scenarios(
            frame,
            direction=-1,
            spec=spec,
            capture_periods=True,
        )

        self.assertGreater(
            positive["0"]["ann_return"],
            negative["0"]["ann_return"],
        )
        self.assertGreaterEqual(
            positive["0"]["ann_return"],
            positive["5"]["ann_return"],
        )
        self.assertGreaterEqual(
            positive["5"]["ann_return"],
            positive["15"]["ann_return"],
        )
        self.assertAlmostEqual(
            positive["0"]["period_rows"][0]["gross_return"],
            -negative["0"]["period_rows"][0]["gross_return"],
            places=12,
        )
        self.assertEqual(
            positive["0"]["integrity"]["protocol"],
            VECTOR_SCREEN_PROTOCOL,
        )
        self.assertTrue(
            positive["0"]["integrity"]["event_replay_required"]
        )
        self.assertFalse(
            positive["0"]["integrity"][
                "settlement_statement_available"
            ]
        )
        self.assertGreater(
            positive["0"]["commission_and_tax"],
            0.0,
        )
        self.assertEqual(
            positive["0"]["portfolio_fingerprint"],
            positive["15"]["portfolio_fingerprint"],
        )
        self.assertNotEqual(
            positive["0"]["portfolio_fingerprint"],
            negative["0"]["portfolio_fingerprint"],
        )
        period = positive["0"]["period_rows"][0]
        self.assertAlmostEqual(
            period["active_return"],
            period["net_return"] - period["benchmark_return"],
            places=12,
        )
        self.assertIn("active_sharpe", positive["0"])
        self.assertIn("benchmark_ann_return", positive["0"])

    def test_long_short_has_two_sided_exposure_and_borrow(self):
        frame = _frame()
        spec = BatchBacktestSpec(
            market="us",
            mode="long_short",
            universe_n=4,
            horizon=5,
            top_fraction=0.25,
            initial_capital=1_000_000.0,
            rebalance_every=5,
            holdout_start="2020-01-01",
            holdout_end="2020-03-10",
            slippage_bps=(0.0,),
            borrow_cost_bps_annual=300.0,
        )
        positive = run_vector_cost_scenarios(
            frame,
            direction=1,
            spec=spec,
            capture_periods=True,
        )["0"]
        negative = run_vector_cost_scenarios(
            frame,
            direction=-1,
            spec=spec,
            capture_periods=True,
        )["0"]

        self.assertAlmostEqual(
            positive["avg_gross_exposure"],
            2.0,
            places=6,
        )
        self.assertAlmostEqual(
            positive["avg_net_exposure"],
            0.0,
            places=6,
        )
        self.assertGreater(positive["borrow_cost"], 0.0)
        self.assertGreater(
            positive["period_rows"][0]["gross_return"],
            0.0,
        )
        self.assertLess(
            negative["period_rows"][0]["gross_return"],
            0.0,
        )

    def test_horizon_must_match_rebalance_period(self):
        frame = _frame()
        spec = BatchBacktestSpec(
            market="us",
            mode="long_only",
            universe_n=4,
            horizon=5,
            top_fraction=0.25,
            rebalance_every=10,
            holdout_start="2020-01-01",
            holdout_end="2020-03-10",
            slippage_bps=(0.0,),
        )
        with self.assertRaisesRegex(
            ValueError,
            "horizon 与 rebalance_every",
        ):
            run_vector_cost_scenarios(
                frame,
                direction=1,
                spec=spec,
            )

    def test_finalist_selection_skips_identical_portfolio_paths(self):
        common = {
            "ann_return_bps_0": 0.2,
            "ann_return_bps_5": 0.19,
            "ann_return_bps_15": 0.17,
        }
        rows = [
            {
                **common,
                "expression_hash": "first",
                "portfolio_fingerprint_bps_15": "same-path",
            },
            {
                **common,
                "expression_hash": "alias",
                "portfolio_fingerprint_bps_15": "same-path",
            },
            {
                **common,
                "expression_hash": "independent",
                "portfolio_fingerprint_bps_15": "other-path",
            },
        ]
        selected = _select_independent_ledger_rows(rows, 2)
        self.assertEqual(
            [row["expression_hash"] for row in selected],
            ["first", "independent"],
        )


if __name__ == "__main__":
    unittest.main()
