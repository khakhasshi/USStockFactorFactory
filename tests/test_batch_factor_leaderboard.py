import unittest
from datetime import date, timedelta

import polars as pl

from backend.app.backtest.batch import (
    BatchBacktestSpec,
    batch_protocol_for_market,
    canonical_expression,
    information_coefficients,
    oriented_expression_hash,
    rank_factor_results,
    run_cost_scenarios,
    select_training_direction,
)
from backend.scripts.factor_leaderboard import (
    SOURCE_POLICY_US_PLUS_ASHARE_PRICE_VOLUME,
    _selection_reason,
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


def _us_event_frame(
    start: date,
    days: int = 65,
    symbols: int = 60,
) -> pl.DataFrame:
    rows = []
    for day_index in range(days):
        trade_date = start + timedelta(days=day_index)
        for symbol_index in range(symbols):
            price = 20.0 + symbol_index / 10.0 + day_index / 100.0
            rows.append({
                "trade_date": trade_date,
                "ts_code": f"U{symbol_index:03d}",
                "name": f"U{symbol_index:03d}",
                "factor": float(symbol_index),
                "univ_rank": symbol_index + 1,
                "raw_open": price,
                "raw_close": price + symbol_index / 10_000.0,
                "vol": 10_000_000.0,
                "amount": 100_000_000.0,
                "adjustment_factor": 1.0,
                "can_buy_open_proxy": True,
                "can_sell_open_proxy": True,
                "fwd_1": symbol_index / 100_000.0,
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
            "max_drawdown_bps_0": 0.12,
            "max_drawdown_bps_5": 0.14,
            "max_drawdown_bps_15": 0.18,
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
            "max_drawdown_bps_0": 0.12,
            "max_drawdown_bps_5": 0.14,
            "max_drawdown_bps_15": 0.18,
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

    def test_portfolio_path_fingerprint_overrides_ic_differences(self):
        common = {
            "status": "ok",
            "portfolio_fingerprint_bps_15": "same-holdings",
            "source_record_count": 1,
            "ann_return_bps_0": 0.20,
            "ann_return_bps_5": 0.18,
            "ann_return_bps_15": 0.12,
            "sharpe_bps_15": 1.0,
            "max_drawdown_bps_0": 0.12,
            "max_drawdown_bps_5": 0.14,
            "max_drawdown_bps_15": 0.18,
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
            {
                **common,
                "expression_hash": "a",
                "evaluation_fingerprint": "different-ic-a",
            },
            {
                **common,
                "expression_hash": "b",
                "evaluation_fingerprint": "different-ic-b",
                "oos_ic_mean": 0.031,
            },
        ])
        self.assertEqual(
            sum(row["economic_representative"] for row in ranked),
            1,
        )
        self.assertEqual(
            {row["overall_rank"] for row in ranked},
            {1},
        )

    def test_us_batch_replay_keeps_long_short_borrow_and_ibkr_costs(self):
        frame = _us_event_frame(date(2023, 1, 2))
        spec = BatchBacktestSpec(
            market="us",
            mode="long_short",
            horizon=1,
            universe_n=60,
            top_fraction=0.20,
            initial_capital=100_000,
            rebalance_every=5,
            max_volume_participation=1.0,
            holdout_start="2023-01-01",
            holdout_end="2023-12-31",
            slippage_bps=(0.0,),
            borrow_cost_bps_annual=300.0,
        )
        result = run_cost_scenarios(
            frame,
            direction=1,
            spec=spec,
        )["0"]
        self.assertEqual(result["fee_profile"], "ibkr_pro_fixed_us_v1")
        self.assertEqual(result["currency"], "USD")
        self.assertGreater(result["borrow_cost"], 0)
        self.assertGreater(result["avg_gross_exposure"], 1.5)
        self.assertLess(abs(result["avg_net_exposure"]), 0.10)
        self.assertTrue(result["integrity"]["all_pass"])

    def test_us_long_only_has_distinct_protocol_and_no_short_financing(self):
        self.assertEqual(
            batch_protocol_for_market("us", "long_only"),
            "us_cross_task_long_only_factor_leaderboard_v1",
        )
        self.assertEqual(
            batch_protocol_for_market("us", "long_short"),
            "us_cross_task_long_short_factor_leaderboard_v1",
        )
        frame = _us_event_frame(date(2023, 1, 2))
        spec = BatchBacktestSpec(
            market="us",
            mode="long_only",
            horizon=1,
            universe_n=60,
            top_fraction=0.20,
            initial_capital=100_000,
            rebalance_every=5,
            max_volume_participation=1.0,
            holdout_start="2023-01-01",
            holdout_end="2023-12-31",
            slippage_bps=(0.0,),
            borrow_cost_bps_annual=0.0,
        )
        result = run_cost_scenarios(
            frame,
            direction=1,
            spec=spec,
        )["0"]
        self.assertEqual(result["fee_profile"], "ibkr_pro_fixed_us_v1")
        self.assertEqual(result["currency"], "USD")
        self.assertEqual(result["borrow_cost"], 0.0)
        self.assertGreater(result["avg_net_exposure"], 0.50)
        self.assertAlmostEqual(
            result["avg_gross_exposure"],
            result["avg_net_exposure"],
            places=6,
        )
        self.assertEqual(
            result["integrity"]["long_only_negative_position_violations"],
            0,
        )
        self.assertTrue(result["integrity"]["all_pass"])

    def test_us_source_policy_includes_us_and_only_portable_ashare(self):
        policy = SOURCE_POLICY_US_PLUS_ASHARE_PRICE_VOLUME
        us_fundamental = {
            "origin_markets": ["us"],
            "profile": {"fields": ["turnover_rate"]},
        }
        ashare_price_volume = {
            "origin_markets": ["ashare"],
            "profile": {"fields": ["close", "amount"]},
        }
        ashare_fundamental = {
            "origin_markets": ["ashare"],
            "profile": {"fields": ["close", "pe_ttm"]},
        }
        both_price_volume = {
            "origin_markets": ["ashare", "us"],
            "profile": {"fields": ["close", "vol"]},
        }
        self.assertEqual(
            _selection_reason(us_fundamental, policy),
            "us_origin",
        )
        self.assertEqual(
            _selection_reason(ashare_price_volume, policy),
            "ashare_price_volume_transfer",
        )
        self.assertIsNone(
            _selection_reason(ashare_fundamental, policy),
        )
        self.assertEqual(
            _selection_reason(both_price_volume, policy),
            "us_origin_and_ashare_price_volume",
        )


if __name__ == "__main__":
    unittest.main()
