import json
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import polars as pl

from backend.app.backtest.engine import (
    EventBacktestConfig,
    StepEventBacktester,
    _write_artifacts,
)
from backend.app.backtest.fees import calculate_trade_fees


def _rows(day_index: int, market: str = "ashare") -> list[dict]:
    current = date(2024, 1, 2) + timedelta(days=day_index)
    rows = []
    for index, (symbol, factor) in enumerate(
        [("AAA", 4.0), ("BBB", 3.0), ("CCC", 2.0), ("DDD", 1.0)],
        start=1,
    ):
        rows.append({
            "trade_date": current,
            "ts_code": symbol,
            "name": symbol,
            "factor": factor,
            "univ_rank": index,
            # More precision than the statement price to catch hidden-price
            # cash reconciliation regressions.
            "raw_open": 10.123456789 + day_index,
            "raw_close": 10.223456789 + day_index,
            "vol": 1_000_000.0,
            "amount": 10_000_000.0,
            "adjustment_factor": 1.0,
            "can_buy_open_proxy": True,
            "can_sell_open_proxy": True,
            "market": market,
        })
    return rows


class FeeScheduleRegressionTests(unittest.TestCase):
    def test_ashare_wan2_no_min_and_sell_taxes(self):
        buy = calculate_trade_fees(
            market="ashare",
            side="BUY",
            quantity=100,
            price=10,
            trade_date=date(2024, 1, 2),
        )
        self.assertEqual(buy.commission, 0.20)
        self.assertEqual(buy.transfer_fee, 0.01)
        self.assertEqual(buy.stamp_duty, 0.0)
        self.assertEqual(buy.total, 0.21)
        self.assertLess(buy.total, 5.0, "万2免5不能偷偷恢复最低五元")

        sell = calculate_trade_fees(
            market="ashare",
            side="SELL",
            quantity=100,
            price=10,
            trade_date=date(2024, 1, 2),
        )
        self.assertEqual(sell.commission, 0.20)
        self.assertEqual(sell.transfer_fee, 0.01)
        self.assertEqual(sell.stamp_duty, 0.50)
        self.assertEqual(sell.total, 0.71)

    def test_ashare_historical_rate_boundaries(self):
        before_transfer_cut = calculate_trade_fees(
            market="ashare",
            side="SELL",
            quantity=1_000,
            price=10,
            trade_date=date(2022, 4, 28),
        )
        after_transfer_cut = calculate_trade_fees(
            market="ashare",
            side="SELL",
            quantity=1_000,
            price=10,
            trade_date=date(2022, 4, 29),
        )
        self.assertEqual(before_transfer_cut.transfer_fee, 0.20)
        self.assertEqual(after_transfer_cut.transfer_fee, 0.10)
        before_stamp_cut = calculate_trade_fees(
            market="ashare",
            side="SELL",
            quantity=1_000,
            price=10,
            trade_date=date(2023, 8, 27),
        )
        after_stamp_cut = calculate_trade_fees(
            market="ashare",
            side="SELL",
            quantity=1_000,
            price=10,
            trade_date=date(2023, 8, 28),
        )
        self.assertEqual(before_stamp_cut.stamp_duty, 10.0)
        self.assertEqual(after_stamp_cut.stamp_duty, 5.0)

    def test_ibkr_pro_fixed_minimum_and_cap(self):
        standard = calculate_trade_fees(
            market="us",
            side="BUY",
            quantity=100,
            price=10,
            trade_date=date(2026, 8, 5),
        )
        self.assertEqual(standard.commission, 1.0)
        large = calculate_trade_fees(
            market="us",
            side="SELL",
            quantity=1_000,
            price=100,
            trade_date=date(2026, 8, 5),
        )
        self.assertEqual(large.commission, 5.0)
        penny_order = calculate_trade_fees(
            market="us",
            side="BUY",
            quantity=1,
            price=10,
            trade_date=date(2026, 8, 5),
        )
        self.assertEqual(penny_order.commission, 0.10)


class EventLedgerRegressionTests(unittest.TestCase):
    def test_signal_fills_only_on_next_open_and_cash_reconciles(self):
        config = EventBacktestConfig(
            market="ashare",
            mode="long_only",
            universe_n=4,
            top_fraction=0.25,
            initial_capital=100_000,
            rebalance_every=1,
            slippage_bps=0,
            max_volume_participation=1.0,
        )
        runner = StepEventBacktester(config)
        day0 = date(2024, 1, 2)
        day1 = date(2024, 1, 3)
        runner.step(
            trade_date=day0,
            rows=_rows(0),
            next_trade_date=day1,
            rebalance=True,
        )
        self.assertEqual(runner.trades, [])
        self.assertGreater(len(runner.pending_orders), 0)
        runner.step(
            trade_date=day1,
            rows=_rows(1),
            next_trade_date=None,
            rebalance=False,
        )
        self.assertEqual(len(runner.trades), 1)
        fill = runner.trades[0]
        self.assertEqual(fill["signal_date"], str(day0))
        self.assertEqual(fill["trade_date"], str(day1))
        signed = fill["filled_quantity"]
        expected_cash = (
            fill["cash_before"]
            - signed * fill["fill_price"]
            - fill["total_fees"]
        )
        self.assertAlmostEqual(expected_cash, fill["cash_after"], places=5)
        result = runner.result()
        self.assertTrue(result["integrity"]["all_pass"])
        self.assertEqual(
            result["integrity"]["same_day_signal_fill_violations"], 0
        )
        self.assertEqual(
            result["integrity"]["long_only_negative_position_violations"], 0
        )

    def test_us_long_short_has_both_legs_and_ibkr_statement(self):
        config = EventBacktestConfig(
            market="us",
            mode="long_short",
            universe_n=4,
            top_fraction=0.25,
            initial_capital=100_000,
            rebalance_every=1,
            slippage_bps=0,
            max_volume_participation=1.0,
            borrow_cost_bps_annual=300,
        )
        runner = StepEventBacktester(config)
        runner.step(
            trade_date=date(2024, 1, 2),
            rows=_rows(0, "us"),
            next_trade_date=date(2024, 1, 3),
            rebalance=True,
        )
        runner.step(
            trade_date=date(2024, 1, 3),
            rows=_rows(1, "us"),
            next_trade_date=None,
            rebalance=False,
        )
        self.assertEqual({trade["side"] for trade in runner.trades}, {"BUY", "SELL"})
        self.assertTrue(all(
            trade["fee_profile"] == "ibkr_pro_fixed_us_v1"
            for trade in runner.trades
        ))
        self.assertTrue(any(quantity < 0 for quantity in runner.positions.values()))
        result = runner.result()
        self.assertGreater(result["stats"]["borrow_cost"], 0)
        self.assertTrue(result["integrity"]["all_pass"])

    def test_statement_artifacts_are_hashed_and_row_complete(self):
        config = EventBacktestConfig(
            market="ashare",
            mode="long_only",
            universe_n=4,
            top_fraction=0.25,
            initial_capital=100_000,
            slippage_bps=0,
            max_volume_participation=1.0,
        )
        runner = StepEventBacktester(config)
        runner.step(
            trade_date=date(2024, 1, 2),
            rows=_rows(0),
            next_trade_date=date(2024, 1, 3),
            rebalance=True,
        )
        runner.step(
            trade_date=date(2024, 1, 3),
            rows=_rows(1),
            next_trade_date=None,
            rebalance=False,
        )
        result = runner.result()
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_artifacts(result, Path(tmp))
            self.assertEqual(
                manifest["files"]["statement_csv"]["rows"],
                len(result["trades"]),
            )
            self.assertEqual(len(manifest["files"]["statement_csv"]["sha256"]), 64)
            stored = json.loads((Path(tmp) / "manifest.json").read_text())
            self.assertTrue(stored["integrity"]["all_pass"])
            statement_path = Path(tmp) / "settlement_statement.csv"
            statement = statement_path.read_text()
            self.assertIn("commission", statement)
            self.assertIn("cash_after", statement)
            csv_frame = pl.read_csv(statement_path)
            parquet_frame = pl.read_parquet(
                Path(tmp) / "settlement_statement.parquet"
            )
            self.assertEqual(csv_frame.height, len(result["trades"]))
            self.assertEqual(parquet_frame.height, len(result["trades"]))
            self.assertAlmostEqual(
                csv_frame["total_fees"].sum(),
                sum(row["total_fees"] for row in result["trades"]),
                places=6,
            )

    def test_integrity_gate_rejects_a_tampered_statement(self):
        config = EventBacktestConfig(
            market="ashare",
            mode="long_only",
            universe_n=4,
            top_fraction=0.25,
            initial_capital=100_000,
            slippage_bps=0,
            max_volume_participation=1.0,
        )
        runner = StepEventBacktester(config)
        runner.step(
            trade_date=date(2024, 1, 2),
            rows=_rows(0),
            next_trade_date=date(2024, 1, 3),
            rebalance=True,
        )
        runner.step(
            trade_date=date(2024, 1, 3),
            rows=_rows(1),
            next_trade_date=None,
            rebalance=False,
        )
        runner.trades[0]["cash_after"] += 1.0
        checks = runner.result()["integrity"]
        self.assertFalse(checks["all_pass"])
        self.assertGreater(checks["cash_reconciliation_max_error"], 0.9)

    def test_fill_rate_counts_rejected_day_orders(self):
        config = EventBacktestConfig(
            market="ashare",
            mode="long_only",
            universe_n=4,
            top_fraction=0.25,
            initial_capital=100_000,
            slippage_bps=0,
            max_volume_participation=1.0,
        )
        runner = StepEventBacktester(config)
        runner.step(
            trade_date=date(2024, 1, 2),
            rows=_rows(0),
            next_trade_date=date(2024, 1, 3),
            rebalance=True,
        )
        blocked_rows = _rows(1)
        for row in blocked_rows:
            row["can_buy_open_proxy"] = False
        runner.step(
            trade_date=date(2024, 1, 3),
            rows=blocked_rows,
            next_trade_date=None,
            rebalance=False,
        )
        stats = runner.result()["stats"]
        self.assertEqual(stats["orders_executed"], 1)
        self.assertEqual(stats["rejected_orders"], 1)
        self.assertEqual(stats["fill_rate"], 0.0)

    def test_fractional_corporate_action_quantity_reconciles(self):
        config = EventBacktestConfig(
            market="ashare",
            mode="long_only",
            universe_n=4,
            top_fraction=0.25,
            initial_capital=100_000,
            rebalance_every=1,
            slippage_bps=5,
            max_volume_participation=1.0,
        )
        runner = StepEventBacktester(config)
        runner.step(
            trade_date=date(2024, 1, 2),
            rows=_rows(0),
            next_trade_date=date(2024, 1, 3),
            rebalance=True,
        )
        rotated = _rows(1)
        for row in rotated:
            row["factor"] = 10.0 if row["ts_code"] == "BBB" else -10.0
        runner.step(
            trade_date=date(2024, 1, 3),
            rows=rotated,
            next_trade_date=date(2024, 1, 4),
            rebalance=True,
        )
        adjusted = _rows(2)
        for row in adjusted:
            if row["ts_code"] == "AAA":
                row["adjustment_factor"] = 1.123456789
        runner.step(
            trade_date=date(2024, 1, 4),
            rows=adjusted,
            next_trade_date=None,
            rebalance=False,
        )
        sells = [
            trade for trade in runner.trades
            if trade["symbol"] == "AAA" and trade["side"] == "SELL"
        ]
        self.assertEqual(len(sells), 1)
        self.assertNotEqual(sells[0]["filled_quantity"] % 1, 0)
        checks = runner.result()["integrity"]
        self.assertTrue(checks["all_pass"])
        self.assertLessEqual(checks["cash_reconciliation_max_error"], 1e-5)


if __name__ == "__main__":
    unittest.main()
