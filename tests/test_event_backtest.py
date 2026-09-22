import json
import io
from backend.app.artifact_exports import csv_chunks
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl

from backend.app.backtest.engine import (
    EventBacktestConfig,
    StepEventBacktester,
    _prepare_backtest_frame,
    _write_artifacts,
)
from backend.app.backtest.fees import calculate_trade_fees
from backend.app.config import ASHARE_PANEL_GLOB, US_PANEL_GLOB
from backend.app.data.panel import PanelStore


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

    def test_reverse_direction_is_preserved_in_orders_and_statement(self):
        config = EventBacktestConfig(
            market="us",
            mode="long_short",
            direction=-1,
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
        by_symbol = {trade["symbol"]: trade for trade in runner.trades}
        self.assertEqual(by_symbol["DDD"]["side"], "BUY")
        self.assertEqual(by_symbol["AAA"]["side"], "SELL")
        result = runner.result()
        self.assertEqual(result["config"]["direction"], -1)
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
            self.assertFalse(statement_path.exists())
            statement = b"".join(csv_chunks(statement_path)).decode()
            self.assertIn("commission", statement)
            self.assertIn("cash_after", statement)
            csv_frame = pl.read_csv(io.BytesIO(statement.encode()))
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

    def test_artifact_schema_scans_late_fractional_execution_values(self):
        # Polars normally infers a sequence-of-dicts schema from the first 100
        # rows.  A long integer-looking prefix followed by a fractional US fill
        # must still publish a complete audit ledger.
        trades = [
            {"fill_id": index, "quantity": index + 1, "price": 10.0}
            for index in range(101)
        ]
        trades.append({"fill_id": 101, "quantity": 17.055546, "price": 10.0})
        result = {
            "trades": trades,
            "events": [],
            "daily_steps": [],
            "round_trips": [],
            "config": {},
            "fee_schedule": {},
            "stats": {},
            "integrity": {"all_pass": True},
        }
        with tempfile.TemporaryDirectory() as tmp:
            manifest = _write_artifacts(result, Path(tmp))
            stored = pl.read_parquet(Path(tmp) / "settlement_statement.parquet")
            self.assertEqual(manifest["files"]["trades"]["rows"], 102)
            self.assertEqual(stored.height, 102)
            self.assertAlmostEqual(stored["quantity"][-1], 17.055546)

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

    def test_summary_capture_matches_full_ledger_and_audits_every_fill(self):
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
        full = StepEventBacktester(config)
        summary = StepEventBacktester(config, capture_detail=False)
        dates = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
        for index, trade_date in enumerate(dates):
            rows = _rows(index)
            next_date = dates[index + 1] if index + 1 < len(dates) else None
            for runner in (full, summary):
                runner.step(
                    trade_date=trade_date,
                    rows=rows,
                    next_trade_date=next_date,
                    rebalance=True,
                )
        full_result = full.result()
        summary_result = summary.result()
        self.assertEqual(full_result["stats"], summary_result["stats"])
        self.assertEqual(summary_result["trades"], [])
        self.assertEqual(summary_result["events"], [])
        self.assertEqual(
            summary_result["integrity"]["statement_rows"],
            full_result["stats"]["fills"],
        )
        self.assertEqual(
            summary_result["integrity"]["audit_mode"],
            "online_fill_reconciliation",
        )
        self.assertTrue(summary_result["integrity"]["all_pass"])

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


class PanelRoutingRegressionTests(unittest.TestCase):
    def test_explicit_market_uses_its_own_default_panel(self):
        self.assertEqual(
            PanelStore(panel_glob=None, market="ashare").panel_glob,
            ASHARE_PANEL_GLOB,
        )
        self.assertEqual(
            PanelStore(panel_glob=None, market="us").panel_glob,
            US_PANEL_GLOB,
        )

    def test_factor_is_computed_before_liquidity_union_filter(self):
        dates = [date(2023, 1, 2) + timedelta(days=index) for index in range(65)]
        rows = []
        for trade_date in dates:
            for code, close, univ_rank in (
                ("A", 1.0, 1),
                ("B", 2.0, 2),
                # Never enters Top-2 but must remain in the cross-section used
                # by zscore before A/B are selected for the event ledger.
                ("OUT", 100.0, 3),
            ):
                rows.append({
                    "trade_date": trade_date,
                    "ts_code": code,
                    "name": code,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "vol": 1_000_000.0,
                    "amount": 10_000_000.0,
                    "raw_open": close,
                    "raw_close": close,
                    "adjustment_factor": 1.0,
                    "can_buy_open_proxy": True,
                    "can_sell_open_proxy": True,
                    "univ_rank": univ_rank,
                })
        panel = pl.DataFrame(rows)

        class FakeStore:
            trading_dates = dates

            @staticmethod
            def ensure_loaded():
                return panel

        with patch(
            "backend.app.backtest.engine.PanelStore.get",
            return_value=FakeStore(),
        ):
            frame, _ = _prepare_backtest_frame(
                expression="zscore(close)",
                universe_n=2,
                start=str(dates[0]),
                end=str(dates[-1]),
                panel_glob="unused",
                market="us",
            )
        first_a = frame.filter(
            (pl.col("trade_date") == dates[0])
            & (pl.col("ts_code") == "A")
        )["factor"][0]
        expected = (1.0 - (1.0 + 2.0 + 100.0) / 3.0) / pl.Series(
            [1.0, 2.0, 100.0]
        ).std()
        union_only = (1.0 - 1.5) / pl.Series([1.0, 2.0]).std()
        self.assertAlmostEqual(first_a, expected, places=10)
        self.assertNotAlmostEqual(first_a, union_only, places=5)
        self.assertEqual(set(frame["ts_code"].unique()), {"A", "B"})


if __name__ == "__main__":
    unittest.main()
