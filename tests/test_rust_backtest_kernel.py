import unittest
from datetime import date, timedelta

import polars as pl

from backend.app.backtest.engine import EventBacktestConfig, StepEventBacktester
from backend.app.backtest.rust_kernel import (
    align_shadow_results,
    run_rust_kernel,
    rust_kernel_capabilities,
)


def _session(day_index: int) -> list[dict]:
    trade_date = date(2024, 1, 2) + timedelta(days=day_index)
    base = [("AAA", 4.0), ("BBB", 3.0), ("CCC", 2.0), ("DDD", 1.0)]
    if day_index % 2:
        base = [(symbol, 5.0 - factor) for symbol, factor in base]
    return [
        {
            "trade_date": trade_date,
            "ts_code": symbol,
            "name": symbol,
            "factor": factor,
            "univ_rank": rank,
            "raw_open": 10.0 + day_index * 0.1 + rank * 0.01,
            "raw_high": 10.1 + day_index * 0.1 + rank * 0.01,
            "raw_low": 9.9 + day_index * 0.1 + rank * 0.01,
            "raw_close": 10.05 + day_index * 0.1 + rank * 0.01,
            "vol": 1_000_000.0,
            "amount": 10_000_000.0,
            "adjustment_factor": 1.0,
            "can_buy_open_proxy": True,
            "can_sell_open_proxy": True,
            "_adv20_prev": 1_000_000.0,
            "_atr_pct": 0.02,
            "_vol20_prev": 0.2,
        }
        for rank, (symbol, factor) in enumerate(base, start=1)
    ]


class RustKernelAlignmentTests(unittest.TestCase):
    def test_library_is_loadable(self):
        self.assertTrue(rust_kernel_capabilities()["available"])

    def test_three_primary_paths_match_python_ledgers(self):
        for market, mode in (
            ("ashare", "long_only"),
            ("us", "long_only"),
            ("us", "long_short"),
        ):
            with self.subTest(market=market, mode=mode):
                config = EventBacktestConfig(
                    market=market,
                    mode=mode,
                    universe_n=4,
                    top_fraction=0.25,
                    initial_capital=100_000.0,
                    rebalance_every=1,
                    slippage_bps=0.0,
                    max_volume_participation=1.0,
                    borrow_cost_bps_annual=300.0,
                )
                runner = StepEventBacktester(config)
                frame_rows = []
                for day_index in range(6):
                    rows = _session(day_index)
                    frame_rows.extend(rows)
                    runner.step(
                        trade_date=rows[0]["trade_date"],
                        rows=rows,
                        next_trade_date=(
                            _session(day_index + 1)[0]["trade_date"]
                            if day_index < 5 else None
                        ),
                        rebalance=True,
                    )
                rust = run_rust_kernel(pl.DataFrame(frame_rows), config)
                alignment = align_shadow_results(runner.result(), rust)
                self.assertTrue(alignment["all_pass"], alignment)

    def test_square_root_impact_matches_python(self):
        config = EventBacktestConfig(
            market="us",
            mode="long_short",
            universe_n=4,
            top_fraction=0.25,
            initial_capital=100_000.0,
            rebalance_every=1,
            slippage_bps=2.0,
            spread_bps=1.0,
            impact_model="square_root",
            impact_coefficient_bps=20.0,
            max_volume_participation=1.0,
            borrow_cost_bps_annual=300.0,
        )
        runner = StepEventBacktester(config)
        frame_rows = []
        for day_index in range(6):
            rows = _session(day_index)
            frame_rows.extend(rows)
            runner.step(
                trade_date=rows[0]["trade_date"],
                rows=rows,
                next_trade_date=(
                    _session(day_index + 1)[0]["trade_date"]
                    if day_index < 5 else None
                ),
                rebalance=True,
            )
        alignment = align_shadow_results(
            runner.result(), run_rust_kernel(pl.DataFrame(frame_rows), config)
        )
        self.assertTrue(alignment["all_pass"], alignment)


if __name__ == "__main__":
    unittest.main()
