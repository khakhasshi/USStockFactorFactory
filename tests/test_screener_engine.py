import unittest
from datetime import date, timedelta

import polars as pl

from backend.app.screener import screen_cross_section


def _panel() -> tuple[pl.DataFrame, list[date]]:
    dates = [date(2024, 1, 2) + timedelta(days=index) for index in range(40)]
    rows = []
    for day_index, trade_date in enumerate(dates):
        for symbol_index in range(100):
            price = 10 + symbol_index * 0.1 + day_index * (symbol_index + 1) * 0.001
            rows.append({
                "trade_date": trade_date,
                "ts_code": f"S{symbol_index:03d}",
                "name": f"Stock {symbol_index:03d}",
                "close": price,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "raw_close": price,
                "vol": 1_000_000.0,
                "amount": price * 1_000_000,
                "univ_rank": symbol_index + 1,
            })
    return pl.DataFrame(rows), dates


class ScreenerEngineTests(unittest.TestCase):
    def test_single_plan_components_and_cache(self):
        frame, dates = _panel()
        kwargs = {
            "df": frame,
            "trading_dates": dates,
            "panel_identity": "synthetic-screen-v1",
            "target_date": dates[-1],
            "factors": [
                {"expression": "rank(close)", "weight": 2, "direction": 1},
                {"expression": "rank(ts_delta(close, 5))", "weight": 1, "direction": 1},
            ],
            "fields": ["open", "high", "low", "close", "vol", "amount"],
            "universe_n": 100,
            "top_n": 10,
            "direction": "top",
        }
        first = screen_cross_section(**kwargs)
        second = screen_cross_section(**kwargs)
        self.assertFalse(first["performance"]["cache_hit"])
        self.assertTrue(second["performance"]["cache_hit"])
        self.assertEqual(len(first["stocks"]), 10)
        self.assertEqual(len(first["stocks"][0]["components"]), 2)
        contribution = sum(
            row["contribution"] for row in first["stocks"][0]["components"]
        )
        self.assertAlmostEqual(
            contribution,
            first["stocks"][0]["score"],
            places=3,
        )
        self.assertLess(first["required_history"], 20)


if __name__ == "__main__":
    unittest.main()
