import sys
import unittest
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.dsl.engine import parse, validate


def _reference_ts_rank(series: pl.Series) -> float | None:
    last = series[-1]
    if last is None:
        return None
    return float((series <= last).sum()) / len(series)


class DslEngineTests(unittest.TestCase):
    def test_native_ts_rank_matches_legacy_percentile_semantics(self):
        frame = pl.DataFrame({
            "ts_code": ["A"] * 8 + ["B"] * 8,
            "trade_date": list(range(8)) * 2,
            "close": [
                1.0, 2.0, 2.0, 3.0, None, 2.0, 4.0, 4.0,
                5.0, 5.0, 4.0, None, 4.0, 3.0, 3.0, 6.0,
            ],
        })
        actual = (
            parse("ts_rank(close, 4)", fields=["close"])
            .apply(frame.lazy())
            .collect()["factor"]
        )
        expected = frame.with_columns(
            pl.col("close")
            .rolling_map(_reference_ts_rank, 4, min_samples=4)
            .over("ts_code", order_by="trade_date")
            .alias("factor")
        )["factor"]

        self.assertEqual(actual.null_count(), expected.null_count())
        differences = (
            pl.DataFrame({"actual": actual, "expected": expected})
            .drop_nulls()
            .select((pl.col("actual") - pl.col("expected")).abs().max())
            .item()
        )
        self.assertAlmostEqual(float(differences), 0.0, places=12)

    def test_native_ts_rank_remains_composable(self):
        frame = pl.DataFrame({
            "ts_code": ["A"] * 10,
            "trade_date": list(range(10)),
            "close": [float(value) for value in range(10)],
        })
        result = (
            parse(
                "ts_delta(ts_rank(close, 4), 1)",
                fields=["close"],
            )
            .apply(frame.lazy())
            .collect()["factor"]
        )
        self.assertEqual(result.null_count(), 4)
        self.assertTrue(all(value == 0.0 for value in result.drop_nulls()))

    def test_external_returns_matches_lagged_price_return(self):
        frame = pl.DataFrame({
            "ts_code": ["A"] * 5,
            "trade_date": list(range(5)),
            "close": [10.0, 11.0, 12.1, 12.1, 13.31],
        })
        result = (
            parse("returns(close, 1)", fields=["close"])
            .apply(frame.lazy())
            .collect()["factor"]
        )
        self.assertIsNone(result[0])
        expected = [0.1, 0.1, 0.0, 0.1]
        for value, target in zip(result.drop_nulls(), expected, strict=True):
            self.assertAlmostEqual(float(value), target, places=10)

    def test_external_winsor_mad_uses_cross_sectional_mad(self):
        frame = pl.DataFrame({
            "ts_code": ["A", "B", "C", "D", "E"],
            "trade_date": [1] * 5,
            "close": [1.0, 2.0, 3.0, 4.0, 100.0],
        })
        result = (
            parse("winsor_mad(close, 1)", fields=["close"])
            .apply(frame.lazy())
            .collect()["factor"]
        )
        self.assertAlmostEqual(float(result[-1]), 3.0 + 1.4826, places=6)
        self.assertAlmostEqual(float(result[0]), 3.0 - 1.4826, places=6)

    def test_external_one_year_windows_accept_252_but_not_more(self):
        self.assertIsNone(validate("ts_mean(close, 252)", ["close"]))
        self.assertIn("1..252", validate("ts_mean(close, 253)", ["close"]))


if __name__ == "__main__":
    unittest.main()
