import sys
import unittest
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.dsl.engine import parse


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


if __name__ == "__main__":
    unittest.main()
