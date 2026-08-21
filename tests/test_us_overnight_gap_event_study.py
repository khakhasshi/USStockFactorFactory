import unittest
from datetime import date, timedelta

import polars as pl

from backend.scripts.us_overnight_gap_event_study import (
    build_gap_buckets,
    build_statistics,
    parse_thresholds,
    summarize_group,
)


def _events() -> pl.DataFrame:
    rows = []
    start = date(2024, 1, 2)
    for symbol_index in range(3):
        symbol = f"T{symbol_index}"
        for day in range(35):
            gap = 0.06 if day in (5, 20) else 0.0
            rows.append(
                {
                    "trade_date": start + timedelta(days=day),
                    "event_trade_date": start + timedelta(days=day + 1),
                    "ts_code": symbol,
                    "name": symbol,
                    "calendar_idx": day,
                    "event_calendar_idx": day + 1,
                    "raw_gap": gap,
                    "adj_gap": gap,
                    "raw_fwd_open_h5": 0.02 if gap else 0.0,
                    "raw_fwd_open_h10": 0.03 if gap else 0.0,
                    "raw_fwd_open_h20": 0.04 if gap else 0.0,
                    "adjusted_fwd_open_h5": 0.02 if gap else 0.0,
                    "adjusted_fwd_open_h10": 0.03 if gap else 0.0,
                    "adjusted_fwd_open_h20": 0.04 if gap else 0.0,
                    "future_event_spacing_h5": 5,
                    "future_event_spacing_h10": 10,
                    "future_event_spacing_h20": 20,
                }
            )
    return pl.DataFrame(rows)


class OvernightGapEventStudyTests(unittest.TestCase):
    def test_thresholds_are_normalized_and_validated(self):
        self.assertEqual(parse_thresholds("0.10, 0.05, 0.05"), (0.05, 0.10))
        with self.assertRaises(ValueError):
            parse_thresholds("0,1.2")

    def test_positive_gap_has_positive_forward_result(self):
        frame = _events()
        row = summarize_group(
            frame,
            gap_column="raw_gap",
            forward_column="raw_fwd_open_h10",
            threshold=0.05,
            sign="positive",
            horizon=10,
            sample_kind="all",
            bootstrap_reps=20,
            seed=1,
        )
        self.assertEqual(row["events"], 6)
        self.assertAlmostEqual(row["mean_return"], 0.03)
        self.assertAlmostEqual(row["win_rate"], 1.0)

    def test_cooldown_removes_clustered_events(self):
        frame = _events()
        all_row = summarize_group(
            frame,
            gap_column="raw_gap",
            forward_column="raw_fwd_open_h5",
            threshold=0.05,
            sign="positive",
            horizon=5,
            sample_kind="all",
            bootstrap_reps=0,
            seed=1,
        )
        cooldown_row = summarize_group(
            frame,
            gap_column="raw_gap",
            forward_column="raw_fwd_open_h5",
            threshold=0.05,
            sign="positive",
            horizon=5,
            sample_kind="cooldown20d",
            bootstrap_reps=0,
            seed=1,
        )
        self.assertEqual(all_row["events"], 6)
        self.assertEqual(cooldown_row["events"], 3)

    def test_statistics_include_both_modes_directions_and_horizons(self):
        result = build_statistics(_events(), (0.05,), bootstrap_reps=0)
        self.assertEqual(result.height, 24)
        self.assertEqual(set(result.get_column("price_mode")), {"raw", "adjusted"})
        self.assertEqual(set(result.get_column("sign")), {"positive", "negative"})
        self.assertEqual(set(result.get_column("horizon")), {5, 10, 20})

    def test_gap_buckets_are_nonempty_for_large_positive_events(self):
        result = build_gap_buckets(_events())
        self.assertGreater(result.height, 0)
        self.assertEqual(result.get_column("events").sum(), 6)

