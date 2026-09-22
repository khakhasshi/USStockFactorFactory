"""Metamorphic causal checks: change future data, not the training decision."""

import math
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest

from backend.app.config import DIRECTION_POLICY_BOTH, evaluation_config, get_layer_bounds
from backend.app.data.panel import with_market_calendar_labels
from backend.app.dsl.engine import parse
from backend.app.eval.harness import (
    _direction_aggregates,
    _evaluate_discovery_orientations,
    _prepare_direction_work,
    _prepare_factor_base,
    preflight_expression,
)


@pytest.fixture(scope="module")
def source_prices():
    days = []
    current = date(2019, 7, 1)
    while current <= date(2023, 5, 31):
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    bounds = get_layer_bounds("us")
    rows = []
    for i, day in enumerate(days):
        layer = next(k for k, (lo, hi) in bounds.items() if lo <= str(day) <= hi)
        for stock in range(80):
            price = (100 + stock * 2) * (1 + i * 0.0002 + math.sin(i / 17 + stock / 23) * 0.04)
            rows.append({
                "trade_date": day, "ts_code": f"S{stock:03d}",
                "open": price, "high": price * 1.01, "low": price * 0.99,
                "close": price * (1 + math.sin(i / 9 + stock) * 0.002),
                "vol": 1e7, "amount": 1e9, "univ_rank": stock + 1,
                "era": day.year * 10 + (1 if day.month <= 6 else 2), "layer": layer,
            })
    return pl.DataFrame(rows).sort("ts_code", "trade_date")


@pytest.fixture(scope="module")
def panel(source_prices):
    return with_market_calendar_labels(source_prices, [1, 5, 10, 20])


def _base(frame, *, expression="close", layers=None):
    with patch("backend.app.eval.harness._panel_frame", return_value=frame):
        base, fwd = _prepare_factor_base(expression, 80, 5, "synthetic", "us", layers or ["META_TRAIN"])
    return base, fwd


def _work(frame, *, expression="close", layers=None):
    base, fwd = _base(frame, expression=expression, layers=layers)
    return _prepare_direction_work(base, 5, "long_short", 1, evaluation_config("us")), fwd


def test_future_2023_prices_cannot_change_training_metrics_or_direction(source_prices, panel):
    changed = source_prices.with_columns([
        pl.when(pl.col("trade_date") >= date(2023, 1, 1))
        .then(pl.col(column) * (3 + pl.col("univ_rank") / 3))
        .otherwise(pl.col(column)).alias(column)
        for column in ["open", "high", "low", "close"]
    ])
    mutated = with_market_calendar_labels(changed, [1, 5, 10, 20])
    results = []
    for frame in [panel, mutated]:
        with patch("backend.app.eval.harness._panel_frame", return_value=frame):
            layers, cfg, discovery, _ = _evaluate_discovery_orientations(
                "ts_mean(close, 30)", 80, 5, "long_short", 1,
                "synthetic", 15, "us", None, DIRECTION_POLICY_BOTH,
            )
            results.append((layers, cfg, discovery))
    assert results[0] == results[1]


def test_purge_and_embargo_use_label_exit_and_market_calendar(panel):
    base, _ = _base(panel)
    actual = base.collect()
    layer_calendar = panel.filter(pl.col("layer") == "META_TRAIN")["trade_date"].unique().sort()
    assert actual["trade_date"].min() == layer_calendar[5]
    assert actual["_label_exit_date"].max() <= date(2022, 12, 31)
    assert actual["_label_entry_date"].min() > actual["trade_date"].min()


def test_validation_keeps_pre_boundary_rolling_warmup(panel):
    expression = "ts_mean(close, 30)"
    base, _ = _base(panel, expression=expression, layers=["META_HOLDOUT"])
    actual = base.collect().sort("trade_date", "ts_code")
    holdout_calendar = panel.filter(pl.col("layer") == "META_HOLDOUT")["trade_date"].unique().sort()
    expected_first = holdout_calendar[5]
    assert actual["trade_date"].min() == expected_first
    expected = parse(expression).apply(panel.lazy()).filter(pl.col("trade_date") == expected_first).sort("ts_code").collect()
    observed = actual.filter(pl.col("trade_date") == expected_first).sort("ts_code")
    assert observed["factor"].to_list() == expected["factor"].to_list()
    assert observed.height == 80


def test_factor_null_day_does_not_shift_later_rebalance_cohorts(panel):
    work, _ = _work(panel)
    original = work.collect().sort("trade_date", "ts_code")
    excluded_day = original["trade_date"].unique().sort()[3]
    mutated = panel.with_columns(
        pl.when(pl.col("trade_date") == excluded_day).then(None)
        .otherwise(pl.col("close")).alias("close")
    )
    work_after, _ = _work(mutated)
    actual = work_after.collect().sort("trade_date", "ts_code")
    fields = ["trade_date", "ts_code", "_date_seq", "_weight"]
    assert actual.select(fields).equals(original.filter(pl.col("trade_date") != excluded_day).select(fields))


def test_future_label_availability_cannot_select_or_reweight_stocks(panel):
    work, _ = _work(panel)
    original = work.collect().sort("trade_date", "ts_code")
    selected = original.filter(pl.col("_weight").abs() > 0).row(0, named=True)
    mutated = panel.with_columns(
        pl.when((pl.col("trade_date") == selected["trade_date"]) & (pl.col("ts_code") == selected["ts_code"]))
        .then(None).otherwise(pl.col("fwd_5")).alias("fwd_5")
    )
    work_after, fwd = _work(mutated)
    actual = work_after.collect().sort("trade_date", "ts_code")
    fields = ["trade_date", "ts_code", "_eligible_n", "_factor_pct", "_long_w", "_short_w", "_weight"]
    assert actual.select(fields).equals(original.select(fields))
    daily, _ = _direction_aggregates(work_after, fwd, "long_short", 1e6)
    assert daily.collect()["missing_selected_labels"].sum() == 1


def test_first_embargoed_cohort_has_only_initial_entry_turnover(panel):
    work, fwd = _work(panel)
    daily, _ = _direction_aggregates(work, fwd, "long_short", 1e6)
    first = daily.collect().sort("trade_date").row(0, named=True)
    # No prior sleeve exists at this independent evaluation layer. The first
    # rebalance must buy 1x and sell 1x, not invent another 2x liquidation.
    assert first["turnover"] == pytest.approx(2.0)


def test_signal_preflight_does_not_condition_admission_on_future_labels(panel):
    missing_returns = panel.with_columns(pl.lit(None, dtype=pl.Float64).alias("fwd_5"))
    outcomes = []
    for frame in [panel, missing_returns]:
        synthetic = SimpleNamespace(ensure_loaded=lambda frame=frame: frame)
        with patch("backend.app.eval.harness._panel_frame", return_value=frame), patch(
            "backend.app.eval.harness.PanelStore.get", return_value=synthetic
        ):
            outcomes.append(preflight_expression("close", 80, 5, "synthetic", "us", sample_modulus=1))
    for field in ["accepted", "failure_reasons", "eligible_observations",
                  "finite_observations", "finite_coverage", "usable_dates"]:
        assert outcomes[0][field] == outcomes[1][field]
