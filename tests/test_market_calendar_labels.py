from datetime import date, timedelta

import polars as pl
import pytest

from backend.app.data.panel import PanelStore, with_market_calendar_labels


def _prices():
    # The second security is suspended on market sessions 2 and 4. Weekends
    # are omitted deliberately: horizons count sessions, not calendar days.
    days = [date(2024, 1, 4), date(2024, 1, 5), date(2024, 1, 8),
            date(2024, 1, 9), date(2024, 1, 10), date(2024, 1, 11),
            date(2024, 1, 12), date(2024, 1, 15)]
    records = [
        {"trade_date": day, "ts_code": code, "open": float(100 + i * 10)}
        for code in ["A", "B"]
        for i, day in enumerate(days)
        if code == "A" or i not in {2, 4}
    ]
    return pl.DataFrame(records), days


def test_labels_use_market_sessions_not_next_security_observations():
    prices, days = _prices()
    result = with_market_calendar_labels(prices, [1, 2, 5])
    b0 = result.filter((pl.col("ts_code") == "B") & (pl.col("trade_date") == days[0])).row(0, named=True)
    assert b0["label_entry_date"] == days[1]
    assert b0["label_exit_date_1"] == days[2]
    assert b0["fwd_1"] is None  # Must not use B's next observed date Jan 9.
    assert b0["label_exit_date_2"] == days[3]
    assert b0["fwd_2"] == pytest.approx(130 / 110 - 1)
    # An intermediate suspension does not invalidate exact available endpoints.
    assert b0["label_exit_date_5"] == days[6]
    assert b0["fwd_5"] == pytest.approx(160 / 110 - 1)
    b1 = result.filter((pl.col("ts_code") == "B") & (pl.col("trade_date") == days[1])).row(0, named=True)
    assert b1["label_entry_date"] == days[2]
    assert b1["fwd_1"] is None  # Entry missing, even though exit is available.


def test_calendar_labels_keep_input_order_and_terminal_dates_are_null():
    prices, days = _prices()
    prices = prices.reverse()
    result = with_market_calendar_labels(prices, [1])
    assert result.select("ts_code", "trade_date").equals(prices.select("ts_code", "trade_date"))
    last = result.filter(pl.col("trade_date") == days[-1])
    assert last["label_entry_date"].null_count() == last.height
    assert last["label_exit_date_1"].null_count() == last.height
    assert last["fwd_1"].null_count() == last.height
    assert last["market_session_index"].to_list() == [7, 7]


def test_forward_returns_do_not_accept_zero_negative_or_nonfinite_opens():
    days = [date(2024, 1, 1) + timedelta(days=i) for i in range(4)]
    for invalid in [0.0, -1.0, float("nan"), float("inf"), None]:
        frame = pl.DataFrame({"trade_date": days, "ts_code": ["A"] * 4,
                              "open": [100.0, invalid, 105.0, 110.0]})
        result = with_market_calendar_labels(frame, [1])
        assert result["fwd_1"][0] is None


def test_duplicate_endpoint_quotes_fail_closed():
    prices, _ = _prices()
    with pytest.raises(pl.exceptions.ComputeError, match="validation"):
        with_market_calendar_labels(pl.concat([prices, prices.head(1)]), [1])


def test_panel_store_materializes_calendar_labels_and_policy(tmp_path):
    prices, _ = _prices()
    prices = prices.with_columns(
        pl.col("open").alias("high"), pl.col("open").alias("low"),
        pl.col("open").alias("close"), pl.lit(1000.0).alias("vol"),
        (pl.col("open") * 1000.0).alias("amount"),
    )
    path = tmp_path / "prices.parquet"
    prices.write_parquet(path)
    store = PanelStore(str(path), "us", ["open", "high", "low", "close", "vol", "amount"])
    frame = store.ensure_loaded()
    assert "market_session_index" in frame.columns
    assert "label_exit_date_20" in frame.columns
    assert store.summary()["forward_label_policy"] == "exact_market_session_forward_open_v2"
    assert store.trading_dates == sorted(prices["trade_date"].unique().to_list())


def test_negative_horizon_is_rejected():
    prices, _ = _prices()
    with pytest.raises(ValueError, match="positive"):
        with_market_calendar_labels(prices, [0])
