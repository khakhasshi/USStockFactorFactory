from datetime import date, timedelta

import polars as pl
import pytest

from backend.app.backtest.engine import EventBacktestConfig, StepEventBacktester, _closed_trade_disclosure
from backend.app.backtest.stability import analyze_signal_diagnostics


def test_open_winning_inventory_is_not_a_winning_closed_trade():
    runner = StepEventBacktester(EventBacktestConfig(market="us", mode="long_only", top_fraction=.5))
    first = date(2024, 1, 2)
    for i in range(4):
        day = first + timedelta(days=i)
        rows = [{"trade_date": day, "ts_code": symbol, "name": symbol, "factor": float(j),
                 "univ_rank": j + 1, "raw_open": 10. + i, "raw_high": 11. + i,
                 "raw_low": 10. + i, "raw_close": 10.5 + i, "vol": 1e8,
                 "amount": 1e9, "_adv20_prev": 1e8, "adjustment_factor": 1.,
                 "can_buy_open_proxy": True, "can_sell_open_proxy": True}
                for j, symbol in enumerate(["AAA", "BBB", "CCC", "DDD"])]
        runner.step(trade_date=day, rows=rows, next_trade_date=day + timedelta(days=1) if i < 3 else None, rebalance=i == 0)
    stats = runner.result()["stats"]
    assert stats["open_positions"] > 0
    assert stats["open_unrealized_pnl_after_entry_fees"] > 0
    assert stats["closed_lots"] == 0
    assert stats["closed_lot_win_rate"] is None
    assert stats["win_rate"] is None
    assert stats["closed_lot_profit_factor"] is None
    assert stats["trade_statistics_status"] == "NO_CLOSED_LOTS"


def test_closed_partial_lots_and_financing_are_explicitly_named():
    result = _closed_trade_disclosure({"closed_trades": 3, "profit_factor": 2., "win_rate": 2 / 3,
                                       "borrow_cost": 40., "margin_interest": 10.}, -1000.)
    assert result["closed_lots"] == 3
    assert result["closed_lot_profit_factor"] == 2.
    assert result["unallocated_financing_cost"] == 50.
    assert result["open_unrealized_pnl_after_entry_fees"] == -1000.
    assert result["trade_statistics_excludes_open_positions"]
    assert result["trade_statistics_excludes_financing"]


def test_ic_does_not_read_labels_settling_after_event_end():
    first = date(2024, 1, 1)
    dates = [first + timedelta(days=i) for i in range(30)]
    rows = [{"trade_date": day, "factor": float(symbol), "univ_rank": symbol + 1,
             "fwd_5": float(symbol) * .01, "label_exit_date_5": day + timedelta(days=6)}
            for day in dates for symbol in range(50)]
    frame = pl.DataFrame(rows)
    result = analyze_signal_diagnostics(frame, horizon=5, universe_n=50, direction=1)
    path_dates = [date.fromisoformat(row["trade_date"]) for row in result["path"]]
    assert path_dates
    assert all(day + timedelta(days=6) <= dates[-1] for day in path_dates)
    mutated = frame.with_columns(pl.when(pl.col("label_exit_date_5") > dates[-1]).then(-pl.col("fwd_5")).otherwise(pl.col("fwd_5")).alias("fwd_5"))
    assert analyze_signal_diagnostics(mutated, horizon=5, universe_n=50, direction=1)["path"] == result["path"]
