from __future__ import annotations

from datetime import date, timedelta

import polars as pl
import pytest

from backend.app.backtest.stability import (
    analyze_return_stability,
    analyze_signal_diagnostics,
    factor_performance_correlation,
    monte_carlo_analysis,
)


def _weekdays(start: date, count: int) -> list[date]:
    output = []
    current = start
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current)
        current += timedelta(days=1)
    return output


def _ledger_and_sleeves() -> tuple[list[dict], list[dict]]:
    dates = _weekdays(date(2021, 1, 4), 756)
    first_nlv = 500_000.0
    second_nlv = 500_000.0
    first_values = []
    second_values = []
    daily = []
    total_previous = 1_000_000.0
    for trade_date in dates:
        first_return = 0.001 if trade_date.year == 2021 else -0.0006
        second_return = -0.0004 if trade_date.year == 2021 else 0.0008
        first_nlv *= 1.0 + first_return
        second_nlv *= 1.0 + second_return
        total = first_nlv + second_nlv
        daily.append({
            "trade_date": trade_date,
            "daily_return": total / total_previous - 1.0,
            "turnover": 0.02,
        })
        first_values.append(first_nlv)
        second_values.append(second_nlv)
        total_previous = total
    sleeves = [
        {"factor_id": "F01", "name": "A", "normalized_weight": 0.5, "initial_capital": 500_000.0, "nlv": first_values},
        {"factor_id": "F02", "name": "B", "normalized_weight": 0.5, "initial_capital": 500_000.0, "nlv": second_values},
    ]
    return daily, sleeves


def test_time_slices_and_material_regime_reversal_are_ledger_based():
    daily, sleeves = _ledger_and_sleeves()
    result = analyze_return_stability(
        daily,
        total_initial_capital=1_000_000.0,
        sleeves=sleeves,
    )
    assert result["status"] == "OK"
    assert {row["period"] for row in result["annual"]} >= {"2021", "2022"}
    assert result["latest_rolling"]["12m"] is not None
    assert result["latest_rolling"]["24m"] is not None
    events = result["regime_reversal"]["events"]
    assert {row["factor_id"] for row in events} == {"F01", "F02"}
    assert result["regime_reversal"]["material_reversal_count"] == 2
    for period in result["annual"]:
        assert abs(period["contribution_reconciliation_error"]) <= 1e-8


def test_causal_ic_reports_pearson_rank_and_annualised_ir():
    dates = _weekdays(date(2022, 1, 3), 130)
    rows = []
    for day_index, trade_date in enumerate(dates):
        for symbol_index in range(100):
            signal = float(symbol_index) + day_index * 1e-4
            rows.append({
                "trade_date": trade_date,
                "ts_code": f"S{symbol_index:03d}",
                "univ_rank": symbol_index + 1,
                "factor": signal,
                "fwd_5": signal / 1000.0,
            })
    result = analyze_signal_diagnostics(
        pl.DataFrame(rows),
        horizon=5,
        universe_n=100,
        direction=1,
    )
    assert result["status"] == "OK"
    assert result["overall"]["ic_mean"] == pytest.approx(1.0)
    assert result["overall"]["rank_ic_mean"] == pytest.approx(1.0)
    assert result["overall"]["n_dates"] >= 20
    reverse = analyze_signal_diagnostics(
        pl.DataFrame(rows), horizon=5, universe_n=100, direction=-1
    )
    assert reverse["overall"]["rank_ic_mean"] == pytest.approx(-1.0)


def test_moving_block_monte_carlo_is_deterministic_and_joint_for_sleeves():
    daily, sleeves = _ledger_and_sleeves()
    first = monte_carlo_analysis(
        daily, simulations=300, block_size_sessions=20, seed=7, sleeves=sleeves
    )
    second = monte_carlo_analysis(
        daily, simulations=300, block_size_sessions=20, seed=7, sleeves=sleeves
    )
    assert first == second
    assert first["status"] == "OK"
    assert first["terminal_return"]["p05"] <= first["terminal_return"]["p50"]
    assert first["max_drawdown"]["p50"] <= first["max_drawdown"]["p95"]
    assert len(first["sleeves"]) == 2


def test_factor_performance_correlation_separates_realised_and_ic_paths():
    daily, sleeves = _ledger_and_sleeves()
    stability = analyze_return_stability(
        daily, total_initial_capital=1_000_000.0, sleeves=sleeves
    )
    diagnostics = {
        "factors": [
            {"factor_id": "F01", "diagnostics": {"path": [
                {"trade_date": "2021-01-04", "rank_ic": 0.1},
                {"trade_date": "2021-01-11", "rank_ic": 0.2},
                {"trade_date": "2021-01-18", "rank_ic": 0.3},
            ]}},
            {"factor_id": "F02", "diagnostics": {"path": [
                {"trade_date": "2021-01-04", "rank_ic": -0.1},
                {"trade_date": "2021-01-11", "rank_ic": -0.2},
                {"trade_date": "2021-01-18", "rank_ic": -0.3},
            ]}},
        ]
    }
    result = factor_performance_correlation(
        dates=[row["trade_date"] for row in daily],
        sleeves=sleeves,
        signal_diagnostics=diagnostics,
        stability_analysis=stability,
    )
    assert result["status"] == "OK"
    assert len(result["pairs"]) == 1
    assert result["pairs"][0]["rank_ic_path_correlation"] == pytest.approx(-1.0)
    assert result["matrices"]["daily_return"][0][0] == 1.0
