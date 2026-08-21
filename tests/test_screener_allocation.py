import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import polars as pl
from fastapi import HTTPException

from backend.app.api import routes
from backend.app.models import ScreenerRun
from backend.app.portfolio_allocation import build_purchase_allocation


def _trading_dates(count: int, start: date = date(2024, 1, 2)) -> list[date]:
    output = []
    current = start
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current)
        current += timedelta(days=1)
    return output


def _panel(
    *,
    count: int = 190,
    symbols: tuple[str, ...] = ("LOW", "MID", "HIGH", "ALT"),
) -> tuple[pl.DataFrame, list[date]]:
    dates = _trading_dates(count)
    rng = np.random.default_rng(20260818)
    common = rng.normal(0.00015, 0.003, count)
    sigmas = (0.003, 0.008, 0.016, 0.011)
    rows = []
    for index, symbol in enumerate(symbols):
        returns = common + rng.normal(0.0, sigmas[index], count)
        price = 100.0
        for trading_date, daily_return in zip(dates, returns):
            price *= 1.0 + daily_return
            rows.append({
                "trade_date": trading_date,
                "ts_code": symbol,
                "close": price,
            })
    return pl.DataFrame(rows), dates


def _selected(symbols: tuple[str, ...] = ("LOW", "MID", "HIGH", "ALT")) -> list[dict]:
    return [
        {
            "ts_code": symbol,
            "name": symbol.title(),
            "side": "top",
            "side_rank": index + 1,
            "score": 96.0 - index * 3.0,
        }
        for index, symbol in enumerate(symbols)
    ]


class PortfolioAllocationTests(unittest.TestCase):
    def test_robust_weights_are_causal_normalised_and_capped(self):
        frame, dates = _panel()
        target = dates[160]
        kwargs = {
            "target_date": target,
            "selected": _selected(),
            "lookback": 120,
            "method": "robust_risk_budget",
            "max_weight": 0.30,
            "score_tilt": 0.35,
        }
        with_future = build_purchase_allocation(df=frame, **kwargs)
        without_future = build_purchase_allocation(
            df=frame.filter(pl.col("trade_date") <= target),
            **kwargs,
        )

        weights = [row["purchase_weight"] for row in with_future["allocations"]]
        self.assertAlmostEqual(sum(weights), 1.0, places=7)
        self.assertLessEqual(max(weights), 0.30 + 1e-8)
        self.assertEqual(
            with_future["allocations"],
            without_future["allocations"],
        )
        self.assertEqual(with_future["diagnostics"]["sample_end"], str(target))
        self.assertEqual(with_future["diagnostics"]["observations"], 120)
        self.assertTrue(with_future["diagnostics"]["converged"])

    def test_risk_budget_gives_lower_volatility_stock_more_capital(self):
        frame, dates = _panel(symbols=("LOW", "MID", "HIGH", "ALT"))
        result = build_purchase_allocation(
            df=frame,
            target_date=dates[-1],
            selected=_selected(),
            lookback=120,
            method="robust_risk_budget",
            max_weight=1.0,
            score_tilt=0.0,
        )
        weights = {row["ts_code"]: row["purchase_weight"] for row in result["allocations"]}
        self.assertGreater(weights["LOW"], weights["HIGH"])
        self.assertGreater(
            result["diagnostics"]["equal_weight_annualised_volatility"],
            result["diagnostics"]["portfolio_annualised_volatility"],
        )

    def test_single_stock_is_one_hundred_percent_and_cap_is_made_feasible(self):
        frame, dates = _panel(symbols=("LOW",))
        result = build_purchase_allocation(
            df=frame,
            target_date=dates[-1],
            selected=_selected(("LOW",)),
            max_weight=0.20,
        )
        self.assertEqual(result["allocations"][0]["purchase_weight"], 1.0)
        self.assertEqual(result["parameters"]["effective_max_weight"], 1.0)
        self.assertTrue(any("不可行" in warning for warning in result["warnings"]))

    def test_insufficient_joint_history_fails_closed(self):
        frame, dates = _panel(count=35)
        with self.assertRaisesRegex(ValueError, "共同有效收益样本"):
            build_purchase_allocation(
                df=frame,
                target_date=dates[-1],
                selected=_selected(),
                lookback=120,
            )


class _ReadSession:
    def __init__(self, row):
        self.row = row

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def scalar(self, query):
        return self.row


class _Frame:
    height = 1234


class _Panel:
    def read_snapshot(self):
        return _Frame(), (date(2024, 1, 3), date(2024, 1, 4)), "new", 2


def _stored_run() -> ScreenerRun:
    return ScreenerRun(
        id=88,
        experiment_id=7,
        schema_version=routes.SCREENER_RUN_SCHEMA_VERSION,
        market="us",
        portfolio_mode="long_only",
        target_date=date(2024, 1, 4),
        requested_date=None,
        direction="top",
        panel_identity="us:old:g1:old:1200:2024-01-04",
        request_spec={"panel_glob": "panel/*.parquet"},
        result_snapshot={
            "stocks": [
                {"ts_code": "AAA", "name": "Alpha", "score": 95.0, "side": "top", "side_rank": 1},
                {"ts_code": "BBB", "name": "Beta", "score": 90.0, "side": "top", "side_rank": 2},
            ]
        },
        factor_count=1,
        eligible_count=100,
        result_count=2,
        cache_hit=False,
        status="done",
        error="",
        created_at=datetime(2026, 8, 18, 12, 0, 0),
    )


class ScreenerAllocationRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_route_binds_symbols_to_same_task_snapshot(self):
        allocation = {
            "protocol": "test",
            "method": "robust_risk_budget",
            "target_date": "2024-01-04",
            "selection_count": 1,
            "allocations": [],
            "parameters": {},
            "diagnostics": {},
            "warnings": [],
        }
        builder = MagicMock(return_value=allocation)
        with (
            patch.object(routes, "_experiment_context", AsyncMock(return_value=(7, {}))),
            patch.object(routes, "SessionLocal", return_value=_ReadSession(_stored_run())),
            patch.object(routes.PanelStore, "get", return_value=_Panel()),
            patch.object(routes, "build_purchase_allocation", builder),
        ):
            response = await routes.allocate_screener_selection(
                routes.ScreenerAllocationReq(
                    experiment_id=7,
                    run_id=88,
                    symbols=["BBB"],
                )
            )

        self.assertEqual(builder.call_args.kwargs["selected"][0]["ts_code"], "BBB")
        self.assertEqual(response["run_id"], 88)
        self.assertTrue(response["panel_changed"])
        self.assertTrue(any("面板标识" in warning for warning in response["warnings"]))

    async def test_route_rejects_symbol_outside_frozen_candidate_list(self):
        with (
            patch.object(routes, "_experiment_context", AsyncMock(return_value=(7, {}))),
            patch.object(routes, "SessionLocal", return_value=_ReadSession(_stored_run())),
        ):
            with self.assertRaises(HTTPException) as raised:
                await routes.allocate_screener_selection(
                    routes.ScreenerAllocationReq(
                        experiment_id=7,
                        run_id=88,
                        symbols=["NOT_IN_SNAPSHOT"],
                    )
                )
        self.assertEqual(raised.exception.status_code, 400)

    def test_frontend_exposes_arbitrary_selection_and_allocation_surface(self):
        source = (
            Path(__file__).resolve().parents[1] / "frontend" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn('data-testid="screener-allocation"', source)
        self.assertIn('api("/screener/allocate"', source)
        self.assertIn("toggleStockSelection", source)
        self.assertIn("计算购买比例", source)


if __name__ == "__main__":
    unittest.main()
