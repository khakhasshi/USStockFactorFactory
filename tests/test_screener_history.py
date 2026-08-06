import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from backend.app.api import routes
from backend.app.models import ScreenerRun


def _screen_result() -> dict:
    return {
        "stocks": [
            {
                "rank": 1,
                "side": "top",
                "side_rank": 1,
                "head_rank": 1,
                "tail_rank": 500,
                "ts_code": "AAA",
                "name": "Alpha",
                "score": 98.25,
                "components": [
                    {
                        "expression": "rank(close)",
                        "direction": -1,
                        "weight": 1.0,
                        "value": 12.3,
                        "rank_score": 98.25,
                        "contribution": 98.25,
                    }
                ],
            }
        ],
        "eligible_count": 500,
        "history_start": "2023-12-20",
        "required_history": 3,
        "ranking_semantics": {},
        "performance": {
            "engine": "polars_single_lazy_plan_v3_tail_rank",
            "cache_hit": False,
            "elapsed_ms": 12.5,
        },
    }


def _stored_run(
    *,
    run_id: int = 91,
    experiment_id: int = 7,
) -> ScreenerRun:
    result = {
        "experiment_id": experiment_id,
        "date": "2024-01-04",
        "requested_date": "2024-01-05",
        "date_adjusted": True,
        "universe_n": 500,
        "top_n": 20,
        "factor_count": 1,
        "expression_mode": True,
        "market": "us",
        "portfolio_mode": "long_short",
        "direction": "top",
        **_screen_result(),
    }
    return ScreenerRun(
        id=run_id,
        experiment_id=experiment_id,
        schema_version=routes.SCREENER_RUN_SCHEMA_VERSION,
        market="us",
        portfolio_mode="long_short",
        target_date=date(2024, 1, 4),
        requested_date=date(2024, 1, 5),
        direction="top",
        panel_identity="us:test-panel:1000:2024-01-04",
        request_spec={
            "expression_mode": True,
            "universe_n": 500,
            "top_n": 20,
            "factors": [
                {
                    "expression": "rank(close)",
                    "weight": 1.0,
                    "direction": -1,
                }
            ],
        },
        result_snapshot=result,
        factor_count=1,
        eligible_count=500,
        result_count=1,
        cache_hit=False,
        elapsed_ms=18.75,
        status="done",
        error="",
        created_at=datetime(2026, 8, 6, 12, 30, 15),
    )


class _Frame:
    height = 1000


class _Panel:
    trading_dates = [
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 4),
    ]

    def ensure_loaded(self):
        return _Frame()


class _WriteSession:
    def __init__(self):
        self.added = []
        self.commits = 0
        self.refreshes = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    def add(self, row):
        row.id = 901
        row.created_at = datetime(2026, 8, 6, 13, 15, 0)
        self.added.append(row)

    async def commit(self):
        self.commits += 1

    async def refresh(self, row):
        self.refreshes += 1


class _ReadSession:
    def __init__(self, *, rows=None, scalar_value=None):
        self.rows = list(rows or [])
        self.scalar_value = scalar_value
        self.scalar_queries = []
        self.scalars_queries = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

    async def scalar(self, query):
        self.scalar_queries.append(query)
        return self.scalar_value

    async def scalars(self, query):
        self.scalars_queries.append(query)

        class _Rows:
            def __init__(self, values):
                self.values = values

            def all(self):
                return self.values

        return _Rows(self.rows)


class ScreenerHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_screener_binds_explicit_task_and_persists_full_snapshot(self):
        session = _WriteSession()
        context = AsyncMock(return_value=(
            42,
            {
                "market": "us",
                "portfolio_mode": "long_short",
                "panel_glob": "test-panel/*.parquet",
                "evaluation_protocol": "v4.0",
            },
        ))
        screened = _screen_result()
        with (
            patch.object(routes, "_experiment_context", context),
            patch.object(routes.PanelStore, "get", return_value=_Panel()),
            patch.object(
                routes,
                "screen_cross_section",
                return_value=screened,
            ) as screen,
            patch.object(routes, "SessionLocal", return_value=session),
        ):
            response = await routes.screener(routes.ScreenerReq(
                experiment_id=42,
                expression="rank(close)",
                expression_direction=-1,
                date="2024-01-04",
                universe_n=500,
                top_n=20,
                direction="top",
            ))

        context.assert_awaited_once_with(42)
        self.assertEqual(screen.call_args.kwargs["factors"], [{
            "expression": "rank(close)",
            "weight": 1.0,
            "direction": -1,
        }])
        self.assertEqual(session.commits, 1)
        self.assertEqual(session.refreshes, 1)
        self.assertEqual(len(session.added), 1)
        stored = session.added[0]
        self.assertIsInstance(stored, ScreenerRun)
        self.assertEqual(stored.experiment_id, 42)
        self.assertEqual(stored.market, "us")
        self.assertEqual(stored.target_date, date(2024, 1, 4))
        self.assertEqual(
            stored.request_spec["factors"][0]["direction"],
            -1,
        )
        self.assertEqual(
            stored.result_snapshot["stocks"][0]["ts_code"],
            "AAA",
        )
        self.assertEqual(response["experiment_id"], 42)
        self.assertEqual(response["run_id"], 901)
        self.assertEqual(response["recorded_at"], "2026-08-06T13:15:00")

    async def test_history_list_is_task_scoped_and_returns_lightweight_rows(self):
        row = _stored_run(experiment_id=7)
        session = _ReadSession(rows=[row], scalar_value=1)
        context = AsyncMock(return_value=(
            7,
            {"market": "us", "portfolio_mode": "long_short"},
        ))
        with (
            patch.object(routes, "_experiment_context", context),
            patch.object(routes, "SessionLocal", return_value=session),
        ):
            response = await routes.list_screener_runs(
                experiment_id=7,
                limit=30,
                offset=0,
            )

        context.assert_awaited_once_with(7)
        compiled = session.scalars_queries[0].compile()
        self.assertIn(7, compiled.params.values())
        self.assertIn(
            "screener_runs.experiment_id",
            str(session.scalars_queries[0]),
        )
        self.assertEqual(response["experiment_id"], 7)
        self.assertEqual(response["total"], 1)
        summary = response["runs"][0]
        self.assertEqual(summary["id"], row.id)
        self.assertEqual(summary["stock_preview"][0]["ts_code"], "AAA")
        self.assertEqual(
            summary["factor_preview"][0]["expression"],
            "rank(close)",
        )
        self.assertNotIn("result", summary)
        self.assertNotIn("request_spec", summary)

    async def test_history_detail_requires_same_task_and_returns_full_snapshot(self):
        row = _stored_run(run_id=92, experiment_id=7)
        session = _ReadSession(scalar_value=row)
        with (
            patch.object(
                routes,
                "_experiment_context",
                AsyncMock(return_value=(7, {})),
            ),
            patch.object(routes, "SessionLocal", return_value=session),
        ):
            response = await routes.screener_run_detail(
                run_id=92,
                experiment_id=7,
            )

        compiled = session.scalar_queries[0].compile()
        self.assertIn(7, compiled.params.values())
        self.assertIn(92, compiled.params.values())
        detail = response["run"]
        self.assertEqual(detail["experiment_id"], 7)
        self.assertEqual(detail["request_spec"]["universe_n"], 500)
        self.assertEqual(detail["result"]["run_id"], 92)
        self.assertEqual(detail["result"]["stocks"][0]["ts_code"], "AAA")

        missing = _ReadSession(scalar_value=None)
        with (
            patch.object(
                routes,
                "_experiment_context",
                AsyncMock(return_value=(8, {})),
            ),
            patch.object(routes, "SessionLocal", return_value=missing),
        ):
            with self.assertRaises(HTTPException) as raised:
                await routes.screener_run_detail(
                    run_id=92,
                    experiment_id=8,
                )
        self.assertEqual(raised.exception.status_code, 404)

    def test_frontend_sends_task_id_and_exposes_history_surface(self):
        source = (
            Path(__file__).resolve().parents[1] / "frontend" / "app.js"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "const experimentId = Number(appState.experimentId) || undefined",
            source,
        )
        self.assertIn("experiment_id: experimentId", source)
        self.assertIn('data-testid="screener-history"', source)
        self.assertIn('api(`/screener/runs${query}`)', source)
        self.assertIn("async function openHistory(runRow)", source)


if __name__ == "__main__":
    unittest.main()
