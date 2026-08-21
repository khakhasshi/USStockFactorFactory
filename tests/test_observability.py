import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl

from backend.app.data.panel import PanelStore
from backend.app.api.routes import _provider_observability
from backend.app.main import request_telemetry
from backend.app.observability import (
    AsyncTTLCache,
    RuntimeObservability,
    build_findings,
    build_slo,
    normalize_route,
    overall_health,
    redact_text,
    redact_value,
)
from backend.app.orchestrator import Engine
from backend.app.screener import _ScreenerCache
from starlette.requests import Request


class ObservabilityTests(unittest.TestCase):
    def test_route_cardinality_and_secret_redaction(self):
        self.assertEqual(
            normalize_route("/api/experiments/123/factors/456?full=true"),
            "/api/experiments/:id/factors/:id",
        )
        self.assertEqual(
            normalize_route(
                "/api/runs/123e4567-e89b-12d3-a456-426614174000/events"
            ),
            "/api/runs/:uuid/events",
        )
        redacted = redact_text(
            "Authorization=Bearer abc.def api_key=topsecret password=hunter2 "
            "postgresql://researcher:dsn-secret@localhost/factors"
        )
        self.assertNotIn("abc.def", redacted)
        self.assertNotIn("topsecret", redacted)
        self.assertNotIn("hunter2", redacted)
        self.assertNotIn("dsn-secret", redacted)
        nested = redact_value({
            "provider": {
                "api_key": "secret",
                "model": "test-model",
                "headers": {"Authorization": "Bearer hidden"},
            }
        })
        self.assertEqual(nested["provider"]["api_key"], "[REDACTED]")
        self.assertEqual(
            nested["provider"]["headers"]["Authorization"],
            "[REDACTED]",
        )
        self.assertEqual(nested["provider"]["model"], "test-model")

    def test_request_metrics_and_incident_are_bounded_and_safe(self):
        telemetry = RuntimeObservability()
        request_id, started = telemetry.begin_request()
        telemetry.finish_request(
            request_id=request_id,
            method="GET",
            path="/api/factors/123",
            status=500,
            started=started,
            error=ValueError("token=do-not-leak"),
        )
        snapshot = telemetry.request_snapshot(window_seconds=300)
        self.assertEqual(snapshot["in_flight"], 0)
        self.assertEqual(snapshot["lifetime"]["requests"], 1)
        self.assertEqual(snapshot["lifetime"]["server_errors"], 1)
        self.assertEqual(snapshot["routes"][0]["route"], "GET /api/factors/:id")
        self.assertEqual(snapshot["routes"][0]["errors"], 1)
        self.assertNotIn(
            "do-not-leak",
            snapshot["incidents"][0].get("error", ""),
        )
        self.assertNotIn("_epoch", snapshot["recent_requests"][0])
        process = telemetry.process_snapshot()
        self.assertGreater(process["pid"], 0)
        self.assertIn("asyncio_tasks", process)

    def test_rolling_window_latency_and_route_series_are_bounded(self):
        telemetry = RuntimeObservability(route_series_limit=8)
        for index in range(20):
            request_id, started = telemetry.begin_request()
            time.sleep(0.0001)
            telemetry.finish_request(
                request_id=request_id,
                method="GET",
                path=f"/api/unmatched/slug-{index}",
                status=200,
                started=started,
            )
        snapshot = telemetry.request_snapshot(window_seconds=300)
        self.assertEqual(snapshot["window"]["latency_sample_size"], 20)
        self.assertGreater(snapshot["window"]["latency_ms"]["p95"], 0)
        self.assertLessEqual(len(snapshot["routes"]), 8)
        self.assertIn(
            "GET /__overflow__",
            {route["route"] for route in snapshot["routes"]},
        )
        self.assertEqual(
            snapshot["lifetime"]["latency_scope"],
            "last_2000_requests",
        )

    def test_single_flight_cache_and_background_task_supervision(self):
        async def scenario():
            cache = AsyncTTLCache(ttl_seconds=60)
            calls = 0

            async def loader():
                nonlocal calls
                calls += 1
                return {"calls": calls}

            first = await cache.get("key", loader)
            second = await cache.get("key", loader)
            forced = await cache.get("key", loader, force=True)
            telemetry = RuntimeObservability()

            async def fail():
                raise RuntimeError("token=background-secret")

            task = telemetry.track_task(
                asyncio.create_task(fail(), name="test.failure")
            )
            await asyncio.gather(task, return_exceptions=True)
            await asyncio.sleep(0)
            return first, second, forced, calls, cache.stats(), telemetry

        first, second, forced, calls, stats, telemetry = asyncio.run(scenario())
        self.assertEqual(first, second)
        self.assertEqual(calls, 2)
        self.assertEqual(forced["calls"], 2)
        self.assertGreaterEqual(stats["hits"], 1)
        supervised = telemetry.background_snapshot()
        self.assertEqual(supervised["failed"], 1)
        self.assertNotIn(
            "background-secret",
            repr(supervised),
        )

    def test_incident_journal_survives_restart_and_redacts(self):
        with tempfile.TemporaryDirectory() as directory:
            journal = Path(directory) / "incidents.jsonl"

            async def first_process():
                telemetry = RuntimeObservability(journal_path=journal)
                await telemetry.start()
                request_id, started = telemetry.begin_request()
                telemetry.finish_request(
                    request_id=request_id,
                    method="GET",
                    path="/api/failure",
                    status=500,
                    started=started,
                    error=RuntimeError("api_key=never-write-this"),
                )
                await telemetry.stop()

            asyncio.run(first_process())
            raw = journal.read_text(encoding="utf-8")
            self.assertNotIn("never-write-this", raw)
            self.assertIn("service_started", raw)
            second = RuntimeObservability(journal_path=journal)
            second._start_journal()
            snapshot = second.journal_snapshot()
            second._stop_journal()
            self.assertGreaterEqual(len(snapshot["recent_events"]), 3)

    def test_uncaught_http_error_returns_correlation_id(self):
        request = Request({
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
            "scheme": "http",
            "path": "/api/test/123",
            "raw_path": b"/api/test/123",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 10010),
            "route": SimpleNamespace(path="/api/test/{test_id}"),
        })

        async def fail(_request):
            raise RuntimeError("simulated failure")

        with self.assertLogs("factorfactory.http", level="ERROR"):
            response = asyncio.run(request_telemetry(request, fail))
        payload = json.loads(response.body)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(payload["request_id"], response.headers["X-Request-ID"])
        self.assertIn("app;dur=", response.headers["Server-Timing"])
        self.assertNotIn("simulated failure", response.body.decode())

    def test_deterministic_findings_cover_db_http_panel_and_worker(self):
        snapshot = {
            "database": {
                "status": "error",
                "error": "connection refused",
                "pool": {"utilization": 0.9},
            },
            "requests": {
                "window": {
                    "seconds": 300,
                    "requests": 4,
                    "server_errors": 1,
                    "latency_ms": {"p95": 1300},
                },
                "event_loop": {"p95_lag_ms": 150},
            },
            "data": {
                "panels": [{
                    "market": "ashare",
                    "state": "error",
                    "load_error": "bad parquet",
                }],
            },
            "workers": [
                {
                    "experiment_id": 7,
                    "running": True,
                    "heartbeat_stale": True,
                    "heartbeat_age_seconds": 600,
                    "phase": "factor_evaluation",
                    "evaluation_deadline_exceeded": True,
                    "evaluation_elapsed_seconds": 601,
                    "evaluation_soft_deadline_seconds": 180,
                    "task_done": False,
                    "task_exception": None,
                },
                {
                    "experiment_id": 8,
                    "running": False,
                    "heartbeat_stale": False,
                    "phase": "failed",
                    "last_error": "evaluation crashed",
                    "task_done": True,
                    "task_exception": None,
                },
            ],
            "providers": {
                "inner_provider_configured": False,
                "outer_provider_configured": False,
            },
        }
        findings = build_findings(snapshot)
        codes = {row["code"] for row in findings}
        self.assertTrue({
            "database_unavailable",
            "database_pool_pressure",
            "recent_http_5xx",
            "slow_http_p95",
            "event_loop_lag",
            "panel_contract_error",
            "worker_heartbeat_stale",
            "worker_evaluation_slow",
            "worker_failed",
        }.issubset(codes))
        self.assertEqual(overall_health(findings), "unhealthy")

    def test_engine_diagnostics_exposes_phase_without_secrets(self):
        engine = Engine()
        engine.running = True
        engine.task_config = {
            "market": "us",
            "provider_token": "hidden",
        }
        engine._set_phase(
            "factor_evaluation",
            progress=True,
            current_task="liquid500_5d",
            current_budget_index=3,
            current_budget_total=10,
        )
        diagnostics = engine.diagnostics(include_logs=False)
        self.assertEqual(diagnostics["phase"], "factor_evaluation")
        self.assertEqual(diagnostics["current_task"], "liquid500_5d")
        self.assertEqual(diagnostics["current_budget_index"], 3)
        self.assertEqual(
            diagnostics["task_config"]["provider_token"],
            "[REDACTED]",
        )
        self.assertNotIn("logs", diagnostics)

    def test_provider_snapshot_exposes_routing_but_never_credential(self):
        diagnostics = _provider_observability({
            "inner_provider": "primary",
            "outer_provider": "primary",
            "providers": [{
                "name": "primary",
                "format": "openai",
                "model": "reasoning-model",
                "base_url": "https://llm.example.test/v1",
                "api_key": "sk-private-value",
            }],
        })
        provider = diagnostics["providers"][0]
        self.assertEqual(provider["endpoint_host"], "llm.example.test")
        self.assertTrue(provider["api_key_present"])
        self.assertNotIn("api_key", provider)
        self.assertNotIn(
            "sk-private-value",
            repr(diagnostics),
        )

    def test_cache_and_panel_inventory_counters(self):
        cache = _ScreenerCache(maxsize=2)
        self.assertIsNone(cache.get("missing"))
        cache.put("a", {"value": 1})
        cache.put("b", {"value": 2})
        self.assertEqual(cache.get("a"), {"value": 1})
        cache.put("c", {"value": 3})
        removed = cache.clear()
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["evictions"], 1)
        self.assertEqual(removed, 2)
        self.assertEqual(stats["entries"], 0)
        self.assertEqual(stats["clears"], 1)
        self.assertEqual(stats["cleared_entries"], 2)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "part-000.parquet"
            pl.DataFrame({
                "trade_date": [date(2026, 1, 2)],
                "ts_code": ["TEST"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.8],
                "close": [10.2],
                "vol": [1000.0],
                "amount": [10_000.0],
            }).write_parquet(source)
            panel = PanelStore(str(Path(directory) / "*.parquet"), "us")
            diagnostics = panel.diagnostics()
            self.assertEqual(diagnostics["state"], "cold")
            self.assertEqual(diagnostics["file_count"], 1)
            self.assertGreater(diagnostics["total_bytes"], 0)
            self.assertIsNotNone(diagnostics["identity"])
            self.assertEqual(diagnostics["schema_status"], "ok")
            self.assertEqual(diagnostics["missing_dsl_fields"], [])
            ashare = PanelStore(
                str(Path(directory) / "*.parquet"),
                "ashare",
            ).diagnostics()
            self.assertEqual(ashare["state"], "error")
            self.assertIn("pe_ttm", ashare["missing_dsl_fields"])

    def test_panel_hot_reload_detects_stable_change_and_swaps_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "part-000.parquet"

            def write_panel(trade_date: date, close: float) -> None:
                pl.DataFrame({
                    "trade_date": [trade_date],
                    "ts_code": ["TEST"],
                    "name": ["Test"],
                    "open": [close],
                    "high": [close + 0.5],
                    "low": [close - 0.5],
                    "close": [close],
                    "vol": [1000.0],
                    "amount": [close * 1000.0],
                }).write_parquet(source)

            write_panel(date(2026, 1, 2), 10.0)
            panel = PanelStore(str(Path(directory) / "*.parquet"), "us")
            _, dates_v1, identity_v1, generation_v1 = panel.read_snapshot()
            self.assertEqual(dates_v1[-1], date(2026, 1, 2))
            self.assertEqual(generation_v1, 1)

            write_panel(date(2026, 1, 5), 11.0)
            detected = panel.reload_if_changed(require_stable=True)
            self.assertEqual(detected["status"], "change_detected")
            self.assertEqual(panel.read_snapshot()[3], generation_v1)
            reloaded = panel.reload_if_changed(require_stable=True)
            self.assertEqual(reloaded["status"], "reloaded")

            frame_v2, dates_v2, identity_v2, generation_v2 = panel.read_snapshot()
            self.assertEqual(dates_v2[-1], date(2026, 1, 5))
            self.assertEqual(float(frame_v2["close"][0]), 11.0)
            self.assertNotEqual(identity_v1, identity_v2)
            self.assertEqual(generation_v2, 2)
            diagnostics = panel.diagnostics()
            self.assertFalse(diagnostics["stale"])
            self.assertEqual(diagnostics["reload_count"], 1)

    def test_panel_hot_reload_failure_keeps_previous_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "part-000.parquet"
            pl.DataFrame({
                "trade_date": [date(2026, 1, 2)],
                "ts_code": ["TEST"],
                "name": ["Test"],
                "open": [10.0],
                "high": [10.5],
                "low": [9.5],
                "close": [10.0],
                "vol": [1000.0],
                "amount": [10_000.0],
            }).write_parquet(source)
            panel = PanelStore(str(Path(directory) / "*.parquet"), "us")
            frame_v1, dates_v1, identity_v1, generation_v1 = panel.read_snapshot()

            pl.DataFrame({
                "trade_date": [date(2026, 1, 5)],
                "ts_code": ["BROKEN"],
            }).write_parquet(source)
            failed = panel.reload_if_changed()
            self.assertEqual(failed["status"], "reload_failed")
            self.assertTrue(failed["serving_continued"])

            frame_after, dates_after, identity_after, generation_after = (
                panel.read_snapshot()
            )
            self.assertIs(frame_after, frame_v1)
            self.assertEqual(dates_after, dates_v1)
            self.assertEqual(identity_after, identity_v1)
            self.assertEqual(generation_after, generation_v1)
            diagnostics = panel.diagnostics()
            self.assertTrue(diagnostics["stale"])
            self.assertIsNotNone(diagnostics["reload_error"])

    def test_panel_reads_continue_while_new_generation_builds(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "part-000.parquet"

            def write_panel(trade_date: date) -> None:
                pl.DataFrame({
                    "trade_date": [trade_date],
                    "ts_code": ["TEST"],
                    "name": ["Test"],
                    "open": [10.0],
                    "high": [10.5],
                    "low": [9.5],
                    "close": [10.0],
                    "vol": [1000.0],
                    "amount": [10_000.0],
                }).write_parquet(source)

            write_panel(date(2026, 1, 2))
            panel = PanelStore(str(Path(directory) / "*.parquet"), "us")
            old_frame, old_dates, _, old_generation = panel.read_snapshot()
            write_panel(date(2026, 1, 5))
            original_load = panel._load
            reload_started = threading.Event()
            allow_reload = threading.Event()

            def slow_load():
                reload_started.set()
                self.assertTrue(allow_reload.wait(timeout=2.0))
                return original_load()

            panel._load = slow_load
            result: dict = {}

            def run_reload() -> None:
                result.update(panel.reload_if_changed())

            thread = threading.Thread(target=run_reload)
            thread.start()
            self.assertTrue(reload_started.wait(timeout=2.0))
            serving_frame, serving_dates, _, serving_generation = (
                panel.read_snapshot()
            )
            self.assertIs(serving_frame, old_frame)
            self.assertEqual(serving_dates, old_dates)
            self.assertEqual(serving_generation, old_generation)
            self.assertEqual(panel.diagnostics()["state"], "reloading")
            allow_reload.set()
            thread.join(timeout=2.0)
            self.assertFalse(thread.is_alive())
            self.assertEqual(result["status"], "reloaded")
            self.assertEqual(panel.read_snapshot()[3], old_generation + 1)

    def test_slo_exposes_explicit_objectives(self):
        snapshot = {
            "requests": {
                "window": {
                    "requests": 20,
                    "error_rate": 0.0,
                    "latency_ms": {"p95": 30.0},
                },
                "event_loop": {"sample_count": 5, "p95_lag_ms": 2.0},
            },
            "database": {"status": "ok", "latency_ms": 8.0},
            "process": {
                "workspace_disk": {
                    "free_bytes": 40,
                    "total_bytes": 100,
                },
            },
            "engine": {"stale_heartbeat_count": 0},
            "data": {
                "panels": [{
                    "required": True,
                    "state": "cold",
                    "schema_status": "ok",
                }],
            },
        }
        slo = build_slo(snapshot)
        self.assertEqual(slo["status"], "pass")
        self.assertEqual(slo["failed"], 0)
        self.assertEqual(len(slo["objectives"]), 8)


class EngineHeartbeatTests(unittest.IsolatedAsyncioTestCase):
    async def test_blocking_evaluation_emits_heartbeats_and_runtime(self):
        engine = Engine()

        def slow_result():
            time.sleep(0.065)
            return {"ok": True}

        with patch.dict(os.environ, {
            "FF_EVALUATION_HEARTBEAT_SECONDS": "0.01",
            "FF_EVALUATION_SOFT_DEADLINE_SECONDS": "0.02",
        }):
            result = await engine._run_blocking_with_heartbeat(slow_result)

        self.assertEqual(result, {"ok": True})
        self.assertFalse(engine.status["evaluation_active"])
        self.assertGreaterEqual(
            engine.status["evaluation_heartbeat_count"],
            2,
        )
        self.assertTrue(engine.status["evaluation_deadline_exceeded"])
        self.assertGreaterEqual(
            engine.status["last_evaluation_duration_seconds"],
            0.05,
        )

    async def test_cancellation_waits_for_blocking_evaluation_to_drain(self):
        engine = Engine()
        completed: list[bool] = []

        def slow_result():
            time.sleep(0.06)
            completed.append(True)

        with patch.dict(os.environ, {
            "FF_EVALUATION_HEARTBEAT_SECONDS": "0.01",
            "FF_EVALUATION_SOFT_DEADLINE_SECONDS": "0.02",
        }):
            task = asyncio.create_task(
                engine._run_blocking_with_heartbeat(slow_result)
            )
            await asyncio.sleep(0.015)
            cancelled_at = time.monotonic()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        self.assertEqual(completed, [True])
        self.assertGreaterEqual(time.monotonic() - cancelled_at, 0.03)
        self.assertFalse(engine.status["evaluation_active"])
        self.assertEqual(
            engine.status["current_operation"],
            "factor_evaluation_draining",
        )


if __name__ == "__main__":
    unittest.main()
