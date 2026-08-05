import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from backend.app.data.panel import PanelStore
from backend.app.api.routes import _provider_observability
from backend.app.main import request_telemetry
from backend.app.observability import (
    RuntimeObservability,
    build_findings,
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
            "Authorization=Bearer abc.def api_key=topsecret password=hunter2"
        )
        self.assertNotIn("abc.def", redacted)
        self.assertNotIn("topsecret", redacted)
        self.assertNotIn("hunter2", redacted)
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
                },
                "lifetime": {"latency_ms": {"p95": 1300}},
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
            "panel_load_error",
            "worker_heartbeat_stale",
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
        stats = cache.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["evictions"], 1)
        self.assertEqual(stats["entries"], 2)

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "part-000.parquet"
            source.write_bytes(b"inventory-only")
            panel = PanelStore(str(Path(directory) / "*.parquet"), "us")
            diagnostics = panel.diagnostics()
            self.assertEqual(diagnostics["state"], "cold")
            self.assertEqual(diagnostics["file_count"], 1)
            self.assertEqual(diagnostics["total_bytes"], len(b"inventory-only"))
            self.assertIsNotNone(diagnostics["identity"])


if __name__ == "__main__":
    unittest.main()
