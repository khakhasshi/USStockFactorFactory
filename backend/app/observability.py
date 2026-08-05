"""Low-overhead, secret-safe runtime telemetry for the local research service."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import platform
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import Counter, OrderedDict, deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Awaitable, Callable

from .config import (
    ALLOW_REMOTE_UNAUTHENTICATED,
    HOST,
    PORT,
    is_loopback_host,
)

try:
    import psutil
except ImportError:  # pragma: no cover - run.sh installs it; fallback keeps boot safe
    psutil = None


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INCIDENT_JOURNAL = ROOT / "var" / "observability" / "incidents.jsonl"
MAX_ROUTE_SERIES = 128
RECENT_REQUEST_CAPACITY = 5000
_NUMERIC_SEGMENT = re.compile(r"/\d+(?=/|$)")
_UUID_SEGMENT = re.compile(
    r"/[0-9a-fA-F]{8}-[0-9a-fA-F-]{27,36}(?=/|$)"
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(
        r"(?i)\b([a-z][a-z0-9+.-]*://[^:/\s]+:)([^@\s]+)(@)"
    ),
    re.compile(
        r"(?i)\b(api[_-]?key|authorization|password|secret|token)"
        r"(\s*[=:]\s*)[^\s,;}\]]+"
    ),
)
_SECRET_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "credential",
    "password",
    "secret",
    "token",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def redact_text(value: object, limit: int = 1200) -> str:
    """Redact common credentials before text reaches telemetry or the UI."""
    text = str(value)
    text = _SECRET_PATTERNS[0].sub("Bearer [REDACTED]", text)
    text = _SECRET_PATTERNS[1].sub("sk-[REDACTED]", text)
    text = _SECRET_PATTERNS[2].sub(r"\1[REDACTED]\3", text)
    text = _SECRET_PATTERNS[3].sub(r"\1\2[REDACTED]", text)
    return text[:limit]


def redact_value(value: Any, *, depth: int = 0) -> Any:
    """Recursively sanitize telemetry payloads while keeping useful structure."""
    if depth >= 8:
        return "[MAX_DEPTH]"
    if isinstance(value, dict):
        sanitized = {}
        for raw_key, item in list(value.items())[:200]:
            key = str(raw_key)
            lowered = key.lower()
            if any(part in lowered for part in _SECRET_KEY_PARTS):
                sanitized[key] = "[REDACTED]" if item else None
            else:
                sanitized[key] = redact_value(item, depth=depth + 1)
        return sanitized
    if isinstance(value, (list, tuple, set)):
        return [redact_value(item, depth=depth + 1) for item in list(value)[:200]]
    if isinstance(value, str):
        return redact_text(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return redact_text(value)


def normalize_route(path: str) -> str:
    """Bound route cardinality without losing the endpoint's troubleshooting value."""
    normalized = _UUID_SEGMENT.sub("/:uuid", path.split("?", 1)[0])
    return _NUMERIC_SEGMENT.sub("/:id", normalized)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("fastapi", "uvicorn", "sqlalchemy", "asyncpg", "polars", "psutil"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    return versions


def _git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=ROOT,
                capture_output=True,
                check=False,
                text=True,
                timeout=0.5,
            )
            return completed.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    commit = run("rev-parse", "HEAD")
    branch = run("branch", "--show-current")
    dirty = bool(run("status", "--porcelain"))
    return {
        "commit": commit,
        "commit_short": commit[:12] if commit else "unknown",
        "branch": branch or "detached",
        "dirty_at_start": dirty,
    }


class _TelemetryLogHandler(logging.Handler):
    def __init__(self, owner: "RuntimeObservability") -> None:
        super().__init__(level=logging.INFO)
        self.owner = owner

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.owner.record_log(
                logger=record.name,
                level=record.levelname.lower(),
                message=redact_text(record.getMessage()),
            )
        except Exception:  # noqa: BLE001
            self.handleError(record)


class RuntimeObservability:
    """Bounded in-memory telemetry; no request body or secret is retained."""

    def __init__(
        self,
        *,
        route_series_limit: int = MAX_ROUTE_SERIES,
        recent_request_capacity: int = RECENT_REQUEST_CAPACITY,
        journal_path: Path | None = None,
    ) -> None:
        self.started_epoch = time.time()
        self.started_monotonic = time.monotonic()
        self.started_at = utc_now()
        self.deployment = _git_metadata()
        self.package_versions = _package_versions()
        self._route_series_limit = max(8, int(route_series_limit))
        self._recent_request_capacity = max(100, int(recent_request_capacity))
        self._lock = threading.Lock()
        self._in_flight = 0
        self._total_requests = 0
        self._server_errors = 0
        self._client_errors = 0
        self._slow_requests = 0
        self._status_classes: Counter[str] = Counter()
        self._durations: deque[float] = deque(maxlen=2000)
        self._routes: dict[str, dict[str, Any]] = {}
        self._recent_requests: deque[dict] = deque(
            maxlen=self._recent_request_capacity
        )
        self._incidents: deque[dict] = deque(maxlen=120)
        self._logs: deque[dict] = deque(maxlen=300)
        self._loop_lag_ms: deque[float] = deque(maxlen=300)
        self._loop_monitor: asyncio.Task | None = None
        self._log_handler: _TelemetryLogHandler | None = None
        self._process = psutil.Process(os.getpid()) if psutil else None
        self._background_tasks: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._journal_path = journal_path
        self._journal_handler: RotatingFileHandler | None = None
        self._journal_logger: logging.Logger | None = None
        self._journal_events: deque[dict[str, Any]] = deque(maxlen=120)
        self._journal_error: str | None = None

    async def start(self) -> None:
        self._start_journal()
        if self._log_handler is None:
            self._log_handler = _TelemetryLogHandler(self)
            logging.getLogger().addHandler(self._log_handler)
        if self._process:
            self._process.cpu_percent(interval=None)
        if self._loop_monitor is None or self._loop_monitor.done():
            self._loop_monitor = asyncio.create_task(
                self._monitor_event_loop(),
                name="observability.event_loop_lag",
            )
        self._persist_event({
            "kind": "service_started",
            "pid": os.getpid(),
            "commit": self.deployment.get("commit_short"),
            "branch": self.deployment.get("branch"),
        })

    async def stop(self) -> None:
        self._persist_event({
            "kind": "service_stopped",
            "pid": os.getpid(),
            "uptime_seconds": round(
                max(0.0, time.monotonic() - self.started_monotonic),
                3,
            ),
        })
        if self._loop_monitor and not self._loop_monitor.done():
            self._loop_monitor.cancel()
            try:
                await self._loop_monitor
            except asyncio.CancelledError:
                pass
        self._loop_monitor = None
        if self._log_handler is not None:
            logging.getLogger().removeHandler(self._log_handler)
            self._log_handler = None
        self._stop_journal()

    def _start_journal(self) -> None:
        if self._journal_path is None or self._journal_handler is not None:
            return
        try:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal_events.clear()
            if self._journal_path.exists():
                with self._journal_path.open("r", encoding="utf-8") as stream:
                    for line in deque(stream, maxlen=self._journal_events.maxlen):
                        try:
                            value = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(value, dict):
                            self._journal_events.append(redact_value(value))
            logger = logging.getLogger(
                f"factorfactory.incident_journal.{id(self)}"
            )
            logger.setLevel(logging.INFO)
            logger.propagate = False
            handler = RotatingFileHandler(
                self._journal_path,
                maxBytes=2_000_000,
                backupCount=4,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
            self._journal_logger = logger
            self._journal_handler = handler
            self._journal_error = None
        except (OSError, ValueError) as exc:
            self._journal_error = redact_text(exc, 400)

    def _stop_journal(self) -> None:
        if self._journal_handler is None or self._journal_logger is None:
            return
        try:
            self._journal_logger.removeHandler(self._journal_handler)
            self._journal_handler.close()
        finally:
            self._journal_handler = None
            self._journal_logger = None

    def _persist_event(self, payload: dict[str, Any]) -> None:
        if self._journal_logger is None:
            return
        row = redact_value({"at": utc_now(), **payload})
        try:
            self._journal_logger.info(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"))
            )
            with self._lock:
                self._journal_events.append(row)
        except (OSError, ValueError, TypeError) as exc:
            self._journal_error = redact_text(exc, 400)

    def journal_snapshot(self) -> dict:
        with self._lock:
            events = list(reversed([dict(row) for row in self._journal_events]))
        size = 0
        if self._journal_path is not None:
            try:
                size = self._journal_path.stat().st_size
            except OSError:
                pass
        return {
            "enabled": self._journal_path is not None,
            "path": str(self._journal_path) if self._journal_path else None,
            "current_bytes": size,
            "rotation_bytes": 2_000_000,
            "retained_files": 5,
            "error": self._journal_error,
            "recent_events": events[:80],
        }

    def track_task(
        self,
        task: asyncio.Task,
        *,
        name: str | None = None,
    ) -> asyncio.Task:
        """Supervise a background task and surface silent startup failures."""
        task_name = name or task.get_name() or "background"
        key = f"{task_name}:{id(task)}"
        row = {
            "name": task_name,
            "state": "running",
            "started_at": utc_now(),
            "finished_at": None,
            "exception": None,
        }
        with self._lock:
            self._background_tasks[key] = row
            while len(self._background_tasks) > 40:
                self._background_tasks.popitem(last=False)

        def completed(done: asyncio.Task) -> None:
            state = "completed"
            exception = None
            if done.cancelled():
                state = "cancelled"
            else:
                try:
                    error = done.exception()
                except asyncio.CancelledError:
                    state = "cancelled"
                    error = None
                if error is not None:
                    state = "failed"
                    exception = redact_text(error, 800)
            with self._lock:
                current = self._background_tasks.get(key)
                if current is not None:
                    current.update({
                        "state": state,
                        "finished_at": utc_now(),
                        "exception": exception,
                    })
                    if exception:
                        self._incidents.append({
                            "kind": "background_task_failure",
                            "at": current["finished_at"],
                            "task": task_name,
                            "message": exception,
                        })
            if exception:
                self._persist_event({
                    "kind": "background_task_failure",
                    "task": task_name,
                    "message": exception,
                })

        task.add_done_callback(completed)
        return task

    def background_snapshot(self) -> dict:
        with self._lock:
            tasks = [dict(row) for row in self._background_tasks.values()]
        return {
            "tracked": len(tasks),
            "running": sum(row["state"] == "running" for row in tasks),
            "failed": sum(row["state"] == "failed" for row in tasks),
            "tasks": list(reversed(tasks)),
        }

    async def _monitor_event_loop(self) -> None:
        interval = 1.0
        expected = time.monotonic() + interval
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            lag = max(0.0, (now - expected) * 1000.0)
            with self._lock:
                self._loop_lag_ms.append(lag)
            expected = now + interval

    def begin_request(self) -> tuple[str, float]:
        request_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._in_flight += 1
        return request_id, time.perf_counter()

    def finish_request(
        self,
        *,
        request_id: str,
        method: str,
        path: str,
        status: int,
        started: float,
        error: BaseException | None = None,
    ) -> float:
        duration_ms = max(0.0, (time.perf_counter() - started) * 1000.0)
        route = f"{method.upper()} {normalize_route(path)}"
        now_epoch = time.time()
        with self._lock:
            if (
                route not in self._routes
                and len(self._routes) >= self._route_series_limit - 1
            ):
                route = f"{method.upper()} /__overflow__"
            record = {
                "request_id": request_id,
                "route": route,
                "status": int(status),
                "duration_ms": round(duration_ms, 3),
                "at": datetime.fromtimestamp(now_epoch, timezone.utc).isoformat(
                    timespec="milliseconds"
                ),
                "_epoch": now_epoch,
            }
            if error is not None:
                record["error_type"] = type(error).__name__
                record["error"] = redact_text(error, 400)
            self._in_flight = max(0, self._in_flight - 1)
            self._total_requests += 1
            self._durations.append(duration_ms)
            status_class = f"{status // 100}xx"
            self._status_classes[status_class] += 1
            if status >= 500:
                self._server_errors += 1
            elif status >= 400:
                self._client_errors += 1
            if duration_ms >= 1000.0:
                self._slow_requests += 1
            self._recent_requests.append(record)

            metric = self._routes.setdefault(
                route,
                {
                    "count": 0,
                    "errors": 0,
                    "client_errors": 0,
                    "total_ms": 0.0,
                    "max_ms": 0.0,
                    "durations": deque(maxlen=300),
                    "status_counts": Counter(),
                    "last_status": None,
                    "last_at": None,
                    "last_request_id": None,
                },
            )
            metric["count"] += 1
            metric["errors"] += int(status >= 500)
            metric["client_errors"] += int(400 <= status < 500)
            metric["total_ms"] += duration_ms
            metric["max_ms"] = max(metric["max_ms"], duration_ms)
            metric["durations"].append(duration_ms)
            metric["status_counts"][str(status)] += 1
            metric["last_status"] = status
            metric["last_at"] = record["at"]
            metric["last_request_id"] = request_id
            if status >= 500 or error is not None:
                self._incidents.append({
                    **record,
                    "kind": "http_exception" if error else "http_5xx",
                })
        if status >= 500 or error is not None:
            persistent = {
                key: value
                for key, value in record.items()
                if key != "_epoch"
            }
            self._persist_event({
                **persistent,
                "kind": "http_exception" if error else "http_5xx",
            })
        return duration_ms

    def record_log(self, *, logger: str, level: str, message: str) -> None:
        row = {
            "at": utc_now(),
            "logger": logger,
            "level": level,
            "message": redact_text(message),
        }
        with self._lock:
            self._logs.append(row)
            if level in {"error", "critical"}:
                self._incidents.append({
                    "kind": "runtime_log",
                    **row,
                })
        if level in {"error", "critical"}:
            self._persist_event({"kind": "runtime_log", **row})

    def request_snapshot(self, window_seconds: int = 300) -> dict:
        now = time.time()
        with self._lock:
            durations = list(self._durations)
            recent = [dict(row) for row in self._recent_requests]
            window = [
                row for row in recent
                if now - float(row.get("_epoch", 0.0)) <= window_seconds
            ]
            routes = []
            for route, metric in self._routes.items():
                samples = list(metric["durations"])
                routes.append({
                    "route": route,
                    "count": metric["count"],
                    "errors": metric["errors"],
                    "client_errors": metric["client_errors"],
                    "error_rate": round(
                        metric["errors"] / max(1, metric["count"]),
                        6,
                    ),
                    "avg_ms": round(metric["total_ms"] / max(1, metric["count"]), 3),
                    "latency_sample_size": len(samples),
                    "latency_scope": "last_300_requests_for_route",
                    "p50_ms": round(_percentile(samples, 0.50), 3),
                    "p95_ms": round(_percentile(samples, 0.95), 3),
                    "p99_ms": round(_percentile(samples, 0.99), 3),
                    "max_ms": round(metric["max_ms"], 3),
                    "status_counts": dict(metric["status_counts"]),
                    "last_status": metric["last_status"],
                    "last_at": metric["last_at"],
                    "last_request_id": metric["last_request_id"],
                })
            routes.sort(key=lambda row: (-row["p95_ms"], -row["count"], row["route"]))
            loop_lag = list(self._loop_lag_ms)
            incidents = list(reversed([dict(row) for row in self._incidents]))
            logs = list(reversed([dict(row) for row in self._logs]))
            in_flight = self._in_flight
            total = self._total_requests
            server_errors = self._server_errors
            client_errors = self._client_errors
            slow = self._slow_requests
            status_classes = dict(self._status_classes)

        for row in recent:
            row.pop("_epoch", None)
        window_durations = [float(row["duration_ms"]) for row in window]
        window_errors = sum(int(row["status"] >= 500) for row in window)
        window_client_errors = sum(
            int(400 <= row["status"] < 500) for row in window
        )
        window_slow = sum(int(row["duration_ms"] >= 1000.0) for row in window)
        window_status_classes = Counter(
            f"{int(row['status']) // 100}xx" for row in window
        )
        uptime = max(0.001, time.monotonic() - self.started_monotonic)
        effective_window_seconds = max(0.001, min(float(window_seconds), uptime))
        window_truncated = (
            len(recent) >= self._recent_request_capacity
            and bool(window)
            and len(window) >= self._recent_request_capacity
        )
        return {
            "in_flight": in_flight,
            "lifetime": {
                "requests": total,
                "server_errors": server_errors,
                "client_errors": client_errors,
                "slow_requests": slow,
                "error_rate": round(server_errors / max(1, total), 6),
                "requests_per_second": round(total / uptime, 4),
                "status_classes": status_classes,
                "latency_sample_size": len(durations),
                "latency_scope": "last_2000_requests",
                "latency_ms": {
                    "p50": round(_percentile(durations, 0.50), 3),
                    "p95": round(_percentile(durations, 0.95), 3),
                    "p99": round(_percentile(durations, 0.99), 3),
                    "max": round(max(durations, default=0.0), 3),
                },
            },
            "window": {
                "seconds": window_seconds,
                "requests": len(window),
                "server_errors": window_errors,
                "client_errors": window_client_errors,
                "slow_requests": window_slow,
                "error_rate": round(window_errors / max(1, len(window)), 6),
                "requests_per_second": round(
                    len(window) / effective_window_seconds,
                    4,
                ),
                "status_classes": dict(window_status_classes),
                "latency_sample_size": len(window_durations),
                "sample_capacity": self._recent_request_capacity,
                "sample_truncated": window_truncated,
                "latency_ms": {
                    "p50": round(_percentile(window_durations, 0.50), 3),
                    "p95": round(_percentile(window_durations, 0.95), 3),
                    "p99": round(_percentile(window_durations, 0.99), 3),
                    "max": round(max(window_durations, default=0.0), 3),
                },
            },
            "routes": routes,
            "recent_requests": list(reversed(recent[-80:])),
            "incidents": incidents[:80],
            "runtime_logs": logs[:100],
            "event_loop": {
                "sample_count": len(loop_lag),
                "last_lag_ms": round(loop_lag[-1], 3) if loop_lag else None,
                "p95_lag_ms": round(_percentile(loop_lag, 0.95), 3),
                "max_lag_ms": round(max(loop_lag, default=0.0), 3),
            },
        }

    def process_snapshot(self) -> dict:
        threads = threading.enumerate()
        try:
            tasks = list(asyncio.all_tasks())
        except RuntimeError:
            tasks = []
        task_names = Counter(task.get_name() for task in tasks if not task.done())
        process = {
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "cwd": str(ROOT),
            "python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "thread_count": len(threads),
            "threads": [thread.name for thread in threads[:40]],
            "asyncio_tasks": {
                "total": len(tasks),
                "active": sum(not task.done() for task in tasks),
                "by_name": dict(task_names.most_common(30)),
            },
            "supervised_tasks": self.background_snapshot(),
        }
        if self._process is not None:
            try:
                with self._process.oneshot():
                    memory = self._process.memory_info()
                    cpu_times = self._process.cpu_times()
                    process.update({
                        "cpu_percent": round(self._process.cpu_percent(interval=None), 2),
                        "cpu_times": {
                            "user_s": round(cpu_times.user, 3),
                            "system_s": round(cpu_times.system, 3),
                        },
                        "rss_bytes": memory.rss,
                        "vms_bytes": memory.vms,
                        "rss_mb": round(memory.rss / 1024**2, 2),
                        "memory_percent": round(self._process.memory_percent(), 3),
                        "native_threads": self._process.num_threads(),
                    })
                    if hasattr(self._process, "open_files"):
                        process["open_files"] = len(self._process.open_files())
                    if hasattr(self._process, "num_fds"):
                        process["file_descriptors"] = self._process.num_fds()
                    if hasattr(self._process, "io_counters"):
                        io = self._process.io_counters()
                        process["io"] = {
                            "read_bytes": io.read_bytes,
                            "write_bytes": io.write_bytes,
                            "read_count": io.read_count,
                            "write_count": io.write_count,
                        }
                    else:
                        process["io"] = {
                            "available": False,
                            "reason": "not supported by psutil on this platform",
                        }
            except Exception as exc:  # noqa: BLE001 - telemetry must fail open
                process["resource_error"] = redact_text(exc, 400)
        try:
            load_1m, load_5m, load_15m = os.getloadavg()
            process["host_load"] = {
                "1m": round(load_1m, 3),
                "5m": round(load_5m, 3),
                "15m": round(load_15m, 3),
                "cpu_count": os.cpu_count(),
            }
        except OSError:
            pass
        if psutil is not None:
            try:
                host_memory = psutil.virtual_memory()
                disk = psutil.disk_usage(str(ROOT))
                process["host_memory"] = {
                    "total_bytes": host_memory.total,
                    "available_bytes": host_memory.available,
                    "percent": host_memory.percent,
                }
                process["workspace_disk"] = {
                    "total_bytes": disk.total,
                    "free_bytes": disk.free,
                    "percent": disk.percent,
                }
            except (psutil.Error, OSError):
                pass
        return process

    def service_snapshot(self) -> dict:
        uptime = max(0.0, time.monotonic() - self.started_monotonic)
        loopback = is_loopback_host(HOST)
        return {
            "name": "FactorFactory",
            "pid": os.getpid(),
            "host": HOST,
            "port": PORT,
            "network": {
                "bind": f"{HOST}:{PORT}",
                "scope": "loopback" if loopback else "remote",
                "loopback_only": loopback,
                "remote_unauthenticated_opt_in": (
                    not loopback and ALLOW_REMOTE_UNAUTHENTICATED
                ),
            },
            "started_at": self.started_at,
            "generated_at": utc_now(),
            "uptime_seconds": round(uptime, 3),
            "deployment": self.deployment,
            "packages": self.package_versions,
            "journal": self.journal_snapshot(),
            "environment": {
                "FF_HOST": HOST,
                "FF_PORT": str(PORT),
                "FF_EXPERIMENT_ID": os.environ.get("FF_EXPERIMENT_ID"),
                "FF_MARKET": os.environ.get("FF_MARKET", "us"),
                "remote_unauthenticated_allowed": ALLOW_REMOTE_UNAUTHENTICATED,
                "database_url_overridden": "FF_DATABASE_URL" in os.environ,
                "panel_glob_overridden": "FF_PANEL_GLOB" in os.environ,
                "backtest_root_overridden": "FF_BACKTEST_ARTIFACT_ROOT" in os.environ,
            },
}


OBSERVABILITY = RuntimeObservability(journal_path=DEFAULT_INCIDENT_JOURNAL)


class AsyncTTLCache:
    """Small single-flight TTL cache with measurable observer overhead."""

    def __init__(self, *, ttl_seconds: float, capacity: int = 16) -> None:
        self.ttl_seconds = max(0.1, float(ttl_seconds))
        self.capacity = max(1, int(capacity))
        self._entries: OrderedDict[
            object,
            tuple[float, float, Any],
        ] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._refreshes = 0
        self._failures = 0
        self._evictions = 0
        self._last_refresh_ms: float | None = None
        self._last_refresh_at: str | None = None

    def _read(self, key: object, now: float) -> Any | None:
        entry = self._entries.get(key)
        if entry is None or entry[0] <= now:
            return None
        self._entries.move_to_end(key)
        return copy.deepcopy(entry[2])

    async def get(
        self,
        key: object,
        loader: Callable[[], Awaitable[Any]],
        *,
        force: bool = False,
    ) -> Any:
        now = time.monotonic()
        if not force:
            value = self._read(key, now)
            if value is not None:
                self._hits += 1
                return value
        self._misses += 1
        async with self._lock:
            now = time.monotonic()
            if not force:
                value = self._read(key, now)
                if value is not None:
                    self._hits += 1
                    return value
            started = time.perf_counter()
            try:
                value = await loader()
            except Exception:
                self._failures += 1
                raise
            self._refreshes += 1
            self._last_refresh_ms = round(
                (time.perf_counter() - started) * 1000.0,
                3,
            )
            self._last_refresh_at = utc_now()
            self._entries[key] = (
                time.monotonic() + self.ttl_seconds,
                time.time(),
                copy.deepcopy(value),
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)
                self._evictions += 1
            return copy.deepcopy(value)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict:
        now = time.time()
        ages = [
            max(0.0, now - generated_epoch)
            for _, generated_epoch, _ in self._entries.values()
        ]
        attempts = self._hits + self._misses
        return {
            "entries": len(self._entries),
            "capacity": self.capacity,
            "ttl_seconds": self.ttl_seconds,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(1, attempts), 6),
            "refreshes": self._refreshes,
            "failures": self._failures,
            "evictions": self._evictions,
            "oldest_entry_age_seconds": round(max(ages), 3) if ages else None,
            "last_refresh_ms": self._last_refresh_ms,
            "last_refresh_at": self._last_refresh_at,
        }


def build_slo(snapshot: dict) -> dict:
    """Evaluate explicit operational objectives without hiding missing samples."""
    requests = snapshot.get("requests") or {}
    window = requests.get("window") or {}
    database = snapshot.get("database") or {}
    process = snapshot.get("process") or {}
    engine = snapshot.get("engine") or {}
    panels = (snapshot.get("data") or {}).get("panels") or []
    required_panels = [panel for panel in panels if panel.get("required")]
    objectives: list[dict[str, Any]] = []

    def objective(
        code: str,
        label: str,
        value: object,
        target: str,
        passed: bool | None,
    ) -> None:
        objectives.append({
            "code": code,
            "label": label,
            "value": value,
            "target": target,
            "status": (
                "no_data" if passed is None else "pass" if passed else "fail"
            ),
        })

    request_count = int(window.get("requests") or 0)
    error_rate = float(window.get("error_rate") or 0.0)
    p95 = float((window.get("latency_ms") or {}).get("p95") or 0.0)
    objective(
        "http_5xx_rate",
        "HTTP 5xx 率",
        error_rate,
        "< 1%",
        error_rate < 0.01 if request_count else None,
    )
    objective(
        "http_p95",
        "HTTP P95",
        p95,
        "< 500 ms",
        p95 < 500.0 if request_count else None,
    )
    db_latency = float(database.get("latency_ms") or 0.0)
    objective(
        "database",
        "数据库可用性 / 延迟",
        db_latency,
        "up 且 < 250 ms",
        database.get("status") == "ok" and db_latency < 250.0,
    )
    loop_p95 = float(
        ((requests.get("event_loop") or {}).get("p95_lag_ms")) or 0.0
    )
    loop_samples = int(
        ((requests.get("event_loop") or {}).get("sample_count")) or 0
    )
    objective(
        "event_loop",
        "事件循环 P95",
        loop_p95,
        "< 100 ms",
        loop_p95 < 100.0 if loop_samples else None,
    )
    disk = process.get("workspace_disk") or {}
    free_ratio = (
        float(disk.get("free_bytes") or 0)
        / max(1.0, float(disk.get("total_bytes") or 0))
    )
    objective(
        "disk_free",
        "工作区剩余磁盘",
        free_ratio,
        "> 10%",
        free_ratio > 0.10 if disk else None,
    )
    stale_workers = int(engine.get("stale_heartbeat_count") or 0)
    objective(
        "worker_heartbeat",
        "Worker 心跳",
        stale_workers,
        "0 stale",
        stale_workers == 0,
    )
    panel_failures = [
        panel for panel in required_panels
        if panel.get("state") == "error"
        or panel.get("source_error")
        or panel.get("schema_status") == "error"
    ]
    objective(
        "panel_contract",
        "活动面板数据契约",
        len(panel_failures),
        "0 failed",
        not panel_failures if required_panels else None,
    )
    failed = sum(row["status"] == "fail" for row in objectives)
    no_data = sum(row["status"] == "no_data" for row in objectives)
    return {
        "status": "fail" if failed else "pass",
        "failed": failed,
        "no_data": no_data,
        "objectives": objectives,
        "note": "运行 SLO 只描述工程健康，不代表因子或回测具备实盘资格。",
    }


def build_findings(snapshot: dict) -> list[dict]:
    """Turn raw telemetry into deterministic, operator-facing diagnoses."""
    findings: list[dict] = []

    def add(
        severity: str,
        code: str,
        title: str,
        detail: str,
        action: str,
    ) -> None:
        findings.append({
            "severity": severity,
            "code": code,
            "title": title,
            "detail": detail,
            "action": action,
        })

    database = snapshot.get("database") or {}
    if database.get("status") != "ok":
        add(
            "critical",
            "database_unavailable",
            "数据库不可用",
            database.get("error") or "SELECT 1 未通过",
            "检查 PostgreSQL 进程、连接 URL、权限和连接池状态。",
        )
    pool = database.get("pool") or {}
    if float(pool.get("utilization") or 0.0) >= 0.80:
        add(
            "warning",
            "database_pool_pressure",
            "数据库连接池接近耗尽",
            f"当前占用率 {float(pool['utilization']):.0%}",
            "定位慢请求和未释放事务；确认 checked_out 能随请求结束下降。",
        )

    requests = snapshot.get("requests") or {}
    window = requests.get("window") or {}
    if int(window.get("server_errors") or 0) > 0:
        add(
            "critical",
            "recent_http_5xx",
            "最近窗口出现 HTTP 5xx",
            (
                f"{window.get('seconds', 300)} 秒内 "
                f"{window.get('server_errors')} / {window.get('requests')} 个请求失败"
            ),
            "按请求 ID查看最近异常，再对照路由延迟表和运行日志。",
        )
    p95 = float(((window.get("latency_ms") or {}).get("p95")) or 0)
    if p95 >= 1000.0:
        add(
            "warning",
            "slow_http_p95",
            "API P95 延迟偏高",
            f"{window.get('seconds', 300)} 秒窗口 P95 为 {p95:.1f} ms",
            "在路由表中按 P95 排序，区分面板冷加载、数据库等待和计算型请求。",
        )
    if window.get("sample_truncated"):
        add(
            "warning",
            "http_window_truncated",
            "HTTP 观测窗口已截断",
            (
                f"请求量超过内存样本上限 "
                f"{window.get('sample_capacity')}，窗口统计只覆盖最近样本"
            ),
            "接入 Prometheus 时序存储，或在确认内存预算后提高样本容量。",
        )
    loop_p95 = float(((requests.get("event_loop") or {}).get("p95_lag_ms")) or 0)
    if loop_p95 >= 100.0:
        add(
            "warning",
            "event_loop_lag",
            "事件循环存在阻塞",
            f"事件循环延迟 P95 为 {loop_p95:.1f} ms",
            "检查是否有 Polars/文件扫描未通过 asyncio.to_thread 隔离。",
        )

    for panel in ((snapshot.get("data") or {}).get("panels") or []):
        if (
            panel.get("required", True)
            and (
                panel.get("state") == "error"
                or panel.get("schema_status") == "error"
            )
        ):
            missing = panel.get("missing_dsl_fields") or panel.get(
                "missing_required_columns"
            )
            add(
                "critical",
                "panel_contract_error",
                f"{panel.get('market', 'unknown')} 面板数据契约失败",
                (
                    panel.get("load_error")
                    or panel.get("schema_error")
                    or panel.get("source_error")
                    or f"缺少字段: {', '.join(missing or [])}"
                ),
                "检查面板 glob、Parquet schema、文件权限和剩余磁盘空间。",
            )

    for worker in snapshot.get("workers") or []:
        if worker.get("running") and worker.get("heartbeat_stale"):
            add(
                "warning",
                "worker_heartbeat_stale",
                f"研究任务 {worker.get('experiment_id')} 心跳过旧",
                f"阶段 {worker.get('phase')}，心跳年龄 {worker.get('heartbeat_age_seconds')} 秒",
                "检查当前阶段耗时、最近日志、线程/CPU/内存以及面板或 LLM 调用。",
            )
        if worker.get("task_done") and worker.get("task_exception"):
            add(
                "critical",
                "worker_task_exception",
                f"研究任务 {worker.get('experiment_id')} 异常退出",
                worker.get("task_exception"),
                "查看同一任务的最近 error 事件和完整 traceback。",
            )
        if worker.get("phase") == "failed" and worker.get("last_error"):
            add(
                "critical",
                "worker_failed",
                f"研究任务 {worker.get('experiment_id')} 失败停止",
                worker.get("last_error"),
                "按任务 ID 和最近阶段过滤持久化事件，修复后显式重新启动任务。",
            )

    network = ((snapshot.get("service") or {}).get("network") or {})
    if not network.get("loopback_only", True):
        add(
            "warning",
            "remote_unauthenticated_bind",
            "服务已暴露到非本机网络",
            f"当前监听 {network.get('bind')}，控制 API 没有身份认证。",
            "仅在隔离网络中使用；完成排障后恢复 FF_HOST=127.0.0.1。",
        )

    journal = ((snapshot.get("service") or {}).get("journal") or {})
    if journal.get("error"):
        add(
            "warning",
            "incident_journal_error",
            "跨重启事故日志不可写",
            journal["error"],
            "检查 var/observability 的目录权限与剩余磁盘空间。",
        )

    supervised = ((snapshot.get("process") or {}).get("supervised_tasks") or {})
    if int(supervised.get("failed") or 0):
        add(
            "critical",
            "background_task_failure",
            "后台任务静默失败",
            f"{supervised.get('failed')} 个受监管后台任务失败。",
            "在任务明细和跨重启事故日志中查看异常，修复后重启服务。",
        )

    disk = ((snapshot.get("process") or {}).get("workspace_disk") or {})
    if disk:
        free_ratio = (
            float(disk.get("free_bytes") or 0)
            / max(1.0, float(disk.get("total_bytes") or 0))
        )
        if free_ratio <= 0.10:
            add(
                "critical",
                "workspace_disk_pressure",
                "工作区磁盘空间不足",
                f"剩余空间 {free_ratio:.1%}",
                "清理可重建缓存或迁移回测产物；不要删除数据库与历史实验。",
            )

    providers = snapshot.get("providers") or {}
    if not providers.get("inner_provider_configured"):
        add(
            "info",
            "inner_provider_fallback",
            "内层 LLM 未配置",
            "内层提案将使用随机基线或回退路径。",
            "若预期使用 LLM，请在设置页绑定内层 provider；仅配置不代表网络可达。",
        )
    if not providers.get("outer_provider_configured"):
        add(
            "info",
            "outer_provider_fallback",
            "外层 LLM 未配置",
            "外层优化将使用随机扰动或回退路径。",
            "若预期使用 LLM，请在设置页绑定外层 provider；仅配置不代表网络可达。",
        )

    if not any(
        row["severity"] in {"critical", "warning"} for row in findings
    ):
        add(
            "ok",
            "no_active_fault",
            "未发现活动故障",
            "数据库、请求窗口、面板注册表和 worker 心跳未触发告警规则。",
            "继续观察 P95、连接池、面板身份和最近异常；健康不等于研究结果可用于实盘。",
        )
    severity_order = {"critical": 0, "warning": 1, "info": 2, "ok": 3}
    return sorted(findings, key=lambda row: severity_order.get(row["severity"], 9))


def overall_health(findings: list[dict]) -> str:
    severities = {row.get("severity") for row in findings}
    if "critical" in severities:
        return "unhealthy"
    if "warning" in severities:
        return "degraded"
    return "healthy"


def fingerprint_payload(value: object) -> str:
    """Short stable identity for configs and data inventories."""
    return hashlib.sha256(repr(value).encode("utf-8")).hexdigest()[:16]
