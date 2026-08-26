"""Thread-safe runtime progress registry for heterogeneous computations.

Durable task owners (research workers, backtests and combination experiments)
remain the source of truth.  This registry adds live, stage-level heartbeats for
calculations whose database row cannot be updated cheaply from a worker thread.
The API layer merges both sources and de-duplicates by ``job_id``.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Any


ACTIVE_STATES = frozenset({"queued", "starting", "running", "stopping"})
TERMINAL_STATES = frozenset({"done", "failed", "stopped", "cancelled"})
_UNSET = object()


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ComputeProgressRegistry:
    """Bounded in-memory registry; safe to update from executor threads."""

    def __init__(self, *, capacity: int = 200) -> None:
        self.capacity = max(20, int(capacity))
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _numbers(completed: Any, total: Any) -> tuple[float | None, float | None, float | None]:
        try:
            done = float(completed) if completed is not None else None
        except (TypeError, ValueError):
            done = None
        try:
            size = float(total) if total is not None else None
        except (TypeError, ValueError):
            size = None
        if size is None or size <= 0 or done is None:
            return done, size, None
        return done, size, max(0.0, min(1.0, done / size))

    def start(
        self,
        job_id: str,
        *,
        kind: str,
        title: str,
        phase: str = "starting",
        message: str = "",
        completed: float | int | None = None,
        total: float | int | None = None,
        cancellable: bool = False,
        experiment_id: int | None = None,
        metadata: dict | None = None,
    ) -> dict:
        now = utc_iso()
        monotonic = time.monotonic()
        done, size, ratio = self._numbers(completed, total)
        payload = {
            "job_id": str(job_id),
            "kind": str(kind),
            "title": str(title),
            "state": "running",
            "phase": str(phase),
            "message": str(message or ""),
            "completed": done,
            "total": size,
            "progress": ratio,
            "indeterminate": ratio is None,
            "cancellable": bool(cancellable),
            "experiment_id": experiment_id,
            "metadata": dict(metadata or {}),
            "error": "",
            "started_at": now,
            "updated_at": now,
            "completed_at": None,
            "heartbeat_age_seconds": 0.0,
            "elapsed_seconds": 0.0,
            "_started_monotonic": monotonic,
            "_updated_monotonic": monotonic,
        }
        with self._lock:
            self._jobs[str(job_id)] = payload
            self._prune_locked()
        return self.get(str(job_id)) or {}

    def update(
        self,
        job_id: str,
        *,
        phase: str | None = None,
        message: str | None = None,
        completed: float | int | None | object = _UNSET,
        total: float | int | None | object = _UNSET,
        state: str | None = None,
        metadata: dict | None = None,
    ) -> dict | None:
        with self._lock:
            row = self._jobs.get(str(job_id))
            if row is None:
                return None
            if phase is not None:
                row["phase"] = str(phase)
            if message is not None:
                row["message"] = str(message)
            if state is not None:
                row["state"] = str(state)
            if completed is not _UNSET or total is not _UNSET:
                resolved_completed = None if completed is _UNSET else completed
                resolved_total = None if total is _UNSET else total
                done, size, ratio = self._numbers(resolved_completed, resolved_total)
                row.update({
                    "completed": done,
                    "total": size,
                    "progress": ratio,
                    "indeterminate": ratio is None,
                })
            if metadata:
                row["metadata"] = {**dict(row.get("metadata") or {}), **dict(metadata)}
            row["updated_at"] = utc_iso()
            row["_updated_monotonic"] = time.monotonic()
        return self.get(str(job_id))

    def finish(
        self,
        job_id: str,
        *,
        state: str = "done",
        message: str = "",
        error: str = "",
        metadata: dict | None = None,
    ) -> dict | None:
        if state not in TERMINAL_STATES:
            raise ValueError(f"invalid terminal state: {state}")
        with self._lock:
            row = self._jobs.get(str(job_id))
            if row is None:
                return None
            finished_monotonic = time.monotonic()
            if row.get("total") is not None and state == "done":
                row["completed"] = row["total"]
                row["progress"] = 1.0
                row["indeterminate"] = False
            row.update({
                "state": state,
                "phase": state,
                "message": str(message or row.get("message") or ""),
                "error": str(error or "")[:4000],
                "updated_at": utc_iso(),
                "completed_at": utc_iso(),
                "cancellable": False,
                "_updated_monotonic": finished_monotonic,
                "_completed_monotonic": finished_monotonic,
            })
            if metadata:
                row["metadata"] = {**dict(row.get("metadata") or {}), **dict(metadata)}
        return self.get(str(job_id))

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            row = self._jobs.get(str(job_id))
            return self._public(row) if row is not None else None

    def snapshot(self, *, include_recent: bool = True, limit: int = 100) -> list[dict]:
        with self._lock:
            rows = [self._public(row) for row in self._jobs.values()]
        if not include_recent:
            rows = [row for row in rows if row["state"] in ACTIVE_STATES]
        rows.sort(
            key=lambda row: (
                row["state"] not in ACTIVE_STATES,
                str(row.get("updated_at") or ""),
            ),
            reverse=False,
        )
        active = [row for row in rows if row["state"] in ACTIVE_STATES]
        recent = sorted(
            (row for row in rows if row["state"] not in ACTIVE_STATES),
            key=lambda row: str(row.get("updated_at") or ""),
            reverse=True,
        )
        return (active + recent)[: max(1, int(limit))]

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()

    def _public(self, row: dict) -> dict:
        now = time.monotonic()
        result = {key: value for key, value in row.items() if not key.startswith("_")}
        elapsed_end = row.get("_completed_monotonic", now)
        result["elapsed_seconds"] = round(
            max(0.0, elapsed_end - row["_started_monotonic"]), 3
        )
        result["heartbeat_age_seconds"] = round(max(0.0, now - row["_updated_monotonic"]), 3)
        return result

    def _prune_locked(self) -> None:
        if len(self._jobs) <= self.capacity:
            return
        terminal = sorted(
            (
                (key, row) for key, row in self._jobs.items()
                if row.get("state") in TERMINAL_STATES
            ),
            key=lambda item: item[1].get("_updated_monotonic", 0.0),
        )
        for key, _ in terminal[: max(0, len(self._jobs) - self.capacity)]:
            self._jobs.pop(key, None)


COMPUTE_PROGRESS = ComputeProgressRegistry()
