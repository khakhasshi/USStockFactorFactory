"""API audit bookkeeping tests: isolated fake sessions, no live DB writes."""
import asyncio
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from backend.app.api import routes
from backend.app.eval import harness


@pytest.fixture
def api(monkeypatch):
    rows = []
    class Session:
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        def add(self, row):
            row.id = len(rows) + 1
            rows.append(row)
        async def commit(self): pass
        async def refresh(self, row): pass
        async def get(self, model, key): return rows[key - 1]
    async def context(eid):
        return 999, {"market": "us", "portfolio_mode": "long_only"}
    monkeypatch.setattr(routes, "SessionLocal", Session)
    monkeypatch.setattr(routes, "_experiment_context", context)
    monkeypatch.setattr(routes, "_batch_evaluation_slots", asyncio.Semaphore(1))
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), rows


def test_batch_registers_before_compute_and_records_failure(api, monkeypatch):
    client, rows = api
    def compute(expression, tasks, **kwargs):
        assert len(rows) == 2
        assert all(r.statistic["status"] == "pending" for r in rows)
        assert all(not r.selected for r in rows)
        kwargs["progress_callback"]({"completed": 2, "total": 2})
        return {"results": [
            {"status": "ok", "metrics": {"direction": -1, "raw_return_evidence": {"test": True}}},
            {"status": "error", "error": "invalid coverage"}], "formal_eligible": False}
    monkeypatch.setattr(harness, "evaluate_batch", compute)
    r = client.post("/api/evaluations/batch", json={"expression": "rank(close)",
        "tasks": [{"horizon": 5}, {"horizon": 10}]})
    assert r.status_code == 200, r.text
    assert r.json()["trials_registered"]
    assert [x["trial_id"] for x in r.json()["results"]] == [1, 2]
    assert rows[0].statistic["raw_return_evidence"] == {"test": True}
    assert rows[1].failure_reason == "invalid coverage"
    assert all(not r.statistic["formal_eligible"] for r in rows)


@pytest.mark.parametrize("task", [{"horizon": 2}, {"direction": 0},
    {"portfolio_mode": "invalid"}, {"scope": "full_audit"}])
def test_batch_rejects_invalid_request_before_trial(api, task):
    client, rows = api
    r = client.post("/api/evaluations/batch", json={"expression": "rank(close)", "tasks": [task]})
    assert r.status_code in {400, 422}
    assert not rows


def test_batch_failure_keeps_attempts(api, monkeypatch):
    client, rows = api
    def fail(*args, **kwargs): raise RuntimeError("panel unavailable")
    monkeypatch.setattr(harness, "evaluate_batch", fail)
    with pytest.raises(RuntimeError, match="panel unavailable"):
        client.post("/api/evaluations/batch", json={"expression": "rank(close)", "tasks": [{}]})
    assert rows[0].failure_reason == "batch_pending_evidence"
