from contextlib import contextmanager
from copy import deepcopy
from datetime import date, timedelta
import hashlib
from types import SimpleNamespace

import pytest

from backend.app import audit_snapshot, research_overfit
from backend.app.config import evaluation_config
from backend.app.eval import event_audit, harness


def _provenance(available=True):
    return {"immutable_inputs_available": available, "actual_start": "2023-01-01", "actual_end": "2024-12-31",
            "actual_sessions": 500, "expression": "rank(close)",
            "expression_sha256": hashlib.sha256(b"rank(close)").hexdigest(), "config_sha256": "c" * 64,
            "run_fingerprint": "f" * 64, "requested_start": "2023-01-01", "requested_end": "2024-12-31",
            "panel": {"immutable_inputs_available": available, "status": "FROZEN",
                      "data_sha256": "d" * 64, "snapshot_id": "s" * 64,
                      "market_calendar_sha256": "b" * 64, "files": [{"sha256": "d" * 64}],
                      "code": {"code_sha256": "c" * 64, "persisted": available, "source_files": [{"sha256": "c" * 64}]}}}


def _result(**stats):
    return {"stats": {"sharpe": 1.2, "ann_ret": .15, "max_dd": .15, "avg_daily_turnover": .02, **stats},
            "integrity": {"all_pass": True}, "input_provenance": _provenance(), "config": {}}


def test_event_pass_requires_real_integrity_key_and_provenance():
    cfg = evaluation_config("us")
    result = _result()
    assert event_audit.event_metric_gate(result, cfg)["passed"]
    for bad in ({"all_pass": False}, {"all_passed": True}, {}):
        result["integrity"] = bad
        assert not event_audit.event_metric_gate(result, cfg)["passed"]
    result = _result()
    result.pop("input_provenance")
    assert not event_audit.event_metric_gate(result, cfg)["passed"]
    result["input_provenance"] = _provenance(False)
    assert not event_audit.event_metric_gate(result, cfg)["passed"]


@pytest.mark.parametrize("bad_metrics,reason", [
    ({"max_dd": .6}, "event_daily_drawdown_above_gate"),
    ({"sharpe": .1}, "event_net_sharpe_below_gate"),
    ({"ann_ret": -.1}, "event_net_return_not_positive"),
    ({"avg_daily_turnover": .8}, "event_turnover_above_gate"),
])
def test_event_daily_risk_failure_is_a_hard_gate(bad_metrics, reason):
    result = event_audit.event_metric_gate(_result(**bad_metrics), evaluation_config("us"))
    assert not result["passed"]
    assert reason in result["failure_reasons"]


def _snapshot_dates():
    start = date(2020, 1, 1)
    return tuple(start + timedelta(days=i) for i in range(2446) if (start + timedelta(days=i)).weekday() < 5)


def test_all_three_event_windows_freeze_execution_parameters(monkeypatch):
    calls = []
    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return _result()
    monkeypatch.setattr(event_audit, "run_backtest", run)
    cfg = evaluation_config("us")
    snapshot = SimpleNamespace(trading_dates=_snapshot_dates())
    audit = event_audit.run_event_audit("rank(close)", 500, 5, "long_short", -1, "us", "frozen/*.parquet", cfg, snapshot)
    assert audit["passed"]
    assert set(audit["windows"]) == {"holdout", "vault", "rating"}
    assert len(calls) == 3
    for args, kwargs in calls:
        assert args[:2] == ("rank(close)", 500)
        assert kwargs["direction"] == -1
        assert kwargs["rebalance_every"] == 5
        assert kwargs["mode"] == "long_short"
        assert kwargs["top_fraction"] == cfg["top_fraction"]
        assert kwargs["slippage_bps"] == cfg["base_cost_bps"]
        assert kwargs["borrow_cost_bps_annual"] == cfg["borrow_cost_bps_annual"]
        assert kwargs["initial_capital"] == cfg["target_capital"]
        assert kwargs["execution_backend"] == "python"
        assert kwargs["max_volume_participation"] == cfg["max_adv_participation"]
        assert kwargs["liquidate_at_end"] is True
    assert audit["windows"]["holdout"]["actual_start"] > "2023-01-01"
    assert audit["windows"]["rating"]["actual_start"] == "2020-01-01"
    assert audit["visible_to_research_llms"] is False


def test_event_execution_errors_and_unsupported_contract_fail_closed(monkeypatch):
    snapshot = SimpleNamespace(trading_dates=_snapshot_dates())
    def explode(*args, **kwargs):
        raise ValueError("inconsistent frozen panel")
    monkeypatch.setattr(event_audit, "run_backtest", explode)
    result = event_audit.run_event_audit("close", 500, 5, "long_short", 1, "us", None, evaluation_config("us"), snapshot)
    assert not result["passed"]
    assert all(window["status"] == "ERROR" for window in result["windows"].values())
    monkeypatch.setattr(event_audit, "run_backtest", lambda *a, **kw: _result())
    cfg = evaluation_config("us", {"tail_fraction": .1})
    result = event_audit.run_event_audit("close", 500, 5, "long_short", 1, "us", None, cfg, snapshot)
    assert not result["passed"]
    assert all(window["status"] == "FAIL" for window in result["windows"].values())


@pytest.mark.parametrize("event_pass,snapshot_pass,overfit_pass", [(False, True, True), (True, False, True), (True, True, False)])
def test_vector_champion_never_promoted_if_new_hard_evidence_fails(monkeypatch, event_pass, snapshot_pass, overfit_pass):
    snapshot = SimpleNamespace(trading_dates=_snapshot_dates(), manifest={"immutable_inputs_available": snapshot_pass})
    @contextmanager
    def freeze(*args, **kwargs):
        yield snapshot
    monkeypatch.setattr(audit_snapshot, "frozen_panel", freeze)
    monkeypatch.setattr(audit_snapshot, "build_run_provenance", lambda *a, **kw: _provenance(snapshot_pass))
    monkeypatch.setattr(harness, "_evaluate_full_vector", lambda *a, **kw: {
        "direction": -1, "eligibility": {"research_pass": True, "holdout_pass": True, "vault_pass": True,
        "capacity_pass": True, "eligible": True, "grade": "F5", "failure_reasons": []}, "ranking": {},
    })
    monkeypatch.setattr(event_audit, "run_event_audit", lambda *a, **kw: {"passed": event_pass})
    monkeypatch.setattr(research_overfit, "formal_overfit_governance", lambda *a, **kw: {"passed": overfit_pass, "reasons": []})
    result = harness.evaluate_full("rank(close)", trial_history=[])
    assert result["eligibility"]["eligible"] is False
    assert result["ranking"]["formal_gate_passed"] is False
    assert result["eligibility"]["grade"] != "F5"
