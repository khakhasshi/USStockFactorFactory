import time

from backend.app.compute_progress import ComputeProgressRegistry


def test_registry_reports_real_ratio_and_terminal_state():
    registry = ComputeProgressRegistry()
    registry.start(
        "backtest:7",
        kind="backtest",
        title="事件回测 #7",
        completed=0,
        total=100,
    )
    row = registry.update(
        "backtest:7",
        phase="event_simulation",
        completed=40,
        total=100,
    )
    assert row["progress"] == 0.4
    assert row["indeterminate"] is False
    completed = registry.finish("backtest:7", message="完成")
    assert completed["state"] == "done"
    assert completed["progress"] == 1.0
    assert completed["cancellable"] is False


def test_registry_can_switch_to_an_indeterminate_phase():
    registry = ComputeProgressRegistry()
    registry.start(
        "joint:a",
        kind="qlib_joint",
        title="联合模型",
        completed=8,
        total=10,
    )
    row = registry.update(
        "joint:a",
        phase="feature_materialize",
        completed=None,
        total=None,
        message="总耗时不可预估",
    )
    assert row["progress"] is None
    assert row["indeterminate"] is True
    assert row["completed"] is None
    assert row["total"] is None


def test_registry_hides_recent_jobs_when_requested():
    registry = ComputeProgressRegistry()
    registry.start("research:1", kind="research", title="研究")
    registry.start("backtest:2", kind="backtest", title="回测")
    registry.finish("backtest:2")
    active = registry.snapshot(include_recent=False)
    assert [row["job_id"] for row in active] == ["research:1"]


def test_terminal_elapsed_time_stops_growing():
    registry = ComputeProgressRegistry()
    registry.start("backtest:stable", kind="backtest", title="回测")
    time.sleep(0.002)
    completed = registry.finish("backtest:stable")
    time.sleep(0.002)
    assert registry.get("backtest:stable")["elapsed_seconds"] == completed["elapsed_seconds"]
