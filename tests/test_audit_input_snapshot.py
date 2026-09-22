from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from backend.app import audit_snapshot as snapshots


def _store(tmp_path):
    path = tmp_path / "input.parquet"
    frame = pl.DataFrame({"trade_date": [date(2024, 1, 2), date(2024, 1, 3)], "close": [10., 11.]})
    frame.write_parquet(path)
    store = SimpleNamespace(market="us", panel_glob=str(path), frame=frame)
    store.read_snapshot = lambda: (store.frame, tuple(store.frame["trade_date"]), "loaded-1", 1)
    return store, path


@pytest.fixture(autouse=True)
def small_code_snapshot(monkeypatch):
    monkeypatch.setattr(snapshots, "code_snapshot", lambda *a, **kw: {
        "git_commit": "test-commit", "tracked_diff_sha256": "diff", "code_sha256": "source", "persisted": True,
    })


def test_snapshot_copies_inputs_and_pins_audit_despite_reload(tmp_path):
    store, source = _store(tmp_path)
    with snapshots.frozen_panel(store=store, root=tmp_path / "snapshots") as frozen:
        assert frozen.manifest["immutable_inputs_available"]
        copy = frozen.manifest["files"][0]["snapshot_path"]
        old_hash = snapshots.hash_file(source)
        store.frame = store.frame.with_columns(pl.lit(999.).alias("close"))
        store.frame.write_parquet(source)
        read, dates, identity, generation = snapshots.read_frozen_panel(store)
        assert read["close"].to_list() == [10., 11.]
        assert snapshots.hash_file(__import__("pathlib").Path(copy)) == old_hash
        assert snapshots.hash_file(source) != old_hash
        provenance = snapshots.build_run_provenance("close", {"direction": 1}, "2020-01-01", "2030-01-01", dates, read)
        assert provenance["actual_start"] == "2024-01-02"
        assert provenance["actual_end"] == "2024-01-03"
        assert provenance["expression_sha256"]
        assert provenance["panel"]["code"]["git_commit"] == "test-commit"
    assert snapshots.active_snapshot() is None


def test_synthetic_frame_does_not_claim_persisted_input_proof(tmp_path):
    store, source = _store(tmp_path)
    store.panel_glob = str(tmp_path / "missing-*.parquet")
    snap = snapshots.freeze_panel(store, root=tmp_path / "snapshots")
    assert snap.manifest["status"] == "IN_MEMORY_ONLY"
    assert not snap.manifest["immutable_inputs_available"]


def test_stale_memory_cannot_be_certified_with_new_disk_hash(tmp_path):
    store, _ = _store(tmp_path)
    store._source_inventory = lambda **kwargs: {"identity": "different-data"}
    with pytest.raises(ValueError, match="Loaded panel and source partitions differ"):
        snapshots.freeze_panel(store, root=tmp_path / "snapshots")


def test_nested_sleeves_share_first_snapshot_and_reset_context(tmp_path):
    store, _ = _store(tmp_path)
    @snapshots.pin_backtest_inputs
    def sleeve():
        return snapshots.freeze_panel(store, root=tmp_path / "snapshots")
    @snapshots.pin_backtest_inputs
    def ensemble():
        first = sleeve()
        store.frame = store.frame.with_columns(pl.lit(77.).alias("close"))
        return first, sleeve()
    first, second = ensemble()
    assert first is second
    assert first.frame["close"].to_list() == [10., 11.]
    assert snapshots.active_snapshot() is None

