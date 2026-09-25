from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from types import SimpleNamespace

import polars as pl
from polars.testing import assert_series_equal
import pytest

from backend.app import factor_compute_cache as cc
from backend.app.dsl.engine import parse


@pytest.fixture
def source(monkeypatch):
    import math
    rows = [{"ts_code": f"S{s:02}", "trade_date": date(2020, 1, 1) + timedelta(days=d),
             "close": 50 + s + math.sin(d / 4 + s) * 3,
             "vol": 1000. + s * 13 + d * 17} for s in range(12) for d in range(90)]
    df = pl.DataFrame(rows)
    snap = SimpleNamespace(frame=df, market="us", store=object(), generation=1,
        loaded_identity="test", manifest={"snapshot_id": "test-v1", "immutable_inputs_available": True,
                                          "code": {"code_sha256": "code-v1"}})
    monkeypatch.setattr(cc, "_CACHE", cc.ColumnCache(1024 * 1024))
    monkeypatch.setenv("FF_FACTOR_COMPUTE_CACHE", "1")
    return df, snap


@pytest.mark.parametrize("expr", ["rank(close)", "ts_mean(rank(close),5)",
    "rank(ts_mean(close,5)/ts_std(close,20))", "rank(ts_corr(close,vol,10))",
    "winsor_mad(ts_delta(close,5),3)", "zscore(returns(close,5))",
    "ts_rank(close,20)", "ts_slope(close,10)", "ts_rsquare(close,10)",
    "ts_argmin(close,10)", "ts_quantile(close,10,0.2)", "close", "2.0"])
def test_cold_warm_and_bounded_interval_match_reference(source, expr):
    df, snap = source
    for start, end in ((None, None), (date(2020,1,15), date(2020,3,1))):
        lf = df.lazy()
        if start: lf = lf.filter(pl.col("trade_date") >= start)
        if end: lf = lf.filter(pl.col("trade_date") <= end)
        reference = parse(expr, ["close", "vol"]).apply(lf).collect()["factor"]
        for _ in range(2):
            actual = cc.factor_column(df, expr, ["close", "vol"], "us", snapshot=snap, start=start, end=end)
            selected = df.with_columns(actual)
            if start: selected = selected.filter(pl.col("trade_date") >= start)
            if end: selected = selected.filter(pl.col("trade_date") <= end)
            assert_series_equal(reference, selected["factor"], check_exact=True)


def test_common_subexpression_hit_and_sign_not_canonicalized(source):
    df, snap = source
    a = cc.factor_column(df, "rank(ts_mean(close,5))", ["close", "vol"], "us", snapshot=snap)
    expr = "rank(ts_mean(close,5)/ts_std(close,7))"
    actual = cc.factor_column(df, expr, ["close", "vol"], "us", snapshot=snap)
    assert_series_equal(actual, parse(expr, ["close", "vol"]).apply(df.lazy()).collect()["factor"], check_exact=True)
    assert cc.cache_stats()["subexpression_reuses"] > 0
    negative = cc.factor_column(df, "-rank(ts_mean(close,5))", ["close", "vol"], "us", snapshot=snap)
    assert_series_equal(negative, -a, check_exact=True)


def test_snapshot_version_scope_field_validation_and_copy_isolation(source):
    df, snap = source
    args = (df, "rank(close)", ["close", "vol"], "us")
    first = cc.factor_column(*args, snapshot=snap)
    first.scatter([0], [999.])
    assert cc.factor_column(*args, snapshot=snap)[0] != 999.
    snap.manifest["code"]["code_sha256"] = "code-v2"
    cc.factor_column(*args, snapshot=snap)
    snap.manifest["snapshot_id"] = "test-v2"
    cc.factor_column(*args, snapshot=snap)
    cc.factor_column(*args, snapshot=snap, end=date(2020,2,1))
    assert cc.cache_stats()["factor_materializations"] == 4
    with pytest.raises(ValueError):
        cc.factor_column(df, "rank(close)", ["vol"], "us", snapshot=snap)
    assert cc.factor_column(df.reverse(), "close", ["close"], "us", snapshot=snap) is None


def test_memory_eviction_and_disk_checksum(tmp_path):
    cache = cc.ColumnCache(100, tmp_path, 10000)
    a = pl.Series("value", range(10), dtype=pl.Float64)
    cache.put("a"*64, a, disk=True)
    cache.put("b"*64, a)
    assert cache.stats()["memory_bytes"] <= 100 and cache.stats()["evictions"] == 1
    assert_series_equal(cache.get("a"*64, 10, disk=True), a)
    (tmp_path / ("a"*64 + ".arrow")).write_bytes(b"corrupt")
    cache.put("c"*64, a)
    assert cache.get("a"*64, 10, disk=True) is None
    assert cache.stats()["disk_invalid_or_unavailable"] == 1


def test_concurrent_same_expression_materializes_once(source):
    df, snap = source
    def run(_):
        return cc.factor_column(df, "rank(ts_mean(close,10))", ["close", "vol"], "us", snapshot=snap)
    with ThreadPoolExecutor(max_workers=4) as pool:
        values = list(pool.map(run, range(8)))
    assert cc.cache_stats()["factor_materializations"] == 1
    for value in values[1:]:
        assert_series_equal(values[0], value, check_exact=True)


def test_batch_scope_configurations_and_failure_are_independent(monkeypatch):
    from contextlib import contextmanager
    from backend.app.eval import harness
    from backend.app import audit_snapshot
    @contextmanager
    def frozen(*args):
        yield SimpleNamespace(manifest={"snapshot_id": "test"})
    calls = []
    def evaluate(expr, **kwargs):
        calls.append(kwargs)
        if kwargs["horizon"] == 19: raise ValueError("invalid")
        return {"scope": "discovery", "parameters": kwargs}
    monkeypatch.setattr(audit_snapshot, "frozen_panel", frozen)
    monkeypatch.setattr(harness, "evaluate", evaluate)
    progress = []
    result = harness.evaluate_batch("rank(close)", [{"horizon": 5, "universe_n": 500},
        {"horizon": 19}, {"horizon": 20, "universe_n": 1500}], progress_callback=progress.append)
    assert [r["status"] for r in result["results"]] == ["ok", "error", "ok"]
    assert len(calls) == 3 and progress[-1]["completed"] == 3
    assert result["configuration_trials"] == 3 and not result["holdout_vault_consumed"]
    with pytest.raises(ValueError):
        harness.evaluate_batch("close", [{"scope": "full_audit"}])
