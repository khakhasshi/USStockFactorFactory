#!/usr/bin/env python3
"""Paired real-data verification; no production trials/tasks are created."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.eval.harness import evaluate, evaluate_batch, _evaluate_full_vector
from backend.app.backtest.engine import run_backtest
from backend.app import factor_compute_cache as cc
from scripts.benchmark_python_optimizations import EXPRESSION, digest, evidence


def reset():
    with cc._CACHE.lock:
        cc._CACHE.entries.clear()
        cc._CACHE.bytes = 0
        cc._CACHE.counters.clear()


def score_evidence(r):
    return {k: v for k, v in r.items() if k != "runtime"}


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round(time.perf_counter() - start, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    if out.exists(): raise SystemExit("Refusing to overwrite report")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "factorfactory.compute-cache-benchmark/v1", "cases": []}
    def record(row):
        report["cases"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        temp = out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temp.replace(out)
    tasks = [{"name": "T1", "universe_n": 500, "horizon": 5},
             {"name": "T2", "universe_n": 1500, "horizon": 10},
             {"name": "T3", "universe_n": 500, "horizon": 20}]
    for market, mode in (("us", "long_only"), ("us", "long_short"), ("ashare", "long_only")):
        os.environ["FF_FACTOR_COMPUTE_CACHE"] = "0"
        evaluate(EXPRESSION, market=market, portfolio_mode=mode)  # panel warmup
        requests = [{**t, "portfolio_mode": mode} for t in tasks]
        reference, reference_seconds = timed(evaluate_batch, EXPRESSION, requests, market=market)
        os.environ["FF_FACTOR_COMPUTE_CACHE"] = "1"
        reset()
        cold, cold_seconds = timed(evaluate_batch, EXPRESSION, requests, market=market)
        warm, warm_seconds = timed(evaluate_batch, EXPRESSION, requests, market=market)
        hashes = [digest([score_evidence(r["metrics"]) for r in batch["results"]]) for batch in (reference, cold, warm)]
        record({"market": market, "mode": mode, "kind": "three_configuration_batch",
                "seconds": {"reference": reference_seconds, "cold_cache": cold_seconds, "warm_cache": warm_seconds},
                "exact_equal": len(set(hashes)) == 1, "hashes": hashes, "cache": cc.cache_stats(),
                "snapshot": warm["input_snapshot_id"]})
        # Related expression exercises reusing measured intermediate columns.
        related = EXPRESSION.replace("ts_mean(amount,120)", "ts_mean(amount,90)")
        os.environ["FF_FACTOR_COMPUTE_CACHE"] = "0"
        a, ta = timed(evaluate, related, market=market, portfolio_mode=mode)
        os.environ["FF_FACTOR_COMPUTE_CACHE"] = "1"
        b, tb = timed(evaluate, related, market=market, portfolio_mode=mode)
        record({"market": market, "mode": mode, "kind": "related_expression",
                "reference_seconds": ta, "cached_seconds": tb,
                "exact_equal": score_evidence(a) == score_evidence(b), "cache": cc.cache_stats()})
        # Confirm HOLDOUT, Vault, rating and direction-freeze semantics too.
        values = []
        for flag in ("0", "1"):
            os.environ["FF_FACTOR_COMPUTE_CACHE"] = flag
            values.append(_evaluate_full_vector(EXPRESSION, market=market, portfolio_mode=mode))
        record({"market": market, "mode": mode, "kind": "full_vector_audit",
                "exact_equal": score_evidence(values[0]) == score_evidence(values[1])})
        for stress in (False, True):
            kwargs = dict(expression=EXPRESSION, universe_n=500, start="2023-01-01", end="2026-09-23",
                          market=market, mode=mode, rebalance_every=5, top_fraction=.2,
                          execution_backend="python", response_trade_limit=10**9, response_daily_limit=None,
                          monte_carlo_enabled=True, monte_carlo_simulations=2000)
            # Explicit operational headroom. The initial no-headroom run is
            # retained separately as a negative-control report, not overwritten.
            if mode == "long_short":
                kwargs.update(long_gross_target=.9, short_gross_target=.9)
            if stress:
                kwargs.update(position_sizing="inverse_volatility", unfilled_order_policy="carry",
                              liquidate_at_end=True, portfolio_stop_drawdown_pct=.15,
                              risk_cooldown_sessions=5, impact_model="square_root",
                              exit_policy={"fixed_stop_loss_pct": .08, "atr_stop_multiple": 2.5,
                                           "atr_trailing_multiple": 3., "time_stop_sessions": 60})
            hashes, seconds, integrity, materialize = [], [], [], []
            for flag in ("0", "1", "1"):
                os.environ["FF_FACTOR_COMPUTE_CACHE"] = flag
                result, elapsed = timed(run_backtest, **kwargs)
                seconds.append(elapsed)
                hashes.append(digest(evidence(result)))
                integrity.append(result["integrity"]["all_pass"])
                materialize.append(result["execution"]["timing_seconds"]["materialization"])
            record({"market": market, "mode": mode, "kind": "event_backtest", "stress": stress,
                    "seconds_reference_cold_warm": seconds, "materialization_reference_cold_warm": materialize,
                    "exact_equal": len(set(hashes)) == 1, "hashes": hashes,
                    "integrity_pass": all(integrity), "integrity_results": integrity,
                    "gross_targets": [kwargs.get("long_gross_target", 1.), kwargs.get("short_gross_target", 0.)],
                    "cache": cc.cache_stats()})
    report["passed"] = all(row["exact_equal"] and row.get("integrity_pass", True) for row in report["cases"])
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
