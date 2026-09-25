#!/usr/bin/env python3
"""Real-data paired reference/optimized replay, no production task/DB writes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.backtest.engine import run_backtest
from backend.app.eval.harness import evaluate
from backend.app.config import default_panel_glob

EXPRESSION = "rank((delay(close,21)/delay(close,252)-1)/(ts_std(ts_delta(close,1),60)+1e-9)*(ts_mean(amount,20)/(ts_mean(amount,120)+1e-9)))"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def evidence(result):
    return {k: v for k, v in result.items() if k not in {
        "execution", "input_provenance", "artifacts", "export_metadata",
    }}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--end", default="2026-09-23")
    args = p.parse_args()
    out = Path(args.output)
    if out.exists():
        raise SystemExit("Refusing to overwrite existing benchmark")
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {"protocol": "exact_python_optimization_parity_v1", "cases": [], "evaluation": [],
              "expression": EXPRESSION, "end": args.end,
              "engine_sha256": hashlib.sha256((ROOT / "backend/app/backtest/engine.py").read_bytes()).hexdigest(),
              "harness_sha256": hashlib.sha256((ROOT / "backend/app/eval/harness.py").read_bytes()).hexdigest()}
    def checkpoint():
        temp = out.with_suffix(".tmp")
        temp.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        temp.replace(out)
    for market, mode in (("us", "long_only"), ("us", "long_short"), ("ashare", "long_only")):
        # Warm panel and immutable source snapshot; don't credit cold-load savings.
        evaluate(EXPRESSION, market=market, portfolio_mode=mode, panel_glob=default_panel_glob(market))
        eval_rows, eval_hashes = {}, {}
        for flag in ("0", "1"):
            os.environ["FF_EVALUATION_WINDOW_OPTIMIZATIONS"] = flag
            started = time.perf_counter()
            value = evaluate(EXPRESSION, market=market, portfolio_mode=mode, panel_glob=default_panel_glob(market))
            eval_rows[flag] = round(time.perf_counter() - started, 6)
            # Runtime contains timing and raw path evidence, not score contract.
            eval_hashes[flag] = digest({k: v for k, v in value.items() if k not in {"runtime", "raw_return_evidence"}})
        row = {"market": market, "mode": mode, "seconds": eval_rows, "exact_equal": eval_hashes["0"] == eval_hashes["1"]}
        report["evaluation"].append(row)
        print(json.dumps(row), flush=True)
        for stress in (False, True):
            kw = dict(expression=EXPRESSION, universe_n=500, start="2023-01-01" if stress else "2021-01-01",
                      end=args.end, market=market, mode=mode, panel_glob=default_panel_glob(market),
                      rebalance_every=5, top_fraction=.2, execution_backend="python",
                      capture_detail=True, response_trade_limit=10**9, response_daily_limit=None,
                      monte_carlo_enabled=True, monte_carlo_simulations=2000)
            if stress:
                kw.update(position_sizing="inverse_volatility", unfilled_order_policy="carry",
                          liquidate_at_end=True, portfolio_stop_drawdown_pct=.15,
                          portfolio_daily_loss_pct=.05, risk_cooldown_sessions=5,
                          impact_model="square_root", impact_coefficient_bps=10.,
                          exit_policy={"fixed_stop_loss_pct": .08, "fixed_take_profit_pct": .2,
                                       "atr_stop_multiple": 2.5, "atr_trailing_multiple": 3.,
                                       "break_even_activation_pct": .1, "time_stop_sessions": 60})
            timings = {"0": [], "1": []}
            hashes = set()
            passes = []
            for repeat in range(args.repeats):
                for flag in (("0", "1") if repeat % 2 == 0 else ("1", "0")):
                    os.environ["FF_PYTHON_EVENT_OPTIMIZATIONS"] = flag
                    started = time.perf_counter()
                    result = run_backtest(**kw)
                    timings[flag].append(time.perf_counter() - started)
                    hashes.add(digest(evidence(result)))
                    passes.append(result["integrity"]["all_pass"])
            row = {"market": market, "mode": mode, "stress": stress,
                   "seconds": timings, "speedup": statistics.median(timings["0"]) / statistics.median(timings["1"]),
                   "exact_equal": len(hashes) == 1, "integrity_pass": all(passes),
                   "days": result["stats"]["days"], "fills": result["stats"]["fills"],
                   "ledger_hashes": sorted(hashes), "data_snapshot": result["input_provenance"], "config": result["config"]}
            report["cases"].append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in {"data_snapshot", "config"}}), flush=True)
            checkpoint()
    report["passed"] = all(r["exact_equal"] for r in report["evaluation"]) and all(
        r["exact_equal"] and r["integrity_pass"] for r in report["cases"])
    checkpoint()
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
