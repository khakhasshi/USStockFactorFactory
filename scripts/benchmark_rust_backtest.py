#!/usr/bin/env python3
"""Reproducible dual-market Python/Rust event-kernel alignment benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.backtest.engine import run_backtest
from backend.app.config import default_panel_glob


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", default="rank(returns(close, 20))")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--universe", type=int, default=100)
    args = parser.parse_args()
    failures = 0
    for market, mode in (
        ("ashare", "long_only"),
        ("us", "long_only"),
        ("us", "long_short"),
    ):
        result = run_backtest(
            args.expression,
            universe_n=args.universe,
            start=args.start,
            end=args.end,
            direction=1,
            mode=mode,
            panel_glob=default_panel_glob(market),
            market=market,
            top_fraction=0.1,
            rebalance_every=5,
            slippage_bps=2.0,
            max_volume_participation=0.1,
            borrow_cost_bps_annual=300.0 if mode == "long_short" else 0.0,
            response_trade_limit=0,
            response_daily_limit=0,
            capture_detail=True,
            execution_backend="rust_shadow",
        )
        execution = result["execution"]
        alignment = execution.get("alignment") or {}
        row = {
            "market": market,
            "mode": mode,
            "days": result["stats"]["days"],
            "fills": result["stats"]["fills"],
            "final_nlv": result["stats"]["final_nlv"],
            "integrity_pass": result["integrity"]["all_pass"],
            "alignment_pass": alignment.get("all_pass", False),
            "trade_identity_mismatches": alignment.get("trade_identity_mismatches"),
            "max_trade_numeric_error": alignment.get("max_trade_numeric_error"),
            "max_daily_numeric_error": alignment.get("max_daily_numeric_error"),
            **execution["timing_seconds"],
        }
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        failures += int(not row["integrity_pass"] or not row["alignment_pass"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
