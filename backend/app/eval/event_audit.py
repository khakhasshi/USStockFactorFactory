"""Mandatory Python event-ledger confirmation; never proposal feedback."""
from __future__ import annotations

import math
from pathlib import Path

from ..backtest.engine import run_backtest
from ..config import get_layer_bounds

PROTOCOL = "factorfactory.event-promotion-gate/v1"


def event_metric_gate(result: dict, cfg: dict, *, minimum_sharpe=None) -> dict:
    stats = result.get("stats") or {}
    reasons = []
    def finite(key):
        value = stats.get({"ann_return": "ann_ret", "max_drawdown": "max_dd", "daily_turnover": "avg_daily_turnover"}.get(key, key))
        try:
            return float(value) if value is not None and math.isfinite(float(value)) else None
        except (ValueError, TypeError, OverflowError):
            return None
    from ..factor_lifecycle import _frozen_input_pass
    provenance = result.get("input_provenance") or {}
    if not _frozen_input_pass(provenance, provenance.get("expression", "")):
        reasons.append("event_immutable_provenance_missing")
    if not (result.get("integrity") or {}).get("all_pass"):
        reasons.append("event_ledger_integrity_failed")
    if int(stats.get("open_positions") or 0) > 0:
        reasons.append("event_terminal_liquidation_incomplete")
    if (finite("sharpe") is None or finite("sharpe") < (float(cfg["min_oos_sharpe"]) if minimum_sharpe is None else minimum_sharpe)):
        reasons.append("event_net_sharpe_below_gate")
    if finite("ann_return") is None or finite("ann_return") <= 0:
        reasons.append("event_net_return_not_positive")
    if finite("max_drawdown") is None or finite("max_drawdown") > float(cfg["max_drawdown"]):
        reasons.append("event_daily_drawdown_above_gate")
    if finite("daily_turnover") is None or finite("daily_turnover") > float(cfg["max_daily_turnover"]):
        reasons.append("event_turnover_above_gate")
    return {"passed": not reasons, "failure_reasons": reasons, "stats": stats,
            "integrity": result.get("integrity"), "input_provenance": result.get("input_provenance"),
            "manifest": result.get("artifacts")}


def run_event_audit(expression, universe_n, horizon, mode, direction, market, panel_glob, cfg,
                    snapshot, artifact_root: str | Path | None = None) -> dict:
    bounds = get_layer_bounds(market)
    dates = list(snapshot.trading_dates)
    windows = {"holdout": bounds["META_HOLDOUT"], "vault": bounds["FACTOR_VAULT"],
               "rating": ("2020-01-01", str(dates[-1]))}
    results = {}
    for name, (start, end) in windows.items():
        window_dates = [d for d in dates if start <= str(d) <= end]
        if name != "rating":
            window_dates = window_dates[horizon:]
        if len(window_dates) < int(cfg["min_layer_days"]):
            results[name] = {"passed": False, "status": "INSUFFICIENT_DATA", "failure_reasons": ["event_window_too_short"]}
            continue
        try:
            result = run_backtest(
                expression, universe_n, str(window_dates[0]), str(window_dates[-1]),
                direction=direction, mode=mode, market=market, panel_glob=panel_glob,
                initial_capital=float(cfg["target_capital"]), rebalance_every=horizon,
                top_fraction=float(cfg["top_fraction"]),
                slippage_bps=float(cfg["base_cost_bps"]),
                borrow_cost_bps_annual=float(cfg["borrow_cost_bps_annual"]),
                max_volume_participation=float(cfg["max_adv_participation"]),
                long_gross_target=1.0, short_gross_target=1.0 if mode == "long_short" else 0.0,
                max_gross_leverage=2.0, liquidate_at_end=True,
                execution_backend="python", response_trade_limit=0, response_daily_limit=0,
                artifact_dir=Path(artifact_root) / name if artifact_root else None,
                capture_detail=True, monte_carlo_enabled=False,
            )
            gate = event_metric_gate(result, cfg)
            gate.update({"status": "PASS" if gate["passed"] else "FAIL",
                         "actual_start": str(window_dates[0]), "actual_end": str(window_dates[-1]),
                         "market_sessions": len(window_dates), "config": result.get("config")})
            if float(cfg["tail_fraction"]) != float(cfg["top_fraction"]) and mode == "long_short":
                gate["passed"] = False
                gate["status"] = "FAIL"
                gate["failure_reasons"].append("asymmetric_tail_not_supported_by_event_contract")
            results[name] = gate
        except Exception as exc:
            results[name] = {"passed": False, "status": "ERROR", "failure_reasons": [type(exc).__name__ + ": " + str(exc)[:500]]}
    return {"protocol": PROTOCOL, "passed": all(v["passed"] for v in results.values()),
            "status": "PASS" if all(v["passed"] for v in results.values()) else "FAIL",
            "windows": results, "visible_to_research_llms": False,
            "contract": {"engine": "step_event_v2", "authority": "python_daily_event_ledger",
                "direction_frozen": direction, "universe_n": universe_n, "rebalance_sessions": horizon,
                "entry": "t_close_signal_t_plus_1_open", "terminal_liquidation": True,
                "cost_policy": "base_bps_execution_slippage_plus_market_fee_schedule_and_borrow",
                "note": "Event daily MDD and net PnL are mandatory confirmation, not equality with vector proxies."}}
