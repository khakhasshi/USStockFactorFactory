"""Pure allocation authority shared by event backtests and live execution.

This module deliberately imports no data, database, broker or API components.
Inputs must be frozen at signal close. The caller owns rounding to tradable lots
and the separate execution/cash ledger. Do not replace missing causal inputs
with information observed at the next open.
"""

from __future__ import annotations


def allocation_dollars(
    rows: list[dict],
    *,
    nlv: float,
    gross_target: float,
    position_sizing: str,
    max_position_weight: float,
    risk_per_position_fraction: float,
    atr_stop_multiple: float | None,
) -> list[float]:
    """The event engine's allocation, extracted without arithmetic changes."""
    if not rows or nlv <= 0 or gross_target <= 0:
        return []
    budget = nlv * gross_target
    cap = nlv * max_position_weight
    if position_sizing == "atr_risk":
        stop_multiple = atr_stop_multiple or 2.0
        values = []
        for row in rows:
            atr_pct = float(row.get("_atr_pct") or 0.0)
            risk_dollars = nlv * risk_per_position_fraction / max(0.0025, atr_pct * stop_multiple)
            values.append(min(cap, risk_dollars))
        total = sum(values)
        scale = min(1.0, budget / total) if total > 0 else 0.0
        return [value * scale for value in values]
    if position_sizing == "inverse_volatility":
        raw = [1.0 / max(0.01, float(row.get("_vol20_prev") or 0.0)) for row in rows]
    elif position_sizing == "equal_weight":
        raw = [1.0] * len(rows)
    else:
        raise ValueError("position_sizing 不受支持")
    total = sum(raw)
    values = [min(cap, budget * value / total) for value in raw]
    for _ in range(4):
        residual = budget - sum(values)
        available = [index for index, value in enumerate(values) if value < cap - 1e-8]
        if residual <= 1e-8 or not available:
            break
        share = residual / len(available)
        for index in available:
            values[index] = min(cap, values[index] + share)
    return values


def effective_gross_targets(
    *, long_gross_target: float, short_gross_target: float,
    max_gross_leverage: float, cash_buffer_fraction: float,
) -> tuple[float, float]:
    configured = long_gross_target + short_gross_target
    operating_cap = max_gross_leverage * (1.0 - cash_buffer_fraction)
    scale = min(1.0, max(0.0, operating_cap) / configured) if configured > 1e-12 else 0.0
    return long_gross_target * scale, short_gross_target * scale
