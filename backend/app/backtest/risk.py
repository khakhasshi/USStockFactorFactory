"""Position lifecycle and deterministic daily-bar exit policies.

The event engine deliberately keeps accounting and risk decisions separate.
This module owns only position state and causal exit-level evaluation; fills,
fees, cash and the immutable event ledger remain the responsibility of the
backtester.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any


def _positive_optional(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
        raise ValueError(f"{name} 必须为正数或留空")


@dataclass(frozen=True)
class ExitPolicyConfig:
    """Composable position-exit policy for long and short books.

    Percentages are decimals: ``0.08`` means 8%.  ATR levels are based on the
    signal-close ATR percentage mapped to the actual entry price.  Rolling ATR
    can update at completed closes, never with information from the bar being
    evaluated.
    """

    fixed_stop_loss_pct: float | None = None
    fixed_take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    atr_period: int = 14
    atr_stop_multiple: float | None = None
    atr_take_profit_multiple: float | None = None
    atr_trailing_multiple: float | None = None
    break_even_activation_pct: float | None = None
    time_stop_sessions: int | None = None
    intrabar_conflict_policy: str = "conservative"

    def validate(self) -> None:
        for field_name in (
            "fixed_stop_loss_pct",
            "fixed_take_profit_pct",
            "trailing_stop_pct",
            "atr_stop_multiple",
            "atr_take_profit_multiple",
            "atr_trailing_multiple",
            "break_even_activation_pct",
        ):
            _positive_optional(getattr(self, field_name), field_name)
        for field_name in (
            "fixed_stop_loss_pct",
            "fixed_take_profit_pct",
            "trailing_stop_pct",
            "break_even_activation_pct",
        ):
            value = getattr(self, field_name)
            if value is not None and float(value) >= 1:
                raise ValueError(f"{field_name} 必须小于 1")
        if not 2 <= int(self.atr_period) <= 252:
            raise ValueError("atr_period 必须在 2..252")
        if self.time_stop_sessions is not None and not 1 <= int(self.time_stop_sessions) <= 2520:
            raise ValueError("time_stop_sessions 必须在 1..2520 或留空")
        if self.intrabar_conflict_policy not in {"conservative", "optimistic"}:
            raise ValueError("intrabar_conflict_policy 必须为 conservative 或 optimistic")

    @property
    def enabled(self) -> bool:
        return any(
            value is not None
            for value in (
                self.fixed_stop_loss_pct,
                self.fixed_take_profit_pct,
                self.trailing_stop_pct,
                self.atr_stop_multiple,
                self.atr_take_profit_multiple,
                self.atr_trailing_multiple,
                self.break_even_activation_pct,
                self.time_stop_sessions,
            )
        )

    def snapshot(self) -> dict[str, Any]:
        return asdict(self) | {"enabled": self.enabled}


@dataclass
class PositionState:
    symbol: str
    quantity: float
    avg_entry_price: float
    entry_date: str
    entry_session_index: int
    highest_price: float
    lowest_price: float
    initial_atr: float | None = None
    current_atr: float | None = None
    holding_sessions: int = 0
    settled_quantity: float = 0.0
    entry_fees_remaining: float = 0.0
    realized_pnl: float = 0.0
    mfe_pct: float = 0.0
    mae_pct: float = 0.0

    @property
    def direction(self) -> int:
        return 1 if self.quantity > 0 else -1

    def update_completed_bar(
        self,
        *,
        high: float,
        low: float,
        atr: float | None,
    ) -> None:
        if high > 0:
            self.highest_price = max(self.highest_price, float(high))
        if low > 0:
            self.lowest_price = min(self.lowest_price, float(low))
        if atr is not None and math.isfinite(float(atr)) and float(atr) > 0:
            self.current_atr = float(atr)
        if self.avg_entry_price > 0:
            if self.direction > 0:
                self.mfe_pct = max(
                    self.mfe_pct,
                    self.highest_price / self.avg_entry_price - 1.0,
                )
                self.mae_pct = min(
                    self.mae_pct,
                    self.lowest_price / self.avg_entry_price - 1.0,
                )
            else:
                self.mfe_pct = max(
                    self.mfe_pct,
                    1.0 - self.lowest_price / self.avg_entry_price,
                )
                self.mae_pct = min(
                    self.mae_pct,
                    1.0 - self.highest_price / self.avg_entry_price,
                )
        self.holding_sessions += 1

    def snapshot(self) -> dict[str, Any]:
        return asdict(self) | {"direction": self.direction}


@dataclass(frozen=True)
class ExitTrigger:
    reason: str
    reference_price: float
    trigger_price: float
    gap: bool = False
    ambiguous: bool = False


def risk_levels(
    state: PositionState,
    policy: ExitPolicyConfig,
) -> tuple[float | None, float | None, list[str]]:
    """Return effective stop, take-profit and the mechanisms setting them."""
    direction = state.direction
    entry = state.avg_entry_price
    stop_candidates: list[tuple[float, str]] = []
    target_candidates: list[tuple[float, str]] = []
    initial_atr = state.initial_atr
    current_atr = state.current_atr or initial_atr

    if policy.fixed_stop_loss_pct is not None:
        stop_candidates.append((
            entry * (1.0 - policy.fixed_stop_loss_pct * direction),
            "fixed_stop_loss",
        ))
    if policy.atr_stop_multiple is not None and initial_atr:
        stop_candidates.append((
            entry - direction * policy.atr_stop_multiple * initial_atr,
            "atr_stop_loss",
        ))
    if policy.trailing_stop_pct is not None:
        anchor = state.highest_price if direction > 0 else state.lowest_price
        stop_candidates.append((
            anchor * (1.0 - direction * policy.trailing_stop_pct),
            "trailing_stop",
        ))
    if policy.atr_trailing_multiple is not None and current_atr:
        anchor = state.highest_price if direction > 0 else state.lowest_price
        stop_candidates.append((
            anchor - direction * policy.atr_trailing_multiple * current_atr,
            "atr_trailing_stop",
        ))
    if policy.break_even_activation_pct is not None:
        favorable = (
            state.highest_price / entry - 1.0
            if direction > 0
            else 1.0 - state.lowest_price / entry
        )
        if favorable >= policy.break_even_activation_pct:
            stop_candidates.append((entry, "break_even_stop"))

    if policy.fixed_take_profit_pct is not None:
        target_candidates.append((
            entry * (1.0 + policy.fixed_take_profit_pct * direction),
            "fixed_take_profit",
        ))
    if policy.atr_take_profit_multiple is not None and initial_atr:
        target_candidates.append((
            entry + direction * policy.atr_take_profit_multiple * initial_atr,
            "atr_take_profit",
        ))

    if direction > 0:
        stop = max(stop_candidates, default=(None, ""), key=lambda item: -math.inf if item[0] is None else item[0])
        target = min(target_candidates, default=(None, ""), key=lambda item: math.inf if item[0] is None else item[0])
    else:
        stop = min(stop_candidates, default=(None, ""), key=lambda item: math.inf if item[0] is None else item[0])
        target = max(target_candidates, default=(None, ""), key=lambda item: -math.inf if item[0] is None else item[0])
    mechanisms = [item[1] for item in stop_candidates + target_candidates]
    return stop[0], target[0], mechanisms


def evaluate_exit(
    state: PositionState,
    policy: ExitPolicyConfig,
    *,
    open_price: float,
    high_price: float | None = None,
    low_price: float | None = None,
    stage: str,
) -> ExitTrigger | None:
    """Evaluate a position against levels known before the current bar.

    ``stage='open'`` handles gap and time exits. ``stage='intraday'`` uses the
    completed daily high/low but never updates a trailing anchor until after
    the trigger decision.  When both exits are touched and no intraday path is
    available, the default policy chooses the adverse exit.
    """
    if not policy.enabled:
        return None
    stop, target, _ = risk_levels(state, policy)
    direction = state.direction
    if stage == "open":
        stop_hit = stop is not None and (
            open_price <= stop if direction > 0 else open_price >= stop
        )
        target_hit = target is not None and (
            open_price >= target if direction > 0 else open_price <= target
        )
        if stop_hit:
            return ExitTrigger("risk_stop_gap", open_price, float(stop), gap=True)
        if target_hit:
            return ExitTrigger("risk_take_profit_gap", open_price, float(target), gap=True)
        if (
            policy.time_stop_sessions is not None
            and state.holding_sessions >= policy.time_stop_sessions
        ):
            return ExitTrigger("risk_time_stop", open_price, open_price)
        return None

    if stage != "intraday" or high_price is None or low_price is None:
        return None
    stop_hit = stop is not None and (
        low_price <= stop if direction > 0 else high_price >= stop
    )
    target_hit = target is not None and (
        high_price >= target if direction > 0 else low_price <= target
    )
    if stop_hit and target_hit:
        choose_stop = policy.intrabar_conflict_policy == "conservative"
        return ExitTrigger(
            "risk_stop_intraday" if choose_stop else "risk_take_profit_intraday",
            float(stop if choose_stop else target),
            float(stop if choose_stop else target),
            ambiguous=True,
        )
    if stop_hit:
        return ExitTrigger("risk_stop_intraday", float(stop), float(stop))
    if target_hit:
        return ExitTrigger("risk_take_profit_intraday", float(target), float(target))
    return None
