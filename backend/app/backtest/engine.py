"""Chronological step-event equity backtester.

Protocol
--------
1. A factor is observed only after session ``t`` closes.
2. Target-share orders are created at that close.
3. Orders are eligible to fill at raw open ``t+1`` (never on signal day).
4. Every fill updates cash and positions, with an explicit fee breakdown.
5. The same ledger is the source of NAV, statistics, CSV statements, and
   integrity checks.  There is no separate vector-return truth path.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import polars as pl

from ..config import DEFAULT_PORTFOLIO_MODE, get_dsl_fields
from ..data.panel import PanelStore
from ..dsl.engine import parse, required_history
from .fees import (
    ASHARE_WAN2_NO_MIN_PROFILE,
    IBKR_PRO_FIXED_PROFILE,
    calculate_trade_fees,
    fee_schedule_snapshot,
)
from .risk import (
    ExitPolicyConfig,
    PositionState,
    evaluate_exit,
    risk_levels,
)

BACKTEST_PROTOCOL = "step_event_v2"
_EVENT_PHASE_ORDER = {
    "SESSION_OPEN": 0,
    "OPEN_CORPORATE_ACTION": 1,
    "OPEN_RISK_TRIGGER": 2,
    "OPEN_EXECUTION": 2,
    "INTRADAY_RISK_TRIGGER": 3,
    "INTRADAY_EXECUTION": 3,
    "CLOSE_FINANCING": 4,
    "CLOSE_RISK": 5,
    "CLOSE_SIGNAL": 6,
    "SESSION_CLOSE": 7,
}


def _finite(value: Any) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _round(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


@dataclass(frozen=True)
class EventBacktestConfig:
    market: str = "us"
    mode: str = DEFAULT_PORTFOLIO_MODE
    direction: int = 1
    universe_n: int = 500
    top_fraction: float = 0.20
    initial_capital: float = 1_000_000.0
    rebalance_every: int = 5
    slippage_bps: float = 2.0
    max_volume_participation: float = 0.10
    borrow_cost_bps_annual: float = 0.0
    fee_profile: str | None = None
    account_type: str = "auto"
    cash_buffer_fraction: float = 0.0
    max_gross_leverage: float = 2.0
    margin_interest_bps_annual: float = 0.0
    position_sizing: str = "equal_weight"
    max_positions: int = 10_000
    max_position_weight: float = 1.0
    min_trade_notional: float = 0.0
    rebalance_buffer_pct: float = 0.0
    long_gross_target: float = 1.0
    short_gross_target: float = 1.0
    risk_per_position_fraction: float = 0.01
    spread_bps: float = 0.0
    impact_model: str = "fixed"
    impact_coefficient_bps: float = 0.0
    unfilled_order_policy: str = "cancel"
    max_order_age_sessions: int = 1
    max_stale_sessions: int = 20
    liquidate_at_end: bool = False
    portfolio_stop_drawdown_pct: float | None = None
    portfolio_daily_loss_pct: float | None = None
    risk_cooldown_sessions: int = 0
    exit_policy: ExitPolicyConfig = field(default_factory=ExitPolicyConfig)

    def validate(self) -> None:
        if self.market not in {"us", "ashare"}:
            raise ValueError("market 必须是 us 或 ashare")
        if self.mode not in {"long_short", "long_only"}:
            raise ValueError("mode 必须是 long_short 或 long_only")
        if self.market == "ashare" and self.mode != "long_only":
            raise ValueError("A股事件回测只允许纯多头")
        if self.direction not in {-1, 1}:
            raise ValueError("direction 必须为 1 或 -1")
        if not 1 <= self.universe_n <= 10_000:
            raise ValueError("universe_n 必须在 1..10000")
        if not 0 < self.top_fraction <= 0.5:
            raise ValueError("top_fraction 必须在 (0, 0.5]")
        if self.initial_capital <= 0:
            raise ValueError("initial_capital 必须为正数")
        if not 1 <= self.rebalance_every <= 252:
            raise ValueError("rebalance_every 必须在 1..252")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps 不能为负数")
        if not 0 < self.max_volume_participation <= 1:
            raise ValueError("max_volume_participation 必须在 (0, 1]")
        if self.borrow_cost_bps_annual < 0:
            raise ValueError("borrow_cost_bps_annual 不能为负数")
        if self.account_type not in {"auto", "cash", "margin"}:
            raise ValueError("account_type 必须为 auto、cash 或 margin")
        if self.mode == "long_short" and self.account_type == "cash":
            raise ValueError("long_short 不能使用 cash 账户")
        if not 0 <= self.cash_buffer_fraction < 1:
            raise ValueError("cash_buffer_fraction 必须在 [0, 1)")
        if not 1 <= self.max_gross_leverage <= 10:
            raise ValueError("max_gross_leverage 必须在 1..10")
        if self.margin_interest_bps_annual < 0:
            raise ValueError("margin_interest_bps_annual 不能为负数")
        if self.position_sizing not in {"equal_weight", "inverse_volatility", "atr_risk"}:
            raise ValueError("position_sizing 不受支持")
        if not 1 <= self.max_positions <= 10_000:
            raise ValueError("max_positions 必须在 1..10000")
        if not 0 < self.max_position_weight <= self.max_gross_leverage:
            raise ValueError("max_position_weight 必须为正且不超过最大总杠杆")
        if self.min_trade_notional < 0:
            raise ValueError("min_trade_notional 不能为负数")
        if not 0 <= self.rebalance_buffer_pct < 1:
            raise ValueError("rebalance_buffer_pct 必须在 [0, 1)")
        if not 0 <= self.long_gross_target <= self.max_gross_leverage:
            raise ValueError("long_gross_target 超出范围")
        if not 0 <= self.short_gross_target <= self.max_gross_leverage:
            raise ValueError("short_gross_target 超出范围")
        if self.mode == "long_only" and self.short_gross_target != 0:
            object.__setattr__(self, "short_gross_target", 0.0)
        if self.long_gross_target + self.short_gross_target > self.max_gross_leverage + 1e-12:
            raise ValueError("多空目标总敞口超过 max_gross_leverage")
        if not 0 < self.risk_per_position_fraction <= 0.25:
            raise ValueError("risk_per_position_fraction 必须在 (0, 0.25]")
        if self.spread_bps < 0 or self.impact_coefficient_bps < 0:
            raise ValueError("价差和冲击参数不能为负数")
        if self.impact_model not in {"fixed", "linear", "square_root"}:
            raise ValueError("impact_model 必须为 fixed、linear 或 square_root")
        if self.unfilled_order_policy not in {"cancel", "carry"}:
            raise ValueError("unfilled_order_policy 必须为 cancel 或 carry")
        if not 1 <= self.max_order_age_sessions <= 20:
            raise ValueError("max_order_age_sessions 必须在 1..20")
        if not 1 <= self.max_stale_sessions <= 252:
            raise ValueError("max_stale_sessions 必须在 1..252")
        for name in ("portfolio_stop_drawdown_pct", "portfolio_daily_loss_pct"):
            value = getattr(self, name)
            if value is not None and not 0 < value < 1:
                raise ValueError(f"{name} 必须在 (0, 1) 或留空")
        if not 0 <= self.risk_cooldown_sessions <= 252:
            raise ValueError("risk_cooldown_sessions 必须在 0..252")
        self.exit_policy.validate()

    @property
    def resolved_fee_profile(self) -> str:
        if self.fee_profile:
            return self.fee_profile
        return (
            ASHARE_WAN2_NO_MIN_PROFILE
            if self.market == "ashare"
            else IBKR_PRO_FIXED_PROFILE
        )

    @property
    def currency(self) -> str:
        return "CNY" if self.market == "ashare" else "USD"

    @property
    def lot_size(self) -> int:
        return 100 if self.market == "ashare" else 1

    @property
    def resolved_account_type(self) -> str:
        if self.account_type != "auto":
            return self.account_type
        return "margin" if self.mode == "long_short" else "cash"

    @property
    def configured_gross_target(self) -> float:
        return self.long_gross_target + self.short_gross_target

    @property
    def effective_gross_target_scale(self) -> float:
        """Keep an explicit operating band below the hard leverage ceiling."""
        configured = self.configured_gross_target
        if configured <= 1e-12:
            return 0.0
        operating_cap = self.max_gross_leverage * (1.0 - self.cash_buffer_fraction)
        return min(1.0, max(0.0, operating_cap) / configured)

    @property
    def effective_long_gross_target(self) -> float:
        return self.long_gross_target * self.effective_gross_target_scale

    @property
    def effective_short_gross_target(self) -> float:
        return self.short_gross_target * self.effective_gross_target_scale

    @property
    def leverage_headroom(self) -> float:
        return max(
            0.0,
            self.max_gross_leverage
            - self.effective_long_gross_target
            - self.effective_short_gross_target,
        )


class StepEventBacktester:
    """Stateful engine whose ``step`` method advances exactly one session."""

    def __init__(
        self,
        config: EventBacktestConfig,
        *,
        capture_detail: bool = True,
    ) -> None:
        config.validate()
        # Validate the selected fee profile before the first event is emitted.
        fee_schedule_snapshot(config.market, config.resolved_fee_profile)
        self.config = config
        self.capture_detail = bool(capture_detail)
        self.cash = float(config.initial_capital)
        self.positions: dict[str, float] = {}
        self.position_states: dict[str, PositionState] = {}
        self.names: dict[str, str] = {}
        self.last_close: dict[str, float] = {}
        self.last_adjustment: dict[str, float] = {}
        self.pending_orders: list[dict] = []
        self.trades: list[dict] = []
        self.events: list[dict] = []
        self.daily: list[dict] = []
        self._event_seq = 0
        self._order_seq = 0
        self._fill_seq = 0
        self._previous_close_nlv = float(config.initial_capital)
        self._cumulative_fees = 0.0
        self._cumulative_slippage = 0.0
        self._cumulative_borrow = 0.0
        self._cumulative_margin_interest = 0.0
        self._cumulative_traded_notional = 0.0
        self._rejected_orders = 0
        self._partial_orders = 0
        self._executed_orders = 0
        self._requested_execution_quantity = 0.0
        self._filled_execution_quantity = 0.0
        self._session_index = -1
        self._session_filled_by_symbol: dict[str, float] = {}
        self._blocked_symbols_today: set[str] = set()
        self._missing_position_sessions: dict[str, int] = {}
        self._round_trips: list[dict] = []
        self._exit_reason_counts: dict[str, int] = {}
        self._peak_nlv = float(config.initial_capital)
        self._risk_cooldown_remaining = 0
        self._portfolio_liquidations = 0
        self._portfolio_risk_active = False
        self._portfolio_risk_trigger_events = 0
        self._portfolio_risk_rearms = 0
        self._portfolio_risk_active_sessions = 0
        self._portfolio_risk_last_trigger: str | None = None
        self._max_portfolio_risk_cycle_drawdown = 0.0
        self._online_integrity = {
            "cash_reconciliation_max_error": 0.0,
            "fee_formula_max_error": 0.0,
            "fee_component_sum_max_error": 0.0,
            "gross_amount_max_error": 0.0,
            "slippage_formula_max_error": 0.0,
            "position_formula_max_error": 0.0,
            "same_day_signal_fill_violations": 0,
            "scheduled_execution_date_violations": 0,
            "duplicate_fill_id_violations": 0,
            "nonpositive_fill_violations": 0,
            "side_sign_violations": 0,
            "fee_profile_violations": 0,
            "ashare_buy_lot_violations": 0,
            "event_phase_order_violations": 0,
            "cash_account_negative_cash_violations": 0,
            "gross_leverage_violations": 0,
            "gross_leverage_breach_events": 0,
            "automatic_deleveraging_events": 0,
            "leverage_limited_orders": 0,
            "max_open_gross_leverage_observed": 0.0,
            "max_open_gross_leverage_after_control": 0.0,
            "position_state_quantity_violations": 0,
            "stale_position_writeoffs": 0,
            "intrabar_ambiguities": 0,
        }
        self._seen_fill_ids: set[str] = set()
        self._last_event_date = ""
        self._last_event_phase = -1
        self._gross_leverage_violation_details: list[dict] = []

    def _emit(
        self,
        trade_date: date,
        phase: str,
        event_type: str,
        *,
        symbol: str = "",
        order_id: str = "",
        message: str = "",
        payload: dict | None = None,
    ) -> None:
        self._event_seq += 1
        date_key = str(trade_date)
        phase_order = _EVENT_PHASE_ORDER.get(phase, 99)
        if date_key == self._last_event_date and phase_order < self._last_event_phase:
            self._online_integrity["event_phase_order_violations"] += 1
        if date_key != self._last_event_date:
            self._last_event_date = date_key
            self._last_event_phase = -1
        self._last_event_phase = phase_order
        if self.capture_detail:
            self.events.append({
                "seq": self._event_seq,
                "trade_date": date_key,
                "phase": phase,
                "event_type": event_type,
                "symbol": symbol,
                "order_id": order_id,
                "message": message,
                "payload_json": json.dumps(
                    payload or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            })

    def _audit_trade_online(self, trade: dict) -> None:
        """Reconcile one transient fill even when its statement is not retained."""
        signed = (
            trade["filled_quantity"]
            if trade["side"] == "BUY"
            else -trade["filled_quantity"]
        )
        values = {
            "cash_reconciliation_max_error": abs(
                trade["cash_before"]
                - signed * trade["fill_price"]
                - trade["total_fees"]
                - trade["cash_after"]
            ),
            "gross_amount_max_error": abs(
                trade["filled_quantity"] * trade["fill_price"]
                - trade["gross_amount"]
            ),
            "slippage_formula_max_error": abs(
                trade["filled_quantity"]
                * abs(trade["fill_price"] - trade["reference_price"])
                - trade["slippage_cost"]
            ),
            "position_formula_max_error": abs(
                trade["position_before"] + signed - trade["position_after"]
            ),
            "fee_component_sum_max_error": abs(
                trade["commission"]
                + trade["stamp_duty"]
                + trade["transfer_fee"]
                + trade["regulatory_fee"]
                + trade["exchange_fee"]
                - trade["total_fees"]
            ),
        }
        recalculated = calculate_trade_fees(
            market=trade["market"],
            side=trade["side"],
            quantity=trade["filled_quantity"],
            price=trade["fill_price"],
            trade_date=date.fromisoformat(trade["trade_date"]),
            profile=trade["fee_profile"],
        )
        values["fee_formula_max_error"] = abs(
            recalculated.total - trade["total_fees"]
        )
        for key, value in values.items():
            self._online_integrity[key] = max(
                float(self._online_integrity[key]),
                float(value),
            )
        if (
            trade.get("reason") == "factor_rebalance_next_open"
            and trade["signal_date"] >= trade["trade_date"]
        ):
            self._online_integrity["same_day_signal_fill_violations"] += 1
        if trade.get("scheduled_execute_date") != trade["trade_date"]:
            self._online_integrity["scheduled_execution_date_violations"] += 1
        if trade["fill_id"] in self._seen_fill_ids:
            self._online_integrity["duplicate_fill_id_violations"] += 1
        self._seen_fill_ids.add(trade["fill_id"])
        if trade["filled_quantity"] <= 0 or trade["fill_price"] <= 0:
            self._online_integrity["nonpositive_fill_violations"] += 1
        if (
            trade["side"] == "BUY"
            and trade["position_after"] < trade["position_before"] - 1e-8
        ) or (
            trade["side"] == "SELL"
            and trade["position_after"] > trade["position_before"] + 1e-8
        ):
            self._online_integrity["side_sign_violations"] += 1
        if trade["fee_profile"] != self.config.resolved_fee_profile:
            self._online_integrity["fee_profile_violations"] += 1
        if (
            self.config.market == "ashare"
            and trade["side"] == "BUY"
            and abs(trade["filled_quantity"] % self.config.lot_size) > 1e-8
        ):
            self._online_integrity["ashare_buy_lot_violations"] += 1

    def _price(self, symbol: str, market: dict[str, dict], field: str) -> float:
        row = market.get(symbol)
        if row and _finite(row.get(field)) and float(row[field]) > 0:
            return float(row[field])
        return float(self.last_close.get(symbol, 0.0))

    def _nlv(self, market: dict[str, dict], field: str) -> tuple[float, float, float]:
        long_value = 0.0
        short_value = 0.0
        for symbol, quantity in self.positions.items():
            price = self._price(symbol, market, field)
            value = quantity * price
            if value >= 0:
                long_value += value
            else:
                short_value += value
        return self.cash + long_value + short_value, long_value, short_value

    def _gross_leverage(
        self,
        market: dict[str, dict],
        field: str = "raw_open",
    ) -> tuple[float, float, float]:
        nlv, long_value, short_value = self._nlv(market, field)
        gross_value = long_value + abs(short_value)
        leverage = gross_value / nlv if nlv > 1e-12 else math.inf
        return nlv, gross_value, leverage

    def _project_fill_leverage(
        self,
        *,
        symbol: str,
        signed_quantity: float,
        fill_price: float,
        trade_date: date,
        market: dict[str, dict],
    ) -> tuple[float, float, float]:
        """Project NLV and gross leverage using the actual executable price."""
        current = float(self.positions.get(symbol, 0.0))
        valuation_price = self._price(symbol, market, "raw_open")
        current_nlv, current_gross, _ = self._gross_leverage(market, "raw_open")
        after = current + signed_quantity
        gross_after = (
            current_gross
            - abs(current) * valuation_price
            + abs(after) * valuation_price
        )
        fees = calculate_trade_fees(
            market=self.config.market,
            side="BUY" if signed_quantity > 0 else "SELL",
            quantity=abs(signed_quantity),
            price=fill_price,
            trade_date=trade_date,
            profile=self.config.resolved_fee_profile,
        )
        # A fill executed away from the valuation reference and its fees both
        # reduce NLV.  Cash transfer at the reference itself is balance-sheet
        # neutral, including when opening or covering a short.
        nlv_after = (
            current_nlv
            - signed_quantity * (fill_price - valuation_price)
            - fees.total
        )
        leverage_after = gross_after / nlv_after if nlv_after > 1e-12 else math.inf
        return nlv_after, gross_after, leverage_after

    def _leverage_capped_quantity(
        self,
        *,
        symbol: str,
        requested_abs: float,
        signed_direction: float,
        fill_price: float,
        trade_date: date,
        market: dict[str, dict],
    ) -> tuple[float, bool, float]:
        """Cap only the exposure-increasing part of a proposed fill.

        Reductions and covers are always allowed, even when the account is
        already above its ceiling.  Any part that opens or adds exposure is
        admitted only if the post-fee, post-slippage balance sheet remains
        within the declared hard limit.
        """
        requested_abs = max(0.0, float(requested_abs))
        signed_direction = 1.0 if signed_direction > 0 else -1.0
        current = float(self.positions.get(symbol, 0.0))
        full_signed = signed_direction * requested_abs
        _, _, full_leverage = self._project_fill_leverage(
            symbol=symbol,
            signed_quantity=full_signed,
            fill_price=fill_price,
            trade_date=trade_date,
            market=market,
        )
        if abs(current + full_signed) <= abs(current) + 1e-10:
            return requested_abs, False, full_leverage
        if full_leverage <= self.config.max_gross_leverage + 1e-10:
            return requested_abs, False, full_leverage

        # When an order crosses zero, the closing portion is unconditionally
        # safe.  Search only the new exposure beyond that point.
        reducing_base = (
            min(requested_abs, abs(current))
            if current * signed_direction < 0
            else 0.0
        )
        low, high = reducing_base, requested_abs
        if reducing_base > 0:
            _, _, base_leverage = self._project_fill_leverage(
                symbol=symbol,
                signed_quantity=signed_direction * reducing_base,
                fill_price=fill_price,
                trade_date=trade_date,
                market=market,
            )
            if base_leverage > self.config.max_gross_leverage + 1e-10:
                return reducing_base, True, full_leverage
        else:
            _, _, current_leverage = self._gross_leverage(market, "raw_open")
            if current_leverage > self.config.max_gross_leverage + 1e-10:
                return 0.0, True, full_leverage

        for _ in range(48):
            middle = (low + high) / 2.0
            _, _, leverage = self._project_fill_leverage(
                symbol=symbol,
                signed_quantity=signed_direction * middle,
                fill_price=fill_price,
                trade_date=trade_date,
                market=market,
            )
            if leverage <= self.config.max_gross_leverage + 1e-10:
                low = middle
            else:
                high = middle

        lot = float(self.config.lot_size)
        opening_units = math.floor(max(0.0, low - reducing_base) / lot + 1e-12)
        allowed = reducing_base + opening_units * lot
        return min(requested_abs, _round(allowed, 6)), True, full_leverage

    def _settle_positions_for_session(self) -> None:
        if self.config.market != "ashare":
            return
        for state in self.position_states.values():
            if state.quantity > 0:
                state.settled_quantity = abs(state.quantity)

    def _position_state_error(self) -> float:
        symbols = set(self.positions) | set(self.position_states)
        return max(
            (
                abs(
                    float(self.positions.get(symbol, 0.0))
                    - float(self.position_states.get(symbol).quantity if symbol in self.position_states else 0.0)
                )
                for symbol in symbols
            ),
            default=0.0,
        )

    def _liquidity_capacity(self, row: dict) -> tuple[float, str]:
        adv = row.get("_adv20_prev")
        basis = "previous_20_session_adv"
        if not _finite(adv) or float(adv) <= 0:
            # Compatibility for direct StepEventBacktester integrations.  Real
            # panel runs always materialize the causal previous-session ADV.
            adv = row.get("vol")
            basis = "same_row_volume_fallback"
        return max(0.0, float(adv or 0.0)), basis

    def _impact_bps(self, participation: float) -> float:
        base = self.config.slippage_bps + self.config.spread_bps / 2.0
        if self.config.impact_model == "linear":
            return base + self.config.impact_coefficient_bps * participation
        if self.config.impact_model == "square_root":
            return base + self.config.impact_coefficient_bps * math.sqrt(max(0.0, participation))
        return base

    def _max_affordable_buy(
        self,
        *,
        requested: float,
        fill_price: float,
        trade_date: date,
    ) -> float:
        if self.config.resolved_account_type != "cash":
            return requested
        lot = self.config.lot_size
        reserve = max(0.0, self._previous_close_nlv) * self.config.cash_buffer_fraction
        available = max(0.0, self.cash - reserve)
        quantity = math.floor(min(requested, available / max(fill_price, 1e-12)) / lot) * lot
        while quantity > 0:
            fee = calculate_trade_fees(
                market=self.config.market,
                side="BUY",
                quantity=quantity,
                price=fill_price,
                trade_date=trade_date,
                profile=self.config.resolved_fee_profile,
            )
            if quantity * fill_price + fee.total <= available + 1e-9:
                return float(quantity)
            quantity -= lot
        return 0.0

    def _record_round_trip(
        self,
        *,
        state: PositionState,
        close_quantity: float,
        exit_price: float,
        exit_date: date,
        reason: str,
        entry_fee: float,
        exit_fee: float,
    ) -> float:
        direction = state.direction
        gross = close_quantity * (exit_price - state.avg_entry_price) * direction
        net = gross - entry_fee - exit_fee
        trade_return = (
            net / max(1e-12, close_quantity * state.avg_entry_price)
        )
        self._round_trips.append({
            "symbol": state.symbol,
            "entry_date": state.entry_date,
            "exit_date": str(exit_date),
            "direction": direction,
            "quantity": _round(close_quantity, 6),
            "entry_price": _round(state.avg_entry_price, 6),
            "exit_price": _round(exit_price, 6),
            "gross_pnl": _round(gross, 6),
            "net_pnl": _round(net, 6),
            "return": _round(trade_return, 8),
            "holding_sessions": state.holding_sessions,
            "mfe_pct": _round(state.mfe_pct, 8),
            "mae_pct": _round(state.mae_pct, 8),
            "exit_reason": reason,
        })
        self._exit_reason_counts[reason] = self._exit_reason_counts.get(reason, 0) + 1
        return net

    def _update_position_state_for_fill(
        self,
        *,
        symbol: str,
        signed_quantity: float,
        fill_price: float,
        trade_date: date,
        atr_pct: float | None,
        fees: float,
        reason: str,
    ) -> tuple[float | None, float, int | None]:
        state = self.position_states.get(symbol)
        old_quantity = float(state.quantity) if state else 0.0
        new_quantity = _round(old_quantity + signed_quantity, 6)
        entry_before = state.avg_entry_price if state else None
        realized = 0.0
        holding = state.holding_sessions if state else None
        if state is None or abs(old_quantity) < 1e-8:
            initial_atr = (
                fill_price * float(atr_pct)
                if _finite(atr_pct) and float(atr_pct) > 0
                else None
            )
            created = PositionState(
                symbol=symbol,
                quantity=new_quantity,
                avg_entry_price=fill_price,
                entry_date=str(trade_date),
                entry_session_index=self._session_index,
                highest_price=fill_price,
                lowest_price=fill_price,
                initial_atr=initial_atr,
                current_atr=initial_atr,
                settled_quantity=(abs(new_quantity) if self.config.market != "ashare" or new_quantity < 0 else 0.0),
                entry_fees_remaining=fees,
            )
            self.position_states[symbol] = created
            return entry_before, realized, holding

        same_direction_add = old_quantity * signed_quantity > 0
        if same_direction_add:
            old_abs = abs(old_quantity)
            add_abs = abs(signed_quantity)
            state.avg_entry_price = (
                state.avg_entry_price * old_abs + fill_price * add_abs
            ) / max(1e-12, old_abs + add_abs)
            state.quantity = new_quantity
            state.entry_fees_remaining += fees
            if self.config.market != "ashare" or new_quantity < 0:
                state.settled_quantity = abs(new_quantity)
            return entry_before, realized, holding

        close_quantity = min(abs(old_quantity), abs(signed_quantity))
        close_fraction = close_quantity / max(1e-12, abs(old_quantity))
        entry_fee = state.entry_fees_remaining * close_fraction
        exit_fee = fees * (close_quantity / max(1e-12, abs(signed_quantity)))
        realized = self._record_round_trip(
            state=state,
            close_quantity=close_quantity,
            exit_price=fill_price,
            exit_date=trade_date,
            reason=reason,
            entry_fee=entry_fee,
            exit_fee=exit_fee,
        )
        state.realized_pnl += realized
        state.entry_fees_remaining -= entry_fee
        if self.config.market == "ashare" and state.quantity > 0:
            state.settled_quantity = max(0.0, state.settled_quantity - close_quantity)

        if abs(new_quantity) < 1e-8:
            self.position_states.pop(symbol, None)
        elif old_quantity * new_quantity > 0:
            state.quantity = new_quantity
            if self.config.market != "ashare":
                state.settled_quantity = abs(new_quantity)
        else:
            open_quantity = abs(new_quantity)
            open_fee = max(0.0, fees - exit_fee)
            initial_atr = (
                fill_price * float(atr_pct)
                if _finite(atr_pct) and float(atr_pct) > 0
                else None
            )
            self.position_states[symbol] = PositionState(
                symbol=symbol,
                quantity=new_quantity,
                avg_entry_price=fill_price,
                entry_date=str(trade_date),
                entry_session_index=self._session_index,
                highest_price=fill_price,
                lowest_price=fill_price,
                initial_atr=initial_atr,
                current_atr=initial_atr,
                settled_quantity=(open_quantity if self.config.market != "ashare" or new_quantity < 0 else 0.0),
                entry_fees_remaining=open_fee,
            )
        return entry_before, realized, holding

    def _apply_corporate_actions(self, trade_date: date, market: dict[str, dict]) -> None:
        for symbol in list(self.positions):
            row = market.get(symbol)
            if not row or not _finite(row.get("adjustment_factor")):
                continue
            previous = self.last_adjustment.get(symbol)
            current = float(row["adjustment_factor"])
            if not previous or previous <= 0 or current <= 0:
                continue
            ratio = current / previous
            if abs(ratio - 1.0) <= 1e-8:
                continue
            before = self.positions[symbol]
            # The panel's adjustment factor can imply fractional synthetic
            # shares. Freeze share precision in the state itself so the
            # statement never hides extra quantity decimals.
            after = _round(before * ratio, 6)
            self.positions[symbol] = after
            state = self.position_states.get(symbol)
            if state is not None:
                state.quantity = after
                state.avg_entry_price = state.avg_entry_price / ratio
                state.highest_price = state.highest_price / ratio
                state.lowest_price = state.lowest_price / ratio
                if state.initial_atr:
                    state.initial_atr /= ratio
                if state.current_atr:
                    state.current_atr /= ratio
                state.settled_quantity = _round(state.settled_quantity * ratio, 6)
            for order in self.pending_orders:
                if order["symbol"] == symbol:
                    order["target_quantity"] = _round(
                        order["target_quantity"] * ratio,
                        6,
                    )
            self._emit(
                trade_date,
                "OPEN_CORPORATE_ACTION",
                "CORPORATE_ACTION_ADJUSTMENT",
                symbol=symbol,
                message="按复权因子保持经济敞口连续",
                payload={
                    "quantity_before": before,
                    "quantity_after": after,
                    "adjustment_ratio": ratio,
                    "semantics": "split_dividend_reinvestment_proxy",
                },
            )

    def _volume_limit(self, symbol: str, row: dict, requested_abs: float) -> tuple[float, float, str]:
        volume, basis = self._liquidity_capacity(row)
        if volume <= 0:
            return 0.0, 0.0, basis
        cap = volume * self.config.max_volume_participation
        used = self._session_filled_by_symbol.get(symbol, 0.0)
        remaining = max(0.0, cap - used)
        lot = self.config.lot_size
        capped = min(requested_abs, math.floor(remaining / lot) * lot)
        participation = capped / volume if volume > 0 else 0.0
        return max(0.0, capped), participation, basis

    def _record_rejection(
        self,
        *,
        trade_date: date,
        order: dict,
        reason: str,
        requested: float,
        phase: str = "OPEN_EXECUTION",
    ) -> None:
        self._rejected_orders += 1
        self._emit(
            trade_date,
            phase,
            "ORDER_REJECTED",
            symbol=order["symbol"],
            order_id=order["order_id"],
            message=reason,
            payload={"requested_quantity": requested},
        )

    def _execute_order(
        self,
        trade_date: date,
        order: dict,
        market: dict[str, dict],
        *,
        reference_price_override: float | None = None,
        phase: str = "OPEN_EXECUTION",
    ) -> dict:
        symbol = order["symbol"]
        current = float(self.positions.get(symbol, 0.0))
        requested_signed = _round(
            float(order["target_quantity"]) - current,
            6,
        )
        if abs(requested_signed) < 1e-8:
            return {"filled": 0.0, "unfilled": 0.0, "status": "noop"}
        requested_abs = abs(requested_signed)
        self._executed_orders += 1
        self._requested_execution_quantity += requested_abs
        row = market.get(symbol)
        if row is None or not _finite(row.get("raw_open")) or float(row["raw_open"]) <= 0:
            self._record_rejection(
                trade_date=trade_date,
                order=order,
                reason="开盘行情缺失，DAY 订单取消",
                requested=requested_signed,
                phase=phase,
            )
            return {"filled": 0.0, "unfilled": requested_abs, "status": "rejected"}

        side = "BUY" if requested_signed > 0 else "SELL"
        permission = "can_buy_open_proxy" if side == "BUY" else "can_sell_open_proxy"
        if row.get(permission) is False:
            self._record_rejection(
                trade_date=trade_date,
                order=order,
                reason=f"{permission}=false，停牌或涨跌停代理阻止成交",
                requested=requested_signed,
                phase=phase,
            )
            return {"filled": 0.0, "unfilled": requested_abs, "status": "rejected"}

        filled_abs, participation, liquidity_basis = self._volume_limit(
            symbol, row, requested_abs
        )
        # The price written to the statement is the exact price used for cash,
        # fees, affordability, and positions.  Never keep a hidden
        # higher-precision execution price behind a rounded statement value.
        reference_price = _round(
            float(reference_price_override)
            if reference_price_override is not None
            else float(row["raw_open"]),
            6,
        )
        impact_bps = self._impact_bps(participation)
        slip = impact_bps / 10_000.0
        fill_price = _round(
            reference_price * (1.0 + slip if side == "BUY" else 1.0 - slip),
            6,
        )
        if side == "BUY":
            # A cash account cannot silently borrow after an overnight gap.
            # A-share buys additionally preserve board-lot granularity.
            filled_abs = self._max_affordable_buy(
                requested=filled_abs,
                fill_price=fill_price,
                trade_date=trade_date,
            )
        if self.config.market == "ashare" and side == "SELL":
            state = self.position_states.get(symbol)
            available = state.settled_quantity if state and state.quantity > 0 else requested_abs
            filled_abs = min(filled_abs, max(0.0, available))
        leverage_limited = False
        projected_full_leverage = 0.0
        if filled_abs > 0:
            filled_abs, leverage_limited, projected_full_leverage = (
                self._leverage_capped_quantity(
                    symbol=symbol,
                    requested_abs=filled_abs,
                    signed_direction=1.0 if side == "BUY" else -1.0,
                    fill_price=fill_price,
                    trade_date=trade_date,
                    market=market,
                )
            )
            if leverage_limited:
                self._online_integrity["leverage_limited_orders"] += 1
                volume, _ = self._liquidity_capacity(row)
                participation = filled_abs / volume if volume > 0 else 0.0
                self._emit(
                    trade_date,
                    phase,
                    "ORDER_CLIPPED_BY_LEVERAGE",
                    symbol=symbol,
                    order_id=order["order_id"],
                    message="成交量按实时总杠杆硬上限截断",
                    payload={
                        "requested_quantity": requested_abs,
                        "capped_quantity": filled_abs,
                        "projected_full_leverage": projected_full_leverage,
                        "max_gross_leverage": self.config.max_gross_leverage,
                    },
                )
        filled_abs = _round(filled_abs, 6)
        if filled_abs <= 0:
            self._record_rejection(
                trade_date=trade_date,
                order=order,
                reason=(
                    "实时总杠杆硬上限没有新增敞口空间"
                    if leverage_limited
                    else "成交量参与率或可用现金限制导致零成交"
                ),
                requested=requested_signed,
                phase=phase,
            )
            return {"filled": 0.0, "unfilled": requested_abs, "status": "rejected"}

        self._filled_execution_quantity += filled_abs
        signed_quantity = filled_abs if side == "BUY" else -filled_abs
        fees = calculate_trade_fees(
            market=self.config.market,
            side=side,
            quantity=filled_abs,
            price=fill_price,
            trade_date=trade_date,
            profile=self.config.resolved_fee_profile,
        )
        cash_before = self.cash
        position_before = current
        atr_pct = order.get("atr_pct_at_signal")
        self.cash -= signed_quantity * fill_price + fees.total
        position_after = _round(current + signed_quantity, 6)
        if abs(position_after) < 1e-8:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = position_after
        entry_price_before, realized_pnl, holding_sessions = self._update_position_state_for_fill(
            symbol=symbol,
            signed_quantity=signed_quantity,
            fill_price=fill_price,
            trade_date=trade_date,
            atr_pct=atr_pct,
            fees=fees.total,
            reason=str(order.get("reason") or "unspecified"),
        )
        self._session_filled_by_symbol[symbol] = (
            self._session_filled_by_symbol.get(symbol, 0.0) + filled_abs
        )
        self.names[symbol] = str(row.get("name") or symbol)
        slippage_cost = filled_abs * abs(fill_price - reference_price)
        self._cumulative_fees += fees.total
        self._cumulative_slippage += slippage_cost
        self._cumulative_traded_notional += filled_abs * fill_price
        self._fill_seq += 1
        nlv_after, _, _ = self._nlv(market, "raw_open")
        _, _, gross_leverage_after = self._gross_leverage(market, "raw_open")
        unfilled = max(0.0, requested_abs - filled_abs)
        if unfilled > 1e-8:
            self._partial_orders += 1
        trade = {
            "fill_id": f"FILL-{self._fill_seq:08d}",
            "order_id": order["order_id"],
            "signal_date": order["signal_date"],
            "scheduled_execute_date": order["execute_date"],
            "trade_date": str(trade_date),
            "market": self.config.market,
            "currency": self.config.currency,
            "symbol": symbol,
            "name": self.names[symbol],
            "side": side,
            "reason": order["reason"],
            "order_type": order.get("order_type", "MARKET"),
            "time_in_force": order.get("time_in_force", "DAY"),
            "requested_quantity": _round(requested_abs, 6),
            "filled_quantity": _round(filled_abs, 6),
            "unfilled_quantity": _round(unfilled, 6),
            "reference_price": _round(reference_price, 6),
            "fill_price": _round(fill_price, 6),
            "gross_amount": _round(filled_abs * fill_price, 6),
            "commission": fees.commission,
            "stamp_duty": fees.stamp_duty,
            "transfer_fee": fees.transfer_fee,
            "regulatory_fee": fees.regulatory_fee,
            "exchange_fee": fees.exchange_fee,
            "total_fees": fees.total,
            "slippage_cost": _round(slippage_cost, 6),
            "impact_bps": _round(impact_bps, 6),
            "participation": _round(participation, 8),
            "liquidity_basis": liquidity_basis,
            "cash_before": _round(cash_before, 6),
            "cash_after": _round(self.cash, 6),
            "position_before": _round(position_before, 6),
            "position_after": _round(position_after, 6),
            "entry_price_before": _round(entry_price_before, 6) if entry_price_before else None,
            "realized_pnl": _round(realized_pnl, 6),
            "holding_sessions": holding_sessions,
            "nlv_after": _round(nlv_after, 6),
            "gross_leverage_after": _round(gross_leverage_after, 8),
            "leverage_limited": leverage_limited,
            "fee_profile": fees.profile,
            "fee_notes": fees.notes,
        }
        self._audit_trade_online(trade)
        if self.capture_detail:
            self.trades.append(trade)
        self._emit(
            trade_date,
            phase,
            "FILL" if unfilled <= 1e-8 else "PARTIAL_FILL",
            symbol=symbol,
            order_id=order["order_id"],
            message=f"{side} {filled_abs:g} @ {fill_price:.4f}",
            payload={
                "fill_id": trade["fill_id"],
                "total_fees": fees.total,
                "slippage_cost": slippage_cost,
                "unfilled_quantity": unfilled,
                "impact_bps": impact_bps,
                "liquidity_basis": liquidity_basis,
            },
        )
        return {
            "filled": filled_abs,
            "unfilled": unfilled,
            "status": "filled" if unfilled <= 1e-8 else "partial",
        }

    def _execute_pending(
        self,
        trade_date: date,
        market: dict[str, dict],
        next_trade_date: date | None,
    ) -> None:
        due = [order for order in self.pending_orders if order["execute_date"] == str(trade_date)]
        self.pending_orders = [order for order in self.pending_orders if order["execute_date"] != str(trade_date)]
        factor_due = [
            order for order in due
            if order.get("reason") == "factor_rebalance_next_open"
        ]
        if factor_due:
            projected = dict(self.positions)
            for order in factor_due:
                projected[order["symbol"]] = float(order["target_quantity"])
            open_nlv, _, _ = self._nlv(market, "raw_open")
            projected_gross = sum(
                abs(quantity) * self._price(symbol, market, "raw_open")
                for symbol, quantity in projected.items()
            )
            operating_target = (
                self.config.effective_long_gross_target
                + self.config.effective_short_gross_target
            )
            allowed_gross = max(0.0, open_nlv) * min(
                self.config.max_gross_leverage,
                operating_target if operating_target > 0 else self.config.max_gross_leverage,
            )
            if projected_gross > allowed_gross + 1e-8 and projected_gross > 0:
                scale = allowed_gross / projected_gross
                lot = self.config.lot_size
                for order in factor_due:
                    target = float(order["target_quantity"])
                    signed = 1.0 if target >= 0 else -1.0
                    order["target_quantity"] = signed * math.floor(abs(target) * scale / lot) * lot
                self._emit(
                    trade_date,
                    "OPEN_EXECUTION",
                    "TARGETS_SCALED_TO_LEVERAGE",
                    message=f"隔夜价格变化后按 {scale:.4f} 缩放目标仓位",
                    payload={
                        "projected_gross": projected_gross,
                        "allowed_gross": allowed_gross,
                        "scale": scale,
                    },
                )
        # Execute every exposure-reducing order before any order that adds
        # gross risk.  This ordering is valid for long, short and flip orders;
        # a simple SELL-before-BUY rule is insufficient for long-short books.
        due.sort(
            key=lambda order: (
                abs(float(order["target_quantity"]))
                > abs(float(self.positions.get(order["symbol"], 0.0))) + 1e-8,
                order["order_id"],
            )
        )
        for order in due:
            if order["symbol"] in self._blocked_symbols_today and not str(order.get("reason", "")).startswith("risk_"):
                self._emit(
                    trade_date,
                    "OPEN_EXECUTION",
                    "ORDER_CANCELLED_BY_RISK_EXIT",
                    symbol=order["symbol"],
                    order_id=order["order_id"],
                    message="当日风险退出优先，取消因子调仓订单",
                )
                continue
            outcome = self._execute_order(trade_date, order, market)
            should_carry = (
                outcome["unfilled"] > 1e-8
                and next_trade_date is not None
                and (
                    self.config.unfilled_order_policy == "carry"
                    or str(order.get("reason", "")).startswith("risk_")
                    or str(order.get("reason", "")).startswith("portfolio_")
                )
            )
            age = int(order.get("age_sessions", 0)) + 1
            if should_carry and age < self.config.max_order_age_sessions:
                carried = dict(order)
                carried["execute_date"] = str(next_trade_date)
                carried["age_sessions"] = age
                self.pending_orders.append(carried)
                self._emit(
                    trade_date,
                    "OPEN_EXECUTION",
                    "ORDER_CARRIED",
                    symbol=order["symbol"],
                    order_id=order["order_id"],
                    message=f"未成交目标续挂至 {next_trade_date}",
                    payload={"age_sessions": age, "unfilled_quantity": outcome["unfilled"]},
                )

    def _enforce_open_leverage(
        self,
        *,
        trade_date: date,
        market: dict[str, dict],
        next_trade_date: date | None,
    ) -> dict:
        """Repair passive or residual open leverage before intraday trading."""
        nlv_before, gross_before, leverage_before = self._gross_leverage(
            market, "raw_open"
        )
        self._online_integrity["max_open_gross_leverage_observed"] = max(
            float(self._online_integrity["max_open_gross_leverage_observed"]),
            float(leverage_before if math.isfinite(leverage_before) else 0.0),
        )
        hard_limit = self.config.max_gross_leverage
        # This is a hard execution invariant, not a reporting threshold.  Keep
        # only a tiny numerical tolerance for fee/rounding arithmetic.
        tolerance = max(1e-8, hard_limit * 1e-6)
        if leverage_before <= hard_limit + tolerance:
            self._online_integrity["max_open_gross_leverage_after_control"] = max(
                float(self._online_integrity["max_open_gross_leverage_after_control"]),
                float(leverage_before if math.isfinite(leverage_before) else 0.0),
            )
            return {
                "breach": False,
                "resolved": True,
                "before": leverage_before,
                "after": leverage_before,
                "orders": 0,
            }

        self._online_integrity["gross_leverage_breach_events"] += 1
        self._emit(
            trade_date,
            "OPEN_EXECUTION",
            "PORTFOLIO_LEVERAGE_BREACH",
            message=f"开盘总杠杆 {leverage_before:.4f} 超过硬上限 {hard_limit:.4f}",
            payload={
                "open_nlv": nlv_before,
                "gross_value": gross_before,
                "gross_leverage": leverage_before,
                "hard_limit": hard_limit,
            },
        )

        operating_target = (
            self.config.effective_long_gross_target
            + self.config.effective_short_gross_target
        )
        # Even a user-declared target equal to the ceiling needs a minimum
        # recovery band so fees and rounding do not immediately re-breach it.
        recovery_band = max(0.01, self.config.cash_buffer_fraction)
        recovery_target = min(
            operating_target if operating_target > 0 else hard_limit,
            hard_limit * (1.0 - recovery_band),
        )
        scale = (
            max(0.0, nlv_before) * max(0.0, recovery_target) / gross_before
            if gross_before > 1e-12
            else 0.0
        )
        scale = min(1.0, max(0.0, scale))
        orders = 0
        for symbol, current in sorted(
            list(self.positions.items()),
            key=lambda item: abs(item[1]) * self._price(item[0], market, "raw_open"),
            reverse=True,
        ):
            if abs(current) < 1e-8:
                continue
            lot = float(self.config.lot_size)
            target_abs = math.floor(abs(current) * scale / lot + 1e-12) * lot
            target = math.copysign(target_abs, current)
            if abs(target - current) < 1e-8:
                continue
            self._order_seq += 1
            order = {
                "order_id": f"ORD-{self._order_seq:08d}",
                "signal_date": str(trade_date),
                "execute_date": str(trade_date),
                "symbol": symbol,
                "target_quantity": target,
                "quantity_at_signal": current,
                "reason": "portfolio_leverage_deleveraging",
                "order_type": "MARKET",
                "time_in_force": "GTC",
                "age_sessions": 0,
                "atr_pct_at_signal": None,
            }
            outcome = self._execute_order(
                trade_date,
                order,
                market,
                phase="OPEN_EXECUTION",
            )
            orders += int(outcome["filled"] > 0)
            if outcome["unfilled"] > 1e-8 and next_trade_date is not None:
                carried = dict(order)
                carried["execute_date"] = str(next_trade_date)
                carried["age_sessions"] = 1
                self.pending_orders.append(carried)

        self._online_integrity["automatic_deleveraging_events"] += 1
        nlv_after, gross_after, leverage_after = self._gross_leverage(
            market, "raw_open"
        )
        self._online_integrity["max_open_gross_leverage_after_control"] = max(
            float(self._online_integrity["max_open_gross_leverage_after_control"]),
            float(leverage_after if math.isfinite(leverage_after) else 0.0),
        )
        resolved = leverage_after <= hard_limit + tolerance
        if not resolved:
            self._online_integrity["gross_leverage_violations"] += 1
        detail = {
            "trade_date": str(trade_date),
            "before": _round(leverage_before, 8),
            "after": _round(leverage_after, 8),
            "hard_limit": hard_limit,
            "recovery_target": _round(recovery_target, 8),
            "orders": orders,
            "resolved": resolved,
            "nlv_before": _round(nlv_before, 6),
            "nlv_after": _round(nlv_after, 6),
            "gross_before": _round(gross_before, 6),
            "gross_after": _round(gross_after, 6),
        }
        self._gross_leverage_violation_details.append(detail)
        self._emit(
            trade_date,
            "OPEN_EXECUTION",
            "PORTFOLIO_AUTO_DELEVERAGED" if resolved else "PORTFOLIO_DELEVERAGING_INCOMPLETE",
            message=(
                f"自动去杠杆后 {leverage_after:.4f}"
                if resolved
                else f"自动去杠杆未完成，剩余 {leverage_after:.4f}"
            ),
            payload=detail,
        )
        return {
            "breach": True,
            "resolved": resolved,
            "before": leverage_before,
            "after": leverage_after,
            "orders": orders,
        }

    def _target_quantities(self, candidates: list[dict], nlv: float) -> dict[str, float]:
        eligible = [
            row for row in candidates
            if int(row.get("univ_rank") or 10**9) <= self.config.universe_n
            and _finite(row.get("factor"))
            and _finite(row.get("raw_close"))
            and float(row["raw_close"]) > 0
        ]
        eligible.sort(
            key=lambda row: float(row["factor"]) * self.config.direction,
            reverse=True,
        )
        if not eligible:
            return {}
        count = max(1, int(len(eligible) * self.config.top_fraction))
        per_leg_cap = (
            max(1, self.config.max_positions // 2)
            if self.config.mode == "long_short"
            else self.config.max_positions
        )
        count = min(count, per_leg_cap)
        longs = eligible[:count]
        shorts = eligible[-count:] if self.config.mode == "long_short" else []
        target: dict[str, float] = {}
        lot = self.config.lot_size

        def dollars_for(rows: list[dict], gross_target: float) -> list[float]:
            if not rows or nlv <= 0 or gross_target <= 0:
                return []
            budget = nlv * gross_target
            cap = nlv * self.config.max_position_weight
            if self.config.position_sizing == "atr_risk":
                stop_multiple = self.config.exit_policy.atr_stop_multiple or 2.0
                values = []
                for row in rows:
                    atr_pct = float(row.get("_atr_pct") or 0.0)
                    risk_dollars = (
                        nlv * self.config.risk_per_position_fraction
                        / max(0.0025, atr_pct * stop_multiple)
                    )
                    values.append(min(cap, risk_dollars))
                total = sum(values)
                scale = min(1.0, budget / total) if total > 0 else 0.0
                return [value * scale for value in values]
            if self.config.position_sizing == "inverse_volatility":
                raw = [1.0 / max(0.01, float(row.get("_vol20_prev") or 0.0)) for row in rows]
            else:
                raw = [1.0] * len(rows)
            total = sum(raw)
            values = [min(cap, budget * value / total) for value in raw]
            # Redistribute residual budget without breaching the name cap.
            for _ in range(4):
                residual = budget - sum(values)
                available = [index for index, value in enumerate(values) if value < cap - 1e-8]
                if residual <= 1e-8 or not available:
                    break
                share = residual / len(available)
                for index in available:
                    values[index] = min(cap, values[index] + share)
            return values

        long_target = self.config.effective_long_gross_target
        for row, dollars in zip(longs, dollars_for(longs, long_target)):
            quantity = math.floor(dollars / float(row["raw_close"]) / lot) * lot
            if quantity > 0:
                target[row["ts_code"]] = float(quantity)
                self.names[row["ts_code"]] = str(row.get("name") or row["ts_code"])
        for row, dollars in zip(
            shorts,
            dollars_for(shorts, self.config.effective_short_gross_target),
        ):
            quantity = math.floor(dollars / float(row["raw_close"]) / lot) * lot
            if quantity > 0:
                target[row["ts_code"]] = -float(quantity)
                self.names[row["ts_code"]] = str(row.get("name") or row["ts_code"])
        return target

    def _create_orders(
        self,
        *,
        signal_date: date,
        execute_date: date,
        candidates: list[dict],
        close_nlv: float,
    ) -> int:
        targets = self._target_quantities(candidates, close_nlv)
        candidate_by_symbol = {str(row["ts_code"]): row for row in candidates}
        created = 0
        for symbol in sorted(set(self.positions) | set(targets)):
            target = float(targets.get(symbol, 0.0))
            current = float(self.positions.get(symbol, 0.0))
            if abs(target - current) < 1e-8:
                continue
            reference = float(candidate_by_symbol.get(symbol, {}).get("raw_close") or self.last_close.get(symbol, 0.0))
            trade_notional = abs(target - current) * reference
            target_notional = abs(target) * reference
            if trade_notional < self.config.min_trade_notional:
                continue
            if target_notional > 0 and trade_notional / target_notional < self.config.rebalance_buffer_pct:
                continue
            self._order_seq += 1
            order = {
                "order_id": f"ORD-{self._order_seq:08d}",
                "signal_date": str(signal_date),
                "execute_date": str(execute_date),
                "symbol": symbol,
                "target_quantity": target,
                "quantity_at_signal": current,
                "reason": "factor_rebalance_next_open",
                "order_type": "MARKET",
                "time_in_force": "DAY" if self.config.unfilled_order_policy == "cancel" else "GTC",
                "age_sessions": 0,
                "atr_pct_at_signal": candidate_by_symbol.get(symbol, {}).get("_atr_pct"),
            }
            self.pending_orders.append(order)
            created += 1
            self._emit(
                signal_date,
                "CLOSE_SIGNAL",
                "ORDER_CREATED",
                symbol=symbol,
                order_id=order["order_id"],
                message=f"目标持仓 {target:g}，计划 {execute_date} 开盘执行",
                payload={
                    "target_quantity": target,
                    "quantity_at_signal": current,
                    "execute_date": str(execute_date),
                },
            )
        self._emit(
            signal_date,
            "CLOSE_SIGNAL",
            "SIGNAL_SNAPSHOT",
            message=f"生成 {created} 笔次日开盘订单",
            payload={
                "eligible": sum(
                    int(row.get("univ_rank") or 10**9) <= self.config.universe_n
                    and _finite(row.get("factor"))
                    for row in candidates
                ),
                "targets": len(targets),
                "orders": created,
            },
        )
        return created

    def _queue_liquidation_order(
        self,
        *,
        symbol: str,
        signal_date: date,
        execute_date: date,
        reason: str,
        atr_pct: float | None = None,
    ) -> dict:
        existing = [
            order for order in self.pending_orders
            if order["symbol"] == symbol
        ]
        for order in existing:
            self.pending_orders.remove(order)
        self._order_seq += 1
        order = {
            "order_id": f"ORD-{self._order_seq:08d}",
            "signal_date": str(signal_date),
            "execute_date": str(execute_date),
            "symbol": symbol,
            "target_quantity": 0.0,
            "quantity_at_signal": float(self.positions.get(symbol, 0.0)),
            "reason": reason,
            "order_type": "STOP_MARKET" if reason.startswith("risk_") else "MARKET",
            "time_in_force": "GTC",
            "age_sessions": 0,
            "atr_pct_at_signal": atr_pct,
        }
        self.pending_orders.append(order)
        return order

    def _evaluate_risk_stage(
        self,
        *,
        trade_date: date,
        next_trade_date: date | None,
        market: dict[str, dict],
        stage: str,
    ) -> None:
        policy = self.config.exit_policy
        if not policy.enabled:
            return
        trigger_phase = "OPEN_RISK_TRIGGER" if stage == "open" else "INTRADAY_RISK_TRIGGER"
        execution_phase = "OPEN_EXECUTION" if stage == "open" else "INTRADAY_EXECUTION"
        for symbol in sorted(list(self.position_states)):
            if symbol in self._blocked_symbols_today:
                continue
            state = self.position_states.get(symbol)
            row = market.get(symbol)
            if state is None or row is None:
                continue
            open_price = float(row.get("raw_open") or 0.0)
            high_price = float(row.get("raw_high") or 0.0)
            low_price = float(row.get("raw_low") or 0.0)
            if open_price <= 0:
                continue
            trigger = evaluate_exit(
                state,
                policy,
                open_price=open_price,
                high_price=high_price if high_price > 0 else None,
                low_price=low_price if low_price > 0 else None,
                stage=stage,
            )
            if trigger is None:
                continue
            if trigger.ambiguous:
                self._online_integrity["intrabar_ambiguities"] += 1
            stop_price, take_profit_price, mechanisms = risk_levels(state, policy)
            self._emit(
                trade_date,
                trigger_phase,
                "RISK_EXIT_TRIGGERED",
                symbol=symbol,
                message=trigger.reason,
                payload={
                    "reason": trigger.reason,
                    "trigger_price": trigger.trigger_price,
                    "reference_price": trigger.reference_price,
                    "gap": trigger.gap,
                    "ambiguous": trigger.ambiguous,
                    "conflict_policy": policy.intrabar_conflict_policy,
                    "stop_price": stop_price,
                    "take_profit_price": take_profit_price,
                    "mechanisms": mechanisms,
                },
            )
            self.pending_orders = [
                order for order in self.pending_orders
                if order["symbol"] != symbol
            ]
            self._order_seq += 1
            order = {
                "order_id": f"ORD-{self._order_seq:08d}",
                "signal_date": state.entry_date,
                "execute_date": str(trade_date),
                "symbol": symbol,
                "target_quantity": 0.0,
                "quantity_at_signal": state.quantity,
                "reason": trigger.reason,
                "order_type": "STOP_MARKET",
                "time_in_force": "GTC",
                "age_sessions": 0,
                "atr_pct_at_signal": None,
            }
            outcome = self._execute_order(
                trade_date,
                order,
                market,
                reference_price_override=trigger.reference_price,
                phase=execution_phase,
            )
            self._blocked_symbols_today.add(symbol)
            if outcome["unfilled"] > 1e-8 and next_trade_date is not None:
                queued = self._queue_liquidation_order(
                    symbol=symbol,
                    signal_date=trade_date,
                    execute_date=next_trade_date,
                    reason=trigger.reason,
                )
                self._emit(
                    trade_date,
                    execution_phase,
                    "RISK_EXIT_CARRIED",
                    symbol=symbol,
                    order_id=queued["order_id"],
                    message=f"风险退出未完成，续挂至 {next_trade_date}",
                    payload={"unfilled_quantity": outcome["unfilled"]},
                )

    def _handle_stale_positions(
        self,
        trade_date: date,
        market: dict[str, dict],
    ) -> None:
        for symbol in list(self.positions):
            row = market.get(symbol)
            has_price = row is not None and _finite(row.get("raw_close")) and float(row["raw_close"]) > 0
            if has_price:
                self._missing_position_sessions[symbol] = 0
                continue
            missing = self._missing_position_sessions.get(symbol, 0) + 1
            self._missing_position_sessions[symbol] = missing
            if missing < self.config.max_stale_sessions:
                continue
            state = self.position_states.get(symbol)
            quantity = float(self.positions.get(symbol, 0.0))
            if state is not None:
                exit_price = 0.0 if quantity > 0 else float(self.last_close.get(symbol, 0.0))
                if quantity < 0 and exit_price > 0:
                    self.cash -= abs(quantity) * exit_price
                self._record_round_trip(
                    state=state,
                    close_quantity=abs(quantity),
                    exit_price=exit_price,
                    exit_date=trade_date,
                    reason="stale_position_writeoff",
                    entry_fee=state.entry_fees_remaining,
                    exit_fee=0.0,
                )
            self.positions.pop(symbol, None)
            self.position_states.pop(symbol, None)
            self._missing_position_sessions.pop(symbol, None)
            self._online_integrity["stale_position_writeoffs"] += 1
            self._emit(
                trade_date,
                "OPEN_RISK_TRIGGER",
                "STALE_POSITION_WRITEOFF",
                symbol=symbol,
                message=f"连续 {missing} 个交易日缺少行情，按保守退市规则处置",
                payload={"quantity": quantity, "long_exit_price": 0.0},
            )

    def _update_position_bars(self, market: dict[str, dict]) -> None:
        for symbol, state in list(self.position_states.items()):
            row = market.get(symbol)
            if row is None:
                continue
            high = float(row.get("raw_high") or row.get("raw_close") or 0.0)
            low = float(row.get("raw_low") or row.get("raw_close") or 0.0)
            atr_pct = row.get("_atr_pct")
            close = float(row.get("raw_close") or 0.0)
            atr = close * float(atr_pct) if _finite(atr_pct) and close > 0 else None
            state.update_completed_bar(high=high, low=low, atr=atr)

    def _queue_portfolio_liquidation(
        self,
        *,
        trade_date: date,
        next_trade_date: date,
        reason: str,
    ) -> int:
        count = 0
        for symbol in sorted(self.positions):
            self._queue_liquidation_order(
                symbol=symbol,
                signal_date=trade_date,
                execute_date=next_trade_date,
                reason=reason,
            )
            count += 1
        if count:
            self._portfolio_liquidations += 1
            self._portfolio_risk_active = True
            self._portfolio_risk_trigger_events += 1
            self._portfolio_risk_last_trigger = reason
            self._risk_cooldown_remaining = max(
                self._risk_cooldown_remaining,
                self.config.risk_cooldown_sessions,
            )
        return count

    def _liquidate_terminal(
        self,
        *,
        trade_date: date,
        market: dict[str, dict],
    ) -> None:
        if not self.config.liquidate_at_end:
            return
        for symbol in sorted(list(self.positions)):
            row = market.get(symbol)
            if row is None or not _finite(row.get("raw_close")):
                continue
            self._order_seq += 1
            order = {
                "order_id": f"ORD-{self._order_seq:08d}",
                "signal_date": str(trade_date),
                "execute_date": str(trade_date),
                "symbol": symbol,
                "target_quantity": 0.0,
                "quantity_at_signal": self.positions[symbol],
                "reason": "terminal_liquidation",
                "order_type": "MARKET_ON_CLOSE",
                "time_in_force": "DAY",
                "age_sessions": 0,
                "atr_pct_at_signal": None,
            }
            self._execute_order(
                trade_date,
                order,
                market,
                reference_price_override=float(row["raw_close"]),
                phase="INTRADAY_EXECUTION",
            )

    def step(
        self,
        *,
        trade_date: date,
        rows: list[dict],
        next_trade_date: date | None,
        rebalance: bool,
        market_by_symbol: dict[str, dict] | None = None,
    ) -> dict:
        """Advance one session and return that session's reconciled state."""
        market = market_by_symbol or {
            str(row["ts_code"]): row for row in rows
        }
        self._session_index += 1
        self._session_filled_by_symbol = {}
        self._blocked_symbols_today = set()
        if self._risk_cooldown_remaining > 0:
            self._risk_cooldown_remaining -= 1
        event_start = self._event_seq
        fill_start = self._fill_seq
        notional_start = self._cumulative_traded_notional
        self._emit(trade_date, "SESSION_OPEN", "SESSION_OPEN", message="进入交易日")
        self._settle_positions_for_session()
        self._apply_corporate_actions(trade_date, market)
        self._handle_stale_positions(trade_date, market)
        open_nlv_before, _, _ = self._nlv(market, "raw_open")
        self._evaluate_risk_stage(
            trade_date=trade_date,
            next_trade_date=next_trade_date,
            market=market,
            stage="open",
        )
        self._execute_pending(trade_date, market, next_trade_date)
        leverage_control = self._enforce_open_leverage(
            trade_date=trade_date,
            market=market,
            next_trade_date=next_trade_date,
        )
        open_nlv_after, open_long_value, open_short_value = self._nlv(market, "raw_open")
        open_gross_exposure = (
            (open_long_value + abs(open_short_value)) / open_nlv_after
            if open_nlv_after > 0 else math.inf
        )
        self._evaluate_risk_stage(
            trade_date=trade_date,
            next_trade_date=next_trade_date,
            market=market,
            stage="intraday",
        )
        if next_trade_date is None:
            self._liquidate_terminal(trade_date=trade_date, market=market)

        close_nlv_before_financing, long_value, short_value = self._nlv(market, "raw_close")
        borrow_fee = 0.0
        if self.config.mode == "long_short" and short_value < 0:
            borrow_fee = abs(short_value) * self.config.borrow_cost_bps_annual / 10_000.0 / 252.0
            if borrow_fee > 0:
                self.cash -= borrow_fee
                self._cumulative_borrow += borrow_fee
                self._emit(
                    trade_date,
                    "CLOSE_FINANCING",
                    "SHORT_BORROW_FEE",
                    message=f"空头融资费 {borrow_fee:.4f}",
                    payload={
                        "short_market_value": abs(short_value),
                        "annual_bps": self.config.borrow_cost_bps_annual,
                        "fee": borrow_fee,
                    },
                )
        margin_interest = 0.0
        if self.cash < 0 and self.config.resolved_account_type == "margin":
            margin_interest = (
                abs(self.cash)
                * self.config.margin_interest_bps_annual
                / 10_000.0
                / 252.0
            )
            if margin_interest > 0:
                self.cash -= margin_interest
                self._cumulative_margin_interest += margin_interest
                self._emit(
                    trade_date,
                    "CLOSE_FINANCING",
                    "MARGIN_INTEREST",
                    message=f"融资利息 {margin_interest:.4f}",
                    payload={
                        "negative_cash": abs(self.cash) - margin_interest,
                        "annual_bps": self.config.margin_interest_bps_annual,
                        "fee": margin_interest,
                    },
                )
        close_nlv = close_nlv_before_financing - borrow_fee - margin_interest
        traded_notional = self._cumulative_traded_notional - notional_start
        daily_return = close_nlv / self._previous_close_nlv - 1.0 \
            if self._previous_close_nlv > 0 else 0.0
        gross_exposure = (
            (long_value + abs(short_value)) / close_nlv
            if close_nlv > 0 else math.inf
        )
        if self.config.resolved_account_type == "cash" and self.cash < -1e-6:
            self._online_integrity["cash_account_negative_cash_violations"] += 1
        position_error = self._position_state_error()
        if position_error > 1e-6:
            self._online_integrity["position_state_quantity_violations"] += 1

        self._update_position_bars(market)
        self._peak_nlv = max(self._peak_nlv, close_nlv)
        portfolio_drawdown = (
            1.0 - close_nlv / self._peak_nlv
            if self._peak_nlv > 0 else 1.0
        )
        self._max_portfolio_risk_cycle_drawdown = max(
            self._max_portfolio_risk_cycle_drawdown,
            portfolio_drawdown,
        )
        portfolio_trigger = None
        # A portfolio kill switch is an edge-triggered state machine.  Once it
        # fires, the book must liquidate and finish its cooldown before the
        # high-water mark is reset and the rule is armed again.  Re-testing the
        # old peak while flat would otherwise lock the strategy in cash forever.
        if self.positions and not self._portfolio_risk_active:
            if (
                self.config.portfolio_stop_drawdown_pct is not None
                and portfolio_drawdown >= self.config.portfolio_stop_drawdown_pct
            ):
                portfolio_trigger = "portfolio_drawdown_exit"
            if (
                self.config.portfolio_daily_loss_pct is not None
                and daily_return <= -self.config.portfolio_daily_loss_pct
            ):
                portfolio_trigger = portfolio_trigger or "portfolio_daily_loss_exit"
        portfolio_orders = 0
        if portfolio_trigger and next_trade_date is not None:
            portfolio_orders = self._queue_portfolio_liquidation(
                trade_date=trade_date,
                next_trade_date=next_trade_date,
                reason=portfolio_trigger,
            )
            self._emit(
                trade_date,
                "CLOSE_RISK",
                "PORTFOLIO_RISK_TRIGGERED",
                message=portfolio_trigger,
                payload={
                    "drawdown": portfolio_drawdown,
                    "daily_return": daily_return,
                    "liquidation_orders": portfolio_orders,
                    "cooldown_sessions": self.config.risk_cooldown_sessions,
                },
            )
        portfolio_risk_rearmed = False
        liquidation_pending = any(
            str(order.get("reason", "")).startswith("portfolio_")
            for order in self.pending_orders
        )
        if (
            self._portfolio_risk_active
            and not self.positions
            and not liquidation_pending
            and self._risk_cooldown_remaining == 0
        ):
            previous_peak = self._peak_nlv
            self._peak_nlv = max(0.0, close_nlv)
            self._portfolio_risk_active = False
            self._portfolio_risk_rearms += 1
            portfolio_risk_rearmed = True
            self._emit(
                trade_date,
                "CLOSE_RISK",
                "PORTFOLIO_RISK_REARMED",
                message="组合风控冷却结束，重置高水位并允许下一交易日重新入场",
                payload={
                    "previous_peak_nlv": previous_peak,
                    "new_peak_nlv": self._peak_nlv,
                    "last_trigger": self._portfolio_risk_last_trigger,
                },
            )

        if self._portfolio_risk_active:
            self._portfolio_risk_active_sessions += 1

        created_orders = 0
        if (
            not portfolio_trigger
            and not portfolio_risk_rearmed
            and not self._portfolio_risk_active
            and self._risk_cooldown_remaining == 0
            and rebalance
            and next_trade_date is not None
            and close_nlv > 0
        ):
            created_orders = self._create_orders(
                signal_date=trade_date,
                execute_date=next_trade_date,
                candidates=rows,
                close_nlv=close_nlv,
            )

        for symbol, row in market.items():
            if _finite(row.get("raw_close")) and float(row["raw_close"]) > 0:
                self.last_close[symbol] = float(row["raw_close"])
            if _finite(row.get("adjustment_factor")) and float(row["adjustment_factor"]) > 0:
                self.last_adjustment[symbol] = float(row["adjustment_factor"])
        gross_proxy_nlv = (
            close_nlv
            + self._cumulative_fees
            + self._cumulative_slippage
            + self._cumulative_borrow
            + self._cumulative_margin_interest
        )
        row = {
            "trade_date": str(trade_date),
            "open_nlv_before_fills": _round(open_nlv_before, 6),
            "open_nlv_after_fills": _round(open_nlv_after, 6),
            "open_gross_exposure_before_control": _round(
                leverage_control["before"]
                if math.isfinite(leverage_control["before"])
                else 0.0,
                8,
            ),
            "open_gross_exposure": _round(
                open_gross_exposure if math.isfinite(open_gross_exposure) else 0.0,
                8,
            ),
            "leverage_control_orders": int(leverage_control["orders"]),
            "leverage_control_resolved": bool(leverage_control["resolved"]),
            "close_nlv": _round(close_nlv, 6),
            "net_nav": _round(close_nlv / self.config.initial_capital, 8),
            "same_orders_cost_free_nav_proxy": _round(
                gross_proxy_nlv / self.config.initial_capital, 8
            ),
            "daily_return": _round(daily_return, 8),
            "cash": _round(self.cash, 6),
            "long_market_value": _round(long_value, 6),
            "short_market_value": _round(short_value, 6),
            "gross_exposure": _round(gross_exposure if math.isfinite(gross_exposure) else 0.0, 6),
            "net_exposure": _round(
                (long_value + short_value) / close_nlv if close_nlv else 0.0, 6
            ),
            "turnover": _round(
                traded_notional / self._previous_close_nlv
                if self._previous_close_nlv > 0 else 0.0,
                6,
            ),
            "fills": self._fill_seq - fill_start,
            # SESSION_CLOSE is emitted immediately after this snapshot.
            "events": self._event_seq - event_start + 1,
            "orders_created": created_orders,
            "positions": len(self.positions),
            "borrow_fee": _round(borrow_fee, 6),
            "margin_interest": _round(margin_interest, 6),
            "portfolio_drawdown": _round(portfolio_drawdown, 8),
            "portfolio_risk_orders": portfolio_orders,
            "portfolio_risk_active": self._portfolio_risk_active,
            "portfolio_risk_rearmed": portfolio_risk_rearmed,
            "risk_cooldown_remaining": self._risk_cooldown_remaining,
            "position_state_max_error": _round(position_error, 8),
        }
        self.daily.append(row)
        self._previous_close_nlv = close_nlv
        self._emit(
            trade_date,
            "SESSION_CLOSE",
            "SESSION_CLOSE",
            message=f"NAV={row['net_nav']:.6f}",
            payload={
                "close_nlv": close_nlv,
                "cash": self.cash,
                "positions": len(self.positions),
                "daily_return": daily_return,
            },
        )
        return row

    def _integrity_checks(self) -> dict:
        cash_errors = []
        fee_errors = []
        fee_component_errors = []
        gross_amount_errors = []
        slippage_errors = []
        position_formula_errors = []
        same_day = 0
        schedule_violations = 0
        nonpositive_fills = 0
        side_sign_violations = 0
        fee_profile_violations = 0
        ashare_buy_lot_violations = 0
        for trade in self.trades:
            signed = trade["filled_quantity"] if trade["side"] == "BUY" else -trade["filled_quantity"]
            expected_cash = trade["cash_before"] - signed * trade["fill_price"] - trade["total_fees"]
            cash_errors.append(abs(expected_cash - trade["cash_after"]))
            gross_amount_errors.append(
                abs(
                    trade["filled_quantity"] * trade["fill_price"]
                    - trade["gross_amount"]
                )
            )
            slippage_errors.append(
                abs(
                    trade["filled_quantity"]
                    * abs(trade["fill_price"] - trade["reference_price"])
                    - trade["slippage_cost"]
                )
            )
            position_formula_errors.append(
                abs(
                    trade["position_before"] + signed
                    - trade["position_after"]
                )
            )
            fee_component_errors.append(
                abs(
                    trade["commission"]
                    + trade["stamp_duty"]
                    + trade["transfer_fee"]
                    + trade["regulatory_fee"]
                    + trade["exchange_fee"]
                    - trade["total_fees"]
                )
            )
            recalculated = calculate_trade_fees(
                market=trade["market"],
                side=trade["side"],
                quantity=trade["filled_quantity"],
                price=trade["fill_price"],
                trade_date=date.fromisoformat(trade["trade_date"]),
                profile=trade["fee_profile"],
            )
            fee_errors.append(abs(recalculated.total - trade["total_fees"]))
            if (
                trade.get("reason") == "factor_rebalance_next_open"
                and trade["signal_date"] >= trade["trade_date"]
            ):
                same_day += 1
            if trade.get("scheduled_execute_date") != trade["trade_date"]:
                schedule_violations += 1
            if trade["filled_quantity"] <= 0 or trade["fill_price"] <= 0:
                nonpositive_fills += 1
            if (
                trade["side"] == "BUY"
                and trade["position_after"] < trade["position_before"] - 1e-8
            ) or (
                trade["side"] == "SELL"
                and trade["position_after"] > trade["position_before"] + 1e-8
            ):
                side_sign_violations += 1
            if trade["fee_profile"] != self.config.resolved_fee_profile:
                fee_profile_violations += 1
            if (
                self.config.market == "ashare"
                and trade["side"] == "BUY"
                and abs(trade["filled_quantity"] % self.config.lot_size) > 1e-8
            ):
                ashare_buy_lot_violations += 1
        order_violations = 0
        by_date: dict[str, list[int]] = {}
        for event in self.events:
            by_date.setdefault(event["trade_date"], []).append(
                _EVENT_PHASE_ORDER.get(event["phase"], 99)
            )
        for phases in by_date.values():
            order_violations += sum(
                phases[index] < phases[index - 1]
                for index in range(1, len(phases))
            )
        long_only_short_positions = sum(
            quantity < -1e-8 for quantity in self.positions.values()
        ) if self.config.mode == "long_only" else 0
        checks = {
            "statement_rows": len(self.trades),
            "cash_reconciliation_max_error": _round(max(cash_errors, default=0.0), 8),
            "fee_formula_max_error": _round(max(fee_errors, default=0.0), 8),
            "fee_component_sum_max_error": _round(
                max(fee_component_errors, default=0.0), 8
            ),
            "gross_amount_max_error": _round(
                max(gross_amount_errors, default=0.0), 8
            ),
            "slippage_formula_max_error": _round(
                max(slippage_errors, default=0.0), 8
            ),
            "position_formula_max_error": _round(
                max(position_formula_errors, default=0.0), 8
            ),
            "same_day_signal_fill_violations": same_day,
            "scheduled_execution_date_violations": schedule_violations,
            "duplicate_fill_id_violations": (
                len(self.trades)
                - len({trade["fill_id"] for trade in self.trades})
            ),
            "nonpositive_fill_violations": nonpositive_fills,
            "side_sign_violations": side_sign_violations,
            "fee_profile_violations": fee_profile_violations,
            "ashare_buy_lot_violations": ashare_buy_lot_violations,
            "event_phase_order_violations": order_violations,
            "long_only_negative_position_violations": long_only_short_positions,
            "cash_account_negative_cash_violations": int(
                self._online_integrity["cash_account_negative_cash_violations"]
            ),
            "gross_leverage_violations": int(
                self._online_integrity["gross_leverage_violations"]
            ),
            "gross_leverage_breach_events": int(
                self._online_integrity["gross_leverage_breach_events"]
            ),
            "automatic_deleveraging_events": int(
                self._online_integrity["automatic_deleveraging_events"]
            ),
            "leverage_limited_orders": int(
                self._online_integrity["leverage_limited_orders"]
            ),
            "max_open_gross_leverage_observed": _round(
                self._online_integrity["max_open_gross_leverage_observed"], 8
            ),
            "max_open_gross_leverage_after_control": _round(
                self._online_integrity["max_open_gross_leverage_after_control"], 8
            ),
            "gross_leverage_violation_details": list(
                self._gross_leverage_violation_details
            ),
            "position_state_quantity_violations": int(
                self._online_integrity["position_state_quantity_violations"]
            ),
            "stale_position_writeoffs": int(
                self._online_integrity["stale_position_writeoffs"]
            ),
            "intrabar_ambiguities": int(
                self._online_integrity["intrabar_ambiguities"]
            ),
            "ledger_source_of_truth": True,
        }
        checks["all_pass"] = (
            checks["cash_reconciliation_max_error"] <= 1e-5
            and checks["fee_formula_max_error"] <= 1e-8
            and checks["fee_component_sum_max_error"] <= 1e-8
            and checks["gross_amount_max_error"] <= 1e-5
            and checks["slippage_formula_max_error"] <= 1e-5
            and checks["position_formula_max_error"] <= 1e-5
            and same_day == 0
            and schedule_violations == 0
            and checks["duplicate_fill_id_violations"] == 0
            and nonpositive_fills == 0
            and side_sign_violations == 0
            and fee_profile_violations == 0
            and ashare_buy_lot_violations == 0
            and order_violations == 0
            and long_only_short_positions == 0
            and checks["cash_account_negative_cash_violations"] == 0
            and checks["gross_leverage_violations"] == 0
            and checks["position_state_quantity_violations"] == 0
        )
        return checks

    def _online_integrity_checks(self) -> dict:
        long_only_short_positions = (
            sum(quantity < -1e-8 for quantity in self.positions.values())
            if self.config.mode == "long_only"
            else 0
        )
        checks = {
            "statement_rows": self._fill_seq,
            "materialized_statement_rows": 0,
            "audit_mode": "online_fill_reconciliation",
            **{
                key: (
                    _round(value, 8)
                    if key.endswith("_max_error") or key.startswith("max_open_")
                    else int(value)
                )
                for key, value in self._online_integrity.items()
            },
            "gross_leverage_violation_details": list(
                self._gross_leverage_violation_details
            ),
            "long_only_negative_position_violations": long_only_short_positions,
            "ledger_source_of_truth": True,
        }
        checks["all_pass"] = (
            checks["cash_reconciliation_max_error"] <= 1e-5
            and checks["fee_formula_max_error"] <= 1e-8
            and checks["fee_component_sum_max_error"] <= 1e-8
            and checks["gross_amount_max_error"] <= 1e-5
            and checks["slippage_formula_max_error"] <= 1e-5
            and checks["position_formula_max_error"] <= 1e-5
            and checks["same_day_signal_fill_violations"] == 0
            and checks["scheduled_execution_date_violations"] == 0
            and checks["duplicate_fill_id_violations"] == 0
            and checks["nonpositive_fill_violations"] == 0
            and checks["side_sign_violations"] == 0
            and checks["fee_profile_violations"] == 0
            and checks["ashare_buy_lot_violations"] == 0
            and checks["event_phase_order_violations"] == 0
            and long_only_short_positions == 0
            and checks["cash_account_negative_cash_violations"] == 0
            and checks["gross_leverage_violations"] == 0
            and checks["position_state_quantity_violations"] == 0
        )
        return checks

    def result(self) -> dict:
        if not self.daily:
            raise ValueError("回测没有产生交易日")
        returns = [float(row["daily_return"]) for row in self.daily]
        net_nav = [float(row["net_nav"]) for row in self.daily]
        gross_proxy = [float(row["same_orders_cost_free_nav_proxy"]) for row in self.daily]
        n = len(returns)
        final_nav = net_nav[-1]
        ann_return = final_nav ** (252.0 / max(1, n)) - 1.0 if final_nav > 0 else -1.0
        mean_return = sum(returns) / n
        variance = sum((value - mean_return) ** 2 for value in returns) / max(1, n - 1)
        volatility = math.sqrt(variance)
        sharpe = mean_return / volatility * math.sqrt(252.0) if volatility > 1e-12 else 0.0
        peak = 1.0
        max_drawdown = 0.0
        for value in net_nav:
            peak = max(peak, value)
            max_drawdown = max(max_drawdown, 1.0 - value / peak if peak else 0.0)
        downside = [min(0.0, value) for value in returns]
        downside_variance = sum(value * value for value in downside) / max(1, len(downside))
        downside_vol = math.sqrt(downside_variance)
        sortino = mean_return / downside_vol * math.sqrt(252.0) if downside_vol > 1e-12 else 0.0
        calmar = ann_return / max_drawdown if max_drawdown > 1e-12 else 0.0
        wins = [row for row in self._round_trips if float(row["net_pnl"]) > 0]
        losses = [row for row in self._round_trips if float(row["net_pnl"]) < 0]
        gross_profit = sum(float(row["net_pnl"]) for row in wins)
        gross_loss = abs(sum(float(row["net_pnl"]) for row in losses))
        avg_win = gross_profit / len(wins) if wins else 0.0
        avg_loss = gross_loss / len(losses) if losses else 0.0
        fee_total = self._cumulative_fees
        slippage_total = self._cumulative_slippage
        terminal_flat_sessions = 0
        for daily_row in reversed(self.daily):
            if (
                int(daily_row.get("positions", 0)) == 0
                and int(daily_row.get("fills", 0)) == 0
                and int(daily_row.get("orders_created", 0)) == 0
                and abs(float(daily_row.get("daily_return", 0.0))) <= 1e-12
            ):
                terminal_flat_sessions += 1
                continue
            break
        stats = {
            "protocol": BACKTEST_PROTOCOL,
            "days": n,
            "initial_capital": self.config.initial_capital,
            "final_nlv": self.daily[-1]["close_nlv"],
            "final_nav": _round(final_nav, 6),
            "ann_ret": _round(ann_return, 6),
            "ann_vol": _round(volatility * math.sqrt(252.0), 6),
            "sharpe": _round(sharpe, 4),
            "sortino": _round(sortino, 4),
            "calmar": _round(calmar, 4),
            "max_dd": _round(max_drawdown, 6),
            "avg_daily_turnover": _round(
                sum(float(row["turnover"]) for row in self.daily) / n, 6
            ),
            "avg_gross_exposure": _round(
                sum(float(row["gross_exposure"]) for row in self.daily) / n,
                6,
            ),
            "avg_net_exposure": _round(
                sum(float(row["net_exposure"]) for row in self.daily) / n,
                6,
            ),
            "fills": self._fill_seq,
            "orders": self._order_seq,
            "orders_executed": self._executed_orders,
            "rejected_orders": self._rejected_orders,
            "partial_orders": self._partial_orders,
            "fill_rate": _round(
                self._filled_execution_quantity
                / self._requested_execution_quantity
                if self._requested_execution_quantity
                else 1.0,
                6,
            ),
            "commission_and_tax": _round(fee_total, 6),
            "slippage_cost": _round(slippage_total, 6),
            "borrow_cost": _round(self._cumulative_borrow, 6),
            "margin_interest": _round(self._cumulative_margin_interest, 6),
            "total_execution_cost": _round(
                fee_total
                + slippage_total
                + self._cumulative_borrow
                + self._cumulative_margin_interest,
                6,
            ),
            "fee_profile": self.config.resolved_fee_profile,
            "currency": self.config.currency,
            "open_positions": len(self.positions),
            "same_orders_cost_free_final_nav_proxy": _round(gross_proxy[-1], 6),
            "closed_trades": len(self._round_trips),
            "win_rate": _round(len(wins) / len(self._round_trips), 6) if self._round_trips else 0.0,
            "profit_factor": _round(gross_profit / gross_loss, 6) if gross_loss > 1e-12 else None,
            "payoff_ratio": _round(avg_win / avg_loss, 6) if avg_loss > 1e-12 else None,
            "avg_trade_return": _round(
                sum(float(row["return"]) for row in self._round_trips) / len(self._round_trips),
                8,
            ) if self._round_trips else 0.0,
            "avg_holding_sessions": _round(
                sum(int(row["holding_sessions"]) for row in self._round_trips) / len(self._round_trips),
                4,
            ) if self._round_trips else 0.0,
            "portfolio_liquidations": self._portfolio_liquidations,
            "portfolio_risk_trigger_events": self._portfolio_risk_trigger_events,
            "portfolio_risk_rearms": self._portfolio_risk_rearms,
            "portfolio_risk_active_sessions": self._portfolio_risk_active_sessions,
            "portfolio_risk_active_at_end": self._portfolio_risk_active,
            "portfolio_risk_last_trigger": self._portfolio_risk_last_trigger,
            "max_portfolio_risk_cycle_drawdown": _round(
                self._max_portfolio_risk_cycle_drawdown, 8
            ),
            "terminal_flat_sessions": terminal_flat_sessions,
            "exit_reason_counts": dict(sorted(self._exit_reason_counts.items())),
            "gross_leverage_breach_events": int(
                self._online_integrity["gross_leverage_breach_events"]
            ),
            "automatic_deleveraging_events": int(
                self._online_integrity["automatic_deleveraging_events"]
            ),
            "leverage_limited_orders": int(
                self._online_integrity["leverage_limited_orders"]
            ),
            "max_open_gross_leverage_observed": _round(
                self._online_integrity["max_open_gross_leverage_observed"], 8
            ),
            "max_open_gross_leverage_after_control": _round(
                self._online_integrity["max_open_gross_leverage_after_control"], 8
            ),
        }
        return {
            "protocol": BACKTEST_PROTOCOL,
            "config": asdict(self.config) | {
                "fee_profile": self.config.resolved_fee_profile,
                "currency": self.config.currency,
                "signal_timing": "t_close",
                "execution_timing": "t_plus_1_raw_open",
                "valuation_timing": "session_raw_close",
                "intraday_resolution": "daily_ohlc_conservative_path",
                "liquidity_timing": "previous_20_session_adv",
                "account_type": self.config.resolved_account_type,
                "configured_gross_target": self.config.configured_gross_target,
                "effective_long_gross_target": self.config.effective_long_gross_target,
                "effective_short_gross_target": self.config.effective_short_gross_target,
                "leverage_headroom": self.config.leverage_headroom,
                "leverage_control_protocol": "per_fill_hard_cap_and_open_auto_deleverage_v1",
            },
            "fee_schedule": fee_schedule_snapshot(
                self.config.market, self.config.resolved_fee_profile
            ),
            "stats": stats,
            "curve": {
                "dates": [row["trade_date"] for row in self.daily],
                "equity": net_nav,
                "cost_free_proxy": gross_proxy,
                "daily_ret": returns,
            },
            "daily_steps": self.daily,
            "trades": self.trades,
            "events": self.events,
            "round_trips": self._round_trips,
            "positions": [
                {
                    "symbol": symbol,
                    "name": self.names.get(symbol, symbol),
                    "quantity": _round(quantity, 6),
                    "last_price": _round(self.last_close.get(symbol, 0.0), 6),
                    "market_value": _round(quantity * self.last_close.get(symbol, 0.0), 6),
                    "state": self.position_states[symbol].snapshot()
                    if symbol in self.position_states else None,
                    "risk_levels": (
                        {
                            "stop_price": risk_levels(
                                self.position_states[symbol], self.config.exit_policy
                            )[0],
                            "take_profit_price": risk_levels(
                                self.position_states[symbol], self.config.exit_policy
                            )[1],
                        }
                        if symbol in self.position_states else None
                    ),
                }
                for symbol, quantity in sorted(self.positions.items())
            ],
            "integrity": (
                self._integrity_checks()
                if self.capture_detail
                else self._online_integrity_checks()
            ),
            "detail_capture": (
                "full_statement" if self.capture_detail else "summary_online_audit"
            ),
        }


def _prepare_backtest_frame(
    *,
    expression: str,
    universe_n: int,
    start: str,
    end: str,
    panel_glob: str | None,
    market: str,
    forward_horizon: int | None = None,
    dsl_fields: list[str] | tuple[str, ...] | None = None,
    atr_period: int = 14,
) -> tuple[pl.DataFrame, PanelStore]:
    fields = list(dsl_fields or get_dsl_fields(market))
    store = PanelStore.get(panel_glob, market, factor_fields=fields)
    snapshot_reader = getattr(store, "read_snapshot", None)
    if callable(snapshot_reader):
        df, dates, _, _ = snapshot_reader()
    else:  # Compatibility with lightweight integrations and test doubles.
        df = store.ensure_loaded()
        dates = tuple(store.trading_dates)
    pipe = parse(expression, fields)
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    in_range = [value for value in dates if start_date <= value <= end_date]
    if len(in_range) < 60:
        raise ValueError("回测样本不足 (有效交易日 < 60)")
    first_index = dates.index(in_range[0])
    history_need = max(required_history(expression), int(atr_period) + 22)
    history_start = dates[max(0, first_index - history_need - 2)]

    # Keep every security that enters the requested liquidity universe during
    # the test.  Its rows remain available after it leaves the universe, so an
    # existing position can still be marked and sold rather than disappearing.
    symbols = (
        df.lazy()
        .filter(
            pl.col("trade_date").is_between(start_date, end_date)
            & (pl.col("univ_rank") <= universe_n)
        )
        .select("ts_code")
        .unique()
        .collect()["ts_code"]
        .to_list()
    )
    select_columns = [
        "trade_date",
        "ts_code",
        "name",
        "factor",
        "univ_rank",
        "raw_open",
        "raw_high",
        "raw_low",
        "raw_close",
        "vol",
        "amount",
        "adjustment_factor",
        "can_buy_open_proxy",
        "can_sell_open_proxy",
        "_adv20_prev",
        "_atr_pct",
        "_vol20_prev",
    ]
    if forward_horizon is not None:
        forward_column = f"fwd_{int(forward_horizon)}"
        if forward_column not in df.columns:
            raise ValueError(f"不支持的 forward_horizon: {forward_horizon}")
        select_columns.append(forward_column)
    # Factor semantics must match the evaluator: calculate every DSL stage on
    # the complete market cross-section first, then select securities needed
    # by the requested liquidity universe and position ledger.  Filtering to
    # the future union of Top-N names before rank/zscore/winsor would leak
    # future universe membership into earlier cross-sections and change nested
    # time-series expressions.
    source = df.lazy()
    if "raw_high" not in df.columns:
        source = source.with_columns(pl.col("high").alias("raw_high"))
    if "raw_low" not in df.columns:
        source = source.with_columns(pl.col("low").alias("raw_low"))
    enriched = (
        pipe.apply(
            source.filter(pl.col("trade_date").is_between(history_start, end_date))
        )
        .with_columns(
            pl.col("close").shift(1).over("ts_code").alias("_prev_close_for_atr"),
            pl.col("vol")
            .shift(1)
            .rolling_mean(window_size=20, min_samples=1)
            .over("ts_code")
            .alias("_adv20_prev"),
            (pl.col("close") / pl.col("close").shift(1).over("ts_code") - 1.0)
            .alias("_ret1_for_vol"),
        )
        .with_columns(
            pl.max_horizontal(
                (pl.col("high") - pl.col("low")).abs(),
                (pl.col("high") - pl.col("_prev_close_for_atr")).abs(),
                (pl.col("low") - pl.col("_prev_close_for_atr")).abs(),
            ).alias("_true_range")
        )
        .with_columns(
            (
                pl.col("_true_range")
                .rolling_mean(
                    window_size=int(atr_period),
                    min_samples=max(2, int(atr_period) // 2),
                )
                .over("ts_code")
                / pl.col("close").abs()
            ).alias("_atr_pct"),
            (
                pl.col("_ret1_for_vol")
                .shift(1)
                .rolling_std(window_size=20, min_samples=5)
                .over("ts_code")
                * math.sqrt(252.0)
            ).alias("_vol20_prev"),
        )
    )
    full_cross_section = (
        enriched
        .select(select_columns)
        .cache()
    )
    frame = (
        full_cross_section
        .filter(pl.col("trade_date").is_between(start_date, end_date))
        .filter(pl.col("ts_code").is_in(symbols))
        .sort("trade_date", "ts_code")
        # Keep the post-factor symbol/date filters behind the cache barrier.
        # This is intentionally explicit rather than relying on the optimizer
        # to infer that predicate pushdown across window expressions is unsafe.
        .collect(
            optimizations=pl.QueryOptFlags(predicate_pushdown=False)
        )
    )
    return frame, store


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_artifacts(result: dict, artifact_dir: Path) -> dict:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    frame_specs = {
        "trades": (result["trades"], artifact_dir / "settlement_statement.parquet"),
        "events": (result["events"], artifact_dir / "event_ledger.parquet"),
        "daily_steps": (result["daily_steps"], artifact_dir / "daily_ledger.parquet"),
        "round_trips": (result.get("round_trips", []), artifact_dir / "round_trip_ledger.parquet"),
    }
    files: dict[str, dict] = {}
    for name, (rows, path) in frame_specs.items():
        if rows:
            # Execution ledgers can legitimately contain an integer-looking
            # prefix followed by fractional quantities (US partial fills and
            # portfolio liquidations).  Inspect the full ledger when inferring
            # schema so a late float does not fail artifact publication.
            pl.DataFrame(rows, infer_schema_length=None).write_parquet(
                path, compression="zstd"
            )
        else:
            pl.DataFrame({"empty": []}).write_parquet(path, compression="zstd")
        files[name] = {
            "filename": path.name,
            "sha256": _hash_file(path),
            "rows": len(rows),
        }
    csv_path = artifact_dir / "settlement_statement.csv"
    if result["trades"]:
        pl.DataFrame(result["trades"], infer_schema_length=None).write_csv(csv_path)
    else:
        csv_path.write_text("fill_id,order_id,signal_date,trade_date\n", encoding="utf-8")
    files["statement_csv"] = {
        "filename": csv_path.name,
        "sha256": _hash_file(csv_path),
        "rows": len(result["trades"]),
    }
    round_trip_csv = artifact_dir / "round_trip_statement.csv"
    if result.get("round_trips"):
        pl.DataFrame(
            result["round_trips"], infer_schema_length=None
        ).write_csv(round_trip_csv)
    else:
        round_trip_csv.write_text(
            "symbol,entry_date,exit_date,direction,quantity,entry_price,exit_price,net_pnl,exit_reason\n",
            encoding="utf-8",
        )
    files["round_trip_csv"] = {
        "filename": round_trip_csv.name,
        "sha256": _hash_file(round_trip_csv),
        "rows": len(result.get("round_trips", [])),
    }
    manifest = {
        "protocol": BACKTEST_PROTOCOL,
        "config": result["config"],
        "fee_schedule": result["fee_schedule"],
        "stats": result["stats"],
        "integrity": result["integrity"],
        "files": files,
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest["manifest_sha256"] = _hash_file(manifest_path)
    return manifest


def run_backtest(
    expression: str,
    universe_n: int = 500,
    start: str = "2015-01-01",
    end: str = "2024-12-31",
    cost_bps: float | None = None,
    direction: int = 1,
    mode: str = DEFAULT_PORTFOLIO_MODE,
    panel_glob: str | None = None,
    market: str = "us",
    borrow_cost_bps_annual: float = 0.0,
    top_fraction: float = 0.20,
    *,
    initial_capital: float = 1_000_000.0,
    rebalance_every: int = 5,
    slippage_bps: float | None = None,
    max_volume_participation: float = 0.10,
    fee_profile: str | None = None,
    account_type: str = "auto",
    cash_buffer_fraction: float = 0.0,
    max_gross_leverage: float = 2.0,
    margin_interest_bps_annual: float = 0.0,
    position_sizing: str = "equal_weight",
    max_positions: int = 10_000,
    max_position_weight: float = 1.0,
    min_trade_notional: float = 0.0,
    rebalance_buffer_pct: float = 0.0,
    long_gross_target: float = 1.0,
    short_gross_target: float | None = None,
    risk_per_position_fraction: float = 0.01,
    spread_bps: float = 0.0,
    impact_model: str = "fixed",
    impact_coefficient_bps: float = 0.0,
    unfilled_order_policy: str = "cancel",
    max_order_age_sessions: int = 1,
    max_stale_sessions: int = 20,
    liquidate_at_end: bool = False,
    portfolio_stop_drawdown_pct: float | None = None,
    portfolio_daily_loss_pct: float | None = None,
    risk_cooldown_sessions: int = 0,
    exit_policy: ExitPolicyConfig | dict | None = None,
    artifact_dir: str | Path | None = None,
    response_trade_limit: int = 200,
    capture_detail: bool = True,
) -> dict:
    """Run the event engine.

    ``cost_bps`` remains in the signature for API compatibility but is not used
    as a synthetic return haircut.  When ``slippage_bps`` is omitted, it is
    treated as the legacy caller's requested market-impact assumption.
    """
    if artifact_dir is not None and not capture_detail:
        raise ValueError("写入交割单产物时 capture_detail 必须为 true")
    resolved_exit_policy = (
        exit_policy
        if isinstance(exit_policy, ExitPolicyConfig)
        else ExitPolicyConfig(**(exit_policy or {}))
    )
    resolved_slippage = (
        float(slippage_bps)
        if slippage_bps is not None
        else float(cost_bps or (5.0 if market == "ashare" else 2.0))
    )
    config = EventBacktestConfig(
        market=market,
        mode=mode,
        direction=direction,
        universe_n=universe_n,
        top_fraction=top_fraction,
        initial_capital=initial_capital,
        rebalance_every=rebalance_every,
        slippage_bps=resolved_slippage,
        max_volume_participation=max_volume_participation,
        borrow_cost_bps_annual=borrow_cost_bps_annual,
        fee_profile=fee_profile,
        account_type=account_type,
        cash_buffer_fraction=cash_buffer_fraction,
        max_gross_leverage=max_gross_leverage,
        margin_interest_bps_annual=margin_interest_bps_annual,
        position_sizing=position_sizing,
        max_positions=max_positions,
        max_position_weight=max_position_weight,
        min_trade_notional=min_trade_notional,
        rebalance_buffer_pct=rebalance_buffer_pct,
        long_gross_target=long_gross_target,
        short_gross_target=(
            float(short_gross_target)
            if short_gross_target is not None
            else (1.0 if mode == "long_short" else 0.0)
        ),
        risk_per_position_fraction=risk_per_position_fraction,
        spread_bps=spread_bps,
        impact_model=impact_model,
        impact_coefficient_bps=impact_coefficient_bps,
        unfilled_order_policy=unfilled_order_policy,
        max_order_age_sessions=max_order_age_sessions,
        max_stale_sessions=max_stale_sessions,
        liquidate_at_end=liquidate_at_end,
        portfolio_stop_drawdown_pct=portfolio_stop_drawdown_pct,
        portfolio_daily_loss_pct=portfolio_daily_loss_pct,
        risk_cooldown_sessions=risk_cooldown_sessions,
        exit_policy=resolved_exit_policy,
    )
    # Reject invalid account, leverage, sizing and exit semantics before the
    # potentially expensive panel scan.
    config.validate()
    frame, _ = _prepare_backtest_frame(
        expression=expression,
        universe_n=universe_n,
        start=start,
        end=end,
        panel_glob=panel_glob,
        market=market,
        atr_period=resolved_exit_policy.atr_period,
    )
    runner = StepEventBacktester(config, capture_detail=capture_detail)
    sessions = frame.partition_by("trade_date", maintain_order=True)
    session_dates = [session["trade_date"][0] for session in sessions]
    for index, session in enumerate(sessions):
        runner.step(
            trade_date=session_dates[index],
            rows=session.to_dicts(),
            next_trade_date=(
                session_dates[index + 1]
                if index + 1 < len(session_dates)
                else None
            ),
            rebalance=index % rebalance_every == 0,
        )
    result = runner.result()
    manifest = None
    if artifact_dir is not None:
        manifest = _write_artifacts(result, Path(artifact_dir))
    compact = {
        **result,
        "trades": result["trades"][:max(0, response_trade_limit)],
        "events": result["events"][:max(0, response_trade_limit)],
        "round_trips": result.get("round_trips", [])[:max(0, response_trade_limit)],
        "daily_steps": result["daily_steps"][-min(120, len(result["daily_steps"])):],
        "trade_page": {
            "offset": 0,
            "limit": max(0, response_trade_limit),
            "returned": min(len(result["trades"]), max(0, response_trade_limit)),
            "total": len(result["trades"]),
        },
        "event_page": {
            "offset": 0,
            "limit": max(0, response_trade_limit),
            "returned": min(len(result["events"]), max(0, response_trade_limit)),
            "total": len(result["events"]),
        },
        "artifacts": manifest,
    }
    return compact
