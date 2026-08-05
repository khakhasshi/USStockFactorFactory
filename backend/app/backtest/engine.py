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
from dataclasses import asdict, dataclass
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

BACKTEST_PROTOCOL = "step_event_v1"
_EVENT_PHASE_ORDER = {
    "SESSION_OPEN": 0,
    "OPEN_CORPORATE_ACTION": 1,
    "OPEN_EXECUTION": 2,
    "CLOSE_FINANCING": 3,
    "CLOSE_SIGNAL": 4,
    "SESSION_CLOSE": 5,
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
        self._cumulative_traded_notional = 0.0
        self._rejected_orders = 0
        self._partial_orders = 0
        self._executed_orders = 0
        self._requested_execution_quantity = 0.0
        self._filled_execution_quantity = 0.0
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
        }
        self._seen_fill_ids: set[str] = set()
        self._last_event_date = ""
        self._last_event_phase = -1

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
        if trade["signal_date"] >= trade["trade_date"]:
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

    def _volume_limit(self, row: dict, requested_abs: float) -> float:
        volume = float(row.get("vol") or 0.0)
        if volume <= 0:
            return 0.0
        cap = volume * self.config.max_volume_participation
        lot = self.config.lot_size
        capped = min(requested_abs, math.floor(cap / lot) * lot)
        return max(0.0, capped)

    def _max_affordable_ashare_buy(
        self,
        requested: float,
        fill_price: float,
        trade_date: date,
    ) -> float:
        lot = self.config.lot_size
        quantity = math.floor(min(requested, self.cash / max(fill_price, 1e-12)) / lot) * lot
        while quantity > 0:
            fee = calculate_trade_fees(
                market="ashare",
                side="BUY",
                quantity=quantity,
                price=fill_price,
                trade_date=trade_date,
                profile=self.config.resolved_fee_profile,
            )
            if quantity * fill_price + fee.total <= self.cash + 1e-9:
                return float(quantity)
            quantity -= lot
        return 0.0

    def _record_rejection(
        self,
        *,
        trade_date: date,
        order: dict,
        reason: str,
        requested: float,
    ) -> None:
        self._rejected_orders += 1
        self._emit(
            trade_date,
            "OPEN_EXECUTION",
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
    ) -> None:
        symbol = order["symbol"]
        current = float(self.positions.get(symbol, 0.0))
        requested_signed = _round(
            float(order["target_quantity"]) - current,
            6,
        )
        if abs(requested_signed) < 1e-8:
            return
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
            )
            return

        side = "BUY" if requested_signed > 0 else "SELL"
        permission = "can_buy_open_proxy" if side == "BUY" else "can_sell_open_proxy"
        if row.get(permission) is False:
            self._record_rejection(
                trade_date=trade_date,
                order=order,
                reason=f"{permission}=false，停牌或涨跌停代理阻止成交",
                requested=requested_signed,
            )
            return

        filled_abs = self._volume_limit(row, requested_abs)
        # The price written to the statement is the exact price used for cash,
        # fees, affordability, and positions.  Never keep a hidden
        # higher-precision execution price behind a rounded statement value.
        reference_price = _round(float(row["raw_open"]), 6)
        slip = self.config.slippage_bps / 10_000.0
        fill_price = _round(
            reference_price * (1.0 + slip if side == "BUY" else 1.0 - slip),
            6,
        )
        if self.config.market == "ashare" and side == "BUY":
            filled_abs = self._max_affordable_ashare_buy(filled_abs, fill_price, trade_date)
        filled_abs = _round(filled_abs, 6)
        if filled_abs <= 0:
            self._record_rejection(
                trade_date=trade_date,
                order=order,
                reason="成交量参与率或可用现金限制导致零成交",
                requested=requested_signed,
            )
            return

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
        self.cash -= signed_quantity * fill_price + fees.total
        position_after = _round(current + signed_quantity, 6)
        if abs(position_after) < 1e-8:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = position_after
        self.names[symbol] = str(row.get("name") or symbol)
        slippage_cost = filled_abs * abs(fill_price - reference_price)
        self._cumulative_fees += fees.total
        self._cumulative_slippage += slippage_cost
        self._cumulative_traded_notional += filled_abs * fill_price
        self._fill_seq += 1
        nlv_after, _, _ = self._nlv(market, "raw_open")
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
            "cash_before": _round(cash_before, 6),
            "cash_after": _round(self.cash, 6),
            "position_before": _round(position_before, 6),
            "position_after": _round(position_after, 6),
            "nlv_after": _round(nlv_after, 6),
            "fee_profile": fees.profile,
            "fee_notes": fees.notes,
        }
        self._audit_trade_online(trade)
        if self.capture_detail:
            self.trades.append(trade)
        self._emit(
            trade_date,
            "OPEN_EXECUTION",
            "FILL" if unfilled <= 1e-8 else "PARTIAL_FILL",
            symbol=symbol,
            order_id=order["order_id"],
            message=f"{side} {filled_abs:g} @ {fill_price:.4f}",
            payload={
                "fill_id": trade["fill_id"],
                "total_fees": fees.total,
                "slippage_cost": slippage_cost,
                "unfilled_quantity": unfilled,
            },
        )

    def _execute_pending(self, trade_date: date, market: dict[str, dict]) -> None:
        due = [order for order in self.pending_orders if order["execute_date"] == str(trade_date)]
        self.pending_orders = [order for order in self.pending_orders if order["execute_date"] != str(trade_date)]
        # Sells/shorts release cash before buys.  Stable order_id preserves a
        # deterministic statement under identical input data.
        due.sort(
            key=lambda order: (
                float(order["target_quantity"]) - float(self.positions.get(order["symbol"], 0.0)) > 0,
                order["order_id"],
            )
        )
        for order in due:
            self._execute_order(trade_date, order, market)

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
        longs = eligible[:count]
        shorts = eligible[-count:] if self.config.mode == "long_short" else []
        target: dict[str, float] = {}
        lot = self.config.lot_size
        long_dollars = max(0.0, nlv) / len(longs)
        for row in longs:
            quantity = math.floor(long_dollars / float(row["raw_close"]) / lot) * lot
            if quantity > 0:
                target[row["ts_code"]] = float(quantity)
                self.names[row["ts_code"]] = str(row.get("name") or row["ts_code"])
        if shorts:
            short_dollars = max(0.0, nlv) / len(shorts)
            for row in shorts:
                quantity = math.floor(short_dollars / float(row["raw_close"]) / lot) * lot
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
        created = 0
        for symbol in sorted(set(self.positions) | set(targets)):
            target = float(targets.get(symbol, 0.0))
            current = float(self.positions.get(symbol, 0.0))
            if abs(target - current) < 1e-8:
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
        event_start = self._event_seq
        fill_start = self._fill_seq
        notional_start = self._cumulative_traded_notional
        self._emit(trade_date, "SESSION_OPEN", "SESSION_OPEN", message="进入交易日")
        self._apply_corporate_actions(trade_date, market)
        open_nlv_before, _, _ = self._nlv(market, "raw_open")
        self._execute_pending(trade_date, market)
        open_nlv_after, _, _ = self._nlv(market, "raw_open")

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
        close_nlv = close_nlv_before_financing - borrow_fee
        traded_notional = self._cumulative_traded_notional - notional_start
        daily_return = close_nlv / self._previous_close_nlv - 1.0 \
            if self._previous_close_nlv > 0 else 0.0
        created_orders = 0
        if rebalance and next_trade_date is not None and close_nlv > 0:
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
        )
        row = {
            "trade_date": str(trade_date),
            "open_nlv_before_fills": _round(open_nlv_before, 6),
            "open_nlv_after_fills": _round(open_nlv_after, 6),
            "close_nlv": _round(close_nlv, 6),
            "net_nav": _round(close_nlv / self.config.initial_capital, 8),
            "same_orders_cost_free_nav_proxy": _round(
                gross_proxy_nlv / self.config.initial_capital, 8
            ),
            "daily_return": _round(daily_return, 8),
            "cash": _round(self.cash, 6),
            "long_market_value": _round(long_value, 6),
            "short_market_value": _round(short_value, 6),
            "gross_exposure": _round(
                (long_value + abs(short_value)) / close_nlv if close_nlv else 0.0, 6
            ),
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
            if trade["signal_date"] >= trade["trade_date"]:
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
                    if key.endswith("_max_error")
                    else int(value)
                )
                for key, value in self._online_integrity.items()
            },
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
        fee_total = self._cumulative_fees
        slippage_total = self._cumulative_slippage
        stats = {
            "protocol": BACKTEST_PROTOCOL,
            "days": n,
            "initial_capital": self.config.initial_capital,
            "final_nlv": self.daily[-1]["close_nlv"],
            "final_nav": _round(final_nav, 6),
            "ann_ret": _round(ann_return, 6),
            "ann_vol": _round(volatility * math.sqrt(252.0), 6),
            "sharpe": _round(sharpe, 4),
            "max_dd": _round(max_drawdown, 6),
            "avg_daily_turnover": _round(
                sum(float(row["turnover"]) for row in self.daily) / n, 6
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
            "total_execution_cost": _round(
                fee_total + slippage_total + self._cumulative_borrow, 6
            ),
            "fee_profile": self.config.resolved_fee_profile,
            "currency": self.config.currency,
            "open_positions": len(self.positions),
            "same_orders_cost_free_final_nav_proxy": _round(gross_proxy[-1], 6),
        }
        return {
            "protocol": BACKTEST_PROTOCOL,
            "config": asdict(self.config) | {
                "fee_profile": self.config.resolved_fee_profile,
                "currency": self.config.currency,
                "signal_timing": "t_close",
                "execution_timing": "t_plus_1_raw_open",
                "valuation_timing": "session_raw_close",
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
            "positions": [
                {
                    "symbol": symbol,
                    "name": self.names.get(symbol, symbol),
                    "quantity": _round(quantity, 6),
                    "last_price": _round(self.last_close.get(symbol, 0.0), 6),
                    "market_value": _round(quantity * self.last_close.get(symbol, 0.0), 6),
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
) -> tuple[pl.DataFrame, PanelStore]:
    store = PanelStore.get(panel_glob, market)
    df = store.ensure_loaded()
    fields = get_dsl_fields(market)
    pipe = parse(expression, fields)
    start_date = date.fromisoformat(start)
    end_date = date.fromisoformat(end)
    dates = store.trading_dates
    in_range = [value for value in dates if start_date <= value <= end_date]
    if len(in_range) < 60:
        raise ValueError("回测样本不足 (有效交易日 < 60)")
    first_index = dates.index(in_range[0])
    history_start = dates[max(0, first_index - required_history(expression) - 2)]

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
        "raw_close",
        "vol",
        "amount",
        "adjustment_factor",
        "can_buy_open_proxy",
        "can_sell_open_proxy",
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
    full_cross_section = (
        pipe.apply(
            df.lazy().filter(
                pl.col("trade_date").is_between(history_start, end_date)
            )
        )
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
    }
    files: dict[str, dict] = {}
    for name, (rows, path) in frame_specs.items():
        if rows:
            pl.DataFrame(rows).write_parquet(path, compression="zstd")
        else:
            pl.DataFrame({"empty": []}).write_parquet(path, compression="zstd")
        files[name] = {
            "filename": path.name,
            "sha256": _hash_file(path),
            "rows": len(rows),
        }
    csv_path = artifact_dir / "settlement_statement.csv"
    if result["trades"]:
        pl.DataFrame(result["trades"]).write_csv(csv_path)
    else:
        csv_path.write_text("fill_id,order_id,signal_date,trade_date\n", encoding="utf-8")
    files["statement_csv"] = {
        "filename": csv_path.name,
        "sha256": _hash_file(csv_path),
        "rows": len(result["trades"]),
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
    frame, _ = _prepare_backtest_frame(
        expression=expression,
        universe_n=universe_n,
        start=start,
        end=end,
        panel_glob=panel_glob,
        market=market,
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
