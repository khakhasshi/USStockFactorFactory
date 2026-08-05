"""Deterministic, auditable equity fee schedules.

The fee engine intentionally returns a component breakdown for every fill.  A
backtest is not allowed to collapse these charges into a single bps haircut:
the settlement statement and regression tests must be able to recompute each
line independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from decimal import Decimal, ROUND_HALF_UP


ASHARE_WAN2_NO_MIN_PROFILE = "ashare_wan2_no_min_v1"
IBKR_PRO_FIXED_PROFILE = "ibkr_pro_fixed_us_v1"


def _money(value: float) -> float:
    """Round a cash charge to cents using broker-style half-up rounding."""
    return float(Decimal(str(max(0.0, value))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


@dataclass(frozen=True)
class FeeBreakdown:
    profile: str
    currency: str
    commission: float
    stamp_duty: float = 0.0
    transfer_fee: float = 0.0
    regulatory_fee: float = 0.0
    exchange_fee: float = 0.0
    total: float = 0.0
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def ashare_stamp_rate(trade_date: date) -> float:
    """A-share seller stamp duty: 1‰ historically, 0.5‰ since 2023-08-28."""
    return 0.0005 if trade_date >= date(2023, 8, 28) else 0.001


def ashare_transfer_rate(trade_date: date) -> float:
    """ChinaClear A-share transfer fee, charged on both sides."""
    return 0.00001 if trade_date >= date(2022, 4, 29) else 0.00002


def calculate_trade_fees(
    *,
    market: str,
    side: str,
    quantity: float,
    price: float,
    trade_date: date,
    profile: str | None = None,
) -> FeeBreakdown:
    """Return the exact fee components for one executed order.

    ``quantity`` is absolute executed shares and ``price`` is the executed raw
    market price.  Slippage and stock-borrow financing are deliberately not
    included here; they are separate execution/financing ledger entries.
    """
    side = side.upper()
    if side not in {"BUY", "SELL"}:
        raise ValueError("side 必须是 BUY 或 SELL")
    if quantity <= 0 or price <= 0:
        raise ValueError("quantity 和 price 必须为正数")
    notional = float(quantity) * float(price)

    if market == "ashare":
        resolved = profile or ASHARE_WAN2_NO_MIN_PROFILE
        if resolved != ASHARE_WAN2_NO_MIN_PROFILE:
            raise ValueError(f"不支持的 A 股费率: {resolved}")
        commission = _money(notional * 0.0002)  # 万2，明确免除每笔最低 5 元
        transfer = _money(notional * ashare_transfer_rate(trade_date))
        stamp = _money(notional * ashare_stamp_rate(trade_date)) if side == "SELL" else 0.0
        total = _money(commission + transfer + stamp)
        return FeeBreakdown(
            profile=resolved,
            currency="CNY",
            commission=commission,
            stamp_duty=stamp,
            transfer_fee=transfer,
            total=total,
            notes="券商佣金万2免5；佣金已含交易规费；印花税仅卖出；过户费双向",
        )

    if market == "us":
        resolved = profile or IBKR_PRO_FIXED_PROFILE
        if resolved != IBKR_PRO_FIXED_PROFILE:
            raise ValueError(f"不支持的美股费率: {resolved}")
        # IBKR Pro Fixed US equities: USD 0.005/share, USD 1 minimum,
        # capped at 1% of trade value.  The fixed plan is treated as
        # all-inclusive for exchange/clearing/regulatory transaction charges.
        commission = _money(min(max(float(quantity) * 0.005, 1.0), notional * 0.01))
        return FeeBreakdown(
            profile=resolved,
            currency="USD",
            commission=commission,
            total=commission,
            notes="IBKR Pro Fixed；每股 $0.005，最低 $1，最高成交额 1%；交易规费包含在固定佣金内",
        )

    raise ValueError("market 必须是 ashare 或 us")


def fee_schedule_snapshot(market: str, profile: str | None = None) -> dict:
    if market == "ashare":
        resolved = profile or ASHARE_WAN2_NO_MIN_PROFILE
        if resolved != ASHARE_WAN2_NO_MIN_PROFILE:
            raise ValueError(f"不支持的 A 股费率: {resolved}")
        return {
            "profile": resolved,
            "schedule_version": "2026-08-05",
            "currency": "CNY",
            "commission_rate": 0.0002,
            "commission_minimum": 0.0,
            "commission_source": "task_policy_user_specified_wan2_no_minimum",
            "stamp_duty_sell": {
                "through_2023-08-27": 0.001,
                "from_2023-08-28": 0.0005,
            },
            "transfer_fee_both_sides": {
                "through_2022-04-28": 0.00002,
                "from_2022-04-29": 0.00001,
            },
            "rounding": "CNY cent, ROUND_HALF_UP per component",
            "source_urls": {
                "stamp_duty": "https://fgk.chinatax.gov.cn/zcfgk/c102416/c5211343/content.html",
                "sse_fee_reference": "https://one.sse.com.cn/onething/gptz/",
                "transfer_fee_cut": "https://www.xinhuanet.com/2022-04/28/c_1128605983.htm",
            },
        }
    if market == "us":
        resolved = profile or IBKR_PRO_FIXED_PROFILE
        if resolved != IBKR_PRO_FIXED_PROFILE:
            raise ValueError(f"不支持的美股费率: {resolved}")
        return {
            "profile": resolved,
            "schedule_version": "2026-08-05",
            "currency": "USD",
            "commission_per_share": 0.005,
            "minimum_per_order": 1.0,
            "maximum_fraction_of_trade_value": 0.01,
            "exchange_clearing_regulatory": "included_in_fixed_commission",
            "rounding": "USD cent, ROUND_HALF_UP per order",
            "source_urls": {
                "ibkr_official": "https://www.interactivebrokers.com/en/pricing/commissions-stocks.php",
            },
        }
    raise ValueError("market 必须是 ashare 或 us")
