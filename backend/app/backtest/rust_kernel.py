"""Safe ctypes bridge and shadow-alignment policy for the Rust event kernel."""

from __future__ import annotations

import ctypes as ct
import os
import platform
import time
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


RUST_KERNEL_ABI = 2
RUST_KERNEL_PROTOCOL = "step_event_v3_rust_shadow"
DEFAULT_BACKTEST_BACKEND = os.getenv("FF_BACKTEST_BACKEND", "python").strip().lower()
ALIGNMENT_ABSOLUTE_TOLERANCE = float(
    os.getenv("FF_RUST_ALIGNMENT_ABS_TOLERANCE", "0.00001")
)


class _Columns(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("row_count", ct.c_size_t),
        ("session_count", ct.c_size_t),
        ("session_offsets", ct.POINTER(ct.c_size_t)),
        ("day", ct.POINTER(ct.c_int32)),
        ("symbol", ct.POINTER(ct.c_int32)),
        ("univ_rank", ct.POINTER(ct.c_int32)),
        ("factor", ct.POINTER(ct.c_double)),
        ("raw_open", ct.POINTER(ct.c_double)),
        ("raw_close", ct.POINTER(ct.c_double)),
        ("volume", ct.POINTER(ct.c_double)),
        ("adv20_prev", ct.POINTER(ct.c_double)),
        ("atr_pct", ct.POINTER(ct.c_double)),
        ("vol20_prev", ct.POINTER(ct.c_double)),
        ("adjustment_factor", ct.POINTER(ct.c_double)),
        ("can_buy", ct.POINTER(ct.c_uint8)),
        ("can_sell", ct.POINTER(ct.c_uint8)),
    ]


class _Config(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("market", ct.c_uint8),
        ("mode", ct.c_uint8),
        ("direction", ct.c_int8),
        ("account_type", ct.c_uint8),
        ("impact_model", ct.c_uint8),
        ("universe_n", ct.c_int32),
        ("rebalance_every", ct.c_int32),
        ("max_positions", ct.c_int32),
        ("initial_capital", ct.c_double),
        ("top_fraction", ct.c_double),
        ("slippage_bps", ct.c_double),
        ("spread_bps", ct.c_double),
        ("max_volume_participation", ct.c_double),
        ("cash_buffer_fraction", ct.c_double),
        ("max_gross_leverage", ct.c_double),
        ("max_position_weight", ct.c_double),
        ("min_trade_notional", ct.c_double),
        ("rebalance_buffer_pct", ct.c_double),
        ("long_gross_target", ct.c_double),
        ("short_gross_target", ct.c_double),
        ("borrow_cost_bps_annual", ct.c_double),
        ("margin_interest_bps_annual", ct.c_double),
        ("impact_coefficient_bps", ct.c_double),
    ]


class _Trade(ct.Structure):
    _fields_ = [
        ("fill_seq", ct.c_uint64),
        ("order_seq", ct.c_uint64),
        ("signal_day", ct.c_int32),
        ("trade_day", ct.c_int32),
        ("symbol", ct.c_int32),
        ("side", ct.c_int8),
        ("requested_quantity", ct.c_double),
        ("filled_quantity", ct.c_double),
        ("reference_price", ct.c_double),
        ("fill_price", ct.c_double),
        ("commission", ct.c_double),
        ("stamp_duty", ct.c_double),
        ("transfer_fee", ct.c_double),
        ("total_fees", ct.c_double),
        ("cash_before", ct.c_double),
        ("cash_after", ct.c_double),
        ("position_before", ct.c_double),
        ("position_after", ct.c_double),
        ("participation", ct.c_double),
        ("slippage_cost", ct.c_double),
    ]


class _Daily(ct.Structure):
    _fields_ = [
        ("day", ct.c_int32),
        ("close_nlv", ct.c_double),
        ("net_nav", ct.c_double),
        ("daily_return", ct.c_double),
        ("cash", ct.c_double),
        ("long_market_value", ct.c_double),
        ("short_market_value", ct.c_double),
        ("gross_exposure", ct.c_double),
        ("net_exposure", ct.c_double),
        ("turnover", ct.c_double),
        ("borrow_fee", ct.c_double),
        ("margin_interest", ct.c_double),
        ("fills", ct.c_uint64),
        ("orders_created", ct.c_uint64),
        ("positions", ct.c_uint64),
    ]


class _Summary(ct.Structure):
    _fields_ = [
        ("abi_version", ct.c_uint32),
        ("status", ct.c_int32),
        ("trade_count", ct.c_size_t),
        ("daily_count", ct.c_size_t),
        ("order_count", ct.c_uint64),
        ("rejected_orders", ct.c_uint64),
        ("partial_orders", ct.c_uint64),
        ("final_nlv", ct.c_double),
        ("cumulative_fees", ct.c_double),
        ("cumulative_slippage", ct.c_double),
        ("cumulative_borrow", ct.c_double),
        ("cumulative_margin_interest", ct.c_double),
    ]


def _library_candidates() -> list[Path]:
    root = Path(__file__).resolve().parents[3]
    suffix = {"Darwin": "dylib", "Linux": "so", "Windows": "dll"}.get(
        platform.system(), "so"
    )
    filename = (
        "factorfactory_rust_backtest.dll"
        if suffix == "dll"
        else f"libfactorfactory_rust_backtest.{suffix}"
    )
    explicit = os.getenv("FF_RUST_KERNEL_LIBRARY")
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        root / "rust-backtest-kernel" / "target" / "release" / filename,
        root / "rust-backtest-kernel" / "target" / "debug" / filename,
    ])
    return candidates


def _load_library() -> tuple[ct.CDLL | None, str | None]:
    errors: list[str] = []
    for path in _library_candidates():
        if not path.exists():
            continue
        try:
            library = ct.CDLL(str(path))
            library.ff_abi_version.restype = ct.c_uint32
            library.ff_version.restype = ct.c_char_p
            library.ff_run_v1.argtypes = [
                ct.POINTER(_Columns), ct.POINTER(_Config), ct.POINTER(_Trade),
                ct.c_size_t, ct.POINTER(_Daily), ct.c_size_t, ct.POINTER(_Summary),
            ]
            library.ff_run_v1.restype = ct.c_int32
            if int(library.ff_abi_version()) != RUST_KERNEL_ABI:
                errors.append(f"{path}: ABI不匹配")
                continue
            return library, None
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return None, "; ".join(errors) or "未找到已编译 Rust 动态库"


_LIBRARY, _LOAD_ERROR = _load_library()


def reload_rust_kernel() -> None:
    """Reload a just-built kernel without restarting a test process."""
    global _LIBRARY, _LOAD_ERROR
    _LIBRARY, _LOAD_ERROR = _load_library()


def rust_kernel_capabilities() -> dict[str, Any]:
    version = None
    if _LIBRARY is not None:
        raw = _LIBRARY.ff_version()
        version = raw.decode("utf-8") if raw else "unknown"
    return {
        "available": _LIBRARY is not None,
        "load_error": _LOAD_ERROR,
        "version": version,
        "abi_version": RUST_KERNEL_ABI,
        "protocol": RUST_KERNEL_PROTOCOL,
        "configured_backend": DEFAULT_BACKTEST_BACKEND,
        "supported_paths": ["ashare_long_only", "us_long_only", "us_long_short"],
        "position_sizing": ["equal_weight"],
        "impact_models": ["fixed", "linear", "square_root"],
        "order_policies": ["cancel"],
        "alignment_absolute_tolerance": ALIGNMENT_ABSOLUTE_TOLERANCE,
        "safety": "Python step_event_v2 remains authoritative; shadow mismatch returns Python",
    }


def rust_eligibility(config: Any) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if _LIBRARY is None:
        reasons.append(_LOAD_ERROR or "Rust动态库不可用")
    if config.position_sizing != "equal_weight":
        reasons.append("首版Rust内核仅支持equal_weight")
    if config.unfilled_order_policy != "cancel":
        reasons.append("首版Rust内核尚未支持carry订单")
    if config.exit_policy.enabled:
        reasons.append("高级逐股退出规则仍由Python权威引擎执行")
    if config.liquidate_at_end:
        reasons.append("期末强平仍由Python权威引擎执行")
    if config.portfolio_stop_drawdown_pct is not None or config.portfolio_daily_loss_pct is not None:
        reasons.append("组合级风控状态机仍由Python权威引擎执行")
    if config.risk_cooldown_sessions:
        reasons.append("组合风控冷却仍由Python权威引擎执行")
    return not reasons, reasons


def _contiguous(frame: pl.DataFrame, column: str, dtype: np.dtype, *, fill: Any = None) -> np.ndarray:
    series = frame[column]
    if fill is not None:
        series = series.fill_null(fill)
    return np.ascontiguousarray(series.to_numpy(), dtype=dtype)


def _pointer(array: np.ndarray, ctype: Any) -> Any:
    return array.ctypes.data_as(ct.POINTER(ctype))


_EPOCH = date(1970, 1, 1)


def _iso(epoch_day: int) -> str:
    return str(_EPOCH + timedelta(days=int(epoch_day)))


def run_rust_kernel(frame: pl.DataFrame, config: Any) -> dict[str, Any]:
    eligible, reasons = rust_eligibility(config)
    if not eligible:
        raise ValueError("; ".join(reasons))
    assert _LIBRARY is not None
    symbol_lookup = sorted(str(value) for value in frame["ts_code"].unique().to_list())
    symbol_ids = {symbol: index for index, symbol in enumerate(symbol_lookup)}
    prepared = frame.with_columns(
        pl.col("trade_date").cast(pl.Int32).alias("_epoch_day"),
        pl.col("ts_code").replace_strict(symbol_ids).cast(pl.Int32).alias("_symbol_id"),
    )
    arrays = {
        "day": _contiguous(prepared, "_epoch_day", np.int32),
        "symbol": _contiguous(prepared, "_symbol_id", np.int32),
        "rank": _contiguous(prepared, "univ_rank", np.int32, fill=2**31 - 1),
        "factor": _contiguous(prepared, "factor", np.float64, fill=float("nan")),
        "open": _contiguous(prepared, "raw_open", np.float64, fill=float("nan")),
        "close": _contiguous(prepared, "raw_close", np.float64, fill=float("nan")),
        "volume": _contiguous(prepared, "vol", np.float64, fill=0.0),
        "adv": _contiguous(prepared, "_adv20_prev", np.float64, fill=float("nan")),
        "atr": _contiguous(prepared, "_atr_pct", np.float64, fill=float("nan")),
        "vol20": _contiguous(prepared, "_vol20_prev", np.float64, fill=float("nan")),
        "adjustment": _contiguous(prepared, "adjustment_factor", np.float64, fill=float("nan")),
        "can_buy": _contiguous(prepared, "can_buy_open_proxy", np.uint8, fill=False),
        "can_sell": _contiguous(prepared, "can_sell_open_proxy", np.uint8, fill=False),
    }
    boundaries = np.flatnonzero(np.diff(arrays["day"])) + 1
    offsets = np.ascontiguousarray(
        np.concatenate(([0], boundaries, [frame.height])), dtype=np.uintp
    )
    columns = _Columns(
        RUST_KERNEL_ABI, frame.height, len(offsets) - 1,
        _pointer(offsets, ct.c_size_t), _pointer(arrays["day"], ct.c_int32),
        _pointer(arrays["symbol"], ct.c_int32), _pointer(arrays["rank"], ct.c_int32),
        _pointer(arrays["factor"], ct.c_double), _pointer(arrays["open"], ct.c_double),
        _pointer(arrays["close"], ct.c_double), _pointer(arrays["volume"], ct.c_double),
        _pointer(arrays["adv"], ct.c_double), _pointer(arrays["atr"], ct.c_double),
        _pointer(arrays["vol20"], ct.c_double), _pointer(arrays["adjustment"], ct.c_double),
        _pointer(arrays["can_buy"], ct.c_uint8), _pointer(arrays["can_sell"], ct.c_uint8),
    )
    cfg = _Config(
        RUST_KERNEL_ABI, 2 if config.market == "ashare" else 1,
        2 if config.mode == "long_short" else 1, config.direction,
        1 if config.resolved_account_type == "cash" else 2,
        {"fixed": 1, "linear": 2, "square_root": 3}[config.impact_model],
        config.universe_n, config.rebalance_every, config.max_positions,
        config.initial_capital, config.top_fraction, config.slippage_bps,
        config.spread_bps, config.max_volume_participation,
        config.cash_buffer_fraction, config.max_gross_leverage,
        config.max_position_weight, config.min_trade_notional,
        config.rebalance_buffer_pct, config.long_gross_target,
        config.short_gross_target, config.borrow_cost_bps_annual,
        config.margin_interest_bps_annual,
        config.impact_coefficient_bps,
    )
    trade_buffer = (_Trade * max(1, frame.height))()
    daily_buffer = (_Daily * max(1, len(offsets) - 1))()
    summary = _Summary()
    started = time.perf_counter()
    status = _LIBRARY.ff_run_v1(
        ct.byref(columns), ct.byref(cfg), trade_buffer, len(trade_buffer),
        daily_buffer, len(daily_buffer), ct.byref(summary),
    )
    kernel_seconds = time.perf_counter() - started
    if status != 0:
        raise RuntimeError(f"Rust内核返回错误码 {status}")
    trades = []
    for row in trade_buffer[: summary.trade_count]:
        trades.append({
            "fill_id": f"FILL-{row.fill_seq:08d}",
            "order_id": f"ORD-{row.order_seq:08d}",
            "signal_date": _iso(row.signal_day), "trade_date": _iso(row.trade_day),
            "symbol": symbol_lookup[row.symbol], "side": "BUY" if row.side > 0 else "SELL",
            "requested_quantity": row.requested_quantity, "filled_quantity": row.filled_quantity,
            "reference_price": row.reference_price, "fill_price": row.fill_price,
            "commission": row.commission, "stamp_duty": row.stamp_duty,
            "transfer_fee": row.transfer_fee, "total_fees": row.total_fees,
            "cash_before": row.cash_before, "cash_after": row.cash_after,
            "position_before": row.position_before, "position_after": row.position_after,
            "participation": row.participation, "slippage_cost": row.slippage_cost,
        })
    daily = []
    for row in daily_buffer[: summary.daily_count]:
        daily.append({
            "trade_date": _iso(row.day), "close_nlv": row.close_nlv,
            "net_nav": row.net_nav, "daily_return": row.daily_return, "cash": row.cash,
            "long_market_value": row.long_market_value,
            "short_market_value": row.short_market_value,
            "gross_exposure": row.gross_exposure, "net_exposure": row.net_exposure,
            "turnover": row.turnover, "borrow_fee": row.borrow_fee,
            "margin_interest": row.margin_interest, "fills": row.fills,
            "orders_created": row.orders_created, "positions": row.positions,
        })
    return {
        "protocol": RUST_KERNEL_PROTOCOL,
        "kernel_seconds": kernel_seconds,
        "frame_conversion_seconds": None,
        "trades": trades,
        "daily_steps": daily,
        "summary": {
            "trade_count": summary.trade_count, "daily_count": summary.daily_count,
            "order_count": summary.order_count, "rejected_orders": summary.rejected_orders,
            "partial_orders": summary.partial_orders, "final_nlv": summary.final_nlv,
            "cumulative_fees": summary.cumulative_fees,
            "cumulative_slippage": summary.cumulative_slippage,
            "cumulative_borrow": summary.cumulative_borrow,
            "cumulative_margin_interest": summary.cumulative_margin_interest,
        },
        "config": asdict(config),
    }


def align_shadow_results(python_result: dict, rust_result: dict) -> dict[str, Any]:
    py_trades = python_result.get("trades", [])
    rs_trades = rust_result.get("trades", [])
    py_daily = python_result.get("daily_steps", [])
    rs_daily = rust_result.get("daily_steps", [])
    trade_identity_fields = ("trade_date", "symbol", "side")
    trade_numeric_fields = (
        "requested_quantity", "filled_quantity", "reference_price", "fill_price",
        "commission", "stamp_duty", "transfer_fee", "total_fees", "cash_before",
        "cash_after", "position_before", "position_after", "participation", "slippage_cost",
    )
    daily_numeric_fields = (
        "close_nlv", "net_nav", "daily_return", "cash", "long_market_value",
        "short_market_value", "gross_exposure", "net_exposure", "turnover",
        "borrow_fee", "margin_interest",
    )
    identity_mismatches = 0
    max_trade_error = 0.0
    first_trade_mismatch = None
    first_trade_mismatch_detail = None
    first_trade_numeric_mismatch = None
    for index, (left, right) in enumerate(zip(py_trades, rs_trades)):
        if any(left.get(field) != right.get(field) for field in trade_identity_fields):
            identity_mismatches += 1
            if first_trade_mismatch is None:
                first_trade_mismatch = index
                first_trade_mismatch_detail = {
                    "python": {field: left.get(field) for field in trade_identity_fields},
                    "rust": {field: right.get(field) for field in trade_identity_fields},
                }
        for field in trade_numeric_fields:
            field_error = abs(float(left.get(field) or 0.0) - float(right.get(field) or 0.0))
            max_trade_error = max(max_trade_error, field_error)
            if field_error > ALIGNMENT_ABSOLUTE_TOLERANCE and first_trade_numeric_mismatch is None:
                first_trade_numeric_mismatch = {
                    "index": index,
                    "field": field,
                    "python": left.get(field),
                    "rust": right.get(field),
                    "identity": {name: left.get(name) for name in trade_identity_fields},
                }
    max_daily_error = 0.0
    first_daily_mismatch = None
    for index, (left, right) in enumerate(zip(py_daily, rs_daily)):
        if left.get("trade_date") != right.get("trade_date"):
            first_daily_mismatch = first_daily_mismatch or index
        row_error = max(
            abs(float(left.get(field) or 0.0) - float(right.get(field) or 0.0))
            for field in daily_numeric_fields
        )
        max_daily_error = max(max_daily_error, row_error)
        if row_error > ALIGNMENT_ABSOLUTE_TOLERANCE and first_daily_mismatch is None:
            first_daily_mismatch = index
    counts_equal = len(py_trades) == len(rs_trades) and len(py_daily) == len(rs_daily)
    all_pass = (
        counts_equal and identity_mismatches == 0
        and max_trade_error <= ALIGNMENT_ABSOLUTE_TOLERANCE
        and max_daily_error <= ALIGNMENT_ABSOLUTE_TOLERANCE
    )
    return {
        "all_pass": all_pass,
        "absolute_tolerance": ALIGNMENT_ABSOLUTE_TOLERANCE,
        "python_trade_count": len(py_trades), "rust_trade_count": len(rs_trades),
        "python_daily_count": len(py_daily), "rust_daily_count": len(rs_daily),
        "trade_identity_mismatches": identity_mismatches,
        "max_trade_numeric_error": max_trade_error,
        "max_daily_numeric_error": max_daily_error,
        "first_trade_mismatch_index": first_trade_mismatch,
        "first_trade_mismatch_detail": first_trade_mismatch_detail,
        "first_trade_numeric_mismatch": first_trade_numeric_mismatch,
        "first_daily_mismatch_index": first_daily_mismatch,
        "promotion_allowed": all_pass,
    }
