import pytest

from backend.app.api.routes import _resolve_manual_backtest_execution


def test_us_manual_backtest_can_override_task_to_long_only():
    mode, borrow_cost = _resolve_manual_backtest_execution(
        "long_only", "long_short", "us", 350.0, 125.0
    )

    assert mode == "long_only"
    assert borrow_cost == 0.0


def test_us_manual_backtest_long_short_uses_explicit_borrow_cost():
    mode, borrow_cost = _resolve_manual_backtest_execution(
        "long_short", "long_only", "us", 240.0, 125.0
    )

    assert mode == "long_short"
    assert borrow_cost == 240.0


def test_ashare_manual_backtest_rejects_long_short():
    with pytest.raises(ValueError, match="A 股手动回测仅支持 long_only"):
        _resolve_manual_backtest_execution(
            "long_short", "long_only", "ashare", None, 0.0
        )


def test_manual_backtest_rejects_unknown_mode():
    with pytest.raises(ValueError, match="long_only 或 long_short"):
        _resolve_manual_backtest_execution("market_neutral", "long_only", "us", None, 0.0)
