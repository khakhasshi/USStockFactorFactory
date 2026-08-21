from datetime import date, timedelta

from backend.scripts.validate_ashare_factor_pair import _period_stats


def test_period_stats_are_compounded_and_window_scoped():
    start = date(2023, 1, 2)
    rows = [
        {
            "trade_date": str(start + timedelta(days=index)),
            "daily_return": value,
            "turnover": 0.1,
            "fills": index + 1,
        }
        for index, value in enumerate((0.10, -0.05, 0.02))
    ]
    result = _period_stats(
        rows,
        start=start,
        end=start + timedelta(days=1),
    )
    assert result["sessions"] == 2
    assert result["total_return"] == 0.045
    assert result["mean_daily_turnover"] == 0.1
    assert result["fills"] == 3
