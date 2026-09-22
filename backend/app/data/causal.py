"""Shared label-boundary rules for research learning branches."""

from datetime import date

import numpy as np
import polars as pl

from ..config import get_layer_bounds


def purged_layer_predicate(panel, market, horizon, layers):
    exit_column = f"label_exit_date_{horizon}"
    if exit_column not in panel.columns:
        raise ValueError("Exact market-calendar label dates required for causal research")
    calendar = panel.select("trade_date", "layer").unique().sort("trade_date")
    bounds = get_layer_bounds(market)
    safe = pl.lit(False)
    for layer in layers:
        days = calendar.filter(pl.col("layer") == layer)["trade_date"].to_list()
        embargo = 0 if layer == "INNER_PUBLIC" else int(horizon)
        if len(days) > embargo:
            safe = safe | ((pl.col("layer") == layer)
                & (pl.col("trade_date") >= days[embargo])
                & (pl.col(exit_column) <= date.fromisoformat(bounds[layer][1])))
    return safe


def purged_fold_masks(dates, test_dates, *, label_exit_dates=None, embargo_sessions=0):
    """Never fit on a label unavailable before the fold's first signal."""
    dates = np.asarray(dates)
    test_dates = np.asarray(test_dates)
    first = test_dates[0]
    train = dates < first
    if label_exit_dates is not None:
        exits = np.asarray(label_exit_dates)
        if len(exits) != len(dates):
            raise ValueError("label_exit_dates must match observation rows")
        train &= exits < first
    allowed_test = test_dates[max(0, int(embargo_sessions)):]
    return train, np.isin(dates, allowed_test)
