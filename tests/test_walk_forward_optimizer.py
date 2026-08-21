from datetime import date

import numpy as np

from backend.app.factors.walk_forward_optimizer import (
    DEFAULT_FOLDS,
    WALK_FORWARD_PROTOCOL,
    WalkForwardConfig,
    search_walk_forward_weights,
)
from backend.app.factors.weight_optimizer import CrossSectionSlice, WeightSearchConfig


def _walk_forward_slices() -> list[CrossSectionSlice]:
    rng = np.random.default_rng(20260810)
    output = []
    symbols = np.arange(120, dtype=np.int32)
    for year in range(2010, 2023):
        for month in range(1, 13):
            primary = rng.normal(size=len(symbols))
            secondary = rng.normal(size=len(symbols))
            defensive = rng.normal(size=len(symbols))
            market = 0.012 if year not in {2011, 2018, 2022} else -0.012
            returns = (
                market
                + 0.025 * primary
                + 0.012 * secondary
                + 0.008 * defensive
                + rng.normal(scale=0.025, size=len(symbols))
            )
            components = np.column_stack([primary, secondary, defensive])
            order = np.argsort(returns, kind="mergesort")
            return_ranks = np.empty(len(symbols), dtype=float)
            return_ranks[order] = np.arange(len(symbols), dtype=float)
            output.append(CrossSectionSlice(
                trade_date=date(year, month, 15),
                layer="INNER_PUBLIC" if year <= 2019 else "META_TRAIN",
                era=year * 10 + (1 if month <= 6 else 2),
                symbols=symbols,
                components=components,
                forward_returns=returns,
                return_ranks=return_ranks,
            ))
    return output


def test_walk_forward_search_is_deterministic_and_never_reads_private_layers():
    slices = _walk_forward_slices()
    config = WeightSearchConfig(
        min_factors=3,
        max_factors=3,
        coarse_step=0.20,
        refine_step=0.10,
        min_active_weight=0.10,
        max_weight=0.60,
        max_mechanism_weight=0.60,
        max_active_pair_similarity=1.0,
        refine_starts=4,
        refine_iterations=3,
        horizon=20,
        cost_bps=5.0,
        stress_cost_bps=15.0,
    )
    kwargs = {
        "factor_groups": ("volume_price_interaction", "size", "volatility"),
        "folds": DEFAULT_FOLDS,
        "walk_forward": WalkForwardConfig(min_mechanism_groups=3),
    }

    first = search_walk_forward_weights(slices, 3, config, **kwargs)
    second = search_walk_forward_weights(slices, 3, config, **kwargs)

    assert first["best"] == second["best"]
    assert first["protocol"] == WALK_FORWARD_PROTOCOL
    assert first["holdout_or_vault_read"] is False
    assert first["llm_used"] is False
    assert first["best"]["active_factors"] == 3
    assert first["best"]["active_mechanism_groups"] == 3
    assert set(first["best"]["folds"]) == {"WF1", "WF2", "WF3", "WF4"}
    for fold in first["best"]["folds"].values():
        assert fold["purged_tail_slices"] == 1
        assert fold["train_slices"] >= 8
        assert fold["validation_slices"] >= 8
        assert "active_sharpe" in fold["validation"]
        assert "long_sharpe" in fold["validation"]


def test_walk_forward_requires_requested_mechanism_coverage():
    slices = _walk_forward_slices()
    config = WeightSearchConfig(
        min_factors=2,
        max_factors=3,
        coarse_step=0.20,
        refine_step=0.10,
        min_active_weight=0.10,
        max_weight=0.80,
        max_mechanism_weight=1.0,
        max_active_pair_similarity=1.0,
    )

    result = search_walk_forward_weights(
        slices,
        3,
        config,
        factor_groups=("price", "price", "volatility"),
        walk_forward=WalkForwardConfig(min_mechanism_groups=2),
    )

    assert result["coarse_candidates_after_mechanism_filter"] < result[
        "coarse_candidates_before_filters"
    ]
    assert result["best"]["active_mechanism_groups"] == 2
