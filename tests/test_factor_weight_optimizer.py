from datetime import date, timedelta

import numpy as np
import polars as pl

from backend.app.factors.weight_optimizer import (
    COMBINATION_PROTOCOL,
    CrossSectionSlice,
    FactorComponent,
    WeightSearchConfig,
    build_slices,
    combination_promotion_gate,
    enumerate_coarse_weights,
    evaluate_weights,
    search_optimal_weights,
)


def test_build_slices_purges_forward_labels_at_layer_boundaries():
    rows = []
    for layer_index, layer in enumerate(("INNER_PUBLIC", "META_TRAIN")):
        layer_start = date(2010 + layer_index, 1, 1)
        for day_index in range(261):
            trade_date = layer_start + timedelta(days=day_index)
            for symbol in range(60):
                rows.append({
                    "trade_date": trade_date,
                    "layer": layer,
                    "era": 20101 + layer_index * 10,
                    "ts_code": f"S{symbol:03d}",
                    "univ_rank": symbol + 1,
                    "fwd_20": 0.001 * symbol,
                    "component_0": float(symbol),
                })
    slices, summary = build_slices(
        pl.DataFrame(rows),
        component_count=1,
        horizon=20,
        universe_n=500,
    )
    # 261 dates - 21 purged boundary dates = 240 usable dates; stepping by 20
    # leaves exactly twelve signal dates in each immutable layer.
    assert summary["INNER_PUBLIC"] == 12
    assert summary["META_TRAIN"] == 12
    assert len(slices) == 24


def test_non_database_component_requires_an_auditable_source_reference():
    component = FactorComponent(
        factor_id=None,
        name="historical",
        expression="rank(close)",
        direction=1,
        source="historical_leaderboard",
    )
    with np.testing.assert_raises_regex(ValueError, "source_ref"):
        component.validate()

    FactorComponent(
        factor_id=None,
        name="historical",
        expression="rank(close)",
        direction=1,
        source="historical_leaderboard",
        source_ref="report.json#overall_rank=1",
    ).validate()


def _synthetic_slices() -> list[CrossSectionSlice]:
    rng = np.random.default_rng(20260810)
    output = []
    start = date(2010, 1, 1)
    symbol_count = 100
    for layer_index, (layer, periods) in enumerate(
        (("INNER_PUBLIC", 36), ("META_TRAIN", 18))
    ):
        for index in range(periods):
            useful = rng.normal(size=symbol_count)
            noise = rng.normal(size=symbol_count)
            returns = 0.03 * useful + rng.normal(scale=0.02, size=symbol_count)
            components = np.column_stack([
                useful,
                noise,
                0.25 * useful + rng.normal(size=symbol_count),
            ])
            order = np.argsort(returns, kind="mergesort")
            ranks = np.empty(symbol_count, dtype=float)
            ranks[order] = np.arange(symbol_count, dtype=float)
            trade_date = start + timedelta(days=(layer_index * 1000 + index * 20))
            output.append(CrossSectionSlice(
                trade_date=trade_date,
                layer=layer,
                era=trade_date.year * 10 + (1 if trade_date.month <= 6 else 2),
                symbols=np.arange(symbol_count, dtype=np.int32),
                components=components,
                forward_returns=returns,
                return_ranks=ranks,
            ))
    return output


def test_complete_coarse_grid_respects_support_and_bounds():
    config = WeightSearchConfig(
        min_factors=2,
        max_factors=3,
        coarse_step=0.10,
        min_active_weight=0.10,
        max_weight=0.70,
    )
    weights = enumerate_coarse_weights(3, config)
    assert weights
    assert len(weights) == len(set(weights))
    for row in weights:
        active = [value for value in row if value > 0]
        assert 2 <= len(active) <= 3
        assert abs(sum(row) - 1.0) < 1e-12
        assert min(active) >= 0.10
        assert max(active) <= 0.70


def test_mechanism_family_cap_rejects_duplicate_family_concentration():
    config = WeightSearchConfig(
        min_factors=2,
        max_factors=3,
        coarse_step=0.10,
        min_active_weight=0.10,
        max_weight=0.70,
        max_mechanism_weight=0.60,
    )
    weights = enumerate_coarse_weights(
        3, config, factor_groups=("valuation", "valuation", "flow")
    )
    assert weights
    assert all(row[0] + row[1] <= 0.60 + 1e-12 for row in weights)


def test_evaluator_prefers_weight_on_predictive_component():
    slices = _synthetic_slices()
    config = WeightSearchConfig(
        min_factors=2,
        max_factors=3,
        coarse_step=0.10,
        refine_step=0.05,
        min_active_weight=0.10,
        max_weight=0.80,
        max_mechanism_weight=0.80,
        max_active_pair_similarity=1.0,
        cost_bps=0.0,
        stress_cost_bps=0.0,
    )
    results = evaluate_weights(
        slices,
        [(0.8, 0.2, 0.0), (0.2, 0.8, 0.0)],
        config,
    )
    assert results[0]["robust_score"] > results[1]["robust_score"]
    assert results[0]["worst_icir"] > results[1]["worst_icir"]
    for key in (
        "worst_time_block_sharpe",
        "worst_tail_sharpe",
        "worst_tail_monotonicity",
        "worst_ic_tail_conversion_rate",
        "return_source_independence",
    ):
        assert key in results[0]


def test_return_path_similarity_penalizes_related_components():
    slices = _synthetic_slices()
    config = WeightSearchConfig(
        min_factors=2,
        max_factors=3,
        max_weight=0.80,
        max_mechanism_weight=0.80,
        max_active_pair_similarity=1.0,
        cost_bps=0.0,
        stress_cost_bps=0.0,
    )
    related, independent = evaluate_weights(
        slices,
        [(0.5, 0.0, 0.5), (0.5, 0.5, 0.0)],
        config,
    )
    assert independent["return_source_independence"] > related["return_source_independence"]


def test_search_is_deterministic_and_uses_two_to_three_factors():
    slices = _synthetic_slices()
    config = WeightSearchConfig(
        min_factors=2,
        max_factors=3,
        coarse_step=0.20,
        refine_step=0.10,
        min_active_weight=0.10,
        max_weight=0.80,
        max_mechanism_weight=0.80,
        refine_starts=4,
        refine_iterations=4,
        cost_bps=0.0,
        stress_cost_bps=0.0,
    )
    first = search_optimal_weights(slices, 3, config)
    second = search_optimal_weights(slices, 3, config)
    assert first["best"] == second["best"]
    assert first["protocol"] == COMBINATION_PROTOCOL
    assert first["llm_used"] is False
    assert first["holdout_or_vault_read"] is False
    assert 2 <= first["best"]["active_factors"] <= 3
    similarity = np.asarray(first["return_path_similarity_matrix"])
    assert np.allclose(similarity, similarity.T)
    assert np.allclose(np.diag(similarity), 1.0)
    assert first["search_stability"]["near_optimal_candidates"] >= 1


def test_promotion_gate_blocks_correlated_high_drawdown_winner():
    result = {
        "best": {
            "weights": [0.5, 0.5],
            "return_source_independence": 0.40,
            "worst_tail_sharpe": 1.0,
        },
        "return_path_similarity_matrix": [
            [1.0, 0.92],
            [0.92, 1.0],
        ],
        "event_verification": {
            "scenarios": {
                layer: {
                    "15": {
                        "sharpe": 0.70,
                        "max_drawdown": 0.40,
                        "integrity": {"all_pass": True},
                    }
                }
                for layer in ("INNER_PUBLIC", "META_TRAIN")
            }
        },
    }
    gate = combination_promotion_gate(result)
    assert gate["decision"] == "RESEARCH_ONLY_BLOCKED"
    assert gate["production_eligible"] is False
    assert set(gate["failed_rules"]) == {
        "event_drawdown",
        "active_pairwise_return_path_similarity",
    }
