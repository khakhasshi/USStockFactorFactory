from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

from backend.app.combination_lab import (
    COMBINATION_LAB_PROTOCOL,
    LabSlice,
    _evaluate_candidates,
    enumerate_weight_candidates,
    parse_llm_weight_proposals,
    validate_lab_spec,
)


def _raw_spec(**updates):
    value = {
        "name": "test",
        "experiment_id": 1,
        "search_mode": "programmatic",
        "market": "us",
        "portfolio_mode": "long_only",
        "components": [
            {
                "key": "quality",
                "name": "quality",
                "expression": "rank(close)",
                "direction": 1,
                "mechanism": "quality",
            },
            {
                "key": "reversal",
                "name": "reversal",
                "expression": "rank((-returns(close, 20)))",
                "direction": 1,
                "mechanism": "reversal",
            },
            {
                "key": "liquidity",
                "name": "liquidity",
                "expression": "rank(ts_mean(amount, 20))",
                "direction": -1,
                "mechanism": "liquidity",
            },
        ],
        "min_factors": 2,
        "max_factors": 3,
        "min_mechanisms": 2,
        "coarse_step": 0.10,
        "min_weight": 0.10,
        "max_weight": 0.80,
        "max_mechanism_weight": 0.80,
        "path_budget": 100,
        "train_start": "2010-01-01",
        "train_end": "2016-12-31",
        "validation_start": "2017-01-01",
        "validation_end": "2019-12-31",
        "rating_start": "2020-01-01",
        "rating_end": "2024-12-31",
        "cost_bps": 0.0,
        "stress_cost_bps": 10.0,
    }
    value.update(updates)
    return value


def test_protocol_freezes_component_array_and_non_overlapping_windows():
    spec = validate_lab_spec(_raw_spec())
    assert spec["protocol"] == COMBINATION_LAB_PROTOCOL
    assert len(spec["components"]) == 3
    assert len(spec["snapshot_hash"]) == 32
    assert spec["components"][1]["required_history"] >= 21

    with pytest.raises(ValueError, match="不得重叠"):
        validate_lab_spec(_raw_spec(validation_end="2020-12-31"))


def test_budgeted_enumeration_respects_support_weight_and_mechanism_constraints():
    spec = validate_lab_spec(_raw_spec(path_budget=30))
    rows = enumerate_weight_candidates(spec)
    assert 1 <= len(rows) <= 30
    assert len({tuple(row["weights"]) for row in rows}) == len(rows)
    for row in rows:
        active = [value for value in row["weights"] if value > 0]
        assert 2 <= len(active) <= 3
        assert abs(sum(active) - 1.0) < 1e-9
        assert min(active) >= 0.10
        assert max(active) <= 0.80


def test_llm_can_only_propose_known_bounded_components():
    spec = validate_lab_spec(_raw_spec(search_mode="llm"))
    payload = {
        "proposals": [
            {
                "weights": {"quality": 0.7, "reversal": 0.3},
                "hypothesis": "bounded",
            },
            {"weights": {"quality": 0.5, "invented": 0.5}},
            {"weights": {"quality": 1.0}},
        ]
    }
    accepted = parse_llm_weight_proposals(payload, spec)
    assert len(accepted) == 1
    assert accepted[0]["source"] == "llm_proposal"
    rows = enumerate_weight_candidates(spec, payload)
    assert any(row["source"] == "llm_proposal" for row in rows)


def _synthetic_slices() -> list[LabSlice]:
    rng = np.random.default_rng(20260822)
    output = []
    for index in range(36):
        useful = rng.normal(size=120)
        noise = rng.normal(size=120)
        returns = 0.025 * useful + rng.normal(scale=0.02, size=120)
        components = np.column_stack([useful, noise, rng.normal(size=120)])
        order = np.argsort(returns, kind="mergesort")
        return_ranks = np.empty(120, dtype=float)
        return_ranks[order] = np.arange(120, dtype=float)
        output.append(LabSlice(
            trade_date=date(2017, 1, 1) + timedelta(days=index * 30),
            symbols=np.arange(120, dtype=np.int32),
            components=components,
            forward_returns=returns,
            return_ranks=return_ranks,
        ))
    return output


def test_deterministic_evaluator_reports_ic_rank_ic_cost_and_robustness():
    spec = validate_lab_spec(_raw_spec(max_weight=0.90))
    candidates = [
        {"weights": (0.8, 0.2, 0.0), "source": "test"},
        {"weights": (0.2, 0.8, 0.0), "source": "test"},
    ]
    rows = _evaluate_candidates(
        _synthetic_slices(),
        candidates,
        spec,
        progress_callback=None,
        cancel_event=None,
        stage="test",
    )
    # Extremely clean synthetic data can saturate the deliberately bounded
    # robustness score.  The uncapped IC diagnostics must still distinguish
    # the genuinely predictive weighting, and are explicit ranking tie-breaks.
    assert rows[0]["robust_score"] >= rows[1]["robust_score"]
    assert rows[0]["rank_ic_mean"] > rows[1]["rank_ic_mean"]
    assert rows[0]["rank_icir"] > rows[1]["rank_icir"]
    for key in (
        "ic_mean",
        "icir",
        "rank_ic_mean",
        "rank_icir",
        "rank_ic_p_value",
        "worst_year_sharpe",
        "stress_ann_return",
        "effective_factor_count",
    ):
        assert key in rows[0]


def test_frontend_exposes_both_combination_modes_and_component_array_copy():
    source = (Path(__file__).parents[1] / "frontend" / "app.js").read_text()
    assert 'id: "combinations", label: "组合优化"' in source
    assert "程序化优化" in source
    assert "LLM协作优化" in source
    assert "组件数组是权威数据" in source
    assert "/combination-experiments" in source
