import pytest

from backend.app.factors.combination_candidates import (
    select_diverse_combination_candidates,
)


def _candidate(
    key: str,
    expression: str,
    mechanism: str,
    *,
    priority: int = 2,
    score: float = 1.0,
    rank: int = 1,
    vector: list[float] | None = None,
) -> dict:
    return {
        "component_key": key,
        "expression": expression,
        "mechanism_family": mechanism,
        "source_priority": priority,
        "quality_score": score,
        "source_rank": rank,
        "training_return_path_signature": (
            {"available": True, "vector": vector} if vector is not None else None
        ),
    }


def test_structural_duplicate_keeps_current_protocol_over_old_leaderboard():
    current = _candidate(
        "db:1",
        "rank(ts_mean(abs(ts_delta(close,1)/delay(close,1))/(amount+1e-9),120))",
        "volume_price_interaction",
        priority=2,
        score=1.5,
    )
    historical = _candidate(
        "leaderboard:2",
        "ts_mean(winsor(abs(ts_delta(close,1)/delay(close,1))/(amount+1e-9)),120)",
        "volume_price_interaction",
        priority=1,
        score=97.0,
    )
    size = _candidate("db:2", "rank(log(total_mv))", "size")

    result = select_diverse_combination_candidates([historical, current, size])

    assert {row["component_key"] for row in result["selected"]} == {
        "db:1",
        "db:2",
    }
    assert {
        (row["component_key"], row["reason"], row.get("kept_component_key"))
        for row in result["excluded"]
    } == {("leaderboard:2", "structural_duplicate", "db:1")}


def test_within_mechanism_training_return_path_correlation_is_rejected():
    vector = [-2.0, -1.0, 0.5, 1.5, 2.0]
    rows = [
        _candidate(
            "price:a",
            "rank(ts_mean(close,20))",
            "momentum",
            score=2.0,
            vector=vector,
        ),
        _candidate(
            "price:b",
            "rank(ts_delta(close,5))",
            "momentum",
            score=1.0,
            vector=[value * 3 for value in vector],
        ),
        _candidate("size", "rank(log(total_mv))", "size"),
    ]

    result = select_diverse_combination_candidates(
        rows,
        return_path_correlation_cap=0.80,
    )

    assert {row["component_key"] for row in result["selected"]} == {
        "price:a",
        "size",
    }
    rejected = next(
        row for row in result["excluded"] if row["component_key"] == "price:b"
    )
    assert rejected["reason"] == "training_return_path_too_correlated"
    assert rejected["correlation"] == pytest.approx(1.0)


def test_within_mechanism_negative_return_path_is_a_diversifier():
    vector = [-2.0, -1.0, 0.5, 1.5, 2.0]
    rows = [
        _candidate(
            "price:a",
            "rank(ts_mean(close,20))",
            "momentum",
            score=2.0,
            vector=vector,
        ),
        _candidate(
            "price:b",
            "rank(ts_delta(close,5))",
            "momentum",
            score=1.0,
            vector=[-value for value in vector],
        ),
        _candidate("size", "rank(log(total_mv))", "size"),
    ]

    result = select_diverse_combination_candidates(
        rows,
        return_path_correlation_cap=0.80,
    )

    assert {row["component_key"] for row in result["selected"]} == {
        "price:a",
        "price:b",
        "size",
    }


def test_round_robin_preserves_mechanism_coverage_before_depth():
    rows = [
        _candidate(
            f"volume:{index}",
            f"rank(ts_mean(close*volume_ratio,{20 + index}))",
            "volume_price_interaction",
            score=10.0 - index,
        )
        for index in range(4)
    ]
    rows.extend([
        _candidate("size", "rank(log(total_mv))", "size", score=1.0),
        _candidate("vol", "rank(ts_std(close,40))", "volatility", score=1.0),
    ])

    result = select_diverse_combination_candidates(
        rows,
        max_per_mechanism=4,
        max_candidates=4,
    )

    mechanisms = [row["mechanism_family"] for row in result["selected"]]
    assert set(mechanisms) == {"volume_price_interaction", "size", "volatility"}
    assert mechanisms.count("volume_price_interaction") == 2


def test_candidate_validation_rejects_duplicate_keys():
    row = _candidate("same", "rank(close)", "momentum")
    with pytest.raises(ValueError, match="component_key 不得重复"):
        select_diverse_combination_candidates([row, dict(row)])
