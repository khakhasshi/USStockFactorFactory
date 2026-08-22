import pytest

from backend.app.blind_review import build_review_packets, seal_reviews
from backend.app.document_dsl import document_to_dsl
from backend.app.mechanism_catalog import catalog


def test_catalog_has_exactly_33_base_mechanisms_without_parameter_grid():
    result = catalog()
    assert result["count"] == 33
    assert result["lens_count"] == 119
    assert all(row["factor_expression"] is None for row in result["lenses"])
    assert result["policy"] == "idea_map_only_no_score_bonus_no_bulk_grid"


def test_document_candidates_are_valid_and_never_receive_score_bonus():
    result = document_to_dsl(
        "研究假设：急跌放量后成交压力逐步衰减，可能代表边际卖压耗尽并发生短期反转。",
        market="us",
    )
    assert result["candidates"]
    assert all(row["dsl_valid"] for row in result["candidates"])
    assert all(row["score_bonus"] == 0 for row in result["candidates"])


def test_blind_review_rejects_cross_packet_or_stale_submission():
    packets = build_review_packets(expression="rank(returns(close, 20))", hypothesis="trend", market="us")
    with pytest.raises(ValueError, match="hash mismatch"):
        seal_reviews(
            packets=packets,
            method_review={"packet_hash": "wrong", "passed": True},
            code_review={"packet_hash": packets["code_packet"]["packet_hash"], "passed": True},
        )
