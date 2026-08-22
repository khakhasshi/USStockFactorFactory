from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from backend.app.dsl.engine import MAX_EXPRESSION_LENGTH, validate
from backend.app.factor_tools import (
    _analyse_materialized_frame,
    build_combination_expression,
    factor_tool_capabilities,
    validate_correlation_spec,
)


ASHARE_FIELDS = factor_tool_capabilities("ashare")["dsl_fields"]


def _components():
    return [
        {"key": "flow", "expression": "rank(ts_mean(net_mf_amount,20))", "direction": 1, "weight": 2},
        {"key": "value", "expression": "rank(-ps_ttm)", "direction": 1, "weight": 1},
        {"key": "reversal", "expression": "rank(returns(close,20))", "direction": -1, "weight": 1},
    ]


def test_expression_builder_embeds_direction_weights_and_market_whitelist():
    result = build_combination_expression({
        "market": "ashare",
        "components": _components(),
        "normalization": "rank",
    })
    assert result["protocol"] == "multi_factor_expression_builder_v1"
    assert result["direction"] == 1
    assert result["weights"] == [0.5, 0.25, 0.25]
    assert "-" in result["expression"]
    assert "+-" not in result["expression"]
    assert validate(result["expression"], ASHARE_FIELDS) is None
    assert result["component_snapshot"][1]["expression"] == "rank(-ps_ttm)"


def test_equal_weight_builder_omits_only_a_common_rank_invariant_scale():
    components = [{**row, "weight": 1} for row in _components()]
    result = build_combination_expression({
        "market": "ashare",
        "components": components,
        "normalization": "zscore",
        "omit_common_scale": True,
    })
    assert result["equal_weight"] is True
    assert "省略共同系数" in result["scale_note"]
    assert "/3" not in result["expression"]
    assert "+-" not in result["expression"]
    assert validate(result["expression"], ASHARE_FIELDS) is None


def test_dsl_accepts_audited_composites_beyond_legacy_500_character_limit():
    expression = "+".join(["rank(ts_mean(close,20))"] * 24)
    assert 500 < len(expression) < MAX_EXPRESSION_LENGTH
    assert validate(expression, ASHARE_FIELDS) is None
    assert "最大" in str(validate("rank(close)+" * 1000, ASHARE_FIELDS))


def test_correlation_spec_freezes_common_execution_semantics():
    spec = validate_correlation_spec({
        "market": "ashare",
        "components": _components(),
        "start": "2020-01-01",
        "end": "2024-12-31",
        "horizon": 5,
        "universe_n": 500,
        "cost_bps": 20,
    })
    assert spec["portfolio_mode"] == "long_only"
    assert spec["components"][2]["direction"] == -1
    assert len(spec["request_hash"]) == 32


def test_materialized_correlation_reports_signal_ic_and_return_path_matrices():
    rng = np.random.default_rng(20260822)
    rows = []
    for day_index in range(12):
        signal = rng.normal(size=120)
        returns = 0.02 * signal + rng.normal(scale=0.01, size=120)
        ranks = np.argsort(np.argsort(signal)).astype(float) / 120.0
        for symbol_index in range(120):
            rows.append({
                "trade_date": date(2020, 1, 1) + timedelta(days=day_index),
                "ts_code": f"S{symbol_index:04d}",
                "univ_rank": symbol_index + 1,
                "fwd_1": float(returns[symbol_index]),
                "component_0": float(ranks[symbol_index]),
                "component_1": float(ranks[symbol_index]),
                "component_2": float(1.0 - ranks[symbol_index]),
            })
    frame = pl.DataFrame(rows)
    spec = {
        "components": [
            {"key": "same_a", "name": "same_a", "direction": 1},
            {"key": "same_b", "name": "same_b", "direction": 1},
            {"key": "inverse", "name": "inverse", "direction": -1},
        ],
        "horizon": 1,
        "top_fraction": 0.20,
        "portfolio_mode": "long_only",
        "cost_bps": 0.0,
        "borrow_cost_bps_annual": 0.0,
    }
    result = _analyse_materialized_frame(frame, spec)
    assert result["signal_rank_correlation"][0][1] > 0.999
    assert result["signal_rank_correlation"][0][2] < -0.999
    assert result["portfolio_return_correlation"][0][1] > 0.999
    assert len(result["factor_stats"]) == 3


def test_frontend_exposes_factor_tools_and_direct_backtest_handoff():
    source = (Path(__file__).parents[1] / "frontend" / "app.js").read_text(encoding="utf-8")
    assert '{ id: "tools", label: "因子工具" }' in source
    assert "因子相关性与组合表达式工具" in source
    assert "/factor-tools/correlation" in source
    assert "/factor-tools/build-expression" in source
    assert "送入回测模块" in source
