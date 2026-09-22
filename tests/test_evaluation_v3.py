import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import (
    DIRECTION_POLICY_BOTH,
    DIRECTION_POLICY_FIXED,
    EVALUATION_PROTOCOL_VERSION,
    evaluation_config,
    get_dsl_fields,
)
from app.dsl.engine import parse, validate
from app.eval.harness import (
    FROZEN_RATING_LAYER,
    _prepare_daily,
    evaluate,
    evaluate_full,
    preflight_expression,
)
from app.miner.agent import _build_system_prompt
from app.config import DEFAULT_MINER_TEMPLATE


class _SyntheticPanel:
    def __init__(self, frame: pl.DataFrame):
        self.frame = frame

    def ensure_loaded(self) -> pl.DataFrame:
        return self.frame


def _panel() -> pl.DataFrame:
    rows = []
    layers = ["INNER_PUBLIC", "META_TRAIN", "META_HOLDOUT", "FACTOR_VAULT"]
    start = date(2020, 1, 1)
    for layer_index, layer in enumerate(layers):
        for day_index in range(160):
            trade_date = start + timedelta(days=layer_index * 180 + day_index)
            era = layer_index * 10 + day_index // 35
            for security in range(100):
                rank = security + 1
                signal = (rank - 50.5) / 50.0
                forward = (
                    signal * (0.008 + math.sin(day_index / 19.0) * 0.002)
                    + math.sin(security * 0.71 + day_index * 0.29) * 0.003
                )
                rows.append({
                    "trade_date": trade_date,
                    "ts_code": f"S{security:03d}",
                    "name": f"Stock {security}",
                    "open": 10.0 + rank / 10.0,
                    "high": 10.2 + rank / 10.0,
                    "low": 9.8 + rank / 10.0,
                    "close": float(rank),
                    "vol": 1_000_000.0,
                    "amount": 1_000_000_000.0,
                    "pb": 1.0 + rank / 100.0,
                    "fwd_1": forward,
                    "fwd_5": forward,
                    "fwd_10": forward,
                    "fwd_20": forward,
                    "univ_rank": rank,
                    "era": era,
                    "layer": layer,
                })
    return pl.DataFrame(rows)


def _membership_turnover_panel() -> pl.DataFrame:
    rows = []
    for day_index in range(2):
        trade_date = date(2020, 1, 2 + day_index)
        for security in range(120):
            if day_index == 0:
                eligible = security < 100
                univ_rank = security + 1 if eligible else 101 + security
                close = 200.0 - security
            else:
                eligible = security >= 20
                univ_rank = security - 19 if eligible else 101 + security
                close = float(security)
            rows.append({
                "trade_date": trade_date,
                "ts_code": f"S{security:03d}",
                "name": f"Stock {security}",
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "vol": 1_000_000.0,
                "amount": 1_000_000_000.0,
                "fwd_1": (security - 60) / 100_000.0,
                "fwd_5": (security - 60) / 100_000.0,
                "fwd_10": (security - 60) / 100_000.0,
                "fwd_20": (security - 60) / 100_000.0,
                "univ_rank": univ_rank,
                "era": 20201,
                "layer": "INNER_PUBLIC",
            })
    return pl.DataFrame(rows)


class EvaluationV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = _panel()

    def test_market_specific_dsl_fields(self):
        self.assertIsNone(validate("rank(pb)", get_dsl_fields("ashare")))
        self.assertIn("未知字段", validate("rank(pb)", get_dsl_fields("us")))

    def test_market_specific_protocol_defaults(self):
        ashare = evaluation_config("ashare")
        us = evaluation_config("us")
        self.assertEqual(ashare["base_cost_bps"], 20.0)
        self.assertEqual(us["borrow_cost_bps_annual"], 300.0)
        self.assertNotEqual(ashare["target_capital"], us["target_capital"])
        with self.assertRaises(ValueError):
            evaluation_config("ashare", {"stress_cost_bps": []})

    def test_discovery_never_returns_private_layers(self):
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(self.frame)):
            result = evaluate(
                "rank(close)",
                universe_n=100,
                horizon=5,
                portfolio_mode="long_only",
                direction=1,
                panel_glob="synthetic",
                cost_bps=20,
                market="ashare",
            )
        self.assertEqual(result["protocol_version"], EVALUATION_PROTOCOL_VERSION)
        self.assertEqual(result["scope"], "discovery")
        self.assertNotIn("holdout", result)
        self.assertNotIn("vault", result)
        self.assertIn("active", result["public"])
        self.assertIn("cost_stress", result["gate"])

    def test_constant_cross_section_is_rejected_before_tie_order_can_trade(self):
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(self.frame)):
            with self.assertRaisesRegex(ValueError, "有效评估样本为空"):
                evaluate(
                    "1",
                    universe_n=100,
                    horizon=5,
                    portfolio_mode="long_only",
                    direction=1,
                    panel_glob="synthetic",
                    cost_bps=20,
                    market="ashare",
                )

    def test_cheap_preflight_rejects_constant_and_accepts_varying_signal(self):
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(self.frame)):
            constant = preflight_expression(
                "1", 100, 5, "synthetic", "ashare", sample_modulus=1
            )
            varying = preflight_expression(
                "rank(close)", 100, 5, "synthetic", "ashare", sample_modulus=1
            )
        self.assertFalse(constant["accepted"])
        self.assertEqual(constant["usable_dates"], 0)
        self.assertTrue(varying["accepted"])
        self.assertGreater(varying["finite_coverage"], 0.9)
        self.assertFalse(varying["evaluation_performed"])

    def test_full_audit_has_four_isolation_layers_plus_2020_latest_rating(self):
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(self.frame)):
            result = evaluate_full(
                "rank(close)",
                universe_n=100,
                horizon=5,
                portfolio_mode="long_only",
                direction=1,
                panel_glob="synthetic",
                cost_bps=20,
                market="ashare",
                evaluation_overrides={"target_capital": 100_000.0},
            )
        self.assertEqual(
            set(result["layers"]),
            {"public", "gate", "holdout", "vault", "rating"},
        )
        self.assertTrue(result["eligibility"]["research_pass"])
        self.assertTrue(result["eligibility"]["holdout_pass"])
        self.assertTrue(result["eligibility"]["vault_pass"])
        self.assertEqual(result["eligibility"]["grade"], "F4")
        self.assertFalse(result["eligibility"]["eligible"])
        self.assertFalse(result["eligibility"]["snapshot_pass"])
        self.assertFalse(result["eligibility"]["overfit_pass"])
        self.assertFalse(result["eligibility"]["production_approved"])
        self.assertTrue(result["ranking"]["available"])
        self.assertIsNotNone(result["ranking"]["score"])
        self.assertIn("return_confidence", result["holdout"])
        self.assertGreater(result["holdout"]["cost_cushion_multiple"], 1.0)
        self.assertEqual(result["rating"]["window_start"], "2020-01-01")
        self.assertEqual(
            result["rating"]["window_end"],
            str(self.frame["trade_date"].max()),
        )
        self.assertEqual(
            result["ranking"]["rating_window"]["end"],
            str(self.frame["trade_date"].max()),
        )
        self.assertFalse(
            result["ranking"]["rating_window"]["independent_out_of_sample"]
        )
        self.assertIsNone(result["ranking"]["score_pre_vault"])
        self.assertIsNotNone(result["ranking"]["score_frozen_rating"])

    def test_long_short_reports_separate_legs_and_borrow(self):
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(self.frame)):
            result = evaluate(
                "rank(close)",
                universe_n=100,
                horizon=5,
                portfolio_mode="long_short",
                direction=1,
                panel_glob="synthetic",
                cost_bps=15,
                market="us",
            )
        self.assertIsNotNone(result["public"]["short_leg"])
        self.assertEqual(result["public"]["borrow_cost_bps_annual"], 300.0)
        self.assertIn("net", result["public"])

    def test_each_new_candidate_tests_both_signs_and_selects_reverse(self):
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(self.frame)):
            result = evaluate(
                "-rank(close)",
                universe_n=100,
                horizon=5,
                portfolio_mode="long_only",
                direction=1,
                panel_glob="synthetic",
                cost_bps=20,
                market="ashare",
            )
        selection = result["discovery"]["direction_selection"]
        self.assertEqual(result["direction_policy"], DIRECTION_POLICY_BOTH)
        self.assertEqual(result["direction"], -1)
        self.assertEqual(result["public"]["direction"], -1)
        self.assertEqual(set(selection["candidates"]), {"+1", "-1"})
        self.assertGreater(
            selection["candidates"]["-1"]["learning_score"],
            selection["candidates"]["+1"]["learning_score"],
        )
        self.assertEqual(selection["trials_multiplier"], 2)
        self.assertEqual(
            selection["effective_multiple_testing_trials"],
            2 * evaluation_config("ashare")["multiple_testing_trials"],
        )

    def test_both_directions_share_one_factor_plan(self):
        with (
            patch(
                "app.eval.harness.PanelStore.get",
                return_value=_SyntheticPanel(self.frame),
            ),
            patch("app.eval.harness.parse", wraps=parse) as parser,
        ):
            result = evaluate(
                "rank(close)",
                universe_n=100,
                horizon=5,
                portfolio_mode="long_only",
                direction=1,
                panel_glob="synthetic",
                cost_bps=20,
                market="ashare",
            )
        self.assertEqual(parser.call_count, 1)
        self.assertEqual(result["runtime"]["direction_count"], 2)
        self.assertEqual(result["runtime"]["factor_materializations"], 1)
        self.assertEqual(
            result["runtime"]["execution"],
            "polars_native_shared_subplan",
        )

    def test_full_audit_freezes_direction_before_validation_layers(self):
        reversed_validation = self.frame.with_columns(
            pl.when(
                pl.col("layer").is_in(["META_HOLDOUT", "FACTOR_VAULT"])
            )
            .then(-pl.col("fwd_5"))
            .otherwise(pl.col("fwd_5"))
            .alias("fwd_5")
        )
        with patch(
            "app.eval.harness.PanelStore.get",
            return_value=_SyntheticPanel(reversed_validation),
        ):
            result = evaluate_full(
                "rank(close)",
                universe_n=100,
                horizon=5,
                portfolio_mode="long_only",
                direction=1,
                panel_glob="synthetic",
                cost_bps=20,
                market="ashare",
            )
        self.assertEqual(result["direction"], 1)
        self.assertEqual(
            result["discovery"]["direction_selection"]["selection_scope"],
            "training_safe_discovery_only",
        )
        self.assertLess(result["holdout"]["icir"], 0)

    def test_fixed_direction_remains_available_for_reaudit(self):
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(self.frame)):
            result = evaluate(
                "-rank(close)",
                universe_n=100,
                horizon=5,
                portfolio_mode="long_only",
                direction=1,
                panel_glob="synthetic",
                cost_bps=20,
                market="ashare",
                direction_policy=DIRECTION_POLICY_FIXED,
            )
        self.assertEqual(result["direction"], 1)
        self.assertEqual(
            set(result["discovery"]["direction_selection"]["candidates"]),
            {"+1"},
        )

    def test_turnover_counts_positions_that_leave_the_universe(self):
        frame = _membership_turnover_panel()
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(frame)):
            daily, _ = _prepare_daily(
                "close",
                universe_n=100,
                horizon=1,
                portfolio_mode="long_only",
                direction=1,
                panel_glob="synthetic",
                market="ashare",
                layers=["INNER_PUBLIC"],
                cfg=evaluation_config("ashare", {"target_capital": 100_000.0}),
            )
        self.assertEqual(daily.height, 2)
        self.assertAlmostEqual(float(daily["turnover"][0]), 1.0, places=6)
        self.assertAlmostEqual(float(daily["turnover"][1]), 2.0, places=6)

    def test_rating_date_window_includes_latest_rows_outside_declared_layers(self):
        latest = self.frame["trade_date"].max()
        frame = self.frame.with_columns(
            pl.when(pl.col("trade_date") == latest)
            .then(pl.lit("NONE"))
            .otherwise(pl.col("layer"))
            .alias("layer")
        )
        with patch("app.eval.harness.PanelStore.get", return_value=_SyntheticPanel(frame)):
            daily, _ = _prepare_daily(
                "rank(close)",
                universe_n=100,
                horizon=1,
                portfolio_mode="long_only",
                direction=1,
                panel_glob="synthetic",
                market="ashare",
                layers=[],
                cfg=evaluation_config("ashare"),
                date_window=("2020-01-01", None),
                layer_name_override=FROZEN_RATING_LAYER,
            )
        self.assertEqual(daily["trade_date"].max(), latest)
        self.assertEqual(daily["layer"].unique().to_list(), [FROZEN_RATING_LAYER])

    def test_llm_prompt_receives_long_only_and_dual_direction_policy(self):
        prompt = _build_system_prompt(
            DEFAULT_MINER_TEMPLATE,
            fields=["close"],
            portfolio_mode="long_only",
            market="ashare",
            direction=-1,
        )
        self.assertIn("持仓模式: long_only", prompt)
        self.assertIn("同时评价 +1", prompt)
        self.assertIn("-1（低值偏多）", prompt)
        self.assertIn("完全同分时优先 -1", prompt)
        self.assertIn("不建立空头", prompt)

        fixed = _build_system_prompt(
            DEFAULT_MINER_TEMPLATE,
            fields=["close"],
            portfolio_mode="long_only",
            market="ashare",
            direction=-1,
            direction_policy=DIRECTION_POLICY_FIXED,
        )
        self.assertIn("方向冻结为 -1", fixed)


if __name__ == "__main__":
    unittest.main()
