import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import EVALUATION_PROTOCOL_VERSION, evaluation_config, get_dsl_fields
from app.dsl.engine import validate
from app.eval.harness import _prepare_daily, evaluate, evaluate_full
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

    def test_full_audit_has_four_layers_and_lifecycle(self):
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
        self.assertEqual(set(result["layers"]), {"public", "gate", "holdout", "vault"})
        self.assertTrue(result["eligibility"]["research_pass"])
        self.assertTrue(result["eligibility"]["holdout_pass"])
        self.assertTrue(result["eligibility"]["vault_pass"])
        self.assertEqual(result["eligibility"]["grade"], "F5")
        self.assertFalse(result["eligibility"]["production_approved"])
        self.assertTrue(result["ranking"]["available"])
        self.assertIsNotNone(result["ranking"]["score"])
        self.assertIn("return_confidence", result["holdout"])
        self.assertGreater(result["holdout"]["cost_cushion_multiple"], 1.0)

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

    def test_llm_prompt_receives_long_only_and_frozen_direction(self):
        prompt = _build_system_prompt(
            DEFAULT_MINER_TEMPLATE,
            fields=["close"],
            portfolio_mode="long_only",
            market="ashare",
            direction=-1,
        )
        self.assertIn("持仓模式: long_only", prompt)
        self.assertIn("方向冻结为 -1", prompt)
        self.assertIn("因子值越低", prompt)
        self.assertIn("不建立空头", prompt)


if __name__ == "__main__":
    unittest.main()
