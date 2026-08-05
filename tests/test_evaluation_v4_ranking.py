import unittest

from backend.app.config import (
    EVALUATION_PROTOCOL_VERSION,
    evaluation_config,
    resolve_engine_tasks,
)
from backend.app.eval.harness import _discovery_score
from backend.app.eval.ranking import build_live_ranking, ranking_diagnostics
from backend.app.models import Factor


def _layer(
    *,
    sharpe: float,
    ann_return: float,
    sharpe_lcb: float,
    ann_return_lcb: float,
    return_t: float,
    icir: float = 1.0,
    stress_sharpe: float = 0.6,
    cost_cushion: float = 3.0,
) -> dict:
    branch = {
        "sharpe": sharpe,
        "ann_return": ann_return,
        "max_drawdown": 0.12,
        "calmar": ann_return / 0.12,
    }
    return {
        "available": True,
        "ic_mean": 0.03,
        "icir": icir,
        "monotonicity": 0.70,
        "era_consistency": 0.80,
        "profitable_era_rate": 0.80,
        "profitable_year_rate": 0.75,
        "worst_era_sharpe": 0.20,
        "daily_turnover": 0.04,
        "adv_participation_p95": 0.01,
        "cost_breakeven_bps": 45.0,
        "cost_cushion_multiple": cost_cushion,
        "active": branch,
        "net": branch,
        "return_confidence": {
            "sharpe_lcb": sharpe_lcb,
            "ann_return_lcb": ann_return_lcb,
            "hac_t_stat": return_t,
            "probabilistic_sharpe_gt_zero": 0.98,
        },
        "absolute_return_confidence": {
            "sharpe_lcb": sharpe_lcb,
            "ann_return_lcb": ann_return_lcb,
        },
        "cost_stress": [{"cost_bps": 40.0, "sharpe": stress_sharpe}],
    }


def _eligibility() -> dict:
    return {
        "research_pass": True,
        "holdout_pass": True,
        "vault_pass": True,
        "capacity_pass": True,
        "failure_reasons": [],
    }


class EvaluationV4RankingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = evaluation_config(
            "us",
            {
                "multiple_testing_trials": 1000,
                "target_rank_ann_return": 0.12,
            },
        )

    def test_fee_after_profitability_dominates_high_ic_point_estimate(self):
        strong = {
            "public": _layer(
                sharpe=1.6,
                ann_return=0.16,
                sharpe_lcb=1.1,
                ann_return_lcb=0.10,
                return_t=5.0,
                icir=0.8,
            ),
            "gate": _layer(
                sharpe=1.4,
                ann_return=0.14,
                sharpe_lcb=0.9,
                ann_return_lcb=0.08,
                return_t=4.5,
                icir=0.8,
            ),
            "holdout": _layer(
                sharpe=1.3,
                ann_return=0.13,
                sharpe_lcb=0.8,
                ann_return_lcb=0.07,
                return_t=4.2,
                icir=0.8,
            ),
            "vault": _layer(
                sharpe=0.5,
                ann_return=0.05,
                sharpe_lcb=0.1,
                ann_return_lcb=0.01,
                return_t=1.5,
            ),
        }
        weak = {
            "public": _layer(
                sharpe=0.5,
                ann_return=0.04,
                sharpe_lcb=0.15,
                ann_return_lcb=0.01,
                return_t=2.0,
                icir=3.0,
                stress_sharpe=-0.2,
                cost_cushion=1.0,
            ),
            "gate": _layer(
                sharpe=0.4,
                ann_return=0.03,
                sharpe_lcb=0.10,
                ann_return_lcb=0.005,
                return_t=1.8,
                icir=3.0,
                stress_sharpe=-0.3,
                cost_cushion=0.8,
            ),
            "holdout": _layer(
                sharpe=0.3,
                ann_return=0.02,
                sharpe_lcb=0.05,
                ann_return_lcb=0.002,
                return_t=1.4,
                icir=3.0,
                stress_sharpe=-0.5,
                cost_cushion=0.6,
            ),
            "vault": _layer(
                sharpe=2.0,
                ann_return=0.20,
                sharpe_lcb=1.0,
                ann_return_lcb=0.10,
                return_t=5.0,
            ),
        }
        strong_rank = build_live_ranking(strong, _eligibility(), "long_short", self.cfg)
        weak_rank = build_live_ranking(weak, _eligibility(), "long_short", self.cfg)
        self.assertGreater(strong_rank["score"], weak_rank["score"])
        self.assertGreater(
            strong_rank["components"]["net_profitability_lcb"],
            weak_rank["components"]["net_profitability_lcb"],
        )

    def test_numeric_vault_outcome_cannot_change_pre_vault_score(self):
        layers = {
            "public": _layer(
                sharpe=1.0,
                ann_return=0.10,
                sharpe_lcb=0.6,
                ann_return_lcb=0.05,
                return_t=4.0,
            ),
            "gate": _layer(
                sharpe=0.9,
                ann_return=0.09,
                sharpe_lcb=0.5,
                ann_return_lcb=0.04,
                return_t=3.8,
            ),
            "holdout": _layer(
                sharpe=0.8,
                ann_return=0.08,
                sharpe_lcb=0.4,
                ann_return_lcb=0.03,
                return_t=3.5,
            ),
            "vault": _layer(
                sharpe=-3.0,
                ann_return=-0.40,
                sharpe_lcb=-4.0,
                ann_return_lcb=-0.50,
                return_t=-5.0,
            ),
        }
        first = build_live_ranking(layers, _eligibility(), "long_short", self.cfg)
        layers["vault"] = _layer(
            sharpe=5.0,
            ann_return=0.80,
            sharpe_lcb=4.0,
            ann_return_lcb=0.60,
            return_t=8.0,
        )
        second = build_live_ranking(layers, _eligibility(), "long_short", self.cfg)
        self.assertEqual(first["score_pre_vault"], second["score_pre_vault"])
        self.assertEqual(first["score"], second["score"])

    def test_ranking_diagnostics_detects_monotone_future_profit(self):
        rows = [
            {
                "factor_id": index,
                "score_pre_vault": float(index * 10),
                "vault_ann_return": -0.09 + index * 0.03,
                "vault_sharpe": -0.9 + index * 0.3,
            }
            for index in range(1, 9)
        ]
        result = ranking_diagnostics(rows)
        self.assertEqual(result["status"], "calibrated")
        self.assertAlmostEqual(
            result["metrics"]["spearman_score_vs_vault_return"],
            1.0,
        )
        self.assertGreater(
            result["metrics"]["top_quartile_positive_rate"],
            result["metrics"]["bottom_quartile_positive_rate"],
        )

    def test_ranking_diagnostics_fail_closed_on_small_sample(self):
        result = ranking_diagnostics(
            [{
                "factor_id": 1,
                "score_pre_vault": 50,
                "vault_ann_return": 0.1,
                "vault_sharpe": 1.0,
            }]
        )
        self.assertEqual(result["status"], "insufficient_sample")
        self.assertIsNone(result["metrics"])

    def test_reaudit_state_labels_fit_database_columns(self):
        lifecycle = "configuration_changed_requires_reaudit"
        provenance = "configuration_changed_requires_revalidation"
        self.assertGreaterEqual(
            Factor.__table__.c.lifecycle_stage.type.length,
            len(lifecycle),
        )
        self.assertGreaterEqual(
            Factor.__table__.c.provenance_status.type.length,
            len(provenance),
        )

    def test_task_costs_are_market_aware_and_local_overrides_win(self):
        shared = [
            {"name": "liquid", "universe_n": 500, "horizon": 5, "cost_bps": 15},
            {"name": "broad", "universe_n": 1500, "horizon": 10, "cost_bps": 25},
        ]
        ashare = resolve_engine_tasks(shared, "ashare", "long_only", 1)
        us = resolve_engine_tasks(shared, "us", "long_short", 1)
        self.assertEqual([row["cost_bps"] for row in ashare], [20.0, 30.0])
        self.assertEqual([row["cost_bps"] for row in us], [15.0, 25.0])
        self.assertTrue(all(row["mode"] == "long_only" for row in ashare))
        local = resolve_engine_tasks(
            [{"name": "custom", "universe_n": 500, "horizon": 5, "cost_bps": 37}],
            "ashare",
            "long_only",
            -1,
            preserve_declared_costs=True,
        )
        self.assertEqual(local[0]["cost_bps"], 37.0)
        self.assertEqual(local[0]["direction"], -1)
        self.assertEqual(
            local[0]["direction_policy"],
            "both_train_select",
        )

    def test_failed_discovery_scores_keep_a_continuous_learning_gradient(self):
        mild_public = _layer(
            sharpe=-0.25,
            ann_return=-0.02,
            sharpe_lcb=-0.30,
            ann_return_lcb=-0.02,
            return_t=-0.40,
            icir=-0.20,
            stress_sharpe=-0.40,
            cost_cushion=-0.50,
        )
        mild_gate = _layer(
            sharpe=-0.35,
            ann_return=-0.03,
            sharpe_lcb=-0.45,
            ann_return_lcb=-0.03,
            return_t=-0.60,
            icir=-0.30,
            stress_sharpe=-0.60,
            cost_cushion=-0.80,
        )
        severe_public = _layer(
            sharpe=-2.0,
            ann_return=-0.30,
            sharpe_lcb=-2.5,
            ann_return_lcb=-0.35,
            return_t=-4.0,
            icir=-1.8,
            stress_sharpe=-2.5,
            cost_cushion=-6.0,
        )
        severe_gate = _layer(
            sharpe=-2.5,
            ann_return=-0.40,
            sharpe_lcb=-3.0,
            ann_return_lcb=-0.45,
            return_t=-5.0,
            icir=-2.2,
            stress_sharpe=-3.0,
            cost_cushion=-8.0,
        )
        for layer in (
            mild_public,
            mild_gate,
            severe_public,
            severe_gate,
        ):
            layer["coverage"] = 0.95
            layer["hac_p_value"] = 0.9

        mild = _discovery_score(
            mild_public,
            mild_gate,
            "long_short",
            self.cfg,
        )
        severe = _discovery_score(
            severe_public,
            severe_gate,
            "long_short",
            self.cfg,
        )

        self.assertFalse(mild["passed"])
        self.assertFalse(severe["passed"])
        self.assertGreater(mild["score"], 0.0)
        self.assertGreater(severe["score"], 0.0)
        self.assertGreater(mild["score"], severe["score"])
        self.assertEqual(
            mild["score_semantics"],
            "continuous_failure_margin_v4.2",
        )
        self.assertIn("gate_score", mild)
        self.assertIn("gate_components", mild)

    def test_protocol_override_cannot_downgrade_current_evaluator(self):
        cfg = evaluation_config(
            "us",
            {"protocol_version": "v4.0"},
        )
        self.assertEqual(
            cfg["protocol_version"],
            EVALUATION_PROTOCOL_VERSION,
        )


if __name__ == "__main__":
    unittest.main()
