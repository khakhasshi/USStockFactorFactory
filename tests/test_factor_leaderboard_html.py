import hashlib
import importlib.util
import json
import re
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "backend/scripts/render_factor_leaderboard_html.py"
)
SPEC = importlib.util.spec_from_file_location(
    "render_factor_leaderboard_html",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class FactorLeaderboardHtmlTests(unittest.TestCase):
    def test_us_presentation_exposes_long_short_cost_boundary(self):
        presentation = MODULE._presentation(
            {
                "market": "us",
                "mode": "long_short",
                "initial_capital": 1_000_000,
                "borrow_cost_bps_annual": 300,
            },
            {"target_market": "us"},
        )
        self.assertIn("美股", presentation["title"])
        self.assertIn("多空", presentation["heading_html"])
        self.assertIn("IBKR Pro", presentation["fee_summary"])
        self.assertIn("3.00%", presentation["fee_summary"])
        self.assertEqual(presentation["currency_symbol"], "$")

    def test_us_long_only_presentation_exposes_beta_and_no_borrow(self):
        presentation = MODULE._presentation(
            {
                "market": "us",
                "mode": "long_only",
                "initial_capital": 1_000_000,
                "borrow_cost_bps_annual": 0,
            },
            {"target_market": "us"},
        )
        self.assertIn("美股", presentation["title"])
        self.assertIn("纯多头", presentation["heading_html"])
        self.assertIn("IBKR Pro", presentation["fee_summary"])
        self.assertIn("无借券费", presentation["fee_summary"])
        self.assertIn("市场 Beta", presentation["boundary"])
        self.assertNotIn("多空", presentation["title"])
        self.assertEqual(presentation["currency_symbol"], "$")

    def test_vector_presentation_discloses_screening_boundary(self):
        presentation = MODULE._presentation(
            {
                "market": "ashare",
                "mode": "long_only",
                "initial_capital": 10_000_000,
            },
            {"target_market": "ashare"},
            protocol_id=(
                "dual_direction_full_window_vector_screen_leaderboard_v1"
            ),
            scenario_engine="vector_screen",
        )
        self.assertTrue(presentation["screening_only"])
        self.assertFalse(presentation["has_vault"])
        self.assertIn("向量筛选", presentation["title"])
        self.assertIn("不是独立样本外验证", presentation["boundary"])
        self.assertIn("事件复核", presentation["validation_label"])

    def test_render_accepts_vector_ledger_freeze_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            winner = {
                "overall_rank": 1,
                "status": "ok",
                "economic_representative": True,
                "economic_duplicate_of": "",
                "equivalence_group_size": 1,
                "expression_hash": "winner-plus",
                "expression": "rank(close)",
                "origin_scope": "ashare",
                "direction": 1,
                "practical_pass": True,
                "robust_score": 90.0,
                "ann_return_bps_0": 0.12,
                "ann_return_bps_5": 0.10,
                "ann_return_bps_15": 0.08,
                "sharpe_bps_0": 1.0,
                "sharpe_bps_5": 0.8,
                "sharpe_bps_15": 0.6,
                "oos_ic_mean": 0.03,
                "oos_icir": 1.2,
                "oos_rank_ic_mean": 0.04,
                "oos_rank_icir": 1.4,
                "fields": ["close"],
                "operators": ["rank"],
                "experiment_ids": [1],
            }
            files = {
                "leaderboard_full.json": [winner],
                "snapshot.json": {
                    "snapshot_at_asia_shanghai": "2026-08-07 00:00:00",
                    "cutoffs": {
                        "max_experiment_id": 1,
                        "max_node_id": 1,
                        "max_factor_id": 1,
                    },
                    "source_rows": 1,
                    "unique_expressions": 1,
                    "valid_target_expressions": 1,
                    "invalid_target_expressions": 0,
                    "invalid": [],
                    "target_market": "ashare",
                    "experiments": [{
                        "id": 1,
                        "name": "vector",
                        "market": "ashare",
                        "status": "stopped",
                        "created_at": "2026-08-07",
                    }],
                },
                "manifest.json": {
                    "protocol": (
                        "dual_direction_full_window_"
                        "vector_screen_leaderboard_v1"
                    ),
                    "scenario_engine": "vector_screen",
                    "screening_only": True,
                    "completed_at": "2026-08-07",
                    "duration_seconds": 1,
                    "successful_orientations": 1,
                    "failed_orientations": 0,
                    "economic_equivalence_groups": 1,
                    "direction_counts": {"+1": 1},
                    "practical_pass": 1,
                    "finalist_ledgers": 1,
                    "event_verified_orientations": 1,
                    "window_start": "2020-01-01",
                    "window_end": "2026-08-06",
                    "artifacts": {},
                },
                "protocol.json": {
                    "protocol": (
                        "dual_direction_full_window_"
                        "vector_screen_leaderboard_v1"
                    ),
                    "spec": {
                        "market": "ashare",
                        "mode": "long_only",
                        "initial_capital": 10_000_000,
                    },
                    "panel_glob": "panel/*.parquet",
                    "panel_identity_path_size_mtime_sha256": "abc123",
                    "workers": 1,
                    "threads_per_worker": 1,
                    "ranking_uses_vault": False,
                },
                "finalists_frozen_before_ledger.json": {
                    "orientation_ids": ["winner-plus"],
                    "ranking_uses_post_ranking_ledger": False,
                },
                "vault_finalists.json": {},
                "finalist_ledger_audits.json": {
                    "winner-plus": {
                        "status": "ok",
                        "manifest": {
                            "stats": {
                                "ann_ret": 0.07,
                                "sharpe": 0.5,
                                "max_dd": 0.2,
                                "fill_rate": 0.99,
                                "fills": 10,
                            },
                            "integrity": {"all_pass": True},
                        },
                    },
                },
            }
            for filename, value in files.items():
                (root / filename).write_text(
                    json.dumps(value, ensure_ascii=False),
                    encoding="utf-8",
                )

            manifest = MODULE.render_report(root)
            html = (root / "leaderboard.html").read_text(
                encoding="utf-8"
            )
            self.assertEqual(manifest["embedded_results"], 1)
            self.assertIn("向量筛选与事件复核榜", html)
            self.assertIn("event_ann_return_bps_15", html)
            self.assertIn(
                "finalists_frozen_before_ledger.json",
                manifest["source_files"],
            )

    def test_render_report_is_self_contained_and_embeds_all_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            winner = {
                "overall_rank": 1,
                "status": "ok",
                "economic_representative": True,
                "economic_duplicate_of": "",
                "equivalence_group_size": 1,
                "expression_hash": "winner",
                "expression": "rank(close)",
                "origin_scope": "ashare",
                "direction": 1,
                "practical_pass": True,
                "robust_score": 90.0,
                "ann_return_bps_0": 0.12,
                "ann_return_bps_5": 0.10,
                "ann_return_bps_15": 0.08,
                "sharpe_bps_0": 1.0,
                "sharpe_bps_5": 0.8,
                "sharpe_bps_15": 0.6,
                "oos_ic_mean": 0.03,
                "oos_icir": 1.2,
                "oos_rank_ic_mean": 0.04,
                "oos_rank_icir": 1.4,
                "vault_status": "ok",
                "vault_ann_return_bps_15": 0.05,
                "vault_sharpe_bps_15": 0.4,
                "fields": ["close"],
                "operators": ["rank"],
                "experiment_ids": [1],
            }
            files = {
                "leaderboard_full.json": [winner],
                "snapshot.json": {
                    "snapshot_at_asia_shanghai": "2026-08-06 00:00:00",
                    "cutoffs": {
                        "max_experiment_id": 1,
                        "max_node_id": 1,
                        "max_factor_id": 1,
                    },
                    "source_rows": 1,
                    "unique_expressions": 1,
                    "valid_ashare_expressions": 1,
                    "invalid_ashare_expressions": 0,
                    "invalid": [],
                    "experiments": [{
                        "id": 1,
                        "name": "</script><script>alert(1)</script>",
                        "market": "ashare",
                        "status": "open",
                        "created_at": "2026-08-06",
                    }],
                },
                "manifest.json": {
                    "protocol": "test",
                    "completed_at": "2026-08-06",
                    "duration_seconds": 1,
                    "successful_expressions": 1,
                    "failed_expressions": 0,
                    "economic_equivalence_groups": 1,
                    "origin_scope_counts": {"ashare": 1},
                    "direction_counts": {"+1": 1},
                    "practical_pass": 1,
                    "vault_opened_after_ranking": 1,
                    "finalist_ledgers": 0,
                    "result_generation_protocol": "protocol.json",
                    "artifacts": {},
                },
                "protocol.json": {
                    "spec": {"initial_capital": 10_000_000},
                    "panel_glob": "panel/*.parquet",
                    "panel_identity_path_size_mtime_sha256": "abc123",
                    "workers": 1,
                    "threads_per_worker": 1,
                    "ranking_uses_vault": False,
                },
                "finalists_frozen_before_vault.json": {
                    "ranking_uses_vault": False,
                    "overall_hashes": ["winner"],
                    "dimension_champions": {
                        "robust_score": "winner",
                        "ann_return_bps_15": "winner",
                        "oos_icir": "winner",
                        "oos_rank_icir": "winner",
                    },
                    "hashes": ["winner"],
                },
                "vault_finalists.json": {
                    "winner": {"status": "ok"},
                },
                "finalist_ledger_audits.json": {},
            }
            for filename, value in files.items():
                (root / filename).write_text(
                    json.dumps(value, ensure_ascii=False),
                    encoding="utf-8",
                )

            manifest = MODULE.render_report(root)
            output = root / "leaderboard.html"
            html = output.read_text(encoding="utf-8")
            match = re.search(
                r'<script id="report-data" type="application/json">'
                r"([\s\S]*?)</script>",
                html,
            )
            self.assertIsNotNone(match)
            payload = json.loads(match.group(1))
            self.assertEqual(len(payload["results"]), 1)
            self.assertIn(r"\operatorname{Rank}", payload["results"][0]["latex"])
            self.assertNotIn("</script><script>alert(1)</script>", html)
            self.assertNotRegex(html, r'(?:src|href)=["\']https?://')
            self.assertEqual(manifest["network_dependencies"], [])
            self.assertEqual(
                manifest["html"]["sha256"],
                hashlib.sha256(output.read_bytes()).hexdigest(),
            )


if __name__ == "__main__":
    unittest.main()
