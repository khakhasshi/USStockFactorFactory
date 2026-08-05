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
