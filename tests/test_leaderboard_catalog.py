import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from backend.app.leaderboards import (
    build_leaderboard_catalog,
    load_leaderboard_factor_detail,
    resolve_leaderboard_file,
)


class LeaderboardCatalogTests(unittest.TestCase):
    def _write_report(
        self,
        directory: Path,
        *,
        market: str,
        mode: str,
        protocol: str,
        generated_at: str,
        window_end: str,
        html: str,
    ) -> str:
        directory.mkdir(parents=True)
        (directory / "leaderboard.html").write_text(html, encoding="utf-8")
        digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
        (directory / "protocol.json").write_text(
            json.dumps({
                "protocol": protocol,
                "target_market": market,
                "portfolio_mode": mode,
                "policy_label": "NON_PIT_RESEARCH",
                "ranking_window": {
                    "start": "2020-01-01",
                    "end": window_end,
                },
                "screening_only": "vector_screen" in protocol,
                "spec": {"market": market, "mode": mode},
            }),
            encoding="utf-8",
        )
        (directory / "leaderboard_manifest.json").write_text(
            json.dumps({
                "generated_at": generated_at,
                "embedded_results": 12,
                "html": {"sha256": digest},
            }),
            encoding="utf-8",
        )
        (directory / "progress.json").write_text(
            json.dumps({"percent": 100, "errors": 0}),
            encoding="utf-8",
        )
        (directory / "manifest.json").write_text("{}", encoding="utf-8")
        (directory / "finalist_ledger_audits.json").write_text(
            "{}", encoding="utf-8"
        )
        return digest

    def test_catalog_deduplicates_archive_and_marks_latest_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            active = root / "var/reports"
            archived = (
                root
                / "var/archives/factor-leaderboards/20260807-120000/reports"
            )
            old_html = "<html><body id='overview'>old</body></html>"
            self._write_report(
                active / "ashare-old",
                market="ashare",
                mode="long_only",
                protocol="ashare_cross_task_factor_leaderboard_v1",
                generated_at="2026-08-06T01:00:00+08:00",
                window_end="2024-12-31",
                html=old_html,
            )
            self._write_report(
                archived / "ashare-old-copy",
                market="ashare",
                mode="long_only",
                protocol="ashare_cross_task_factor_leaderboard_v1",
                generated_at="2026-08-06T01:00:00+08:00",
                window_end="2024-12-31",
                html=old_html,
            )
            new_digest = self._write_report(
                active / "ashare-vector",
                market="ashare",
                mode="long_only",
                protocol=(
                    "dual_direction_full_window_"
                    "vector_screen_leaderboard_v1"
                ),
                generated_at="2026-08-07T14:00:00+08:00",
                window_end="2026-08-06",
                html="<html><body id='overview'>new</body></html>",
            )

            catalog = build_leaderboard_catalog(root)

            self.assertEqual(catalog["summary"]["logical_versions"], 2)
            self.assertEqual(catalog["summary"]["archive_copies"], 1)
            self.assertEqual(catalog["summary"]["complete_versions"], 2)
            newest = catalog["reports"][0]
            self.assertEqual(newest["id"], new_digest[:24])
            self.assertEqual(newest["version_kind"], "full_window_vector")
            self.assertTrue(newest["is_latest_for_scope"])
            older = catalog["reports"][1]
            self.assertFalse(older["is_latest_for_scope"])
            self.assertEqual(older["archive_copy_count"], 1)
            self.assertEqual(older["copies"][0]["location"], "active")

    def test_report_file_resolution_is_catalogued_and_traversal_safe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_dir = root / "var/reports/us-report"
            digest = self._write_report(
                report_dir,
                market="us",
                mode="long_short",
                protocol="us_cross_task_long_short_factor_leaderboard_v1",
                generated_at="2026-08-06T02:00:00+08:00",
                window_end="2024-12-31",
                html="<html>report</html>",
            )
            resolved = resolve_leaderboard_file(
                digest[:24], "leaderboard.html", root
            )
            self.assertEqual(resolved, (report_dir / "leaderboard.html").resolve())
            with self.assertRaises(ValueError):
                resolve_leaderboard_file(digest[:24], "../protocol.json", root)
            with self.assertRaises(ValueError):
                resolve_leaderboard_file(digest[:24], "secret.env", root)
            with self.assertRaises(FileNotFoundError):
                resolve_leaderboard_file("missing", "leaderboard.html", root)

    def test_factor_detail_uses_frozen_report_row_and_explains_direction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report_dir = root / "var/reports/ashare-report"
            digest = self._write_report(
                report_dir,
                market="ashare",
                mode="long_only",
                protocol=(
                    "dual_direction_full_window_"
                    "vector_screen_leaderboard_v1"
                ),
                generated_at="2026-08-07T14:00:00+08:00",
                window_end="2026-08-07",
                html="<html>report</html>",
            )
            (report_dir / "leaderboard_full.json").write_text(
                json.dumps([{
                    "expression_hash": "volatility-minus",
                    "oriented_expression_hash": "volatility",
                    "expression": "ts_std((high-low)/close, 40)",
                    "direction": -1,
                    "overall_rank": 1,
                    "portfolio_metric_basis": "active",
                    "ann_return_bps_15": 0.12,
                    "sharpe_bps_15": 1.1,
                    "active_ann_return_bps_15": 0.07,
                    "active_sharpe_bps_15": 0.6,
                    "oos_ic_mean": 0.02,
                    "oos_icir": 0.8,
                    "oos_rank_ic_mean": 0.03,
                    "oos_rank_icir": 1.0,
                    "fields": ["close", "high", "low"],
                    "operators": ["ts_std"],
                    "factor_ids": [7],
                    "experiment_ids": [3],
                    "origin_scope": "ashare",
                }]),
                encoding="utf-8",
            )

            detail = load_leaderboard_factor_detail(
                digest[:24],
                "volatility-minus",
                root,
            )

            self.assertEqual(detail["expression"], "ts_std((high-low)/close, 40)")
            self.assertEqual(detail["direction"], -1)
            self.assertEqual(
                detail["economics"]["mechanism_family"],
                "volatility",
            )
            self.assertIn("低值端", detail["economics"]["direction_interpretation"])
            self.assertIn("不是因果证明", detail["economics"]["disclaimer"])
            self.assertIn(r"\operatorname{Std}", detail["latex"])
            self.assertTrue(detail["report"]["screening_only"])
            self.assertEqual(
                detail["metrics"]["ranking_ann_return_bps_15"],
                0.07,
            )
            self.assertEqual(
                detail["metrics"]["ranking_sharpe_bps_15"],
                0.6,
            )

            with self.assertRaises(FileNotFoundError):
                load_leaderboard_factor_detail(
                    digest[:24],
                    "missing-factor",
                    root,
                )

    def test_frontend_exposes_top_tab_filters_and_quick_jumps(self):
        frontend = Path(__file__).resolve().parents[1] / "frontend/app.js"
        source = frontend.read_text(encoding="utf-8")
        self.assertIn('{ id: "leaderboards", label: "榜单" }', source)
        self.assertIn('data-testid="leaderboards-page"', source)
        self.assertIn('aria-label="报告快速跳转"', source)
        self.assertIn('aria-label="榜单因子详情"', source)
        self.assertIn('data-testid="leaderboard-factor-latex"', source)
        self.assertIn('event.target?.closest', source)
        self.assertIn('target="_blank" rel="noopener"', source)
        self.assertIn("const ScreenerView", source)
        app_components = source.split("const App =", 1)[1].split("template:", 1)[0]
        self.assertNotIn("ScreenerView", app_components)


if __name__ == "__main__":
    unittest.main()
