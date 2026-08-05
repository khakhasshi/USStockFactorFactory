"""Render a self-contained interactive HTML report from a frozen leaderboard.

The generated file has no network dependencies. It embeds a compact projection
of every factor result, the frozen experiment inventory, Vault checks, invalid
expressions, and finalist-ledger audit metadata.

Example:
    .venv/bin/python backend/scripts/render_factor_leaderboard_html.py \
      --report-dir \
      var/reports/ashare-all-factor-leaderboard-corrected-snapshot-2208-1522
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.dsl.engine import expression_to_latex  # noqa: E402


HTML_REPORT_PROTOCOL = "factor_leaderboard_html_v1"
REQUIRED_FILES = (
    "leaderboard_full.json",
    "snapshot.json",
    "manifest.json",
    "protocol.json",
    "finalists_frozen_before_vault.json",
    "vault_finalists.json",
    "finalist_ledger_audits.json",
)
RESULT_FIELDS = (
    "overall_rank",
    "practical_pass",
    "multiple_test_pass",
    "cost_monotonic",
    "robust_score",
    "expression_hash",
    "oriented_expression_hash",
    "evaluation_fingerprint",
    "origin_scope",
    "origin_markets",
    "direction",
    "economic_representative",
    "economic_duplicate_of",
    "equivalence_group_size",
    "experiment_ids",
    "factor_ids",
    "node_ids",
    "source_record_count",
    "fields",
    "operators",
    "required_history",
    "complexity",
    "train_ic_mean",
    "train_icir",
    "train_rank_ic_mean",
    "train_rank_icir",
    "oos_ic_mean",
    "oos_icir",
    "oos_rank_ic_mean",
    "oos_rank_icir",
    "oos_rank_ic_bh_q",
    "ann_return_bps_0",
    "ann_return_bps_5",
    "ann_return_bps_15",
    "sharpe_bps_0",
    "sharpe_bps_5",
    "sharpe_bps_15",
    "max_drawdown_bps_15",
    "avg_daily_turnover_bps_15",
    "fill_rate_bps_15",
    "fills_bps_15",
    "commission_and_tax_bps_15",
    "slippage_cost_bps_15",
    "total_execution_cost_bps_15",
    "vault_status",
    "vault_ann_return_bps_15",
    "vault_sharpe_bps_15",
    "vault_ic_mean",
    "vault_icir",
    "vault_rank_ic_mean",
    "vault_rank_icir",
    "expression",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _project_result(row: dict) -> dict:
    projected = {key: row.get(key) for key in RESULT_FIELDS}
    expression = str(row.get("expression") or "")
    try:
        projected["latex"] = expression_to_latex(expression)
    except (SyntaxError, TypeError, ValueError):
        projected["latex"] = expression
    return projected


def _build_payload(report_dir: Path) -> dict:
    paths = {name: report_dir / name for name in REQUIRED_FILES}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"报告目录缺少文件: {', '.join(missing)}")

    leaderboard = _read_json(paths["leaderboard_full.json"])
    snapshot = _read_json(paths["snapshot.json"])
    manifest = _read_json(paths["manifest.json"])
    protocol = _read_json(paths["protocol.json"])
    finalists = _read_json(paths["finalists_frozen_before_vault.json"])
    vault = _read_json(paths["vault_finalists.json"])
    ledger_audits = _read_json(paths["finalist_ledger_audits.json"])

    if len(leaderboard) != int(manifest["successful_expressions"]):
        raise ValueError("leaderboard 行数与 manifest 成功数不一致")
    representative_count = sum(
        bool(row.get("economic_representative")) for row in leaderboard
    )
    if representative_count != int(manifest["economic_equivalence_groups"]):
        raise ValueError("经济评估组数量与 manifest 不一致")

    experiments = [
        {
            "id": row["id"],
            "name": row["name"],
            "market": row["market"],
            "status": row["status"],
            "created_at": row["created_at"],
        }
        for row in snapshot["experiments"]
    ]
    invalid = [
        {
            "expression_hash": row["expression_hash"],
            "expression": row["expression"],
            "origin_scope": row["origin_scope"],
            "experiment_ids": row["experiment_ids"],
            "validation_error": row["validation_error"],
        }
        for row in snapshot["invalid"]
    ]
    ledger_rows = {}
    for expression_hash, audit in ledger_audits.items():
        nested_manifest = audit.get("manifest") or {}
        ledger_rows[expression_hash] = {
            "status": audit.get("status"),
            "artifact_dir": audit.get("artifact_dir"),
            "runtime_seconds": audit.get("runtime_seconds"),
            "summary": audit.get("summary"),
            "stats": nested_manifest.get("stats"),
            "integrity": nested_manifest.get("integrity"),
            "files": nested_manifest.get("files"),
            "fee_schedule": nested_manifest.get("fee_schedule"),
        }

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    return {
        "report_protocol": HTML_REPORT_PROTOCOL,
        "generated_at": generated_at,
        "policy_label": "NON_PIT_RESEARCH",
        "snapshot": {
            "snapshot_at": snapshot["snapshot_at_asia_shanghai"],
            "cutoffs": snapshot["cutoffs"],
            "source_rows": snapshot["source_rows"],
            "unique_expressions": snapshot["unique_expressions"],
            "valid_expressions": snapshot["valid_ashare_expressions"],
            "invalid_expressions": snapshot["invalid_ashare_expressions"],
        },
        "manifest": {
            key: manifest.get(key)
            for key in (
                "protocol",
                "completed_at",
                "duration_seconds",
                "successful_expressions",
                "failed_expressions",
                "economic_equivalence_groups",
                "origin_scope_counts",
                "direction_counts",
                "practical_pass",
                "vault_opened_after_ranking",
                "finalist_ledgers",
                "result_generation_protocol",
            )
        }
        | {"artifact_count": len(manifest.get("artifacts") or {})},
        "protocol": {
            "spec": protocol["spec"],
            "panel_glob": protocol["panel_glob"],
            "panel_identity": protocol[
                "panel_identity_path_size_mtime_sha256"
            ],
            "workers": protocol["workers"],
            "threads_per_worker": protocol["threads_per_worker"],
            "ranking_uses_vault": protocol["ranking_uses_vault"],
            "result_generation_protocol_file": protocol.get(
                "result_generation_protocol_file"
            ),
            "result_generation_sources_match_current": protocol.get(
                "result_generation_sources_match_current"
            ),
        },
        "experiments": experiments,
        "results": [_project_result(row) for row in leaderboard],
        "finalists": finalists,
        "vault": vault,
        "ledgers": ledger_rows,
        "invalid": invalid,
        "source_files": {
            name: {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
    }


def render_report(
    report_dir: Path,
    output_path: Path | None = None,
) -> dict:
    report_dir = report_dir.resolve()
    output_path = (
        output_path.resolve()
        if output_path
        else report_dir / "leaderboard.html"
    )
    payload = _build_payload(report_dir)
    html = (
        _HTML_TEMPLATE.replace(
            "__REPORT_DATA__",
            _json_for_script(payload),
        )
        .replace("__GENERATED_AT__", payload["generated_at"])
        .replace("__SNAPSHOT_AT__", str(payload["snapshot"]["snapshot_at"]))
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    derivative_manifest = {
        "protocol": HTML_REPORT_PROTOCOL,
        "generated_at": payload["generated_at"],
        "html": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": _sha256(output_path),
        },
        "source_files": payload["source_files"],
        "embedded_results": len(payload["results"]),
        "embedded_invalid_expressions": len(payload["invalid"]),
        "network_dependencies": [],
    }
    derivative_path = output_path.with_name(
        f"{output_path.stem}_manifest.json"
    )
    derivative_path.write_text(
        json.dumps(
            derivative_manifest,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    derivative_manifest["manifest_path"] = str(derivative_path)
    return derivative_manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = render_report(args.report_dir, args.output)
    print(
        "HTML 报表完成 · "
        f"{manifest['embedded_results']} 个因子 · "
        f"{manifest['html']['bytes'] / 1024 / 1024:.2f} MiB · "
        f"{manifest['html']['path']}"
    )


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="dark">
  <meta name="robots" content="noindex,nofollow">
  <title>FactorFactory · A股全任务因子审计榜</title>
  <link rel="icon" href="data:,">
  <style>
    :root {
      --bg: #090d12;
      --panel: #11171f;
      --panel-2: #151d27;
      --panel-3: #0c1219;
      --border: #283341;
      --border-soft: #1d2733;
      --text: #e8edf3;
      --muted: #8d99a8;
      --faint: #596575;
      --blue: #58a6ff;
      --blue-2: #79c0ff;
      --green: #3fb950;
      --green-soft: rgba(63,185,80,.12);
      --amber: #d9a441;
      --amber-soft: rgba(217,164,65,.12);
      --red: #ff6b63;
      --red-soft: rgba(255,107,99,.11);
      --cyan: #4ec9b0;
      --shadow: 0 18px 50px rgba(0,0,0,.28);
      --mono: "SFMono-Regular", "SF Mono", Menlo, Consolas, monospace;
      --sans: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; background: var(--bg); }
    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 75% -10%, rgba(31,111,235,.14), transparent 33rem),
        linear-gradient(180deg, #0b1017 0, var(--bg) 34rem);
      font-family: var(--sans);
      font-size: 13px;
      line-height: 1.5;
    }
    button, input, select { font: inherit; }
    button { color: inherit; }
    a { color: var(--blue-2); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code, .mono { font-family: var(--mono); }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 40;
      display: flex;
      align-items: center;
      gap: 18px;
      min-height: 58px;
      padding: 0 28px;
      border-bottom: 1px solid rgba(40,51,65,.86);
      background: rgba(9,13,18,.88);
      backdrop-filter: blur(18px);
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 10px;
      flex: 0 0 auto;
      font-family: var(--mono);
      font-size: 15px;
      font-weight: 750;
      color: var(--blue);
      letter-spacing: -.02em;
    }
    .brand-mark {
      display: grid;
      place-items: center;
      width: 25px;
      height: 25px;
      border: 1px solid #28619d;
      border-radius: 7px;
      background: #0c2035;
      box-shadow: inset 0 0 18px rgba(88,166,255,.12);
    }
    .nav { display: flex; gap: 3px; margin-left: 8px; }
    .nav a {
      color: var(--muted);
      border-radius: 6px;
      padding: 6px 9px;
      font-size: 12px;
    }
    .nav a:hover { color: var(--text); background: var(--panel-2); text-decoration: none; }
    .top-meta {
      margin-left: auto;
      display: flex;
      align-items: center;
      gap: 9px;
      min-width: 0;
    }
    .badge, .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      white-space: nowrap;
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 3px 9px;
      color: var(--muted);
      background: rgba(17,23,31,.8);
      font-family: var(--mono);
      font-size: 10px;
    }
    .badge::before {
      content: "";
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: currentColor;
      box-shadow: 0 0 8px currentColor;
    }
    .badge.ok { color: var(--green); border-color: rgba(63,185,80,.35); }
    .badge.warn { color: var(--amber); border-color: rgba(217,164,65,.4); }
    .shell { width: min(1680px, 100%); margin: 0 auto; padding: 30px 28px 72px; }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(320px, .6fr);
      gap: 20px;
      align-items: stretch;
      margin-bottom: 18px;
    }
    .hero-copy, .hero-aside, .card {
      border: 1px solid var(--border);
      border-radius: 12px;
      background: linear-gradient(145deg, rgba(21,29,39,.95), rgba(13,19,27,.95));
      box-shadow: var(--shadow);
    }
    .hero-copy { padding: 28px 30px; position: relative; overflow: hidden; }
    .hero-copy::after {
      content: "";
      position: absolute;
      width: 280px;
      height: 280px;
      right: -110px;
      top: -150px;
      border-radius: 50%;
      border: 1px solid rgba(88,166,255,.20);
      box-shadow: 0 0 80px rgba(88,166,255,.12);
    }
    .eyebrow {
      color: var(--blue);
      font-family: var(--mono);
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .15em;
      text-transform: uppercase;
    }
    h1 {
      max-width: 820px;
      margin: 9px 0 10px;
      font-size: clamp(26px, 3vw, 42px);
      line-height: 1.08;
      letter-spacing: -.045em;
    }
    .hero-copy p { max-width: 850px; margin: 0; color: var(--muted); font-size: 14px; }
    .hero-copy .meta-line {
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin-top: 22px;
      color: var(--faint);
      font-family: var(--mono);
      font-size: 10px;
    }
    .hero-aside { padding: 22px; display: flex; flex-direction: column; justify-content: space-between; }
    .boundary-title { color: var(--amber); font-family: var(--mono); font-size: 11px; font-weight: 700; }
    .hero-aside p { color: var(--muted); margin: 8px 0 18px; }
    .hero-aside strong { color: var(--text); }
    .mini-links { display: flex; flex-wrap: wrap; gap: 7px; }
    .mini-links a {
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 5px 8px;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 10px;
    }
    .mini-links a:hover { border-color: var(--blue); color: var(--blue-2); text-decoration: none; }
    .kpis {
      display: grid;
      grid-template-columns: repeat(6, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }
    .kpi {
      min-height: 104px;
      padding: 15px 16px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(17,23,31,.78);
    }
    .kpi-label { color: var(--muted); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; }
    .kpi-value { margin: 7px 0 2px; font-family: var(--mono); font-size: 25px; font-weight: 750; letter-spacing: -.04em; }
    .kpi-foot { color: var(--faint); font-size: 10px; }
    .kpi.accent { border-color: rgba(88,166,255,.42); background: linear-gradient(145deg, rgba(17,38,61,.85), rgba(17,23,31,.8)); }
    .kpi.accent .kpi-value { color: var(--blue-2); }
    .section { margin-top: 18px; scroll-margin-top: 76px; }
    .section-heading {
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 18px;
      margin: 0 2px 10px;
    }
    .section-heading h2 { margin: 0; font-size: 15px; letter-spacing: -.01em; }
    .section-heading p { margin: 3px 0 0; color: var(--muted); font-size: 11px; }
    .card { padding: 16px; box-shadow: none; }
    .insight-grid { display: grid; grid-template-columns: 1.35fr 1fr 1fr; gap: 12px; }
    .insight-card { min-height: 190px; position: relative; overflow: hidden; }
    .insight-card.primary { border-color: rgba(88,166,255,.40); background: linear-gradient(145deg, #12243a, #111820 65%); }
    .insight-title { color: var(--muted); font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: .1em; }
    .insight-card h3 { margin: 8px 0 8px; font-size: 18px; line-height: 1.25; }
    .formula {
      display: block;
      padding: 10px 11px;
      border: 1px solid var(--border-soft);
      border-radius: 7px;
      background: var(--panel-3);
      color: var(--blue-2);
      font-family: var(--mono);
      font-size: 11px;
      line-height: 1.6;
      overflow-wrap: anywhere;
    }
    .metric-row { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 12px; }
    .metric-chip {
      min-width: 82px;
      padding: 6px 8px;
      border: 1px solid var(--border);
      border-radius: 7px;
      background: rgba(9,13,18,.52);
    }
    .metric-chip span { display: block; color: var(--faint); font-size: 9px; text-transform: uppercase; }
    .metric-chip b { font-family: var(--mono); font-size: 12px; }
    .insight-card p { color: var(--muted); font-size: 12px; }
    .signal { display: inline-block; border-radius: 4px; padding: 1px 5px; font-family: var(--mono); font-size: 9px; }
    .signal.good { color: var(--green); background: var(--green-soft); }
    .signal.warn { color: var(--amber); background: var(--amber-soft); }
    .signal.bad { color: var(--red); background: var(--red-soft); }
    .charts { display: grid; grid-template-columns: 1.35fr .65fr; gap: 12px; }
    .chart-card { padding: 0; overflow: hidden; }
    .card-head { display: flex; justify-content: space-between; align-items: start; gap: 12px; padding: 14px 16px 8px; }
    .card-head h3 { margin: 0; font-size: 12px; }
    .card-head p { margin: 3px 0 0; color: var(--muted); font-size: 10px; }
    .chart-wrap { position: relative; height: 330px; padding: 0 8px 10px; }
    canvas { display: block; width: 100%; height: 100%; }
    .tooltip {
      position: absolute;
      z-index: 5;
      display: none;
      max-width: 330px;
      pointer-events: none;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 8px 9px;
      background: rgba(9,13,18,.96);
      box-shadow: var(--shadow);
      color: var(--text);
      font-family: var(--mono);
      font-size: 10px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(260px, 1.5fr) repeat(3, minmax(130px, .45fr)) auto auto;
      gap: 8px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--border);
      background: rgba(9,13,18,.35);
    }
    input, select {
      min-width: 0;
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 8px 10px;
      color: var(--text);
      background: var(--panel-3);
      outline: none;
    }
    input:focus, select:focus { border-color: var(--blue); box-shadow: 0 0 0 2px rgba(88,166,255,.10); }
    .toggle {
      display: flex;
      align-items: center;
      gap: 7px;
      padding: 0 8px;
      color: var(--muted);
      white-space: nowrap;
      font-size: 11px;
    }
    .toggle input { width: auto; accent-color: var(--blue); }
    .btn {
      border: 1px solid var(--border);
      border-radius: 7px;
      padding: 7px 10px;
      background: var(--panel-2);
      cursor: pointer;
      white-space: nowrap;
    }
    .btn:hover { border-color: var(--blue); color: var(--blue-2); }
    .table-card { padding: 0; overflow: hidden; }
    .table-scroll { overflow: auto; max-height: 720px; }
    table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
    th, td { padding: 8px 9px; border-bottom: 1px solid var(--border-soft); text-align: right; white-space: nowrap; font-size: 10.5px; }
    th {
      position: sticky;
      top: 0;
      z-index: 2;
      color: var(--muted);
      background: #111820;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: .04em;
      cursor: pointer;
      user-select: none;
    }
    th:hover { color: var(--blue-2); }
    th.left, td.left { text-align: left; }
    tbody tr { cursor: pointer; }
    tbody tr:hover { background: rgba(88,166,255,.055); }
    tbody tr.selected { background: rgba(88,166,255,.10); outline: 1px solid rgba(88,166,255,.35); outline-offset: -1px; }
    .rank { color: var(--muted); font-family: var(--mono); }
    .rank.top { color: var(--green); font-weight: 800; }
    .factor-cell { max-width: 330px; min-width: 240px; }
    .hash { color: var(--blue-2); font-family: var(--mono); font-size: 10px; }
    .expr {
      display: block;
      max-width: 330px;
      overflow: hidden;
      color: var(--muted);
      font-family: var(--mono);
      font-size: 9px;
      text-overflow: ellipsis;
    }
    .metric-stack { display: grid; grid-template-columns: repeat(3, minmax(43px, auto)); gap: 3px; }
    .metric-stack span { padding: 2px 4px; border-radius: 3px; background: rgba(255,255,255,.025); font-family: var(--mono); }
    .metric-stack .stress { color: var(--amber); }
    .positive { color: var(--green); }
    .negative { color: var(--red); }
    .neutral { color: var(--muted); }
    .origin { text-transform: uppercase; }
    .table-foot {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 14px;
      color: var(--muted);
      font-size: 10px;
      background: rgba(9,13,18,.35);
    }
    .pager { display: flex; gap: 6px; align-items: center; }
    .pager button:disabled { opacity: .35; cursor: not-allowed; }
    .vault-grid { display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(340px, .75fr); gap: 12px; }
    .vault-table th, .vault-table td { font-size: 10px; }
    .champions { display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 8px; }
    .champion {
      display: grid;
      grid-template-columns: minmax(115px,.7fr) 1.2fr auto;
      align-items: center;
      gap: 8px;
      padding: 9px 10px;
      border: 1px solid var(--border-soft);
      border-radius: 7px;
      background: var(--panel-3);
    }
    .champion-label { color: var(--muted); font-size: 10px; }
    .champion-hash { overflow: hidden; color: var(--blue-2); font-family: var(--mono); font-size: 10px; text-overflow: ellipsis; }
    .champion-value { font-family: var(--mono); font-size: 11px; text-align: right; }
    .audit-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .timeline { display: grid; gap: 8px; }
    .timeline-row {
      display: grid;
      grid-template-columns: 100px 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 9px 10px;
      border: 1px solid var(--border-soft);
      border-radius: 7px;
      background: var(--panel-3);
    }
    .timeline-row b { font-family: var(--mono); font-size: 10px; }
    .timeline-row span { color: var(--muted); font-size: 10px; }
    .ledger-list { display: grid; gap: 8px; }
    .ledger-row {
      padding: 10px;
      border: 1px solid var(--border-soft);
      border-radius: 7px;
      background: var(--panel-3);
    }
    .ledger-row-head { display: flex; justify-content: space-between; gap: 10px; }
    .ledger-row .metrics { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; margin-top: 9px; }
    .ledger-row .metrics span { color: var(--muted); font-size: 9px; }
    .ledger-row .metrics b { display: block; margin-top: 2px; font-family: var(--mono); font-size: 11px; color: var(--text); }
    details { border-top: 1px solid var(--border); padding: 12px 0 0; margin-top: 12px; }
    summary { color: var(--muted); cursor: pointer; }
    .invalid-list { margin-top: 10px; display: grid; gap: 6px; }
    .invalid-row { padding: 8px 9px; border: 1px solid var(--border-soft); border-radius: 6px; background: var(--panel-3); }
    .invalid-row code { display: block; margin-top: 4px; color: var(--muted); overflow-wrap: anywhere; }
    .drawer-backdrop {
      position: fixed;
      inset: 0;
      z-index: 79;
      display: none;
      background: rgba(0,0,0,.55);
      backdrop-filter: blur(2px);
    }
    .drawer-backdrop.open { display: block; }
    .drawer {
      position: fixed;
      z-index: 80;
      top: 0;
      right: 0;
      width: min(720px, 94vw);
      height: 100vh;
      padding: 22px;
      overflow: auto;
      border-left: 1px solid var(--border);
      background: #101720;
      box-shadow: -20px 0 70px rgba(0,0,0,.55);
      transform: translateX(105%);
      transition: transform .18s ease;
    }
    .drawer.open { transform: translateX(0); }
    .drawer-top { display: flex; justify-content: space-between; gap: 16px; align-items: start; }
    .drawer h2 { margin: 5px 0 3px; font-family: var(--mono); font-size: 17px; }
    .drawer-close { width: 32px; height: 32px; padding: 0; }
    .drawer-section { margin-top: 16px; }
    .drawer-section h3 { margin: 0 0 8px; color: var(--muted); font-family: var(--mono); font-size: 10px; text-transform: uppercase; letter-spacing: .08em; }
    .drawer-formula { font-size: 12px; }
    .latex {
      margin-top: 7px;
      padding: 9px 10px;
      border: 1px dashed var(--border);
      border-radius: 7px;
      color: var(--cyan);
      background: rgba(9,13,18,.5);
      font-family: var(--mono);
      font-size: 10px;
      overflow-wrap: anywhere;
    }
    .metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 8px; }
    .detail-metric { padding: 9px; border: 1px solid var(--border-soft); border-radius: 7px; background: var(--panel-3); }
    .detail-metric span { color: var(--faint); font-size: 9px; }
    .detail-metric b { display: block; margin-top: 4px; font-family: var(--mono); font-size: 13px; }
    .layer-table th { position: static; cursor: default; }
    .layer-table th, .layer-table td { font-size: 10px; }
    .tags { display: flex; flex-wrap: wrap; gap: 6px; }
    .footer {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      margin-top: 22px;
      padding: 16px 2px 0;
      border-top: 1px solid var(--border);
      color: var(--faint);
      font-family: var(--mono);
      font-size: 9px;
    }
    @media (max-width: 1200px) {
      .kpis { grid-template-columns: repeat(3, 1fr); }
      .insight-grid { grid-template-columns: 1fr 1fr; }
      .insight-card.primary { grid-column: 1 / -1; }
      .charts, .vault-grid { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr 1fr 1fr; }
    }
    @media (max-width: 760px) {
      .topbar { padding: 8px 12px; flex-wrap: wrap; }
      .nav { order: 3; width: 100%; overflow: auto; margin-left: 0; }
      .top-meta { margin-left: auto; }
      .top-meta .pill { display: none; }
      .shell { padding: 18px 12px 50px; }
      .hero { grid-template-columns: 1fr; }
      .hero-copy { padding: 22px 20px; }
      .kpis { grid-template-columns: repeat(2, 1fr); }
      .insight-grid, .audit-grid { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr 1fr; }
      .toolbar input:first-child { grid-column: 1 / -1; }
      .champions { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2,1fr); }
      .drawer { width: 100vw; }
      .footer { flex-direction: column; }
    }
    @media print {
      body { background: #fff; color: #111; }
      .topbar, .toolbar, .pager, .drawer, .drawer-backdrop, .chart-card { display: none !important; }
      .shell { width: 100%; padding: 0; }
      .card, .hero-copy, .hero-aside, .kpi { box-shadow: none; background: #fff; border-color: #bbb; break-inside: avoid; }
      .hero { grid-template-columns: 1fr; }
      .table-scroll { max-height: none; overflow: visible; }
      th { position: static; background: #eee; color: #333; }
      td, .expr, .hero-copy p, .insight-card p { color: #333; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div class="brand"><span class="brand-mark">×</span> FactorFactory</div>
    <nav class="nav" aria-label="报告导航">
      <a href="#overview">结论</a>
      <a href="#charts">分布</a>
      <a href="#leaderboard">全量榜单</a>
      <a href="#vault">Vault</a>
      <a href="#audit">审计</a>
    </nav>
    <div class="top-meta">
      <span class="pill">冻结 __SNAPSHOT_AT__</span>
      <span class="badge ok">complete</span>
    </div>
  </header>

  <main class="shell">
    <section class="hero" id="overview">
      <div class="hero-copy">
        <div class="eyebrow">A-share event replay · cross-task history</div>
        <h1>全任务历史因子<br>多维实战审计榜</h1>
        <p>将 A 股与美股任务发现的全部历史 DSL 表达式，统一放入 A 股纯多头事件引擎，在真实费税基础上追加 0 / 5 / 15 BPS 双边滑点，并以训练定向、独立榜单期和一次性 Vault 分层验证。</p>
        <div class="meta-line">
          <span id="heroProtocol">protocol —</span>
          <span id="heroPanel">panel —</span>
          <span>generated __GENERATED_AT__</span>
        </div>
      </div>
      <aside class="hero-aside">
        <div>
          <div class="boundary-title">RESEARCH BOUNDARY / 研究边界</div>
          <p>当前面板为<strong>非 PIT 当前成分股研究面板</strong>。榜单通过意味着比较口径可复现、成本已计入且账本自洽，不代表可以直接用于实盘。</p>
          <span class="badge warn">NON_PIT_RESEARCH</span>
        </div>
        <div class="mini-links">
          <a href="leaderboard_full.csv">完整 CSV</a>
          <a href="top_by_dimension.csv">维度榜</a>
          <a href="manifest.json">审计 Manifest</a>
          <a href="invalid_expressions.csv">无效表达式</a>
        </div>
      </aside>
    </section>

    <section class="kpis" id="kpis" aria-label="运行摘要"></section>

    <section class="section">
      <div class="section-heading">
        <div><h2>决策摘要</h2><p>先看稳定性与失效证据，再看单项冠军。</p></div>
      </div>
      <div class="insight-grid" id="insights"></div>
    </section>

    <section class="section" id="charts">
      <div class="section-heading">
        <div><h2>收益与预测力分布</h2><p>每个点是一个经济评估代表；点击点可打开因子详情。</p></div>
      </div>
      <div class="charts">
        <article class="card chart-card">
          <div class="card-head">
            <div><h3>RankIC × 15bp 年化收益</h3><p>绿色为实战硬筛通过，蓝色描边为已开 Vault。</p></div>
            <span class="pill" id="scatterCount">—</span>
          </div>
          <div class="chart-wrap">
            <canvas id="scatterCanvas" aria-label="RankIC 与 15bp 年化收益散点图"></canvas>
            <div class="tooltip" id="scatterTip"></div>
          </div>
        </article>
        <article class="card chart-card">
          <div class="card-head">
            <div><h3>成本压力曲线</h3><p>总榜前五与当前选中因子的年化收益。</p></div>
          </div>
          <div class="chart-wrap">
            <canvas id="costCanvas" aria-label="0 5 15 BPS 成本压力曲线"></canvas>
          </div>
        </article>
      </div>
    </section>

    <section class="section" id="leaderboard">
      <div class="section-heading">
        <div><h2>全量多维榜单</h2><p>默认仅显示经济代表；完整数据包含所有历史重复表达式。</p></div>
        <span class="pill" id="tableCount">—</span>
      </div>
      <article class="card table-card">
        <div class="toolbar">
          <input id="searchInput" type="search" placeholder="搜索 Hash、DSL、字段、实验 ID…" aria-label="搜索因子">
          <select id="originFilter" aria-label="来源市场">
            <option value="">全部来源</option>
            <option value="ashare">A股任务</option>
            <option value="us">美股任务</option>
            <option value="both">双来源</option>
          </select>
          <select id="passFilter" aria-label="硬筛状态">
            <option value="">全部硬筛状态</option>
            <option value="pass">仅通过</option>
            <option value="fail">仅未通过</option>
          </select>
          <select id="vaultFilter" aria-label="Vault 状态">
            <option value="">全部 Vault 状态</option>
            <option value="ok">已开 Vault</option>
            <option value="not_opened">未开 Vault</option>
          </select>
          <label class="toggle"><input id="representativeOnly" type="checkbox" checked>仅经济代表</label>
          <button class="btn" id="exportButton" type="button">导出筛选 CSV</button>
        </div>
        <div class="table-scroll">
          <table aria-label="全量因子榜单">
            <thead>
              <tr>
                <th data-sort="overall_rank">总榜</th>
                <th class="left" data-sort="expression_hash">因子 / 表达式</th>
                <th data-sort="origin_scope">来源</th>
                <th data-sort="direction">方向</th>
                <th data-sort="robust_score">稳健分</th>
                <th data-sort="ann_return_bps_15">年化 0/5/15</th>
                <th data-sort="sharpe_bps_15">夏普 0/5/15</th>
                <th data-sort="oos_ic_mean">IC / ICIR</th>
                <th data-sort="oos_rank_ic_mean">RankIC / IR</th>
                <th data-sort="vault_ann_return_bps_15">Vault</th>
                <th data-sort="practical_pass">判定</th>
              </tr>
            </thead>
            <tbody id="leaderboardBody"></tbody>
          </table>
        </div>
        <div class="table-foot">
          <span id="filterSummary">—</span>
          <div class="pager">
            <select id="pageSize" aria-label="每页数量">
              <option value="25">25 / 页</option>
              <option value="50" selected>50 / 页</option>
              <option value="100">100 / 页</option>
            </select>
            <button class="btn" id="prevPage" type="button">上一页</button>
            <span id="pageLabel">—</span>
            <button class="btn" id="nextPage" type="button">下一页</button>
          </div>
        </div>
      </article>
    </section>

    <section class="section" id="vault">
      <div class="section-heading">
        <div><h2>Vault 与维度冠军</h2><p>候选 Hash 在开 Vault 前已冻结；Vault 不参与总榜排序。</p></div>
        <span class="badge ok" id="vaultStatus">—</span>
      </div>
      <div class="vault-grid">
        <article class="card table-card">
          <div class="card-head">
            <div><h3>独立期生存结果</h3><p>2025-01-01 至 2026-08-04，15 BPS。</p></div>
          </div>
          <div class="table-scroll" style="max-height:470px">
            <table class="vault-table">
              <thead><tr>
                <th>总榜</th><th class="left">Hash</th><th>榜单年化</th><th>Vault 年化</th><th>Δ</th><th>Vault 夏普</th><th>Vault RankIC</th><th>结论</th>
              </tr></thead>
              <tbody id="vaultBody"></tbody>
            </table>
          </div>
        </article>
        <article class="card">
          <div class="card-head" style="padding:0 0 10px">
            <div><h3>各维度第一名</h3><p>同一 Hash 可同时占据多个维度。</p></div>
          </div>
          <div class="champions" id="champions"></div>
        </article>
      </div>
    </section>

    <section class="section" id="audit">
      <div class="section-heading">
        <div><h2>可复现性与交割审计</h2><p>数据身份、分层边界、费用与逐笔账本均有独立证据。</p></div>
      </div>
      <div class="audit-grid">
        <article class="card">
          <div class="card-head" style="padding:0 0 10px"><div><h3>冻结研究链</h3><p>方向、排名与 Vault 严格分层。</p></div></div>
          <div class="timeline" id="timeline"></div>
        </article>
        <article class="card">
          <div class="card-head" style="padding:0 0 10px"><div><h3>前三名完整交割单</h3><p>批量摘要与明细账逐项一致。</p></div></div>
          <div class="ledger-list" id="ledgerList"></div>
        </article>
      </div>
      <article class="card" style="margin-top:12px">
        <div class="card-head" style="padding:0">
          <div><h3>审计附录</h3><p id="auditSummary">—</p></div>
          <span class="badge ok">hash verified</span>
        </div>
        <details>
          <summary>查看 6 个退化表达式</summary>
          <div class="invalid-list" id="invalidList"></div>
        </details>
      </article>
    </section>

    <footer class="footer">
      <span>FactorFactory · A-share cross-task factor leaderboard</span>
      <span id="footerIdentity">—</span>
    </footer>
  </main>

  <div class="drawer-backdrop" id="drawerBackdrop"></div>
  <aside class="drawer" id="factorDrawer" aria-hidden="true" aria-label="因子详情">
    <div id="drawerContent"></div>
  </aside>

  <script id="report-data" type="application/json">__REPORT_DATA__</script>
  <script>
  (() => {
    "use strict";
    const report = JSON.parse(document.getElementById("report-data").textContent);
    const rows = report.results;
    const byHash = new Map(rows.map(row => [row.expression_hash, row]));
    const dimensionLabels = {
      robust_score: "综合稳健分",
      ann_return_bps_0: "0bp 年化收益",
      ann_return_bps_5: "5bp 年化收益",
      ann_return_bps_15: "15bp 年化收益",
      sharpe_bps_0: "0bp 夏普",
      sharpe_bps_5: "5bp 夏普",
      sharpe_bps_15: "15bp 夏普",
      oos_ic_mean: "Pearson IC",
      oos_icir: "Pearson ICIR",
      oos_rank_ic_mean: "RankIC",
      oos_rank_icir: "RankICIR"
    };
    let filtered = [];
    let page = 1;
    let pageSize = 50;
    let sortKey = "overall_rank";
    let sortDirection = 1;
    let selectedHash = rows[0]?.expression_hash || "";
    let scatterPoints = [];

    const $ = id => document.getElementById(id);
    const finite = value => Number.isFinite(Number(value));
    const number = (value, digits=3) => finite(value) ? Number(value).toFixed(digits) : "—";
    const percent = (value, digits=2) => finite(value) ? `${(Number(value) * 100).toFixed(digits)}%` : "—";
    const integer = value => finite(value) ? Number(value).toLocaleString("zh-CN") : "—";
    const signed = (value, digits=2) => finite(value) ? `${Number(value) >= 0 ? "+" : ""}${(Number(value) * 100).toFixed(digits)}%` : "—";
    const escapeHtml = value => String(value ?? "").replace(/[&<>"']/g, char => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;"
    })[char]);
    const tone = value => !finite(value) ? "neutral" : Number(value) > 0 ? "positive" : Number(value) < 0 ? "negative" : "neutral";
    const shortHash = value => String(value || "").slice(0, 8);
    const metricValue = (dimension, row) => {
      const value = row?.[dimension];
      if (dimension.includes("return")) return percent(value);
      if (dimension === "robust_score") return number(value, 2);
      return number(value, dimension.endsWith("_mean") ? 4 : 3);
    };
    const classification = row => {
      if (!row || row.vault_status !== "ok") return ["未开封", "neutral"];
      const ann = Number(row.vault_ann_return_bps_15);
      const ric = Number(row.vault_rank_ic_mean);
      const delta = ann - Number(row.ann_return_bps_15);
      if (ann <= 0 || ric <= 0) return ["失效", "bad"];
      if (delta < -.05) return ["显著衰减", "warn"];
      if (ann > 0 && ric > 0) return ["生存", "good"];
      return ["观察", "neutral"];
    };

    function renderHeader() {
      $("heroProtocol").textContent = `protocol ${report.manifest.protocol}`;
      $("heroPanel").textContent = `panel ${report.protocol.panel_identity.slice(0, 12)}`;
      $("footerIdentity").textContent = `panel ${report.protocol.panel_identity.slice(0, 16)} · ${report.report_protocol}`;
      const kpis = [
        ["来源记录", report.snapshot.source_rows, "任务 1–8，节点 + 因子"],
        ["可执行因子", report.snapshot.valid_expressions, `${report.snapshot.invalid_expressions} 个退化式隔离`],
        ["经济评估组", report.manifest.economic_equivalence_groups, "相同评估结果合并排名"],
        ["实战硬筛", report.manifest.practical_pass, "成本、IC、BH q 同时通过"],
        ["Vault 复核", report.manifest.vault_opened_after_ranking, "总榜前 20 + 维度冠军"],
        ["运行失败", report.manifest.failed_expressions, `${report.manifest.artifact_count} 个产物哈希`, true]
      ];
      $("kpis").innerHTML = kpis.map(([label, value, foot, accent]) => `
        <article class="kpi ${accent ? "accent" : ""}">
          <div class="kpi-label">${escapeHtml(label)}</div>
          <div class="kpi-value">${integer(value)}</div>
          <div class="kpi-foot">${escapeHtml(foot)}</div>
        </article>`).join("");
    }

    function renderInsights() {
      const champion = byHash.get(report.finalists.dimension_champions.robust_score) || rows[0];
      const returnChampion = byHash.get(report.finalists.dimension_champions.ann_return_bps_15);
      const icirChampion = byHash.get(report.finalists.dimension_champions.oos_icir);
      const rankIcirChampion = byHash.get(report.finalists.dimension_champions.oos_rank_icir);
      const returnClass = classification(returnChampion);
      const rankClass = classification(rankIcirChampion);
      $("insights").innerHTML = `
        <article class="card insight-card primary">
          <div class="insight-title">Balanced champion · 综合第一</div>
          <h3>#${champion.overall_rank} · ${escapeHtml(champion.expression_hash)}</h3>
          <code class="formula">${escapeHtml(champion.expression)}</code>
          <div class="metric-row">
            <div class="metric-chip"><span>15bp 年化</span><b>${percent(champion.ann_return_bps_15)}</b></div>
            <div class="metric-chip"><span>15bp 夏普</span><b>${number(champion.sharpe_bps_15)}</b></div>
            <div class="metric-chip"><span>RankIC</span><b>${number(champion.oos_rank_ic_mean, 4)}</b></div>
            <div class="metric-chip"><span>Vault 年化</span><b>${percent(champion.vault_ann_return_bps_15)}</b></div>
          </div>
        </article>
        <article class="card insight-card">
          <div class="insight-title">Cost champion · 单项高收益</div>
          <h3>${escapeHtml(returnChampion.expression_hash)}</h3>
          <p>15bp 榜单期年化 <strong>${percent(returnChampion.ann_return_bps_15)}</strong>，但 Vault 年化降至 <strong>${percent(returnChampion.vault_ann_return_bps_15)}</strong>，Pearson IC 为 ${number(returnChampion.vault_ic_mean, 4)}。</p>
          <span class="signal ${returnClass[1]}">${returnClass[0]} · 不替代综合排名</span>
        </article>
        <article class="card insight-card">
          <div class="insight-title">Structure · 因子簇集中</div>
          <h3>前列集中在低振幅 / 低波动</h3>
          <p>总榜前十多数只是同一经济机制的非线性或窗口变体，不能把十行榜单当成十个独立 Alpha。组合前应再做相关性聚类和风险暴露残差化。</p>
          <span class="signal warn">避免重复下注</span>
        </article>
        <article class="card insight-card">
          <div class="insight-title">ICIR champion · 指标陷阱</div>
          <h3>${escapeHtml(icirChampion.expression_hash)}</h3>
          <p>榜单期 ICIR ${number(icirChampion.oos_icir)}，但 15bp 年化 ${percent(icirChampion.ann_return_bps_15)}，Vault 年化 ${percent(icirChampion.vault_ann_return_bps_15)}。</p>
          <span class="signal bad">IC 高不等于可交易</span>
        </article>
        <article class="card insight-card">
          <div class="insight-title">RankICIR champion · 时期敏感</div>
          <h3>${escapeHtml(rankIcirChampion.expression_hash)}</h3>
          <p>榜单期 15bp 年化仅 ${percent(rankIcirChampion.ann_return_bps_15)}，Vault 却为 ${percent(rankIcirChampion.vault_ann_return_bps_15)}，跨期差异很大。</p>
          <span class="signal ${rankClass[1]}">需做时期与暴露归因</span>
        </article>`;
    }

    function applyFilters() {
      const query = $("searchInput").value.trim().toLowerCase();
      const origin = $("originFilter").value;
      const pass = $("passFilter").value;
      const vault = $("vaultFilter").value;
      const representativeOnly = $("representativeOnly").checked;
      filtered = rows.filter(row => {
        if (representativeOnly && !row.economic_representative) return false;
        if (origin && row.origin_scope !== origin) return false;
        if (pass === "pass" && !row.practical_pass) return false;
        if (pass === "fail" && row.practical_pass) return false;
        if (vault && row.vault_status !== vault) return false;
        if (query) {
          const haystack = [
            row.expression_hash, row.expression, row.origin_scope,
            ...(row.fields || []), ...(row.operators || []),
            ...(row.experiment_ids || [])
          ].join(" ").toLowerCase();
          if (!haystack.includes(query)) return false;
        }
        return true;
      });
      filtered.sort((a, b) => {
        const av = a[sortKey];
        const bv = b[sortKey];
        if (av == null && bv == null) return 0;
        if (av == null) return 1;
        if (bv == null) return -1;
        if (typeof av === "string" || typeof bv === "string") {
          return String(av).localeCompare(String(bv)) * sortDirection;
        }
        return (Number(av) - Number(bv)) * sortDirection;
      });
      page = 1;
      renderTable();
    }

    function stack(values, formatter, classLast=false) {
      return `<div class="metric-stack">${values.map((value, index) =>
        `<span class="${classLast && index === values.length - 1 ? "stress" : ""}">${formatter(value)}</span>`
      ).join("")}</div>`;
    }

    function renderTable() {
      const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
      page = Math.min(page, totalPages);
      const start = (page - 1) * pageSize;
      const slice = filtered.slice(start, start + pageSize);
      $("leaderboardBody").innerHTML = slice.map(row => {
        const vaultText = row.vault_status === "ok"
          ? `<span class="${tone(row.vault_ann_return_bps_15)}">${percent(row.vault_ann_return_bps_15)}</span><br><span class="neutral">${number(row.vault_sharpe_bps_15)}</span>`
          : `<span class="neutral">未开封</span>`;
        const representativeTag = row.economic_representative
          ? ""
          : `<span class="signal warn">重复 ${escapeHtml(shortHash(row.economic_duplicate_of))}</span>`;
        return `<tr data-hash="${escapeHtml(row.expression_hash)}" class="${row.expression_hash === selectedHash ? "selected" : ""}">
          <td><span class="rank ${row.overall_rank <= 10 ? "top" : ""}">#${integer(row.overall_rank)}</span></td>
          <td class="left factor-cell">
            <span class="hash">${escapeHtml(row.expression_hash)}</span> ${representativeTag}
            <span class="expr" title="${escapeHtml(row.expression)}">${escapeHtml(row.expression)}</span>
          </td>
          <td><span class="pill origin">${escapeHtml(row.origin_scope)}</span></td>
          <td class="${row.direction < 0 ? "negative" : "positive"}">${row.direction > 0 ? "+1" : "−1"}</td>
          <td>${number(row.robust_score, 2)}</td>
          <td>${stack([row.ann_return_bps_0,row.ann_return_bps_5,row.ann_return_bps_15], percent, true)}</td>
          <td>${stack([row.sharpe_bps_0,row.sharpe_bps_5,row.sharpe_bps_15], value => number(value), true)}</td>
          <td><span class="${tone(row.oos_ic_mean)}">${number(row.oos_ic_mean,4)}</span><br><span class="neutral">${number(row.oos_icir)}</span></td>
          <td><span class="${tone(row.oos_rank_ic_mean)}">${number(row.oos_rank_ic_mean,4)}</span><br><span class="neutral">${number(row.oos_rank_icir)}</span></td>
          <td>${vaultText}</td>
          <td><span class="signal ${row.practical_pass ? "good" : "bad"}">${row.practical_pass ? "PASS" : "FAIL"}</span></td>
        </tr>`;
      }).join("");
      $("tableCount").textContent = `${integer(filtered.length)} / ${integer(rows.length)} rows`;
      $("filterSummary").textContent = `显示 ${integer(start + 1)}–${integer(Math.min(start + pageSize, filtered.length))}，筛选后 ${integer(filtered.length)} 条`;
      $("pageLabel").textContent = `${page} / ${totalPages}`;
      $("prevPage").disabled = page <= 1;
      $("nextPage").disabled = page >= totalPages;
      document.querySelectorAll("#leaderboardBody tr").forEach(tr => {
        tr.addEventListener("click", () => openDrawer(tr.dataset.hash));
      });
    }

    function openDrawer(hash) {
      const row = byHash.get(hash);
      if (!row) return;
      selectedHash = hash;
      const [statusText, statusClass] = classification(row);
      const experiments = (row.experiment_ids || []).map(id => {
        const experiment = report.experiments.find(item => item.id === id);
        return experiment ? `#${id} ${experiment.name}` : `#${id}`;
      });
      const tags = [
        `来源 ${row.origin_scope}`, `方向 ${row.direction > 0 ? "+1" : "−1"}`,
        `历史 ${row.required_history}d`, `复杂度 ${row.complexity}`,
        `组大小 ${row.equivalence_group_size}`
      ];
      $("drawerContent").innerHTML = `
        <div class="drawer-top">
          <div>
            <div class="eyebrow">Factor detail · #${integer(row.overall_rank)}</div>
            <h2>${escapeHtml(row.expression_hash)}</h2>
            <span class="signal ${row.practical_pass ? "good" : "bad"}">${row.practical_pass ? "PRACTICAL PASS" : "SCREENED OUT"}</span>
            <span class="signal ${statusClass}">${statusText}</span>
          </div>
          <button class="btn drawer-close" type="button" aria-label="关闭详情">×</button>
        </div>
        <div class="drawer-section">
          <h3>DSL expression</h3>
          <code class="formula drawer-formula">${escapeHtml(row.expression)}</code>
          <div class="latex">${escapeHtml(row.latex)}</div>
        </div>
        <div class="drawer-section">
          <h3>核心指标</h3>
          <div class="metric-grid">
            ${[
              ["稳健分", number(row.robust_score,2)],
              ["15bp 年化", percent(row.ann_return_bps_15)],
              ["15bp 夏普", number(row.sharpe_bps_15)],
              ["最大回撤", percent(row.max_drawdown_bps_15)],
              ["IC", number(row.oos_ic_mean,4)],
              ["ICIR", number(row.oos_icir)],
              ["RankIC", number(row.oos_rank_ic_mean,4)],
              ["RankICIR", number(row.oos_rank_icir)]
            ].map(([label,value]) => `<div class="detail-metric"><span>${label}</span><b>${value}</b></div>`).join("")}
          </div>
        </div>
        <div class="drawer-section">
          <h3>训练 → 榜单期 → Vault</h3>
          <table class="layer-table">
            <thead><tr><th class="left">层</th><th>年化 15bp</th><th>夏普</th><th>IC</th><th>ICIR</th><th>RankIC</th><th>RankICIR</th></tr></thead>
            <tbody>
              <tr><td class="left">META_TRAIN</td><td>—</td><td>—</td><td>${number(row.train_ic_mean,4)}</td><td>${number(row.train_icir)}</td><td>${number(row.train_rank_ic_mean,4)}</td><td>${number(row.train_rank_icir)}</td></tr>
              <tr><td class="left">META_HOLDOUT</td><td>${percent(row.ann_return_bps_15)}</td><td>${number(row.sharpe_bps_15)}</td><td>${number(row.oos_ic_mean,4)}</td><td>${number(row.oos_icir)}</td><td>${number(row.oos_rank_ic_mean,4)}</td><td>${number(row.oos_rank_icir)}</td></tr>
              <tr><td class="left">FACTOR_VAULT</td><td>${percent(row.vault_ann_return_bps_15)}</td><td>${number(row.vault_sharpe_bps_15)}</td><td>${number(row.vault_ic_mean,4)}</td><td>${number(row.vault_icir)}</td><td>${number(row.vault_rank_ic_mean,4)}</td><td>${number(row.vault_rank_icir)}</td></tr>
            </tbody>
          </table>
        </div>
        <div class="drawer-section">
          <h3>执行质量（15bp）</h3>
          <div class="metric-grid">
            ${[
              ["日均换手", percent(row.avg_daily_turnover_bps_15)],
              ["成交率", percent(row.fill_rate_bps_15)],
              ["成交笔数", integer(row.fills_bps_15)],
              ["总执行成本", finite(row.total_execution_cost_bps_15) ? `¥${integer(Math.round(row.total_execution_cost_bps_15))}` : "—"],
              ["费税", finite(row.commission_and_tax_bps_15) ? `¥${integer(Math.round(row.commission_and_tax_bps_15))}` : "—"],
              ["滑点", finite(row.slippage_cost_bps_15) ? `¥${integer(Math.round(row.slippage_cost_bps_15))}` : "—"],
              ["RankIC BH q", number(row.oos_rank_ic_bh_q,6)],
              ["来源记录", integer(row.source_record_count)]
            ].map(([label,value]) => `<div class="detail-metric"><span>${label}</span><b>${value}</b></div>`).join("")}
          </div>
        </div>
        <div class="drawer-section">
          <h3>结构与来源</h3>
          <div class="tags">${tags.map(tag => `<span class="pill">${escapeHtml(tag)}</span>`).join("")}</div>
          <p class="neutral" style="font-size:10px;margin-top:9px">字段：${escapeHtml((row.fields || []).join(", ") || "—")}<br>算子：${escapeHtml((row.operators || []).join(", ") || "—")}<br>实验：${escapeHtml(experiments.join(" · ") || "—")}</p>
        </div>`;
      document.querySelector(".drawer-close").addEventListener("click", closeDrawer);
      $("factorDrawer").classList.add("open");
      $("factorDrawer").setAttribute("aria-hidden", "false");
      $("drawerBackdrop").classList.add("open");
      renderTable();
      drawCostChart();
    }

    function closeDrawer() {
      $("factorDrawer").classList.remove("open");
      $("factorDrawer").setAttribute("aria-hidden", "true");
      $("drawerBackdrop").classList.remove("open");
    }

    function renderVault() {
      const vaultRows = report.finalists.hashes.map(hash => byHash.get(hash)).filter(Boolean);
      $("vaultStatus").textContent = `${vaultRows.length} / ${vaultRows.length} integrity pass`;
      $("vaultBody").innerHTML = vaultRows.map(row => {
        const delta = Number(row.vault_ann_return_bps_15) - Number(row.ann_return_bps_15);
        const [label, style] = classification(row);
        return `<tr data-hash="${escapeHtml(row.expression_hash)}">
          <td>#${integer(row.overall_rank)}</td>
          <td class="left"><span class="hash">${escapeHtml(row.expression_hash)}</span></td>
          <td>${percent(row.ann_return_bps_15)}</td>
          <td class="${tone(row.vault_ann_return_bps_15)}">${percent(row.vault_ann_return_bps_15)}</td>
          <td class="${tone(delta)}">${signed(delta)}</td>
          <td>${number(row.vault_sharpe_bps_15)}</td>
          <td>${number(row.vault_rank_ic_mean,4)}</td>
          <td><span class="signal ${style}">${label}</span></td>
        </tr>`;
      }).join("");
      document.querySelectorAll("#vaultBody tr").forEach(tr => tr.addEventListener("click", () => openDrawer(tr.dataset.hash)));

      $("champions").innerHTML = Object.entries(report.finalists.dimension_champions).map(([dimension, hash]) => {
        const row = byHash.get(hash);
        return `<button class="champion" type="button" data-hash="${escapeHtml(hash)}">
          <span class="champion-label">${escapeHtml(dimensionLabels[dimension] || dimension)}</span>
          <span class="champion-hash">${escapeHtml(hash)}</span>
          <span class="champion-value">${metricValue(dimension, row)}</span>
        </button>`;
      }).join("");
      document.querySelectorAll(".champion").forEach(button => button.addEventListener("click", () => openDrawer(button.dataset.hash)));
    }

    function renderAudit() {
      const spec = report.protocol.spec;
      const rows = [
        ["META_TRAIN", "2020–2022 · 仅冻结正/反方向", "read-only direction"],
        ["META_HOLDOUT", "2023–2024 · 排名、IC 与 0/5/15bp", "ranking"],
        ["FREEZE", `${report.finalists.hashes.length} 个 Hash 在 Vault 前冻结`, "hash sealed"],
        ["FACTOR_VAULT", "2025–2026 · 不进入分数", "one-time"],
        ["LEDGER", `${report.manifest.finalist_ledgers} 个完整逐笔交割单`, "reconciled"]
      ];
      $("timeline").innerHTML = rows.map(([label, text, state]) => `
        <div class="timeline-row"><b>${label}</b><span>${text}</span><span class="signal good">${state}</span></div>`
      ).join("");
      $("ledgerList").innerHTML = Object.entries(report.ledgers).map(([hash, ledger]) => {
        const stats = ledger.stats || {};
        const integrity = ledger.integrity || {};
        return `<div class="ledger-row">
          <div class="ledger-row-head">
            <span class="hash">${escapeHtml(hash)}</span>
            <span class="signal ${integrity.all_pass ? "good" : "bad"}">${integrity.all_pass ? "ALL PASS" : "FAILED"}</span>
          </div>
          <div class="metrics">
            <div><span>成交</span><b>${integer(integrity.statement_rows)}</b></div>
            <div><span>15bp 年化</span><b>${percent(stats.ann_ret)}</b></div>
            <div><span>最大回撤</span><b>${percent(stats.max_dd)}</b></div>
            <div><span>成交率</span><b>${percent(stats.fill_rate)}</b></div>
          </div>
          <div class="mini-links" style="margin-top:9px">
            <a href="finalist_ledgers/${escapeHtml(hash)}/settlement_statement.csv">CSV 交割单</a>
            <a href="finalist_ledgers/${escapeHtml(hash)}/manifest.json">子 Manifest</a>
          </div>
        </div>`;
      }).join("");
      $("auditSummary").textContent =
        `${report.manifest.artifact_count} 个源产物哈希；${report.manifest.successful_expressions} 个结果，${report.manifest.failed_expressions} 个运行失败；` +
        `A股万二免五，卖出印花税与双向过户费，额外滑点 0/5/15 BPS；初始资金 CNY ${integer(spec.initial_capital)}。`;
      $("invalidList").innerHTML = report.invalid.map(row => `
        <div class="invalid-row">
          <span class="signal bad">${escapeHtml(row.validation_error)}</span>
          <code>${escapeHtml(row.expression)}</code>
        </div>`).join("");
    }

    function setupCanvas(canvas) {
      const rect = canvas.getBoundingClientRect();
      const ratio = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.round(rect.width * ratio);
      canvas.height = Math.round(rect.height * ratio);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return {ctx, width: rect.width, height: rect.height};
    }

    function drawAxes(ctx, width, height, bounds, labels) {
      const pad = {left:54, right:18, top:18, bottom:36};
      const plotW = width - pad.left - pad.right;
      const plotH = height - pad.top - pad.bottom;
      ctx.clearRect(0,0,width,height);
      ctx.strokeStyle = "#273341";
      ctx.fillStyle = "#778493";
      ctx.lineWidth = 1;
      ctx.font = "9px SFMono-Regular, Menlo, monospace";
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      for (let i=0;i<=4;i++) {
        const y = pad.top + plotH * i / 4;
        const value = bounds.yMax - (bounds.yMax-bounds.yMin)*i/4;
        ctx.beginPath(); ctx.moveTo(pad.left,y); ctx.lineTo(width-pad.right,y); ctx.stroke();
        ctx.fillText(labels.y(value), pad.left-7, y);
      }
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      for (let i=0;i<=4;i++) {
        const x = pad.left + plotW * i / 4;
        const value = bounds.xMin + (bounds.xMax-bounds.xMin)*i/4;
        ctx.beginPath(); ctx.moveTo(x,pad.top); ctx.lineTo(x,height-pad.bottom); ctx.stroke();
        ctx.fillText(labels.x(value), x, height-pad.bottom+9);
      }
      return {
        pad, plotW, plotH,
        x: value => pad.left + (value-bounds.xMin)/(bounds.xMax-bounds.xMin)*plotW,
        y: value => pad.top + (bounds.yMax-value)/(bounds.yMax-bounds.yMin)*plotH
      };
    }

    function drawScatter() {
      const canvas = $("scatterCanvas");
      const {ctx,width,height} = setupCanvas(canvas);
      const source = rows.filter(row => row.economic_representative && finite(row.oos_rank_ic_mean) && finite(row.ann_return_bps_15));
      const xs = source.map(row => Number(row.oos_rank_ic_mean));
      const ys = source.map(row => Number(row.ann_return_bps_15));
      const bounds = {
        xMin: Math.min(-.01, ...xs), xMax: Math.max(.12, ...xs),
        yMin: Math.min(-.35, ...ys), yMax: Math.max(.16, ...ys)
      };
      const axes = drawAxes(ctx,width,height,bounds,{x:value=>value.toFixed(2),y:value=>`${(value*100).toFixed(0)}%`});
      scatterPoints = source.map(row => ({
        row,
        x: axes.x(Number(row.oos_rank_ic_mean)),
        y: axes.y(Number(row.ann_return_bps_15))
      }));
      scatterPoints.forEach(point => {
        const row = point.row;
        const vault = row.vault_status === "ok";
        ctx.beginPath();
        ctx.arc(point.x, point.y, row.overall_rank <= 20 ? 3.2 : 2.1, 0, Math.PI*2);
        ctx.fillStyle = row.practical_pass ? "rgba(63,185,80,.62)" : "rgba(111,126,143,.30)";
        ctx.fill();
        if (vault || row.expression_hash === selectedHash) {
          ctx.strokeStyle = row.expression_hash === selectedHash ? "#79c0ff" : "rgba(88,166,255,.70)";
          ctx.lineWidth = row.expression_hash === selectedHash ? 2 : 1;
          ctx.stroke();
        }
      });
      $("scatterCount").textContent = `${source.length} economic groups`;
    }

    function drawCostChart() {
      const canvas = $("costCanvas");
      const {ctx,width,height} = setupCanvas(canvas);
      const selected = byHash.get(selectedHash);
      const chartRows = [...rows.filter(row => row.economic_representative).slice(0,5)];
      if (selected && !chartRows.some(row => row.expression_hash === selected.expression_hash)) chartRows.push(selected);
      const values = chartRows.flatMap(row => [row.ann_return_bps_0,row.ann_return_bps_5,row.ann_return_bps_15]).filter(finite).map(Number);
      const bounds = {xMin:0,xMax:15,yMin:Math.min(0,...values)-.015,yMax:Math.max(.15,...values)+.015};
      const axes = drawAxes(ctx,width,height,bounds,{x:value=>`${value.toFixed(0)}bp`,y:value=>`${(value*100).toFixed(0)}%`});
      const palette = ["#58a6ff","#3fb950","#d9a441","#b392f0","#4ec9b0","#ff6b63"];
      chartRows.forEach((row,index) => {
        const points = [[0,row.ann_return_bps_0],[5,row.ann_return_bps_5],[15,row.ann_return_bps_15]];
        ctx.beginPath();
        points.forEach(([x,y],pointIndex) => {
          const px=axes.x(x), py=axes.y(Number(y));
          if (pointIndex===0) ctx.moveTo(px,py); else ctx.lineTo(px,py);
        });
        ctx.strokeStyle = row.expression_hash === selectedHash ? "#ffffff" : palette[index % palette.length];
        ctx.lineWidth = row.expression_hash === selectedHash ? 2.5 : 1.35;
        ctx.globalAlpha = row.expression_hash === selectedHash ? 1 : .78;
        ctx.stroke();
        ctx.globalAlpha = 1;
        points.forEach(([x,y]) => {
          ctx.beginPath(); ctx.arc(axes.x(x),axes.y(Number(y)),2.6,0,Math.PI*2);
          ctx.fillStyle = row.expression_hash === selectedHash ? "#ffffff" : palette[index % palette.length];
          ctx.fill();
        });
        ctx.fillStyle = row.expression_hash === selectedHash ? "#ffffff" : palette[index % palette.length];
        ctx.font = "9px SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "left";
        ctx.fillText(`#${row.overall_rank}`, axes.x(15)+5, axes.y(Number(row.ann_return_bps_15)));
      });
    }

    function nearestScatter(event) {
      const rect = $("scatterCanvas").getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      let nearest = null;
      let best = 90;
      scatterPoints.forEach(point => {
        const distance = (point.x-x)**2 + (point.y-y)**2;
        if (distance < best) { best = distance; nearest = point; }
      });
      return nearest;
    }

    function exportCsv() {
      const columns = [
        "overall_rank","expression_hash","origin_scope","direction","practical_pass","robust_score",
        "ann_return_bps_0","ann_return_bps_5","ann_return_bps_15",
        "sharpe_bps_0","sharpe_bps_5","sharpe_bps_15",
        "oos_ic_mean","oos_icir","oos_rank_ic_mean","oos_rank_icir",
        "vault_ann_return_bps_15","vault_sharpe_bps_15","expression"
      ];
      const csvCell = value => `"${String(value ?? "").replaceAll('"','""')}"`;
      const content = [columns.join(","), ...filtered.map(row => columns.map(key => csvCell(row[key])).join(","))].join("\n");
      const blob = new Blob(["\ufeff",content],{type:"text/csv;charset=utf-8"});
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = "factor_leaderboard_filtered.csv";
      link.click();
      setTimeout(() => URL.revokeObjectURL(link.href), 1000);
    }

    function bindEvents() {
      ["searchInput","originFilter","passFilter","vaultFilter","representativeOnly"].forEach(id => {
        $(id).addEventListener(id === "searchInput" ? "input" : "change", applyFilters);
      });
      $("pageSize").addEventListener("change", event => { pageSize=Number(event.target.value); page=1; renderTable(); });
      $("prevPage").addEventListener("click", () => { if (page>1) { page--; renderTable(); } });
      $("nextPage").addEventListener("click", () => { if (page*pageSize<filtered.length) { page++; renderTable(); } });
      $("exportButton").addEventListener("click", exportCsv);
      $("drawerBackdrop").addEventListener("click", closeDrawer);
      document.addEventListener("keydown", event => { if (event.key === "Escape") closeDrawer(); });
      document.querySelectorAll("th[data-sort]").forEach(th => th.addEventListener("click", () => {
        const next = th.dataset.sort;
        if (sortKey === next) sortDirection *= -1;
        else { sortKey=next; sortDirection = ["overall_rank","expression_hash","origin_scope"].includes(next) ? 1 : -1; }
        applyFilters();
      }));
      $("scatterCanvas").addEventListener("mousemove", event => {
        const point = nearestScatter(event);
        const tip = $("scatterTip");
        if (!point) { tip.style.display="none"; return; }
        tip.style.display="block";
        tip.style.left=`${Math.min(event.offsetX+14,$("scatterCanvas").clientWidth-250)}px`;
        tip.style.top=`${Math.max(8,event.offsetY-54)}px`;
        tip.innerHTML=`#${point.row.overall_rank} ${escapeHtml(point.row.expression_hash)}<br>RankIC ${number(point.row.oos_rank_ic_mean,4)} · 15bp ${percent(point.row.ann_return_bps_15)}`;
      });
      $("scatterCanvas").addEventListener("mouseleave", () => $("scatterTip").style.display="none");
      $("scatterCanvas").addEventListener("click", event => {
        const point = nearestScatter(event);
        if (point) openDrawer(point.row.expression_hash);
      });
      const redraw = () => { drawScatter(); drawCostChart(); };
      if ("ResizeObserver" in window) {
        const observer = new ResizeObserver(redraw);
        observer.observe($("scatterCanvas").parentElement);
        observer.observe($("costCanvas").parentElement);
      } else {
        window.addEventListener("resize", redraw);
      }
    }

    renderHeader();
    renderInsights();
    renderVault();
    renderAudit();
    bindEvents();
    applyFilters();
    requestAnimationFrame(() => { drawScatter(); drawCostChart(); });
  })();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
