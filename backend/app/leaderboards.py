"""Discover and safely serve immutable factor-leaderboard reports."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .dsl.engine import expression_to_latex, expression_profile
from .factors.economics import explain_factor_economics


_ACTIVE_REPORTS = Path("var/reports")
_ARCHIVE_REPORTS = Path("var/archives/factor-leaderboards")
_ALLOWED_REPORT_SUFFIXES = {
    ".csv",
    ".html",
    ".json",
    ".md",
    ".parquet",
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _report_directories(project_root: Path) -> list[tuple[Path, str, str | None]]:
    candidates: list[tuple[Path, str, str | None]] = []
    active_root = project_root / _ACTIVE_REPORTS
    if active_root.is_dir():
        for report_dir in sorted(active_root.iterdir()):
            if report_dir.is_dir() and (report_dir / "leaderboard.html").is_file():
                candidates.append((report_dir, "active", None))

    archive_root = project_root / _ARCHIVE_REPORTS
    if archive_root.is_dir():
        for snapshot_dir in sorted(archive_root.iterdir(), reverse=True):
            reports_root = snapshot_dir / "reports"
            if not reports_root.is_dir():
                continue
            for report_dir in sorted(reports_root.iterdir()):
                if report_dir.is_dir() and (report_dir / "leaderboard.html").is_file():
                    candidates.append((report_dir, "archive", snapshot_dir.name))
    return candidates


def _html_identity(report_dir: Path, leaderboard_manifest: dict[str, Any]) -> str:
    html_meta = leaderboard_manifest.get("html")
    if isinstance(html_meta, dict) and html_meta.get("sha256"):
        return str(html_meta["sha256"])
    html_path = report_dir / "leaderboard.html"
    digest = hashlib.sha256()
    with html_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version_copy(
    project_root: Path,
    report_dir: Path,
    location: str,
    archive_snapshot: str | None,
) -> dict[str, Any]:
    return {
        "location": location,
        "archive_snapshot": archive_snapshot,
        "relative_path": str(report_dir.relative_to(project_root)),
    }


def _protocol_presentation(protocol_id: str) -> tuple[str, str, str]:
    if protocol_id == "dual_direction_full_window_vector_screen_leaderboard_v1":
        return (
            "全区间双向榜",
            "full_window_vector",
            "2020 年至报告截止日同时评价正反方向，先做 0/5/15bps 向量筛选，"
            "再对独立前列组合做逐笔事件账本复核。",
        )
    if "cross_task" in protocol_id and "leaderboard" in protocol_id:
        return (
            "样本外事件榜",
            "holdout_event",
            "训练期冻结方向，2023–2024 榜单期使用步进事件引擎；"
            "2025 年后的 Vault 复核不参与总榜排序。",
        )
    return (
        protocol_id or "未标记协议",
        "other",
        "请打开协议与审计区确认样本划分、方向、费用和执行口径。",
    )


def _report_record(
    project_root: Path,
    report_dir: Path,
    location: str,
    archive_snapshot: str | None,
) -> tuple[str, dict[str, Any]]:
    protocol = _read_json(report_dir / "protocol.json")
    leaderboard_manifest = _read_json(report_dir / "leaderboard_manifest.json")
    progress = _read_json(report_dir / "progress.json")
    manifest = _read_json(report_dir / "manifest.json")
    spec = protocol.get("spec") if isinstance(protocol.get("spec"), dict) else {}

    html_sha256 = _html_identity(report_dir, leaderboard_manifest)
    report_id = html_sha256[:24]
    protocol_id = str(protocol.get("protocol") or manifest.get("protocol") or "")
    protocol_label, version_kind, description = _protocol_presentation(protocol_id)
    market = str(protocol.get("target_market") or spec.get("market") or "unknown")
    mode = str(protocol.get("portfolio_mode") or spec.get("mode") or "unknown")
    market_label = {"ashare": "A股", "us": "美股"}.get(market, market)
    mode_label = {"long_only": "纯多", "long_short": "多空"}.get(mode, mode)

    ranking_window = protocol.get("ranking_window")
    if not isinstance(ranking_window, dict):
        ranking_window = {
            "start": spec.get("holdout_start"),
            "end": spec.get("holdout_end"),
            "semantics": "independent_holdout" if version_kind == "holdout_event" else None,
        }
    generated_at = leaderboard_manifest.get("generated_at")
    if not generated_at:
        generated_at = datetime.fromtimestamp(
            (report_dir / "leaderboard.html").stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()

    completed = float(progress.get("percent") or 0) >= 100.0
    errors = int(progress.get("errors") or progress.get("failed_orientations") or 0)
    has_vault = (report_dir / "vault_finalists.json").is_file() and (
        report_dir / "vault_finalists.json"
    ).stat().st_size > 3
    has_event_replay = (report_dir / "finalist_ledger_audits.json").is_file()
    screening_only = bool(protocol.get("screening_only"))
    policy_label = str(protocol.get("policy_label") or "NON_PIT_RESEARCH")

    record = {
        "id": report_id,
        "directory_name": report_dir.name,
        "title": f"{market_label} · {mode_label} · {protocol_label}",
        "market": market,
        "market_label": market_label,
        "mode": mode,
        "mode_label": mode_label,
        "protocol": protocol_id,
        "protocol_label": protocol_label,
        "version_kind": version_kind,
        "description": description,
        "generated_at": str(generated_at),
        "ranking_window": ranking_window,
        "data_end": (
            protocol.get("panel_date_max")
            or spec.get("vault_end")
            or ranking_window.get("end")
        ),
        "policy_label": policy_label,
        "production_eligible": bool(protocol.get("production_eligible", False)),
        "screening_only": screening_only,
        "sample_is_out_of_sample": (
            False if screening_only else version_kind == "holdout_event"
        ),
        "direction_policy": (
            protocol.get("direction_selection")
            or ("train_frozen" if version_kind == "holdout_event" else "unspecified")
        ),
        "result_count": int(leaderboard_manifest.get("embedded_results") or 0),
        "status": "complete" if completed and errors == 0 else "completed_with_gaps" if completed else "incomplete",
        "progress_percent": float(progress.get("percent") or 0),
        "errors": errors,
        "has_vault": has_vault,
        "has_event_replay": has_event_replay,
        "html_sha256": html_sha256,
        "file_url": f"/api/leaderboards/{report_id}/files/leaderboard.html",
        "quick_links": [
            {"id": "overview", "label": "结论"},
            {"id": "charts", "label": "分布"},
            {"id": "leaderboard", "label": "全量榜单"},
            {"id": "vault", "label": "Vault / 事件复核"},
            {"id": "audit", "label": "审计"},
        ],
        "copies": [
            _version_copy(project_root, report_dir, location, archive_snapshot)
        ],
    }
    return html_sha256, record


def build_leaderboard_catalog(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return logical report versions, deduplicating archived byte-identical copies."""
    project_root = project_root.resolve()
    by_identity: dict[str, dict[str, Any]] = {}
    archive_copy_count = 0
    for report_dir, location, archive_snapshot in _report_directories(project_root):
        identity, record = _report_record(
            project_root,
            report_dir,
            location,
            archive_snapshot,
        )
        if location == "archive":
            archive_copy_count += 1
        existing = by_identity.get(identity)
        if existing is None:
            by_identity[identity] = record
            continue
        existing["copies"].extend(record["copies"])
        if existing["copies"][0]["location"] != "active" and location == "active":
            record["copies"].extend(existing["copies"][:-1])
            by_identity[identity] = record

    reports = list(by_identity.values())
    reports.sort(
        key=lambda item: (
            str(item.get("generated_at") or ""),
            str(item.get("title") or ""),
        ),
        reverse=True,
    )
    newest_by_scope: dict[tuple[str, str], str] = {}
    for report in reports:
        scope = (report["market"], report["mode"])
        newest_by_scope.setdefault(scope, report["id"])
    for report in reports:
        report["is_latest_for_scope"] = (
            newest_by_scope.get((report["market"], report["mode"])) == report["id"]
        )
        report["archive_copy_count"] = sum(
            copy["location"] == "archive" for copy in report["copies"]
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "logical_versions": len(reports),
            "active_versions": sum(
                any(copy["location"] == "active" for copy in report["copies"])
                for report in reports
            ),
            "archive_copies": archive_copy_count,
            "complete_versions": sum(report["status"] == "complete" for report in reports),
        },
        "reports": reports,
        "guidance": [
            {
                "title": "样本外事件榜",
                "text": "用于观察冻结方向后的 2023–2024 样本外执行与 2025+ Vault；优先看跨期稳定性。",
            },
            {
                "title": "全区间双向榜",
                "text": "用于诊断 2020 至截止日的长期方向与成本敏感度；全样本参与排名，不是独立样本外。",
            },
            {
                "title": "统一边界",
                "text": "所有现有报告均为 NON-PIT 研究产物；榜单名次、绿色状态或纯多收益都不等于实盘批准。",
            },
        ],
    }


def resolve_leaderboard_file(
    report_id: str,
    file_path: str,
    project_root: Path = PROJECT_ROOT,
) -> Path:
    """Resolve one catalogued report file without permitting path traversal."""
    project_root = project_root.resolve()
    report_dir: Path | None = None
    for candidate, location, archive_snapshot in _report_directories(project_root):
        del location, archive_snapshot
        identity = _html_identity(
            candidate,
            _read_json(candidate / "leaderboard_manifest.json"),
        )
        if identity[:24] == report_id:
            report_dir = candidate.resolve()
            if str(candidate).startswith(str(project_root / _ACTIVE_REPORTS)):
                break
    if report_dir is None:
        raise FileNotFoundError("leaderboard version not found")

    relative = Path(file_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("invalid leaderboard file path")
    if relative.suffix.lower() not in _ALLOWED_REPORT_SUFFIXES:
        raise ValueError("unsupported leaderboard file type")
    resolved = (report_dir / relative).resolve()
    if resolved != report_dir and report_dir not in resolved.parents:
        raise ValueError("leaderboard file escaped report directory")
    if not resolved.is_file():
        raise FileNotFoundError("leaderboard file not found")
    return resolved


def load_leaderboard_factor_detail(
    report_id: str,
    expression_hash: str,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    """Read one factor from the immutable report snapshot and explain it."""
    leaderboard_path = resolve_leaderboard_file(
        report_id,
        "leaderboard_full.json",
        project_root,
    )
    try:
        rows = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeError) as exc:
        raise ValueError("leaderboard snapshot is unreadable") from exc
    if not isinstance(rows, list):
        raise ValueError("leaderboard snapshot must be a list")
    row = next(
        (
            candidate
            for candidate in rows
            if isinstance(candidate, dict)
            and str(candidate.get("expression_hash") or "") == expression_hash
        ),
        None,
    )
    if row is None:
        raise FileNotFoundError("leaderboard factor not found")

    protocol_path = resolve_leaderboard_file(
        report_id,
        "protocol.json",
        project_root,
    )
    protocol = _read_json(protocol_path)
    spec = protocol.get("spec") if isinstance(protocol.get("spec"), dict) else {}
    market = str(protocol.get("target_market") or spec.get("market") or "ashare")
    portfolio_mode = str(
        protocol.get("portfolio_mode")
        or spec.get("mode")
        or ("long_only" if market == "ashare" else "long_short")
    )
    expression = str(row.get("expression") or "")
    try:
        profile = expression_profile(expression)
        latex = str(profile.get("latex") or expression_to_latex(expression))
    except (SyntaxError, TypeError, ValueError):
        profile = {
            "fields": list(row.get("fields") or []),
            "operators": list(row.get("operators") or []),
        }
        latex = expression
    fields = list(row.get("fields") or profile.get("fields") or [])
    operators = list(row.get("operators") or profile.get("operators") or [])
    direction = -1 if int(row.get("direction") or 1) < 0 else 1
    metrics = {
        key: row.get(key)
        for key in (
            "overall_rank",
            "practical_pass",
            "qualification_status",
            "absolute_quality_score",
            "robust_score",
            "portfolio_metric_basis",
            "ann_return_bps_0",
            "ann_return_bps_5",
            "ann_return_bps_15",
            "sharpe_bps_0",
            "sharpe_bps_5",
            "sharpe_bps_15",
            "oos_ic_mean",
            "oos_icir",
            "oos_rank_ic_mean",
            "oos_rank_icir",
            "vault_status",
            "vault_ann_return_bps_15",
            "vault_sharpe_bps_15",
            "event_replay_status",
            "event_ann_return_bps_15",
            "event_sharpe_bps_15",
        )
    }
    metric_basis = str(row.get("portfolio_metric_basis") or "total")
    if row.get("ranking_ann_return_bps_15") is not None:
        ranking_ann_return = row.get("ranking_ann_return_bps_15")
    elif metric_basis == "active":
        ranking_ann_return = row.get("active_ann_return_bps_15")
    else:
        ranking_ann_return = row.get("ann_return_bps_15")
    if row.get("ranking_sharpe_bps_15") is not None:
        ranking_sharpe = row.get("ranking_sharpe_bps_15")
    elif metric_basis == "active":
        ranking_sharpe = row.get("active_sharpe_bps_15")
    else:
        ranking_sharpe = row.get("sharpe_bps_15")
    metrics.update({
        "ranking_ann_return_bps_15": ranking_ann_return,
        "ranking_sharpe_bps_15": ranking_sharpe,
        "ranking_max_drawdown_bps_15": row.get(
            "ranking_max_drawdown_bps_15"
        ),
    })
    return {
        "report_id": report_id,
        "identity": {
            "expression_hash": expression_hash,
            "oriented_expression_hash": row.get("oriented_expression_hash"),
            "factor_ids": list(row.get("factor_ids") or []),
            "experiment_ids": list(row.get("experiment_ids") or []),
            "origin_scope": row.get("origin_scope"),
        },
        "expression": expression,
        "latex": latex,
        "direction": direction,
        "fields": fields,
        "operators": operators,
        "required_history": row.get(
            "required_history",
            profile.get("required_history"),
        ),
        "complexity": row.get("complexity", profile.get("complexity")),
        "metrics": metrics,
        "report": {
            "market": market,
            "portfolio_mode": portfolio_mode,
            "protocol": protocol.get("protocol"),
            "policy_label": protocol.get("policy_label", "NON_PIT_RESEARCH"),
            "ranking_window": protocol.get("ranking_window") or {},
            "screening_only": bool(protocol.get("screening_only")),
        },
        "economics": explain_factor_economics(
            expression,
            direction=direction,
            market=market,
            portfolio_mode=portfolio_mode,
            fields=fields,
            operators=operators,
            screening_only=bool(protocol.get("screening_only")),
        ),
    }
