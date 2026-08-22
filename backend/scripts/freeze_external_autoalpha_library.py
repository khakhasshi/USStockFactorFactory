"""Freeze a legacy AutoAlpha SQLite ``factor_pool`` for local leaderboards.

The source database is opened read-only and is never migrated or modified.
Structured AutoAlpha expression trees are translated losslessly into the
FactorFactory text DSL, validated against the actual target panel schema, and
AST-deduplicated while retaining every source factor ID and task provenance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from app.backtest.batch import canonical_expression  # noqa: E402
from app.config import ASHARE_PANEL_GLOB, US_PANEL_GLOB  # noqa: E402
from app.dsl.engine import expression_profile, validate  # noqa: E402


TRANSLATION_PROTOCOL = "autoalpha_expression_tree_to_factorfactory_dsl_v1"
SNAPSHOT_PROTOCOL = "external_autoalpha_factor_pool_snapshot_v1"
SOURCE_POLICY = "external_factor_pool_all"
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BINARY = {
    "add": "+",
    "subtract": "-",
    "multiply": "*",
    "divide": "/",
}
_ROLLING = {
    "rolling_mean": "ts_mean",
    "rolling_sum": "ts_sum",
    "rolling_std": "ts_std",
    "rolling_min": "ts_min",
    "rolling_max": "ts_max",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _number(value: Any, *, name: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    numeric = float(value)
    if not numeric == numeric or abs(numeric) == float("inf"):
        raise ValueError(f"{name} must be finite")
    return str(int(numeric)) if numeric.is_integer() else repr(numeric)


def translate_expression_tree(raw: dict[str, Any]) -> str:
    """Translate one canonical AutoAlpha expression tree without approximation."""
    if not isinstance(raw, dict):
        raise TypeError("expression node must be an object")
    operator = str(raw.get("operator") or "")
    arguments = raw.get("arguments") or []
    parameters = raw.get("parameters") or {}
    if not isinstance(arguments, list) or not isinstance(parameters, dict):
        raise TypeError("expression arguments/parameters have invalid types")

    if operator == "field":
        name = str(parameters.get("name") or "")
        if not _IDENTIFIER.fullmatch(name):
            raise ValueError(f"unsafe field name: {name!r}")
        return name
    if operator == "constant":
        return _number(parameters.get("value"), name="constant value")
    if operator in _BINARY:
        if len(arguments) != 2:
            raise ValueError(f"{operator} requires two arguments")
        left = translate_expression_tree(arguments[0])
        right = translate_expression_tree(arguments[1])
        return f"({left} {_BINARY[operator]} {right})"
    if operator == "negate":
        if len(arguments) != 1:
            raise ValueError("negate requires one argument")
        return f"(-{translate_expression_tree(arguments[0])})"
    if operator == "absolute":
        if len(arguments) != 1:
            raise ValueError("absolute requires one argument")
        return f"abs({translate_expression_tree(arguments[0])})"
    if operator in {"cs_rank", "cs_zscore"}:
        if len(arguments) != 1:
            raise ValueError(f"{operator} requires one argument")
        name = "rank" if operator == "cs_rank" else "zscore"
        return f"{name}({translate_expression_tree(arguments[0])})"
    if operator == "winsorize_mad":
        if len(arguments) != 1:
            raise ValueError("winsorize_mad requires one argument")
        threshold = _number(
            parameters.get("threshold", 3.0),
            name="winsorize_mad threshold",
        )
        return (
            f"winsor_mad({translate_expression_tree(arguments[0])}, "
            f"{threshold})"
        )
    if operator in {"delay", "delta", "returns"}:
        if len(arguments) != 1:
            raise ValueError(f"{operator} requires one argument")
        periods = _number(parameters.get("periods"), name=f"{operator} periods")
        name = "ts_delta" if operator == "delta" else operator
        return f"{name}({translate_expression_tree(arguments[0])}, {periods})"
    if operator in _ROLLING:
        if len(arguments) != 1:
            raise ValueError(f"{operator} requires one argument")
        window = _number(parameters.get("window"), name=f"{operator} window")
        return (
            f"{_ROLLING[operator]}("
            f"{translate_expression_tree(arguments[0])}, {window})"
        )
    raise ValueError(f"unsupported AutoAlpha operator: {operator}")


def _panel_fields(panel_glob: str) -> list[str]:
    return sorted(
        pl.scan_parquet(panel_glob, hive_partitioning=True)
        .collect_schema()
        .names()
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("expression_hash,expression,validation_error\n", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            })


def freeze_snapshot(
    *,
    database: Path,
    source_market: str,
    target_market: str,
    panel_glob: str,
    source_label: str,
) -> dict[str, Any]:
    database = database.resolve()
    before = database.stat()
    source_sha256 = _sha256(database)
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA query_only=ON")
        integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
        if integrity.lower() != "ok":
            raise ValueError(f"source SQLite integrity_check failed: {integrity}")
        rows = connection.execute(
            """
            SELECT factor_id, source_task_id, source_iteration, name, family,
                   proposal_json, metrics_json, status, status_reason,
                   created_at, updated_at
            FROM factor_pool
            ORDER BY factor_id
            """
        ).fetchall()
    finally:
        connection.close()
    after = database.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("source SQLite changed while the snapshot was read")

    panel_fields = _panel_fields(panel_glob)
    tasks = sorted({str(row["source_task_id"]) for row in rows})
    task_ids = {task: index for index, task in enumerate(tasks, start=1)}
    expressions: dict[str, dict[str, Any]] = {}
    invalid: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    source_field_counts: Counter[str] = Counter()

    for row in rows:
        factor_id = str(row["factor_id"])
        status_counts[str(row["status"])] += 1
        try:
            proposal = json.loads(str(row["proposal_json"]))
            tree = proposal["expression"]
            expression = translate_expression_tree(tree)
            expression_hash, canonical = canonical_expression(expression)
            profile = expression_profile(expression)
            source_field_counts.update(profile["fields"])
            validation_error = validate(expression, panel_fields)
            if validation_error:
                raise ValueError(validation_error)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            invalid.append({
                "expression_hash": hashlib.sha256(
                    f"{factor_id}:{row['proposal_json']}".encode("utf-8")
                ).hexdigest()[:16],
                "expression": "",
                "origin_scope": source_market,
                "experiment_ids": [task_ids[str(row["source_task_id"])]],
                "factor_ids": [factor_id],
                "validation_error": str(exc),
            })
            continue

        item = expressions.setdefault(expression_hash, {
            "expression_hash": expression_hash,
            "expression": expression,
            "canonical_expression": canonical,
            "origin_markets": [source_market],
            "origin_scope": source_market,
            "experiment_ids": [],
            "factor_ids": [],
            "node_ids": [],
            "provenance": [],
            "profile": profile,
            "selection_reason": "external_factor_pool_all",
        })
        task_id = task_ids[str(row["source_task_id"])]
        if task_id not in item["experiment_ids"]:
            item["experiment_ids"].append(task_id)
        item["factor_ids"].append(factor_id)
        item["provenance"].append({
            "source": "external_factor_pool",
            "factor_id": factor_id,
            "source_task_id": str(row["source_task_id"]),
            "source_iteration": int(row["source_iteration"]),
            "name": str(row["name"]),
            "family": str(row["family"]),
            "status": str(row["status"]),
            "status_reason": str(row["status_reason"]),
            "expected_direction": int(proposal.get("expected_direction") or 1),
            "created_at": str(row["created_at"]),
            "updated_at": str(row["updated_at"]),
        })

    valid = list(expressions.values())
    for item in valid:
        item["experiment_ids"].sort()
        item["factor_ids"].sort()
        item["source_record_count"] = len(item["factor_ids"])
    valid.sort(key=lambda item: (-int(item["profile"]["complexity"]), item["expression_hash"]))
    invalid.sort(key=lambda item: item["factor_ids"][0])
    created = [str(row["created_at"]) for row in rows]
    experiments = [
        {
            "id": task_ids[task],
            "external_task_id": task,
            "name": task,
            "market": source_market,
            "status": "external_snapshot",
            "created_at": min(
                str(row["created_at"])
                for row in rows
                if str(row["source_task_id"]) == task
            ),
        }
        for task in tasks
    ]
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    return {
        "protocol": SNAPSHOT_PROTOCOL,
        "translation_protocol": TRANSLATION_PROTOCOL,
        "candidate_source_kind": "external_sqlite_factor_pool",
        "candidate_source_label": source_label,
        "source_policy": SOURCE_POLICY,
        "target_market": target_market,
        "source_market": source_market,
        "snapshot_at_asia_shanghai": now,
        "source_database": str(database),
        "source_database_role": "read_only_sqlite_migration_source",
        "source_database_bytes": before.st_size,
        "source_database_mtime_ns": before.st_mtime_ns,
        "source_database_sha256": source_sha256,
        "source_database_integrity": integrity,
        "cutoffs": {
            "source_factor_rows": len(rows),
            "max_source_iteration": max(
                (int(row["source_iteration"]) for row in rows),
                default=0,
            ),
        },
        "experiments": experiments,
        "source_rows": len(rows),
        "unique_expressions": len(valid),
        "selected_source_rows": len(rows),
        "selected_unique_expressions": len(valid) + len(invalid),
        "valid_target_expressions": len(valid),
        "invalid_target_expressions": len(invalid),
        f"valid_{target_market}_expressions": len(valid),
        f"invalid_{target_market}_expressions": len(invalid),
        "selection_counts": {
            "source_factor_rows": len(rows),
            "ast_unique_valid_expressions": len(valid),
            "invalid_expressions": len(invalid),
            "source_task_count": len(tasks),
            "source_status_counts": dict(sorted(status_counts.items())),
        },
        "target_panel_fields": panel_fields,
        "source_field_counts": dict(sorted(source_field_counts.items())),
        "syntax_failures": invalid,
        "invalid": invalid,
        "excluded": [],
        "expressions": valid,
        "source_created_at_min": min(created) if created else None,
        "source_created_at_max": max(created) if created else None,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--source-market", choices=("ashare", "us"), required=True)
    parser.add_argument("--target-market", choices=("ashare", "us"), required=True)
    parser.add_argument("--source-label", required=True)
    parser.add_argument("--panel-glob")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    panel_glob = args.panel_glob or (
        ASHARE_PANEL_GLOB if args.target_market == "ashare" else US_PANEL_GLOB
    )
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot = freeze_snapshot(
        database=args.database,
        source_market=args.source_market,
        target_market=args.target_market,
        panel_glob=panel_glob,
        source_label=args.source_label,
    )
    _json(output_dir / "snapshot.json", snapshot)
    protocol = {
        "protocol": SNAPSHOT_PROTOCOL,
        "translation_protocol": TRANSLATION_PROTOCOL,
        "candidate_source_kind": snapshot["candidate_source_kind"],
        "candidate_source_label": snapshot["candidate_source_label"],
        "source_database": snapshot["source_database"],
        "source_database_sha256": snapshot["source_database_sha256"],
        "target_market": args.target_market,
        "source_market": args.source_market,
        "source_policy": SOURCE_POLICY,
        "spec": {
            "market": args.target_market,
            "universe_n": 500,
            "horizon": 5,
            "top_fraction": 0.20,
            "rebalance_every": 5,
            "max_volume_participation": 0.05,
        },
    }
    _json(output_dir / "protocol.json", protocol)
    _write_csv(output_dir / "invalid_expressions.csv", snapshot["invalid"])
    _write_csv(output_dir / "excluded_expressions.csv", [])
    manifest = {
        "protocol": SNAPSHOT_PROTOCOL,
        "created_at": snapshot["snapshot_at_asia_shanghai"],
        "source_database_sha256": snapshot["source_database_sha256"],
        "source_rows": snapshot["source_rows"],
        "valid_target_expressions": snapshot["valid_target_expressions"],
        "invalid_target_expressions": snapshot["invalid_target_expressions"],
        "artifacts": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            manifest["artifacts"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _json(output_dir / "source_manifest.json", manifest)
    print(
        f"frozen {snapshot['source_rows']} source factors -> "
        f"{snapshot['valid_target_expressions']} AST-unique valid expressions; "
        f"invalid={snapshot['invalid_target_expressions']} · {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
