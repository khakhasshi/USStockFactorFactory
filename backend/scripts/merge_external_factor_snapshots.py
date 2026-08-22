"""Merge frozen external AutoAlpha libraries for one target market.

All source snapshots remain immutable.  Expressions are AST-deduplicated
across libraries and admitted only when every referenced field exists in the
target panel.  Cross-market exclusions are retained as an audited inventory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import polars as pl  # noqa: E402

from app.config import ASHARE_PANEL_GLOB, US_PANEL_GLOB  # noqa: E402
from app.dsl.engine import expression_profile, validate  # noqa: E402


MERGE_PROTOCOL = "external_joint_factor_pool_portability_snapshot_v1"
SOURCE_POLICY = "external_joint_libraries_target_panel_portability"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("expression_hash,expression,validation_error\n", encoding="utf-8")
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                if isinstance(value, (dict, list))
                else value
                for key, value in row.items()
            })


def _read_source(directory: Path) -> tuple[dict[str, Any], str]:
    path = directory.resolve() / "snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if snapshot.get("candidate_source_kind") != "external_sqlite_factor_pool":
        raise ValueError(f"not an external SQLite factor snapshot: {directory}")
    return snapshot, _sha256(path)


def merge_snapshots(
    *,
    source_directories: list[Path],
    target_market: str,
    panel_glob: str,
    source_label: str,
) -> dict[str, Any]:
    loaded = [_read_source(directory) for directory in source_directories]
    panel_fields = sorted(
        pl.scan_parquet(panel_glob, hive_partitioning=True)
        .collect_schema()
        .names()
    )
    panel_field_set = set(panel_fields)

    task_keys = sorted({
        (str(snapshot["source_market"]), str(experiment["external_task_id"]))
        for snapshot, _ in loaded
        for experiment in snapshot.get("experiments", [])
    })
    task_ids = {key: index for index, key in enumerate(task_keys, start=1)}
    experiments = [
        {
            "id": task_ids[key],
            "external_task_id": key[1],
            "name": f"{key[0]}:{key[1]}",
            "market": key[0],
            "status": "external_joint_snapshot",
            "created_at": next(
                str(experiment["created_at"])
                for snapshot, _ in loaded
                if str(snapshot["source_market"]) == key[0]
                for experiment in snapshot.get("experiments", [])
                if str(experiment["external_task_id"]) == key[1]
            ),
        }
        for key in task_keys
    ]

    merged: dict[str, dict[str, Any]] = {}
    excluded: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    source_records_by_market: Counter[str] = Counter()
    selected_records_by_market: Counter[str] = Counter()
    database_sources: list[dict[str, Any]] = []

    for snapshot, snapshot_sha in loaded:
        source_market = str(snapshot["source_market"])
        source_records_by_market[source_market] += int(snapshot["source_rows"])
        database_sources.append({
            "source_market": source_market,
            "source_label": snapshot.get("candidate_source_label"),
            "source_database": snapshot.get("source_database"),
            "source_database_sha256": snapshot.get("source_database_sha256"),
            "source_snapshot_sha256": snapshot_sha,
            "source_rows": snapshot.get("source_rows"),
        })
        for row in snapshot.get("invalid", []):
            invalid.append({
                **row,
                "factor_ids": [
                    f"{source_market}:{factor_id}"
                    for factor_id in row.get("factor_ids", [])
                ],
                "origin_scope": source_market,
                "source_snapshot_sha256": snapshot_sha,
            })

        for row in snapshot.get("expressions", []):
            profile = row.get("profile") or expression_profile(row["expression"])
            missing_fields = sorted(set(profile.get("fields") or []) - panel_field_set)
            qualified_factor_ids = [
                f"{source_market}:{factor_id}"
                for factor_id in row.get("factor_ids", [])
            ]
            base_audit = {
                "expression_hash": row["expression_hash"],
                "expression": row["expression"],
                "origin_scope": source_market,
                "origin_markets": [source_market],
                "factor_ids": qualified_factor_ids,
                "experiment_ids": sorted({
                    task_ids[(source_market, str(provenance["source_task_id"]))]
                    for provenance in row.get("provenance", [])
                }),
                "source_record_count": int(row.get("source_record_count") or 0),
            }
            if missing_fields:
                excluded.append({
                    **base_audit,
                    "selection_exclusion": "target_panel_missing_fields",
                    "missing_fields": missing_fields,
                    "validation_error": (
                        f"{target_market} panel missing fields: "
                        + ", ".join(missing_fields)
                    ),
                })
                continue
            error = validate(str(row["expression"]), panel_fields)
            if error:
                invalid.append({**base_audit, "validation_error": error})
                continue

            item = merged.setdefault(str(row["expression_hash"]), {
                "expression_hash": str(row["expression_hash"]),
                "expression": str(row["expression"]),
                "canonical_expression": row.get("canonical_expression"),
                "origin_markets": set(),
                "experiment_ids": set(),
                "factor_ids": [],
                "node_ids": [],
                "provenance": [],
                "profile": profile,
                "selection_reason": "target_panel_portable",
            })
            item["origin_markets"].add(source_market)
            item["experiment_ids"].update(base_audit["experiment_ids"])
            item["factor_ids"].extend(qualified_factor_ids)
            item["provenance"].extend([
                {
                    **provenance,
                    "source_market": source_market,
                    "qualified_factor_id": (
                        f"{source_market}:{provenance['factor_id']}"
                    ),
                    "source_snapshot_sha256": snapshot_sha,
                }
                for provenance in row.get("provenance", [])
            ])
            selected_records_by_market[source_market] += int(
                row.get("source_record_count") or 0
            )

    expressions = list(merged.values())
    for item in expressions:
        item["origin_markets"] = sorted(item["origin_markets"])
        item["origin_scope"] = (
            "both" if len(item["origin_markets"]) > 1 else item["origin_markets"][0]
        )
        item["experiment_ids"] = sorted(item["experiment_ids"])
        item["factor_ids"] = sorted(set(item["factor_ids"]))
        item["source_record_count"] = len(item["provenance"])
    expressions.sort(
        key=lambda row: (-int(row["profile"]["complexity"]), row["expression_hash"])
    )
    excluded.sort(key=lambda row: (row["expression_hash"], row["origin_scope"]))
    invalid.sort(key=lambda row: (row["expression_hash"], row["origin_scope"]))
    combined_source_sha = hashlib.sha256(
        "\n".join(sorted(
            f"{row['source_market']}:{row['source_database_sha256']}"
            for row in database_sources
        )).encode("utf-8")
    ).hexdigest()
    now = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
    return {
        "protocol": MERGE_PROTOCOL,
        "candidate_source_kind": "external_sqlite_factor_pool",
        "candidate_source_label": source_label,
        "source_policy": SOURCE_POLICY,
        "target_market": target_market,
        "source_markets": sorted(source_records_by_market),
        "snapshot_at_asia_shanghai": now,
        "source_database_sha256": combined_source_sha,
        "source_databases": database_sources,
        "cutoffs": {
            "source_factor_rows": sum(source_records_by_market.values()),
            "source_snapshots": len(loaded),
        },
        "experiments": experiments,
        "source_rows": sum(source_records_by_market.values()),
        "unique_expressions": len(expressions),
        "selected_source_rows": sum(selected_records_by_market.values()),
        "selected_unique_expressions": len(expressions),
        "valid_target_expressions": len(expressions),
        "invalid_target_expressions": len(invalid),
        f"valid_{target_market}_expressions": len(expressions),
        f"invalid_{target_market}_expressions": len(invalid),
        "selection_counts": {
            "source_factor_rows_by_market": dict(sorted(source_records_by_market.items())),
            "portable_source_rows_by_market": dict(sorted(selected_records_by_market.items())),
            "ast_unique_portable_expressions": len(expressions),
            "cross_market_or_field_exclusions": len(excluded),
            "invalid_expressions": len(invalid),
            "source_task_count": len(experiments),
        },
        "target_panel_fields": panel_fields,
        "syntax_failures": invalid,
        "invalid": invalid,
        "excluded": excluded,
        "expressions": expressions,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-report", type=Path, action="append", required=True)
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
    snapshot = merge_snapshots(
        source_directories=args.source_report,
        target_market=args.target_market,
        panel_glob=panel_glob,
        source_label=args.source_label,
    )
    _write_json(output_dir / "snapshot.json", snapshot)
    protocol = {
        "protocol": MERGE_PROTOCOL,
        "candidate_source_kind": snapshot["candidate_source_kind"],
        "candidate_source_label": snapshot["candidate_source_label"],
        "source_database_sha256": snapshot["source_database_sha256"],
        "source_databases": snapshot["source_databases"],
        "target_market": args.target_market,
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
    _write_json(output_dir / "protocol.json", protocol)
    _write_csv(output_dir / "invalid_expressions.csv", snapshot["invalid"])
    _write_csv(output_dir / "excluded_expressions.csv", snapshot["excluded"])
    manifest = {
        "protocol": MERGE_PROTOCOL,
        "created_at": snapshot["snapshot_at_asia_shanghai"],
        "source_database_sha256": snapshot["source_database_sha256"],
        "source_rows": snapshot["source_rows"],
        "valid_target_expressions": snapshot["valid_target_expressions"],
        "invalid_target_expressions": snapshot["invalid_target_expressions"],
        "excluded_expressions": len(snapshot["excluded"]),
        "artifacts": {},
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file():
            manifest["artifacts"][path.name] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    _write_json(output_dir / "source_manifest.json", manifest)
    print(
        f"merged {snapshot['source_rows']} source rows -> "
        f"{snapshot['valid_target_expressions']} portable AST-unique expressions; "
        f"excluded={len(snapshot['excluded'])}; "
        f"invalid={snapshot['invalid_target_expressions']} · {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
