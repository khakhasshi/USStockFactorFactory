import json
import sqlite3
import tempfile
from pathlib import Path

import polars as pl

from backend.scripts.freeze_external_autoalpha_library import (
    freeze_snapshot,
    translate_expression_tree,
)
from backend.scripts.merge_external_factor_snapshots import merge_snapshots


def _field(name: str) -> dict:
    return {"operator": "field", "arguments": [], "parameters": {"name": name}}


def test_translate_expression_tree_preserves_special_operators():
    tree = {
        "operator": "cs_rank",
        "arguments": [{
            "operator": "winsorize_mad",
            "arguments": [{
                "operator": "returns",
                "arguments": [_field("adj_close")],
                "parameters": {"periods": 252},
            }],
            "parameters": {"threshold": 5},
        }],
        "parameters": {},
    }

    assert translate_expression_tree(tree) == (
        "rank(winsor_mad(returns(adj_close, 252), 5))"
    )


def test_freeze_snapshot_is_read_only_ast_deduplicated_and_auditable():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        database = root / "autoalpha.sqlite3"
        connection = sqlite3.connect(database)
        connection.execute(
            """
            CREATE TABLE factor_pool (
                factor_id TEXT PRIMARY KEY,
                source_task_id TEXT NOT NULL,
                source_iteration INTEGER NOT NULL,
                name TEXT NOT NULL,
                family TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                status TEXT NOT NULL,
                status_reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        expression = {
            "operator": "cs_rank",
            "arguments": [{
                "operator": "rolling_mean",
                "arguments": [_field("close")],
                "parameters": {"window": 20},
            }],
            "parameters": {},
        }
        for index, task in enumerate(("task-a", "task-b"), start=1):
            proposal = {
                "name": f"factor-{index}",
                "family": "price",
                "expected_direction": 1,
                "expression": expression,
            }
            connection.execute(
                "INSERT INTO factor_pool VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    f"F_{index}", task, index, f"factor-{index}", "price",
                    json.dumps(proposal), "{}", "ELIGIBLE", "test",
                    "2026-01-01", "2026-01-02",
                ),
            )
        connection.commit()
        connection.close()
        panel = root / "panel.parquet"
        pl.DataFrame({
            "trade_date": ["2026-01-01"],
            "ts_code": ["A"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "vol": [1.0],
            "amount": [1.0],
        }).write_parquet(panel)
        before = database.read_bytes()

        snapshot = freeze_snapshot(
            database=database,
            source_market="ashare",
            target_market="ashare",
            panel_glob=str(panel),
            source_label="test external library",
        )

        assert database.read_bytes() == before
        assert snapshot["source_rows"] == 2
        assert snapshot["unique_expressions"] == 1
        assert snapshot["valid_target_expressions"] == 1
        assert snapshot["invalid_target_expressions"] == 0
        assert snapshot["expressions"][0]["factor_ids"] == ["F_1", "F_2"]
        assert snapshot["expressions"][0]["source_record_count"] == 2
        assert snapshot["candidate_source_kind"] == "external_sqlite_factor_pool"
        assert snapshot["source_database_integrity"] == "ok"


def test_joint_snapshot_keeps_cross_market_provenance_and_excludes_missing_fields():
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        panel = root / "us.parquet"
        pl.DataFrame({
            "trade_date": ["2026-01-01"],
            "ts_code": ["A"],
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "vol": [1.0], "amount": [1.0],
        }).write_parquet(panel)

        def source(market: str, rows: list[dict]) -> Path:
            directory = root / market
            directory.mkdir()
            snapshot = {
                "candidate_source_kind": "external_sqlite_factor_pool",
                "candidate_source_label": market,
                "source_market": market,
                "source_database": f"/{market}.sqlite3",
                "source_database_sha256": market * 8,
                "source_rows": len(rows),
                "experiments": [{
                    "id": 1,
                    "external_task_id": "task-1",
                    "created_at": "2026-01-01",
                }],
                "invalid": [],
                "expressions": rows,
            }
            (directory / "snapshot.json").write_text(json.dumps(snapshot))
            return directory

        common = {
            "expression_hash": "same",
            "expression": "rank(close)",
            "canonical_expression": "rank(close)",
            "factor_ids": ["F_close"],
            "source_record_count": 1,
            "profile": {
                "fields": ["close"], "operators": ["rank"],
                "complexity": 4, "required_history": 1,
            },
        }
        ashare = source("ashare", [
            {
                **common,
                "provenance": [{"factor_id": "F_close", "source_task_id": "task-1"}],
            },
            {
                "expression_hash": "pb-only",
                "expression": "rank(pb)",
                "canonical_expression": "rank(pb)",
                "factor_ids": ["F_pb"],
                "source_record_count": 1,
                "profile": {
                    "fields": ["pb"], "operators": ["rank"],
                    "complexity": 4, "required_history": 1,
                },
                "provenance": [{"factor_id": "F_pb", "source_task_id": "task-1"}],
            },
        ])
        us = source("us", [{
            **common,
            "provenance": [{"factor_id": "F_close", "source_task_id": "task-1"}],
        }])

        merged = merge_snapshots(
            source_directories=[ashare, us],
            target_market="us",
            panel_glob=str(panel),
            source_label="joint",
        )

        assert merged["source_rows"] == 3
        assert merged["valid_target_expressions"] == 1
        assert len(merged["excluded"]) == 1
        assert merged["excluded"][0]["missing_fields"] == ["pb"]
        row = merged["expressions"][0]
        assert row["origin_scope"] == "both"
        assert row["factor_ids"] == ["ashare:F_close", "us:F_close"]
        assert row["source_record_count"] == 2
