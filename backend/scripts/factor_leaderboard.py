"""Backtest persisted factor candidates on one frozen event-engine protocol.

Example:
    ../.venv/bin/python -m scripts.factor_leaderboard \
      --max-experiment-id 8 --max-node-id 2208 --max-factor-id 1522 \
      --workers 2 --threads-per-worker 4

The command is read-only with respect to PostgreSQL.  It writes a frozen input
snapshot, resumable JSONL results, CSV/Parquet leaderboards, finalist ledgers,
and a manifest below ``var/reports``.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import glob
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg  # noqa: E402
import polars as pl  # noqa: E402

from app.backtest.batch import (  # noqa: E402
    BATCH_PROTOCOL,
    BatchBacktestSpec,
    batch_protocol_for_market,
    canonical_expression,
    flatten_factor_result,
    information_coefficients,
    oriented_expression_hash,
    rank_factor_results,
    run_cost_scenarios,
    select_training_direction,
)
from app.backtest.engine import _prepare_backtest_frame, _write_artifacts  # noqa: E402
from app.config import (  # noqa: E402
    ASHARE_PANEL_GLOB,
    US_PANEL_GLOB,
    get_dsl_fields,
    get_layer_bounds,
)
from app.dsl.engine import expression_profile, validate  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_ROOT = PROJECT_ROOT / "var" / "reports"
_WORKER_SPEC: BatchBacktestSpec | None = None
_WORKER_PANEL_GLOB = ASHARE_PANEL_GLOB
_WORKER_PROTOCOL = BATCH_PROTOCOL
SOURCE_POLICY_ALL = "all"
SOURCE_POLICY_US_PLUS_ASHARE_PRICE_VOLUME = (
    "us_plus_ashare_price_volume"
)
PURE_PRICE_VOLUME_FIELDS = frozenset(
    {"open", "high", "low", "close", "vol", "amount"}
)
REPORT_DIMENSIONS = (
    "absolute_quality_score",
    "robust_score",
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
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, Path)):
        return str(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"无法序列化 {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    _write_json(temporary, value)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _database_value(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _selection_reason(
    item: dict,
    source_policy: str,
) -> str | None:
    """Return why one AST-unique expression belongs to the frozen run."""
    if source_policy == SOURCE_POLICY_ALL:
        return "all_sources"
    if source_policy != SOURCE_POLICY_US_PLUS_ASHARE_PRICE_VOLUME:
        raise ValueError(f"未知来源筛选策略: {source_policy}")
    origins = set(item.get("origin_markets") or [])
    fields = set((item.get("profile") or {}).get("fields") or [])
    us_origin = "us" in origins
    ashare_price_volume = (
        "ashare" in origins
        and bool(item.get("profile"))
        and fields <= PURE_PRICE_VOLUME_FIELDS
    )
    if us_origin and ashare_price_volume:
        return "us_origin_and_ashare_price_volume"
    if us_origin:
        return "us_origin"
    if ashare_price_volume:
        return "ashare_price_volume_transfer"
    return None


async def _freeze_snapshot(
    *,
    max_experiment_id: int,
    max_node_id: int,
    max_factor_id: int,
    target_market: str = "ashare",
    source_policy: str = SOURCE_POLICY_ALL,
    protocol: str | None = None,
) -> dict:
    if target_market not in {"ashare", "us"}:
        raise ValueError("target_market 必须是 ashare 或 us")
    protocol = protocol or batch_protocol_for_market(target_market)
    connection = await asyncpg.connect(database="factor_factory")
    try:
        server_time = await connection.fetchval(
            "SELECT now() AT TIME ZONE 'Asia/Shanghai'"
        )
        experiments = await connection.fetch(
            """
            SELECT id, name, status, research_config, created_at
            FROM experiments
            WHERE id <= $1
            ORDER BY id
            """,
            max_experiment_id,
        )
        records = await connection.fetch(
            """
            SELECT
                'factor' AS source,
                f.id,
                f.experiment_id,
                f.expression,
                f.status,
                f.lifecycle_stage AS source_detail,
                f.created_at
            FROM factors f
            WHERE f.id <= $1 AND f.experiment_id <= $3
            UNION ALL
            SELECT
                'node' AS source,
                n.id,
                n.experiment_id,
                n.expression,
                n.status,
                n.op AS source_detail,
                n.created_at
            FROM nodes n
            WHERE n.id <= $2 AND n.experiment_id <= $3
            ORDER BY source, id
            """,
            max_factor_id,
            max_node_id,
            max_experiment_id,
        )
    finally:
        await connection.close()

    experiment_rows: list[dict] = []
    experiment_market: dict[int, str] = {}
    for record in experiments:
        config = _database_value(record["research_config"]) or {}
        # Experiments 1 and 2 predate explicit task markets and came from the
        # original US-only system.  Preserve that provenance instead of
        # guessing from the current process default.
        market = str(config.get("market") or "us")
        experiment_market[int(record["id"])] = market
        experiment_rows.append({
            "id": int(record["id"]),
            "name": str(record["name"]),
            "status": str(record["status"]),
            "market": market,
            "market_source": (
                "research_config" if config.get("market") else "legacy_us_default"
            ),
            "created_at": str(record["created_at"]),
            "research_config": config,
        })

    expressions: dict[str, dict] = {}
    syntax_failures: list[dict] = []
    for record in records:
        expression = str(record["expression"] or "").strip()
        try:
            expression_hash, canonical = canonical_expression(expression)
        except (SyntaxError, ValueError) as exc:
            syntax_failures.append({
                "source": record["source"],
                "id": int(record["id"]),
                "experiment_id": int(record["experiment_id"]),
                "expression": expression,
                "error": str(exc),
            })
            continue
        item = expressions.setdefault(
            expression_hash,
            {
                "expression_hash": expression_hash,
                # Execute the shortest original source text.  ast.unparse is
                # retained only as identity metadata because its inserted
                # whitespace can push a valid <=500-character historical DSL
                # expression over the parser's safety limit.
                "expression": expression,
                "canonical_expression": canonical,
                "provenance": [],
                "origin_markets": set(),
                "experiment_ids": set(),
                "factor_ids": [],
                "node_ids": [],
            },
        )
        if len(expression) < len(item["expression"]):
            item["expression"] = expression
        source = str(record["source"])
        experiment_id = int(record["experiment_id"])
        market = experiment_market.get(experiment_id, "us")
        item["origin_markets"].add(market)
        item["experiment_ids"].add(experiment_id)
        item[f"{source}_ids"].append(int(record["id"]))
        item["provenance"].append({
            "source": source,
            "id": int(record["id"]),
            "experiment_id": experiment_id,
            "market": market,
            "status": str(record["status"]),
            "detail": str(record["source_detail"] or ""),
            "created_at": str(record["created_at"]),
        })

    selected: list[dict] = []
    valid: list[dict] = []
    invalid: list[dict] = []
    excluded: list[dict] = []
    for item in expressions.values():
        item["origin_markets"] = sorted(item["origin_markets"])
        item["experiment_ids"] = sorted(item["experiment_ids"])
        item["origin_scope"] = (
            "both"
            if len(item["origin_markets"]) > 1
            else item["origin_markets"][0]
        )
        item["source_record_count"] = len(item["provenance"])
        try:
            item["profile"] = expression_profile(item["expression"])
        except (SyntaxError, TypeError, ValueError) as exc:
            item["profile_error"] = str(exc)
        selection_reason = _selection_reason(item, source_policy)
        if selection_reason is None:
            item["selection_exclusion"] = (
                "not_us_origin_and_not_ashare_price_volume"
            )
            excluded.append(item)
            continue
        item["selection_reason"] = selection_reason
        selected.append(item)
        error = validate(
            item["expression"],
            get_dsl_fields(target_market),
        )
        if error:
            item["validation_error"] = error
            invalid.append(item)
            continue
        if "profile" not in item:
            item["profile"] = expression_profile(item["expression"])
        valid.append(item)
    valid.sort(
        key=lambda row: (
            -int(row["profile"]["complexity"]),
            row["expression_hash"],
        )
    )
    invalid.sort(key=lambda row: row["expression_hash"])
    excluded.sort(key=lambda row: row["expression_hash"])
    all_items = list(expressions.values())
    ashare_price_volume = [
        item
        for item in all_items
        if (
            "ashare" in set(item["origin_markets"])
            and bool(item.get("profile"))
            and set(item["profile"]["fields"]) <= PURE_PRICE_VOLUME_FIELDS
        )
    ]
    selection_counts = {
        "all_source_rows": len(records),
        "all_unique_expressions": len(expressions),
        "us_origin_unique_expressions": sum(
            "us" in set(item["origin_markets"])
            for item in all_items
        ),
        "ashare_origin_unique_expressions": sum(
            "ashare" in set(item["origin_markets"])
            for item in all_items
        ),
        "ashare_price_volume_unique_expressions": len(
            ashare_price_volume
        ),
        "ashare_only_price_volume_unique_expressions": sum(
            set(item["origin_markets"]) == {"ashare"}
            for item in ashare_price_volume
        ),
        "us_ashare_price_volume_overlap_unique_expressions": sum(
            "us" in set(item["origin_markets"])
            for item in ashare_price_volume
        ),
        "selected_unique_before_validation": len(selected),
        "selected_source_records": sum(
            int(item["source_record_count"]) for item in selected
        ),
        "selected_provenance_us_records": sum(
            provenance["market"] == "us"
            for item in selected
            for provenance in item["provenance"]
        ),
        "selected_provenance_ashare_records": sum(
            provenance["market"] == "ashare"
            for item in selected
            for provenance in item["provenance"]
        ),
        "valid_target_expressions": len(valid),
        "valid_target_source_records": sum(
            int(item["source_record_count"]) for item in valid
        ),
        "invalid_target_expressions": len(invalid),
        "invalid_target_source_records": sum(
            int(item["source_record_count"]) for item in invalid
        ),
        "excluded_unique_expressions": len(excluded),
    }
    return {
        "protocol": protocol,
        "target_market": target_market,
        "source_policy": source_policy,
        "pure_price_volume_fields": sorted(PURE_PRICE_VOLUME_FIELDS),
        "snapshot_at_asia_shanghai": str(server_time),
        "cutoffs": {
            "max_experiment_id": max_experiment_id,
            "max_node_id": max_node_id,
            "max_factor_id": max_factor_id,
        },
        "experiments": experiment_rows,
        "source_rows": len(records),
        "unique_expressions": len(expressions),
        "selected_source_rows": selection_counts[
            "selected_source_records"
        ],
        "selected_unique_expressions": len(selected),
        "valid_target_expressions": len(valid),
        "invalid_target_expressions": len(invalid),
        f"valid_{target_market}_expressions": len(valid),
        f"invalid_{target_market}_expressions": len(invalid),
        "selection_counts": selection_counts,
        "syntax_failures": syntax_failures,
        "invalid": invalid,
        "excluded": excluded,
        "expressions": valid,
    }


def _worker_init(
    spec: dict,
    panel_glob: str,
    protocol: str,
) -> None:
    global _WORKER_SPEC, _WORKER_PANEL_GLOB, _WORKER_PROTOCOL
    _WORKER_SPEC = BatchBacktestSpec(**{
        **spec,
        "slippage_bps": tuple(spec["slippage_bps"]),
    })
    _WORKER_PANEL_GLOB = panel_glob
    _WORKER_PROTOCOL = protocol


def _prepare_expression_frame(expression: str) -> pl.DataFrame:
    if _WORKER_SPEC is None:
        raise RuntimeError("worker 未初始化")
    frame, _ = _prepare_backtest_frame(
        expression=expression,
        universe_n=_WORKER_SPEC.universe_n,
        start=_WORKER_SPEC.train_start,
        end=_WORKER_SPEC.holdout_end,
        panel_glob=_WORKER_PANEL_GLOB,
        market=_WORKER_SPEC.market,
        forward_horizon=_WORKER_SPEC.horizon,
    )
    return frame


def _worker_factor(record: dict) -> dict:
    if _WORKER_SPEC is None:
        raise RuntimeError("worker 未初始化")
    started = time.perf_counter()
    base = {
        "protocol": _WORKER_PROTOCOL,
        "status": "ok",
        "market": _WORKER_SPEC.market,
        "portfolio_mode": _WORKER_SPEC.mode,
        "expression_hash": record["expression_hash"],
        "expression": record["expression"],
        "origin_scope": record["origin_scope"],
        "origin_markets": record["origin_markets"],
        "selection_reason": record.get("selection_reason"),
        "experiment_ids": record["experiment_ids"],
        "factor_ids": record["factor_ids"],
        "node_ids": record["node_ids"],
        "source_record_count": record["source_record_count"],
        "complexity": record["profile"]["complexity"],
        "required_history": record["profile"]["required_history"],
        "fields": record["profile"]["fields"],
        "operators": record["profile"]["operators"],
        "worker_pid": os.getpid(),
    }
    try:
        frame = _prepare_expression_frame(record["expression"])
        direction, train_ic = select_training_direction(
            frame,
            _WORKER_SPEC,
        )
        holdout_ic = information_coefficients(
            frame,
            start=_WORKER_SPEC.holdout_start,
            end=_WORKER_SPEC.holdout_end,
            horizon=_WORKER_SPEC.horizon,
            universe_n=_WORKER_SPEC.universe_n,
            direction=direction,
        )
        scenarios = run_cost_scenarios(
            frame,
            direction=direction,
            spec=_WORKER_SPEC,
            capture_detail=False,
        )
        return {
            **base,
            "direction": direction,
            "oriented_expression_hash": oriented_expression_hash(
                record["expression"],
                direction,
            ),
            "train_ic": train_ic,
            "holdout_ic": holdout_ic,
            "scenarios": scenarios,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:  # noqa: BLE001 - retain every candidate failure
        return {
            **base,
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "traceback_tail": traceback.format_exc(limit=5)[-6000:],
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }


def _worker_vault(record: dict, direction: int) -> dict:
    if _WORKER_SPEC is None:
        raise RuntimeError("worker 未初始化")
    vault_spec = replace(
        _WORKER_SPEC,
        holdout_start=_WORKER_SPEC.vault_start,
        holdout_end=_WORKER_SPEC.vault_end,
        slippage_bps=(15.0,),
    )
    started = time.perf_counter()
    try:
        frame, _ = _prepare_backtest_frame(
            expression=record["expression"],
            universe_n=vault_spec.universe_n,
            start=vault_spec.holdout_start,
            end=vault_spec.holdout_end,
            panel_glob=_WORKER_PANEL_GLOB,
            market=vault_spec.market,
            forward_horizon=vault_spec.horizon,
        )
        metrics = information_coefficients(
            frame,
            start=vault_spec.holdout_start,
            end=vault_spec.holdout_end,
            horizon=vault_spec.horizon,
            universe_n=vault_spec.universe_n,
            direction=direction,
        )
        scenario = run_cost_scenarios(
            frame,
            direction=direction,
            spec=vault_spec,
            capture_detail=False,
        )["15"]
        return {
            "status": "ok",
            "expression_hash": record["expression_hash"],
            "direction": direction,
            "ic": metrics,
            "scenario_bps_15": scenario,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "expression_hash": record["expression_hash"],
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }


def _worker_finalist_ledger(
    record: dict,
    direction: int,
    artifact_dir: str,
) -> dict:
    if _WORKER_SPEC is None:
        raise RuntimeError("worker 未初始化")
    audit_spec = replace(_WORKER_SPEC, slippage_bps=(15.0,))
    started = time.perf_counter()
    try:
        frame = _prepare_expression_frame(record["expression"])
        scenario = run_cost_scenarios(
            frame,
            direction=direction,
            spec=audit_spec,
            capture_detail=True,
        )["15"]
        result = scenario.pop("result")
        manifest = _write_artifacts(result, Path(artifact_dir))
        return {
            "status": "ok",
            "expression_hash": record["expression_hash"],
            "direction": direction,
            "summary": scenario,
            "artifact_dir": artifact_dir,
            "manifest": manifest,
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "expression_hash": record["expression_hash"],
            "error_type": type(exc).__name__,
            "error": str(exc)[:4000],
            "runtime_seconds": round(time.perf_counter() - started, 4),
        }


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} 第 {line_number} 行不是有效 JSON"
            ) from exc
    return rows


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=_json_default,
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns = sorted({key for row in rows for key in row})
    preferred = [
        "overall_rank",
        "practical_pass",
        "qualification_status",
        "absolute_quality_score",
        "robust_score",
        "portfolio_metric_basis",
        "expression_hash",
        "origin_scope",
        "selection_reason",
        "direction",
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
        "expression",
    ]
    ordered = [key for key in preferred if key in columns]
    ordered.extend(key for key in columns if key not in ordered)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)


def _percentage(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "—"


def _number(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _market_label(market: str) -> str:
    return "A股" if market == "ashare" else "美股"


def _portfolio_label(mode: str) -> str:
    return "纯多头" if mode == "long_only" else "多空"


def _cost_description(spec: BatchBacktestSpec) -> str:
    if spec.market == "ashare":
        return (
            "万二免五佣金、卖出印花税、双向过户费始终计入；"
            "0/5/15 BPS 是额外买卖滑点。"
        )
    if spec.mode == "long_only":
        return (
            "IBKR Pro Fixed 佣金始终计入；纯多头不产生空头借券费；"
            "0/5/15 BPS 是额外买卖滑点。"
        )
    return (
        "IBKR Pro Fixed 佣金与"
        f"{spec.borrow_cost_bps_annual / 100:.2f}% 年化空头借券代理始终计入；"
        "0/5/15 BPS 是额外买卖滑点。"
    )


def _leaderboard_markdown(
    ranked: list[dict],
    *,
    snapshot: dict,
    vault: dict[str, dict],
    spec: BatchBacktestSpec,
    protocol: str,
) -> str:
    representatives = [
        row for row in ranked if row["economic_representative"]
    ]
    practical = [
        row for row in representatives if row["practical_pass"]
    ]
    display = (practical or representatives)[:30]
    market_label = _market_label(spec.market)
    portfolio_label = _portfolio_label(spec.mode)
    lines = [
        f"# {market_label}{portfolio_label}事件回测：跨任务历史因子多维榜单",
        "",
        f"- 协议：`{protocol}`",
        (
            "- 冻结快照："
            f"{snapshot['snapshot_at_asia_shanghai']}；"
            f"{snapshot['source_rows']} 条来源记录，"
            f"{snapshot['unique_expressions']} 个 AST 去重表达式，"
            f"{snapshot['valid_target_expressions']} 个在"
            f"{market_label}可执行。"
        ),
        (
            "- 排名样本：方向只用 2020–2022 选择；"
            "榜单只用 2023–2024；2025–2026 Vault 不参与排序。"
        ),
        f"- 成本：{_cost_description(spec)}",
        "- 标签：`NON_PIT_RESEARCH`，不是实盘批准。",
        "",
        (
            "|总榜|Hash|来源|方向|年化 0bp|年化 5bp|年化 15bp|"
            "夏普 0bp|夏普 5bp|夏普 15bp|IC|ICIR|RankIC|RankICIR|"
            "稳健分|Vault 15bp|"
        ),
        (
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
            "---:|---:|---:|---:|---:|"
        ),
    ]
    for row in display:
        vault_row = vault.get(row["expression_hash"], {})
        vault_scenario = vault_row.get("scenario_bps_15", {})
        vault_text = (
            _percentage(vault_scenario.get("ann_return"))
            if vault_row.get("status") == "ok"
            else "未复核"
        )
        lines.append(
            "|"
            + "|".join([
                str(row["overall_rank"]),
                f"`{row['expression_hash']}`",
                str(row["origin_scope"]),
                f"{int(row['direction']):+d}",
                _percentage(row.get("ann_return_bps_0")),
                _percentage(row.get("ann_return_bps_5")),
                _percentage(row.get("ann_return_bps_15")),
                _number(row.get("sharpe_bps_0")),
                _number(row.get("sharpe_bps_5")),
                _number(row.get("sharpe_bps_15")),
                _number(row.get("oos_ic_mean"), 4),
                _number(row.get("oos_icir")),
                _number(row.get("oos_rank_ic_mean"), 4),
                _number(row.get("oos_rank_icir")),
                _number(row.get("robust_score"), 2),
                vault_text,
            ])
            + "|"
        )
    lines.extend([
        "",
        "## 排名口径",
        "",
        (
            "稳健分由 15bp 年化收益 20%、15bp 夏普 25%、Pearson IC 10%、"
            "Pearson ICIR 15%、RankIC 10%、RankICIR 15%、成本韧性 5% 的"
            "截面百分位加权。`practical_pass` 还要求三档账本完整性通过、"
            "成本结果单调、15bp 收益与夏普为正、IC 与 RankIC 同为正，"
            "且 RankIC 在全部经济等价组上的 BH q≤0.10。"
        ),
        "",
        "## 因子表达式",
        "",
    ])
    for row in display:
        lines.append(
            f"{row['overall_rank']}. `{row['expression_hash']}` — "
            f"`{row['expression']}`"
        )
    return "\n".join(lines) + "\n"


def _progress_payload(
    *,
    protocol: str,
    started_at: float,
    completed: int,
    total: int,
    ok: int,
    errors: int,
    workers: int,
    phase: str,
) -> dict:
    elapsed = max(0.0, time.time() - started_at)
    rate = completed / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total - completed)
    return {
        "protocol": protocol,
        "phase": phase,
        "completed": completed,
        "total": total,
        "percent": round(100.0 * completed / max(1, total), 3),
        "ok": ok,
        "errors": errors,
        "workers": workers,
        "elapsed_seconds": round(elapsed, 1),
        "factors_per_second": round(rate, 5),
        "eta_seconds": round(remaining / rate, 1) if rate > 0 else None,
        "updated_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-experiment-id", type=int, required=True)
    parser.add_argument("--max-node-id", type=int, required=True)
    parser.add_argument("--max-factor-id", type=int, required=True)
    parser.add_argument(
        "--market",
        choices=("ashare", "us"),
        default="ashare",
    )
    parser.add_argument(
        "--source-policy",
        choices=(
            SOURCE_POLICY_ALL,
            SOURCE_POLICY_US_PLUS_ASHARE_PRICE_VOLUME,
        ),
    )
    parser.add_argument(
        "--portfolio-mode",
        choices=("long_only", "long_short"),
    )
    parser.add_argument("--borrow-cost-bps-annual", type=float)
    parser.add_argument("--initial-capital", type=float)
    parser.add_argument("--panel-glob")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads-per-worker", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--vault-top", type=int, default=20)
    parser.add_argument("--ledger-top", type=int, default=3)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--refresh-snapshot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers 必须在 1..4")
    if not 1 <= args.threads_per_worker <= 10:
        raise SystemExit("--threads-per-worker 必须在 1..10")
    source_policy = args.source_policy or (
        SOURCE_POLICY_US_PLUS_ASHARE_PRICE_VOLUME
        if args.market == "us"
        else SOURCE_POLICY_ALL
    )
    portfolio_mode = args.portfolio_mode or (
        "long_short" if args.market == "us" else "long_only"
    )
    if args.market == "ashare" and portfolio_mode != "long_only":
        raise SystemExit("A股批量事件回测只允许 --portfolio-mode long_only")
    borrow_cost_bps_annual = (
        float(args.borrow_cost_bps_annual)
        if args.borrow_cost_bps_annual is not None
        else (300.0 if args.market == "us" and portfolio_mode == "long_short" else 0.0)
    )
    if borrow_cost_bps_annual < 0:
        raise SystemExit("--borrow-cost-bps-annual 不能为负数")
    initial_capital = (
        float(args.initial_capital)
        if args.initial_capital is not None
        else (1_000_000.0 if args.market == "us" else 10_000_000.0)
    )
    if initial_capital <= 0:
        raise SystemExit("--initial-capital 必须为正数")
    panel_glob = (
        args.panel_glob
        or (US_PANEL_GLOB if args.market == "us" else ASHARE_PANEL_GLOB)
    )
    layers = get_layer_bounds(args.market)
    spec = BatchBacktestSpec(
        market=args.market,
        mode=portfolio_mode,
        initial_capital=initial_capital,
        train_start=layers["META_TRAIN"][0],
        train_end=layers["META_TRAIN"][1],
        holdout_start=layers["META_HOLDOUT"][0],
        holdout_end=layers["META_HOLDOUT"][1],
        vault_start=layers["FACTOR_VAULT"][0],
        vault_end=layers["FACTOR_VAULT"][1],
        borrow_cost_bps_annual=borrow_cost_bps_annual,
    )
    protocol = batch_protocol_for_market(
        args.market,
        portfolio_mode,
    )
    os.environ["POLARS_MAX_THREADS"] = str(args.threads_per_worker)
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else (
            REPORT_ROOT
            / f"{args.market}-factor-leaderboard-{run_stamp}"
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_dir / "snapshot.json"
    if snapshot_path.exists() and not args.refresh_snapshot:
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        expected_cutoffs = {
            "max_experiment_id": args.max_experiment_id,
            "max_node_id": args.max_node_id,
            "max_factor_id": args.max_factor_id,
        }
        if snapshot.get("cutoffs") != expected_cutoffs:
            raise SystemExit(
                "已有快照的数据库冻结点与当前参数不一致；"
                "请换用新的 --output-dir"
            )
        if (
            snapshot.get("target_market", "ashare") != args.market
            or snapshot.get("source_policy", SOURCE_POLICY_ALL)
            != source_policy
        ):
            raise SystemExit(
                "已有快照的目标市场或来源策略与当前参数不一致；"
                "请换用新的 --output-dir"
            )
    else:
        snapshot = asyncio.run(_freeze_snapshot(
            max_experiment_id=args.max_experiment_id,
            max_node_id=args.max_node_id,
            max_factor_id=args.max_factor_id,
            target_market=args.market,
            source_policy=source_policy,
            protocol=protocol,
        ))
        _write_json(snapshot_path, snapshot)
    snapshot.setdefault(
        "valid_target_expressions",
        snapshot.get(f"valid_{args.market}_expressions", 0),
    )
    snapshot.setdefault(
        "invalid_target_expressions",
        snapshot.get(f"invalid_{args.market}_expressions", 0),
    )
    snapshot.setdefault("excluded", [])
    expressions = list(snapshot["expressions"])
    if args.limit > 0:
        expressions = expressions[: args.limit]
    by_hash = {row["expression_hash"]: row for row in expressions}
    results_path = output_dir / "results.jsonl"
    existing = _load_jsonl(results_path)
    completed_hashes = {row["expression_hash"] for row in existing}
    pending = [
        row for row in expressions
        if row["expression_hash"] not in completed_hashes
    ]
    panel_files = [Path(path) for path in sorted(glob.glob(panel_glob))]
    if not panel_files:
        raise SystemExit(f"面板路径未匹配任何文件: {panel_glob}")
    panel_inventory = [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in panel_files
    ]
    panel_identity = hashlib.sha256(
        "\n".join(
            f"{row['path']}|{row['bytes']}|{row['mtime_ns']}"
            for row in panel_inventory
        ).encode("utf-8")
    ).hexdigest()
    execution_sources = [
        PROJECT_ROOT / "backend/app/backtest/engine.py",
        PROJECT_ROOT / "backend/app/backtest/fees.py",
        PROJECT_ROOT / "backend/app/backtest/batch.py",
        PROJECT_ROOT / "backend/app/data/panel.py",
        PROJECT_ROOT / "backend/app/dsl/engine.py",
        Path(__file__).resolve(),
    ]
    protocol_path = output_dir / "protocol.json"
    generation_protocol_path = output_dir / "result_generation_protocol.json"
    previous_protocol = (
        json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol_path.exists()
        else None
    )
    generation_protocol = (
        json.loads(generation_protocol_path.read_text(encoding="utf-8"))
        if generation_protocol_path.exists()
        else previous_protocol
    )
    current_protocol = {
        "protocol": protocol,
        "policy_label": "NON_PIT_RESEARCH",
        "target_market": args.market,
        "source_policy": source_policy,
        "spec": spec.as_dict(),
        "panel_glob": panel_glob,
        "panel_identity_path_size_mtime_sha256": panel_identity,
        "panel_inventory": panel_inventory,
        "execution_source_sha256": {
            str(path.relative_to(PROJECT_ROOT)): _sha256(path)
            for path in execution_sources
        },
        "workers": args.workers,
        "threads_per_worker": args.threads_per_worker,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "ranking_uses_vault": False,
        "short_borrow_proxy": (
            {
                "annual_bps": borrow_cost_bps_annual,
                "semantics": "constant_pressure_proxy_not_historical_locate_data",
            }
            if portfolio_mode == "long_short"
            else None
        ),
    }
    if existing and generation_protocol:
        if (
            generation_protocol.get(
                "panel_identity_path_size_mtime_sha256"
            )
            != current_protocol["panel_identity_path_size_mtime_sha256"]
            or json.dumps(
                generation_protocol.get("spec"),
                ensure_ascii=False,
                sort_keys=True,
            )
            != json.dumps(
                current_protocol["spec"],
                ensure_ascii=False,
                sort_keys=True,
            )
        ):
            raise SystemExit(
                "已有结果的面板身份或回测参数与当前运行不一致；"
                "请换用新的 --output-dir"
            )
        generation_sources = generation_protocol.get(
            "execution_source_sha256"
        )
        current_sources = current_protocol["execution_source_sha256"]
        if pending and generation_sources != current_sources:
            raise SystemExit(
                "已有结果由不同源码版本生成，不能与新结果混合续算；"
                "请换用新的 --output-dir，或仅在全部结果完成后重建报告"
            )
        if not generation_protocol_path.exists():
            _write_json(generation_protocol_path, generation_protocol)
        current_protocol["resumed_result_count"] = len(existing)
        current_protocol["result_generation_protocol_file"] = (
            generation_protocol_path.name
        )
        current_protocol["result_generation_sources_match_current"] = (
            generation_sources == current_sources
        )
    _write_json(protocol_path, current_protocol)
    started_at = time.time()
    print(
        f"{_market_label(args.market)}{_portfolio_label(portfolio_mode)} · "
        f"快照 {snapshot['snapshot_at_asia_shanghai']} · "
        f"待回测 {len(pending)}/{len(expressions)} · "
        f"{args.workers} workers × {args.threads_per_worker} Polars threads",
        flush=True,
    )
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=context,
        initializer=_worker_init,
        initargs=(spec.as_dict(), panel_glob, protocol),
    ) as executor:
        futures = {
            executor.submit(_worker_factor, row): row["expression_hash"]
            for row in pending
        }
        results = list(existing)
        done = len(existing)
        ok = sum(row.get("status") == "ok" for row in existing)
        errors = len(existing) - ok
        total = len(expressions)
        for future in as_completed(futures):
            result = future.result()
            _append_jsonl(results_path, result)
            results.append(result)
            done += 1
            if result.get("status") == "ok":
                ok += 1
            else:
                errors += 1
            progress = _progress_payload(
                protocol=protocol,
                started_at=started_at,
                completed=done,
                total=total,
                ok=ok,
                errors=errors,
                workers=args.workers,
                phase="holdout_backtest",
            )
            _write_json_atomic(output_dir / "progress.json", progress)
            if done % 10 == 0 or done == total:
                eta = progress["eta_seconds"]
                print(
                    f"[{done}/{total} {progress['percent']:.1f}%] "
                    f"ok={ok} error={errors} "
                    f"elapsed={progress['elapsed_seconds']:.0f}s "
                    f"eta={eta if eta is not None else '—'}s",
                    flush=True,
                )

        nested_ok = [row for row in results if row.get("status") == "ok"]
        flat = [flatten_factor_result(row) for row in nested_ok]
        ranked = rank_factor_results(flat)
        ranked_by_hash = {
            row["expression_hash"]: row for row in ranked
        }
        representatives = [
            row for row in ranked if row["economic_representative"]
        ]
        finalists = [
            row for row in representatives if row["practical_pass"]
        ] or representatives
        overall_finalist_hashes = [
            row["expression_hash"]
            for row in finalists[: max(args.vault_top, args.ledger_top)]
        ]
        dimension_champions = {
            dimension: max(
                representatives,
                key=lambda row: float(
                    row.get(dimension, float("-inf"))
                ),
            )["expression_hash"]
            for dimension in REPORT_DIMENSIONS
        }
        vault_hashes = list(dict.fromkeys([
            *overall_finalist_hashes[: args.vault_top],
            *dimension_champions.values(),
        ]))
        ledger_hashes = overall_finalist_hashes[: args.ledger_top]
        _write_json(output_dir / "finalists_frozen_before_vault.json", {
            "ranking_uses_vault": False,
            "frozen_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "overall_hashes": overall_finalist_hashes[
                : args.vault_top
            ],
            "dimension_champions": dimension_champions,
            "hashes": vault_hashes,
        })

        vault_results: dict[str, dict] = {}
        vault_futures = {
            executor.submit(
                _worker_vault,
                by_hash[expression_hash],
                int(ranked_by_hash[expression_hash]["direction"]),
            ): expression_hash
            for expression_hash in vault_hashes
        }
        for future in as_completed(vault_futures):
            row = future.result()
            vault_results[row["expression_hash"]] = row
        _write_json(output_dir / "vault_finalists.json", vault_results)

        ledger_root = output_dir / "finalist_ledgers"
        ledger_root.mkdir(exist_ok=True)
        ledger_futures = {
            executor.submit(
                _worker_finalist_ledger,
                by_hash[expression_hash],
                int(ranked_by_hash[expression_hash]["direction"]),
                str(ledger_root / expression_hash),
            ): expression_hash
            for expression_hash in ledger_hashes
        }
        ledger_results: dict[str, dict] = {}
        for future in as_completed(ledger_futures):
            row = future.result()
            ledger_results[row["expression_hash"]] = row
        _write_json(output_dir / "finalist_ledger_audits.json", ledger_results)

    for row in ranked:
        vault_row = vault_results.get(row["expression_hash"])
        if vault_row and vault_row.get("status") == "ok":
            row.update({
                "vault_status": "ok",
                "vault_ann_return_bps_15": vault_row[
                    "scenario_bps_15"
                ]["ann_return"],
                "vault_sharpe_bps_15": vault_row[
                    "scenario_bps_15"
                ]["sharpe"],
                "vault_ic_mean": vault_row["ic"]["ic_mean"],
                "vault_icir": vault_row["ic"]["icir"],
                "vault_rank_ic_mean": vault_row["ic"]["rank_ic_mean"],
                "vault_rank_icir": vault_row["ic"]["rank_icir"],
            })
        else:
            row["vault_status"] = (
                "error" if vault_row else "not_opened"
            )
    full_csv = output_dir / "leaderboard_full.csv"
    _write_csv(full_csv, ranked)
    if ranked:
        pl.DataFrame(ranked, strict=False).write_parquet(
            output_dir / "leaderboard_full.parquet",
            compression="zstd",
        )
    _write_json(output_dir / "leaderboard_full.json", ranked)
    failures = [
        row for row in _load_jsonl(results_path)
        if row.get("status") != "ok"
    ]
    _write_csv(output_dir / "failures.csv", failures)
    invalid_rows = [
        {
            "expression_hash": row["expression_hash"],
            "expression": row["expression"],
            "origin_scope": row["origin_scope"],
            "experiment_ids": row["experiment_ids"],
            "factor_ids": row["factor_ids"],
            "node_ids": row["node_ids"],
            "validation_error": row["validation_error"],
        }
        for row in snapshot["invalid"]
    ]
    _write_csv(output_dir / "invalid_expressions.csv", invalid_rows)
    excluded_rows = [
        {
            "expression_hash": row["expression_hash"],
            "expression": row["expression"],
            "origin_scope": row["origin_scope"],
            "experiment_ids": row["experiment_ids"],
            "factor_ids": row["factor_ids"],
            "node_ids": row["node_ids"],
            "fields": ",".join(
                (row.get("profile") or {}).get("fields") or []
            ),
            "selection_exclusion": row["selection_exclusion"],
        }
        for row in snapshot.get("excluded", [])
    ]
    _write_csv(output_dir / "excluded_expressions.csv", excluded_rows)
    top_by_dimension = {
        key: sorted(
            [
                row for row in ranked
                if row["economic_representative"]
            ],
            key=lambda row: float(row.get(key, float("-inf"))),
            reverse=True,
        )[:20]
        for key in REPORT_DIMENSIONS
    }
    _write_json(output_dir / "top_by_dimension.json", top_by_dimension)
    dimension_rows = []
    for dimension, rows in top_by_dimension.items():
        for dimension_rank, row in enumerate(rows, start=1):
            dimension_rows.append({
                "dimension": dimension,
                "dimension_rank": dimension_rank,
                "dimension_value": row.get(dimension),
                "overall_rank": row["overall_rank"],
                "expression_hash": row["expression_hash"],
                "origin_scope": row["origin_scope"],
                "direction": row["direction"],
                "ann_return_bps_0": row.get("ann_return_bps_0"),
                "ann_return_bps_5": row.get("ann_return_bps_5"),
                "ann_return_bps_15": row.get("ann_return_bps_15"),
                "sharpe_bps_0": row.get("sharpe_bps_0"),
                "sharpe_bps_5": row.get("sharpe_bps_5"),
                "sharpe_bps_15": row.get("sharpe_bps_15"),
                "oos_ic_mean": row.get("oos_ic_mean"),
                "oos_icir": row.get("oos_icir"),
                "oos_rank_ic_mean": row.get("oos_rank_ic_mean"),
                "oos_rank_icir": row.get("oos_rank_icir"),
                "vault_status": row.get("vault_status"),
                "vault_ann_return_bps_15": row.get(
                    "vault_ann_return_bps_15"
                ),
                "vault_sharpe_bps_15": row.get(
                    "vault_sharpe_bps_15"
                ),
                "vault_ic_mean": row.get("vault_ic_mean"),
                "vault_icir": row.get("vault_icir"),
                "vault_rank_ic_mean": row.get("vault_rank_ic_mean"),
                "vault_rank_icir": row.get("vault_rank_icir"),
                "expression": row["expression"],
            })
    _write_csv(output_dir / "top_by_dimension.csv", dimension_rows)
    report_path = output_dir / "leaderboard.md"
    report_path.write_text(
        _leaderboard_markdown(
            ranked,
            snapshot=snapshot,
            vault=vault_results,
            spec=spec,
            protocol=protocol,
        ),
        encoding="utf-8",
    )
    final_progress = _progress_payload(
        protocol=protocol,
        started_at=started_at,
        completed=len(expressions),
        total=len(expressions),
        ok=len(ranked),
        errors=len(failures),
        workers=args.workers,
        phase="complete",
    )
    _write_json_atomic(output_dir / "progress.json", final_progress)
    artifacts = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != output_dir / "manifest.json":
            artifacts[str(path.relative_to(output_dir))] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    manifest = {
        "protocol": protocol,
        "target_market": args.market,
        "source_policy": source_policy,
        "completed_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "duration_seconds": round(time.time() - started_at, 2),
        "snapshot": snapshot["cutoffs"],
        "result_generation_protocol": (
            generation_protocol_path.name
            if generation_protocol_path.exists()
            else protocol_path.name
        ),
        "source_rows": snapshot["source_rows"],
        "unique_expressions": snapshot["unique_expressions"],
        "selection_counts": snapshot.get("selection_counts", {}),
        "scheduled_expressions": len(expressions),
        "successful_expressions": len(ranked),
        "failed_expressions": len(failures),
        "invalid_expressions": len(snapshot["invalid"]),
        "excluded_expressions": len(snapshot.get("excluded", [])),
        "economic_equivalence_groups": sum(
            row["economic_representative"] for row in ranked
        ),
        "origin_scope_counts": dict(sorted(Counter(
            row["origin_scope"] for row in ranked
        ).items())),
        "direction_counts": dict(sorted(Counter(
            f"{int(row['direction']):+d}" for row in ranked
            if row["economic_representative"]
        ).items())),
        "practical_pass": sum(
            row["practical_pass"] and row["economic_representative"]
            for row in ranked
        ),
        "vault_opened_after_ranking": len(vault_results),
        "finalist_ledgers": len(ledger_results),
        "artifacts": artifacts,
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(
        f"完成 · 成功 {len(ranked)} · 失败 {len(failures)} · "
        f"实战硬筛 {manifest['practical_pass']} · {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
