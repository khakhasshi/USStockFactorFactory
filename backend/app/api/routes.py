import asyncio
import json
import math
import os
import statistics as st
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import polars as pl
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.engine import make_url

from ..backtest.engine import run_backtest, run_multi_factor_backtest
from ..backtest.rust_kernel import rust_kernel_capabilities
from ..blind_review import (
    build_review_packets,
    deterministic_code_review,
    seal_reviews,
)
from ..factor_lifecycle import factor_evidence_state
from ..combination_lab import (
    COMBINATION_LAB_PROTOCOL,
    CombinationLabManager,
    validate_lab_spec,
)
from ..compute_progress import ACTIVE_STATES, COMPUTE_PROGRESS
from ..config import (
    BACKTEST_ARTIFACT_ROOT,
    DATABASE_URL,
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_ENGINE_CONFIG_V2,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_MINER_TEMPLATE,
    DEFAULT_PORTFOLIO_MODE,
    DEFAULT_RESEARCH_DIRECTION_POLICY,
    DIRECTION_POLICY_FIXED,
    EVALUATION_PROTOCOL_VERSION,
    FROZEN_RATING_PROTOCOL_VERSION,
    FROZEN_RATING_WINDOW_END,
    FROZEN_RATING_WINDOW_START,
    PANEL_GLOB,
    SERVICE_ARCHITECTURE,
    SERVICE_INSTANCE,
    default_panel_glob,
    evaluation_config,
    get_dsl_fields,
    resolve_engine_tasks,
)
from ..data.panel import PanelStore
from ..db import (
    SessionLocal,
    active_experiment_setting_key,
    engine as db_engine,
    get_active_experiment_id,
)
from ..dsl.engine import (
    OPERATORS_DOC,
    expression_profile,
    normalize_hash,
    parse,
    validate,
)
from ..eval.harness import evaluate, evaluate_full
from ..eval.ranking import ranking_diagnostics
from ..feedback import (
    FEEDBACK_SCHEMA_VERSION,
    OUTER_REPORT_SCHEMA_VERSION,
)
from ..leaderboards import (
    build_leaderboard_catalog,
    load_leaderboard_factor_detail,
    resolve_leaderboard_file,
)
from ..factors.diversity import infer_mechanism, mechanisms_for_market
from ..factors.return_source_governance import (
    RETURN_SOURCE_GOVERNANCE_PROTOCOL,
    cluster_training_return_sources,
    resolve_return_source_governance,
)
from ..factors.semantics import audit_expression_semantics, field_contract
from ..factors.similarity import (
    build_similarity_index,
    expression_similarity,
    expression_fingerprint,
    nearest_factors,
)
from ..factor_tools import (
    build_combination_expression,
    factor_tool_capabilities,
    run_factor_correlation,
)
from ..models import (
    Backtest,
    CombinationExperiment,
    EngineEvent,
    Experiment,
    Factor,
    LLMCallAudit,
    MinerVersion,
    Node,
    OuterStep,
    ScreenerRun,
    Setting,
    Trial,
)
from ..meta.agent import propose_template
from ..observability import (
    AsyncTTLCache,
    OBSERVABILITY,
    build_findings,
    build_slo,
    fingerprint_payload,
    overall_health,
    redact_text,
    redact_value,
)
from ..orchestrator import EngineManager
from ..runtime_identity import runtime_identity
from ..portfolio_allocation import (
    ALLOCATION_METHODS,
    build_purchase_allocation,
)
from ..qlib_native import (
    alpha158_catalog,
    alpha158_progress,
    latest_alpha158_reports,
    qlib_native_capabilities,
)
from ..qlib_joint import JointModelSpec, latest_joint_result, run_joint_alpha158
from ..research_records import (
    RESEARCH_RECORD_SCHEMA_VERSION,
    research_record_payload,
    task_research_summary,
)
from ..research_documents import (
    research_document_catalog,
    research_document_metadata,
    resolve_research_document,
)
from ..document_dsl import document_to_dsl
from ..mechanism_catalog import catalog as mechanism_catalog
from ..research_overfit import (
    cscv_pbo,
    deflated_sharpe_ratio,
    effective_trial_count,
    harvey_liu_haircut,
    winner_curse,
)
from ..residual_beam import residual_oof_beam_search
from ..research_architecture import (
    RESEARCH_ARCHITECTURE_SCHEMA,
    architecture_catalog,
    resolve_research_architecture,
)
from ..search_pool import (
    ALGORITHM_GROUPS,
    DEFAULT_SEARCH_ALGORITHMS,
    SEARCH_GROUP_WEIGHTS,
    SEARCH_POLICY_SCHEMA,
    SUPPORTED_SEARCH_ALGORITHMS,
)
from ..screener import SCREEN_CACHE, screen_cross_section

router = APIRouter(prefix="/api")
_similarity_cache: OrderedDict[tuple[int, int, int, float], dict] = OrderedDict()
_similarity_cache_lock = asyncio.Lock()
_similarity_cache_stats = {
    "hits": 0,
    "misses": 0,
    "builds": 0,
    "evictions": 0,
    "last_build_ms": None,
    "last_build_at": None,
}
_observability_components_cache = AsyncTTLCache(
    ttl_seconds=10.0,
    capacity=1,
)
_observability_events_cache = AsyncTTLCache(
    ttl_seconds=3.0,
    capacity=32,
)


def _invalidate_observability_components() -> None:
    _observability_components_cache.clear()


def invalidate_panel_dependents() -> dict:
    """Invalidate derived in-process state after a panel generation swap."""
    removed = SCREEN_CACHE.clear()
    _invalidate_observability_components()
    return {
        "screener_entries_removed": removed,
        "observability_components_invalidated": True,
    }


def _campaign_controls(
    config: dict,
    market: str,
    *,
    default_budget: int = 0,
) -> tuple[int, list[str]]:
    try:
        raw_budget = config.get("candidate_evaluation_budget", default_budget)
        budget = int(raw_budget if raw_budget is not None else default_budget)
    except (TypeError, ValueError) as exc:
        raise HTTPException(
            400,
            "candidate_evaluation_budget 必须为非负整数",
        ) from exc
    if budget < 0:
        raise HTTPException(
            400,
            "candidate_evaluation_budget 必须为非负整数",
        )

    configured = config.get("target_mechanisms") or []
    if not isinstance(configured, list):
        raise HTTPException(400, "target_mechanisms 必须为机制名称列表")
    allowed = set(mechanisms_for_market(market))
    targets: list[str] = []
    for value in configured:
        mechanism = str(value).strip()
        if not mechanism or mechanism not in allowed:
            raise HTTPException(
                400,
                f"收益机制 {mechanism or value} 不适用于 {market}",
            )
        if mechanism not in targets:
            targets.append(mechanism)
    return budget, targets


async def _experiment_context(experiment_id: int | None = None) -> tuple[int, dict]:
    eid = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        exp = await s.get(Experiment, eid)
    if not exp:
        raise HTTPException(404, "研究任务不存在")
    return eid, (exp.research_config or {})


def _resolved_task_pool(cfg: dict, global_engine: Setting | None) -> list[dict]:
    market = cfg.get("market", "us")
    mode = cfg.get(
        "portfolio_mode",
        "long_only" if market == "ashare" else "long_short",
    )
    direction = int(cfg.get("direction", 1))
    direction_policy = str(
        cfg.get("direction_policy")
        or DEFAULT_RESEARCH_DIRECTION_POLICY
    )
    local_tasks = (cfg.get("engine_config") or {}).get("tasks")
    global_tasks = (global_engine.value if global_engine else {}).get("tasks")
    return resolve_engine_tasks(
        local_tasks or global_tasks or DEFAULT_ENGINE_CONFIG["tasks"],
        market,
        mode,
        direction,
        direction_policy,
        preserve_declared_costs=bool(local_tasks),
    )


def _compact_layer_metrics(metrics: dict | None) -> dict:
    metrics = metrics or {}
    compact = {
        key: metrics.get(key)
        for key in (
            "available",
            "window_start",
            "window_end",
            "window_policy",
            "ic_mean",
            "icir",
            "hac_p_value",
            "era_consistency",
            "monotonicity",
            "turnover",
            "daily_turnover",
            "cost_breakeven_bps",
            "cost_cushion_multiple",
            "profitable_era_rate",
            "worst_era_sharpe",
            "score",
            "portfolio_mode",
        )
        if key in metrics
    }
    for branch in ("active", "net", "gross", "long_leg", "short_leg"):
        values = metrics.get(branch)
        if isinstance(values, dict):
            compact[branch] = {
                key: values.get(key)
                for key in ("sharpe", "ann_return", "max_drawdown")
                if key in values
            }
    return compact


def _compact_ranking(metrics: dict | None) -> dict:
    ranking = (metrics or {}).get("ranking") or {}
    if not isinstance(ranking, dict):
        return {}
    compact = {
        key: ranking.get(key)
        for key in (
            "available",
            "score",
            "score_frozen_rating",
            "score_pre_vault",
            "rating_protocol_version",
            "status",
            "vault_seal",
            "policy_label",
            "rating_window",
        )
        if key in ranking
    }
    evidence = ranking.get("evidence")
    if isinstance(evidence, dict):
        compact["evidence"] = {
            key: evidence.get(key)
            for key in (
                "holdout_sharpe",
                "holdout_ann_return",
                "holdout_sharpe_lcb",
                "holdout_ann_return_lcb",
                "holdout_return_hac_t",
                "rating_sharpe",
                "rating_ann_return",
                "rating_sharpe_lcb",
                "rating_ann_return_lcb",
                "rating_return_hac_t",
                "cost_breakeven_bps",
                "cost_cushion_multiple",
                "worst_stress_sharpe",
            )
        }
    compact["current"] = (
        ranking.get("rating_protocol_version")
        == FROZEN_RATING_PROTOCOL_VERSION
    )
    return compact


def _factor_payload(f: Factor, include_validation: bool = False) -> dict:
    validation = f.validation_metrics or {}
    full_ranking = dict(validation.get("ranking") or {})
    full_ranking["current"] = (
        full_ranking.get("rating_protocol_version")
        == FROZEN_RATING_PROTOCOL_VERSION
    )
    market = str((f.research_meta or {}).get("market") or "us")
    current_review = deterministic_code_review(f.expression, market)
    evidence_state = factor_evidence_state(
        f,
        current_code_review=current_review,
        current_rating_protocol=FROZEN_RATING_PROTOCOL_VERSION,
    )
    payload = {
        "id": f.id,
        "experiment_id": f.experiment_id,
        "name": f.name,
        "expression": f.expression,
        "status": evidence_state["effective_status"],
        "raw_status": f.status,
        "evidence_state": evidence_state,
        "lifecycle_stage": f.lifecycle_stage or "legacy_unreviewed",
        "provenance_status": f.provenance_status or "unverified",
        "task": f.task_name,
        "hypothesis": f.hypothesis,
        "public": (
            f.public_metrics or {}
            if include_validation
            else _compact_layer_metrics(f.public_metrics)
        ),
        "gate": (
            f.gate_metrics or {}
            if include_validation
            else _compact_layer_metrics(f.gate_metrics)
        ),
        "research_meta": f.research_meta or {},
        "evaluation_protocol": f.evaluation_protocol or "legacy_unoriented",
        "eligibility": f.eligibility or {},
        "ranking": (
            full_ranking
            if include_validation
            else _compact_ranking(validation)
        ),
        "evaluated_at": str(f.evaluated_at) if f.evaluated_at else None,
        "created_at": str(f.created_at),
    }
    if include_validation:
        payload["validation"] = validation
        payload["fingerprint"] = f.fingerprint or {}
        payload["current_code_review"] = current_review
    return payload


# ---------- 引擎 ----------

class EngineStartReq(BaseModel):
    mode: str = "v2"
    experiment_id: int | None = None


class ServiceLLMProbeReq(BaseModel):
    experiment_id: int
    layer: int = Field(default=3, ge=3, le=3)


class ThreeLayerCampaignReq(BaseModel):
    campaign_id: str = "us-v668-three-layer-abcde-v4"
    name_prefix: str = "美股V668-三层架构"
    candidate_evaluation_budget: int = Field(default=120, ge=20, le=2000)
    start: bool = True


@router.post("/engine/start")
async def engine_start(req: EngineStartReq | None = None):
    mode = req.mode if req else "v2"
    if mode not in ("v1", "v2"):
        raise HTTPException(400, "mode 必须为 v1 或 v2")
    if mode == "v1":
        raise HTTPException(
            409,
            "V1 引擎已冻结为历史只读实现；新研究只能使用 V2",
        )
    return await EngineManager.get().start(mode, req.experiment_id if req else None)


@router.post("/engine/stop")
async def engine_stop(req: dict | None = None):
    experiment_id = req.get("experiment_id") if req else None
    return await EngineManager.get().stop(experiment_id)


def _progress_numbers(completed, total) -> tuple[float | None, float | None, float | None]:
    try:
        done = float(completed) if completed is not None else None
    except (TypeError, ValueError):
        done = None
    try:
        size = float(total) if total is not None else None
    except (TypeError, ValueError):
        size = None
    ratio = (
        max(0.0, min(1.0, done / size))
        if done is not None and size is not None and size > 0
        else None
    )
    return done, size, ratio


def _elapsed_seconds(started_at) -> float | None:
    if not started_at:
        return None
    try:
        started = (
            started_at
            if isinstance(started_at, datetime)
            else datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
        )
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        return round(max(0.0, (datetime.now(timezone.utc) - started).total_seconds()), 3)
    except (TypeError, ValueError):
        return None


def _compute_job(
    *,
    job_id: str,
    kind: str,
    title: str,
    state: str,
    phase: str,
    message: str = "",
    completed=None,
    total=None,
    cancellable: bool = False,
    experiment_id: int | None = None,
    error: str = "",
    started_at=None,
    updated_at=None,
    completed_at=None,
    heartbeat_age_seconds=None,
    elapsed_seconds=None,
    metadata: dict | None = None,
    source: str = "durable",
) -> dict:
    done, size, ratio = _progress_numbers(completed, total)
    resolved_elapsed = elapsed_seconds
    if resolved_elapsed is None and started_at and completed_at:
        try:
            started_value = started_at if isinstance(started_at, datetime) else datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))
            completed_value = completed_at if isinstance(completed_at, datetime) else datetime.fromisoformat(str(completed_at).replace("Z", "+00:00"))
            if started_value.tzinfo is None:
                started_value = started_value.replace(tzinfo=timezone.utc)
            if completed_value.tzinfo is None:
                completed_value = completed_value.replace(tzinfo=timezone.utc)
            resolved_elapsed = round(max(0.0, (completed_value - started_value).total_seconds()), 3)
        except (TypeError, ValueError):
            resolved_elapsed = None
    elif resolved_elapsed is None and state in ACTIVE_STATES:
        resolved_elapsed = _elapsed_seconds(started_at)
    return {
        "job_id": job_id,
        "kind": kind,
        "title": title,
        "state": state,
        "phase": phase,
        "message": message,
        "completed": done,
        "total": size,
        "progress": ratio,
        "indeterminate": ratio is None,
        "cancellable": bool(cancellable and state in ACTIVE_STATES),
        "experiment_id": experiment_id,
        "error": str(error or "")[:4000],
        "started_at": started_at.isoformat() if isinstance(started_at, datetime) else started_at,
        "updated_at": updated_at.isoformat() if isinstance(updated_at, datetime) else updated_at,
        "completed_at": completed_at.isoformat() if isinstance(completed_at, datetime) else completed_at,
        "heartbeat_age_seconds": heartbeat_age_seconds,
        "elapsed_seconds": resolved_elapsed,
        "metadata": dict(metadata or {}),
        "source": source,
    }


@router.get("/compute-tasks")
async def compute_tasks(include_recent: bool = True, limit: int = 60):
    """One progress protocol for every material calculation in this process.

    Exact totals are exposed only when the worker knows them.  Continuous
    research, panel collection and other open-ended phases remain explicitly
    indeterminate and rely on phase/heartbeat/elapsed-time observability.
    """
    limit = max(10, min(200, int(limit)))
    jobs: dict[str, dict] = {
        row["job_id"]: {**row, "source": "runtime_registry"}
        for row in COMPUTE_PROGRESS.snapshot(include_recent=include_recent, limit=limit)
    }
    manager = EngineManager.get()
    worker_rows = manager.all_status(include_logs=False)
    experiment_ids = {int(row["experiment_id"]) for row in worker_rows if row.get("experiment_id")}
    async with SessionLocal() as session:
        names = {
            row.id: row.name
            for row in (await session.scalars(
                select(Experiment).where(Experiment.id.in_(experiment_ids))
            )).all()
        } if experiment_ids else {}
        combinations = list((await session.scalars(
            select(CombinationExperiment)
            .order_by(CombinationExperiment.id.desc())
            .limit(30)
        )).all())
        backtests = list((await session.scalars(
            select(Backtest).order_by(Backtest.id.desc()).limit(30)
        )).all())

    for runtime in worker_rows:
        experiment_id = int(runtime.get("experiment_id") or 0)
        state = "running" if runtime.get("running") else str(runtime.get("state") or "stopped")
        if state == "error":
            state = "failed"
        completed = runtime.get("candidate_evaluations", runtime.get("inner_evals"))
        total = runtime.get("candidate_evaluation_budget")
        if not total:
            total = (runtime.get("task_config") or {}).get("candidate_evaluation_budget")
        if not total or float(total) <= 0:
            total = None
        job = _compute_job(
            job_id=f"research:{experiment_id}",
            kind="research",
            title=f"因子研究 · {names.get(experiment_id, f'任务#{experiment_id}')}",
            state=state,
            phase=str(runtime.get("phase") or "not_started"),
            message=str(runtime.get("current_operation") or runtime.get("current_task") or ""),
            completed=completed,
            total=total,
            cancellable=True,
            experiment_id=experiment_id,
            error=str(runtime.get("last_error") or runtime.get("task_exception") or ""),
            started_at=runtime.get("started_at"),
            updated_at=runtime.get("last_progress_at") or runtime.get("last_heartbeat_at"),
            heartbeat_age_seconds=runtime.get("heartbeat_age_seconds"),
            elapsed_seconds=runtime.get("uptime_seconds"),
            metadata={
                "market": (runtime.get("task_config") or {}).get("market"),
                "continuous": total is None,
                "heartbeat_stale": bool(runtime.get("heartbeat_stale")),
                "outer_step": runtime.get("outer_step"),
                "llm_calls": runtime.get("llm_calls"),
                "factor_count": runtime.get("factor_count"),
                "formal_factor_count": runtime.get("formal_factor_count"),
                "research_candidate_count": runtime.get(
                    "research_candidate_count"
                ),
                "hidden_novelty_resamples": runtime.get(
                    "session_hidden_novelty_resamples"
                ),
                "hidden_resample_waste_rate": runtime.get(
                    "hidden_resample_waste_rate"
                ),
                "effective_evaluations_per_hour": runtime.get(
                    "effective_evaluations_per_hour"
                ),
                "duplicate_waste_rate": runtime.get("duplicate_waste_rate"),
                "recent_unique_yield_rate": runtime.get(
                    "recent_unique_yield_rate"
                ),
                "search_space_exhausted": bool(
                    runtime.get("search_space_exhausted")
                ),
            },
        )
        jobs[job["job_id"]] = job

    for row in combinations:
        state_map = {"error": "failed", "draft": "stopped"}
        state = state_map.get(row.status, row.status)
        if not include_recent and state not in ACTIVE_STATES:
            continue
        runtime = CombinationLabManager.get().snapshot(row.id) or dict(row.progress or {})
        job = _compute_job(
            job_id=f"combination:{row.id}",
            kind="combination",
            title=f"组合优化 · {row.name}",
            state=state,
            phase=str(runtime.get("stage") or state),
            message=str(runtime.get("message") or ""),
            completed=runtime.get("completed"),
            total=runtime.get("total"),
            cancellable=True,
            experiment_id=row.experiment_id,
            error=row.error,
            started_at=row.started_at or row.created_at,
            updated_at=runtime.get("updated_at") or row.completed_at or row.created_at,
            completed_at=row.completed_at,
            elapsed_seconds=(row.result or {}).get("elapsed_seconds"),
            metadata={"market": row.market, "search_mode": row.search_mode},
        )
        jobs[job["job_id"]] = job

    for row in backtests:
        job_id = f"backtest:{row.id}"
        if job_id in jobs:
            continue
        state = "failed" if row.status == "failed" else row.status
        if not include_recent and state not in ACTIVE_STATES:
            continue
        job = _compute_job(
            job_id=job_id,
            kind="backtest",
            title=f"事件回测 #{row.id}",
            state=state,
            phase="event_backtest" if state in ACTIVE_STATES else state,
            message=f"{(row.params or {}).get('market', '—')} · {(row.params or {}).get('start', '—')} 至 {(row.params or {}).get('end', '—')}",
            experiment_id=row.experiment_id,
            error=row.error,
            started_at=row.created_at,
            updated_at=row.created_at,
            metadata={"market": (row.params or {}).get("market")},
        )
        jobs[job_id] = job

    try:
        alpha = await asyncio.to_thread(alpha158_progress)
    except (OSError, ValueError, json.JSONDecodeError):
        alpha = {"state": "not_started"}
    if alpha.get("state") != "not_started":
        state = "done" if alpha.get("state") == "complete" else str(alpha.get("state") or "running")
        if include_recent or state in ACTIVE_STATES:
            job = _compute_job(
                job_id="qlib-alpha158:benchmark",
                kind="qlib_alpha158",
                title="Qlib Alpha158 双市场基准",
                state=state,
                phase=str(alpha.get("current") or alpha.get("phase") or state),
                message=f"{alpha.get('market', '—')} · {alpha.get('portfolio_mode', '—')}",
                completed=alpha.get("completed"),
                total=alpha.get("total"),
                started_at=alpha.get("started_at"),
                updated_at=alpha.get("updated_at"),
                error=alpha.get("error", ""),
                metadata={"artifact_path": alpha.get("artifact_path")},
            )
            jobs[job["job_id"]] = job

    panel_registry = await asyncio.to_thread(PanelStore.registry_snapshot)
    for panel in panel_registry.get("panels", []):
        state = str(panel.get("state") or "cold")
        if state not in {"loading", "reloading", "error"}:
            continue
        job = _compute_job(
            job_id=f"panel:{panel.get('id')}",
            kind="panel",
            title=f"数据面板 · {panel.get('market', '—')}",
            state="failed" if state == "error" else "running",
            phase=state,
            message="面板热重载" if state == "reloading" else "加载研究面板",
            error=panel.get("load_error") or panel.get("reload_error") or "",
            started_at=panel.get("reload_started_at") or panel.get("load_started_at"),
            updated_at=panel.get("reloaded_at") or panel.get("loaded_at"),
            metadata={"generation": panel.get("generation")},
        )
        jobs[job["job_id"]] = job

    ordered = sorted(
        jobs.values(),
        key=lambda row: (
            0 if row.get("state") in ACTIVE_STATES else 1,
            -(datetime.fromisoformat(str(row.get("updated_at") or row.get("started_at") or "1970-01-01").replace("Z", "+00:00")).timestamp()),
        ),
    )[:limit]
    active_count = sum(row.get("state") in ACTIVE_STATES for row in ordered)
    failed_count = sum(row.get("state") == "failed" for row in ordered)
    return {
        "schema": "factorfactory.compute-progress/v1",
        "observed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "active_count": active_count,
        "failed_count": failed_count,
        "total": len(ordered),
        "tasks": ordered,
        "semantics": {
            "determinate": "completed/total 仅在总量真实可知时提供",
            "indeterminate": "持续任务或不可预估阶段只展示阶段、心跳和耗时",
        },
    }


@router.post("/compute-tasks/{job_id:path}/cancel")
async def cancel_compute_task(job_id: str):
    if job_id.startswith("research:"):
        experiment_id = int(job_id.split(":", 1)[1])
        return await EngineManager.get().stop(experiment_id)
    if job_id.startswith("combination:"):
        combination_id = int(job_id.split(":", 1)[1])
        return await CombinationLabManager.get().stop(combination_id)
    raise HTTPException(409, "该计算阶段不支持安全取消")


@router.post("/service/llm-probe")
async def service_llm_probe(req: ServiceLLMProbeReq):
    """Exercise the Governor path without changing research state.

    The call appends a normal redacted LLM audit marked ``service_probe`` but
    does not create a miner version, node, trial, factor, or outer step.
    """
    if SERVICE_ARCHITECTURE != "three_layer":
        raise HTTPException(409, "Governor 探针仅适用于三层服务")
    async with SessionLocal() as session:
        experiment = await session.get(Experiment, req.experiment_id)
        if experiment is None:
            raise HTTPException(404, "研究任务不存在")
        config = dict(experiment.research_config or {})
        if (
            str(config.get("service_instance") or "") != SERVICE_INSTANCE
            or not config.get("layer3_enabled")
        ):
            raise HTTPException(409, "任务不属于当前三层服务")
        provider_settings = await session.get(Setting, "llm_providers")
        provider_config = dict(provider_settings.value or {}) if provider_settings else {}
        provider_name = provider_config.get("outer_provider")
        provider = next(
            (
                row
                for row in provider_config.get("providers", [])
                if row.get("name") == provider_name and row.get("api_key")
            ),
            None,
        )
        incumbent = await session.scalar(
            select(MinerVersion)
            .where(
                MinerVersion.experiment_id == req.experiment_id,
                MinerVersion.status == "incumbent",
                MinerVersion.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
            )
            .order_by(MinerVersion.id.desc())
        )
    if provider is None:
        raise HTTPException(503, "outer_provider 未配置")
    template = (
        incumbent.harness_spec
        if incumbent is not None and isinstance(incumbent.harness_spec, dict)
        else DEFAULT_MINER_TEMPLATE
    )
    _, note, source, reflection = await propose_template(
        template,
        [],
        provider,
        market=config.get("market", "us"),
        portfolio_mode=config.get("portfolio_mode", "long_short"),
        direction=int(config.get("direction", 1)),
        direction_policy=config.get("direction_policy", "both_train_select"),
        trace_context={
            "experiment_id": req.experiment_id,
            "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
            "runtime_identity": runtime_identity(),
            "architecture_layer": 3,
            "service_probe": True,
            "service_instance": SERVICE_INSTANCE,
        },
    )
    async with SessionLocal() as session:
        audit = await session.scalar(
            select(LLMCallAudit)
            .where(
                LLMCallAudit.experiment_id == req.experiment_id,
                LLMCallAudit.role == "outer",
            )
            .order_by(LLMCallAudit.id.desc())
        )
    return {
        "ok": source in {"llm", "llm_rejected"},
        "probe_only": True,
        "service_instance": SERVICE_INSTANCE,
        "experiment_id": req.experiment_id,
        "layer": req.layer,
        "source": source,
        "note": note,
        "reflection": reflection,
        "audit": {
            "id": audit.id if audit else None,
            "status": audit.status if audit else None,
            "provider_name": audit.provider_name if audit else None,
            "model": audit.model if audit else None,
            "latency_ms": audit.latency_ms if audit else None,
        },
    }


@router.get("/engine/status")
async def engine_status(experiment_id: int | None = None):
    manager = EngineManager.get()
    exp_id = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        exp = await s.get(Experiment, exp_id)
        protocol = (
            (exp.research_config or {}).get("evaluation_protocol")
            if exp
            else EVALUATION_PROTOCOL_VERSION
        ) or EVALUATION_PROTOCOL_VERSION
        n_factors_all = await s.scalar(
            select(func.count(Factor.id)).where(
                Factor.experiment_id == exp_id
            )
        )
        n_factors = await s.scalar(
            select(func.count(Factor.id)).where(
                Factor.experiment_id == exp_id,
                Factor.evaluation_protocol == protocol,
            )
        )
        n_nodes_all = await s.scalar(
            select(func.count(Node.id)).where(Node.experiment_id == exp_id)
        )
        n_nodes = await s.scalar(
            select(func.count(Node.id)).where(
                Node.experiment_id == exp_id,
                Node.evaluation_protocol == protocol,
            )
        )
        n_steps_all = await s.scalar(
            select(func.count(OuterStep.id)).where(
                OuterStep.experiment_id == exp_id
            )
        )
        n_steps = await s.scalar(
            select(func.count(OuterStep.id)).where(
                OuterStep.experiment_id == exp_id,
                OuterStep.evaluation_protocol == protocol,
            )
        )
        accepted = await s.scalar(
            select(func.count(OuterStep.id)).where(
                OuterStep.accepted,
                OuterStep.experiment_id == exp_id,
                OuterStep.evaluation_protocol == protocol,
            )
        )
        inc = await s.scalar(
            select(MinerVersion)
            .where(
                MinerVersion.status == "incumbent",
                MinerVersion.experiment_id == exp_id,
                MinerVersion.evaluation_protocol == protocol,
            )
            .order_by(MinerVersion.id.desc())
        )
    runtime = manager.status_for(exp_id)
    return {
        **runtime,
        "experiment": {"id": exp_id, "name": exp.name if exp else "?",
                       "status": exp.status if exp else "?"},
        "counts": {
            "evaluation_protocol": protocol,
            "factors": n_factors,
            "factors_all": n_factors_all,
            "nodes": n_nodes,
            "nodes_all": n_nodes_all,
            "outer_steps": n_steps,
            "outer_steps_all": n_steps_all,
            "accepted": accepted,
        },
        "incumbent": {
            "version_no": inc.version_no,
            "meta_score": inc.meta_score,
            "spec": inc.harness_spec,
            "evaluation_protocol": inc.evaluation_protocol,
            "feedback_summary": inc.feedback_summary,
            "reflection": inc.reflection,
            "context_fingerprint": inc.context_fingerprint,
        } if inc else None,
        "logs": runtime.get("logs", [])[-60:],
        "workers": manager.all_status(),
    }


def _three_layer_arm_configs() -> list[dict]:
    all_algorithms = list(DEFAULT_SEARCH_ALGORITHMS)
    return [
        {
            "arm": "A",
            "label": "结构化随机",
            "proposal_mode": "search_pool",
            "memory_mode": "cold",
            "layer2_enabled": False,
            "layer3_enabled": False,
            "search_algorithms": ["structured_random"],
            "estimand": "随机语法基线",
        },
        {
            "arm": "B",
            "label": "第一层算法组合",
            "proposal_mode": "search_pool",
            "memory_mode": "cold",
            "layer2_enabled": False,
            "layer3_enabled": False,
            "search_algorithms": all_algorithms,
            "estimand": "B-A = 算法搜索池增量",
        },
        {
            "arm": "C",
            "label": "算法组合+Researcher冷记忆",
            "proposal_mode": "llm",
            "memory_mode": "cold",
            "layer2_enabled": True,
            "layer3_enabled": False,
            "search_algorithms": all_algorithms,
            "estimand": "C-B = 第二层LLM增量",
        },
        {
            "arm": "D",
            "label": "算法组合+Researcher连续记忆",
            "proposal_mode": "llm",
            "memory_mode": "adaptive",
            "layer2_enabled": True,
            "layer3_enabled": False,
            "search_algorithms": all_algorithms,
            "estimand": "D-C = 任务连续记忆增量",
        },
        {
            "arm": "E",
            "label": "完整三层+Governor",
            "proposal_mode": "llm",
            "memory_mode": "adaptive",
            "layer2_enabled": True,
            "layer3_enabled": True,
            "search_algorithms": all_algorithms,
            "estimand": "E-D = 第三层治理LLM增量",
        },
    ]


@router.post("/campaigns/three-layer")
async def create_three_layer_campaign(req: ThreeLayerCampaignReq):
    """Pre-register and optionally start the five-arm three-layer ablation."""
    campaign_id = req.campaign_id.strip()
    if not campaign_id or len(campaign_id) > 96:
        raise HTTPException(400, "campaign_id 必须为 1-96 字符")
    panel_glob = default_panel_glob("us")
    resolved_eval = evaluation_config("us", None)
    tasks = resolve_engine_tasks(
        DEFAULT_ENGINE_CONFIG_V2["tasks"],
        "us",
        "long_short",
        1,
        DEFAULT_RESEARCH_DIRECTION_POLICY,
        preserve_declared_costs=True,
    )
    created: list[int] = []
    experiment_rows: list[Experiment] = []
    async with SessionLocal() as session:
        for arm in _three_layer_arm_configs():
            name = f"{req.name_prefix}-{arm['arm']} {arm['label']}"
            existing = await session.scalar(
                select(Experiment).where(Experiment.name == name)
            )
            if existing:
                existing_cfg = dict(existing.research_config or {})
                if (
                    existing_cfg.get("campaign_id") != campaign_id
                    or existing_cfg.get("architecture_arm") != arm["arm"]
                ):
                    raise HTTPException(
                        409,
                        f"同名任务 {name} 已存在但不属于本实验；未覆盖任何数据",
                    )
                experiment_rows.append(existing)
                continue
            config = {
                "campaign_id": campaign_id,
                "campaign_schema": "three_layer_abcde_v2",
                "architecture_arm": arm["arm"],
                "estimand": arm["estimand"],
                "market": "us",
                "portfolio_mode": "long_short",
                "panel_glob": panel_glob,
                "engine_mode": "v2",
                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                "evaluation_config": resolved_eval,
                "direction": 1,
                "direction_policy": DEFAULT_RESEARCH_DIRECTION_POLICY,
                "proposal_mode": arm["proposal_mode"],
                "memory_mode": arm["memory_mode"],
                "layer1_enabled": True,
                "layer2_enabled": arm["layer2_enabled"],
                "layer3_enabled": arm["layer3_enabled"],
                "search_algorithms": arm["search_algorithms"],
                "target_factor_count": 0,
                "candidate_evaluation_budget": req.candidate_evaluation_budget,
                "target_mechanisms": list(mechanisms_for_market("us")),
                "return_source_governance": resolve_return_source_governance({
                    "protocol": RETURN_SOURCE_GOVERNANCE_PROTOCOL,
                    "correlation_threshold": 0.85,
                    "required_sources": 5,
                    "meta_score_weight": 0.15,
                    "cross_experiment_admission": False,
                }),
                "engine_config": {
                    **DEFAULT_ENGINE_CONFIG_V2,
                    "tasks": tasks,
                    # 5 seeds x 6 candidates = 30 candidates per fixed-policy
                    # step; 120 therefore closes on complete seed blocks.
                    "n_seeds_per_candidate": 5,
                    "paired_cohorts_per_comparison": 5,
                    "max_llm_calls": 60,
                    "max_outer_steps": 6,
                    "max_runtime_hours": 4.0,
                    "batch_candidates_per_call": 6,
                },
                "preregistration": {
                    "primary_metric": "diversity_adjusted_training_meta_score",
                    "secondary_metrics": [
                        "gate_pass_rate",
                        "valid_candidate_rate",
                        "factor_admission_rate",
                        "mechanism_coverage",
                        "training_return_source_coverage",
                        "wall_clock_seconds",
                        "llm_calls",
                    ],
                    "contrasts": ["B-A", "C-B", "D-C", "E-D"],
                    "sealed_data_in_search": False,
                    "rating_window": "2020-01-01..latest_panel_date",
                    "promotion_rule": "research_only_until_separate_frozen_rating",
                    "compute_policy": "five logical workers; max two concurrent evaluations",
                },
            }
            experiment = Experiment(
                name=name,
                description=(
                    f"三层架构 A-E 预注册消融；{arm['estimand']}。"
                    "搜索仅使用训练安全反馈，冻结评级不进入提示词或调度器。"
                ),
                status="open",
                research_config=config,
            )
            session.add(experiment)
            await session.flush()
            created.append(experiment.id)
            experiment_rows.append(experiment)
        await session.commit()

    starts = []
    if req.start:
        for experiment in experiment_rows:
            starts.append({
                "experiment_id": experiment.id,
                **await EngineManager.get().start("v2", experiment.id),
            })
    _invalidate_observability_components()
    return {
        "ok": True,
        "campaign_id": campaign_id,
        "created_experiment_ids": created,
        "experiments": [
            {
                "id": experiment.id,
                "name": experiment.name,
                "arm": (experiment.research_config or {}).get("architecture_arm"),
            }
            for experiment in experiment_rows
        ],
        "starts": starts,
        "parallel_evaluation_limit": int(
            os.environ.get("FF_MAX_PARALLEL_EVALUATIONS", "2")
        ),
    }


@router.get("/campaigns/three-layer/{campaign_id}")
async def three_layer_campaign_status(campaign_id: str):
    async with SessionLocal() as session:
        all_experiments = (await session.scalars(
            select(Experiment).order_by(Experiment.id)
        )).all()
        experiments = [
            row for row in all_experiments
            if (row.research_config or {}).get("campaign_id") == campaign_id
        ]
        if not experiments:
            raise HTTPException(404, "三层实验不存在")
        ids = [row.id for row in experiments]
        nodes = (await session.scalars(
            select(Node).where(Node.experiment_id.in_(ids))
        )).all()
        llm_calls = (await session.scalars(
            select(LLMCallAudit).where(LLMCallAudit.experiment_id.in_(ids))
        )).all()
        factors = (await session.scalars(
            select(Factor).where(Factor.experiment_id.in_(ids))
        )).all()
        versions = (await session.scalars(
            select(MinerVersion).where(MinerVersion.experiment_id.in_(ids))
        )).all()
    payload = []
    for experiment in experiments:
        arm_nodes = [node for node in nodes if node.experiment_id == experiment.id]
        arm_calls = [row for row in llm_calls if row.experiment_id == experiment.id]
        arm_factors = [row for row in factors if row.experiment_id == experiment.id]
        arm_versions = [row for row in versions if row.experiment_id == experiment.id]
        valid_scores = [
            float(node.public_score or 0.0)
            for node in arm_nodes if node.status == "ok"
        ]
        seed_scores: dict[int, list[float]] = {}
        for node in arm_nodes:
            if node.status == "ok":
                seed_scores.setdefault(int(node.seed or 0), []).append(
                    float(node.public_score or 0.0)
                )
        latest_scored_version = max(
            (row for row in arm_versions if row.meta_score is not None),
            key=lambda row: row.id,
            default=None,
        )
        algorithms: dict[str, int] = {}
        for node in arm_nodes:
            algorithm = str((node.proposal_meta or {}).get("search_algorithm") or "unknown")
            algorithms[algorithm] = algorithms.get(algorithm, 0) + 1
        payload.append({
            "arm": (experiment.research_config or {}).get("architecture_arm"),
            "experiment_id": experiment.id,
            "name": experiment.name,
            "candidate_budget": int(
                (experiment.research_config or {}).get(
                    "candidate_evaluation_budget", 0
                ) or 0
            ),
            "runtime": EngineManager.get().status_for(experiment.id, include_logs=False),
            "nodes": len(arm_nodes),
            "valid_nodes": sum(node.status == "ok" for node in arm_nodes),
            "llm_calls": len(arm_calls),
            "llm_transport_errors": sum(row.status == "transport_error" for row in arm_calls),
            "llm_semantic_rejections": sum(row.status == "rejected" for row in arm_calls),
            "factor_count": len(arm_factors),
            "mean_public_score": (
                round(st.mean(valid_scores), 6) if valid_scores else None
            ),
            "latest_meta_score": (
                round(float(latest_scored_version.meta_score), 6)
                if latest_scored_version is not None else None
            ),
            "seed_mean_public_scores": {
                str(seed): round(st.mean(scores), 6)
                for seed, scores in sorted(seed_scores.items())
            },
            "search_algorithms": algorithms,
        })
    ordered = sorted(payload, key=lambda row: row["arm"])
    contrasts = []
    for left, right in zip(ordered, ordered[1:]):
        left_seed = left["seed_mean_public_scores"]
        right_seed = right["seed_mean_public_scores"]
        shared = sorted(set(left_seed) & set(right_seed))
        paired = [right_seed[key] - left_seed[key] for key in shared]
        contrasts.append({
            "contrast": f"{right['arm']}-{left['arm']}",
            "mean_public_score_delta": (
                round(float(right["mean_public_score"]) - float(left["mean_public_score"]), 6)
                if right["mean_public_score"] is not None
                and left["mean_public_score"] is not None else None
            ),
            "paired_seed_delta_mean": round(st.mean(paired), 6) if paired else None,
            "paired_seed_count": len(paired),
            "valid_rate_delta": round(
                right["valid_nodes"] / max(1, right["nodes"])
                - left["valid_nodes"] / max(1, left["nodes"]),
                6,
            ),
            "factor_count_delta": right["factor_count"] - left["factor_count"],
            "incremental_llm_calls": right["llm_calls"] - left["llm_calls"],
            "final_inference_ready": bool(
                not left["runtime"].get("running")
                and not right["runtime"].get("running")
                and left["nodes"] >= left["candidate_budget"] > 0
                and right["nodes"] >= right["candidate_budget"] > 0
            ),
        })
    return {
        "campaign_id": campaign_id,
        "arms": ordered,
        "contrasts": contrasts,
        "inference_warning": (
            "运行中指标仅用于运维，不得提前选择胜者；"
            "仅 final_inference_ready=true 后执行预注册对比。"
        ),
    }


@router.get("/engine/progress")
async def engine_progress(experiment_id: int | None = None):
    """外层 meta-score 步进序列 (可视化)."""
    exp_id = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        exp = await s.get(Experiment, exp_id)
        protocol = (
            (exp.research_config or {}).get("evaluation_protocol")
            if exp
            else EVALUATION_PROTOCOL_VERSION
        ) or EVALUATION_PROTOCOL_VERSION
        rows = (await s.scalars(
            select(OuterStep).where(
                OuterStep.experiment_id == exp_id,
                OuterStep.evaluation_protocol == protocol,
            ).order_by(OuterStep.step_no))).all()
    return {
        "evaluation_protocol": protocol,
        "steps": [
            {"step": r.step_no, "candidate": r.candidate_score, "incumbent": r.incumbent_score,
             "accepted": r.accepted, "note": r.detail.get("note", ""),
             "evaluation_protocol": r.evaluation_protocol,
             "reflection": r.detail.get("outcome_reflection", {}),
             "comparison": r.detail.get("comparison", {})}
            for r in rows
        ]
    }


# ---------- 研发树 ----------

@router.get("/tree")
async def research_tree(miner_version_id: int | None = None, experiment_id: int | None = None, limit: int = 800):
    exp_id = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        q = select(Node).where(Node.experiment_id == exp_id).order_by(Node.id.desc()).limit(limit)
        if miner_version_id:
            q = q.where(Node.miner_version_id == miner_version_id)
        nodes = list(reversed((await s.scalars(q)).all()))
        versions = (await s.scalars(
            select(MinerVersion).where(MinerVersion.experiment_id == exp_id).order_by(MinerVersion.id))).all()
    return {
        "versions": [
            {"id": v.id, "version_no": v.version_no, "status": v.status, "meta_score": v.meta_score,
             "note": v.proposal_note, "spec": v.harness_spec, "parent_id": v.parent_id,
             "evaluation_protocol": v.evaluation_protocol,
             "feedback_summary": v.feedback_summary,
             "reflection": v.reflection,
             "context_fingerprint": v.context_fingerprint}
            for v in versions
        ],
        "nodes": [
            {"id": n.id, "parent_id": n.parent_id, "miner_version_id": n.miner_version_id,
             "op": n.op, "expression": n.expression,
             "hypothesis": n.hypothesis, "status": n.status,
             "error": redact_text(n.error, 1200),
             "public_score": n.public_score, "source": n.source, "task": n.task_name,
             "learning_score": n.public_score,
             "gate_score": float(
                 ((n.public_metrics or {}).get("discovery") or {}).get(
                     "gate_score",
                     0.0,
                 )
                 or 0.0
             ),
             "selected_direction": int(
                 ((n.public_metrics or {}).get("discovery") or {}).get(
                     "selected_direction",
                     (n.feedback_summary or {}).get(
                         "direction",
                         (n.public_metrics or {}).get("direction", 1),
                     ),
                 )
                 or 1
             ),
             "direction_policy": (
                 ((n.public_metrics or {}).get("discovery") or {}).get(
                     "direction_policy"
                 )
             ),
             "direction_selection": (
                 ((n.public_metrics or {}).get("discovery") or {}).get(
                     "direction_selection"
                 )
             ),
             "score_semantics": (
                 ((n.public_metrics or {}).get("discovery") or {}).get(
                     "score_semantics"
                 )
             ),
             "outer_step": n.outer_step_no, "created_at": str(n.created_at),
             "evaluation_protocol": n.evaluation_protocol, "seed": n.seed,
             "proposal_meta": n.proposal_meta,
             "feedback_summary": n.feedback_summary}
            for n in nodes
        ],
    }


@router.get("/llm/audits")
async def llm_call_audits(
    experiment_id: int | None = None,
    role: str | None = None,
    status: str | None = None,
    limit: int = 100,
    include_content: bool = False,
):
    """Secret-safe prompt lineage for debugging the two-layer feedback loop."""
    if not 1 <= limit <= 500:
        raise HTTPException(400, "limit 必须在 1..500")
    exp_id = experiment_id or await get_active_experiment_id()
    query = select(LLMCallAudit).where(
        LLMCallAudit.experiment_id == exp_id
    )
    if role:
        query = query.where(LLMCallAudit.role == role)
    if status:
        query = query.where(LLMCallAudit.status == status)
    async with SessionLocal() as session:
        rows = (
            await session.scalars(
                query.order_by(LLMCallAudit.id.desc()).limit(limit)
            )
        ).all()
    return {
        "experiment_id": exp_id,
        "include_content": include_content,
        "calls": [
            {
                "id": row.id,
                "role": row.role,
                "phase": row.phase,
                "status": row.status,
                "provider_name": row.provider_name,
                "model": row.model,
                "evaluation_protocol": row.evaluation_protocol,
                "miner_version_id": row.miner_version_id,
                "outer_step_no": row.outer_step_no,
                "task_name": row.task_name,
                "prompt_hash": row.prompt_hash,
                "feedback_fingerprint": row.feedback_fingerprint,
                "latency_ms": row.latency_ms,
                "error": redact_text(row.error, 1200),
                "trace_meta": redact_value(row.trace_meta or {}),
                "created_at": str(row.created_at),
                **({
                    "system_prompt": redact_text(row.system_prompt, 24_000),
                    "user_prompt": redact_text(row.user_prompt, 32_000),
                    "response": redact_text(row.response, 16_000),
                } if include_content else {}),
            }
            for row in rows
        ],
    }


# ---------- 任务研究记录库（非正式因子库） ----------

async def _ranked_research_records(experiment_id: int) -> list[dict]:
    """Build one training-safe, task-ranked view over immutable search nodes."""
    async with SessionLocal() as session:
        nodes = (
            await session.scalars(
                select(Node).where(
                    Node.experiment_id == experiment_id,
                    Node.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
                )
            )
        ).all()
        factors = (
            await session.scalars(
                select(Factor).where(
                    Factor.experiment_id == experiment_id,
                    Factor.node_id.is_not(None),
                )
            )
        ).all()
    factor_by_node = {
        int(row.node_id): row.id for row in factors if row.node_id is not None
    }
    formal_by_node = {}
    for row in factors:
        if row.node_id is None:
            continue
        market = str((row.research_meta or {}).get("market") or "us")
        evidence = factor_evidence_state(
            row,
            current_code_review=deterministic_code_review(
                row.expression, market
            ),
            current_rating_protocol=FROZEN_RATING_PROTOCOL_VERSION,
        )
        if evidence["formal_factor"]:
            formal_by_node[int(row.node_id)] = row.id
    grouped: dict[str, list[Node]] = {}
    for node in nodes:
        grouped.setdefault(node.task_name or "unknown", []).append(node)
    records: list[dict] = []
    for task_nodes in grouped.values():
        task_nodes.sort(
            key=lambda node: (
                node.status == "ok",
                float(node.public_score or 0.0),
                int(node.id or 0),
            ),
            reverse=True,
        )
        for rank, node in enumerate(task_nodes, start=1):
            records.append(
                research_record_payload(
                    node,
                    task_rank=rank,
                    formal_factor_id=formal_by_node.get(int(node.id)),
                    research_factor_id=factor_by_node.get(int(node.id)),
                )
            )
    records.sort(
        key=lambda row: (
            row.get("status") == "ok",
            float(row.get("learning_score") or 0.0),
            int(row.get("id") or 0),
        ),
        reverse=True,
    )
    return records


@router.get("/research-records")
async def list_research_records(
    experiment_id: int | None = None,
    task_name: str | None = None,
    source: str | None = None,
    passed: bool | None = None,
    status: str | None = None,
    q: str | None = None,
    offset: int = 0,
    limit: int = 100,
):
    if offset < 0:
        raise HTTPException(400, "offset 不能为负数")
    if not 1 <= limit <= 500:
        raise HTTPException(400, "limit 必须在 1..500")
    exp_id = experiment_id or await get_active_experiment_id()
    records = await _ranked_research_records(exp_id)
    summaries = task_research_summary(records)
    filtered = records
    if task_name:
        filtered = [row for row in filtered if row.get("task_name") == task_name]
    if source:
        filtered = [row for row in filtered if row.get("source") == source]
    if passed is not None:
        filtered = [row for row in filtered if row.get("discovery_passed") is passed]
    if status:
        filtered = [row for row in filtered if row.get("status") == status]
    if q:
        needle = q.casefold()
        filtered = [
            row for row in filtered
            if needle in " ".join([
                str(row.get("expression") or ""),
                str(row.get("hypothesis") or ""),
                str(row.get("mechanism_family") or ""),
                str(row.get("task_name") or ""),
            ]).casefold()
        ]
    return {
        "schema_version": RESEARCH_RECORD_SCHEMA_VERSION,
        "experiment_id": exp_id,
        "interpretation_boundary": (
            "training research records only; ranking is task-local and does not "
            "grant formal factor, holdout, vault, paper, live, or production approval"
        ),
        "tasks": summaries,
        "total": len(filtered),
        "offset": offset,
        "limit": limit,
        "records": filtered[offset:offset + limit],
    }


@router.get("/research-records/{node_id}")
async def research_record_detail(node_id: int, experiment_id: int | None = None):
    exp_id = experiment_id or await get_active_experiment_id()
    records = await _ranked_research_records(exp_id)
    record = next((row for row in records if row.get("id") == node_id), None)
    if record is None:
        raise HTTPException(404, "研究记录不存在或不属于当前任务")
    async with SessionLocal() as session:
        children = (
            await session.scalars(
                select(Node).where(
                    Node.experiment_id == exp_id,
                    Node.parent_id == node_id,
                    Node.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
                ).order_by(Node.id.asc())
            )
        ).all()
    record_by_id = {row["id"]: row for row in records}
    return {
        **record,
        "parent": record_by_id.get(record.get("parent_id")),
        "children": [record_by_id[row.id] for row in children if row.id in record_by_id],
    }


# ---------- 因子库 ----------

@router.get("/factors")
async def list_factors(
    status: str | None = None,
    lifecycle: str | None = None,
    experiment_id: int | None = None,
    q: str | None = None,
    sort: str = "live_rank",
    direction: str = "desc",
    limit: int = 500,
):
    exp_id = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        query = select(Factor).where(Factor.experiment_id == exp_id).order_by(Factor.id.desc())
        if status:
            query = query.where(Factor.status == status)
        if lifecycle:
            query = query.where(Factor.lifecycle_stage == lifecycle)
        if q:
            needle = f"%{q}%"
            query = query.where((Factor.name.ilike(needle)) | (Factor.expression.ilike(needle)) | (Factor.hypothesis.ilike(needle)))
        query = query.limit(max(1, min(limit, 2000)))
        rows = (await s.scalars(query)).all()
    if sort == "live_rank":
        def live_rank_key(factor: Factor) -> tuple:
            validation = factor.validation_metrics or {}
            ranking = validation.get("ranking") or {}
            grade = str((factor.eligibility or {}).get("grade") or "F0")
            try:
                grade_number = int(grade.removeprefix("F"))
            except ValueError:
                grade_number = 0
            is_current_audit = (
                factor.evaluation_protocol == EVALUATION_PROTOCOL_VERSION
                and ranking.get("rating_protocol_version")
                == FROZEN_RATING_PROTOCOL_VERSION
                and bool(ranking.get("available"))
                and ranking.get("score") is not None
            )
            return (
                int(is_current_audit),
                float(ranking.get("score") or -1.0),
                grade_number,
                float((factor.public_metrics or {}).get("score") or 0.0),
            )

        rows.sort(key=live_rank_key, reverse=direction != "asc")
    elif sort == "score":
        rows.sort(key=lambda f: float((f.public_metrics or {}).get("score") or 0), reverse=direction != "asc")
    elif sort == "icir":
        rows.sort(key=lambda f: float((f.public_metrics or {}).get("icir") or 0), reverse=direction != "asc")
    elif sort == "grade":
        rows.sort(
            key=lambda f: str((f.eligibility or {}).get("grade") or "F0"),
            reverse=direction != "asc",
        )
    payloads = [_factor_payload(f) for f in rows]
    if sort == "live_rank":
        position = 0
        for payload in payloads:
            ranking = payload.get("ranking") or {}
            if ranking.get("available") and ranking.get("current"):
                position += 1
                ranking["position"] = position
    return {
        "factors": payloads,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "sort": sort,
    }


@router.get("/factors/ranking-diagnostics")
async def factor_ranking_diagnostics(experiment_id: int | None = None):
    """Explain why Vault calibration is unavailable for the V4.3 rating."""
    eid, cfg = await _experiment_context(experiment_id)
    if FROZEN_RATING_PROTOCOL_VERSION == "v4.3":
        return {
            "experiment_id": eid,
            "market": cfg.get("market", "us"),
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
            "rating_protocol_version": FROZEN_RATING_PROTOCOL_VERSION,
            "status": "not_applicable_full_window_rating",
            "sample_size": 0,
            "minimum_sample": 0,
            "metrics": None,
            "message": (
                "V4.3 冻结评级覆盖 2020 至最新交易日，已包含 Vault 日期；"
                "不能再用同一 Vault 对该评级做独立校准。HOLDOUT/Vault 仍作为硬门槛。"
            ),
            "basis": "full-history rating; no same-sample vault calibration",
            "uses_vault_to_tune_score": True,
        }
    async with SessionLocal() as s:
        rows = (
            await s.scalars(
                select(Factor).where(
                    Factor.experiment_id == eid,
                    Factor.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
                )
            )
        ).all()
    items = []
    for factor in rows:
        provenance_status = (factor.provenance_status or "").lower()
        if (
            "invalid" in provenance_status
            or factor.lifecycle_stage == "configuration_changed_requires_reaudit"
        ):
            continue
        validation = factor.validation_metrics or {}
        ranking = validation.get("ranking") or {}
        vault = (validation.get("layers") or {}).get("vault") or {}
        # Profit means fee-after absolute P&L for A-share long-only and fee-after
        # market-neutral P&L for US long/short.  In both cases that is `net`.
        outcome = vault.get("net") or {}
        if ranking.get("available"):
            items.append({
                "factor_id": factor.id,
                "score_pre_vault": ranking.get("score_pre_vault"),
                "vault_ann_return": outcome.get("ann_return"),
                "vault_sharpe": outcome.get("sharpe"),
            })
    return {
        "experiment_id": eid,
        "market": cfg.get("market", "us"),
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        **ranking_diagnostics(items),
    }


async def _factor_similarity_index(
    experiment_id: int,
    threshold: float = 0.64,
) -> tuple[dict, list[dict]]:
    async with SessionLocal() as s:
        rows = (await s.scalars(
            select(Factor)
            .where(Factor.experiment_id == experiment_id)
            .order_by(Factor.id)
        )).all()
    items = [
        {
            "id": factor.id,
            "name": factor.name,
            "expression": factor.expression,
            "score": (factor.public_metrics or {}).get("score", 0),
            "grade": (factor.eligibility or {}).get("grade"),
            "lifecycle_stage": factor.lifecycle_stage,
        }
        for factor in rows
    ]
    key = (
        experiment_id,
        len(items),
        max((item["id"] for item in items), default=0),
        round(threshold, 4),
    )
    index = _similarity_cache.get(key)
    if index is not None:
        _similarity_cache_stats["hits"] += 1
        _similarity_cache.move_to_end(key)
        return index, items
    _similarity_cache_stats["misses"] += 1
    async with _similarity_cache_lock:
        index = _similarity_cache.get(key)
        if index is not None:
            _similarity_cache_stats["hits"] += 1
            _similarity_cache.move_to_end(key)
            return index, items
        started = time.perf_counter()
        index = await asyncio.to_thread(build_similarity_index, items, threshold)
        _similarity_cache_stats["builds"] += 1
        _similarity_cache_stats["last_build_ms"] = round(
            (time.perf_counter() - started) * 1000.0,
            3,
        )
        _similarity_cache_stats["last_build_at"] = datetime.now(
            timezone.utc
        ).isoformat(timespec="milliseconds")
        _similarity_cache[key] = index
        _similarity_cache.move_to_end(key)
        while len(_similarity_cache) > 24:
            _similarity_cache.popitem(last=False)
            _similarity_cache_stats["evictions"] += 1
    return index, items


@router.get("/factors/similarity-groups")
async def factor_similarity_groups(
    experiment_id: int | None = None,
    threshold: float = 0.64,
    include_singletons: bool = True,
):
    eid = experiment_id or await get_active_experiment_id()
    try:
        index, _ = await _factor_similarity_index(eid, threshold)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    groups = index["groups"]
    if not include_singletons:
        groups = [group for group in groups if group["size"] > 1]
    return {
        "experiment_id": eid,
        "groups": groups,
        "stats": index["stats"],
    }


@router.get("/factors/{fid}/similar")
async def similar_factors(fid: int, limit: int = 20):
    async with SessionLocal() as s:
        factor = await s.get(Factor, fid)
    if not factor:
        raise HTTPException(404, "因子不存在")
    index, items = await _factor_similarity_index(factor.experiment_id)
    nearest = await asyncio.to_thread(nearest_factors, items, fid, limit)
    return {
        "factor_id": fid,
        "group_id": index["factor_to_group"].get(fid),
        "similar": nearest,
        "algorithm": index["stats"]["algorithm"],
    }


@router.get("/factors/{fid}/detail")
async def factor_detail(fid: int):
    async with SessionLocal() as s:
        f = await s.get(Factor, fid)
        if not f:
            raise HTTPException(404)
        exp = await s.get(Experiment, f.experiment_id)
        global_engine = await s.get(Setting, "engine_config")
    cfg = (exp.research_config or {}) if exp else {}
    market = cfg.get("market", "us")
    resolved_eval = evaluation_config(market, cfg.get("evaluation_config"))
    task_pool = _resolved_task_pool(cfg, global_engine)
    task = next((row for row in task_pool if row.get("name") == f.task_name), {})
    audit_defaults = {
        "universe_n": int(task.get("universe_n", 500)),
        "horizon": int(task.get("horizon", 5)),
        "direction": int(
            (f.research_meta or {}).get(
                "direction",
                task.get("direction", cfg.get("direction", 1)),
            )
        ),
        "direction_policy": DIRECTION_POLICY_FIXED,
        "cost_bps": float(task.get("cost_bps", resolved_eval["base_cost_bps"])),
        "target_capital": float(resolved_eval["target_capital"]),
        "market": market,
        "portfolio_mode": cfg.get(
            "portfolio_mode",
            "long_only" if market == "ashare" else "long_short",
        ),
    }
    # Detail reads are deliberately O(1).  Full four-layer evaluation is an
    # explicit POST /audit action and its result is persisted on the factor.
    try:
        dsl_profile = expression_profile(f.expression)
        similarity_index, items = await _factor_similarity_index(f.experiment_id)
        similar = await asyncio.to_thread(nearest_factors, items, f.id, 12)
    except (SyntaxError, ValueError):
        dsl_profile = {"latex": "", "operators": [], "fields": [], "windows": []}
        similarity_index = {"factor_to_group": {}}
        similar = []
    return {
        "factor": _factor_payload(f, include_validation=True),
        "audit_defaults": audit_defaults,
        "dsl": dsl_profile,
        "semantic_audit": audit_expression_semantics(f.expression, market),
        "mechanism_family": (f.research_meta or {}).get(
            "mechanism_family",
            infer_mechanism(f.expression, f.hypothesis),
        ),
        "similarity": {
            "group_id": similarity_index["factor_to_group"].get(f.id),
            "nearest": similar,
        },
    }


class FactorAuditReq(BaseModel):
    universe_n: int | None = None
    horizon: int | None = None
    direction: int | None = None
    cost_bps: float | None = None
    target_capital: float | None = None


@router.post("/factors/{fid}/audit")
async def audit_factor(fid: int, req: FactorAuditReq | None = None):
    """Run and persist an explicit four-layer V4 audit.

    This endpoint is intentionally separate from mining so HOLDOUT/VAULT never
    become proposal feedback.
    """
    req = req or FactorAuditReq()
    async with SessionLocal() as s:
        f = await s.get(Factor, fid)
        if not f:
            raise HTTPException(404, "因子不存在")
        exp = await s.get(Experiment, f.experiment_id)
        if not exp:
            raise HTTPException(404, "研究任务不存在")
        global_engine = await s.get(Setting, "engine_config")
        actual_trials = int(
            await s.scalar(
                select(func.count(Trial.id)).where(
                    Trial.experiment_id == f.experiment_id
                )
            )
            or 0
        )
    cfg = exp.research_config or {}
    market = cfg.get("market", "us")
    mode = cfg.get(
        "portfolio_mode",
        "long_only" if market == "ashare" else "long_short",
    )
    task_pool = _resolved_task_pool(cfg, global_engine)
    task = next((row for row in task_pool if row.get("name") == f.task_name), {})
    universe_n = req.universe_n or int(task.get("universe_n", 500))
    horizon = req.horizon or int(task.get("horizon", 5))
    direction = req.direction or int(
        (f.research_meta or {}).get(
            "direction",
            task.get("direction", cfg.get("direction", 1)),
        )
    )
    if direction not in {-1, 1}:
        raise HTTPException(400, "direction 必须为 1 或 -1")
    if req.cost_bps is not None and req.cost_bps < 0:
        raise HTTPException(400, "cost_bps 不能为负数")
    factor_meta = dict(f.research_meta or {})
    direction_trials_multiplier = max(
        1,
        int(factor_meta.get("direction_trials_multiplier") or 1),
    )
    overrides = evaluation_config(market, cfg.get("evaluation_config"))
    overrides["multiple_testing_trials"] = max(
        int(overrides["multiple_testing_trials"]),
        actual_trials * direction_trials_multiplier,
        1,
    )
    if req.target_capital is not None:
        if req.target_capital <= 0:
            raise HTTPException(400, "target_capital 必须为正数")
        overrides["target_capital"] = req.target_capital
    try:
        audit = await asyncio.to_thread(
            evaluate_full,
            f.expression,
            universe_n,
            horizon,
            mode,
            direction,
            cfg.get("panel_glob"),
            req.cost_bps if req.cost_bps is not None else task.get("cost_bps"),
            market,
            overrides,
            DIRECTION_POLICY_FIXED,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    provenance_warning = cfg.get("provenance_warning")
    if provenance_warning:
        audit["source_provenance_warning"] = provenance_warning
        provenance_status = "revalidated_expression_source_invalid"
    else:
        provenance_status = "v4_validated_on_configured_panel"
    async with SessionLocal() as s:
        factor = await s.get(Factor, fid)
        meta = dict(factor.research_meta or {})
        previous_validation = factor.validation_metrics or {}
        if previous_validation:
            audit_history = list(meta.get("audit_history") or [])
            audit_history.append({
                "evaluation_protocol": factor.evaluation_protocol,
                "evaluated_at": str(factor.evaluated_at) if factor.evaluated_at else None,
                "eligibility": factor.eligibility or {},
                "validation": previous_validation,
            })
            meta["audit_history"] = audit_history
        factor.public_metrics = {
            **audit["public"],
            "discovery": audit["discovery"],
            "protocol_version": EVALUATION_PROTOCOL_VERSION,
        }
        factor.gate_metrics = audit["gate"]
        factor.validation_metrics = audit
        factor.eligibility = audit["eligibility"]
        factor.evaluation_protocol = EVALUATION_PROTOCOL_VERSION
        factor.lifecycle_stage = audit["eligibility"]["stage"]
        factor.provenance_status = provenance_status
        factor.evaluated_at = datetime.utcnow()
        factor.fingerprint = {
            **(factor.fingerprint or {}),
            **expression_fingerprint(factor.expression),
        }
        meta.update({
            "direction": direction,
            "last_audit_universe_n": universe_n,
            "last_audit_horizon": horizon,
            "policy_label": "NON_PIT_RESEARCH",
            "multiple_testing_trials": overrides["multiple_testing_trials"],
            "direction_trials_multiplier": direction_trials_multiplier,
        })
        factor.research_meta = meta
        s.add(Trial(
            experiment_id=factor.experiment_id,
            expression_hash=(factor.fingerprint or {}).get("expr_hash", "audit"),
            layer="FULL_AUDIT_V4",
            task_name=factor.task_name,
            node_id=factor.node_id,
            expression=factor.expression,
            search_method="manual_full_audit",
            mechanism=infer_mechanism(factor.expression),
            selected=bool(audit["eligibility"].get("eligible")),
            failure_reason="; ".join(audit["eligibility"].get("reasons") or []),
            statistic={
                "grade": audit["eligibility"]["grade"],
                "stage": audit["eligibility"]["stage"],
                "live_rank_score": audit["ranking"].get("score"),
                "score_frozen_rating": audit["ranking"].get(
                    "score_frozen_rating"
                ),
                "score_pre_vault": audit["ranking"].get("score_pre_vault"),
                "rating_window": audit["ranking"].get("rating_window"),
                "rating_protocol_version": FROZEN_RATING_PROTOCOL_VERSION,
                "direction": direction,
                "direction_policy": DIRECTION_POLICY_FIXED,
                "direction_trials_multiplier": direction_trials_multiplier,
                "protocol_version": EVALUATION_PROTOCOL_VERSION,
            },
        ))
        await s.commit()
        await s.refresh(factor)
    _similarity_cache.clear()
    return {"factor": _factor_payload(factor, include_validation=True)}


class FactorStatusReq(BaseModel):
    status: str


@router.post("/factors/{fid}/status")
async def set_factor_status(fid: int, req: FactorStatusReq):
    allowed = {
        "public-leading",
        "library-admitted",
        "public-gate-pass",
        "paper",
        "retired",
        "research-pass",
        "oos-pass",
        "live-candidate",
        "research-candidate",
    }
    if req.status not in allowed:
        raise HTTPException(400, f"status 必须是 {allowed}")
    async with SessionLocal() as s:
        f = await s.get(Factor, fid)
        if not f:
            raise HTTPException(404)
        if req.status in {"library-admitted", "live-candidate"}:
            market = str((f.research_meta or {}).get("market") or "us")
            evidence = factor_evidence_state(
                f,
                current_code_review=deterministic_code_review(
                    f.expression, market
                ),
                current_rating_protocol=FROZEN_RATING_PROTOCOL_VERSION,
            )
            if not evidence["formal_factor"]:
                raise HTTPException(
                    409,
                    "该记录仍是研究候选，未通过 HOLDOUT/Vault/冻结评级/"
                    "双盲审查全部门槛，不能标记为正式或实盘候选",
                )
        f.status = req.status
        await s.commit()
    _invalidate_observability_components()
    return {"ok": True}


class FactorReviewReq(BaseModel):
    tags: list[str] | None = None
    note: str | None = None
    direction: int | None = None
    decision: str | None = None


@router.patch("/factors/{fid}/review")
async def review_factor(fid: int, req: FactorReviewReq):
    async with SessionLocal() as s:
        f = await s.get(Factor, fid)
        if not f:
            raise HTTPException(404)
        meta = dict(f.research_meta or {})
        if req.tags is not None:
            meta["tags"] = sorted({t.strip() for t in req.tags if t.strip()})
        if req.note is not None:
            meta["note"] = req.note[:2000]
        previous_direction = int(meta.get("direction", 1))
        if req.direction in {-1, 1}:
            meta["direction"] = req.direction
            if req.direction != previous_direction:
                meta["metrics_stale_reason"] = "factor_direction_changed"
                f.lifecycle_stage = "configuration_changed_requires_reaudit"
                f.eligibility = {}
                f.evaluated_at = None
        if req.decision:
            meta["decision"] = req.decision
        f.research_meta = meta
        await s.commit()
    return {"ok": True, "research_meta": meta}


class FactorCompareReq(BaseModel):
    factor_ids: list[int] = Field(default_factory=list)
    expressions: list[str] = Field(default_factory=list)
    experiment_id: int | None = None
    universe_n: int = 500
    horizon: int = 5
    portfolio_mode: str | None = None
    direction: int | None = None


def _latest_factor_correlation(
    expressions: list[str],
    directions: list[int],
    panel_glob: str | None,
    universe_n: int,
    market: str,
) -> dict:
    import polars as pl
    df = PanelStore.get(panel_glob, market).ensure_loaded()
    fields = get_dsl_fields(market)
    target = df["trade_date"].max()
    merged = None
    for i, (expression, direction) in enumerate(
        zip(expressions, directions)
    ):
        frame = parse(expression, fields).apply(df.lazy()).filter(
            (pl.col("trade_date") == target) & (pl.col("univ_rank") <= universe_n)
        ).select(
            "ts_code",
            (pl.col("factor") * direction).alias(f"f{i}"),
        ).collect()
        merged = frame if merged is None else merged.join(frame, on="ts_code", how="inner")
    if merged is None or merged.height < 30:
        return {"date": str(target), "n": 0, "matrix": []}
    matrix = []
    for i in range(len(expressions)):
        row = []
        for j in range(len(expressions)):
            row.append(round(float(merged.select(pl.corr(f"f{i}", f"f{j}")).item() or 0), 4))
        matrix.append(row)
    return {"date": str(target), "n": merged.height, "matrix": matrix}


@router.post("/factors/compare")
async def compare_factors(req: FactorCompareReq):
    eid, cfg = await _experiment_context(req.experiment_id)
    async with SessionLocal() as s:
        factors = (
            await s.scalars(select(Factor).where(
                Factor.id.in_(req.factor_ids),
                Factor.experiment_id == eid,
            ))
        ).all() if req.factor_ids else []
    default_direction = req.direction or int(cfg.get("direction", 1))
    items = [
        (expression, default_direction)
        for expression in req.expressions
    ] + [
        (
            factor.expression,
            req.direction or int(
                (factor.research_meta or {}).get("direction", cfg.get("direction", 1))
            ),
        )
        for factor in factors
    ]
    deduplicated: dict[str, int] = {}
    for expression, direction in items:
        deduplicated.setdefault(expression, direction)
    items = list(deduplicated.items())[:12]
    if not items:
        raise HTTPException(400, "至少选择一个因子或表达式")
    if any(direction not in {-1, 1} for _, direction in items):
        raise HTTPException(400, "direction 必须为 1 或 -1")
    expressions = [expression for expression, _ in items]
    directions = [direction for _, direction in items]
    mode = req.portfolio_mode or cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE)
    market = cfg.get("market", "us")
    resolved_eval = evaluation_config(market, cfg.get("evaluation_config"))
    results = []
    for expression, direction in items:
        try:
            metrics = await asyncio.to_thread(
                evaluate,
                expression,
                req.universe_n,
                req.horizon,
                mode,
                direction,
                cfg.get("panel_glob"),
                resolved_eval["base_cost_bps"],
                market,
                cfg.get("evaluation_config"),
                DIRECTION_POLICY_FIXED,
            )
            results.append({"expression": expression, "direction": direction, **metrics})
        except ValueError as exc:
            results.append({"expression": expression, "direction": direction, "error": str(exc)})
    corr = await asyncio.to_thread(
        _latest_factor_correlation,
        expressions,
        directions,
        cfg.get("panel_glob"),
        req.universe_n,
        market,
    )
    return {
        "experiment_id": eid,
        "portfolio_mode": mode,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "results": results,
        "correlation": corr,
        "correlation_semantics": "direction_adjusted_signal",
    }


class EvalReq(BaseModel):
    expression: str
    experiment_id: int | None = None
    universe_n: int = 500
    horizon: int = 5
    portfolio_mode: str | None = None
    direction: int = 1
    direction_policy: str = DEFAULT_RESEARCH_DIRECTION_POLICY
    panel_glob: str | None = None
    cost_bps: float | None = None
    full_audit: bool = False


@router.post("/factors/evaluate")
async def manual_evaluate(req: EvalReq):
    _, cfg = await _experiment_context(req.experiment_id)
    market = cfg.get("market", "us")
    err = validate(req.expression, get_dsl_fields(market))
    if err:
        raise HTTPException(400, f"表达式非法: {err}")
    try:
        mode = req.portfolio_mode or cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE)
        panel_glob = req.panel_glob or cfg.get("panel_glob")
        evaluator = evaluate_full if req.full_audit else evaluate
        metrics = await asyncio.to_thread(
            evaluator,
            req.expression,
            req.universe_n,
            req.horizon,
            mode,
            req.direction,
            panel_glob,
            req.cost_bps,
            market,
            cfg.get("evaluation_config"),
            req.direction_policy,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return metrics


# ---------- 回测 ----------

def _resolve_manual_backtest_execution(
    requested_mode: str | None,
    configured_mode: str,
    market: str,
    requested_borrow_cost: float | None,
    configured_borrow_cost: float,
) -> tuple[str, float]:
    mode = requested_mode or configured_mode
    if mode not in {"long_only", "long_short"}:
        raise ValueError("mode 必须是 long_only 或 long_short")
    if market == "ashare" and mode != "long_only":
        raise ValueError("A 股手动回测仅支持 long_only")
    if mode == "long_only":
        return mode, 0.0
    borrow_cost = (
        requested_borrow_cost
        if requested_borrow_cost is not None
        else configured_borrow_cost
    )
    if borrow_cost < 0:
        raise ValueError("年化借券成本不能为负数")
    return mode, float(borrow_cost)

class ExitPolicyReq(BaseModel):
    fixed_stop_loss_pct: float | None = None
    fixed_take_profit_pct: float | None = None
    trailing_stop_pct: float | None = None
    atr_period: int = 14
    atr_stop_multiple: float | None = None
    atr_take_profit_multiple: float | None = None
    atr_trailing_multiple: float | None = None
    break_even_activation_pct: float | None = None
    time_stop_sessions: int | None = None
    intrabar_conflict_policy: str = "conservative"


class BacktestFactorReq(BaseModel):
    name: str = ""
    expression: str
    weight: float = Field(default=1.0, gt=0)
    direction: int = 1


class BacktestReq(BaseModel):
    expression: str = ""
    factors: list[BacktestFactorReq] = Field(default_factory=list)
    combination_method: str = "independent_capital_sleeves"
    execution_backend: str | None = None
    experiment_id: int | None = None
    universe_n: int = 500
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    cost_bps: float | None = None  # legacy compatibility: interpreted as slippage only
    direction: int = 1
    mode: str | None = None
    panel_glob: str | None = None
    borrow_cost_bps_annual: float | None = None
    top_fraction: float = 0.20
    initial_capital: float = 1_000_000.0
    rebalance_every: int = 5
    slippage_bps: float | None = None
    max_volume_participation: float = 0.10
    fee_profile: str | None = None
    account_type: str = "auto"
    cash_buffer_fraction: float = 0.0
    max_gross_leverage: float = 2.0
    margin_interest_bps_annual: float = 0.0
    position_sizing: str = "equal_weight"
    max_positions: int = 10_000
    max_position_weight: float = 1.0
    min_trade_notional: float = 0.0
    rebalance_buffer_pct: float = 0.0
    long_gross_target: float = 1.0
    short_gross_target: float | None = None
    risk_per_position_fraction: float = 0.01
    spread_bps: float = 0.0
    impact_model: str = "fixed"
    impact_coefficient_bps: float = 0.0
    unfilled_order_policy: str = "cancel"
    max_order_age_sessions: int = 1
    max_stale_sessions: int = 20
    liquidate_at_end: bool = False
    portfolio_stop_drawdown_pct: float | None = None
    portfolio_daily_loss_pct: float | None = None
    risk_cooldown_sessions: int = 0
    monte_carlo_enabled: bool = True
    monte_carlo_simulations: int = Field(default=2000, ge=100, le=20000)
    monte_carlo_block_size_sessions: int = Field(default=20, ge=1, le=252)
    monte_carlo_seed: int = 20260824
    exit_policy: ExitPolicyReq = Field(default_factory=ExitPolicyReq)


@router.get("/backtest/capabilities")
async def backtest_capabilities():
    return {
        "protocol": "step_event_v2",
        "resolution": "daily_ohlc_conservative_path",
        "signal_timing": "t_close",
        "default_execution": "t_plus_1_raw_open",
        "liquidity_basis": "previous_20_session_adv",
        "exit_rules": [
            "fixed_stop_loss",
            "fixed_take_profit",
            "trailing_stop",
            "atr_stop_loss",
            "atr_take_profit",
            "atr_trailing_stop",
            "break_even_stop",
            "time_stop",
            "portfolio_drawdown_exit",
            "portfolio_daily_loss_exit",
        ],
        "position_sizing": ["equal_weight", "inverse_volatility", "atr_risk"],
        "multi_factor": {
            "max_factors": 12,
            "combination_methods": ["independent_capital_sleeves"],
            "weight_semantics": "positive_starting_capital_allocation_normalized_to_one",
            "per_factor_direction": [-1, 1],
            "attribution": "exact_daily_nlv_pnl_and_cost_by_factor_sleeve",
            "cross_factor_order_netting": False,
        },
        "diagnostics": {
            "time_slice_stability": {
                "annual": True,
                "rolling_months": [12, 24],
                "metrics": ["total_return", "cagr", "sharpe", "max_drawdown", "turnover"],
                "sleeve_regime_reversal": True,
            },
            "information_coefficient": {
                "label": "t_close_to_t_plus_1_open_through_t_plus_1_plus_h_open",
                "metrics": ["ic_mean", "icir", "rank_ic_mean", "rank_icir"],
                "non_overlapping_rebalance_dates": True,
            },
            "monte_carlo": {
                "method": "circular_moving_block_bootstrap",
                "simulation_range": [100, 20000],
                "joint_sleeve_sampling": True,
            },
            "factor_performance_correlation": {
                "metrics": [
                    "daily_net_return_correlation",
                    "monthly_net_return_correlation",
                    "rolling_12m_return_correlation",
                    "rank_ic_path_correlation",
                ],
                "source": "independent_fee_after_sleeve_ledgers",
            },
        },
        "impact_models": ["fixed", "linear", "square_root"],
        "account_types": ["auto", "cash", "margin"],
        "order_policies": ["cancel", "carry"],
        "leverage_control": {
            "target_semantics": "operating_target_below_hard_limit",
            "per_fill_hard_cap": True,
            "post_gap_auto_deleverage": True,
            "resolved_breach_is_audited_not_failed": True,
            "unresolved_breach_fails_integrity": True,
        },
        "portfolio_risk_control": {
            "state_machine": ["armed", "liquidating", "cooldown", "rearmed"],
            "high_water_mark_reset_after_cooldown": True,
            "flat_book_does_not_retrigger_drawdown": True,
        },
        "market_constraints": {
            "ashare": ["board_lot_buy", "t_plus_1_sell", "open_limit_proxy"],
            "us": ["cash_or_margin", "short_borrow_cost"],
        },
        "rust_kernel": rust_kernel_capabilities(),
    }


@router.post("/backtest")
async def backtest(req: BacktestReq):
    eid, cfg = await _experiment_context(req.experiment_id)
    market = cfg.get("market", "us")
    if req.direction not in {-1, 1}:
        raise HTTPException(400, "direction 必须为 1 或 -1")
    if req.combination_method != "independent_capital_sleeves":
        raise HTTPException(400, "combination_method 仅支持 independent_capital_sleeves")
    if len(req.factors) > 12:
        raise HTTPException(400, "多因子回测最多支持 12 个因子")
    factor_payload = [item.model_dump() for item in req.factors]
    if factor_payload:
        for index, factor in enumerate(factor_payload, start=1):
            if factor["direction"] not in {-1, 1}:
                raise HTTPException(400, f"第 {index} 个因子方向必须为 1 或 -1")
            err = validate(factor["expression"], get_dsl_fields(market))
            if err:
                raise HTTPException(400, f"第 {index} 个因子表达式非法: {err}")
    else:
        if not req.expression.strip():
            raise HTTPException(400, "至少需要一个因子表达式")
        err = validate(req.expression, get_dsl_fields(market))
        if err:
            raise HTTPException(400, f"表达式非法: {err}")
    panel_glob = req.panel_glob or cfg.get("panel_glob")
    resolved_eval = evaluation_config(market, cfg.get("evaluation_config"))
    try:
        mode, borrow_cost = _resolve_manual_backtest_execution(
            req.mode,
            cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE),
            market,
            req.borrow_cost_bps_annual,
            resolved_eval["borrow_cost_bps_annual"],
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    params = {
        **req.model_dump(),
        "expression": req.expression or (factor_payload[0]["expression"] if factor_payload else ""),
        "requested_mode": req.mode,
        "mode": mode,
        "market": market,
        "panel_glob": panel_glob,
        "borrow_cost_bps_annual": borrow_cost,
        "protocol": (
            "step_event_v2_weighted_sleeves_v1"
            if factor_payload else "step_event_v2"
        ),
    }
    async with SessionLocal() as s:
        record = Backtest(
            experiment_id=eid,
            params=params,
            result={},
            status="running",
        )
        s.add(record)
        await s.commit()
        await s.refresh(record)
        run_id = record.id
    artifact_dir = BACKTEST_ARTIFACT_ROOT / f"{run_id:08d}"
    progress_job_id = f"backtest:{run_id}"
    COMPUTE_PROGRESS.start(
        progress_job_id,
        kind="backtest",
        title=f"事件回测 #{run_id}",
        phase="queued",
        message=f"{market} · {req.start} 至 {req.end}",
        experiment_id=eid,
        metadata={
            "backtest_id": run_id,
            "market": market,
            "mode": mode,
            "factor_count": len(factor_payload) or 1,
        },
    )

    def backtest_progress(payload: dict) -> None:
        COMPUTE_PROGRESS.update(
            progress_job_id,
            phase=payload.get("phase"),
            message=payload.get("message"),
            completed=payload.get("completed"),
            total=payload.get("total"),
        )

    try:
        common_backtest_kwargs = dict(
            universe_n=req.universe_n,
            start=req.start,
            end=req.end,
            cost_bps=req.cost_bps,
            mode=mode,
            panel_glob=panel_glob,
            market=market,
            borrow_cost_bps_annual=borrow_cost,
            top_fraction=req.top_fraction,
            initial_capital=req.initial_capital,
            rebalance_every=req.rebalance_every,
            slippage_bps=req.slippage_bps,
            max_volume_participation=req.max_volume_participation,
            fee_profile=req.fee_profile,
            account_type=req.account_type,
            cash_buffer_fraction=req.cash_buffer_fraction,
            max_gross_leverage=req.max_gross_leverage,
            margin_interest_bps_annual=req.margin_interest_bps_annual,
            position_sizing=req.position_sizing,
            max_positions=req.max_positions,
            max_position_weight=req.max_position_weight,
            min_trade_notional=req.min_trade_notional,
            rebalance_buffer_pct=req.rebalance_buffer_pct,
            long_gross_target=req.long_gross_target,
            short_gross_target=req.short_gross_target,
            risk_per_position_fraction=req.risk_per_position_fraction,
            spread_bps=req.spread_bps,
            impact_model=req.impact_model,
            impact_coefficient_bps=req.impact_coefficient_bps,
            unfilled_order_policy=req.unfilled_order_policy,
            max_order_age_sessions=req.max_order_age_sessions,
            max_stale_sessions=req.max_stale_sessions,
            liquidate_at_end=req.liquidate_at_end,
            portfolio_stop_drawdown_pct=req.portfolio_stop_drawdown_pct,
            portfolio_daily_loss_pct=req.portfolio_daily_loss_pct,
            risk_cooldown_sessions=req.risk_cooldown_sessions,
            monte_carlo_enabled=req.monte_carlo_enabled,
            monte_carlo_simulations=req.monte_carlo_simulations,
            monte_carlo_block_size_sessions=req.monte_carlo_block_size_sessions,
            monte_carlo_seed=req.monte_carlo_seed,
            exit_policy=req.exit_policy.model_dump(),
            execution_backend=req.execution_backend,
            artifact_dir=artifact_dir,
            progress_callback=backtest_progress,
        )
        if factor_payload:
            result = await asyncio.to_thread(
                run_multi_factor_backtest,
                factor_payload,
                **common_backtest_kwargs,
            )
        else:
            result = await asyncio.to_thread(
                run_backtest,
                req.expression,
                direction=req.direction,
                **common_backtest_kwargs,
            )
    except Exception as exc:  # noqa: BLE001 - persist failed runs as audit evidence
        COMPUTE_PROGRESS.finish(
            progress_job_id,
            state="failed",
            message="事件回测失败",
            error=str(exc),
        )
        async with SessionLocal() as s:
            failed = await s.get(Backtest, run_id)
            failed.status = "failed"
            failed.error = str(exc)[:4000]
            await s.commit()
        if isinstance(exc, ValueError):
            raise HTTPException(400, str(exc)) from exc
        raise HTTPException(500, f"事件回测失败: {exc}") from exc
    persisted = {
        key: result[key]
        for key in (
            "protocol",
            "config",
            "fee_schedule",
            "stats",
            "curve",
            "daily_steps",
            "integrity",
            "positions",
            "round_trips",
            "attribution_method",
            "attribution_disclosure",
            "factor_attribution",
            "factor_attribution_curve",
            "stability_analysis",
            "signal_diagnostics",
            "monte_carlo",
            "factor_performance_correlation",
            "execution",
            "artifacts",
        )
        if key in result
    }
    async with SessionLocal() as s:
        completed = await s.get(Backtest, run_id)
        completed.result = persisted
        integrity_passed = bool(result.get("integrity", {}).get("all_pass"))
        completed.status = "done" if integrity_passed else "failed"
        completed.error = (
            ""
            if integrity_passed
            else "交割单完整性检查失败；结果已保留但禁止作为有效回测使用"
        )
        await s.commit()
    if not integrity_passed:
        COMPUTE_PROGRESS.finish(
            progress_job_id,
            state="failed",
            message="交割单完整性检查失败",
            error=completed.error,
        )
        raise HTTPException(
            500,
            "交割单完整性检查失败；失败账本已保留，请检查历史回测详情",
        )
    COMPUTE_PROGRESS.finish(
        progress_job_id,
        message="事件回测、交割单与完整性检查完成",
    )
    return {"id": run_id, **result}


@router.get("/backtests")
async def list_backtests(experiment_id: int | None = None):
    eid = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        rows = (await s.scalars(
            select(Backtest)
            .where(Backtest.experiment_id == eid)
            .order_by(Backtest.id.desc())
            .limit(50)
        )).all()
    return {"backtests": [
        {
            "id": b.id,
            "params": b.params,
            "stats": (
                (b.result or {}).get("stats")
                if isinstance(b.result, dict) and "stats" in b.result
                else b.result
            ),
            "integrity": (b.result or {}).get("integrity", {})
            if isinstance(b.result, dict) else {},
            "protocol": (b.result or {}).get("protocol", "legacy_vector")
            if isinstance(b.result, dict) else "legacy_vector",
            "status": b.status,
            "error": b.error,
            "created_at": str(b.created_at),
        }
        for b in rows
    ]}


def _artifact_path(backtest_id: int, filename: str) -> Path:
    root = BACKTEST_ARTIFACT_ROOT.resolve()
    path = (root / f"{backtest_id:08d}" / filename).resolve()
    if root not in path.parents:
        raise HTTPException(400, "非法回测产物路径")
    return path


def _read_artifact_page(
    backtest_id: int,
    filename: str,
    offset: int,
    limit: int,
) -> dict:
    path = _artifact_path(backtest_id, filename)
    if not path.exists():
        raise HTTPException(404, "该历史回测没有事件账本产物")
    frame = pl.read_parquet(path)
    if frame.columns == ["empty"]:
        return {"rows": [], "offset": offset, "limit": limit, "total": 0}
    total = frame.height
    rows = frame.slice(offset, limit).to_dicts()
    return {
        "rows": rows,
        "offset": offset,
        "limit": limit,
        "total": total,
        "returned": len(rows),
    }


@router.get("/backtests/{backtest_id}")
async def backtest_detail(backtest_id: int):
    async with SessionLocal() as s:
        record = await s.get(Backtest, backtest_id)
    if not record:
        raise HTTPException(404, "回测不存在")
    trades = (
        await asyncio.to_thread(
            _read_artifact_page,
            backtest_id,
            "settlement_statement.parquet",
            0,
            200,
        )
        if _artifact_path(
            backtest_id, "settlement_statement.parquet"
        ).exists()
        else {"rows": [], "offset": 0, "limit": 200, "total": 0}
    )
    events = (
        await asyncio.to_thread(
            _read_artifact_page,
            backtest_id,
            "event_ledger.parquet",
            0,
            200,
        )
        if _artifact_path(
            backtest_id, "event_ledger.parquet"
        ).exists()
        else {"rows": [], "offset": 0, "limit": 200, "total": 0}
    )
    return {
        "id": record.id,
        "status": record.status,
        "error": record.error,
        "params": record.params,
        "result": record.result,
        "trades": trades,
        "events": events,
        "created_at": str(record.created_at),
    }


@router.get("/backtests/{backtest_id}/trades")
async def backtest_trades(backtest_id: int, offset: int = 0, limit: int = 200):
    if offset < 0 or not 1 <= limit <= 1000:
        raise HTTPException(400, "offset 必须非负，limit 必须在 1..1000")
    return await asyncio.to_thread(
        _read_artifact_page,
        backtest_id,
        "settlement_statement.parquet",
        offset,
        limit,
    )


@router.get("/backtests/{backtest_id}/events")
async def backtest_events(backtest_id: int, offset: int = 0, limit: int = 200):
    if offset < 0 or not 1 <= limit <= 1000:
        raise HTTPException(400, "offset 必须非负，limit 必须在 1..1000")
    return await asyncio.to_thread(
        _read_artifact_page,
        backtest_id,
        "event_ledger.parquet",
        offset,
        limit,
    )


@router.get("/backtests/{backtest_id}/round-trips")
async def backtest_round_trips(backtest_id: int, offset: int = 0, limit: int = 200):
    if offset < 0 or not 1 <= limit <= 1000:
        raise HTTPException(400, "offset 必须非负，limit 必须在 1..1000")
    return await asyncio.to_thread(
        _read_artifact_page,
        backtest_id,
        "round_trip_ledger.parquet",
        offset,
        limit,
    )


@router.get("/backtests/{backtest_id}/statement.csv")
async def download_backtest_statement(backtest_id: int):
    path = _artifact_path(backtest_id, "settlement_statement.csv")
    if not path.exists():
        raise HTTPException(404, "该历史回测没有可下载交割单")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"backtest-{backtest_id:08d}-settlement-statement.csv",
    )


@router.get("/backtests/{backtest_id}/round-trips.csv")
async def download_backtest_round_trips(backtest_id: int):
    path = _artifact_path(backtest_id, "round_trip_statement.csv")
    if not path.exists():
        raise HTTPException(404, "该历史回测没有完整交易归因产物")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"backtest-{backtest_id:08d}-round-trips.csv",
    )


@router.get("/backtests/{backtest_id}/factor-attribution.csv")
async def download_backtest_factor_attribution(backtest_id: int):
    path = _artifact_path(backtest_id, "factor_attribution.csv")
    if not path.exists():
        raise HTTPException(404, "该历史回测没有多因子贡献归因产物")
    return FileResponse(
        path,
        media_type="text/csv",
        filename=f"backtest-{backtest_id:08d}-factor-attribution.csv",
    )


# ---------- 榜单版本目录 ----------

@router.get("/leaderboards")
async def leaderboard_catalog():
    return await asyncio.to_thread(build_leaderboard_catalog)


@router.get("/leaderboards/{report_id}/factors/{expression_hash}")
async def leaderboard_factor_detail(report_id: str, expression_hash: str):
    try:
        return await asyncio.to_thread(
            load_leaderboard_factor_detail,
            report_id,
            expression_hash,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/leaderboards/{report_id}/files/{file_path:path}")
async def leaderboard_file(report_id: str, file_path: str):
    try:
        path = await asyncio.to_thread(
            resolve_leaderboard_file,
            report_id,
            file_path,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc
    return FileResponse(path)


# ---------- 重要研究文档 ----------


@router.get("/research-documents")
async def list_research_documents():
    try:
        return await asyncio.to_thread(research_document_catalog)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"研究文档目录读取失败: {exc}") from exc


@router.get("/research-documents/{slug}")
async def research_document_detail(slug: str):
    try:
        return await asyncio.to_thread(research_document_metadata, slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, "研究文档不存在") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"研究文档读取失败: {exc}") from exc


@router.get("/research-documents/{slug}/html")
async def research_document_html(slug: str):
    try:
        path = await asyncio.to_thread(resolve_research_document, slug)
    except FileNotFoundError as exc:
        raise HTTPException(404, "研究文档不存在") from exc
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"研究文档读取失败: {exc}") from exc
    return FileResponse(path, media_type="text/html; charset=utf-8")


# ---------- 内部研究智能（非产品化） ----------


class DocumentToDslRequest(BaseModel):
    text: str = Field(min_length=20, max_length=200_000)
    market: str
    max_candidates: int = Field(default=3, ge=1, le=3)


class ResidualBeamRequest(BaseModel):
    target: list[float]
    incumbent_predictions: list[list[float]]
    candidates: dict[str, list[float]]
    folds: int = Field(default=5, ge=2, le=20)
    beam_width: int = Field(default=5, ge=1, le=50)
    turnover: dict[str, float] = Field(default_factory=dict)
    complexity: dict[str, float] = Field(default_factory=dict)


class OverfitDiagnosticsRequest(BaseModel):
    period_return_matrix: list[list[float]]
    observed_sharpe: float
    observations: int = Field(ge=2)
    skewness: float = 0.0
    kurtosis: float = Field(default=3.0, ge=1.0)


class BlindReviewPacketRequest(BaseModel):
    expression: str = Field(min_length=1, max_length=2_000)
    hypothesis: str = Field(default="", max_length=2_000)
    market: str


class BlindReviewSealRequest(BaseModel):
    packets: dict
    method_review: dict
    code_review: dict


class QlibJointRunRequest(BaseModel):
    experiment_id: int = Field(ge=1)
    task_name: str = Field(min_length=1, max_length=64)
    max_rows: int | None = Field(default=None, ge=5_000, le=5_000_000)


@router.get("/research-intelligence/mechanisms")
async def research_mechanisms(market: str | None = None):
    try:
        return mechanism_catalog(market)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/research-intelligence/document-to-dsl")
async def research_document_to_dsl(req: DocumentToDslRequest):
    try:
        result = document_to_dsl(
            req.text,
            market=req.market,
            max_candidates=req.max_candidates,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    # Similarity is evidence, never a score adjustment.
    async with SessionLocal() as session:
        existing = list(
            (
                await session.scalars(
                    select(Node.expression)
                    .where(Node.status == "ok")
                    .order_by(Node.id.desc())
                    .limit(2_000)
                )
            ).all()
        )
    for candidate in result["candidates"]:
        nearest = max(
            (
                expression_similarity(candidate["expression"], expression)
                for expression in existing
            ),
            default=0.0,
        )
        candidate["nearest_internal_expression_similarity"] = round(nearest, 6)
        candidate["duplicate_review_required"] = nearest >= 0.90
    return result


@router.post("/research-intelligence/residual-beam")
async def research_residual_beam(req: ResidualBeamRequest):
    job_id = f"residual-beam:{uuid.uuid4().hex[:12]}"
    COMPUTE_PROGRESS.start(
        job_id,
        kind="residual_beam",
        title="Residual OOF Beam 搜索",
        phase="walk_forward_oof",
        message=f"{len(req.candidates)} 个候选 · beam {req.beam_width}",
        completed=0,
        total=len(req.candidates),
    )
    try:
        result = await asyncio.to_thread(
            residual_oof_beam_search,
            target=req.target,
            incumbent_predictions=req.incumbent_predictions,
            candidates=req.candidates,
            folds=req.folds,
            beam_width=req.beam_width,
            turnover=req.turnover,
            complexity=req.complexity,
        )
    except (ValueError, ArithmeticError) as exc:
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="残差搜索失败", error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="残差搜索失败", error=str(exc))
        raise
    COMPUTE_PROGRESS.finish(job_id, message="残差搜索完成")
    return result


@router.post("/research-intelligence/overfit-diagnostics")
async def research_overfit_diagnostics(req: OverfitDiagnosticsRequest):
    matrix = req.period_return_matrix
    if not matrix or any(len(row) != len(matrix[0]) for row in matrix):
        raise HTTPException(400, "period_return_matrix 必须为非空矩形")
    job_id = f"overfit-diagnostics:{uuid.uuid4().hex[:12]}"
    COMPUTE_PROGRESS.start(
        job_id,
        kind="diagnostics",
        title="过拟合诊断 · DSR/PBO",
        phase="effective_trials",
        message=f"{len(matrix[0])} 次试验 · {len(matrix)} 个时期",
    )
    try:
        effective = await asyncio.to_thread(effective_trial_count, matrix)
    except Exception as exc:
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="过拟合诊断失败", error=str(exc))
        raise
    values = [value for row in matrix for value in row]
    score_std = st.pstdev(values) if len(values) > 1 else 0.0
    result = {
        "scope": "caller_supplied_non_frozen_period_returns",
        "actual_trials": len(matrix[0]),
        "effective_trials": effective,
        "dsr": deflated_sharpe_ratio(
            req.observed_sharpe,
            observations=req.observations,
            effective_trials=effective,
            skewness=req.skewness,
            kurtosis=req.kurtosis,
            sharpe_std=score_std,
        ),
        "pbo": cscv_pbo(matrix),
        "harvey_liu": harvey_liu_haircut(
            req.observed_sharpe,
            observations=req.observations,
            trials=effective,
        ),
        "winner_curse": winner_curse(
            req.observed_sharpe,
            effective_trials=effective,
            score_std=score_std,
        ),
    }
    COMPUTE_PROGRESS.finish(job_id, message="DSR/PBO 诊断完成")
    return result


@router.get("/research-intelligence/trial-ledger")
async def research_trial_ledger(experiment_id: int):
    async with SessionLocal() as session:
        experiment = await session.get(Experiment, experiment_id)
        if experiment is None:
            raise HTTPException(404, "研究任务不存在")
        trials = list(
            (
                await session.scalars(
                    select(Trial)
                    .where(Trial.experiment_id == experiment_id)
                    .order_by(Trial.id)
                )
            ).all()
        )
        nodes = list(
            (
                await session.scalars(
                    select(Node)
                    .where(Node.experiment_id == experiment_id, Node.status == "ok")
                    .order_by(Node.id)
                )
            ).all()
        )
    method_counts: dict[str, int] = {}
    mechanism_counts: dict[str, int] = {}
    evaluated_trials = []
    for trial in trials:
        method = trial.search_method or "legacy_unrecorded"
        mechanism = trial.mechanism or "legacy_unrecorded"
        method_counts[method] = method_counts.get(method, 0) + 1
        mechanism_counts[mechanism] = mechanism_counts.get(mechanism, 0) + 1
        statistic = dict(trial.statistic or {})
        if statistic.get("evaluation_performed", True) is not False:
            evaluated_trials.append(trial)
    research_config = dict(experiment.research_config or {})
    predeclared_trials = int(
        (research_config.get("evaluation_config") or {}).get(
            "multiple_testing_trials", 1000
        )
        or 1000
    )
    by_task: dict[str, dict] = {}
    for task_name in sorted({node.task_name for node in nodes}):
        task_nodes = [node for node in nodes if node.task_name == task_name]
        # Correlation/PBO/source clustering are quadratic in candidate count.
        # Preserve full trial counts but use a deterministic quality+recency
        # diagnostic sample so the API remains bounded for 7x24 campaigns.
        diagnostic_limit = 300
        quality_nodes = sorted(
            task_nodes,
            key=lambda node: (float(node.public_score or 0.0), node.id),
            reverse=True,
        )[: diagnostic_limit // 2]
        recent_nodes = sorted(task_nodes, key=lambda node: node.id, reverse=True)[
            : diagnostic_limit // 2
        ]
        diagnostic_nodes = []
        diagnostic_ids = set()
        for node in [*quality_nodes, *recent_nodes]:
            if node.id not in diagnostic_ids:
                diagnostic_ids.add(node.id)
                diagnostic_nodes.append(node)
        vectors = []
        valid_nodes = []
        for node in diagnostic_nodes:
            signature = (node.public_metrics or {}).get("training_return_path_signature") or {}
            vector = list(signature.get("vector") or [])
            if vector:
                vectors.append(vector)
                valid_nodes.append(node)
        same_length = len({len(row) for row in vectors}) == 1 if vectors else False
        matrix = list(map(list, zip(*vectors))) if same_length else []
        effective_sample = effective_trial_count(matrix) if matrix else float(len(diagnostic_nodes) or 1)
        effective = min(
            float(len(task_nodes) or 1),
            effective_sample * len(task_nodes) / max(1, len(diagnostic_nodes)),
        )
        pbo = cscv_pbo(matrix) if matrix else {"available": False, "reason": "no_comparable_training_signatures"}
        best = max(valid_nodes or task_nodes, key=lambda node: float(node.public_score or 0.0), default=None)
        branch = dict((best.public_metrics or {}).get("active") or (best.public_metrics or {}).get("net") or {}) if best else {}
        sharpe = float(branch.get("sharpe") or 0.0)
        observations = int((best.public_metrics or {}).get("n_days") or 2) if best else 2
        task_trials = [trial for trial in trials if trial.task_name == task_name]
        task_evaluated = [
            trial for trial in task_trials
            if (trial.statistic or {}).get("evaluation_performed", True) is not False
        ]
        direction_invariant_hashes = set()
        for trial in task_evaluated:
            try:
                direction_invariant_hashes.add(
                    normalize_hash(trial.expression, direction_invariant=True)
                )
            except (SyntaxError, TypeError, ValueError):
                direction_invariant_hashes.add(
                    f"invalid:{trial.expression_hash or trial.id}"
                )
        source_snapshot = cluster_training_return_sources(
            [
                {
                    "id": node.id,
                    "task_name": node.task_name,
                    "market": research_config.get("market"),
                    "portfolio_mode": research_config.get("portfolio_mode"),
                    "public_score": node.public_score,
                    "public_metrics": node.public_metrics or {},
                    "mechanism_family": (
                        (node.proposal_meta or {}).get("target_family") or "other"
                    ),
                }
                for node in diagnostic_nodes
            ],
            correlation_threshold=float(
                (research_config.get("return_source_governance") or {}).get(
                    "correlation_threshold", 0.85
                )
            ),
        )
        signal_rows = []
        for node in sorted(
            diagnostic_nodes,
            key=lambda value: float(value.public_score or 0.0),
            reverse=True,
        ):
            signature = (node.public_metrics or {}).get(
                "training_signal_rank_signature"
            ) or {}
            vector = list(signature.get("vector") or [])
            if signature.get("available") and len(vector) >= 8:
                signal_rows.append((node.id, vector))
        signal_clusters = []
        if signal_rows and len({len(vector) for _, vector in signal_rows}) == 1:
            import numpy as np

            for node_id, vector in signal_rows:
                values = np.asarray(vector, dtype=float)
                assigned = False
                for cluster in signal_clusters:
                    corr = float(np.corrcoef(values, cluster["vector"])[0, 1])
                    if math.isfinite(corr) and abs(corr) >= 0.95:
                        cluster["members"].append(node_id)
                        assigned = True
                        break
                if not assigned:
                    signal_clusters.append({
                        "representative_node_id": node_id,
                        "members": [node_id],
                        "vector": values,
                    })
        algorithm_efficiency = {}
        for trial in task_evaluated:
            method = trial.search_method or "legacy_unrecorded"
            row = algorithm_efficiency.setdefault(method, {
                "evaluations": 0,
                "selected": 0,
                "runtime_seconds": 0.0,
            })
            row["evaluations"] += 1
            row["selected"] += int(bool(trial.selected))
            runtime = ((trial.statistic or {}).get("evaluation_runtime") or {})
            row["runtime_seconds"] += float(runtime.get("total_ms") or 0.0) / 1000.0
        for row in algorithm_efficiency.values():
            row["selected_per_100"] = round(
                100.0 * row["selected"] / max(1, row["evaluations"]), 6
            )
            row["selected_per_cpu_hour"] = round(
                3600.0 * row["selected"] / max(1.0, row["runtime_seconds"]), 6
            )
            row["runtime_seconds"] = round(row["runtime_seconds"], 3)
        dynamic_trials = max(
            predeclared_trials,
            len(task_evaluated),
            int(math.ceil(effective)),
        )
        by_task[task_name] = {
            "actual_trials": len(task_trials),
            "evaluated_trials": len(task_evaluated),
            "pre_evaluation_rejections": len(task_trials) - len(task_evaluated),
            "unique_direction_invariant_ast": len(direction_invariant_hashes),
            "ast_redundancy_rate": round(
                1.0 - len(direction_invariant_hashes) / max(1, len(task_evaluated)),
                6,
            ),
            "comparable_return_paths": len(vectors) if same_length else 0,
            "return_path_scope": "compressed_public_plus_meta_train_not_frozen_rating",
            "effective_trials": effective,
            "diagnostic_sampling": {
                "method": "top_quality_plus_recent_deterministic",
                "total_nodes": len(task_nodes),
                "sampled_nodes": len(diagnostic_nodes),
                "sample_effective_trials": effective_sample,
                "effective_trial_extrapolation": "sample_effective * total/sample, capped_at_total",
            },
            "multiple_testing_trials": {
                "predeclared": predeclared_trials,
                "actual_evaluated": len(task_evaluated),
                "effective_correlated": effective,
                "dynamic_gate_trials": dynamic_trials,
                "policy": "max(predeclared, actual_evaluated, ceil(effective))",
            },
            "return_source_governance": source_snapshot,
            "signal_rank_deduplication": {
                "protocol": "factorfactory.signal-rank-sketch/v1",
                "available_signatures": len(signal_rows),
                "clusters": len(signal_clusters),
                "absolute_correlation_threshold": 0.95,
                "redundancy_rate": round(
                    1.0 - len(signal_clusters) / max(1, len(signal_rows)), 6
                ),
                "representatives": [
                    {
                        "node_id": row["representative_node_id"],
                        "members": row["members"],
                    }
                    for row in signal_clusters
                ],
            },
            "algorithm_efficiency": algorithm_efficiency,
            "best_node_id": best.id if best else None,
            "best_training_sharpe": sharpe,
            "dsr": deflated_sharpe_ratio(
                sharpe,
                observations=observations,
                effective_trials=effective,
            ),
            "pbo": pbo,
        }
    return {
        "schema": "factorfactory.actual-trial-ledger/v1",
        "experiment_id": experiment_id,
        "append_only_trials": len(trials),
        "evaluated_trials": len(evaluated_trials),
        "predeclared_trials": predeclared_trials,
        "selected_trials": sum(bool(trial.selected) for trial in trials),
        "search_method_counts": method_counts,
        "mechanism_family_counts": mechanism_counts,
        "tasks": by_task,
        "disclosure": "历史试验在新字段上线前只保留原statistic；不会伪造回填表达式和搜索方法。",
    }


@router.post("/research-intelligence/blind-review/packets")
async def blind_review_packets(req: BlindReviewPacketRequest):
    if req.market not in {"us", "ashare"}:
        raise HTTPException(400, "market 必须是 us 或 ashare")
    packets = build_review_packets(
        expression=req.expression,
        hypothesis=req.hypothesis,
        market=req.market,
    )
    packets["deterministic_code_review"] = deterministic_code_review(
        req.expression, req.market
    )
    packets["llm_status"] = "optional_pending_independent_reviewers"
    return packets


@router.post("/research-intelligence/blind-review/seal")
async def blind_review_seal(req: BlindReviewSealRequest):
    try:
        return seal_reviews(
            packets=req.packets,
            method_review=req.method_review,
            code_review=req.code_review,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


# ---------- 设置 ----------

@router.get("/settings")
async def get_settings():
    async with SessionLocal() as s:
        llm = await s.get(Setting, "llm_providers")
        eng = await s.get(Setting, "engine_config")
    llm_val = llm.value if llm else {"providers": [], "inner_provider": "", "outer_provider": ""}
    # 掩码 api_key
    masked = {**llm_val, "providers": [
        {**p, "api_key": (p.get("api_key", "")[:6] + "..." if p.get("api_key") else "")}
        for p in llm_val.get("providers", [])
    ]}
    engine_value = {**DEFAULT_ENGINE_CONFIG_V2, **(eng.value if eng else {})}
    engine_value["tasks"] = resolve_engine_tasks(
        engine_value.get("tasks", []),
        "us",
        "long_short",
        1,
    )
    return {
        "llm_providers": masked,
        "engine_config": engine_value,
        "evaluation_protocol": {
            "version": EVALUATION_PROTOCOL_VERSION,
            "rating_version": FROZEN_RATING_PROTOCOL_VERSION,
            "rating_window": {
                "start": FROZEN_RATING_WINDOW_START,
                "end": FROZEN_RATING_WINDOW_END,
            },
            "defaults": DEFAULT_EVALUATION_CONFIG,
            "policy_label": "NON_PIT_RESEARCH",
        },
    }


class SettingsReq(BaseModel):
    llm_providers: dict | None = None
    engine_config: dict | None = None


@router.post("/settings")
async def save_settings(req: SettingsReq):
    async with SessionLocal() as s:
        if req.llm_providers is not None:
            row = await s.get(Setting, "llm_providers")
            old = row.value if row else {"providers": []}
            # api_key 为掩码/空则保留旧值
            old_keys = {p.get("name"): p.get("api_key", "") for p in old.get("providers", [])}
            for p in req.llm_providers.get("providers", []):
                k = p.get("api_key", "")
                if not k or k.endswith("..."):
                    p["api_key"] = old_keys.get(p.get("name"), "")
            if row:
                row.value = req.llm_providers
            else:
                s.add(Setting(key="llm_providers", value=req.llm_providers))
        if req.engine_config is not None:
            row = await s.get(Setting, "engine_config")
            if row:
                row.value = req.engine_config
            else:
                s.add(Setting(key="engine_config", value=req.engine_config))
        await s.commit()
    _invalidate_observability_components()
    return {"ok": True}


# ---------- 研究任务 (实验) ----------

class ExperimentReq(BaseModel):
    name: str
    description: str = ""
    research_config: dict = Field(default_factory=dict)


class ExperimentPatchReq(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None  # open/archived
    research_config: dict | None = None


@router.get("/research-architectures")
async def research_architectures():
    return {
        "schema": RESEARCH_ARCHITECTURE_SCHEMA,
        "templates": architecture_catalog(),
        "customization": {
            "search_algorithms": list(SUPPORTED_SEARCH_ALGORITHMS),
            "recommended_search_algorithms": list(DEFAULT_SEARCH_ALGORITHMS),
            "search_policy": {
                "schema": SEARCH_POLICY_SCHEMA,
                "scheduler": "quota_deficit_plus_ucb1",
                "group_weights": SEARCH_GROUP_WEIGHTS,
                "algorithm_groups": ALGORITHM_GROUPS,
            },
            "memory_modes": ["adaptive", "cold"],
            "rules": [
                "layer3_requires_layer2",
                "layer1_requires_at_least_one_algorithm",
                "proposal_mode_is_server_derived",
            ],
        },
    }


@router.get("/experiments")
async def list_experiments():
    active_id = await get_active_experiment_id()
    async with SessionLocal() as s:
        rows = (await s.scalars(select(Experiment).order_by(Experiment.id))).all()
        if SERVICE_ARCHITECTURE:
            rows = [
                row
                for row in rows
                if str(
                    (row.research_config or {}).get("service_instance") or ""
                )
                == SERVICE_INSTANCE
            ]
        factor_counts = dict((await s.execute(
            select(Factor.experiment_id, func.count(Factor.id)).group_by(Factor.experiment_id)
        )).all())
        node_counts = dict((await s.execute(
            select(Node.experiment_id, func.count(Node.id)).group_by(Node.experiment_id)
        )).all())
        step_counts = dict((await s.execute(
            select(OuterStep.experiment_id, func.count(OuterStep.id)).group_by(OuterStep.experiment_id)
        )).all())
    out = [{
        "id": e.id,
        "name": e.name,
        "description": e.description,
        "status": e.status,
        "research_config": e.research_config or {},
        "active": e.id == active_id,
        "created_at": str(e.created_at),
        "counts": {
            "factors": factor_counts.get(e.id, 0),
            "nodes": node_counts.get(e.id, 0),
            "outer_steps": step_counts.get(e.id, 0),
        },
    } for e in rows]
    return {"experiments": out, "active_id": active_id}


@router.post("/experiments")
async def create_experiment(req: ExperimentReq):
    if not req.name.strip():
        raise HTTPException(400, "名称不能为空")
    market = req.research_config.get("market", "us")
    if market not in {"us", "ashare"}:
        raise HTTPException(400, "market 必须是 us 或 ashare")
    portfolio_mode = req.research_config.get(
        "portfolio_mode", "long_only" if market == "ashare" else "long_short"
    )
    if portfolio_mode not in {"long_only", "long_short"}:
        raise HTTPException(400, "portfolio_mode 必须是 long_only 或 long_short")
    if market == "ashare" and portfolio_mode != "long_only":
        raise HTTPException(400, "A股研究任务只允许纯多头；多空模式仅适用于美股")
    direction = int(req.research_config.get("direction", 1))
    if direction not in {-1, 1}:
        raise HTTPException(400, "direction 必须为 1 或 -1")
    try:
        architecture = resolve_research_architecture(req.research_config)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    proposal_mode = architecture["proposal_mode"]
    memory_mode = architecture["memory_mode"]
    try:
        target_factor_count = int(
            req.research_config.get("target_factor_count") or 0
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, "target_factor_count 必须为非负整数") from exc
    if target_factor_count < 0:
        raise HTTPException(400, "target_factor_count 必须为非负整数")
    candidate_evaluation_budget, target_mechanisms = _campaign_controls(
        req.research_config,
        market,
        default_budget=120,
    )
    try:
        return_source_governance = resolve_return_source_governance(
            req.research_config.get("return_source_governance")
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    direction_policy = DEFAULT_RESEARCH_DIRECTION_POLICY
    try:
        resolved_evaluation = evaluation_config(
            market,
            req.research_config.get("evaluation_config"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    async with SessionLocal() as s:
        dup = await s.scalar(select(Experiment).where(Experiment.name == req.name.strip()))
        if dup:
            raise HTTPException(400, "同名实验已存在")
        e = Experiment(
            name=req.name.strip(), description=req.description, status="open",
            research_config={
                **req.research_config,
                **architecture,
                "market": market,
                "portfolio_mode": portfolio_mode,
                "panel_glob": req.research_config.get("panel_glob") or default_panel_glob(market),
                "engine_mode": "v2",
                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                "evaluation_config": resolved_evaluation,
                "direction": direction,
                "direction_policy": direction_policy,
                "proposal_mode": proposal_mode,
                "memory_mode": memory_mode,
                "target_factor_count": target_factor_count,
                "candidate_evaluation_budget": candidate_evaluation_budget,
                "target_mechanisms": target_mechanisms,
                "return_source_governance": return_source_governance,
            },
        )
        s.add(e)
        await s.commit()
        await s.refresh(e)
    _invalidate_observability_components()
    return {"ok": True, "id": e.id}


@router.patch("/experiments/{eid}")
async def update_experiment(eid: int, req: ExperimentPatchReq):
    async with SessionLocal() as s:
        e = await s.get(Experiment, eid)
        if not e:
            raise HTTPException(404)
        if req.name is not None:
            if not req.name.strip():
                raise HTTPException(400, "名称不能为空")
            dup = await s.scalar(select(Experiment).where(
                Experiment.name == req.name.strip(), Experiment.id != eid))
            if dup:
                raise HTTPException(400, "同名实验已存在")
            e.name = req.name.strip()
        if req.description is not None:
            e.description = req.description
        if req.research_config is not None:
            worker = EngineManager.get().worker(eid)
            if worker and worker.running:
                raise HTTPException(400, "研究任务运行中，修改市场或持仓模式前请先停止该任务")
            previous_config = dict(e.research_config or {})
            merged = {**previous_config, **req.research_config}
            market = merged.get("market", "us")
            mode = merged.get("portfolio_mode", "long_only" if market == "ashare" else "long_short")
            if market not in {"us", "ashare"}:
                raise HTTPException(400, "market 必须是 us 或 ashare")
            if mode not in {"long_only", "long_short"}:
                raise HTTPException(400, "portfolio_mode 必须是 long_only 或 long_short")
            if market == "ashare" and mode != "long_only":
                raise HTTPException(400, "A股研究任务只允许纯多头")
            direction = int(merged.get("direction", 1))
            if direction not in {-1, 1}:
                raise HTTPException(400, "direction 必须为 1 或 -1")
            try:
                architecture = resolve_research_architecture(merged)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            proposal_mode = architecture["proposal_mode"]
            memory_mode = architecture["memory_mode"]
            try:
                target_factor_count = int(merged.get("target_factor_count") or 0)
            except (TypeError, ValueError) as exc:
                raise HTTPException(400, "target_factor_count 必须为非负整数") from exc
            if target_factor_count < 0:
                raise HTTPException(400, "target_factor_count 必须为非负整数")
            candidate_evaluation_budget, target_mechanisms = _campaign_controls(
                merged,
                market,
            )
            try:
                return_source_governance = resolve_return_source_governance(
                    merged.get("return_source_governance")
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            direction_policy = DEFAULT_RESEARCH_DIRECTION_POLICY
            market_changed = (
                "market" in req.research_config
                and req.research_config["market"] != (e.research_config or {}).get("market")
            )
            evaluation_overrides = (
                req.research_config.get("evaluation_config", {})
                if market_changed
                else {
                    **((e.research_config or {}).get("evaluation_config") or {}),
                    **(req.research_config.get("evaluation_config") or {}),
                }
            )
            try:
                merged["evaluation_config"] = evaluation_config(
                    market,
                    evaluation_overrides,
                )
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            if req.research_config.get("market") and "panel_glob" not in req.research_config:
                merged["panel_glob"] = default_panel_glob(market)
            merged["evaluation_protocol"] = EVALUATION_PROTOCOL_VERSION
            merged["engine_mode"] = "v2"
            merged["direction"] = direction
            merged["direction_policy"] = direction_policy
            merged.update(architecture)
            merged["proposal_mode"] = proposal_mode
            merged["memory_mode"] = memory_mode
            merged["target_factor_count"] = target_factor_count
            merged["candidate_evaluation_budget"] = candidate_evaluation_budget
            merged["target_mechanisms"] = target_mechanisms
            merged["return_source_governance"] = return_source_governance
            e.research_config = merged
            material_keys = {
                "market",
                "portfolio_mode",
                "panel_glob",
                "direction",
                "direction_policy",
                "evaluation_protocol",
                "evaluation_config",
                "return_source_governance",
                "architecture_schema",
                "architecture_template",
                "layer1_enabled",
                "layer2_enabled",
                "layer3_enabled",
                "search_algorithms",
                "qlib_integration",
            }
            if any(previous_config.get(key) != merged.get(key) for key in material_keys):
                factors = (
                    await s.scalars(
                        select(Factor).where(Factor.experiment_id == eid)
                    )
                ).all()
                for factor in factors:
                    factor.lifecycle_stage = "configuration_changed_requires_reaudit"
                    factor.eligibility = {}
                    if "invalid" not in (factor.provenance_status or ""):
                        factor.provenance_status = "configuration_changed_requires_revalidation"
        if req.status is not None:
            if req.status not in {"open", "archived"}:
                raise HTTPException(400, "status 必须是 open/archived")
            if req.status == "archived" and eid == await get_active_experiment_id() \
                    and EngineManager.get().worker(eid) and EngineManager.get().worker(eid).running:
                raise HTTPException(400, "引擎运行中, 不能归档活动实验")
            e.status = req.status
        await s.commit()
    _invalidate_observability_components()
    return {"ok": True}


@router.delete("/experiments/{eid}")
async def delete_experiment(eid: int):
    # 历史研究数据是审计记录，平台不再提供物理删除。
    raise HTTPException(400, "研究任务数据必须保留；请使用归档而不是删除")


@router.post("/experiments/{eid}/activate")
async def activate_experiment(eid: int):
    async with SessionLocal() as s:
        e = await s.get(Experiment, eid)
        if not e:
            raise HTTPException(404)
        if e.status == "archived":
            raise HTTPException(400, "实验已归档, 请先重新开放")
        if (
            SERVICE_ARCHITECTURE
            and str(
                (e.research_config or {}).get("service_instance") or ""
            )
            != SERVICE_INSTANCE
        ):
            raise HTTPException(400, "该研究任务不属于当前服务实例")
        setting_key = active_experiment_setting_key()
        row = await s.get(Setting, setting_key)
        if row:
            row.value = {"id": eid}
        else:
            s.add(Setting(key=setting_key, value={"id": eid}))
        await s.commit()
    _invalidate_observability_components()
    return {
        "ok": True,
        "active_id": eid,
        "experiment": {
            "id": e.id,
            "name": e.name,
            "status": e.status,
            "research_config": e.research_config or {},
        },
    }


# ---------- Qlib 原生研究适配 ----------


@router.get("/qlib/capabilities")
async def qlib_capabilities():
    return await asyncio.to_thread(qlib_native_capabilities)


@router.get("/qlib/alpha158/catalog")
async def qlib_alpha158_catalog(market: str | None = None):
    try:
        return alpha158_catalog(market)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.get("/qlib/alpha158/progress")
async def qlib_alpha158_progress():
    return await asyncio.to_thread(alpha158_progress)


@router.get("/qlib/alpha158/results")
async def qlib_alpha158_results(market: str):
    try:
        reports = await asyncio.to_thread(latest_alpha158_reports, market)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {
        "schema": "factorfactory.qlib-alpha158-results/v1",
        "market": market,
        "available": bool(reports),
        "reports": reports,
    }


@router.get("/qlib/alpha158/report")
async def qlib_alpha158_report(market: str, portfolio_mode: str):
    if portfolio_mode not in {"long_only", "long_short"}:
        raise HTTPException(400, "portfolio_mode 必须是 long_only 或 long_short")
    try:
        reports = await asyncio.to_thread(latest_alpha158_reports, market)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    selected = next(
        (row for row in reports if row.get("portfolio_mode") == portfolio_mode),
        None,
    )
    if selected is None:
        raise HTTPException(404, "尚无该市场与模式的 Alpha158 报告")
    summary_path = Path(selected["artifact_path"])
    report_path = summary_path.with_name(
        summary_path.name.replace("-summary.json", "-report.html")
    )
    if not report_path.exists():
        raise HTTPException(404, "报告文件不存在")
    return FileResponse(report_path, media_type="text/html")


@router.get("/qlib/joint/status")
async def qlib_joint_status(experiment_id: int, task_name: str):
    async with SessionLocal() as session:
        experiment = await session.get(Experiment, experiment_id)
        if experiment is None:
            raise HTTPException(404, "研究任务不存在")
    config = dict(experiment.research_config or {})
    tasks = list((config.get("engine_config") or {}).get("tasks") or DEFAULT_ENGINE_CONFIG_V2["tasks"])
    task = next((row for row in tasks if str(row.get("name")) == task_name), None)
    if task is None:
        raise HTTPException(404, "研究子任务不存在")
    key = (
        f"exp-{experiment_id}-{task_name}-{config.get('market', 'us')}-"
        f"u{int(task.get('universe_n') or 500)}-h{int(task.get('horizon') or 5)}"
    )
    return latest_joint_result(key) or {
        "schema": "factorfactory.qlib-joint-residual-distill/v1",
        "state": "not_started",
        "task_key": key,
    }


@router.post("/qlib/joint/run")
async def qlib_joint_run(req: QlibJointRunRequest):
    async with SessionLocal() as session:
        experiment = await session.get(Experiment, req.experiment_id)
        if experiment is None:
            raise HTTPException(404, "研究任务不存在")
        nodes = list((await session.scalars(
            select(Node)
            .where(
                Node.experiment_id == req.experiment_id,
                Node.task_name == req.task_name,
                Node.status == "ok",
            )
            .order_by(Node.public_score.desc(), Node.id.desc())
            .limit(100)
        )).all())
    config = dict(experiment.research_config or {})
    integration = dict(config.get("qlib_integration") or {})
    if not integration.get("joint_model_enabled"):
        raise HTTPException(400, "该研究任务未启用Qlib联合模型")
    tasks = list((config.get("engine_config") or {}).get("tasks") or DEFAULT_ENGINE_CONFIG_V2["tasks"])
    task = next((row for row in tasks if str(row.get("name")) == req.task_name), None)
    if task is None:
        raise HTTPException(404, "研究子任务不存在")
    expressions = []
    seen = set()
    for node in nodes:
        try:
            key = normalize_hash(node.expression, direction_invariant=True)
        except (SyntaxError, TypeError, ValueError):
            continue
        if key not in seen:
            seen.add(key)
            expressions.append(node.expression)
    task_key = (
        f"exp-{req.experiment_id}-{req.task_name}-{config.get('market', 'us')}-"
        f"u{int(task.get('universe_n') or 500)}-h{int(task.get('horizon') or 5)}"
    )
    progress_job_id = f"qlib-joint:{task_key}"
    COMPUTE_PROGRESS.start(
        progress_job_id,
        kind="qlib_joint",
        title=f"Qlib联合模型 · {req.task_name}",
        phase="queued",
        message="等待特征矩阵与严格时序 OOF 计算",
        experiment_id=req.experiment_id,
        metadata={"task_key": task_key, "task_name": req.task_name},
    )

    def joint_progress(payload: dict) -> None:
        COMPUTE_PROGRESS.update(
            progress_job_id,
            phase=payload.get("phase"),
            message=payload.get("message"),
            completed=payload.get("completed"),
            total=payload.get("total"),
        )

    try:
        joint_task = asyncio.create_task(
            asyncio.to_thread(
                run_joint_alpha158,
                JointModelSpec(
                    market=str(config.get("market") or "us"),
                    panel_glob=config.get("panel_glob"),
                    universe_n=int(task.get("universe_n") or 500),
                    horizon=int(task.get("horizon") or 5),
                    max_rows=int(
                        req.max_rows
                        or integration.get("max_training_rows")
                        or 250_000
                    ),
                    min_meta_dates=int(
                        integration.get("min_meta_dates") or 60
                    ),
                    include_low_fidelity_vwap=bool(
                        integration.get("include_low_fidelity_vwap", False)
                    ),
                ),
                task_key=task_key,
                incumbent_expressions=expressions[:5],
                progress_callback=joint_progress,
            ),
            name=f"qlib-joint.manual.{req.experiment_id}.{req.task_name}",
        )
        while not joint_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(joint_task), timeout=5.0
                )
            except asyncio.TimeoutError:
                COMPUTE_PROGRESS.update(
                    progress_job_id,
                    metadata={
                        "worker_heartbeat": datetime.now(
                            timezone.utc
                        ).isoformat()
                    },
                )
        result = await joint_task
    except (ValueError, RuntimeError) as exc:
        COMPUTE_PROGRESS.finish(
            progress_job_id,
            state="failed",
            message="Qlib联合模型失败",
            error=str(exc),
        )
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        COMPUTE_PROGRESS.finish(
            progress_job_id,
            state="failed",
            message="Qlib联合模型失败",
            error=str(exc),
        )
        raise
    COMPUTE_PROGRESS.finish(
        progress_job_id,
        message="联合模型与DSL蒸馏完成",
        metadata={"search_eligible": bool(result.get("search_eligible"))},
    )
    return result


# ---------- 因子组合优化实验台 ----------


class FactorToolComponentReq(BaseModel):
    key: str | None = None
    name: str | None = None
    expression: str
    direction: int = 1
    weight: float = 1.0
    mechanism: str | None = None


class FactorCorrelationReq(BaseModel):
    experiment_id: int | None = None
    market: str | None = None
    portfolio_mode: str | None = None
    panel_glob: str | None = None
    components: list[FactorToolComponentReq]
    start: str = "2020-01-01"
    end: str = "2026-12-31"
    universe_n: int = 500
    horizon: int = 5
    top_fraction: float = 0.20
    cost_bps: float = 15.0
    borrow_cost_bps_annual: float | None = None
    threshold: float = 0.80


class FactorExpressionBuildReq(BaseModel):
    market: str = "ashare"
    components: list[FactorToolComponentReq]
    normalization: str = "rank"
    omit_common_scale: bool = True


@router.get("/factor-tools/capabilities")
async def factor_tools_capabilities(
    market: str | None = None,
    experiment_id: int | None = None,
):
    _, cfg = await _experiment_context(experiment_id)
    resolved_market = str(market or cfg.get("market") or "us")
    try:
        return factor_tool_capabilities(resolved_market)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/factor-tools/correlation")
async def factor_tools_correlation(req: FactorCorrelationReq):
    _, cfg = await _experiment_context(req.experiment_id)
    payload = req.model_dump(exclude_none=True)
    payload["market"] = payload.get("market") or cfg.get("market") or "us"
    payload["portfolio_mode"] = (
        payload.get("portfolio_mode")
        or cfg.get("portfolio_mode")
        or ("long_only" if payload["market"] == "ashare" else "long_short")
    )
    # A manually selected market must never inherit another task's panel.
    # Reuse the active task panel only when the markets match; otherwise the
    # factor tool resolves the market-specific default panel itself.
    payload["panel_glob"] = payload.get("panel_glob") or (
        cfg.get("panel_glob")
        if str(cfg.get("market") or "us") == payload["market"]
        else None
    )
    job_id = f"factor-correlation:{uuid.uuid4().hex[:12]}"
    COMPUTE_PROGRESS.start(
        job_id,
        kind="factor_correlation",
        title="因子相关性检测",
        phase="materialize",
        message=f"{len(payload.get('components') or [])} 个因子 · {payload['market']}",
        experiment_id=req.experiment_id,
    )
    try:
        result = await asyncio.to_thread(run_factor_correlation, payload)
    except ValueError as exc:
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="相关性检测失败", error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="相关性检测失败", error=str(exc))
        raise HTTPException(500, f"相关性检测失败: {exc}") from exc
    COMPUTE_PROGRESS.finish(job_id, message="相关性矩阵与聚类完成")
    return result


@router.post("/factor-tools/build-expression")
async def factor_tools_build_expression(req: FactorExpressionBuildReq):
    try:
        return build_combination_expression(req.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class CombinationComponentReq(BaseModel):
    key: str | None = None
    name: str | None = None
    expression: str
    direction: int = 1
    mechanism: str | None = None
    source: str = "manual"
    source_ref: str = ""


class CombinationExperimentReq(BaseModel):
    name: str = "组合优化实验"
    experiment_id: int | None = None
    search_mode: str = "programmatic"
    market: str | None = None
    portfolio_mode: str | None = None
    panel_glob: str | None = None
    components: list[CombinationComponentReq]
    min_factors: int = 2
    max_factors: int = 5
    min_mechanisms: int = 2
    coarse_step: float = 0.10
    min_weight: float = 0.05
    max_weight: float = 0.65
    max_mechanism_weight: float = 0.70
    max_pair_correlation: float = 0.85
    path_budget: int = 3000
    validation_budget: int = 250
    top_k: int = 10
    universe_n: int = 500
    top_fraction: float = 0.20
    horizon: int = 5
    cost_bps: float = 15.0
    stress_cost_bps: float = 50.0
    borrow_cost_bps_annual: float | None = None
    train_start: str = "2010-01-01"
    train_end: str = "2018-12-31"
    validation_start: str = "2019-01-01"
    validation_end: str = "2022-12-31"
    rating_start: str = "2023-01-01"
    rating_end: str = "2026-12-31"
    llm_max_proposals: int = 8
    initial_capital: float = 1_000_000.0
    max_volume_participation: float = 0.05
    start: bool = True


def _combination_payload(
    row: CombinationExperiment,
    *,
    include_detail: bool,
) -> dict:
    runtime = CombinationLabManager.get().snapshot(row.id)
    progress = runtime or dict(row.progress or {})
    result = dict(row.result or {})
    payload = {
        "id": row.id,
        "experiment_id": row.experiment_id,
        "name": row.name,
        "protocol": row.protocol,
        "search_mode": row.search_mode,
        "market": row.market,
        "portfolio_mode": row.portfolio_mode,
        "status": row.status,
        "snapshot_hash": row.snapshot_hash,
        "component_count": len((row.component_snapshot or {}).get("components") or []),
        "progress": progress,
        "decision": result.get("decision"),
        "result_hash": result.get("result_hash"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "error": row.error,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }
    if include_detail:
        payload.update({
            "request_spec": row.request_spec or {},
            "component_snapshot": row.component_snapshot or {},
            "result": result,
            "llm_trace": row.llm_trace or {},
        })
    return payload


@router.get("/combination-experiments/capabilities")
async def combination_capabilities(experiment_id: int | None = None):
    eid, cfg = await _experiment_context(experiment_id)
    async with SessionLocal() as session:
        settings = await session.get(Setting, "llm_providers")
    providers = dict(settings.value or {}) if settings else {}
    selected = providers.get("inner_provider") or providers.get("outer_provider")
    configured = any(
        row.get("name") == selected
        and row.get("api_key")
        and row.get("base_url")
        and row.get("model")
        for row in providers.get("providers", [])
    )
    market = str(cfg.get("market") or "us")
    portfolio_mode = str(
        cfg.get("portfolio_mode")
        or ("long_only" if market == "ashare" else "long_short")
    )
    return {
        "protocol": COMBINATION_LAB_PROTOCOL,
        "experiment_id": eid,
        "market": market,
        "portfolio_mode": portfolio_mode,
        "panel_glob": cfg.get("panel_glob"),
        "llm": {
            "configured": bool(configured),
            "provider": selected if configured else None,
        },
        "limits": {
            "min_components": 2,
            "max_components": 12,
            "max_path_budget": 20_000,
            "supported_horizons": [1, 5, 20],
            "ashare_modes": ["long_only"],
            "us_modes": ["long_only", "long_short"],
        },
        "defaults": {
            "min_factors": 2,
            "max_factors": 5,
            "min_weight": 0.05,
            "max_weight": 0.65,
            "max_mechanism_weight": 0.70,
            "max_pair_correlation": 0.85,
            "path_budget": 3000,
            "validation_budget": 250,
            "universe_n": 500,
            "top_fraction": 0.20,
            "horizon": 5,
            "cost_bps": 15.0,
            "stress_cost_bps": 50.0,
        },
    }


@router.post("/combination-experiments")
async def create_combination_experiment(req: CombinationExperimentReq):
    eid, cfg = await _experiment_context(req.experiment_id)
    payload = req.model_dump(exclude={"start"}, exclude_none=True)
    payload["experiment_id"] = eid
    payload["market"] = payload.get("market") or cfg.get("market") or "us"
    payload["portfolio_mode"] = (
        payload.get("portfolio_mode")
        or cfg.get("portfolio_mode")
        or ("long_only" if payload["market"] == "ashare" else "long_short")
    )
    payload["panel_glob"] = payload.get("panel_glob") or cfg.get("panel_glob")
    try:
        canonical = validate_lab_spec(payload)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if canonical["search_mode"] == "llm":
        async with SessionLocal() as session:
            settings = await session.get(Setting, "llm_providers")
        value = dict(settings.value or {}) if settings else {}
        provider_name = value.get("inner_provider") or value.get("outer_provider")
        if not any(
            row.get("name") == provider_name and row.get("api_key")
            for row in value.get("providers", [])
        ):
            raise HTTPException(409, "LLM协作模式要求先在设置中配置inner_provider或outer_provider")
    row = CombinationExperiment(
        experiment_id=eid,
        name=canonical["name"],
        protocol=COMBINATION_LAB_PROTOCOL,
        search_mode=canonical["search_mode"],
        market=canonical["market"],
        portfolio_mode=canonical["portfolio_mode"],
        status="queued" if req.start else "draft",
        snapshot_hash=canonical["snapshot_hash"],
        request_spec=canonical,
        component_snapshot={
            "schema": "combination_component_snapshot_v1",
            "snapshot_hash": canonical["snapshot_hash"],
            "components": canonical["components"],
        },
        progress={"stage": "queued" if req.start else "draft", "completed": 0, "total": 1},
    )
    async with SessionLocal() as session:
        session.add(row)
        await session.commit()
        await session.refresh(row)
    if req.start:
        await CombinationLabManager.get().start(row.id)
    return _combination_payload(row, include_detail=True)


@router.get("/combination-experiments")
async def list_combination_experiments(
    experiment_id: int | None = None,
    limit: int = 50,
):
    eid, _ = await _experiment_context(experiment_id)
    limit = max(1, min(200, int(limit)))
    async with SessionLocal() as session:
        rows = list((await session.scalars(
            select(CombinationExperiment)
            .where(CombinationExperiment.experiment_id == eid)
            .order_by(CombinationExperiment.id.desc())
            .limit(limit)
        )).all())
        total = int(await session.scalar(
            select(func.count()).select_from(CombinationExperiment)
            .where(CombinationExperiment.experiment_id == eid)
        ) or 0)
    return {
        "protocol": COMBINATION_LAB_PROTOCOL,
        "experiment_id": eid,
        "total": total,
        "experiments": [
            _combination_payload(row, include_detail=False) for row in rows
        ],
    }


@router.get("/combination-experiments/{combination_id}")
async def get_combination_experiment(combination_id: int):
    async with SessionLocal() as session:
        row = await session.get(CombinationExperiment, combination_id)
    if row is None:
        raise HTTPException(404, "组合实验不存在")
    return _combination_payload(row, include_detail=True)


@router.post("/combination-experiments/{combination_id}/start")
async def start_combination_experiment(combination_id: int):
    async with SessionLocal() as session:
        row = await session.get(CombinationExperiment, combination_id)
        if row is None:
            raise HTTPException(404, "组合实验不存在")
        if row.status in {"queued", "running"}:
            return {"id": row.id, "status": row.status, "already_running": True}
        if row.status == "done":
            raise HTTPException(409, "已完成实验不可改写；请用相同配置创建新版本")
        row.status = "queued"
        row.error = ""
        row.result = {}
        row.progress = {"stage": "queued", "completed": 0, "total": 1}
        row.started_at = None
        row.completed_at = None
        await session.commit()
    return await CombinationLabManager.get().start(combination_id)


@router.post("/combination-experiments/{combination_id}/stop")
async def stop_combination_experiment(combination_id: int):
    async with SessionLocal() as session:
        row = await session.get(CombinationExperiment, combination_id)
    if row is None:
        raise HTTPException(404, "组合实验不存在")
    return await CombinationLabManager.get().stop(combination_id)


# ---------- 选股器 ----------

SCREENER_RUN_SCHEMA_VERSION = "screener_run_v1"


def _screener_run_payload(
    row: ScreenerRun,
    *,
    include_snapshot: bool = False,
) -> dict:
    request_spec = dict(row.request_spec or {})
    result = dict(row.result_snapshot or {})
    stocks = list(result.get("stocks") or [])
    factors = list(request_spec.get("factors") or [])
    payload = {
        "id": row.id,
        "experiment_id": row.experiment_id,
        "schema_version": row.schema_version,
        "market": row.market,
        "portfolio_mode": row.portfolio_mode,
        "target_date": str(row.target_date),
        "requested_date": (
            str(row.requested_date) if row.requested_date else None
        ),
        "date_adjusted": bool(
            row.requested_date and row.requested_date != row.target_date
        ),
        "direction": row.direction,
        "factor_count": row.factor_count,
        "eligible_count": row.eligible_count,
        "result_count": row.result_count,
        "cache_hit": row.cache_hit,
        "elapsed_ms": row.elapsed_ms,
        "status": row.status,
        "error": row.error,
        "expression_mode": bool(request_spec.get("expression_mode")),
        "universe_n": request_spec.get("universe_n"),
        "top_n": request_spec.get("top_n"),
        "factor_preview": [
            {
                "expression": factor.get("expression"),
                "weight": factor.get("weight"),
                "direction": factor.get("direction"),
            }
            for factor in factors[:3]
        ],
        "stock_preview": [
            {
                "ts_code": stock.get("ts_code"),
                "name": stock.get("name"),
                "side": stock.get("side"),
                "side_rank": stock.get("side_rank"),
                "score": stock.get("score"),
            }
            for stock in stocks[:5]
        ],
        "created_at": (
            row.created_at.isoformat()
            if isinstance(row.created_at, datetime)
            else str(row.created_at)
        ),
    }
    if include_snapshot:
        payload["panel_identity"] = row.panel_identity
        payload["request_spec"] = request_spec
        payload["result"] = {
            **result,
            "run_id": row.id,
            "recorded_at": payload["created_at"],
        }
    return payload


class DSLInspectReq(BaseModel):
    expression: str
    experiment_id: int | None = None


@router.post("/dsl/inspect")
async def inspect_dsl(req: DSLInspectReq):
    _, cfg = await _experiment_context(req.experiment_id)
    market = cfg.get("market", "us")
    error = validate(req.expression, get_dsl_fields(market))
    if error:
        raise HTTPException(400, f"表达式非法: {error}")
    return {
        "valid": True,
        "market": market,
        "semantic_audit": audit_expression_semantics(req.expression, market),
        "mechanism_family": infer_mechanism(req.expression),
        **expression_profile(req.expression),
    }


class ScreenerReq(BaseModel):
    experiment_id: int | None = None
    factors: list[dict] = Field(default_factory=list)  # [{"expression": "...", "weight": 1.0}, ...]
    expression: str | None = None  # 单个 DSL 直接选股
    expression_direction: int = 1
    panel_glob: str | None = None
    date: str | None = None  # YYYY-MM-DD, None=最新交易日
    universe_n: int = 500
    top_n: int = 50
    direction: str = "top"  # "top" | "bottom" | "both"


class ScreenerAllocationReq(BaseModel):
    experiment_id: int | None = None
    run_id: int
    symbols: list[str] = Field(default_factory=list)
    method: str = "robust_risk_budget"
    lookback: int = 120
    max_weight: float = 0.35
    score_tilt: float = 0.35


@router.post("/screener")
async def screener(req: ScreenerReq):
    """多因子选股，并追加保存任务隔离、可复查的完整结果快照。"""
    started = time.perf_counter()
    experiment_id, cfg = await _experiment_context(req.experiment_id)
    panel_glob = req.panel_glob or cfg.get("panel_glob")
    market = cfg.get("market", "us")
    portfolio_mode = cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE)
    fields = get_dsl_fields(market)
    if req.direction not in {"top", "bottom", "both"}:
        raise HTTPException(400, "direction 必须是 top/bottom/both")
    if not 1 <= req.universe_n <= 10000:
        raise HTTPException(400, "universe_n 必须在 1 到 10000 之间")
    if not 1 <= req.top_n <= 500:
        raise HTTPException(400, "top_n 必须在 1 到 500 之间")
    factors = list(req.factors)
    if req.expression:
        factors = [{
            "expression": req.expression,
            "weight": 1.0,
            "direction": req.expression_direction,
        }]
    if not factors:
        raise HTTPException(400, "至少需要一个因子")
    if len(factors) > 12:
        raise HTTPException(400, "多因子选股最多支持 12 个因子")
    for factor in factors:
        if not factor.get("expression"):
            raise HTTPException(400, "因子表达式不能为空")
        validation_error = validate(factor["expression"], fields)
        if validation_error:
            raise HTTPException(400, f"表达式非法: {validation_error}")
        if float(factor.get("weight", 1.0)) <= 0:
            raise HTTPException(400, "因子权重必须大于 0")
        factor_direction = int(factor.get("direction", 1))
        if factor_direction not in {-1, 1}:
            raise HTTPException(400, "因子 direction 必须为 1 或 -1")
    factors = [
        {
            **(
                {"name": str(factor.get("name"))[:80]}
                if factor.get("name") else {}
            ),
            "expression": str(factor["expression"]),
            "weight": float(factor.get("weight", 1.0)),
            "direction": int(factor.get("direction", 1)),
        }
        for factor in factors
    ]

    store = PanelStore.get(panel_glob, market)
    snapshot_reader = getattr(store, "read_snapshot", None)
    if callable(snapshot_reader):
        df, trading_dates, loaded_identity, panel_generation = snapshot_reader()
    else:  # Test doubles and legacy extensions keep the previous contract.
        df = store.ensure_loaded()
        trading_dates = tuple(store.trading_dates)
        loaded_identity = None
        panel_generation = 0
    import datetime as _dt
    import bisect
    requested_date = None
    if req.date:
        try:
            requested_date = _dt.date.fromisoformat(req.date)
        except ValueError as exc:
            raise HTTPException(400, "date 必须是 YYYY-MM-DD") from exc
        index = bisect.bisect_right(trading_dates, requested_date) - 1
        if index < 0:
            raise HTTPException(400, "请求日期早于面板首个交易日")
        target_date = trading_dates[index]
    else:
        target_date = trading_dates[-1]
    panel_identity = (
        f"{market}:{panel_glob or 'default'}:g{panel_generation}:"
        f"{loaded_identity or 'legacy'}:{df.height}:{trading_dates[-1]}"
    )
    job_id = f"screener:{uuid.uuid4().hex[:12]}"
    COMPUTE_PROGRESS.start(
        job_id,
        kind="screener",
        title="多因子选股",
        phase="cross_section",
        message=f"{market} · {target_date} · {len(factors)} 个因子",
        completed=0,
        total=len(factors),
        experiment_id=experiment_id,
    )
    try:
        screened = await asyncio.to_thread(
            screen_cross_section,
            df=df,
            trading_dates=list(trading_dates),
            panel_identity=panel_identity,
            target_date=target_date,
            factors=factors,
            fields=fields,
            universe_n=req.universe_n,
            top_n=req.top_n,
            direction=req.direction,
        )
    except ValueError as exc:
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="选股计算失败", error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="选股计算失败", error=str(exc))
        raise

    response = {
        "experiment_id": experiment_id,
        "date": str(target_date),
        "requested_date": str(requested_date) if requested_date else None,
        "date_adjusted": bool(requested_date and target_date != requested_date),
        "universe_n": req.universe_n,
        "top_n": req.top_n,
        "factor_count": len(factors),
        "expression_mode": bool(req.expression),
        "market": market,
        "portfolio_mode": portfolio_mode,
        "direction": req.direction,
        **screened,
    }
    request_spec = {
        "schema_version": SCREENER_RUN_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "market": market,
        "portfolio_mode": portfolio_mode,
        "evaluation_protocol": cfg.get(
            "evaluation_protocol",
            EVALUATION_PROTOCOL_VERSION,
        ),
        "panel_glob": str(panel_glob) if panel_glob else None,
        "panel_identity": panel_identity,
        "requested_date": req.date,
        "target_date": str(target_date),
        "date_adjusted": response["date_adjusted"],
        "universe_n": req.universe_n,
        "top_n": req.top_n,
        "direction": req.direction,
        "expression_mode": bool(req.expression),
        "factors": factors,
    }
    run = ScreenerRun(
        experiment_id=experiment_id,
        schema_version=SCREENER_RUN_SCHEMA_VERSION,
        market=market,
        portfolio_mode=portfolio_mode,
        target_date=target_date,
        requested_date=requested_date,
        direction=req.direction,
        panel_identity=panel_identity,
        request_spec=request_spec,
        result_snapshot=response,
        factor_count=len(factors),
        eligible_count=int(screened.get("eligible_count") or 0),
        result_count=len(screened.get("stocks") or []),
        cache_hit=bool((screened.get("performance") or {}).get("cache_hit")),
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        status="done",
    )
    async with SessionLocal() as session:
        session.add(run)
        await session.commit()
        await session.refresh(run)
    COMPUTE_PROGRESS.finish(
        job_id,
        message=f"选股完成：{len(screened.get('stocks') or [])} 条结果",
    )
    return {
        **response,
        "run_id": run.id,
        "recorded_at": (
            run.created_at.isoformat()
            if isinstance(run.created_at, datetime)
            else str(run.created_at)
        ),
    }


@router.post("/screener/allocate")
async def allocate_screener_selection(req: ScreenerAllocationReq):
    """Size an arbitrary subset from one immutable screener snapshot."""

    experiment_id, cfg = await _experiment_context(req.experiment_id)
    if req.method not in ALLOCATION_METHODS:
        raise HTTPException(400, f"未知配权方法: {req.method}")
    symbols = [str(symbol).strip() for symbol in req.symbols]
    if not 1 <= len(symbols) <= 100:
        raise HTTPException(400, "必须选择 1 到 100 只证券")
    if any(not symbol for symbol in symbols):
        raise HTTPException(400, "选中证券代码不能为空")
    if len(set(symbols)) != len(symbols):
        raise HTTPException(400, "选中证券不能重复")

    async with SessionLocal() as session:
        row = await session.scalar(
            select(ScreenerRun).where(
                ScreenerRun.id == req.run_id,
                ScreenerRun.experiment_id == experiment_id,
            )
        )
    if not row:
        raise HTTPException(404, "该任务下不存在这条选股记录")

    snapshot = dict(row.result_snapshot or {})
    snapshot_stocks = list(snapshot.get("stocks") or [])
    stock_by_symbol = {
        str(stock.get("ts_code")): stock
        for stock in snapshot_stocks
        if stock.get("ts_code")
    }
    unknown = [symbol for symbol in symbols if symbol not in stock_by_symbol]
    if unknown:
        raise HTTPException(
            400,
            f"所选证券不属于记录 #{row.id} 的候选清单: {', '.join(unknown)}",
        )
    selected = [stock_by_symbol[symbol] for symbol in symbols]

    request_spec = dict(row.request_spec or {})
    panel_glob = request_spec.get("panel_glob") or cfg.get("panel_glob")
    store = PanelStore.get(panel_glob, row.market)
    snapshot_reader = getattr(store, "read_snapshot", None)
    if callable(snapshot_reader):
        df, trading_dates, loaded_identity, panel_generation = snapshot_reader()
    else:  # Test doubles and legacy extensions keep the previous contract.
        df = store.ensure_loaded()
        trading_dates = tuple(store.trading_dates)
        loaded_identity = None
        panel_generation = 0
    current_panel_identity = (
        f"{row.market}:{panel_glob or 'default'}:g{panel_generation}:"
        f"{loaded_identity or 'legacy'}:{df.height}:{trading_dates[-1]}"
    )
    job_id = f"allocation:{uuid.uuid4().hex[:12]}"
    COMPUTE_PROGRESS.start(
        job_id,
        kind="allocation",
        title=f"选股配权 · 记录#{row.id}",
        phase="covariance",
        message=f"{len(selected)} 只证券 · {req.method}",
        experiment_id=experiment_id,
    )
    try:
        allocation = await asyncio.to_thread(
            build_purchase_allocation,
            df=df,
            target_date=row.target_date,
            selected=selected,
            lookback=req.lookback,
            method=req.method,
            max_weight=req.max_weight,
            score_tilt=req.score_tilt,
        )
    except ValueError as exc:
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="配权计算失败", error=str(exc))
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        COMPUTE_PROGRESS.finish(job_id, state="failed", message="配权计算失败", error=str(exc))
        raise

    panel_changed = bool(
        row.panel_identity
        and current_panel_identity != row.panel_identity
    )
    if panel_changed:
        allocation["warnings"].append(
            "当前面板标识与选股快照生成时不同；候选清单保持冻结，"
            "风险统计按当前面板中截至原截面日的数据重新计算。"
        )
    COMPUTE_PROGRESS.finish(job_id, message="稳健协方差与风险预算配权完成")
    return {
        "experiment_id": experiment_id,
        "run_id": row.id,
        "market": row.market,
        "portfolio_mode": row.portfolio_mode,
        "snapshot_panel_identity": row.panel_identity,
        "current_panel_identity": current_panel_identity,
        "panel_changed": panel_changed,
        **allocation,
    }


@router.get("/screener/runs")
async def list_screener_runs(
    experiment_id: int | None = None,
    limit: int = 30,
    offset: int = 0,
):
    if not 1 <= limit <= 100:
        raise HTTPException(400, "limit 必须在 1 到 100 之间")
    if offset < 0:
        raise HTTPException(400, "offset 不能小于 0")
    eid, cfg = await _experiment_context(experiment_id)
    where = ScreenerRun.experiment_id == eid
    async with SessionLocal() as session:
        total = int(
            await session.scalar(
                select(func.count(ScreenerRun.id)).where(where)
            )
            or 0
        )
        rows = (
            await session.scalars(
                select(ScreenerRun)
                .where(where)
                .order_by(ScreenerRun.id.desc())
                .offset(offset)
                .limit(limit)
            )
        ).all()
    return {
        "experiment_id": eid,
        "market": cfg.get("market", "us"),
        "portfolio_mode": cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE),
        "total": total,
        "limit": limit,
        "offset": offset,
        "runs": [_screener_run_payload(row) for row in rows],
    }


@router.get("/screener/runs/{run_id}")
async def screener_run_detail(
    run_id: int,
    experiment_id: int | None = None,
):
    eid, _ = await _experiment_context(experiment_id)
    async with SessionLocal() as session:
        row = await session.scalar(
            select(ScreenerRun).where(
                ScreenerRun.id == run_id,
                ScreenerRun.experiment_id == eid,
            )
        )
    if not row:
        raise HTTPException(404, "该任务下不存在这条选股记录")
    return {
        "experiment_id": eid,
        "run": _screener_run_payload(row, include_snapshot=True),
    }


# ---------- 元信息 ----------

@router.get("/meta")
async def meta(experiment_id: int | None = None, load_panel: bool = False):
    _, cfg = await _experiment_context(experiment_id)
    market = cfg.get("market", "us")
    panel = PanelStore.get(cfg.get("panel_glob"), market)
    summary = await asyncio.to_thread(panel.summary, load_panel)
    if panel.df is not None:
        summary["loaded_columns"] = panel.df.columns
    return {
        "panel": summary,
        "market": market,
        "portfolio_mode": cfg.get("portfolio_mode"),
        "direction": int(cfg.get("direction", 1)),
        "direction_policy": cfg.get(
            "direction_policy",
            DEFAULT_RESEARCH_DIRECTION_POLICY,
        ),
        "evaluation_protocol": cfg.get("evaluation_protocol", "legacy"),
        "evaluation_config": evaluation_config(market, cfg.get("evaluation_config")),
        "dsl_fields": get_dsl_fields(cfg.get("market")),
        "dsl_field_contract": field_contract(market),
        "mechanism_families": list(mechanisms_for_market(market)),
        "operators": OPERATORS_DOC,
    }


class PanelReloadReq(BaseModel):
    experiment_id: int | None = None
    force: bool = False


@router.post("/panels/reload")
async def reload_panel(req: PanelReloadReq):
    """Atomically replace one task panel while the old generation keeps serving."""
    experiment_id, cfg = await _experiment_context(req.experiment_id)
    market = cfg.get("market", "us")
    panel_glob = cfg.get("panel_glob") or default_panel_glob(market)
    store = PanelStore.get(panel_glob, market)
    before = await asyncio.to_thread(store.diagnostics)
    result = await asyncio.to_thread(
        store.reload_if_changed,
        force=req.force,
        require_stable=False,
    )
    invalidation = None
    if result.get("status") in {"loaded", "reloaded"}:
        invalidation = invalidate_panel_dependents()
    after = await asyncio.to_thread(store.diagnostics)
    return {
        "experiment_id": experiment_id,
        "market": market,
        "panel_glob": panel_glob,
        "result": result,
        "cache_invalidation": invalidation,
        "before": before,
        "after": after,
        "policy": {
            "swap": "double_buffer_atomic_generation",
            "ongoing_requests": "retain_previous_immutable_frame",
            "failed_reload": "continue_serving_previous_generation",
        },
    }


def _database_pool_snapshot() -> dict:
    pool = db_engine.sync_engine.pool

    def read(name: str, default=None):
        value = getattr(pool, name, default)
        try:
            return value() if callable(value) else value
        except Exception:  # noqa: BLE001 - diagnostics must not break health checks
            return default

    size = int(read("size", 0) or 0)
    checked_out = int(read("checkedout", 0) or 0)
    checked_in = int(read("checkedin", 0) or 0)
    overflow = int(read("overflow", 0) or 0)
    max_overflow = int(getattr(pool, "_max_overflow", 0) or 0)
    capacity = max(1, size + max(0, max_overflow))
    return {
        "class": type(pool).__name__,
        "size": size,
        "max_overflow": max_overflow,
        "capacity": capacity,
        "checked_in": checked_in,
        "checked_out": checked_out,
        "overflow": overflow,
        "utilization": round(checked_out / capacity, 6),
        "timeout_seconds": read("timeout"),
        "status_text": redact_text(read("status", "unavailable"), 400),
    }


async def _database_observability() -> dict:
    started = time.perf_counter()
    result = {
        "status": "unknown",
        "url": make_url(DATABASE_URL).render_as_string(hide_password=True),
        "driver": make_url(DATABASE_URL).drivername,
        "pool": _database_pool_snapshot(),
        "counts": {},
        "backtest_statuses": {},
        "experiment_statuses": {},
        "factor_lifecycle": {},
        "event_levels_1h": {},
        "protocol_lineage": {},
        "llm_pipeline": {},
    }
    try:
        async with SessionLocal() as s:
            await s.execute(text("SELECT 1"))
            counts = (
                await s.execute(text(
                    """
                    SELECT
                      (SELECT COUNT(*) FROM experiments) AS experiments,
                      (SELECT COUNT(*) FROM factors) AS factors,
                      (SELECT COUNT(*) FROM nodes) AS nodes,
                      (SELECT COUNT(*) FROM trials) AS trials,
                      (SELECT COUNT(*) FROM outer_steps) AS outer_steps,
                      (SELECT COUNT(*) FROM backtests) AS backtests,
                      (SELECT COUNT(*) FROM engine_events) AS engine_events,
                      (SELECT COUNT(*) FROM llm_call_audits) AS llm_call_audits
                    """
                ))
            ).mappings().one()
            result["counts"] = {key: int(value or 0) for key, value in counts.items()}
            result["backtest_statuses"] = {
                str(status): int(count)
                for status, count in (
                    await s.execute(text(
                        "SELECT status, COUNT(*) FROM backtests GROUP BY status"
                    ))
                ).all()
            }
            result["experiment_statuses"] = {
                str(status): int(count)
                for status, count in (
                    await s.execute(text(
                        "SELECT status, COUNT(*) FROM experiments GROUP BY status"
                    ))
                ).all()
            }
            result["factor_lifecycle"] = {
                str(status): int(count)
                for status, count in (
                    await s.execute(text(
                        "SELECT lifecycle_stage, COUNT(*) FROM factors "
                        "GROUP BY lifecycle_stage ORDER BY COUNT(*) DESC"
                    ))
                ).all()
            }
            result["event_levels_1h"] = {
                str(level): int(count)
                for level, count in (
                    await s.execute(text(
                        "SELECT level, COUNT(*) FROM engine_events "
                        "WHERE created_at >= NOW() - INTERVAL '1 hour' "
                        "GROUP BY level"
                    ))
                ).all()
            }
            protocol_lineage = {}
            for table_name in (
                "nodes",
                "trials",
                "miner_versions",
                "outer_steps",
                "factors",
                "llm_call_audits",
            ):
                protocol_lineage[table_name] = {
                    str(protocol): int(count)
                    for protocol, count in (
                        await s.execute(text(
                            f"SELECT evaluation_protocol, COUNT(*) "
                            f"FROM {table_name} "
                            "GROUP BY evaluation_protocol "
                            "ORDER BY COUNT(*) DESC"
                        ))
                    ).all()
                }
            result["protocol_lineage"] = {
                "current_protocol": EVALUATION_PROTOCOL_VERSION,
                "tables": protocol_lineage,
                "policy": (
                    "历史协议只读保留；当前 Miner 上下文、meta-score "
                    "比较和外层决策仅使用同协议记录。"
                ),
            }

            node_coverage = (
                await s.execute(text(
                    """
                    SELECT COUNT(*) AS total,
                           COUNT(*) FILTER (
                             WHERE COALESCE(feedback_summary::jsonb, '{}'::jsonb)
                                   <> '{}'::jsonb
                           ) AS covered
                    FROM nodes
                    WHERE evaluation_protocol = :protocol
                    """
                ), {"protocol": EVALUATION_PROTOCOL_VERSION})
            ).mappings().one()
            version_coverage = (
                await s.execute(text(
                    """
                    SELECT COUNT(*) AS total,
                           COUNT(*) FILTER (WHERE meta_score IS NOT NULL)
                             AS evaluated,
                           COUNT(*) FILTER (
                             WHERE meta_score IS NOT NULL
                               AND COALESCE(feedback_summary::jsonb, '{}'::jsonb)
                                   <> '{}'::jsonb
                           ) AS reports,
                           COUNT(*) FILTER (
                             WHERE meta_score IS NOT NULL
                               AND COALESCE(reflection::jsonb, '{}'::jsonb)
                                   <> '{}'::jsonb
                           ) AS reflections
                    FROM miner_versions
                    WHERE evaluation_protocol = :protocol
                    """
                ), {"protocol": EVALUATION_PROTOCOL_VERSION})
            ).mappings().one()
            llm_stats = (
                await s.execute(text(
                    """
                    SELECT COUNT(*) AS total,
                           COUNT(*) FILTER (
                             WHERE created_at >= NOW() - INTERVAL '1 hour'
                           ) AS calls_1h,
                           COUNT(*) FILTER (
                             WHERE status = 'transport_error'
                               AND created_at >= NOW() - INTERVAL '1 hour'
                           ) AS errors_1h,
                           COUNT(*) FILTER (
                             WHERE status = 'rejected'
                               AND created_at >= NOW() - INTERVAL '1 hour'
                           ) AS semantic_rejections_1h,
                           AVG(latency_ms) FILTER (
                             WHERE created_at >= NOW() - INTERVAL '1 hour'
                           ) AS avg_latency_1h,
                           percentile_cont(0.95) WITHIN GROUP (
                             ORDER BY latency_ms
                           ) FILTER (
                             WHERE created_at >= NOW() - INTERVAL '1 hour'
                           ) AS p95_latency_1h
                    FROM llm_call_audits
                    """
                ))
            ).mappings().one()
            calls_by_role = {
                str(role): int(count)
                for role, count in (
                    await s.execute(text(
                        "SELECT role, COUNT(*) FROM llm_call_audits "
                        "GROUP BY role ORDER BY COUNT(*) DESC"
                    ))
                ).all()
            }
            calls_by_status = {
                str(status): int(count)
                for status, count in (
                    await s.execute(text(
                        "SELECT status, COUNT(*) FROM llm_call_audits "
                        "GROUP BY status ORDER BY COUNT(*) DESC"
                    ))
                ).all()
            }
            recent_llm_calls = (
                await s.scalars(
                    select(LLMCallAudit)
                    .order_by(LLMCallAudit.id.desc())
                    .limit(20)
                )
            ).all()

            node_total = int(node_coverage["total"] or 0)
            node_covered = int(node_coverage["covered"] or 0)
            version_evaluated = int(version_coverage["evaluated"] or 0)
            result["llm_pipeline"] = {
                "feedback_schema": FEEDBACK_SCHEMA_VERSION,
                "outer_report_schema": OUTER_REPORT_SCHEMA_VERSION,
                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                "isolation": {
                    "training_feedback": "PUBLIC + META_TRAIN 保守聚合",
                    "sealed_layers_in_prompts": False,
                    "cross_protocol_context": False,
                    "seed_context": "冻结共同历史 + 本种子独立增量",
                },
                "feedback_coverage": {
                    "nodes_total": node_total,
                    "nodes_with_feedback": node_covered,
                    "nodes_ratio": round(
                        node_covered / max(1, node_total),
                        6,
                    ) if node_total else None,
                    "versions_total": int(
                        version_coverage["total"] or 0
                    ),
                    "versions_evaluated": version_evaluated,
                    "versions_with_report": int(
                        version_coverage["reports"] or 0
                    ),
                    "versions_with_reflection": int(
                        version_coverage["reflections"] or 0
                    ),
                    "reports_ratio": round(
                        int(version_coverage["reports"] or 0)
                        / max(1, version_evaluated),
                        6,
                    ) if version_evaluated else None,
                    "reflections_ratio": round(
                        int(version_coverage["reflections"] or 0)
                        / max(1, version_evaluated),
                        6,
                    ) if version_evaluated else None,
                },
                "calls": {
                    "total": int(llm_stats["total"] or 0),
                    "calls_1h": int(llm_stats["calls_1h"] or 0),
                    "errors_1h": int(llm_stats["errors_1h"] or 0),
                    "semantic_rejections_1h": int(
                        llm_stats["semantic_rejections_1h"] or 0
                    ),
                    "avg_latency_ms_1h": (
                        round(float(llm_stats["avg_latency_1h"]), 3)
                        if llm_stats["avg_latency_1h"] is not None
                        else None
                    ),
                    "p95_latency_ms_1h": (
                        round(float(llm_stats["p95_latency_1h"]), 3)
                        if llm_stats["p95_latency_1h"] is not None
                        else None
                    ),
                    "by_role": calls_by_role,
                    "by_status": calls_by_status,
                },
                "recent_calls": [
                    {
                        "id": row.id,
                        "experiment_id": row.experiment_id,
                        "role": row.role,
                        "phase": row.phase,
                        "status": row.status,
                        "provider_name": row.provider_name,
                        "model": row.model,
                        "evaluation_protocol": row.evaluation_protocol,
                        "miner_version_id": row.miner_version_id,
                        "outer_step_no": row.outer_step_no,
                        "task_name": row.task_name,
                        "prompt_hash": row.prompt_hash,
                        "feedback_fingerprint": row.feedback_fingerprint,
                        "latency_ms": row.latency_ms,
                        "error": redact_text(row.error, 500),
                        "created_at": str(row.created_at),
                    }
                    for row in recent_llm_calls
                ],
            }
        result["status"] = "ok"
        result["error"] = None
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = redact_text(exc, 1200)
    result["latency_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
    result["pool"] = _database_pool_snapshot()
    return result


def _provider_observability(value: dict | None) -> dict:
    value = value or {}
    inner_name = value.get("inner_provider") or ""
    outer_name = value.get("outer_provider") or ""
    providers = []
    configured_names = set()
    for provider in (value.get("providers") or [])[:50]:
        name = str(provider.get("name") or "")
        has_key = bool(provider.get("api_key"))
        if has_key:
            configured_names.add(name)
        base_url = str(provider.get("base_url") or provider.get("url") or "")
        parsed = urlparse(base_url)
        providers.append({
            "name": name,
            "format": provider.get("format") or provider.get("provider"),
            "model": provider.get("model"),
            "endpoint_host": parsed.netloc or parsed.path.split("/")[0] or None,
            "api_key_present": has_key,
            "selected_for": [
                role
                for role, selected in (
                    ("inner", inner_name),
                    ("outer", outer_name),
                )
                if selected == name
            ],
        })
    return {
        "inner_provider": inner_name or None,
        "outer_provider": outer_name or None,
        "inner_provider_configured": inner_name in configured_names,
        "outer_provider_configured": outer_name in configured_names,
        "provider_count": len(providers),
        "providers": providers,
        "note": "仅显示配置状态、模型和主机；凭据不会进入诊断快照。",
    }


async def _active_configuration_snapshot() -> tuple[dict, dict]:
    active = {
        "experiment_id": None,
        "name": None,
        "status": "unknown",
        "config_error": None,
    }
    provider_value: dict = {}
    try:
        experiment_id = await get_active_experiment_id()
        async with SessionLocal() as s:
            experiment = await s.get(Experiment, experiment_id)
            providers = await s.get(Setting, "llm_providers")
        provider_value = dict(providers.value or {}) if providers else {}
        if experiment is None:
            raise RuntimeError(f"活动研究任务 {experiment_id} 不存在")
        config = dict(experiment.research_config or {})
        market = config.get("market", "us")
        panel_glob = config.get("panel_glob") or default_panel_glob(market)
        PanelStore.get(panel_glob, market)
        active.update({
            "experiment_id": experiment.id,
            "name": experiment.name,
            "status": experiment.status,
            "created_at": str(experiment.created_at),
            "market": market,
            "portfolio_mode": config.get(
                "portfolio_mode",
                "long_only" if market == "ashare" else "long_short",
            ),
            "direction": int(config.get("direction", 1)),
            "direction_policy": config.get(
                "direction_policy",
                DEFAULT_RESEARCH_DIRECTION_POLICY,
            ),
            "evaluation_protocol": config.get(
                "evaluation_protocol",
                "legacy",
            ),
            "panel_glob": panel_glob,
            "panel_id": fingerprint_payload((market, panel_glob)),
            "config_fingerprint": fingerprint_payload(config),
            "config": redact_value(config),
        })
    except Exception as exc:  # noqa: BLE001
        active["config_error"] = redact_text(exc, 1200)
    return active, _provider_observability(provider_value)


def _artifact_observability() -> dict:
    root = BACKTEST_ARTIFACT_ROOT
    if not root.exists():
        return {
            "root": str(root),
            "exists": False,
            "runs": 0,
            "files": 0,
            "total_bytes": 0,
            "latest_mtime": None,
            "inventory_truncated": False,
        }
    manifests = list(root.glob("*/manifest.json"))
    files = []
    for path in root.rglob("*"):
        if path.is_file():
            files.append(path)
            if len(files) >= 10000:
                break
    total_bytes = 0
    latest_mtime = 0.0
    stat_errors = 0
    for path in files:
        try:
            stat = path.stat()
            total_bytes += stat.st_size
            latest_mtime = max(latest_mtime, stat.st_mtime)
        except OSError:
            stat_errors += 1
    diversity_audit_root = (
        root.parent / "audits" / "p0-return-source-diversity-v1"
    )
    diversity_audit: dict = {
        "root": str(diversity_audit_root),
        "exists": diversity_audit_root.exists(),
    }
    for filename, key in (
        ("progress.json", "progress"),
        ("protocol.json", "protocol"),
        ("summary.json", "summary"),
    ):
        path = diversity_audit_root / filename
        if not path.exists():
            continue
        try:
            diversity_audit[key] = json.loads(
                path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            diversity_audit[f"{key}_error"] = redact_text(exc, 500)
    return {
        "root": str(root),
        "exists": True,
        "runs": len(manifests),
        "files": len(files),
        "total_bytes": total_bytes,
        "latest_mtime": (
            datetime.fromtimestamp(latest_mtime, timezone.utc).isoformat(
                timespec="seconds"
            )
            if latest_mtime
            else None
        ),
        "stat_errors": stat_errors,
        "inventory_truncated": len(files) >= 10000,
        "factor_return_source_audit": diversity_audit,
    }


def _similarity_cache_observability() -> dict:
    stats = dict(_similarity_cache_stats)
    attempts = int(stats["hits"]) + int(stats["misses"])
    stats.update({
        "entries": len(_similarity_cache),
        "capacity": 24,
        "hit_rate": round(int(stats["hits"]) / max(1, attempts), 6),
    })
    return stats


async def _recent_engine_events(
    *,
    limit: int,
    level: str | None,
    experiment_id: int | None,
) -> tuple[list[dict], str | None]:
    if limit <= 0:
        return [], None
    try:
        stmt = select(EngineEvent)
        if level:
            stmt = stmt.where(EngineEvent.level == level)
        if experiment_id is not None:
            stmt = stmt.where(EngineEvent.experiment_id == experiment_id)
        async with SessionLocal() as s:
            rows = (
                await s.scalars(
                    stmt.order_by(EngineEvent.id.desc()).limit(limit)
                )
            ).all()
        return [
            {
                "id": event.id,
                "experiment_id": event.experiment_id,
                "level": event.level,
                "message": redact_text(event.message, 1600),
                "created_at": str(event.created_at),
                "payload": redact_value(event.payload or {}),
            }
            for event in rows
        ], None
    except Exception as exc:  # noqa: BLE001
        return [], redact_text(exc, 1200)


async def _collect_observability_components() -> dict:
    """Collect the slower database/filesystem surfaces once per short TTL."""
    database, active_and_providers = await asyncio.gather(
        _database_observability(),
        _active_configuration_snapshot(),
    )
    active_task, providers = active_and_providers
    data, artifacts = await asyncio.gather(
        asyncio.to_thread(PanelStore.registry_snapshot),
        asyncio.to_thread(_artifact_observability),
    )
    return {
        "collected_at": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "database": database,
        "active_task": active_task,
        "providers": providers,
        "data": data,
        "artifacts": artifacts,
    }


async def _collect_observability(
    *,
    events_limit: int = 50,
    event_level: str | None = None,
    event_experiment_id: int | None = None,
    window_seconds: int = 300,
    force: bool = False,
) -> dict:
    collector_started = time.perf_counter()
    manager = EngineManager.get()
    components = await _observability_components_cache.get(
        "components",
        _collect_observability_components,
        force=force,
    )
    database = components["database"]
    active_task = components["active_task"]
    providers = components["providers"]
    data = components["data"]
    artifacts = components["artifacts"]
    workers = manager.all_status(include_logs=False)
    required_panel_sources = {
        active_task.get("panel_glob")
    } if active_task.get("panel_glob") else set()
    for worker in manager.workers.values():
        if not worker.running:
            continue
        market = worker.task_config.get("market", "us")
        required_panel_sources.add(
            worker.task_config.get("panel_glob") or default_panel_glob(market)
        )
    for panel in data["panels"]:
        panel["required"] = panel.get("source") in required_panel_sources
        if panel.get("source") == active_task.get("panel_glob"):
            active_task["panel_id"] = panel.get("id")
            active_task["panel_identity"] = panel.get("identity")
    event_cache_key = (
        events_limit,
        event_level or "",
        event_experiment_id,
    )

    async def load_events() -> tuple[list[dict], str | None]:
        return await _recent_engine_events(
            limit=events_limit,
            level=event_level,
            experiment_id=event_experiment_id,
        )

    events, events_error = await _observability_events_cache.get(
        event_cache_key,
        load_events,
        force=force,
    )
    active_id = active_task.get("experiment_id")
    active_worker = (
        manager.status_for(active_id, include_logs=False)
        if active_id is not None
        else None
    )
    if active_worker:
        active_task["worker_state"] = active_worker.get("state")
        active_task["worker_phase"] = active_worker.get("phase")
    requests = OBSERVABILITY.request_snapshot(window_seconds)
    service = OBSERVABILITY.service_snapshot()
    snapshot = {
        "schema_version": "factorfactory.observability/v4",
        "generated_at": service["generated_at"],
        "service": service,
        "process": OBSERVABILITY.process_snapshot(),
        "requests": requests,
        "database": database,
        "llm_pipeline": database.get("llm_pipeline", {}),
        "protocol_lineage": database.get("protocol_lineage", {}),
        "data": data,
        "caches": {
            "panel": {
                key: value for key, value in data.items() if key != "panels"
            },
            "screener": SCREEN_CACHE.stats(),
            "factor_similarity": _similarity_cache_observability(),
            "observability_components": (
                _observability_components_cache.stats()
            ),
            "observability_events": _observability_events_cache.stats(),
        },
        "artifacts": artifacts,
        "active_task": active_task,
        "providers": providers,
        "engine": {
            "worker_count": len(manager.workers),
            "running_count": sum(worker.running for worker in manager.workers.values()),
            "stale_heartbeat_count": sum(
                bool(worker.get("heartbeat_stale")) for worker in workers
            ),
        },
        "workers": workers,
        "recent_events": events,
        "recent_events_error": events_error,
        "collector": {
            "components_collected_at": components["collected_at"],
            "forced": force,
            "total_ms": round(
                (time.perf_counter() - collector_started) * 1000.0,
                3,
            ),
            "policy": (
                "进程/请求/worker 每次实时采集；数据库、面板与产物清单缓存 "
                "10 秒；持久化事件缓存 3 秒。"
            ),
        },
    }
    snapshot["slo"] = build_slo(snapshot)
    findings = build_findings(snapshot)
    snapshot["findings"] = findings
    snapshot["health"] = overall_health(findings)
    # Compatibility aliases for existing local clients.
    snapshot["panel_cache"] = snapshot["caches"]["panel"]
    snapshot["screener_cache"] = snapshot["caches"]["screener"]
    snapshot["backtest_artifacts"] = artifacts
    return snapshot


@router.get("/observability")
async def observability(
    events_limit: int = 50,
    event_level: str | None = None,
    event_experiment_id: int | None = None,
    window_seconds: int = 300,
    force: bool = False,
):
    """Secret-safe engineering snapshot for diagnosis and incident hand-off."""
    if not 0 <= events_limit <= 200:
        raise HTTPException(400, "events_limit 必须在 0..200")
    if not 60 <= window_seconds <= 3600:
        raise HTTPException(400, "window_seconds 必须在 60..3600")
    return await _collect_observability(
        events_limit=events_limit,
        event_level=event_level,
        event_experiment_id=event_experiment_id,
        window_seconds=window_seconds,
        force=force,
    )


@router.get("/health/live")
async def health_live():
    """Liveness only: proves the event loop can still serve a request."""
    service = OBSERVABILITY.service_snapshot()
    return {
        "status": "alive",
        "service": service["name"],
        "pid": service["pid"],
        "started_at": service["started_at"],
        "uptime_seconds": service["uptime_seconds"],
        "generated_at": service["generated_at"],
    }


@router.get("/health/ready")
async def health_ready():
    """Readiness: database and configured panel source must be reachable."""
    snapshot = await _collect_observability(events_limit=0, window_seconds=300)
    panel_errors = [
        panel
        for panel in snapshot["data"]["panels"]
        if panel.get("required")
        and (
            panel.get("state") == "error"
            or panel.get("source_error")
            or panel.get("schema_status") == "error"
        )
    ]
    ready = snapshot["database"]["status"] == "ok" and not panel_errors
    payload = {
        "status": "ready" if ready else "not_ready",
        "health": snapshot["health"],
        "database": {
            "status": snapshot["database"]["status"],
            "latency_ms": snapshot["database"]["latency_ms"],
            "error": snapshot["database"].get("error"),
        },
        "panels": {
            "instances": snapshot["data"]["instances"],
            "loaded": snapshot["data"]["loaded"],
            "errors": len(panel_errors),
            "stale": snapshot["data"].get("stale", 0),
            "reloading": snapshot["data"].get("reloading", 0),
            "reload_errors": snapshot["data"].get("reload_errors", 0),
        },
        "findings": snapshot["findings"],
        "generated_at": snapshot["generated_at"],
    }
    return JSONResponse(payload, status_code=200 if ready else 503)


def _prometheus_escape(value: object) -> str:
    return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')


@router.get("/metrics")
async def prometheus_metrics():
    """Small Prometheus-compatible surface; labels are deliberately bounded."""
    snapshot = await _collect_observability(events_limit=0, window_seconds=300)
    requests = snapshot["requests"]
    lifetime = requests["lifetime"]
    window = requests["window"]
    process = snapshot["process"]
    pool = snapshot["database"]["pool"]
    data = snapshot["data"]
    screener_cache = snapshot["caches"]["screener"]
    llm_pipeline = snapshot.get("llm_pipeline") or {}
    llm_calls = llm_pipeline.get("calls") or {}
    feedback_coverage = llm_pipeline.get("feedback_coverage") or {}
    lines = [
        "# HELP factorfactory_info Deployment identity.",
        "# TYPE factorfactory_info gauge",
        (
            'factorfactory_info{commit="'
            f'{_prometheus_escape(snapshot["service"]["deployment"]["commit_short"])}",'
            f'branch="{_prometheus_escape(snapshot["service"]["deployment"]["branch"])}"'
            "} 1"
        ),
        "# TYPE factorfactory_health_status gauge",
        (
            'factorfactory_health_status{status="'
            f'{_prometheus_escape(snapshot["health"])}'
            '"} 1'
        ),
        "# TYPE factorfactory_http_requests_total counter",
        f'factorfactory_http_requests_total {lifetime["requests"]}',
        "# TYPE factorfactory_http_server_errors_total counter",
        f'factorfactory_http_server_errors_total {lifetime["server_errors"]}',
        "# TYPE factorfactory_http_client_errors_total counter",
        f'factorfactory_http_client_errors_total {lifetime["client_errors"]}',
        "# TYPE factorfactory_http_in_flight gauge",
        f'factorfactory_http_in_flight {requests["in_flight"]}',
        "# TYPE factorfactory_http_latency_milliseconds gauge",
        (
            'factorfactory_http_latency_milliseconds{quantile="0.50"} '
            f'{window["latency_ms"]["p50"]}'
        ),
        (
            'factorfactory_http_latency_milliseconds{quantile="0.95"} '
            f'{window["latency_ms"]["p95"]}'
        ),
        (
            'factorfactory_http_latency_milliseconds{quantile="0.99"} '
            f'{window["latency_ms"]["p99"]}'
        ),
        "# TYPE factorfactory_http_window_requests_per_second gauge",
        (
            "factorfactory_http_window_requests_per_second "
            f'{window["requests_per_second"]}'
        ),
        "# TYPE factorfactory_process_resident_memory_bytes gauge",
        f'factorfactory_process_resident_memory_bytes {process.get("rss_bytes", 0)}',
        "# TYPE factorfactory_process_cpu_percent gauge",
        f'factorfactory_process_cpu_percent {process.get("cpu_percent", 0)}',
        "# TYPE factorfactory_event_loop_lag_milliseconds gauge",
        (
            "factorfactory_event_loop_lag_milliseconds "
            f'{requests["event_loop"].get("p95_lag_ms", 0)}'
        ),
        "# TYPE factorfactory_database_pool_checked_out gauge",
        f'factorfactory_database_pool_checked_out {pool.get("checked_out", 0)}',
        "# TYPE factorfactory_database_up gauge",
        (
            "factorfactory_database_up "
            f'{1 if snapshot["database"]["status"] == "ok" else 0}'
        ),
        "# TYPE factorfactory_database_pool_utilization_ratio gauge",
        f'factorfactory_database_pool_utilization_ratio {pool.get("utilization", 0)}',
        "# TYPE factorfactory_research_workers gauge",
        (
            'factorfactory_research_workers{state="running"} '
            f'{snapshot["engine"]["running_count"]}'
        ),
        (
            'factorfactory_research_workers{state="registered"} '
            f'{snapshot["engine"]["worker_count"]}'
        ),
        "# TYPE factorfactory_panel_instances gauge",
        f'factorfactory_panel_instances {data["instances"]}',
        "# TYPE factorfactory_panel_loaded gauge",
        f'factorfactory_panel_loaded {data["loaded"]}',
        "# TYPE factorfactory_panel_errors gauge",
        f'factorfactory_panel_errors {data["errors"]}',
        "# TYPE factorfactory_panel_memory_bytes gauge",
        f'factorfactory_panel_memory_bytes {data["estimated_size_bytes"]}',
        "# TYPE factorfactory_panel_stale gauge",
        f'factorfactory_panel_stale {data.get("stale", 0)}',
        "# TYPE factorfactory_panel_reloading gauge",
        f'factorfactory_panel_reloading {data.get("reloading", 0)}',
        "# TYPE factorfactory_panel_reload_errors gauge",
        f'factorfactory_panel_reload_errors {data.get("reload_errors", 0)}',
        "# TYPE factorfactory_screener_cache_requests_total counter",
        (
            'factorfactory_screener_cache_requests_total{result="hit"} '
            f'{screener_cache["hits"]}'
        ),
        (
            'factorfactory_screener_cache_requests_total{result="miss"} '
            f'{screener_cache["misses"]}'
        ),
        "# TYPE factorfactory_worker_stale_heartbeats gauge",
        (
            "factorfactory_worker_stale_heartbeats "
            f'{snapshot["engine"]["stale_heartbeat_count"]}'
        ),
        "# TYPE factorfactory_backtest_artifact_bytes gauge",
        (
            "factorfactory_backtest_artifact_bytes "
            f'{snapshot["artifacts"]["total_bytes"]}'
        ),
        "# TYPE factorfactory_observability_collection_milliseconds gauge",
        (
            "factorfactory_observability_collection_milliseconds "
            f'{snapshot["collector"]["total_ms"]}'
        ),
        "# TYPE factorfactory_slo_objectives gauge",
        (
            'factorfactory_slo_objectives{status="failed"} '
            f'{snapshot["slo"]["failed"]}'
        ),
        "# TYPE factorfactory_llm_calls gauge",
        (
            'factorfactory_llm_calls{window="lifetime"} '
            f'{llm_calls.get("total", 0)}'
        ),
        (
            'factorfactory_llm_calls{window="1h"} '
            f'{llm_calls.get("calls_1h", 0)}'
        ),
        "# TYPE factorfactory_llm_calls_by_status gauge",
        "# TYPE factorfactory_llm_calls_by_role gauge",
        "# TYPE factorfactory_llm_failures gauge",
        (
            'factorfactory_llm_failures{window="1h"} '
            f'{llm_calls.get("errors_1h", 0)}'
        ),
        "# TYPE factorfactory_llm_latency_milliseconds gauge",
        (
            'factorfactory_llm_latency_milliseconds{window="1h",quantile="0.95"} '
            f'{llm_calls.get("p95_latency_ms_1h") or 0}'
        ),
        "# TYPE factorfactory_feedback_coverage_ratio gauge",
        (
            'factorfactory_feedback_coverage_ratio{artifact="node"} '
            f'{feedback_coverage.get("nodes_ratio") or 0}'
        ),
        (
            'factorfactory_feedback_coverage_ratio{artifact="outer_report"} '
            f'{feedback_coverage.get("reports_ratio") or 0}'
        ),
        (
            'factorfactory_feedback_coverage_ratio{artifact="outer_reflection"} '
            f'{feedback_coverage.get("reflections_ratio") or 0}'
        ),
    ]
    for status, count in (llm_calls.get("by_status") or {}).items():
        lines.append(
            'factorfactory_llm_calls_by_status{status="'
            f'{_prometheus_escape(status)}'
            f'"}} {count}'
        )
    for role, count in (llm_calls.get("by_role") or {}).items():
        lines.append(
            'factorfactory_llm_calls_by_role{role="'
            f'{_prometheus_escape(role)}'
            f'"}} {count}'
        )
    for route in requests["routes"]:
        label = _prometheus_escape(route["route"])
        lines.extend([
            f'factorfactory_http_route_requests_total{{route="{label}"}} {route["count"]}',
            f'factorfactory_http_route_server_errors_total{{route="{label}"}} {route["errors"]}',
            f'factorfactory_http_route_p95_milliseconds{{route="{label}"}} {route["p95_ms"]}',
        ])
    return PlainTextResponse(
        "\n".join(lines) + "\n",
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
