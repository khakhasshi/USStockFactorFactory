import asyncio
from datetime import datetime
from pathlib import Path

import polars as pl
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from ..backtest.engine import run_backtest
from ..config import (
    BACKTEST_ARTIFACT_ROOT,
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_EVALUATION_CONFIG,
    DEFAULT_PORTFOLIO_MODE,
    EVALUATION_PROTOCOL_VERSION,
    PANEL_GLOB,
    default_panel_glob,
    evaluation_config,
    get_dsl_fields,
    resolve_engine_tasks,
)
from ..data.panel import PanelStore
from ..db import SessionLocal, get_active_experiment_id
from ..dsl.engine import (
    OPERATORS_DOC,
    expression_profile,
    parse,
    validate,
)
from ..eval.harness import evaluate, evaluate_full
from ..eval.ranking import ranking_diagnostics
from ..factors.similarity import (
    build_similarity_index,
    expression_fingerprint,
    nearest_factors,
)
from ..models import Backtest, EngineEvent, Experiment, Factor, MinerVersion, Node, OuterStep, Setting, Trial
from ..orchestrator import EngineManager
from ..screener import SCREEN_CACHE, screen_cross_section

router = APIRouter(prefix="/api")
_similarity_cache: dict[tuple[int, int, int, float], dict] = {}


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
    local_tasks = (cfg.get("engine_config") or {}).get("tasks")
    global_tasks = (global_engine.value if global_engine else {}).get("tasks")
    return resolve_engine_tasks(
        local_tasks or global_tasks or DEFAULT_ENGINE_CONFIG["tasks"],
        market,
        mode,
        direction,
        preserve_declared_costs=bool(local_tasks),
    )


def _compact_layer_metrics(metrics: dict | None) -> dict:
    metrics = metrics or {}
    compact = {
        key: metrics.get(key)
        for key in (
            "available",
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
            "score_pre_vault",
            "status",
            "vault_seal",
            "policy_label",
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
                "cost_breakeven_bps",
                "cost_cushion_multiple",
                "worst_stress_sharpe",
            )
        }
    return compact


def _factor_payload(f: Factor, include_validation: bool = False) -> dict:
    validation = f.validation_metrics or {}
    payload = {
        "id": f.id,
        "experiment_id": f.experiment_id,
        "name": f.name,
        "expression": f.expression,
        "status": f.status,
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
            validation.get("ranking") or {}
            if include_validation
            else _compact_ranking(validation)
        ),
        "evaluated_at": str(f.evaluated_at) if f.evaluated_at else None,
        "created_at": str(f.created_at),
    }
    if include_validation:
        payload["validation"] = validation
        payload["fingerprint"] = f.fingerprint or {}
    return payload


# ---------- 引擎 ----------

class EngineStartReq(BaseModel):
    mode: str = "v2"
    experiment_id: int | None = None


@router.post("/engine/start")
async def engine_start(req: EngineStartReq | None = None):
    mode = req.mode if req else "v1"
    if mode not in ("v1", "v2"):
        raise HTTPException(400, "mode 必须为 v1 或 v2")
    return await EngineManager.get().start(mode, req.experiment_id if req else None)


@router.post("/engine/stop")
async def engine_stop(req: dict | None = None):
    experiment_id = req.get("experiment_id") if req else None
    return await EngineManager.get().stop(experiment_id)


@router.get("/engine/status")
async def engine_status():
    manager = EngineManager.get()
    exp_id = await get_active_experiment_id()
    async with SessionLocal() as s:
        exp = await s.get(Experiment, exp_id)
        n_factors = await s.scalar(
            select(func.count(Factor.id)).where(Factor.experiment_id == exp_id))
        n_nodes = await s.scalar(
            select(func.count(Node.id)).where(Node.experiment_id == exp_id))
        n_steps = await s.scalar(
            select(func.count(OuterStep.id)).where(OuterStep.experiment_id == exp_id))
        accepted = await s.scalar(
            select(func.count(OuterStep.id)).where(OuterStep.accepted, OuterStep.experiment_id == exp_id))
        inc = await s.scalar(
            select(MinerVersion)
            .where(MinerVersion.status == "incumbent", MinerVersion.experiment_id == exp_id)
            .order_by(MinerVersion.id.desc())
        )
    runtime = manager.status_for(exp_id)
    return {
        **runtime,
        "experiment": {"id": exp_id, "name": exp.name if exp else "?",
                       "status": exp.status if exp else "?"},
        "counts": {"factors": n_factors, "nodes": n_nodes, "outer_steps": n_steps, "accepted": accepted},
        "incumbent": {
            "version_no": inc.version_no, "meta_score": inc.meta_score, "spec": inc.harness_spec,
        } if inc else None,
        "logs": runtime.get("logs", [])[-60:],
        "workers": manager.all_status(),
    }


@router.get("/engine/progress")
async def engine_progress(experiment_id: int | None = None):
    """外层 meta-score 步进序列 (可视化)."""
    exp_id = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        rows = (await s.scalars(
            select(OuterStep).where(OuterStep.experiment_id == exp_id).order_by(OuterStep.step_no))).all()
    return {
        "steps": [
            {"step": r.step_no, "candidate": r.candidate_score, "incumbent": r.incumbent_score,
             "accepted": r.accepted, "note": r.detail.get("note", "")}
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
             "note": v.proposal_note, "spec": v.harness_spec, "parent_id": v.parent_id}
            for v in versions
        ],
        "nodes": [
            {"id": n.id, "parent_id": n.parent_id, "miner_version_id": n.miner_version_id,
             "op": n.op, "expression": n.expression, "status": n.status,
             "public_score": n.public_score, "source": n.source, "task": n.task_name,
             "outer_step": n.outer_step_no, "created_at": str(n.created_at)}
            for n in nodes
        ],
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
            if ranking.get("available"):
                position += 1
                ranking["position"] = position
    return {
        "factors": payloads,
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "sort": sort,
    }


@router.get("/factors/ranking-diagnostics")
async def factor_ranking_diagnostics(experiment_id: int | None = None):
    """Validate the frozen V4 rank against fee-after Vault outcomes."""
    eid, cfg = await _experiment_context(experiment_id)
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
    if index is None:
        index = await asyncio.to_thread(build_similarity_index, items, threshold)
        _similarity_cache.clear()
        _similarity_cache[key] = index
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
    overrides = evaluation_config(market, cfg.get("evaluation_config"))
    overrides["multiple_testing_trials"] = max(
        int(overrides["multiple_testing_trials"]),
        actual_trials,
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
        })
        factor.research_meta = meta
        s.add(Trial(
            experiment_id=factor.experiment_id,
            expression_hash=(factor.fingerprint or {}).get("expr_hash", "audit"),
            layer="FULL_AUDIT_V4",
            task_name=factor.task_name,
            statistic={
                "grade": audit["eligibility"]["grade"],
                "stage": audit["eligibility"]["stage"],
                "live_rank_score": audit["ranking"].get("score"),
                "score_pre_vault": audit["ranking"].get("score_pre_vault"),
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
    }
    if req.status not in allowed:
        raise HTTPException(400, f"status 必须是 {allowed}")
    async with SessionLocal() as s:
        f = await s.get(Factor, fid)
        if not f:
            raise HTTPException(404)
        f.status = req.status
        await s.commit()
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
    panel_glob: str | None,
    universe_n: int,
    market: str,
) -> dict:
    import polars as pl
    df = PanelStore.get(panel_glob, market).ensure_loaded()
    fields = get_dsl_fields(market)
    target = df["trade_date"].max()
    merged = None
    for i, expression in enumerate(expressions):
        frame = parse(expression, fields).apply(df.lazy()).filter(
            (pl.col("trade_date") == target) & (pl.col("univ_rank") <= universe_n)
        ).select("ts_code", pl.col("factor").alias(f"f{i}")).collect()
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
            )
            results.append({"expression": expression, "direction": direction, **metrics})
        except ValueError as exc:
            results.append({"expression": expression, "direction": direction, "error": str(exc)})
    corr = await asyncio.to_thread(
        _latest_factor_correlation,
        expressions,
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
    }


class EvalReq(BaseModel):
    expression: str
    experiment_id: int | None = None
    universe_n: int = 500
    horizon: int = 5
    portfolio_mode: str | None = None
    direction: int = 1
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
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return metrics


# ---------- 回测 ----------

class BacktestReq(BaseModel):
    expression: str
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


@router.post("/backtest")
async def backtest(req: BacktestReq):
    eid, cfg = await _experiment_context(req.experiment_id)
    market = cfg.get("market", "us")
    err = validate(req.expression, get_dsl_fields(market))
    if err:
        raise HTTPException(400, f"表达式非法: {err}")
    mode = req.mode or cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE)
    panel_glob = req.panel_glob or cfg.get("panel_glob")
    resolved_eval = evaluation_config(market, cfg.get("evaluation_config"))
    borrow_cost = (
        req.borrow_cost_bps_annual
        if req.borrow_cost_bps_annual is not None
        else resolved_eval["borrow_cost_bps_annual"]
    )
    params = {
        **req.model_dump(),
        "mode": mode,
        "market": market,
        "panel_glob": panel_glob,
        "borrow_cost_bps_annual": borrow_cost,
        "protocol": "step_event_v1",
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
    try:
        result = await asyncio.to_thread(
            run_backtest,
            req.expression,
            req.universe_n,
            req.start,
            req.end,
            req.cost_bps,
            req.direction,
            mode,
            panel_glob,
            market,
            borrow_cost,
            req.top_fraction,
            initial_capital=req.initial_capital,
            rebalance_every=req.rebalance_every,
            slippage_bps=req.slippage_bps,
            max_volume_participation=req.max_volume_participation,
            fee_profile=req.fee_profile,
            artifact_dir=artifact_dir,
        )
    except Exception as exc:  # noqa: BLE001 - persist failed runs as audit evidence
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
            "artifacts",
        )
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
        raise HTTPException(
            500,
            "交割单完整性检查失败；失败账本已保留，请检查历史回测详情",
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
    engine_value = {**DEFAULT_ENGINE_CONFIG, **(eng.value if eng else {})}
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


@router.get("/experiments")
async def list_experiments():
    active_id = await get_active_experiment_id()
    async with SessionLocal() as s:
        rows = (await s.scalars(select(Experiment).order_by(Experiment.id))).all()
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
                "market": market,
                "portfolio_mode": portfolio_mode,
                "panel_glob": req.research_config.get("panel_glob") or default_panel_glob(market),
                "engine_mode": "v2",
                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                "evaluation_config": resolved_evaluation,
                "direction": direction,
            },
        )
        s.add(e)
        await s.commit()
        await s.refresh(e)
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
            e.research_config = merged
            material_keys = {
                "market",
                "portfolio_mode",
                "panel_glob",
                "direction",
                "evaluation_protocol",
                "evaluation_config",
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
        row = await s.get(Setting, "active_experiment")
        if row:
            row.value = {"id": eid}
        else:
            s.add(Setting(key="active_experiment", value={"id": eid}))
        await s.commit()
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


# ---------- 选股器 ----------

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
        **expression_profile(req.expression),
    }


class ScreenerReq(BaseModel):
    factors: list[dict] = Field(default_factory=list)  # [{"expression": "...", "weight": 1.0}, ...]
    expression: str | None = None  # 单个 DSL 直接选股
    panel_glob: str | None = None
    date: str | None = None  # YYYY-MM-DD, None=最新交易日
    universe_n: int = 500
    top_n: int = 50
    direction: str = "top"  # "top" | "bottom" | "both"


@router.post("/screener")
async def screener(req: ScreenerReq):
    """多因子选股: 单一 Polars 计划计算并缓存截面排名。"""
    _, cfg = await _experiment_context()
    panel_glob = req.panel_glob or cfg.get("panel_glob")
    market = cfg.get("market", "us")
    fields = get_dsl_fields(market)
    if req.direction not in {"top", "bottom", "both"}:
        raise HTTPException(400, "direction 必须是 top/bottom/both")
    if not 1 <= req.universe_n <= 10000:
        raise HTTPException(400, "universe_n 必须在 1 到 10000 之间")
    if not 1 <= req.top_n <= 500:
        raise HTTPException(400, "top_n 必须在 1 到 500 之间")
    factors = list(req.factors)
    if req.expression:
        factors = [{"expression": req.expression, "weight": 1.0}]
    if not factors:
        raise HTTPException(400, "至少需要一个因子")
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

    store = PanelStore.get(panel_glob, market)
    df = store.ensure_loaded()
    import datetime as _dt
    import bisect
    requested_date = None
    if req.date:
        try:
            requested_date = _dt.date.fromisoformat(req.date)
        except ValueError as exc:
            raise HTTPException(400, "date 必须是 YYYY-MM-DD") from exc
        index = bisect.bisect_right(store.trading_dates, requested_date) - 1
        if index < 0:
            raise HTTPException(400, "请求日期早于面板首个交易日")
        target_date = store.trading_dates[index]
    else:
        target_date = store.trading_dates[-1]
    try:
        screened = await asyncio.to_thread(
            screen_cross_section,
            df=df,
            trading_dates=store.trading_dates,
            panel_identity=(
                f"{market}:{panel_glob or 'default'}:{df.height}:"
                f"{store.trading_dates[-1]}"
            ),
            target_date=target_date,
            factors=factors,
            fields=fields,
            universe_n=req.universe_n,
            top_n=req.top_n,
            direction=req.direction,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return {
        "date": str(target_date),
        "requested_date": str(requested_date) if requested_date else None,
        "date_adjusted": bool(requested_date and target_date != requested_date),
        "universe_n": req.universe_n,
        "top_n": req.top_n,
        "factor_count": len(factors),
        "expression_mode": bool(req.expression),
        "market": market,
        "portfolio_mode": cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE),
        "direction": req.direction,
        **screened,
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
        "evaluation_protocol": cfg.get("evaluation_protocol", "legacy"),
        "evaluation_config": evaluation_config(market, cfg.get("evaluation_config")),
        "dsl_fields": get_dsl_fields(cfg.get("market")),
        "operators": OPERATORS_DOC,
    }


@router.get("/observability")
async def observability():
    """平台运行态、数据身份、任务 worker 和资源信息的统一只读快照。"""
    import os
    import time
    manager = EngineManager.get()
    _, cfg = await _experiment_context()
    market = cfg.get("market", "us")
    panel = PanelStore.get(cfg.get("panel_glob"), market)
    summary = await asyncio.to_thread(panel.summary, False)
    async with SessionLocal() as s:
        events = (await s.scalars(select(EngineEvent).order_by(EngineEvent.id.desc()).limit(100))).all()
    return {
        "service": {"port": int(os.environ.get("FF_PORT", "10010")), "pid": os.getpid(), "epoch": time.time()},
        "data": summary,
        "active_task": {
            "market": market,
            "portfolio_mode": cfg.get("portfolio_mode"),
            "direction": int(cfg.get("direction", 1)),
            "evaluation_protocol": cfg.get("evaluation_protocol", "legacy"),
        },
        "panel_cache": {
            "instances": len(PanelStore._instances),
            "loaded": sum(store.df is not None for store in PanelStore._instances.values()),
        },
        "screener_cache": SCREEN_CACHE.stats(),
        "backtest_artifacts": {
            "root": str(BACKTEST_ARTIFACT_ROOT),
            "runs": len(list(BACKTEST_ARTIFACT_ROOT.glob("*/manifest.json")))
            if BACKTEST_ARTIFACT_ROOT.exists() else 0,
        },
        "workers": manager.all_status(),
        "engine": {"worker_count": len(manager.workers), "running_count": sum(w.running for w in manager.workers.values())},
        "recent_events": [
            {"id": e.id, "experiment_id": e.experiment_id, "level": e.level,
             "message": e.message, "created_at": str(e.created_at), "payload": e.payload}
            for e in events
        ],
    }
