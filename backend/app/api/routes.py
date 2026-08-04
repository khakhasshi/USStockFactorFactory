import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from ..backtest.engine import run_backtest
from ..config import DEFAULT_ENGINE_CONFIG
from ..data.panel import PanelStore
from ..db import SessionLocal, get_active_experiment_id
from ..dsl.engine import OPERATORS_DOC, validate
from ..eval.harness import era_detail, evaluate
from ..models import Backtest, Experiment, Factor, MinerVersion, Node, OuterStep, Setting, Trial
from ..orchestrator import Engine

router = APIRouter(prefix="/api")


# ---------- 引擎 ----------

class EngineStartReq(BaseModel):
    mode: str = "v1"  # "v1" (A组兼容) 或 "v2" (B组: MinerTemplate + multi-seed + t-test)


@router.post("/engine/start")
async def engine_start(req: EngineStartReq | None = None):
    mode = req.mode if req else "v1"
    if mode not in ("v1", "v2"):
        raise HTTPException(400, "mode 必须为 v1 或 v2")
    return await Engine.get().start(mode)


@router.post("/engine/stop")
async def engine_stop():
    return await Engine.get().stop()


@router.get("/engine/status")
async def engine_status():
    eng = Engine.get()
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
    return {
        **eng.status,
        "experiment": {"id": exp_id, "name": exp.name if exp else "?",
                       "status": exp.status if exp else "?"},
        "counts": {"factors": n_factors, "nodes": n_nodes, "outer_steps": n_steps, "accepted": accepted},
        "incumbent": {
            "version_no": inc.version_no, "meta_score": inc.meta_score, "spec": inc.harness_spec,
        } if inc else None,
        "logs": list(eng.logbuf)[-60:],
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
async def list_factors(status: str | None = None, experiment_id: int | None = None):
    exp_id = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        q = select(Factor).where(Factor.experiment_id == exp_id).order_by(Factor.id.desc())
        if status:
            q = q.where(Factor.status == status)
        rows = (await s.scalars(q)).all()
    return {
        "factors": [
            {"id": f.id, "name": f.name, "expression": f.expression, "status": f.status,
             "task": f.task_name, "hypothesis": f.hypothesis,
             "public": f.public_metrics, "gate": f.gate_metrics,
             "created_at": str(f.created_at)}
            for f in rows
        ]
    }


@router.get("/factors/{fid}/detail")
async def factor_detail(fid: int):
    async with SessionLocal() as s:
        f = await s.get(Factor, fid)
        if not f:
            raise HTTPException(404)
    detail = await asyncio.to_thread(era_detail, f.expression, 500, 5)
    return {"factor": {"id": f.id, "name": f.name, "expression": f.expression,
                       "hypothesis": f.hypothesis, "public": f.public_metrics, "gate": f.gate_metrics},
            **detail}


class FactorStatusReq(BaseModel):
    status: str


@router.post("/factors/{fid}/status")
async def set_factor_status(fid: int, req: FactorStatusReq):
    allowed = {"public-leading", "library-admitted", "public-gate-pass", "paper", "retired"}
    if req.status not in allowed:
        raise HTTPException(400, f"status 必须是 {allowed}")
    async with SessionLocal() as s:
        f = await s.get(Factor, fid)
        if not f:
            raise HTTPException(404)
        f.status = req.status
        await s.commit()
    return {"ok": True}


class EvalReq(BaseModel):
    expression: str
    universe_n: int = 500
    horizon: int = 5


@router.post("/factors/evaluate")
async def manual_evaluate(req: EvalReq):
    err = validate(req.expression)
    if err:
        raise HTTPException(400, f"表达式非法: {err}")
    try:
        metrics = await asyncio.to_thread(evaluate, req.expression, req.universe_n, req.horizon)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return metrics


# ---------- 回测 ----------

class BacktestReq(BaseModel):
    expression: str
    universe_n: int = 500
    start: str = "2015-01-01"
    end: str = "2024-12-31"
    cost_bps: float = 15.0
    direction: int = 1
    mode: str = "long_short"


@router.post("/backtest")
async def backtest(req: BacktestReq):
    err = validate(req.expression)
    if err:
        raise HTTPException(400, f"表达式非法: {err}")
    try:
        result = await asyncio.to_thread(
            run_backtest, req.expression, req.universe_n, req.start, req.end,
            req.cost_bps, req.direction, req.mode,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    async with SessionLocal() as s:
        s.add(Backtest(params=req.model_dump(), result=result["stats"]))
        await s.commit()
    return result


@router.get("/backtests")
async def list_backtests():
    async with SessionLocal() as s:
        rows = (await s.scalars(select(Backtest).order_by(Backtest.id.desc()).limit(50))).all()
    return {"backtests": [
        {"id": b.id, "params": b.params, "stats": b.result, "created_at": str(b.created_at)} for b in rows
    ]}


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
    return {"llm_providers": masked, "engine_config": {**DEFAULT_ENGINE_CONFIG, **(eng.value if eng else {})}}


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


class ExperimentPatchReq(BaseModel):
    name: str | None = None
    description: str | None = None
    status: str | None = None  # open/archived


@router.get("/experiments")
async def list_experiments():
    active_id = await get_active_experiment_id()
    async with SessionLocal() as s:
        rows = (await s.scalars(select(Experiment).order_by(Experiment.id))).all()
        out = []
        for e in rows:
            n_factors = await s.scalar(
                select(func.count(Factor.id)).where(Factor.experiment_id == e.id))
            n_nodes = await s.scalar(
                select(func.count(Node.id)).where(Node.experiment_id == e.id))
            n_steps = await s.scalar(
                select(func.count(OuterStep.id)).where(OuterStep.experiment_id == e.id))
            out.append({
                "id": e.id, "name": e.name, "description": e.description, "status": e.status,
                "active": e.id == active_id, "created_at": str(e.created_at),
                "counts": {"factors": n_factors, "nodes": n_nodes, "outer_steps": n_steps},
            })
    return {"experiments": out, "active_id": active_id}


@router.post("/experiments")
async def create_experiment(req: ExperimentReq):
    if not req.name.strip():
        raise HTTPException(400, "名称不能为空")
    async with SessionLocal() as s:
        dup = await s.scalar(select(Experiment).where(Experiment.name == req.name.strip()))
        if dup:
            raise HTTPException(400, "同名实验已存在")
        e = Experiment(name=req.name.strip(), description=req.description, status="open")
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
        if req.status is not None:
            if req.status not in {"open", "archived"}:
                raise HTTPException(400, "status 必须是 open/archived")
            if req.status == "archived" and eid == await get_active_experiment_id() \
                    and Engine.get().running:
                raise HTTPException(400, "引擎运行中, 不能归档活动实验")
            e.status = req.status
        await s.commit()
    return {"ok": True}


@router.delete("/experiments/{eid}")
async def delete_experiment(eid: int):
    active_id = await get_active_experiment_id()
    if eid == active_id:
        raise HTTPException(400, "不能删除活动实验, 请先切换")
    async with SessionLocal() as s:
        e = await s.get(Experiment, eid)
        if not e:
            raise HTTPException(404)
        # 级联删除该实验全部产物 (nodes 先于 miner_versions, 避免外键阻塞)
        from sqlalchemy import delete as sqldelete
        await s.execute(sqldelete(Trial).where(Trial.experiment_id == eid))
        await s.execute(sqldelete(Factor).where(Factor.experiment_id == eid))
        await s.execute(sqldelete(Node).where(Node.experiment_id == eid))
        await s.execute(sqldelete(OuterStep).where(OuterStep.experiment_id == eid))
        await s.execute(sqldelete(MinerVersion).where(MinerVersion.experiment_id == eid))
        await s.delete(e)
        await s.commit()
    return {"ok": True}


@router.post("/experiments/{eid}/activate")
async def activate_experiment(eid: int):
    if Engine.get().running:
        raise HTTPException(400, "引擎运行中, 请先停止再切换实验")
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
    return {"ok": True}


# ---------- 元信息 ----------

@router.get("/meta")
async def meta():
    return {"panel": await asyncio.to_thread(PanelStore.get().summary), "operators": OPERATORS_DOC}
