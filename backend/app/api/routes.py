import asyncio

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import func, select

from ..backtest.engine import run_backtest
from ..config import DEFAULT_ENGINE_CONFIG, DEFAULT_PORTFOLIO_MODE, PANEL_GLOB, default_panel_glob, get_dsl_fields
from ..data.panel import PanelStore
from ..db import SessionLocal, get_active_experiment_id
from ..dsl.engine import OPERATORS_DOC, parse, validate
from ..eval.harness import era_detail, evaluate
from ..models import Backtest, EngineEvent, Experiment, Factor, MinerVersion, Node, OuterStep, Setting, Trial
from ..orchestrator import EngineManager

router = APIRouter(prefix="/api")


async def _experiment_context(experiment_id: int | None = None) -> tuple[int, dict]:
    eid = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        exp = await s.get(Experiment, eid)
    if not exp:
        raise HTTPException(404, "研究任务不存在")
    return eid, (exp.research_config or {})


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
    experiment_id: int | None = None,
    q: str | None = None,
    sort: str = "score",
    direction: str = "desc",
    limit: int = 500,
):
    exp_id = experiment_id or await get_active_experiment_id()
    async with SessionLocal() as s:
        query = select(Factor).where(Factor.experiment_id == exp_id).order_by(Factor.id.desc())
        if status:
            query = query.where(Factor.status == status)
        if q:
            needle = f"%{q}%"
            query = query.where((Factor.name.ilike(needle)) | (Factor.expression.ilike(needle)) | (Factor.hypothesis.ilike(needle)))
        query = query.limit(max(1, min(limit, 2000)))
        rows = (await s.scalars(query)).all()
    if sort == "score":
        rows.sort(key=lambda f: float((f.public_metrics or {}).get("score") or 0), reverse=direction != "asc")
    elif sort == "icir":
        rows.sort(key=lambda f: float((f.public_metrics or {}).get("icir") or 0), reverse=direction != "asc")
    return {
        "factors": [
        {"id": f.id, "name": f.name, "expression": f.expression, "status": f.status,
             "task": f.task_name, "hypothesis": f.hypothesis,
             "public": f.public_metrics, "gate": f.gate_metrics,
             "research_meta": f.research_meta or {},
             "evaluation_protocol": "long_only_v1" if (f.public_metrics or {}).get("portfolio_mode") == "long_only" else "legacy_unoriented",
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
        exp = await s.get(Experiment, f.experiment_id)
    cfg = (exp.research_config or {}) if exp else {}
    detail = await asyncio.to_thread(era_detail, f.expression, 500, 5, cfg.get("panel_glob"))
    return {"factor": {"id": f.id, "name": f.name, "expression": f.expression,
            "hypothesis": f.hypothesis, "public": f.public_metrics, "gate": f.gate_metrics,
            "research_meta": f.research_meta or {}, "experiment_id": f.experiment_id,
            "evaluation_protocol": "long_only_v1" if (f.public_metrics or {}).get("portfolio_mode") == "long_only" else "legacy_unoriented"},
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
        if req.direction in {-1, 1}:
            meta["direction"] = req.direction
        if req.decision:
            meta["decision"] = req.decision
        f.research_meta = meta
        await s.commit()
    return {"ok": True, "research_meta": meta}


class FactorCompareReq(BaseModel):
    factor_ids: list[int] = []
    expressions: list[str] = []
    experiment_id: int | None = None
    universe_n: int = 500
    horizon: int = 5
    portfolio_mode: str | None = None


def _latest_factor_correlation(expressions: list[str], panel_glob: str | None, universe_n: int) -> dict:
    import polars as pl
    df = PanelStore.get(panel_glob).ensure_loaded()
    target = df["trade_date"].max()
    merged = None
    for i, expression in enumerate(expressions):
        frame = parse(expression).apply(df.lazy()).filter(
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
        factors = (await s.scalars(select(Factor).where(Factor.id.in_(req.factor_ids)))).all() if req.factor_ids else []
    expressions = list(req.expressions) + [f.expression for f in factors]
    expressions = list(dict.fromkeys(expressions))[:12]
    if not expressions:
        raise HTTPException(400, "至少选择一个因子或表达式")
    mode = req.portfolio_mode or cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE)
    results = []
    for expression in expressions:
        try:
            metrics = await asyncio.to_thread(
                evaluate, expression, req.universe_n, req.horizon, mode, 1, cfg.get("panel_glob"), cfg.get("cost_bps", 15)
            )
            results.append({"expression": expression, **metrics})
        except ValueError as exc:
            results.append({"expression": expression, "error": str(exc)})
    corr = await asyncio.to_thread(_latest_factor_correlation, expressions, cfg.get("panel_glob"), req.universe_n)
    return {"experiment_id": eid, "portfolio_mode": mode, "results": results, "correlation": corr}


class EvalReq(BaseModel):
    expression: str
    universe_n: int = 500
    horizon: int = 5
    portfolio_mode: str | None = None
    direction: int = 1
    panel_glob: str | None = None
    cost_bps: float = 15.0


@router.post("/factors/evaluate")
async def manual_evaluate(req: EvalReq):
    err = validate(req.expression)
    if err:
        raise HTTPException(400, f"表达式非法: {err}")
    try:
        _, cfg = await _experiment_context()
        mode = req.portfolio_mode or cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE)
        panel_glob = req.panel_glob or cfg.get("panel_glob")
        metrics = await asyncio.to_thread(
            evaluate, req.expression, req.universe_n, req.horizon,
            mode, req.direction, panel_glob, req.cost_bps,
        )
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
    mode: str | None = None
    panel_glob: str | None = None


@router.post("/backtest")
async def backtest(req: BacktestReq):
    err = validate(req.expression)
    if err:
        raise HTTPException(400, f"表达式非法: {err}")
    try:
        _, cfg = await _experiment_context()
        mode = req.mode or cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE)
        panel_glob = req.panel_glob or cfg.get("panel_glob")
        result = await asyncio.to_thread(
            run_backtest, req.expression, req.universe_n, req.start, req.end,
            req.cost_bps, req.direction, mode, panel_glob,
        )
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    async with SessionLocal() as s:
        s.add(Backtest(
            experiment_id=await get_active_experiment_id(),
            params={**req.model_dump(), "mode": mode, "panel_glob": panel_glob}, result=result["stats"],
        ))
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
    research_config: dict = {}


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
                "research_config": e.research_config or {},
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
        e = Experiment(
            name=req.name.strip(), description=req.description, status="open",
            research_config={
                "market": req.research_config.get("market", "us"),
                "portfolio_mode": req.research_config.get("portfolio_mode", "long_short"),
                "panel_glob": req.research_config.get("panel_glob") or default_panel_glob(req.research_config.get("market", "us")),
                "engine_mode": req.research_config.get("engine_mode", "v2"),
                **req.research_config,
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
            e.research_config = {**(e.research_config or {}), **req.research_config}
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
    return {"ok": True}


# ---------- 选股器 ----------

class ScreenerReq(BaseModel):
    factors: list[dict] = []  # [{"expression": "...", "weight": 1.0}, ...]
    expression: str | None = None  # 单个 DSL 直接选股
    panel_glob: str | None = None
    date: str | None = None  # YYYY-MM-DD, None=最新交易日
    universe_n: int = 500
    top_n: int = 50
    direction: str = "top"  # "top" | "bottom" | "both"


@router.post("/screener")
async def screener(req: ScreenerReq):
    """多因子选股: 按加权综合排名返回股票列表。"""
    import polars as pl
    _, cfg = await _experiment_context()
    panel_glob = req.panel_glob or cfg.get("panel_glob")
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
        if validate(factor["expression"]):
            raise HTTPException(400, f"表达式非法: {validate(factor['expression'])}")
        if float(factor.get("weight", 1.0)) <= 0:
            raise HTTPException(400, "因子权重必须大于 0")

    df = PanelStore.get(panel_glob).ensure_loaded()
    import datetime as _dt
    target_date = _dt.date.fromisoformat(req.date) if req.date else df["trade_date"].max()

    # 计算每个因子的截面排名
    rank_cols = []
    for i, f in enumerate(factors):
        expr_str = f["expression"]
        pipe = parse(expr_str)
        work = pipe.apply(df.lazy()).select(
            pl.col("trade_date"), pl.col("ts_code"), pl.col("name"),
            pl.col("univ_rank"), pl.col("factor").alias(f"f{i}")
        )
        # 截面 rank (1=最好)
        ranked = (
            work.filter(pl.col("trade_date") == target_date)
            .filter(pl.col("univ_rank") <= req.universe_n)
            .filter(pl.col(f"f{i}").is_finite())
            .with_columns(pl.col(f"f{i}").rank(descending=True).alias(f"r{i}"))
            # 只在第一列保留名称，后续 join 只携带代码和排名，避免 name_right 重复。
            .select("ts_code", *( ["name"] if i == 0 else [] ), f"r{i}")
        )
        rank_cols.append(ranked.collect())

    # 合并 & 加权
    merged = rank_cols[0]
    for rc in rank_cols[1:]:
        merged = merged.join(rc, on="ts_code", how="inner")

    # 加权综合分
    total_weight = sum(float(f.get("weight", 1.0)) for f in factors)
    score_expr = pl.lit(0.0)
    for i, f in enumerate(factors):
        w = float(f.get("weight", 1.0)) / total_weight
        score_expr = score_expr + pl.col(f"r{i}") * w

    result = merged.with_columns(score_expr.alias("score"))

    # 排序
    if req.direction == "bottom":
        result = result.sort("score")
    elif req.direction == "both":
        result = result.with_columns(
            pl.when(pl.col("score") > pl.col("score").median())
            .then(pl.col("score"))
            .otherwise(-pl.col("score"))
            .alias("score")
        )
        result = result.sort("score", descending=True)
    else:
        result = result.sort("score", descending=True)

    result = result.head(req.top_n)

    return {
        "date": str(target_date),
        "universe_n": req.universe_n,
        "top_n": req.top_n,
        "factor_count": len(factors),
        "expression_mode": bool(req.expression),
        "market": cfg.get("market"),
        "portfolio_mode": cfg.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE),
        "direction": req.direction,
        "stocks": [
            {"rank": j + 1, "ts_code": r["ts_code"], "name": r.get("name", ""),
             "score": round(float(r["score"]), 2)}
            for j, r in enumerate(result.iter_rows(named=True))
        ],
    }


# ---------- 元信息 ----------

@router.get("/meta")
async def meta(experiment_id: int | None = None):
    _, cfg = await _experiment_context(experiment_id)
    panel = PanelStore.get(cfg.get("panel_glob"))
    summary = await asyncio.to_thread(panel.summary)
    if panel.df is not None:
        summary["loaded_columns"] = panel.df.columns
    return {
        "panel": summary,
        "market": cfg.get("market"),
        "portfolio_mode": cfg.get("portfolio_mode"),
        "dsl_fields": get_dsl_fields(cfg.get("market")),
        "operators": OPERATORS_DOC,
    }


@router.get("/observability")
async def observability():
    """平台运行态、数据身份、任务 worker 和资源信息的统一只读快照。"""
    import os
    import time
    manager = EngineManager.get()
    panel = PanelStore.get()
    summary = await asyncio.to_thread(panel.summary)
    async with SessionLocal() as s:
        events = (await s.scalars(select(EngineEvent).order_by(EngineEvent.id.desc()).limit(100))).all()
    return {
        "service": {"port": int(os.environ.get("FF_PORT", "10010")), "pid": os.getpid(), "epoch": time.time()},
        "data": summary,
        "workers": manager.all_status(),
        "engine": {"worker_count": len(manager.workers), "running_count": sum(w.running for w in manager.workers.values())},
        "recent_events": [
            {"id": e.id, "experiment_id": e.experiment_id, "level": e.level,
             "message": e.message, "created_at": str(e.created_at), "payload": e.payload}
            for e in events
        ],
    }
