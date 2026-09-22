"""Manual trading plans. No broker, simulated fills, or automatic order submission.

The event engine remains the sizing/lifecycle authority. Actual fills are replayed
instead of its execution model; unsupported corporate actions fail closed.
"""
from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import math
from dataclasses import fields, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import exchange_calendars as xcals
import polars as pl
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import Integer, String, Text, JSON, UniqueConstraint, select, text
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, SessionLocal
from .models import Backtest
from .config import SERVICE_MARKET, SERVICE_INSTANCE, BACKTEST_ARTIFACT_ROOT
from .backtest.engine import (EventBacktestConfig, StepEventBacktester,
    _prepare_backtest_frame, _weighted_sleeve_specs)
from .backtest.risk import ExitPolicyConfig, PositionState, risk_levels

router = APIRouter(prefix="/api/trade-plans", tags=["manual-trade-plans"])
log = logging.getLogger(__name__)


class PlanInstance(Base):
    __tablename__ = "manual_plan_instances"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(20), default="paused")
    frozen: Mapped[dict] = mapped_column(JSON)
    revision: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[dict] = mapped_column(JSON, default=dict)


class PlanBatch(Base):
    __tablename__ = "manual_plan_batches"
    __table_args__ = (UniqueConstraint("instance_id", "execution_date"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance_id: Mapped[int] = mapped_column(Integer, index=True)
    execution_date: Mapped[str] = mapped_column(String(10))
    snapshot: Mapped[dict] = mapped_column(JSON)
    reconciliation: Mapped[dict] = mapped_column(JSON, default=dict)


class ManualEntry(Base):
    __tablename__ = "manual_plan_entries"
    __table_args__ = (UniqueConstraint("instance_id", "external_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    instance_id: Mapped[int] = mapped_column(Integer, index=True)
    external_id: Mapped[str] = mapped_column(String(128))
    payload: Mapped[dict] = mapped_column(JSON)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        default=str, allow_nan=False, separators=(",", ":")).encode()).hexdigest()


def decision_code_hash():
    root=Path(__file__).parent
    return digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ("trade_plans.py","backtest/engine.py","backtest/decision.py","backtest/risk.py","dsl/engine.py")})


def calendar(market, first, last):
    # Never substitute weekdays if the calendar package has no holiday coverage.
    return xcals.get_calendar("XNYS" if market == "us" else "XSHG",
        start=str(first - timedelta(days=20)), end=str(last + timedelta(days=20)))


def schedule(market, first, execution, period):
    cal = calendar(market, min(first, execution), max(first, execution))
    if not cal.is_session(str(first)) or not cal.is_session(str(execution)):
        raise ValueError("首次开仓日和执行日必须是该市场交易日")
    if execution < first:
        raise ValueError("执行日不能早于首次开仓日")
    previous = cal.previous_session(str(execution)).date()
    offset = len(cal.sessions_in_range(str(first), str(execution))) - 1
    return cal, previous, offset % period == 0


def engine_config(frozen, spec):
    cfg = frozen["config"]
    allowed = {f.name for f in fields(EventBacktestConfig)}
    args = {k: v for k, v in cfg.items() if k in allowed and k != "exit_policy"}
    ep = {f.name for f in fields(ExitPolicyConfig)}
    args["exit_policy"] = ExitPolicyConfig(**{k:v for k,v in cfg.get("exit_policy", {}).items() if k in ep})
    args.update(initial_capital=frozen["capital"] * spec["normalized_weight"],
        direction=spec["direction"], liquidate_at_end=False)
    config = EventBacktestConfig(**args)
    config.validate()
    return config


class ManualReplay(StepEventBacktester):
    """Replay confirmed executions, not hypothetical backtest executions.

Daily OHLC provides retrospective risk checks only. All actual fees/financing
must be recorded by the operator; no estimated costs are booked as actual.
"""
    def __init__(self, config, entries):
        super().__init__(replace(config, borrow_cost_bps_annual=0,
            margin_interest_bps_annual=0, liquidate_at_end=False))
        self.entries = entries
        self.risk_notices = []

    def _execute_order(self, trade_date, order, market, **kwargs):
        remaining = abs(self.positions.get(order["symbol"], 0) - order["target_quantity"])
        self.risk_notices.append({"date": str(trade_date), "symbol": order["symbol"],
            "reason": order["reason"], "classification": "retrospective_not_a_fill"})
        return {"filled": 0., "unfilled": remaining, "status": "manual_confirmation_required"}

    def _apply_corporate_actions(self, trade_date, market):
        for symbol in self.positions:
            row = market.get(symbol)
            old = self.last_adjustment.get(symbol)
            current = row.get("adjustment_factor") if row else None
            if old and current and abs(current / old - 1) > 1e-8:
                raise ValueError(f"CORPORATE_ACTION_REVIEW: {trade_date} {symbol} 复权变化，不能用回测的合成再投资数量替代券商持仓")

    def _handle_stale_positions(self, trade_date, market):
        for symbol in self.positions:
            if not market.get(symbol) or not valid_price(market[symbol].get("raw_close")):
                raise ValueError(f"MISSING_POSITION_QUOTE: {trade_date} {symbol}")

    def _execute_pending(self, trade_date, market, next_trade_date):
        # Risk orders are suggestions. Never consume them as confirmed fills.
        self.pending_orders = [o for o in self.pending_orders
            if self.positions.get(o["symbol"], 0)]
        self._manual_fills_today = set()
        for entry in self.entries:
            if entry["date"] != str(trade_date):
                continue
            if entry["kind"] == "cash":
                self.cash += entry["amount"]
                continue
            symbol, qty, price = entry["symbol"], entry["quantity"], entry["price"]
            self._manual_fills_today.add(symbol)
            old = self.positions.get(symbol, 0.)
            if self.config.mode == "long_only" and old + qty < -1e-8:
                raise ValueError(f"持仓不足: {symbol}")
            if self.config.market == "ashare" and qty < 0:
                settled = self.position_states[symbol].settled_quantity if symbol in self.position_states else 0
                if -qty > settled + 1e-8:
                    raise ValueError(f"A股 T+1 可卖数量不足: {symbol}")
            row = market.get(symbol)
            if not row:
                raise ValueError(f"成交标的行情缺失: {symbol}")
            self._update_position_state_for_fill(symbol=symbol, signed_quantity=qty,
                fill_price=price, trade_date=trade_date, atr_pct=entry.get("atr_pct"),
                fees=entry["fees"], reason="manual_confirmed")
            self.cash -= qty * price + entry["fees"]
            self._cumulative_fees += entry["fees"]
            self.positions[symbol] = old + qty
            if abs(self.positions[symbol]) < 1e-8:
                self.positions.pop(symbol)
            self._cumulative_traded_notional += abs(qty * price)
            self._fill_seq += 1
            if self.config.resolved_account_type == "cash" and self.cash < -1e-6:
                raise ValueError("现金账户成交回填导致负现金，请先核对本金/费用/成交")

    def _evaluate_risk_stage(self, *, trade_date, market, stage, **kwargs):
        # Daily OHLC cannot locate extrema relative to an actual intraday fill.
        # Don't manufacture an exit before a purchase. Those days require the
        # operator's real executions; use completed closes for new anchors.
        if stage == "intraday":
            market = {k:v for k,v in market.items() if k not in getattr(self,"_manual_fills_today",set())}
        return super()._evaluate_risk_stage(trade_date=trade_date,market=market,stage=stage,**kwargs)

    def _update_position_bars(self, market):
        market = {k:dict(v) for k,v in market.items()}
        for symbol in getattr(self,"_manual_fills_today",set()):
            if symbol in market:
                market[symbol]["raw_high"] = market[symbol]["raw_close"]
                market[symbol]["raw_low"] = market[symbol]["raw_close"]
        super()._update_position_bars(market)


def valid_price(v):
    return v is not None and math.isfinite(float(v)) and float(v) > 0


def build_plan(frozen, entries, execution, *, now=None, loader=_prepare_backtest_frame, archive=False, reconciliation_only=False):
    now = now or datetime.now(timezone.utc)
    if frozen.get("decision_code_hash") and frozen["decision_code_hash"] != decision_code_hash():
        raise ValueError("CODE_CHANGED: 冻结决策代码与当前版本不同，请审核后创建新版本实例")
    first = date.fromisoformat(frozen["first_execution"])
    market = frozen["config"]["market"]
    cal, signal, rebalance = schedule(market, first, execution, frozen["config"]["rebalance_every"])
    close = cal.session_close(str(signal)).to_pydatetime()
    if now < close + timedelta(minutes=30):
        raise ValueError("DATA_NOT_READY: 信号日尚未收盘或未到收盘后30分钟")
    if not reconciliation_only and now >= cal.session_open(str(execution)).to_pydatetime():
        raise ValueError("EXPIRED: 执行日已开盘，不事后补造可执行计划")
    start = cal.previous_session(str(first)).date()
    outputs, daily_totals, inputs = [], {}, []
    generation_identity = None
    for spec in frozen["sleeves"]:
        cfg = engine_config(frozen, spec)
        frame, store = loader(expression=spec["expression"], universe_n=cfg.universe_n,
            start=str(start), end=str(signal), panel_glob=frozen.get("panel_glob"),
            market=market, atr_period=cfg.exit_policy.atr_period, minimum_sessions=1)
        if store is not None:
            _, _, identity, generation = store.read_snapshot()
            stamp = (identity, generation)
            if generation_identity is not None and generation_identity != stamp:
                raise ValueError("DATA_CHANGED: 袖套计算期间行情版本改变，请重试")
            generation_identity = stamp
        actual = frame["trade_date"].unique().sort().to_list()
        expected = [v.date() for v in cal.sessions_in_range(str(start), str(signal))]
        if actual != expected:
            raise ValueError(f"DATA_GAP: 需要 {start} 至 {signal} 完整交易日，实际末日 {actual[-1] if actual else '无'}")
        own = [e for e in entries if e["sleeve_id"] == spec["factor_id"]]
        runner = ManualReplay(cfg, sorted(own, key=lambda e:(e["date"], e["sequence"])))
        # Include every held security, including names outside today's Top-N.
        for day in actual:
            rows = frame.filter(pl.col("trade_date") == day).to_dicts()
            rows_by = {r["ts_code"]: r for r in rows}
            runner.step(trade_date=day, rows=rows, next_trade_date=cal.next_session(str(day)).date(), rebalance=False)
            daily_totals[str(day)] = daily_totals.get(str(day), 0) + runner.daily[-1]["close_nlv"]
        rows_by = {r["ts_code"]:r for r in rows}
        eligible = [r for r in rows if int(r.get("univ_rank") or 10**9) <= cfg.universe_n and valid_price(r.get("raw_close")) and r.get("factor") is not None and math.isfinite(r["factor"])]
        if not eligible:
            raise ValueError("INSUFFICIENT_DATA: 无有效候选，不能把空信号当作清仓指令")
        close_nlv = runner.daily[-1]["close_nlv"]
        if close_nlv <= 0:
            raise ValueError("账户净值非正")
        if rebalance and not runner._portfolio_risk_active and runner._risk_cooldown_remaining == 0:
            runner._create_orders(signal_date=signal, execute_date=execution, candidates=rows, close_nlv=close_nlv)
        target = dict(runner.positions)
        reasons = {}
        for order in runner.pending_orders:
            if order["reason"].startswith("factor_"):
                target[order["symbol"]] = order["target_quantity"]
                reasons[order["symbol"]] = order["reason"]
        # Unconfirmed risk exits always take priority over new factor orders.
        for order in runner.pending_orders:
            if not order["reason"].startswith("factor_") and runner.positions.get(order["symbol"], 0):
                target[order["symbol"]] = order["target_quantity"]
                reasons[order["symbol"]] = order["reason"]
        items = []
        for symbol in sorted(set(runner.positions) | set(target)):
            row = rows_by.get(symbol)
            if not row or not valid_price(row.get("raw_close")):
                raise ValueError("目标或持仓缺少收盘价格: " + symbol)
            current, wanted = runner.positions.get(symbol, 0), target.get(symbol, 0)
            state = runner.position_states.get(symbol)
            stop, take, mechanisms = risk_levels(state, cfg.exit_policy) if state else (None,None,[])
            if state and cfg.exit_policy.time_stop_sessions and state.holding_sessions >= cfg.exit_policy.time_stop_sessions:
                wanted = 0; reasons[symbol] = "risk_time_stop"
            delta = wanted - current
            reference = row["raw_close"]
            if not state and wanted:
                atr=reference*float(row.get("_atr_pct") or 0)
                provisional=PositionState(symbol=symbol,quantity=wanted,avg_entry_price=reference,
                    entry_date=str(execution),entry_session_index=0,highest_price=reference,
                    lowest_price=reference,initial_atr=atr or None,current_atr=atr or None)
                stop,take,mechanisms=risk_levels(provisional,cfg.exit_policy)
            capacity = float(row.get("_adv20_prev") or 0) * cfg.max_volume_participation
            action = "持有" if not delta else "新开" if not current else "平仓" if not wanted else "反向" if current*wanted < 0 else "加仓" if abs(wanted)>abs(current) else "减仓"
            items.append({"sleeve_id":spec["factor_id"], "symbol":symbol,
                "name":row.get("name",symbol), "action":action, "current_quantity":current,
                "target_quantity":wanted, "delta_quantity":delta, "reference_price":reference,
                "current_weight":current*reference/close_nlv, "target_weight":wanted*reference/close_nlv,
                "estimated_amount":abs(delta)*reference, "reason":reasons.get(symbol,"持仓延续"),
                "adv_capacity":capacity, "capacity_warning":abs(delta)>capacity,
                "atr_pct":row.get("_atr_pct"), "stop_price":stop, "take_profit_price":take,
                "risk_levels_provisional":not bool(state),
                "holding_sessions":state.holding_sessions if state else 0, "risk_mechanisms":mechanisms,
                "short_requires_borrow_confirmation":wanted<0})
        outputs.append({"sleeve_id":spec["factor_id"],"name":spec["name"],
            "cash":runner.cash,"nlv":close_nlv,"weight":spec["normalized_weight"],"items":items,
            "risk_active":runner._portfolio_risk_active,"cooldown":runner._risk_cooldown_remaining,
            "risk_notices":runner.risk_notices[-100:],"fees":runner._cumulative_fees})
        # Hash exact float64 inputs consumed, not rounded screener display fields.
        fingerprint = {"sleeve_id":spec["factor_id"], "rows_sha256":hashlib.sha256(frame.write_json().encode()).hexdigest(), "rows":frame.height,
            "sessions":{str(part["trade_date"][0]):hashlib.sha256(part.write_json().encode()).hexdigest() for part in frame.partition_by("trade_date",maintain_order=True)}}
        if archive:
            root=Path(BACKTEST_ARTIFACT_ROOT)/"manual-plan-inputs"
            root.mkdir(parents=True,exist_ok=True)
            path=root/(fingerprint["rows_sha256"]+".parquet")
            if not path.exists():
                # Content-addressed inputs, never overwrite a published object.
                with path.open("xb") as handle:frame.write_parquet(handle)
            fingerprint["artifact_path"]=str(path)
        inputs.append(fingerprint)
    total = sum(s["nlv"] for s in outputs)
    for sleeve in outputs:
        for item in sleeve["items"]:
            item["portfolio_target_weight"] = item["target_weight"] * sleeve["nlv"] / total
    result = {"protocol":"manual_trade_plan_v1","signal_date":str(signal),
        "execution_date":str(execution),"rebalance":rebalance,"source_hash":frozen["source_hash"],
        "scheduled_open":cal.session_open(str(execution)).isoformat(),
        "generated_at":now.isoformat(),"expires_at":cal.session_close(str(execution)).isoformat(),
        "execution_window":"next_session_open_manual_review", "sleeves":outputs,
        "nlv":total,"cash":sum(s["cash"] for s in outputs),"input_fingerprints":inputs,
        "panel_version":generation_identity,"calendar_version":xcals.__version__,
        "curve":[{"date":d,"nlv":n} for d,n in sorted(daily_totals.items())],
        "disclosure":"手工指导，未提交订单。数量按昨日收盘估算；开盘需复核跳空、可卖量、借券、资金和成交容量。新开仓止损止盈为参考价估算，成交后需按实际成本重算。日线风控为盘后检测，不是盘中通知。成交当日无法确定高低点先后，仅用收盘更新移动锚点。实际费用仅来自回填，未填费用不代表免收。"}
    result["snapshot_hash"] = digest(result)
    return result


class StrictReq(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CreateReq(StrictReq):
    backtest_id: int = Field(gt=0)
    name: str = Field(min_length=1,max_length=128)
    first_execution: date
    capital: float = Field(gt=0,le=1e12)


class FillReq(StrictReq):
    external_id: str = Field(min_length=1,max_length=128)
    sleeve_id: str
    date: date
    kind: str = "fill"
    symbol: str = ""
    quantity: float = 0
    price: float = 0
    fees: float = Field(default=0,ge=0)
    amount: float = 0
    note: str = Field(default="",max_length=1000)


class GenerateReq(StrictReq):
    execution_date: date


class StatusReq(StrictReq):
    status: str


class ReconcileReq(StrictReq):
    note: str = Field(min_length=1,max_length=1000)


class CsvReq(StrictReq):
    csv_text: str = Field(min_length=1,max_length=500000)


async def initialize():
    """Append-only source evidence; reconciliation is separate mutable metadata."""
    async with SessionLocal() as s:
        await s.execute(text("""CREATE OR REPLACE FUNCTION guard_manual_plan_evidence() RETURNS trigger AS $$
        BEGIN
          IF TG_OP='DELETE' THEN RAISE EXCEPTION 'manual plan history cannot be deleted'; END IF;
          IF TG_TABLE_NAME='manual_plan_entries' THEN RAISE EXCEPTION 'confirmed entries are append only'; END IF;
          IF TG_TABLE_NAME='manual_plan_batches' THEN
            IF OLD.snapshot::jsonb IS DISTINCT FROM NEW.snapshot::jsonb OR OLD.instance_id<>NEW.instance_id OR OLD.execution_date<>NEW.execution_date THEN RAISE EXCEPTION 'plan snapshot is immutable'; END IF;
          END IF;
          IF TG_TABLE_NAME='manual_plan_instances' THEN
            IF OLD.frozen::jsonb IS DISTINCT FROM NEW.frozen::jsonb THEN RAISE EXCEPTION 'strategy definition is immutable'; END IF;
          END IF;
          RETURN NEW;
        END; $$ LANGUAGE plpgsql"""))
        for table in ("manual_plan_instances","manual_plan_batches","manual_plan_entries"):
            await s.execute(text(f"DROP TRIGGER IF EXISTS guard_evidence ON {table}"))
            await s.execute(text(f"CREATE TRIGGER guard_evidence BEFORE UPDATE OR DELETE ON {table} FOR EACH ROW EXECUTE FUNCTION guard_manual_plan_evidence()"))
        # The application is intentionally a single worker; recover abandoned jobs.
        for p in (await s.scalars(select(PlanInstance))).all():
            if p.progress.get("state")=="running":p.progress={"state":"blocked","message":"服务重启中断计算，可安全重试；未生成成交"}
        await s.commit()


def view_instance(p):
    return {"id":p.id,"name":p.name,"status":p.status,"frozen":p.frozen,
        "revision":p.revision,"progress":p.progress}


@router.get("")
async def listing():
    async with SessionLocal() as s:
        return [view_instance(p) for p in (await s.scalars(select(PlanInstance).order_by(PlanInstance.id.desc()))).all()]


@router.post("")
async def create(req: CreateReq):
    async with SessionLocal() as s:
        bt = await s.get(Backtest, req.backtest_id)
        if not bt: raise HTTPException(404,"当前服务找不到回测ID，请确认来源端口")
        result = bt.result or {}
        cfg = result.get("config",{})
        if bt.status != "done" or not result.get("integrity",{}).get("all_pass"):
            raise HTTPException(400,"必须选择已完成且账本PASS的回测")
        if result.get("protocol") != "step_event_v2_weighted_sleeves_v1":
            raise HTTPException(400,"仅支持独立资金袖套事件回测")
        if SERVICE_MARKET and cfg.get("market") != SERVICE_MARKET:
            raise HTTPException(400,"回测与当前服务市场不一致")
        try:
            cal, _, _ = schedule(cfg["market"],req.first_execution,req.first_execution,cfg["rebalance_every"])
            if cal.session_open(str(req.first_execution)).to_pydatetime() <= datetime.now(timezone.utc):
                raise ValueError("首次开仓日必须尚未开盘；不允许倒填启动日伪造前向记录")
            specs = _weighted_sleeve_specs(bt.params.get("factors") or [])
            frozen = {"source_system":SERVICE_INSTANCE,"backtest_id":bt.id,
                "config":cfg,"source_params":bt.params,"source_hash":digest({"params":bt.params,"result":result}),
                "decision_code_hash":decision_code_hash(),
                "created_at":datetime.now(timezone.utc).isoformat(),"capital":req.capital,
                "first_execution":str(req.first_execution),"sleeves":specs,"panel_glob":bt.params.get("panel_glob"),
                "actual_backtest_start":result["curve"]["dates"][0],"actual_backtest_end":result["curve"]["dates"][-1]}
            for spec in specs: engine_config(frozen,spec)
        except (ValueError,KeyError) as exc: raise HTTPException(400,str(exc)) from exc
        p=PlanInstance(name=req.name,status="paused",frozen=frozen,revision=0,progress={})
        s.add(p); await s.commit(); return view_instance(p)


@router.get("/{instance_id}")
async def detail(instance_id:int):
    async with SessionLocal() as s:
        p=await s.get(PlanInstance,instance_id)
        if not p: raise HTTPException(404,"实例不存在")
        batches=(await s.scalars(select(PlanBatch).where(PlanBatch.instance_id==instance_id).order_by(PlanBatch.id.desc()))).all()
        entries=(await s.scalars(select(ManualEntry).where(ManualEntry.instance_id==instance_id).order_by(ManualEntry.id))).all()
        return {**view_instance(p),"batches":[{"id":b.id,"snapshot":b.snapshot,"reconciliation":b.reconciliation} for b in batches],"entries":[{"id":e.id,**e.payload} for e in entries]}


@router.post("/{instance_id}/status")
async def set_status(instance_id:int,req:StatusReq):
    if req.status not in {"active","paused","archived"}:raise HTTPException(400,"状态非法")
    async with SessionLocal() as s:
        p=await s.scalar(select(PlanInstance).where(PlanInstance.id==instance_id).with_for_update())
        if not p:raise HTTPException(404,"实例不存在")
        if p.status=="archived":raise HTTPException(409,"归档实例只读")
        p.status=req.status; p.revision+=1
        await s.commit(); return view_instance(p)


async def generate(instance_id,execution):
    async with SessionLocal() as s:
        p=await s.scalar(select(PlanInstance).where(PlanInstance.id==instance_id).with_for_update())
        if not p:raise HTTPException(404,"实例不存在")
        existing=await s.scalar(select(PlanBatch).where(PlanBatch.instance_id==instance_id,PlanBatch.execution_date==str(execution)))
        if existing:return {"id":existing.id,"snapshot":existing.snapshot,"reconciliation":existing.reconciliation}
        if p.status=="archived":raise HTTPException(409,"实例已归档")
        old_batches=(await s.scalars(select(PlanBatch).where(PlanBatch.instance_id==instance_id))).all()
        for old in old_batches:
            if old.execution_date < str(execution) and not old.reconciliation:
                raise HTTPException(409,"先回填并确认此前计划的实际成交/未成交，禁止假定已执行")
            if old.execution_date > str(execution):raise HTTPException(409,"不能回填历史计划")
        rev=p.revision; frozen=p.frozen
        if p.progress.get("state")=="running" and datetime.now(timezone.utc).timestamp()-p.progress.get("started",0)<3600:
            raise HTTPException(409,"计划正在计算")
        entries=[{**e.payload,"sequence":e.id} for e in (await s.scalars(select(ManualEntry).where(ManualEntry.instance_id==instance_id))).all()]
        p.progress={"state":"running","message":"冻结行情、逐袖套回放成交并生成目标","started":datetime.now(timezone.utc).timestamp()}
        await s.commit()
    try:
        result=await asyncio.to_thread(build_plan,frozen,entries,execution,archive=True)
        current_inputs={v["sleeve_id"]:v["sessions"] for v in result["input_fingerprints"]}
        for old in old_batches:
            for prior in old.snapshot["input_fingerprints"]:
                if any(current_inputs[prior["sleeve_id"]].get(day)!=sha for day,sha in prior["sessions"].items()):
                    raise ValueError("DATA_REVISION: 历史输入被修订，停止生成，需人工审核而不能静默重算已发布历史")
        async with SessionLocal() as s:
            p=await s.scalar(select(PlanInstance).where(PlanInstance.id==instance_id).with_for_update())
            if p.revision!=rev:raise ValueError("计算期间实例状态/成交发生变化，请重新生成")
            b=PlanBatch(instance_id=instance_id,execution_date=str(execution),snapshot=result,reconciliation={})
            s.add(b);p.progress={"state":"done","message":"计划已冻结归档","execution_date":str(execution)}
            await s.commit();return {"id":b.id,"snapshot":result,"reconciliation":{}}
    except Exception as exc:
        async with SessionLocal() as s:
            p=await s.get(PlanInstance,instance_id)
            p.progress={"state":"blocked","message":str(exc)[:1000]};await s.commit()
        raise HTTPException(409,str(exc)) from exc


@router.post("/{instance_id}/generate")
async def generate_route(instance_id:int,req:GenerateReq):
    return await generate(instance_id,req.execution_date)


@router.post("/{instance_id}/entries")
async def add_entry(instance_id:int,req:FillReq):
    async with SessionLocal() as s:
        p=await s.scalar(select(PlanInstance).where(PlanInstance.id==instance_id).with_for_update())
        if not p:raise HTTPException(404,"实例不存在")
        if p.status=="archived":raise HTTPException(409,"实例已归档")
        payload=req.model_dump(mode="json")
        old=await s.scalar(select(ManualEntry).where(ManualEntry.instance_id==instance_id,ManualEntry.external_id==req.external_id))
        if old:
            if any(old.payload.get(k)!=v for k,v in payload.items()):raise HTTPException(409,"相同成交编号内容冲突")
            return {"id":old.id,"duplicate":True}
        if req.sleeve_id not in {v["factor_id"] for v in p.frozen["sleeves"]}:raise HTTPException(400,"袖套不存在")
        if req.kind not in {"fill","cash"}:raise HTTPException(400,"只支持成交或费用现金调整")
        if req.kind=="fill" and (not req.symbol or not req.quantity or req.price<=0):raise HTTPException(400,"成交需标的、非零数量和正价格")
        if req.kind=="cash" and (not req.amount or not req.note):raise HTTPException(400,"现金费用调整必须说明原因；不用于入金出金")
        if req.kind=="fill" and req.quantity != int(req.quantity):raise HTTPException(400,"当前手工账本仅支持整股")
        market=p.frozen["config"]["market"]
        if market=="ashare" and req.quantity>0 and req.quantity%100:raise HTTPException(400,"A股买入必须100股整数倍")
        cal=calendar(market,req.date,req.date)
        if not cal.is_session(str(req.date)) or datetime.now(timezone.utc)<cal.session_close(str(req.date)).to_pydatetime():
            raise HTTPException(400,"请在实际成交日收盘后回填；盘中保持计划供人工执行")
        b=await s.scalar(select(PlanBatch).where(PlanBatch.instance_id==instance_id,PlanBatch.execution_date==str(req.date)))
        if not b or b.reconciliation:raise HTTPException(409,"需对应日期的未确认计划；已对账账本不可追改")
        if req.kind=="fill":
            items=[i for v in b.snapshot["sleeves"] if v["sleeve_id"]==req.sleeve_id for i in v["items"] if i["symbol"]==req.symbol]
            if not items:raise HTTPException(400,"标的不在当日计划/持仓内")
            payload["atr_pct"]=items[0].get("atr_pct")
        prior=[e.payload for e in (await s.scalars(select(ManualEntry).where(ManualEntry.instance_id==instance_id))).all() if e.payload["sleeve_id"]==req.sleeve_id]
        spec=next(v for v in p.frozen["sleeves"] if v["factor_id"]==req.sleeve_id)
        cfg=engine_config(p.frozen,spec)
        cash=cfg.initial_capital+sum(e["amount"] if e["kind"]=="cash" else -e["quantity"]*e["price"]-e["fees"] for e in prior)
        quantity=sum(e["quantity"] for e in prior if e["kind"]=="fill" and e["symbol"]==req.symbol)
        if req.kind=="fill":
            if cfg.mode=="long_only" and quantity+req.quantity < -1e-8:raise HTTPException(400,"不能卖出超过实际持仓的数量")
            if market=="ashare" and req.quantity<0:
                available=sum(e["quantity"] for e in prior if e["kind"]=="fill" and e["symbol"]==req.symbol and (e["date"]<str(req.date) or e["quantity"]<0))
                if -req.quantity>available:raise HTTPException(400,"A股T+1可卖数量不足")
            cash-=req.quantity*req.price+req.fees
        else:cash+=req.amount
        if cfg.resolved_account_type=="cash" and cash < -1e-6:raise HTTPException(400,"现金余额不足，请检查本金、费用与成交")
        e=ManualEntry(instance_id=instance_id,external_id=req.external_id,payload=payload)
        s.add(e);p.revision+=1;await s.commit();return {"id":e.id}


@router.post("/{instance_id}/batches/{batch_id}/reconcile")
async def reconcile(instance_id:int,batch_id:int,req:ReconcileReq):
    async with SessionLocal() as s:
        p=await s.scalar(select(PlanInstance).where(PlanInstance.id==instance_id).with_for_update())
        b=await s.get(PlanBatch,batch_id)
        if not p or not b or b.instance_id!=instance_id:raise HTTPException(404,"计划不存在")
        if p.status=="archived":raise HTTPException(409,"归档只读")
        if b.reconciliation:return b.reconciliation
        cal=calendar(p.frozen["config"]["market"],date.fromisoformat(b.execution_date),date.fromisoformat(b.execution_date))
        if datetime.now(timezone.utc)<cal.session_close(b.execution_date).to_pydatetime():raise HTTPException(409,"请收盘后确认，避免遗漏当日成交")
        entries=[{**e.payload,"sequence":e.id} for e in (await s.scalars(select(ManualEntry).where(ManualEntry.instance_id==instance_id))).all()]
        try:
            check=await asyncio.to_thread(build_plan,p.frozen,entries,cal.next_session(b.execution_date).date(),reconciliation_only=True)
        except Exception as exc:raise HTTPException(409,"对账未完成: "+str(exc)) from exc
        b.reconciliation={"note":req.note,"confirmed_at":datetime.now(timezone.utc).isoformat(),
            "nlv":check["nlv"],"cash":check["cash"],"curve":check["curve"],"sleeves":check["sleeves"]}
        p.revision+=1;await s.commit();return b.reconciliation


@router.post("/{instance_id}/import-csv")
async def import_csv(instance_id:int,req:CsvReq):
    reader=csv.DictReader(io.StringIO(req.csv_text.lstrip("\ufeff")))
    required={"external_id","sleeve_id","date","symbol","quantity","price"}
    if not required <= set(reader.fieldnames or []):raise HTTPException(400,"CSV须包含: "+", ".join(sorted(required)))
    rows=list(reader)
    if not 1<=len(rows)<=1000:raise HTTPException(400,"每批1至1000笔")
    results=[]
    # Per-row transactions are deliberate: report partial imports explicitly;
    # external_id makes a corrected-file retry safe without duplicate entries.
    for index,row in enumerate(rows,start=2):
        try:
            parsed=FillReq(**{k:v for k,v in row.items() if v not in (None,"")})
            result=await add_entry(instance_id,parsed)
            results.append({"line":index,"ok":True,**result})
        except Exception as exc:
            results.append({"line":index,"ok":False,"error":str(getattr(exc,"detail",exc))[:500]})
    return {"rows":results,"saved_or_duplicate":sum(v["ok"] for v in results),"failed":sum(not v["ok"] for v in results),"atomic":False}


@router.get("/{instance_id}/batches/{batch_id}.csv")
async def export(instance_id:int,batch_id:int):
    async with SessionLocal() as s:
        b=await s.get(PlanBatch,batch_id)
        if not b or b.instance_id!=instance_id:raise HTTPException(404,"计划不存在")
        buf=io.StringIO(); keys=["sleeve_id","symbol","name","action","current_quantity","target_quantity","delta_quantity","reference_price","target_weight","portfolio_target_weight","estimated_amount","stop_price","take_profit_price","reason"]
        writer=csv.DictWriter(buf,keys,extrasaction="ignore");writer.writeheader()
        for sleeve in b.snapshot["sleeves"]:
            for row in sleeve["items"]:
                writer.writerow({k:("'"+v if isinstance(v,str) and v.startswith(("=","+","-","@")) else v) for k,v in row.items()})
        return Response("\ufeff"+buf.getvalue(),media_type="text/csv",headers={"Content-Disposition":f'attachment; filename="plan-{instance_id}-{b.execution_date}.csv"'})


async def scheduler():
    while True:
        try:
            async with SessionLocal() as s:
                instances=(await s.scalars(select(PlanInstance).where(PlanInstance.status=="active"))).all()
            now=datetime.now(timezone.utc)
            for p in instances:
                try:
                    market=p.frozen["config"]["market"]
                    cal=calendar(market,now.date()-timedelta(days=10),now.date()+timedelta(days=14))
                    upcoming=[d.date() for d in cal.sessions if cal.session_open(d).to_pydatetime()>now]
                    execution=upcoming[0]
                    if execution<date.fromisoformat(p.frozen["first_execution"]):continue
                    signal=cal.previous_session(str(execution))
                    if now<cal.session_close(signal).to_pydatetime()+timedelta(minutes=30):continue
                    await generate(p.id,execution)
                except Exception as exc:
                    log.info("Manual plan %s: %s",p.id,exc)
                    if not isinstance(exc,HTTPException) or exc.status_code!=409:
                        async with SessionLocal() as s:
                            current=await s.get(PlanInstance,p.id)
                            if current and current.progress.get("state")!="running":
                                current.progress={"state":"blocked","message":str(exc)[:1000]}
                                await s.commit()
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("Manual plan scheduler failed")
        await asyncio.sleep(60)
