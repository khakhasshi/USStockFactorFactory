"""7x24 双层嵌套优化编排器.

外层步 = 提出候选 HarnessSpec -> 用该配置跑一段内层挖掘 (固定评估预算) -> meta-score 对比在位者 -> 接受/拒绝。
内层挖掘产生搜索树节点; public 达标的因子自动注册进因子库 (gate 指标同时存库但不进提示词)。
"""

import asyncio
import traceback
from collections import deque
from datetime import datetime

from sqlalchemy import select

from .config import DEFAULT_ENGINE_CONFIG, DEFAULT_HARNESS_SPEC
from .data.panel import PanelStore
from .db import SessionLocal
from .dsl.engine import normalize_hash
from .eval.harness import evaluate
from .meta.agent import propose_spec
from .miner.agent import propose
from .models import EngineEvent, Factor, MinerVersion, Node, OuterStep, Setting, Trial

import random


class Engine:
    _instance = None

    def __init__(self) -> None:
        self.running = False
        self.task: asyncio.Task | None = None
        self.status: dict = {"state": "stopped", "outer_step": 0, "inner_evals": 0}
        self.logbuf: deque[dict] = deque(maxlen=300)

    @classmethod
    def get(cls) -> "Engine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def log(self, msg: str, level: str = "info") -> None:
        entry = {"t": datetime.now().strftime("%m-%d %H:%M:%S"), "level": level, "msg": msg}
        self.logbuf.append(entry)
        async with SessionLocal() as s:
            s.add(EngineEvent(level=level, message=msg))
            await s.commit()

    async def start(self) -> dict:
        if self.running:
            return {"ok": False, "msg": "已在运行"}
        self.running = True
        self.status["state"] = "starting"
        self.task = asyncio.create_task(self._run())
        return {"ok": True}

    async def stop(self) -> dict:
        self.running = False
        self.status["state"] = "stopping"
        return {"ok": True, "msg": "将在当前评估完成后停止"}

    # ---------- 主循环 ----------

    async def _run(self) -> None:
        try:
            await self.log("引擎启动: 加载数据面板...")
            await asyncio.to_thread(PanelStore.get().ensure_loaded)
            await self.log(f"面板就绪: {PanelStore.get().summary()['rows']} 行")
            incumbent = await self._ensure_incumbent()
            self.status["state"] = "running"
            cfg = await self._config()

            while self.running:
                step_no = await self._next_step_no()
                self.status["outer_step"] = step_no
                incumbent, cfg = await self._outer_step(step_no, incumbent, cfg)
        except Exception:  # noqa: BLE001
            await self.log(f"引擎异常退出:\n{traceback.format_exc()}", "error")
        finally:
            self.status["state"] = "stopped"
            self.running = False

    async def _outer_step(self, step_no: int, incumbent: MinerVersion, cfg: dict):
        provider = await self._provider("outer_provider")
        history = await self._version_history()
        cand_spec, note, source = await propose_spec(incumbent.harness_spec, history, provider)
        async with SessionLocal() as s:
            cand = MinerVersion(
                version_no=len(history) + 1, parent_id=incumbent.id,
                harness_spec=cand_spec, status="candidate", proposal_note=f"[{source}] {note}",
            )
            s.add(cand)
            await s.commit()
            await s.refresh(cand)
        await self.log(f"外层步 {step_no}: 候选 v{cand.version_no} {note} ({source})")

        budget = int(cfg["inner_budget_per_outer_step"])
        cand_score = await self._mining_session(cand, step_no, budget, cfg)

        # 在位者分数: 定期重测 (noise band), 其余沿用缓存
        if incumbent.meta_score is None or step_no % int(cfg["incumbent_remeasure_every"]) == 0:
            inc_score = await self._mining_session(incumbent, step_no, budget, cfg)
            incumbent = await self._update_score(incumbent.id, inc_score)
        inc_score = incumbent.meta_score or 0.0

        accepted = cand_score > inc_score + float(cfg["outer_accept_epsilon"])
        async with SessionLocal() as s:
            s.add(OuterStep(
                step_no=step_no, candidate_id=cand.id, incumbent_id=incumbent.id,
                candidate_score=cand_score, incumbent_score=inc_score, accepted=accepted,
                detail={"note": note, "source": source, "budget": budget},
            ))
            cand_db = await s.get(MinerVersion, cand.id)
            cand_db.meta_score = cand_score
            if accepted:
                cand_db.status = "incumbent"
                inc_db = await s.get(MinerVersion, incumbent.id)
                inc_db.status = "rejected" if inc_db.status == "candidate" else "superseded"
            else:
                cand_db.status = "rejected"
            await s.commit()
        verdict = "接受 ✓" if accepted else "拒绝 ✗"
        await self.log(f"外层步 {step_no}: 候选 {cand_score:.4f} vs 在位 {inc_score:.4f} -> {verdict}")
        if accepted:
            async with SessionLocal() as s:
                incumbent = await s.get(MinerVersion, cand.id)
        return incumbent, cfg

    async def _mining_session(self, miner: MinerVersion, step_no: int, budget: int, cfg: dict) -> float:
        """跑一段内层挖掘, 返回 meta-score = 各任务 top 因子 gate 分数均值."""
        spec = miner.harness_spec
        provider = await self._provider("inner_provider")
        tasks = cfg["tasks"]
        task_best_gate: dict[str, float] = {}

        for i in range(budget):
            if not self.running:
                break
            task = tasks[i % len(tasks)]
            top_nodes = await self._top_nodes(miner.id, task["name"])
            op = "improve" if (top_nodes and random.random() < float(spec.get("improve_bias", 0.6))) else "draft"
            expr, hypo, source = await propose(spec, op, task, top_nodes, provider)

            node = Node(
                miner_version_id=miner.id, outer_step_no=step_no,
                parent_id=top_nodes[0]["id"] if (op == "improve" and top_nodes) else None,
                op=op, expression=expr, hypothesis=hypo, source=source, task_name=task["name"],
            )
            try:
                metrics = await asyncio.to_thread(evaluate, expr, task["universe_n"], task["horizon"])
                node.status = "ok"
                node.public_metrics = metrics["public"]
                node.gate_metrics = metrics["gate"]
                node.public_score = metrics["public"].get("score") or 0.0
                gate_score = metrics["gate"].get("score") or 0.0
                task_best_gate[task["name"]] = max(task_best_gate.get(task["name"], 0.0), gate_score)
            except Exception as e:  # noqa: BLE001
                node.status = "error"
                node.error = str(e)[:500]
            async with SessionLocal() as s:
                s.add(node)
                s.add(Trial(
                    expression_hash=normalize_hash(expr) if node.status == "ok" else "invalid",
                    layer="INNER_PUBLIC+META_TRAIN", task_name=task["name"],
                    statistic={"public_score": node.public_score},
                ))
                await s.commit()
                await s.refresh(node)
            self.status["inner_evals"] = self.status.get("inner_evals", 0) + 1
            if node.status == "ok":
                await self._maybe_register_factor(node, spec)
                await self.log(
                    f"  内层[{task['name']}] {op}/{source} score={node.public_score:.3f} {expr[:80]}", "debug"
                )
            else:
                await self.log(f"  内层[{task['name']}] {op} 失败: {node.error[:80]}", "debug")

        scores = list(task_best_gate.values())
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    async def _maybe_register_factor(self, node: Node, spec: dict) -> None:
        pm = node.public_metrics
        if abs(pm.get("icir") or 0) < float(spec.get("min_public_icir", 0.25)):
            return
        async with SessionLocal() as s:
            exists = await s.scalar(select(Factor).where(Factor.expression == node.expression))
            if exists:
                return
            n = await s.scalar(select(Factor.id).order_by(Factor.id.desc()).limit(1)) or 0
            s.add(Factor(
                name=f"F{n + 1:05d}", expression=node.expression, hypothesis=node.hypothesis,
                status="public-leading", node_id=node.id, task_name=node.task_name,
                public_metrics=node.public_metrics, gate_metrics=node.gate_metrics,
                fingerprint={
                    "miner_version_id": node.miner_version_id, "outer_step": node.outer_step_no,
                    "source": node.source, "expr_hash": normalize_hash(node.expression),
                },
            ))
            await s.commit()
            await self.log(f"  ★ 新因子入库: {node.expression[:80]}")

    # ---------- 辅助 ----------

    async def _ensure_incumbent(self) -> MinerVersion:
        async with SessionLocal() as s:
            inc = await s.scalar(
                select(MinerVersion).where(MinerVersion.status == "incumbent").order_by(MinerVersion.id.desc())
            )
            if inc:
                return inc
            inc = MinerVersion(version_no=0, harness_spec=DEFAULT_HARNESS_SPEC,
                               status="incumbent", proposal_note="Miner_0 基线")
            s.add(inc)
            await s.commit()
            await s.refresh(inc)
            return inc

    async def _update_score(self, mid: int, score: float) -> MinerVersion:
        async with SessionLocal() as s:
            m = await s.get(MinerVersion, mid)
            m.meta_score = score
            await s.commit()
            await s.refresh(m)
            return m

    async def _next_step_no(self) -> int:
        async with SessionLocal() as s:
            last = await s.scalar(select(OuterStep.step_no).order_by(OuterStep.step_no.desc()).limit(1))
            return (last or 0) + 1

    async def _version_history(self) -> list[dict]:
        async with SessionLocal() as s:
            rows = (await s.scalars(select(MinerVersion).order_by(MinerVersion.id))).all()
            return [
                {"version_no": m.version_no, "harness_spec": m.harness_spec,
                 "meta_score": m.meta_score, "status": m.status}
                for m in rows
            ]

    async def _top_nodes(self, miner_id: int, task_name: str, k: int = 10) -> list[dict]:
        async with SessionLocal() as s:
            rows = (await s.scalars(
                select(Node)
                .where(Node.miner_version_id == miner_id, Node.task_name == task_name, Node.status == "ok")
                .order_by(Node.public_score.desc().nulls_last())
                .limit(k)
            )).all()
            return [
                {"id": n.id, "expression": n.expression, "public_score": n.public_score or 0.0,
                 "public_metrics": n.public_metrics}
                for n in rows
            ]

    async def _provider(self, role: str) -> dict | None:
        async with SessionLocal() as s:
            row = await s.get(Setting, "llm_providers")
            if not row:
                return None
            conf = row.value
            name = conf.get(role)
            for p in conf.get("providers", []):
                if p.get("name") == name and p.get("api_key"):
                    return p
            return None

    async def _config(self) -> dict:
        async with SessionLocal() as s:
            row = await s.get(Setting, "engine_config")
            return {**DEFAULT_ENGINE_CONFIG, **(row.value if row else {})}
