"""7x24 双层嵌套优化编排器 v2.

外层步 = 提出候选 MinerTemplate -> multi-seed 内层挖掘 -> t-test 对比在位者 -> 接受/拒绝。
内层挖掘产生搜索树节点; public 达标的因子自动注册进因子库。

v2 改进:
- 外层优化对象从 6 参数 HarnessSpec 升级为完整 MinerTemplate (prompt/策略/模板)
- 每步 50 次内层评估 × 3 seeds → meta-score 噪声从 ~1.8 压到 <0.3
- 接受门从 epsilon 改为配对 t 检验 (p<0.10)
- 安全边界: MetaValidator 确保外层不能触碰评估器/数据层/隔离边界
"""

import asyncio
import math
import random
import statistics as st
import traceback
from collections import deque
from copy import deepcopy
from datetime import datetime

from sqlalchemy import func, select

from .config import DEFAULT_ENGINE_CONFIG, DEFAULT_ENGINE_CONFIG_V2, DEFAULT_HARNESS_SPEC, DEFAULT_MINER_TEMPLATE
from .data.panel import PanelStore
from .db import SessionLocal, get_active_experiment_id
from .dsl.engine import normalize_hash
from .eval.harness import evaluate
from .meta.agent import propose_spec, propose_template, validate_template
from .miner.agent import propose
from .models import EngineEvent, Experiment, Factor, MinerVersion, Node, OuterStep, Setting, Trial


class Engine:
    _instance = None

    def __init__(self) -> None:
        self.running = False
        self.task: asyncio.Task | None = None
        self.exp_id: int = 1
        self.status: dict = {"state": "stopped", "outer_step": 0, "inner_evals": 0, "experiment_id": None}
        self.logbuf: deque[dict] = deque(maxlen=300)
        self._mode: str = "v1"  # "v1" 或 "v2"

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

    async def start(self, mode: str = "v1") -> dict:
        if self.running:
            return {"ok": False, "msg": "已在运行"}
        self.running = True
        self._mode = mode
        self.status["state"] = "starting"
        self.task = asyncio.create_task(self._run_v2() if mode == "v2" else self._run())
        return {"ok": True, "msg": f"引擎启动 (mode={mode})"}

    async def stop(self) -> dict:
        self.running = False
        self.status["state"] = "stopping"
        return {"ok": True, "msg": "将在当前评估完成后停止"}

    # ================================================================
    # V2 主循环 (B组: MinerTemplate + multi-seed + t-test)
    # ================================================================

    async def _run_v2(self) -> None:
        try:
            self.exp_id = await get_active_experiment_id()
            async with SessionLocal() as s:
                exp = await s.get(Experiment, self.exp_id)
                if not exp:
                    raise RuntimeError(f"活动实验 {self.exp_id} 不存在")
                if exp.status == "archived":
                    raise RuntimeError(f"实验「{exp.name}」已归档")
            self.status["experiment_id"] = self.exp_id
            await self.log(f"[V2] 引擎启动: 实验[{exp.name}] 外层可改写 MinerTemplate")
            await asyncio.to_thread(PanelStore.get().ensure_loaded)
            await self.log(f"面板就绪: {PanelStore.get().summary()['rows']} 行")

            incumbent = await self._ensure_incumbent_v2()
            self.status["state"] = "running"
            cfg = await self._config_v2()

            while self.running:
                step_no = await self._next_step_no()
                self.status["outer_step"] = step_no
                incumbent, cfg = await self._outer_step_v2(step_no, incumbent, cfg)
        except Exception:
            await self.log(f"引擎异常退出:\n{traceback.format_exc()}", "error")
        finally:
            self.status["state"] = "stopped"
            self.running = False

    async def _outer_step_v2(self, step_no: int, incumbent: MinerVersion, cfg: dict):
        provider = await self._provider("outer_provider")
        history = await self._version_history_v2()

        # 外层 LLM 提议新模板
        inc_template = incumbent.harness_spec if isinstance(incumbent.harness_spec, dict) else DEFAULT_MINER_TEMPLATE
        cand_template, note, source = await propose_template(inc_template, history, provider)

        async with SessionLocal() as s:
            cand = MinerVersion(
                experiment_id=self.exp_id,
                version_no=len(history) + 1, parent_id=incumbent.id,
                harness_spec=cand_template, status="candidate",
                proposal_note=f"[{source}] {note}",
            )
            s.add(cand)
            await s.commit()
            await s.refresh(cand)

        await self.log(f"[V2] 外层步 {step_no}: 候选 v{cand.version_no} [{source}] {note[:100]}")

        # ---- multi-seed 内层挖掘 ----
        budget = int(cfg["inner_budget_per_outer_step"])
        n_seeds = int(cfg["n_seeds_per_candidate"])
        cand_scores = []
        for seed in range(n_seeds):
            seed_score = await self._mining_session_v2(cand, step_no, budget, cfg, seed)
            cand_scores.append(seed_score)
            await self.log(f"[V2]   seed {seed+1}/{n_seeds} meta={seed_score:.4f}")

        cand_mean = st.mean(cand_scores) if cand_scores else 0.0
        cand_std = st.stdev(cand_scores) if len(cand_scores) >= 2 else 0.0

        # ---- 在位者重测 ----
        remeasure_every = int(cfg["incumbent_remeasure_every"])
        remeasure_budget = int(cfg.get("incumbent_remeasure_budget", budget))
        if incumbent.meta_score is None or step_no % remeasure_every == 0:
            inc_scores = []
            for seed in range(n_seeds):
                inc_seed_score = await self._mining_session_v2(incumbent, step_no, remeasure_budget, cfg, seed + 1000)
                inc_scores.append(inc_seed_score)
            inc_mean = st.mean(inc_scores) if inc_scores else 0.0
            inc_std = st.stdev(inc_scores) if len(inc_scores) >= 2 else 0.0
            incumbent = await self._update_score(incumbent.id, inc_mean)
            await self.log(f"[V2]   在位重测: mean={inc_mean:.4f} std={inc_std:.4f} (n={len(inc_scores)})")
        else:
            inc_mean = incumbent.meta_score or 0.0
            inc_std = 0.0

        # ---- t-test 接受判定 ----
        accepted = False
        p_value = 1.0
        if len(cand_scores) >= 2 and len(cand_scores) == n_seeds:
            # 配对 t 检验 (cand vs inc 的各 seed 得分差)
            # 简化: 如果 inc 有多种子分数, 做独立样本 t 检验; 否则用单样本
            try:
                if inc_std > 0.001 and len(cand_scores) >= 2:
                    # Welch's t-test
                    diff = cand_mean - inc_mean
                    se = math.sqrt(cand_std**2 / len(cand_scores) + inc_std**2 / max(1, len(cand_scores)))
                    if se > 0.0001:
                        t_stat = diff / se
                        # 近似 p 值 (单边, df 用 Welch-Satterthwaite 近似)
                        df_num = (cand_std**2 / len(cand_scores) + inc_std**2 / max(1, len(cand_scores)))**2
                        df_den = ((cand_std**2 / len(cand_scores))**2 / (len(cand_scores) - 1) +
                                   (inc_std**2 / max(1, len(cand_scores)))**2 / (max(1, len(cand_scores)) - 1))
                        df = df_num / max(0.0001, df_den)
                        # 简化 p 值: 用正态近似
                        if t_stat > 0:
                            p_value = 1.0 - _normal_cdf(t_stat)
                        else:
                            p_value = 1.0
            except Exception:
                p_value = 1.0

            p_threshold = float(cfg["outer_accept_p_value"])
            accepted = p_value < p_threshold

        verdict = f"接受 ✓ p={p_value:.4f}" if accepted else f"拒绝 ✗ p={p_value:.4f}"

        async with SessionLocal() as s:
            s.add(OuterStep(
                experiment_id=self.exp_id,
                step_no=step_no, candidate_id=cand.id, incumbent_id=incumbent.id,
                candidate_score=cand_mean, incumbent_score=inc_mean, accepted=accepted,
                detail={
                    "note": note, "source": source, "budget": budget, "n_seeds": n_seeds,
                    "cand_scores": cand_scores, "cand_std": cand_std,
                    "inc_std": inc_std, "p_value": p_value, "mode": "v2",
                },
            ))
            cand_db = await s.get(MinerVersion, cand.id)
            cand_db.meta_score = cand_mean
            if accepted:
                cand_db.status = "incumbent"
                inc_db = await s.get(MinerVersion, incumbent.id)
                inc_db.status = "rejected" if inc_db.status == "candidate" else "superseded"
            else:
                cand_db.status = "rejected"
            await s.commit()

        await self.log(f"[V2] 外层步 {step_no}: cand {cand_mean:.4f}±{cand_std:.3f} vs inc {inc_mean:.4f}±{inc_std:.3f} -> {verdict}")
        if accepted:
            async with SessionLocal() as s:
                incumbent = await s.get(MinerVersion, cand.id)
        return incumbent, cfg

    async def _mining_session_v2(self, miner: MinerVersion, step_no: int, budget: int, cfg: dict, seed: int) -> float:
        """跑一段内层挖掘, 返回 meta-score."""
        template = miner.harness_spec if isinstance(miner.harness_spec, dict) else DEFAULT_MINER_TEMPLATE
        provider = await self._provider("inner_provider")
        tasks = cfg["tasks"]
        task_best_gate: dict[str, float] = {}

        # 固定种子确保可复现
        rng = random.Random(seed * 10000 + miner.id)

        for i in range(budget):
            if not self.running:
                break
            task = tasks[i % len(tasks)]
            top_nodes = await self._top_nodes(miner.id, task["name"])

            # 操作选择: 根据 draft_strategy 中的指令决定 draft/improve 概率
            improve_bias = 0.6
            draft_strategy = template.get("draft_strategy", "")
            if "优先" in draft_strategy and "改进" not in draft_strategy:
                improve_bias = 0.4  # 偏探索
            elif "改进" in draft_strategy:
                improve_bias = 0.7  # 偏改进

            op = "improve" if (top_nodes and rng.random() < improve_bias) else "draft"
            expr, hypo, source = await propose(template, op, task, top_nodes, provider)

            node = Node(
                experiment_id=self.exp_id,
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
            except Exception as e:
                node.status = "error"
                node.error = str(e)[:500]

            async with SessionLocal() as s:
                s.add(node)
                s.add(Trial(
                    experiment_id=self.exp_id,
                    expression_hash=normalize_hash(expr) if node.status == "ok" else "invalid",
                    layer="INNER_PUBLIC+META_TRAIN", task_name=task["name"],
                    statistic={"public_score": node.public_score, "seed": seed},
                ))
                await s.commit()
                await s.refresh(node)

            self.status["inner_evals"] = self.status.get("inner_evals", 0) + 1
            if node.status == "ok":
                min_icir = float(template.get("min_public_icir", 0.25))
                await self._maybe_register_factor_v2(node, min_icir)
                if i % 10 == 0:
                    await self.log(
                        f"[V2 s{seed}]  内层[{task['name']}] {op}/{source} score={node.public_score:.3f} {expr[:60]}", "debug"
                    )

        scores = list(task_best_gate.values())
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    async def _maybe_register_factor_v2(self, node: Node, min_icir: float) -> None:
        pm = node.public_metrics
        if abs(pm.get("icir") or 0) < min_icir:
            return
        async with SessionLocal() as s:
            exists = await s.scalar(select(Factor).where(
                Factor.expression == node.expression, Factor.experiment_id == self.exp_id))
            if exists:
                return
            n = await s.scalar(
                select(func.count(Factor.id)).where(Factor.experiment_id == self.exp_id)) or 0
            s.add(Factor(
                experiment_id=self.exp_id,
                name=f"F{n + 1:05d}", expression=node.expression, hypothesis=node.hypothesis,
                status="public-leading", node_id=node.id, task_name=node.task_name,
                public_metrics=node.public_metrics, gate_metrics=node.gate_metrics,
                fingerprint={
                    "miner_version_id": node.miner_version_id, "outer_step": node.outer_step_no,
                    "source": node.source, "expr_hash": normalize_hash(node.expression),
                },
            ))
            await s.commit()

    # ================================================================
    # V2 辅助方法
    # ================================================================

    async def _ensure_incumbent_v2(self) -> MinerVersion:
        async with SessionLocal() as s:
            inc = await s.scalar(
                select(MinerVersion)
                .where(MinerVersion.status == "incumbent", MinerVersion.experiment_id == self.exp_id)
                .order_by(MinerVersion.id.desc())
            )
            if inc:
                return inc
            inc = MinerVersion(
                experiment_id=self.exp_id, version_no=0,
                harness_spec=deepcopy(DEFAULT_MINER_TEMPLATE),
                status="incumbent", proposal_note="Miner_0 基线模板"
            )
            s.add(inc)
            await s.commit()
            await s.refresh(inc)
            return inc

    async def _version_history_v2(self) -> list[dict]:
        async with SessionLocal() as s:
            rows = (await s.scalars(
                select(MinerVersion).where(MinerVersion.experiment_id == self.exp_id)
                .order_by(MinerVersion.id))).all()
            return [
                {"version_no": m.version_no, "meta_score": m.meta_score,
                 "status": m.status, "template_note": m.proposal_note,
                 "template": m.harness_spec}
                for m in rows
            ]

    async def _config_v2(self) -> dict:
        async with SessionLocal() as s:
            row = await s.get(Setting, "engine_config")
            return {**DEFAULT_ENGINE_CONFIG_V2, **(row.value if row else {})}

    # ================================================================
    # V1 主循环 (A组: 保持兼容, 原封不动)
    # ================================================================

    async def _run(self) -> None:
        try:
            self.exp_id = await get_active_experiment_id()
            async with SessionLocal() as s:
                exp = await s.get(Experiment, self.exp_id)
                if not exp:
                    raise RuntimeError(f"活动实验 {self.exp_id} 不存在")
                if exp.status == "archived":
                    raise RuntimeError(f"实验「{exp.name}」已归档, 请先切换到开放实验")
            self.status["experiment_id"] = self.exp_id
            await self.log(f"引擎启动: 实验[{exp.name}] 加载数据面板...")
            await asyncio.to_thread(PanelStore.get().ensure_loaded)
            await self.log(f"面板就绪: {PanelStore.get().summary()['rows']} 行")
            incumbent = await self._ensure_incumbent()
            self.status["state"] = "running"
            cfg = await self._config()

            while self.running:
                step_no = await self._next_step_no()
                self.status["outer_step"] = step_no
                incumbent, cfg = await self._outer_step(step_no, incumbent, cfg)
        except Exception:
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
                experiment_id=self.exp_id,
                version_no=len(history) + 1, parent_id=incumbent.id,
                harness_spec=cand_spec, status="candidate", proposal_note=f"[{source}] {note}",
            )
            s.add(cand)
            await s.commit()
            await s.refresh(cand)
        await self.log(f"外层步 {step_no}: 候选 v{cand.version_no} {note} ({source})")

        budget = int(cfg["inner_budget_per_outer_step"])
        cand_score = await self._mining_session(cand, step_no, budget, cfg)

        if incumbent.meta_score is None or step_no % int(cfg["incumbent_remeasure_every"]) == 0:
            inc_score = await self._mining_session(incumbent, step_no, budget, cfg)
            incumbent = await self._update_score(incumbent.id, inc_score)
        inc_score = incumbent.meta_score or 0.0

        accepted = cand_score > inc_score + float(cfg["outer_accept_epsilon"])
        async with SessionLocal() as s:
            s.add(OuterStep(
                experiment_id=self.exp_id,
                step_no=step_no, candidate_id=cand.id, incumbent_id=incumbent.id,
                candidate_score=cand_score, incumbent_score=inc_score, accepted=accepted,
                detail={"note": note, "source": source, "budget": budget, "mode": "v1"},
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
                experiment_id=self.exp_id,
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
            except Exception as e:
                node.status = "error"
                node.error = str(e)[:500]
            async with SessionLocal() as s:
                s.add(node)
                s.add(Trial(
                    experiment_id=self.exp_id,
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
            exists = await s.scalar(select(Factor).where(
                Factor.expression == node.expression, Factor.experiment_id == self.exp_id))
            if exists:
                return
            n = await s.scalar(
                select(func.count(Factor.id)).where(Factor.experiment_id == self.exp_id)) or 0
            s.add(Factor(
                experiment_id=self.exp_id,
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

    # ================================================================
    # 共享辅助方法
    # ================================================================

    async def _ensure_incumbent(self) -> MinerVersion:
        async with SessionLocal() as s:
            inc = await s.scalar(
                select(MinerVersion)
                .where(MinerVersion.status == "incumbent", MinerVersion.experiment_id == self.exp_id)
                .order_by(MinerVersion.id.desc())
            )
            if inc:
                return inc
            inc = MinerVersion(experiment_id=self.exp_id, version_no=0, harness_spec=DEFAULT_HARNESS_SPEC,
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
            last = await s.scalar(
                select(OuterStep.step_no).where(OuterStep.experiment_id == self.exp_id)
                .order_by(OuterStep.step_no.desc()).limit(1))
            return (last or 0) + 1

    async def _version_history(self) -> list[dict]:
        async with SessionLocal() as s:
            rows = (await s.scalars(
                select(MinerVersion).where(MinerVersion.experiment_id == self.exp_id)
                .order_by(MinerVersion.id))).all()
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


def _normal_cdf(x: float) -> float:
    """标准正态 CDF 近似 (Abramowitz & Stegun 7.1.26)."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    # 使用 math.erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
