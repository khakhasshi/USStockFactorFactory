"""7x24 双层嵌套优化编排器 v2.

外层步 = 提出候选 MinerTemplate -> multi-seed 内层挖掘 -> t-test 对比在位者 -> 接受/拒绝。
内层挖掘产生搜索树节点; public 达标的因子自动注册进因子库。

v2 改进:
- 外层优化对象从 6 参数 HarnessSpec 升级为完整 MinerTemplate (prompt/策略/模板)
- 每步 50 次内层评估 × 3 seeds → meta-score 噪声从 ~1.8 压到 <0.3
- 接受门从 epsilon 改为单边 Student t 检验（同时重测时用 Welch）(p<0.10)
- 安全边界: MetaValidator 确保外层不能触碰评估器/数据层/隔离边界
"""

import asyncio
import math
import os
import random
import statistics as st
import time
import traceback
from collections import Counter, deque
from copy import deepcopy
from datetime import datetime

from sqlalchemy import func, select

from .config import (
    DEFAULT_RESEARCH_DIRECTION_POLICY,
    DEFAULT_ENGINE_CONFIG,
    DEFAULT_ENGINE_CONFIG_V2,
    DEFAULT_HARNESS_SPEC,
    DEFAULT_MINER_TEMPLATE,
    DEFAULT_PORTFOLIO_MODE,
    EVALUATION_PROTOCOL_VERSION,
    get_dsl_fields,
    resolve_engine_tasks,
)
from .data.panel import PanelStore
from .db import SessionLocal, get_active_experiment_id
from .dsl.engine import normalize_hash
from .eval.harness import evaluate
from .feedback import (
    build_feedback_envelope,
    combine_seed_feedback,
    compare_feedback_reports,
)
from .factors.similarity import expression_fingerprint
from .meta.agent import (
    propose_spec,
    propose_template,
    reflect_on_outcome,
    validate_template,
)
from .miner.agent import propose
from .models import EngineEvent, Experiment, Factor, MinerVersion, Node, OuterStep, Setting, Trial
from .observability import redact_text, redact_value, utc_now


class Engine:
    _instance = None

    def __init__(self) -> None:
        self.running = False
        self.task: asyncio.Task | None = None
        self.exp_id: int = 1
        now = utc_now()
        self.status: dict = {
            "state": "stopped",
            "phase": "idle",
            "phase_started_at": now,
            "outer_step": 0,
            "inner_evals": 0,
            "experiment_id": None,
            "started_at": None,
            "stopped_at": now,
            "last_heartbeat_at": now,
            "last_progress_at": None,
            "last_log_at": None,
            "last_error": None,
            "current_task": None,
            "current_operation": None,
            "current_seed": None,
            "current_budget_index": None,
            "current_budget_total": None,
            "evaluation_active": False,
            "evaluation_started_at": None,
            "evaluation_elapsed_seconds": None,
            "evaluation_heartbeat_count": 0,
            "evaluation_soft_deadline_seconds": None,
            "evaluation_deadline_exceeded": False,
            "last_evaluation_duration_seconds": None,
        }
        self._started_monotonic: float | None = None
        self._last_heartbeat_monotonic = time.monotonic()
        self.logbuf: deque[dict] = deque(maxlen=300)
        self._mode: str = "v1"  # "v1" 或 "v2"
        self.task_config: dict = {}

    @classmethod
    def get(cls) -> "Engine":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _set_phase(self, phase: str, *, progress: bool = False, **detail) -> None:
        now = utc_now()
        if self.status.get("phase") != phase:
            self.status["phase_started_at"] = now
        self.status["phase"] = phase
        self.status["last_heartbeat_at"] = now
        if progress:
            self.status["last_progress_at"] = now
        self._last_heartbeat_monotonic = time.monotonic()
        self.status.update(detail)

    def _touch_progress(self, **detail) -> None:
        self._set_phase(self.status.get("phase") or "running", progress=True, **detail)

    async def _run_blocking_with_heartbeat(
        self,
        func,
        *args,
        operation: str = "factor_evaluation",
    ):
        """Run CPU-bound work without making a healthy worker look dead.

        Cancellation is drained before the research task reports itself
        stopped.  Python cannot safely kill a running thread; waiting here
        avoids the previous state where the UI said "stopped" while Polars was
        still consuming CPU in an orphaned evaluation.
        """
        try:
            heartbeat_seconds = max(
                0.01,
                float(os.environ.get("FF_EVALUATION_HEARTBEAT_SECONDS", "15")),
            )
        except ValueError:
            heartbeat_seconds = 15.0
        try:
            soft_deadline_seconds = max(
                heartbeat_seconds,
                float(os.environ.get("FF_EVALUATION_SOFT_DEADLINE_SECONDS", "180")),
            )
        except ValueError:
            soft_deadline_seconds = 180.0

        started = time.monotonic()
        started_at = utc_now()
        heartbeat_count = 0
        evaluation_task = asyncio.create_task(
            asyncio.to_thread(func, *args),
            name=f"research.evaluation.{self.exp_id}",
        )
        self._set_phase(
            self.status.get("phase") or "candidate_mining",
            current_operation=operation,
            evaluation_active=True,
            evaluation_started_at=started_at,
            evaluation_elapsed_seconds=0.0,
            evaluation_heartbeat_count=0,
            evaluation_soft_deadline_seconds=soft_deadline_seconds,
            evaluation_deadline_exceeded=False,
        )

        async def wait_until_done(*, draining: bool = False):
            nonlocal heartbeat_count
            while True:
                done, _ = await asyncio.wait(
                    {evaluation_task},
                    timeout=heartbeat_seconds,
                )
                elapsed = max(0.0, time.monotonic() - started)
                if done:
                    return elapsed
                heartbeat_count += 1
                self._set_phase(
                    self.status.get("phase") or "candidate_mining",
                    current_operation=(
                        "factor_evaluation_draining"
                        if draining
                        else operation
                    ),
                    evaluation_active=True,
                    evaluation_elapsed_seconds=round(elapsed, 3),
                    evaluation_heartbeat_count=heartbeat_count,
                    evaluation_deadline_exceeded=(
                        elapsed >= soft_deadline_seconds
                    ),
                )

        try:
            elapsed = await wait_until_done()
            return evaluation_task.result()
        except asyncio.CancelledError:
            # Keep the task alive and wait for the underlying thread.  This is
            # intentionally truthful backpressure for the stop endpoint.
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                current_operation="factor_evaluation_draining",
            )
            elapsed = await wait_until_done(draining=True)
            try:
                evaluation_task.result()
            except Exception:
                # The candidate will not be persisted after cancellation, but
                # the exception is consumed so asyncio does not report an
                # unhandled background-task failure.
                pass
            raise
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                evaluation_active=False,
                evaluation_elapsed_seconds=round(elapsed, 3),
                evaluation_heartbeat_count=heartbeat_count,
                evaluation_deadline_exceeded=(
                    elapsed >= soft_deadline_seconds
                ),
                last_evaluation_duration_seconds=round(elapsed, 3),
            )

    async def log(self, msg: str, level: str = "info") -> None:
        now = utc_now()
        safe_msg = redact_text(msg, 4000)
        entry = {
            "t": datetime.now().strftime("%m-%d %H:%M:%S"),
            "at": now,
            "level": level,
            "msg": safe_msg,
            "phase": self.status.get("phase"),
        }
        self.logbuf.append(entry)
        self.status["last_log_at"] = now
        self.status["last_heartbeat_at"] = now
        self._last_heartbeat_monotonic = time.monotonic()
        if level in {"error", "critical"}:
            self.status["last_error"] = safe_msg[-1200:]
        async with SessionLocal() as s:
            s.add(EngineEvent(
                level=level,
                message=safe_msg,
                experiment_id=self.exp_id,
                payload={
                    "state": self.status.get("state"),
                    "phase": self.status.get("phase"),
                    "mode": self._mode,
                    "outer_step": self.status.get("outer_step"),
                    "inner_evals": self.status.get("inner_evals"),
                    "current_task": self.status.get("current_task"),
                },
            ))
            await s.commit()

    def _panel_glob(self) -> str | None:
        return self.task_config.get("panel_glob") or os.environ.get("FF_PANEL_GLOB")

    def _portfolio_mode(self) -> str:
        return self.task_config.get("portfolio_mode", DEFAULT_PORTFOLIO_MODE)

    def _signal_direction(self) -> int:
        direction = int(self.task_config.get("direction", 1))
        if direction not in {-1, 1}:
            raise ValueError("研究任务 direction 必须为 1 或 -1")
        return direction

    def _direction_policy(self) -> str:
        return str(
            self.task_config.get("direction_policy")
            or DEFAULT_RESEARCH_DIRECTION_POLICY
        )

    async def start(self, mode: str = "v2", experiment_id: int | None = None) -> dict:
        if mode != "v2":
            return {
                "ok": False,
                "msg": (
                    "V1 引擎已冻结为历史只读实现；"
                    "新研究只能使用 V2 双层反馈引擎"
                ),
            }
        if self.running:
            return {"ok": False, "msg": "已在运行"}
        # 在创建后台任务前锁定活动实验，避免 UI/API 在启动窗口切换实验后跑错任务。
        self.exp_id = experiment_id or await get_active_experiment_id()
        async with SessionLocal() as s:
            exp = await s.get(Experiment, self.exp_id)
            self.task_config = dict(exp.research_config or {}) if exp else {}
        self.status["experiment_id"] = self.exp_id
        self.running = True
        self._mode = mode
        now = utc_now()
        self._started_monotonic = time.monotonic()
        self.status.update({
            "state": "starting",
            "started_at": now,
            "stopped_at": None,
            "last_error": None,
            "current_task": None,
            "current_operation": None,
            "current_seed": None,
            "current_budget_index": None,
            "current_budget_total": None,
            "evaluation_active": False,
            "evaluation_started_at": None,
            "evaluation_elapsed_seconds": None,
            "evaluation_heartbeat_count": 0,
            "evaluation_soft_deadline_seconds": None,
            "evaluation_deadline_exceeded": False,
        })
        self._set_phase("starting", progress=True)
        self.task = asyncio.create_task(
            self._run_v2() if mode == "v2" else self._run(),
            name=f"research.worker.{self.exp_id}.{mode}",
        )
        return {"ok": True, "msg": f"引擎启动 (mode={mode})"}

    async def stop(self) -> dict:
        self.running = False
        self.status["state"] = "stopping"
        self._set_phase("stopping", progress=True)
        task = self.task
        if task and not task.done() and task is not asyncio.current_task():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.status.update({"state": "stopped", "stopped_at": utc_now()})
        self._set_phase("stopped", progress=True)
        return {"ok": True, "msg": "引擎已停止，已保留已提交研究数据"}

    def diagnostics(self, include_logs: bool = True) -> dict:
        task = self.task
        task_done = bool(task and task.done())
        task_cancelled = bool(task and task.cancelled())
        task_exception = None
        if task_done and not task_cancelled:
            try:
                exc = task.exception()
            except (asyncio.CancelledError, asyncio.InvalidStateError):
                exc = None
            if exc is not None:
                task_exception = redact_text(exc, 1200)
        heartbeat_age = max(0.0, time.monotonic() - self._last_heartbeat_monotonic)
        uptime = (
            max(0.0, time.monotonic() - self._started_monotonic)
            if self._started_monotonic is not None
            else 0.0
        )
        levels = Counter(row.get("level", "info") for row in self.logbuf)
        result = dict(self.status)
        result.update({
            "mode": self._mode,
            "running": self.running,
            "task_created": task is not None,
            "task_done": task_done,
            "task_cancelled": task_cancelled,
            "task_exception": task_exception,
            "heartbeat_age_seconds": round(heartbeat_age, 3),
            "heartbeat_stale": bool(self.running and heartbeat_age >= 300.0),
            "uptime_seconds": round(uptime, 3),
            "log_counts": dict(levels),
            "task_config": redact_value(self.task_config),
        })
        if include_logs:
            result["logs"] = list(self.logbuf)
        return result

    # ================================================================
    # V2 主循环 (B组: MinerTemplate + multi-seed + t-test)
    # ================================================================

    async def _run_v2(self) -> None:
        try:
            self._set_phase("validating_experiment", progress=True)
            async with SessionLocal() as s:
                exp = await s.get(Experiment, self.exp_id)
                if not exp:
                    raise RuntimeError(f"活动实验 {self.exp_id} 不存在")
                if exp.status == "archived":
                    raise RuntimeError(f"实验「{exp.name}」已归档")
                configured_protocol = (
                    exp.research_config or {}
                ).get("evaluation_protocol")
                if configured_protocol != EVALUATION_PROTOCOL_VERSION:
                    raise RuntimeError(
                        f"实验「{exp.name}」评价协议为 "
                        f"{configured_protocol or 'legacy'}，不能与当前 "
                        f"{EVALUATION_PROTOCOL_VERSION} 搜索历史混用；"
                        "请新建或升级任务配置后再启动"
                    )
            self.status["experiment_id"] = self.exp_id
            await self.log(f"[V2] 引擎启动: 实验[{exp.name}] 外层可改写 MinerTemplate")
            self._set_phase("loading_panel", progress=True)
            panel = PanelStore.get(self._panel_glob(), self.task_config.get("market", "us"))
            await asyncio.to_thread(panel.ensure_loaded)
            await self.log(f"面板就绪: {panel.summary()['rows']} 行 · {self.task_config.get('market', 'configured')}")

            self._set_phase("initializing_miner", progress=True)
            incumbent = await self._ensure_incumbent_v2()
            self.status["state"] = "running"
            cfg = await self._config_v2()

            while self.running:
                step_no = await self._next_step_no()
                self.status["outer_step"] = step_no
                self._set_phase(
                    "outer_step",
                    progress=True,
                    current_operation="prepare",
                    current_seed=None,
                    current_budget_index=None,
                    current_budget_total=None,
                )
                incumbent, cfg = await self._outer_step_v2(step_no, incumbent, cfg)
        except Exception:
            self._set_phase("failed")
            await self.log(f"引擎异常退出:\n{traceback.format_exc()}", "error")
        finally:
            self.running = False
            self.status.update({"state": "stopped", "stopped_at": utc_now()})
            if self.status.get("phase") != "failed":
                self._set_phase("stopped", progress=True)

    async def _outer_step_v2(self, step_no: int, incumbent: MinerVersion, cfg: dict):
        self._set_phase(
            "outer_proposal",
            progress=True,
            current_operation="load_context",
            current_task=None,
        )
        provider = await self._provider("outer_provider")
        history = await self._version_history_v2()

        # 外层 LLM 提议新模板
        inc_template = incumbent.harness_spec if isinstance(incumbent.harness_spec, dict) else DEFAULT_MINER_TEMPLATE
        self._set_phase("outer_proposal", current_operation="llm_or_fallback")
        cand_template, note, source, proposal_reflection = await propose_template(
            inc_template, history, provider,
            market=self.task_config.get("market", "us"),
            portfolio_mode=self._portfolio_mode(),
            direction=self._signal_direction(),
            direction_policy=self._direction_policy(),
            trace_context={
                "experiment_id": self.exp_id,
                "outer_step_no": step_no,
                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                "miner_version_id": incumbent.id,
            },
        )

        async with SessionLocal() as s:
            cand = MinerVersion(
                experiment_id=self.exp_id,
                version_no=await self._next_version_no(),
                parent_id=incumbent.id,
                harness_spec=cand_template, status="candidate",
                proposal_note=f"[{source}] {note}",
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                reflection={"proposal": proposal_reflection},
                context_fingerprint=str(
                    proposal_reflection.get(
                        "history_context_fingerprint",
                        "",
                    )
                ),
            )
            s.add(cand)
            await s.commit()
            await s.refresh(cand)

        await self.log(f"[V2] 外层步 {step_no}: 候选 v{cand.version_no} [{source}] {note[:100]}")

        # ---- multi-seed 内层挖掘 ----
        budget = int(cfg["inner_budget_per_outer_step"])
        n_seeds = int(cfg["n_seeds_per_candidate"])
        # Freeze one common history snapshot before either arm runs.  Every
        # seed sees this same baseline plus only its own newly-created nodes,
        # so seed estimates do not learn from one another and candidate versus
        # incumbent comparisons remain attributable to the template change.
        feedback_baseline = await self._feedback_baseline_v2(
            [task["name"] for task in cfg["tasks"]],
        )
        cand_seed_results = []
        for seed in range(n_seeds):
            self._set_phase(
                "candidate_mining",
                progress=True,
                current_operation="seed",
                current_seed=seed,
                current_budget_index=0,
                current_budget_total=budget,
            )
            seed_result = await self._mining_session_v2(
                cand,
                step_no,
                budget,
                cfg,
                seed,
                feedback_baseline,
            )
            cand_seed_results.append(seed_result)
            await self.log(
                f"[V2]   seed {seed+1}/{n_seeds} "
                f"meta={seed_result['score']:.4f} "
                f"pass={seed_result['summary']['pass_rate']:.1%}"
            )

        cand_scores = [row["score"] for row in cand_seed_results]
        cand_mean = st.mean(cand_scores) if cand_scores else 0.0
        cand_std = st.stdev(cand_scores) if len(cand_scores) >= 2 else 0.0
        cand_report = combine_seed_feedback(cand_seed_results)

        # ---- 在位者重测 ----
        remeasure_every = int(cfg["incumbent_remeasure_every"])
        remeasure_budget = int(cfg.get("incumbent_remeasure_budget", budget))
        inc_seed_results = []
        if incumbent.meta_score is None or step_no % remeasure_every == 0:
            for seed in range(n_seeds):
                self._set_phase(
                    "incumbent_remeasure",
                    progress=True,
                    current_operation="seed",
                    current_seed=seed + 1000,
                    current_budget_index=0,
                    current_budget_total=remeasure_budget,
                )
                inc_seed_result = await self._mining_session_v2(
                    incumbent,
                    step_no,
                    remeasure_budget,
                    cfg,
                    seed + 1000,
                    feedback_baseline,
                )
                inc_seed_results.append(inc_seed_result)
            inc_scores = [row["score"] for row in inc_seed_results]
            inc_mean = st.mean(inc_scores) if inc_scores else 0.0
            inc_std = st.stdev(inc_scores) if len(inc_scores) >= 2 else 0.0
            incumbent = await self._update_score(incumbent.id, inc_mean)
            inc_report = combine_seed_feedback(inc_seed_results)
            await self.log(f"[V2]   在位重测: mean={inc_mean:.4f} std={inc_std:.4f} (n={len(inc_scores)})")
        else:
            inc_scores = []
            inc_mean = incumbent.meta_score or 0.0
            inc_std = 0.0
            inc_report = dict(incumbent.feedback_summary or {})

        # ---- 同协议单边统计门 ----
        test = _one_sided_score_test(
            cand_scores,
            inc_scores,
            reference_mean=inc_mean,
        )
        p_value = test["p_value"]
        p_threshold = float(cfg["outer_accept_p_value"])
        gate_non_degrading = float(
            cand_report.get("gate_score_mean") or 0.0
        ) + 1e-4 >= float(
            inc_report.get("gate_score_mean") or 0.0
        )
        pass_rate_non_degrading = float(
            cand_report.get("pass_rate") or 0.0
        ) + 1e-9 >= float(
            inc_report.get("pass_rate") or 0.0
        )
        accepted = bool(
            len(cand_scores) == n_seeds
            and len(cand_scores) >= 2
            and cand_mean > inc_mean
            and p_value < p_threshold
            and gate_non_degrading
            and pass_rate_non_degrading
        )
        comparison = compare_feedback_reports(cand_report, inc_report)
        outcome_reflection, reflection_source = await reflect_on_outcome(
            proposal_reflection=proposal_reflection,
            candidate_report=cand_report,
            incumbent_report=inc_report,
            comparison=comparison,
            accepted=accepted,
            p_value=p_value,
            provider=provider,
            trace_context={
                "experiment_id": self.exp_id,
                "outer_step_no": step_no,
                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                "miner_version_id": cand.id,
            },
        )

        verdict = f"接受 ✓ p={p_value:.4f}" if accepted else f"拒绝 ✗ p={p_value:.4f}"

        self._set_phase(
            "outer_decision",
            progress=True,
            current_operation="persist_verdict",
            current_task=None,
        )
        async with SessionLocal() as s:
            s.add(OuterStep(
                experiment_id=self.exp_id,
                step_no=step_no, candidate_id=cand.id, incumbent_id=incumbent.id,
                candidate_score=cand_mean, incumbent_score=inc_mean, accepted=accepted,
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                context_fingerprint=str(
                    comparison.get("comparison_fingerprint") or ""
                ),
                detail={
                    "protocol_version": EVALUATION_PROTOCOL_VERSION,
                    "note": note, "source": source, "budget": budget, "n_seeds": n_seeds,
                    "cand_scores": cand_scores, "cand_std": cand_std,
                    "inc_scores": inc_scores, "inc_std": inc_std,
                    "p_value": p_value, "test": test, "mode": "v2",
                    "admission_safety": {
                        "gate_score_non_degrading": gate_non_degrading,
                        "pass_rate_non_degrading": pass_rate_non_degrading,
                    },
                    "candidate_report": cand_report,
                    "incumbent_report": inc_report,
                    "comparison": comparison,
                    "proposal_reflection": proposal_reflection,
                    "outcome_reflection": outcome_reflection,
                    "reflection_source": reflection_source,
                },
            ))
            cand_db = await s.get(MinerVersion, cand.id)
            cand_db.meta_score = cand_mean
            cand_db.feedback_summary = cand_report
            cand_db.reflection = {
                "proposal": proposal_reflection,
                "outcome": outcome_reflection,
            }
            cand_db.context_fingerprint = str(
                comparison.get("comparison_fingerprint") or ""
            )
            if inc_seed_results:
                inc_db_for_report = await s.get(
                    MinerVersion,
                    incumbent.id,
                )
                inc_db_for_report.feedback_summary = inc_report
            if accepted:
                cand_db.status = "incumbent"
                inc_db = await s.get(MinerVersion, incumbent.id)
                inc_db.status = "rejected" if inc_db.status == "candidate" else "superseded"
            else:
                cand_db.status = "rejected"
            await s.commit()

        await self.log(
            f"[V2] 外层步 {step_no}: "
            f"learn cand {cand_mean:.4f}±{cand_std:.3f} "
            f"vs inc {inc_mean:.4f}±{inc_std:.3f}; "
            f"gate_safe={gate_non_degrading} "
            f"pass_safe={pass_rate_non_degrading} -> {verdict}"
        )
        if accepted:
            async with SessionLocal() as s:
                incumbent = await s.get(MinerVersion, cand.id)
        return incumbent, cfg

    async def _mining_session_v2(
        self,
        miner: MinerVersion,
        step_no: int,
        budget: int,
        cfg: dict,
        seed: int,
        feedback_baseline: dict[str, list[dict]] | None = None,
    ) -> dict:
        """Run one seed and return score plus its auditable feedback report."""
        template = miner.harness_spec if isinstance(miner.harness_spec, dict) else DEFAULT_MINER_TEMPLATE
        provider = await self._provider("inner_provider")
        tasks = cfg["tasks"]
        task_best_scores: dict[str, float] = {}
        session_envelopes: list[dict] = []

        # 固定种子确保可复现
        rng = random.Random(seed * 10000 + miner.id)
        session_start_node_id = await self._max_node_id()

        for i in range(budget):
            if not self.running:
                break
            task = tasks[i % len(tasks)]
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                progress=True,
                current_task=task["name"],
                current_operation="context_lookup",
                current_seed=seed,
                current_budget_index=i + 1,
                current_budget_total=budget,
            )
            session_nodes = await self._feedback_nodes_v2(
                miner.id,
                task["name"],
                seed=seed,
                min_node_id=session_start_node_id,
            )
            feedback_nodes = self._merge_feedback_nodes(
                (feedback_baseline or {}).get(task["name"], []),
                session_nodes,
            )
            base_node = max(
                (
                    node
                    for node in feedback_nodes
                    if node.get("status") == "ok"
                ),
                key=lambda node: float(node.get("public_score") or 0.0),
                default=None,
            )

            # 操作选择: 根据 draft_strategy 中的指令决定 draft/improve 概率
            improve_bias = 0.6
            draft_strategy = template.get("draft_strategy", "")
            if "优先" in draft_strategy and "改进" not in draft_strategy:
                improve_bias = 0.4  # 偏探索
            elif "改进" in draft_strategy:
                improve_bias = 0.7  # 偏改进

            op = (
                "improve"
                if (base_node and rng.random() < improve_bias)
                else "draft"
            )
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                current_operation=f"proposal:{op}",
            )
            expr, hypo, source, proposal_meta = await propose(
                template, op, task, feedback_nodes, provider,
                fields=get_dsl_fields(self.task_config.get("market")),
                trace_context={
                    "experiment_id": self.exp_id,
                    "outer_step_no": step_no,
                    "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                    "miner_version_id": miner.id,
                    "task_name": task["name"],
                    "seed": seed,
                    "budget_index": i + 1,
                },
                rng=rng,
            )

            node = Node(
                experiment_id=self.exp_id,
                miner_version_id=miner.id, outer_step_no=step_no,
                parent_id=base_node["id"] if (op == "improve" and base_node) else None,
                op=op, expression=expr, hypothesis=hypo, source=source, task_name=task["name"],
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                seed=seed,
                proposal_meta=proposal_meta,
            )
            try:
                self._set_phase(
                    self.status.get("phase") or "candidate_mining",
                    current_operation="factor_evaluation",
                )
                metrics = await self._run_blocking_with_heartbeat(
                    evaluate, expr, task["universe_n"], task["horizon"],
                    task.get("mode", DEFAULT_PORTFOLIO_MODE), task.get("direction", 1),
                    self._panel_glob(), task.get("cost_bps", 15),
                    self.task_config.get("market", "us"),
                    self.task_config.get("evaluation_config"),
                    task.get(
                        "direction_policy",
                        self._direction_policy(),
                    ),
                )
                node.status = "ok"
                node.public_metrics = {
                    **metrics["public"],
                    "discovery": metrics["discovery"],
                    "protocol_version": metrics["protocol_version"],
                    "evaluation_runtime": metrics.get("runtime") or {},
                }
                node.gate_metrics = metrics["gate"]
                node.public_score = metrics["discovery"].get("score") or 0.0
                task_best_scores[task["name"]] = max(
                    task_best_scores.get(task["name"], 0.0),
                    node.public_score,
                )
            except Exception as e:
                node.status = "error"
                node.error = str(e)[:500]

            async with SessionLocal() as s:
                self._set_phase(
                    self.status.get("phase") or "candidate_mining",
                    current_operation="persist_trial",
                )
                s.add(node)
                await s.flush()
                envelope = build_feedback_envelope(
                    node_id=node.id,
                    parent_id=node.parent_id,
                    task_name=node.task_name,
                    expression=node.expression,
                    hypothesis=node.hypothesis,
                    source=node.source,
                    status=node.status,
                    error=node.error,
                    public_score=node.public_score,
                    public_metrics=node.public_metrics,
                    evaluation_protocol=node.evaluation_protocol,
                    market=self.task_config.get("market", "us"),
                    portfolio_mode=self._portfolio_mode(),
                    direction=self._signal_direction(),
                    proposal_meta=node.proposal_meta,
                )
                node.feedback_summary = envelope
                s.add(Trial(
                    experiment_id=self.exp_id,
                    expression_hash=normalize_hash(expr) if node.status == "ok" else "invalid",
                    layer="INNER_PUBLIC+META_TRAIN", task_name=task["name"],
                    evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                    statistic={
                        "public_score": node.public_score,
                        "learning_score": node.public_score,
                        "gate_score": (
                            node.public_metrics.get("discovery") or {}
                        ).get("gate_score", 0.0),
                        "selected_direction": (
                            node.public_metrics.get("discovery") or {}
                        ).get("selected_direction"),
                        "direction_policy": (
                            node.public_metrics.get("discovery") or {}
                        ).get("direction_policy"),
                        "score_semantics": (
                            node.public_metrics.get("discovery") or {}
                        ).get("score_semantics"),
                        "evaluation_runtime": (
                            node.public_metrics.get("evaluation_runtime") or {}
                        ),
                        "seed": seed,
                        "protocol_version": EVALUATION_PROTOCOL_VERSION,
                        "feedback_fingerprint": envelope[
                            "feedback_fingerprint"
                        ],
                    },
                ))
                await s.commit()
                await s.refresh(node)
            session_envelopes.append(envelope)

            self.status["inner_evals"] = self.status.get("inner_evals", 0) + 1
            self._touch_progress(
                current_operation="register_factor" if node.status == "ok" else "candidate_failed",
            )
            if node.status == "ok":
                min_icir = float(template.get("min_public_icir", 0.25))
                await self._maybe_register_factor_v2(node, min_icir)
                if i % 10 == 0:
                    await self.log(
                        f"[V2 s{seed}]  内层[{task['name']}] {op}/{source} "
                        f"learn={node.public_score:.3f} "
                        f"gate={float((node.public_metrics.get('discovery') or {}).get('gate_score') or 0.0):.3f} "
                        f"dir={int((node.public_metrics.get('discovery') or {}).get('selected_direction') or task.get('direction', 1)):+d} "
                        f"{expr[:60]}",
                        "debug",
                    )

        scores = list(task_best_scores.values())
        score = round(sum(scores) / len(scores), 4) if scores else 0.0
        summary = combine_seed_feedback([{
            "seed": seed,
            "score": score,
            "task_best_scores": task_best_scores,
            "envelopes": session_envelopes,
        }])
        return {
            "seed": seed,
            "score": score,
            "task_best_scores": task_best_scores,
            "envelopes": session_envelopes,
            "summary": summary,
        }

    async def _maybe_register_factor_v2(self, node: Node, min_icir: float) -> None:
        pm = node.public_metrics
        discovery = pm.get("discovery") or {}
        if not discovery.get("passed") or (pm.get("icir") or 0) < min_icir:
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
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                lifecycle_stage="research_pass",
                provenance_status="valid_task_config",
                research_meta={
                    "policy_label": "NON_PIT_RESEARCH",
                    "market": self.task_config.get("market", "us"),
                    "portfolio_mode": self._portfolio_mode(),
                    "direction": int(
                        discovery.get(
                            "selected_direction",
                            self._signal_direction(),
                        )
                    ),
                    "preferred_direction": self._signal_direction(),
                    "direction_policy": discovery.get(
                        "direction_policy",
                        self._direction_policy(),
                    ),
                    "direction_selection": discovery.get(
                        "direction_selection",
                        {},
                    ),
                    "direction_trials_multiplier": int(
                        (
                            discovery.get("direction_selection") or {}
                        ).get("trials_multiplier", 1)
                    ),
                },
                fingerprint={
                    "miner_version_id": node.miner_version_id, "outer_step": node.outer_step_no,
                    "source": node.source, **expression_fingerprint(node.expression),
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
                .where(
                    MinerVersion.status == "incumbent",
                    MinerVersion.experiment_id == self.exp_id,
                    MinerVersion.evaluation_protocol
                    == EVALUATION_PROTOCOL_VERSION,
                )
                .order_by(MinerVersion.id.desc())
            )
            required_template_keys = {
                "system_prompt", "draft_strategy", "improve_strategy",
                "context_strategy", "context_policy",
                "diversity_instruction", "scoring_weights",
                "dsl_exploration_templates",
            }
            if inc and isinstance(inc.harness_spec, dict) and required_template_keys.issubset(inc.harness_spec):
                return inc

            # An experiment may have been bootstrapped with the legacy V1
            # HarnessSpec before being resumed as V2. Keep that record and
            # create an explicit MinerTemplate baseline for the V2 lineage.
            if inc:
                inc.status = "superseded"
                max_version = await s.scalar(
                    select(func.max(MinerVersion.version_no)).where(
                        MinerVersion.experiment_id == self.exp_id
                    )
                )
                next_version = (max_version if max_version is not None else -1) + 1
                inc = MinerVersion(
                    experiment_id=self.exp_id, version_no=next_version,
                    parent_id=inc.id,
                    harness_spec=deepcopy(DEFAULT_MINER_TEMPLATE),
                    status="incumbent",
                    proposal_note="V2 MinerTemplate 基线；保留旧 V1 基线及其历史数据",
                    evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                    reflection={
                        "migration": (
                            "模板结构升级；旧协议数据保留但不进入当前上下文"
                        )
                    },
                )
                s.add(inc)
                await s.commit()
                await s.refresh(inc)
                return inc
            max_version = await s.scalar(
                select(func.max(MinerVersion.version_no)).where(
                    MinerVersion.experiment_id == self.exp_id
                )
            )
            latest_protocol_version = await s.scalar(
                select(MinerVersion)
                .where(
                    MinerVersion.experiment_id == self.exp_id,
                    MinerVersion.evaluation_protocol
                    == EVALUATION_PROTOCOL_VERSION,
                )
                .order_by(MinerVersion.id.desc())
                .limit(1)
            )
            inc = MinerVersion(
                experiment_id=self.exp_id,
                version_no=(max_version if max_version is not None else -1) + 1,
                parent_id=(
                    latest_protocol_version.id
                    if latest_protocol_version is not None
                    else None
                ),
                harness_spec=deepcopy(DEFAULT_MINER_TEMPLATE),
                status="incumbent",
                proposal_note=(
                    f"{EVALUATION_PROTOCOL_VERSION} MinerTemplate 基线；"
                    "历史协议完整保留但不参与比较"
                ),
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                reflection={
                    "migration": "新评价协议独立基线，禁止继承旧 meta-score"
                },
            )
            s.add(inc)
            await s.commit()
            await s.refresh(inc)
            return inc

    async def _version_history_v2(self) -> list[dict]:
        async with SessionLocal() as s:
            rows = (await s.scalars(
                select(MinerVersion).where(
                    MinerVersion.experiment_id == self.exp_id,
                    MinerVersion.evaluation_protocol
                    == EVALUATION_PROTOCOL_VERSION,
                )
                .order_by(MinerVersion.id))).all()
            return [
                {"version_no": m.version_no, "meta_score": m.meta_score,
                 "status": m.status, "template_note": m.proposal_note,
                 "template": m.harness_spec,
                 "evaluation_protocol": m.evaluation_protocol,
                 "feedback_summary": m.feedback_summary,
                 "reflection": m.reflection,
                 "context_fingerprint": m.context_fingerprint}
                for m in rows
            ]

    async def _feedback_nodes_v2(
        self,
        miner_id: int | list[int] | None,
        task_name: str,
        *,
        seed: int | None = None,
        min_node_id: int | None = None,
    ) -> list[dict]:
        filters = [
            Node.experiment_id == self.exp_id,
            Node.task_name == task_name,
            Node.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
        ]
        if miner_id is not None:
            miner_ids = (
                [int(value) for value in miner_id]
                if isinstance(miner_id, list)
                else [int(miner_id)]
            )
            filters.append(Node.miner_version_id.in_(miner_ids))
        if seed is not None:
            filters.append(Node.seed == seed)
        if min_node_id is not None:
            filters.append(Node.id > min_node_id)
        async with SessionLocal() as s:
            top = (await s.scalars(
                select(Node)
                .where(*filters, Node.status == "ok")
                .order_by(Node.public_score.desc().nulls_last())
                .limit(12)
            )).all()
            recent = (await s.scalars(
                select(Node)
                .where(*filters)
                .order_by(Node.id.desc())
                .limit(24)
            )).all()
        rows: list[Node] = []
        seen: set[int] = set()
        for node in [*top, *recent]:
            if node.id in seen:
                continue
            seen.add(node.id)
            rows.append(node)
        return [
            {
                "id": node.id,
                "parent_id": node.parent_id,
                "expression": node.expression,
                "hypothesis": node.hypothesis,
                "status": node.status,
                "error": node.error,
                "source": node.source,
                "task_name": node.task_name,
                "evaluation_protocol": node.evaluation_protocol,
                "public_score": node.public_score or 0.0,
                "public_metrics": node.public_metrics,
                "proposal_meta": node.proposal_meta,
                "feedback_summary": node.feedback_summary,
            }
            for node in rows
        ]

    async def _feedback_baseline_v2(
        self,
        task_names: list[str],
    ) -> dict[str, list[dict]]:
        """Freeze all same-protocol pre-step history for every A/B seed."""
        return {
            task_name: await self._feedback_nodes_v2(
                None,
                task_name,
            )
            for task_name in task_names
        }

    @staticmethod
    def _merge_feedback_nodes(
        baseline: list[dict],
        session_nodes: list[dict],
    ) -> list[dict]:
        rows: list[dict] = []
        seen: set[int] = set()
        for row in [*session_nodes, *baseline]:
            node_id = int(row.get("id") or 0)
            if node_id and node_id in seen:
                continue
            if node_id:
                seen.add(node_id)
            rows.append(row)
        return rows

    async def _max_node_id(self) -> int:
        async with SessionLocal() as session:
            value = await session.scalar(
                select(func.max(Node.id)).where(
                    Node.experiment_id == self.exp_id
                )
            )
        return int(value or 0)

    async def _config_v2(self) -> dict:
        async with SessionLocal() as s:
            row = await s.get(Setting, "engine_config")
            cfg = {**DEFAULT_ENGINE_CONFIG_V2, **(row.value if row else {})}
            task_cfg = self.task_config.get("engine_config", {})
            local_tasks = task_cfg.get("tasks")
            cfg.update(task_cfg)
            cfg["tasks"] = resolve_engine_tasks(
                local_tasks or cfg.get("tasks", []),
                self.task_config.get("market", "us"),
                self._portfolio_mode(),
                self._signal_direction(),
                self._direction_policy(),
                preserve_declared_costs=bool(local_tasks),
            )
            return cfg

    # ================================================================
    # V1 主循环 (A组: 保持兼容, 原封不动)
    # ================================================================

    async def _run(self) -> None:
        try:
            self._set_phase("validating_experiment", progress=True)
            async with SessionLocal() as s:
                exp = await s.get(Experiment, self.exp_id)
                if not exp:
                    raise RuntimeError(f"活动实验 {self.exp_id} 不存在")
                if exp.status == "archived":
                    raise RuntimeError(f"实验「{exp.name}」已归档, 请先切换到开放实验")
            self.status["experiment_id"] = self.exp_id
            await self.log(f"引擎启动: 实验[{exp.name}] 加载数据面板...")
            self._set_phase("loading_panel", progress=True)
            panel = PanelStore.get(self._panel_glob(), self.task_config.get("market", "us"))
            await asyncio.to_thread(panel.ensure_loaded)
            await self.log(
                f"面板就绪: {panel.summary()['rows']} 行 · {self.task_config.get('market', 'configured')}"
            )
            self._set_phase("initializing_miner", progress=True)
            incumbent = await self._ensure_incumbent()
            self.status["state"] = "running"
            cfg = await self._config()

            while self.running:
                step_no = await self._next_step_no()
                self.status["outer_step"] = step_no
                self._set_phase(
                    "outer_step",
                    progress=True,
                    current_operation="prepare",
                    current_seed=None,
                    current_budget_index=None,
                    current_budget_total=None,
                )
                incumbent, cfg = await self._outer_step(step_no, incumbent, cfg)
        except Exception:
            self._set_phase("failed")
            await self.log(f"引擎异常退出:\n{traceback.format_exc()}", "error")
        finally:
            self.running = False
            self.status.update({"state": "stopped", "stopped_at": utc_now()})
            if self.status.get("phase") != "failed":
                self._set_phase("stopped", progress=True)

    async def _outer_step(self, step_no: int, incumbent: MinerVersion, cfg: dict):
        self._set_phase(
            "outer_proposal",
            progress=True,
            current_operation="llm_or_fallback",
            current_task=None,
        )
        provider = await self._provider("outer_provider")
        history = await self._version_history()
        cand_spec, note, source = await propose_spec(incumbent.harness_spec, history, provider)
        next_version = await self._next_version_no()
        async with SessionLocal() as s:
            cand = MinerVersion(
                experiment_id=self.exp_id,
                version_no=next_version, parent_id=incumbent.id,
                harness_spec=cand_spec, status="candidate", proposal_note=f"[{source}] {note}",
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
            )
            s.add(cand)
            await s.commit()
            await s.refresh(cand)
        await self.log(f"外层步 {step_no}: 候选 v{cand.version_no} {note} ({source})")

        budget = int(cfg["inner_budget_per_outer_step"])
        self._set_phase(
            "candidate_mining",
            progress=True,
            current_operation="session",
            current_budget_index=0,
            current_budget_total=budget,
        )
        cand_score = await self._mining_session(cand, step_no, budget, cfg)

        if incumbent.meta_score is None or step_no % int(cfg["incumbent_remeasure_every"]) == 0:
            self._set_phase(
                "incumbent_remeasure",
                progress=True,
                current_operation="session",
                current_budget_index=0,
                current_budget_total=budget,
            )
            inc_score = await self._mining_session(incumbent, step_no, budget, cfg)
            incumbent = await self._update_score(incumbent.id, inc_score)
        inc_score = incumbent.meta_score or 0.0

        accepted = cand_score > inc_score + float(cfg["outer_accept_epsilon"])
        self._set_phase(
            "outer_decision",
            progress=True,
            current_operation="persist_verdict",
            current_task=None,
        )
        async with SessionLocal() as s:
            s.add(OuterStep(
                experiment_id=self.exp_id,
                step_no=step_no, candidate_id=cand.id, incumbent_id=incumbent.id,
                candidate_score=cand_score, incumbent_score=inc_score, accepted=accepted,
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                detail={
                    "protocol_version": EVALUATION_PROTOCOL_VERSION,
                    "note": note,
                    "source": source,
                    "budget": budget,
                    "mode": "v1",
                },
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
        task_best_scores: dict[str, float] = {}

        for i in range(budget):
            if not self.running:
                break
            task = tasks[i % len(tasks)]
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                progress=True,
                current_task=task["name"],
                current_operation="context_lookup",
                current_budget_index=i + 1,
                current_budget_total=budget,
            )
            top_nodes = await self._top_nodes(miner.id, task["name"])
            op = "improve" if (top_nodes and random.random() < float(spec.get("improve_bias", 0.6))) else "draft"
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                current_operation=f"proposal:{op}",
            )
            expr, hypo, source, proposal_meta = await propose(
                spec, op, task, top_nodes, provider,
                fields=get_dsl_fields(self.task_config.get("market")),
                trace_context={
                    "experiment_id": self.exp_id,
                    "outer_step_no": step_no,
                    "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                    "miner_version_id": miner.id,
                    "task_name": task["name"],
                },
            )

            node = Node(
                experiment_id=self.exp_id,
                miner_version_id=miner.id, outer_step_no=step_no,
                parent_id=top_nodes[0]["id"] if (op == "improve" and top_nodes) else None,
                op=op, expression=expr, hypothesis=hypo, source=source, task_name=task["name"],
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                proposal_meta=proposal_meta,
            )
            try:
                self._set_phase(
                    self.status.get("phase") or "candidate_mining",
                    current_operation="factor_evaluation",
                )
                metrics = await self._run_blocking_with_heartbeat(
                    evaluate, expr, task["universe_n"], task["horizon"],
                    task.get("mode", DEFAULT_PORTFOLIO_MODE), task.get("direction", 1),
                    self._panel_glob(), task.get("cost_bps", 15),
                    self.task_config.get("market", "us"),
                    self.task_config.get("evaluation_config"),
                    task.get(
                        "direction_policy",
                        self._direction_policy(),
                    ),
                )
                node.status = "ok"
                node.public_metrics = {
                    **metrics["public"],
                    "discovery": metrics["discovery"],
                    "protocol_version": metrics["protocol_version"],
                    "evaluation_runtime": metrics.get("runtime") or {},
                }
                node.gate_metrics = metrics["gate"]
                node.public_score = metrics["discovery"].get("score") or 0.0
                task_best_scores[task["name"]] = max(
                    task_best_scores.get(task["name"], 0.0),
                    node.public_score,
                )
            except Exception as e:
                node.status = "error"
                node.error = str(e)[:500]
            async with SessionLocal() as s:
                self._set_phase(
                    self.status.get("phase") or "candidate_mining",
                    current_operation="persist_trial",
                )
                s.add(node)
                await s.flush()
                node.feedback_summary = build_feedback_envelope(
                    node_id=node.id,
                    parent_id=node.parent_id,
                    task_name=node.task_name,
                    expression=node.expression,
                    hypothesis=node.hypothesis,
                    source=node.source,
                    status=node.status,
                    error=node.error,
                    public_score=node.public_score,
                    public_metrics=node.public_metrics,
                    evaluation_protocol=node.evaluation_protocol,
                    market=self.task_config.get("market", "us"),
                    portfolio_mode=self._portfolio_mode(),
                    direction=self._signal_direction(),
                    proposal_meta=node.proposal_meta,
                )
                s.add(Trial(
                    experiment_id=self.exp_id,
                    expression_hash=normalize_hash(expr) if node.status == "ok" else "invalid",
                    layer="INNER_PUBLIC+META_TRAIN", task_name=task["name"],
                    evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                    statistic={
                        "public_score": node.public_score,
                        "learning_score": node.public_score,
                        "gate_score": (
                            node.public_metrics.get("discovery") or {}
                        ).get("gate_score", 0.0),
                        "selected_direction": (
                            node.public_metrics.get("discovery") or {}
                        ).get("selected_direction"),
                        "direction_policy": (
                            node.public_metrics.get("discovery") or {}
                        ).get("direction_policy"),
                        "score_semantics": (
                            node.public_metrics.get("discovery") or {}
                        ).get("score_semantics"),
                        "evaluation_runtime": (
                            node.public_metrics.get("evaluation_runtime") or {}
                        ),
                        "protocol_version": EVALUATION_PROTOCOL_VERSION,
                        "feedback_fingerprint": node.feedback_summary[
                            "feedback_fingerprint"
                        ],
                    },
                ))
                await s.commit()
                await s.refresh(node)
            self.status["inner_evals"] = self.status.get("inner_evals", 0) + 1
            self._touch_progress(
                current_operation="register_factor" if node.status == "ok" else "candidate_failed",
            )
            if node.status == "ok":
                await self._maybe_register_factor(node, spec)
                await self.log(
                    f"  内层[{task['name']}] {op}/{source} "
                    f"learn={node.public_score:.3f} "
                    f"gate={float((node.public_metrics.get('discovery') or {}).get('gate_score') or 0.0):.3f} "
                    f"dir={int((node.public_metrics.get('discovery') or {}).get('selected_direction') or task.get('direction', 1)):+d} "
                    f"{expr[:80]}",
                    "debug",
                )
            else:
                await self.log(f"  内层[{task['name']}] {op} 失败: {node.error[:80]}", "debug")

        scores = list(task_best_scores.values())
        return round(sum(scores) / len(scores), 4) if scores else 0.0

    async def _maybe_register_factor(self, node: Node, spec: dict) -> None:
        pm = node.public_metrics
        discovery = pm.get("discovery") or {}
        if not discovery.get("passed") or (pm.get("icir") or 0) < float(spec.get("min_public_icir", 0.25)):
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
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                lifecycle_stage="research_pass",
                provenance_status="valid_task_config",
                research_meta={
                    "policy_label": "NON_PIT_RESEARCH",
                    "market": self.task_config.get("market", "us"),
                    "portfolio_mode": self._portfolio_mode(),
                    "direction": int(
                        discovery.get(
                            "selected_direction",
                            self._signal_direction(),
                        )
                    ),
                    "preferred_direction": self._signal_direction(),
                    "direction_policy": discovery.get(
                        "direction_policy",
                        self._direction_policy(),
                    ),
                    "direction_selection": discovery.get(
                        "direction_selection",
                        {},
                    ),
                    "direction_trials_multiplier": int(
                        (
                            discovery.get("direction_selection") or {}
                        ).get("trials_multiplier", 1)
                    ),
                },
                fingerprint={
                    "miner_version_id": node.miner_version_id, "outer_step": node.outer_step_no,
                    "source": node.source, **expression_fingerprint(node.expression),
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
                .where(
                    MinerVersion.status == "incumbent",
                    MinerVersion.experiment_id == self.exp_id,
                    MinerVersion.evaluation_protocol
                    == EVALUATION_PROTOCOL_VERSION,
                )
                .order_by(MinerVersion.id.desc())
            )
            if inc:
                return inc
            inc = MinerVersion(
                experiment_id=self.exp_id,
                version_no=await self._next_version_no(),
                harness_spec=DEFAULT_HARNESS_SPEC,
                status="incumbent",
                proposal_note=f"{EVALUATION_PROTOCOL_VERSION} Miner_0 基线",
                evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
            )
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

    async def _next_version_no(self) -> int:
        async with SessionLocal() as s:
            last = await s.scalar(
                select(func.max(MinerVersion.version_no)).where(
                    MinerVersion.experiment_id == self.exp_id
                )
            )
            return (last if last is not None else -1) + 1

    async def _version_history(self) -> list[dict]:
        async with SessionLocal() as s:
            rows = (await s.scalars(
                select(MinerVersion).where(
                    MinerVersion.experiment_id == self.exp_id,
                    MinerVersion.evaluation_protocol
                    == EVALUATION_PROTOCOL_VERSION,
                )
                .order_by(MinerVersion.id))).all()
            return [
                {"version_no": m.version_no, "harness_spec": m.harness_spec,
                 "meta_score": m.meta_score, "status": m.status,
                 "evaluation_protocol": m.evaluation_protocol}
                for m in rows
            ]

    async def _top_nodes(self, miner_id: int, task_name: str, k: int = 10) -> list[dict]:
        async with SessionLocal() as s:
            rows = (await s.scalars(
                select(Node)
                .where(
                    Node.miner_version_id == miner_id,
                    Node.task_name == task_name,
                    Node.status == "ok",
                    Node.evaluation_protocol
                    == EVALUATION_PROTOCOL_VERSION,
                )
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
            cfg = {**DEFAULT_ENGINE_CONFIG, **(row.value if row else {})}
            task_cfg = self.task_config.get("engine_config", {})
            local_tasks = task_cfg.get("tasks")
            cfg.update(task_cfg)
            cfg["tasks"] = resolve_engine_tasks(
                local_tasks or cfg.get("tasks", []),
                self.task_config.get("market", "us"),
                self._portfolio_mode(),
                self._signal_direction(),
                self._direction_policy(),
                preserve_declared_costs=bool(local_tasks),
            )
            return cfg


class EngineManager:
    """单进程多研究任务调度器；每个任务拥有独立 worker 和日志缓冲。"""

    _instance = None

    def __init__(self) -> None:
        self.workers: dict[int, Engine] = {}

    @classmethod
    def get(cls) -> "EngineManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def running(self) -> bool:
        return any(worker.running for worker in self.workers.values())

    def worker(self, experiment_id: int) -> Engine | None:
        return self.workers.get(experiment_id)

    def status_for(self, experiment_id: int, include_logs: bool = True) -> dict:
        worker = self.workers.get(experiment_id)
        if worker:
            return worker.diagnostics(include_logs=include_logs)
        now = utc_now()
        result = {
            "state": "stopped",
            "phase": "not_started",
            "phase_started_at": None,
            "outer_step": 0,
            "inner_evals": 0,
            "experiment_id": experiment_id,
            "mode": None,
            "running": False,
            "task_created": False,
            "task_done": False,
            "task_cancelled": False,
            "task_exception": None,
            "heartbeat_age_seconds": None,
            "heartbeat_stale": False,
            "uptime_seconds": 0.0,
            "started_at": None,
            "stopped_at": None,
            "last_heartbeat_at": None,
            "last_progress_at": None,
            "last_log_at": None,
            "last_error": None,
            "current_task": None,
            "current_operation": None,
            "current_seed": None,
            "current_budget_index": None,
            "current_budget_total": None,
            "log_counts": {},
            "task_config": {},
            "observed_at": now,
        }
        if include_logs:
            result["logs"] = []
        return result

    def all_status(self, include_logs: bool = False) -> list[dict]:
        return [
            self.status_for(eid, include_logs=include_logs)
            for eid in sorted(self.workers)
        ]

    async def start(self, mode: str = "v2", experiment_id: int | None = None) -> dict:
        eid = experiment_id or await get_active_experiment_id()
        worker = self.workers.get(eid)
        if worker and worker.running:
            return {"ok": False, "msg": f"研究任务 {eid} 已在运行"}
        worker = Engine()
        self.workers[eid] = worker
        return await worker.start(mode, experiment_id=eid)

    async def stop(self, experiment_id: int | None = None) -> dict:
        if experiment_id is not None:
            worker = self.workers.get(experiment_id)
            return await worker.stop() if worker else {"ok": True, "msg": "任务未运行"}
        stopped = 0
        for worker in list(self.workers.values()):
            if worker.running:
                await worker.stop()
                stopped += 1
        return {"ok": True, "msg": f"已停止 {stopped} 个研究任务，已保留已提交研究数据"}


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Numerical Recipes continued fraction for regularized incomplete beta."""
    max_iterations = 240
    epsilon = 3e-14
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, max_iterations + 1):
        twice = 2 * iteration
        numerator = (
            iteration
            * (b - iteration)
            * x
            / ((qam + twice) * (a + twice))
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        numerator = (
            -(a + iteration)
            * (qab + iteration)
            * x
            / ((a + twice) * (qap + twice))
        )
        d = 1.0 + numerator * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + numerator / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) <= epsilon:
            return result
    raise ArithmeticError("incomplete beta continued fraction did not converge")


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        value = front * _beta_continued_fraction(a, b, x) / a
    else:
        value = 1.0 - (
            front
            * _beta_continued_fraction(b, a, 1.0 - x)
            / b
        )
    return max(0.0, min(1.0, value))


def _student_t_survival(t_stat: float, degrees_of_freedom: float) -> float:
    """One-sided P(T >= t) for Student's t; exact up to numeric tolerance."""
    if not math.isfinite(t_stat) or degrees_of_freedom <= 0:
        return 1.0
    x = degrees_of_freedom / (
        degrees_of_freedom + t_stat * t_stat
    )
    right_tail = 0.5 * _regularized_beta(
        x,
        degrees_of_freedom / 2.0,
        0.5,
    )
    return right_tail if t_stat >= 0 else 1.0 - right_tail


def _one_sided_score_test(
    candidate_scores: list[float],
    incumbent_scores: list[float],
    *,
    reference_mean: float,
) -> dict:
    """One-sided Student t test with explicit Welch/fixed-baseline mode."""
    if len(candidate_scores) < 2:
        return {
            "type": "insufficient_seeds",
            "t_stat": None,
            "p_value": 1.0,
            "candidate_n": len(candidate_scores),
            "incumbent_n": len(incumbent_scores),
        }
    cand_mean = st.mean(candidate_scores)
    cand_var = st.variance(candidate_scores)
    if len(incumbent_scores) >= 2:
        inc_mean = st.mean(incumbent_scores)
        inc_var = st.variance(incumbent_scores)
        standard_error = math.sqrt(
            cand_var / len(candidate_scores)
            + inc_var / len(incumbent_scores)
        )
        first = cand_var / len(candidate_scores)
        second = inc_var / len(incumbent_scores)
        denominator = (
            first * first / (len(candidate_scores) - 1)
            + second * second / (len(incumbent_scores) - 1)
        )
        degrees_of_freedom = (
            (first + second) ** 2 / denominator
            if denominator > 1e-18
            else float(
                len(candidate_scores) + len(incumbent_scores) - 2
            )
        )
        test_type = "welch_one_sided_t"
    else:
        inc_mean = float(reference_mean)
        standard_error = math.sqrt(cand_var / len(candidate_scores))
        degrees_of_freedom = float(len(candidate_scores) - 1)
        test_type = "candidate_vs_frozen_incumbent_one_sided_t"
    difference = cand_mean - inc_mean
    if standard_error <= 1e-12:
        t_stat = None
        p_value = 0.0 if difference > 0 else 1.0
    else:
        t_stat = difference / standard_error
        p_value = (
            _student_t_survival(t_stat, degrees_of_freedom)
            if t_stat > 0
            else 1.0
        )
    return {
        "type": test_type,
        "candidate_n": len(candidate_scores),
        "incumbent_n": len(incumbent_scores),
        "candidate_mean": round(cand_mean, 6),
        "incumbent_mean": round(inc_mean, 6),
        "difference": round(difference, 6),
        "standard_error": round(standard_error, 6),
        "degrees_of_freedom": round(degrees_of_freedom, 6),
        "t_stat": round(t_stat, 6) if t_stat is not None else None,
        "p_value": round(max(0.0, min(1.0, p_value)), 8),
    }
