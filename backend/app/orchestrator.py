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
    SERVICE_ARCHITECTURE,
    SERVICE_INSTANCE,
    get_dsl_fields,
    resolve_engine_tasks,
    service_accepts_task,
)
from .data.panel import PanelStore
from .db import SessionLocal, get_active_experiment_id
from .blind_review import deterministic_code_review
from .dsl.engine import normalize_hash
from .eval.harness import evaluate
from .feedback import (
    build_feedback_envelope,
    combine_seed_feedback,
    compare_feedback_reports,
    enrich_feedback_with_factor_admission,
)
from .factors.diversity import (
    diversity_adjusted_score,
    mechanism_from_item,
    mechanisms_for_market,
    select_target_mechanism,
)
from .factors.return_path import (
    combined_training_signature,
    return_path_correlation,
)
from .factors.return_source_governance import (
    resolve_return_source_governance,
)
from .factors.semantics import audit_expression_semantics
from .factors.similarity import expression_fingerprint, expression_similarity
from .meta.agent import (
    propose_spec,
    propose_template,
    reflect_on_outcome,
    validate_template,
)
from .miner.agent import propose, propose_batch
from .models import (
    EngineEvent,
    Experiment,
    Factor,
    LLMCallAudit,
    MinerVersion,
    Node,
    OuterStep,
    Setting,
    Trial,
)
from .observability import redact_text, redact_value, utc_now
from .runtime_identity import runtime_identity
from .scientific_governor import propose_scientific_directive
from .search_pool import DEFAULT_SEARCH_ALGORITHMS, propose_search_seed


# Five experiments may be logically live at once, but full-panel Polars
# evaluations are the expensive shared resource.  Two concurrent evaluations
# retain pipeline overlap without multiplying memory pressure fivefold.
_EVALUATION_SEMAPHORE = asyncio.Semaphore(
    max(1, int(os.environ.get("FF_MAX_PARALLEL_EVALUATIONS", "2")))
)
_PANEL_LOAD_SEMAPHORE = asyncio.Semaphore(
    max(1, int(os.environ.get("FF_MAX_PARALLEL_PANEL_LOADS", "2")))
)


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
            "evaluation_queued": False,
            "evaluation_queue_seconds": None,
            "evaluation_queue_heartbeat_count": 0,
            "evaluation_started_at": None,
            "evaluation_elapsed_seconds": None,
            "evaluation_heartbeat_count": 0,
            "evaluation_soft_deadline_seconds": None,
            "evaluation_deadline_exceeded": False,
            "last_evaluation_duration_seconds": None,
            "llm_active": False,
            "llm_started_at": None,
            "llm_elapsed_seconds": None,
            "llm_heartbeat_count": 0,
            "last_llm_duration_seconds": None,
        }
        self._started_monotonic: float | None = None
        self._last_heartbeat_monotonic = time.monotonic()
        self.logbuf: deque[dict] = deque(maxlen=300)
        self._mode: str = "v1"  # "v1" 或 "v2"
        self.task_config: dict = {}
        self._runtime_identity: dict = runtime_identity()

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

    async def _run_llm_with_heartbeat(
        self,
        awaitable,
        *,
        phase: str,
        operation: str,
        **detail,
    ):
        """Expose slow provider requests as live work, not stale workers."""
        started = time.monotonic()
        heartbeat_count = 0
        task = asyncio.create_task(
            awaitable,
            name=f"research.llm.{self.exp_id}.{operation}",
        )
        self._set_phase(
            phase,
            progress=True,
            current_operation=operation,
            llm_active=True,
            llm_started_at=utc_now(),
            llm_elapsed_seconds=0.0,
            llm_heartbeat_count=0,
            **detail,
        )
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=15.0)
                elapsed = max(0.0, time.monotonic() - started)
                if done:
                    return task.result()
                heartbeat_count += 1
                self._set_phase(
                    phase,
                    current_operation=operation,
                    llm_active=True,
                    llm_elapsed_seconds=round(elapsed, 3),
                    llm_heartbeat_count=heartbeat_count,
                    **detail,
                )
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
        finally:
            elapsed = max(0.0, time.monotonic() - started)
            self._set_phase(
                phase,
                llm_active=False,
                llm_elapsed_seconds=round(elapsed, 3),
                llm_heartbeat_count=heartbeat_count,
                last_llm_duration_seconds=round(elapsed, 3),
            )

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

    async def _run_shared_evaluation(self, *args):
        """Acquire global evaluation capacity with truthful queue heartbeats."""
        queued_at = time.monotonic()
        queue_heartbeats = 0
        self._set_phase(
            self.status.get("phase") or "candidate_mining",
            current_operation="evaluation_queue",
            evaluation_queued=True,
            evaluation_queue_seconds=0.0,
        )
        acquired = False
        try:
            while not acquired:
                try:
                    await asyncio.wait_for(
                        _EVALUATION_SEMAPHORE.acquire(), timeout=15.0
                    )
                    acquired = True
                except TimeoutError:
                    queue_heartbeats += 1
                    self._set_phase(
                        self.status.get("phase") or "candidate_mining",
                        current_operation="evaluation_queue",
                        evaluation_queued=True,
                        evaluation_queue_seconds=round(
                            time.monotonic() - queued_at, 3
                        ),
                        evaluation_queue_heartbeat_count=queue_heartbeats,
                    )
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                current_operation="factor_evaluation",
                evaluation_queued=False,
                evaluation_queue_seconds=round(
                    time.monotonic() - queued_at, 3
                ),
                evaluation_queue_heartbeat_count=queue_heartbeats,
            )
            return await self._run_blocking_with_heartbeat(evaluate, *args)
        finally:
            if acquired:
                _EVALUATION_SEMAPHORE.release()

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

    def _proposal_mode(self) -> str:
        mode = str(self.task_config.get("proposal_mode") or "llm").strip().lower()
        if mode not in {"llm", "random", "search_pool"}:
            raise ValueError(
                "研究任务 proposal_mode 必须为 llm、random 或 search_pool"
            )
        return mode

    def _layer1_enabled(self) -> bool:
        return bool(self.task_config.get("layer1_enabled", False))

    def _layer2_enabled(self) -> bool:
        return bool(
            self.task_config.get(
                "layer2_enabled", self._proposal_mode() == "llm"
            )
        )

    def _layer3_enabled(self) -> bool:
        return bool(self.task_config.get("layer3_enabled", True))

    def _full_llm_architecture(self) -> bool:
        return bool(self.task_config.get("full_llm_architecture", False))

    def _scientific_governor_enabled(self) -> bool:
        return bool(
            self.task_config.get("scientific_governor_enabled", False)
        )

    def _search_algorithms(self) -> tuple[str, ...]:
        configured = self.task_config.get("search_algorithms")
        if configured is None:
            return DEFAULT_SEARCH_ALGORITHMS
        if not isinstance(configured, (list, tuple)):
            raise ValueError("研究任务 search_algorithms 必须为算法名称列表")
        algorithms = tuple(str(value).strip() for value in configured if str(value).strip())
        unknown = sorted(set(algorithms) - set(DEFAULT_SEARCH_ALGORITHMS))
        if not algorithms or unknown:
            raise ValueError(f"研究任务 search_algorithms 非法: {unknown or 'empty'}")
        return algorithms

    def _memory_mode(self) -> str:
        mode = str(self.task_config.get("memory_mode") or "adaptive").strip().lower()
        if mode not in {"adaptive", "cold"}:
            raise ValueError("研究任务 memory_mode 必须为 adaptive 或 cold")
        return mode

    def _target_factor_count(self) -> int:
        try:
            target = int(self.task_config.get("target_factor_count") or 0)
        except (TypeError, ValueError) as exc:
            raise ValueError("研究任务 target_factor_count 必须为非负整数") from exc
        if target < 0:
            raise ValueError("研究任务 target_factor_count 必须为非负整数")
        return target

    def _continuous_operation(self) -> bool:
        return bool(self.task_config.get("continuous_operation", False))

    def _candidate_evaluation_budget(self) -> int:
        try:
            budget = int(
                self.task_config.get("candidate_evaluation_budget") or 0
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "研究任务 candidate_evaluation_budget 必须为非负整数"
            ) from exc
        if budget < 0:
            raise ValueError(
                "研究任务 candidate_evaluation_budget 必须为非负整数"
            )
        return budget

    def _target_mechanisms(self) -> tuple[str, ...]:
        configured = self.task_config.get("target_mechanisms") or []
        if not isinstance(configured, (list, tuple)):
            raise ValueError("研究任务 target_mechanisms 必须为机制名称列表")
        allowed = set(
            mechanisms_for_market(self.task_config.get("market", "us"))
        )
        targets: list[str] = []
        for value in configured:
            mechanism = str(value).strip()
            if not mechanism or mechanism not in allowed:
                raise ValueError(
                    f"研究任务包含不可用于当前市场的收益机制: {mechanism or value}"
                )
            if mechanism not in targets:
                targets.append(mechanism)
        return tuple(targets)

    def _return_source_governance(self) -> dict:
        """Resolve opt-in governance without changing existing V4 runs."""
        return resolve_return_source_governance(
            self.task_config.get("return_source_governance")
        )

    def _target_family_for_attempt(
        self,
        feedback_nodes: list[dict],
        market: str,
        rng: random.Random,
        attempt_offset: int = 0,
    ) -> str:
        targets = self._target_mechanisms()
        if not targets:
            return select_target_mechanism(feedback_nodes, market, rng)
        # Persisted candidate count is restored at startup, so round-robin
        # targeting remains balanced and reproducible across service restarts.
        index = int(self.status.get("candidate_evaluations") or 0) + int(
            attempt_offset
        )
        return targets[index % len(targets)]

    def _mechanism_schedule(
        self,
        template: dict,
        market: str,
    ) -> tuple[str, ...]:
        allowed = tuple(self._target_mechanisms() or mechanisms_for_market(market))
        if not self._full_llm_architecture():
            return allowed
        directive = template.get("_scientific_governor_directive") or {}
        focus = [
            str(item)
            for item in directive.get("focus_mechanisms", [])
            if str(item) in allowed
        ]
        deprioritized = {
            str(item)
            for item in directive.get("deprioritize_mechanisms", [])
            if str(item) in allowed
        }
        focus = list(dict.fromkeys(focus))
        remaining = [
            item for item in allowed
            if item not in focus and item not in deprioritized
        ]
        tail = [item for item in allowed if item in deprioritized]
        try:
            exploration_share = float(directive.get("exploration_share", 0.5))
        except (TypeError, ValueError):
            exploration_share = 0.5
        focus_repeats = max(1, min(4, round((1.0 - exploration_share) * 5)))
        return tuple(focus * focus_repeats + remaining + tail) or allowed

    async def _factor_count(self) -> int:
        async with SessionLocal() as session:
            value = await session.scalar(
                select(func.count(Factor.id)).where(
                    Factor.experiment_id == self.exp_id,
                    Factor.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
                )
            )
        return int(value or 0)

    async def _candidate_evaluation_count(self) -> int:
        async with SessionLocal() as session:
            value = await session.scalar(
                select(func.count(Node.id)).where(
                    Node.experiment_id == self.exp_id,
                    Node.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
                )
            )
        return int(value or 0)

    def _dynamic_evaluation_overrides(self) -> dict:
        overrides = dict(self.task_config.get("evaluation_config") or {})
        governance = dict(self.task_config.get("overfit_governance") or {})
        if governance.get("dynamic_actual_trials"):
            actual = int(self.status.get("candidate_evaluations") or 0) + 1
            direction_multiplier = (
                2 if self._direction_policy() == "both_train_select" else 1
            )
            overrides["multiple_testing_trials"] = max(
                int(overrides.get("multiple_testing_trials") or 1),
                actual * direction_multiplier,
            )
        return overrides

    async def _llm_call_count(self) -> int:
        async with SessionLocal() as session:
            value = await session.scalar(
                select(func.count(LLMCallAudit.id)).where(
                    LLMCallAudit.experiment_id == self.exp_id,
                    LLMCallAudit.evaluation_protocol
                    == EVALUATION_PROTOCOL_VERSION,
                )
            )
        return int(value or 0)

    async def _stop_if_scientific_budget_reached(
        self,
        cfg: dict,
        *,
        next_step_no: int,
    ) -> bool:
        max_outer = int(cfg["max_outer_steps"])
        max_hours = float(cfg["max_runtime_hours"])
        max_llm = int(cfg["max_llm_calls"])
        llm_calls = await self._llm_call_count()
        elapsed = (
            max(0.0, time.monotonic() - self._started_monotonic)
            if self._started_monotonic is not None
            else 0.0
        )
        candidate_count = int(self.status.get("candidate_evaluations") or 0)
        candidate_budget = self._candidate_evaluation_budget()
        if self._continuous_operation():
            self.status.update({
                "budget_mode": "unlimited",
                "max_outer_steps": None,
                "max_runtime_hours": None,
                "max_llm_calls": None,
                "llm_calls": llm_calls,
                "scientific_budget_progress": {},
                "global_progress": None,
                "estimated_remaining_seconds": None,
                "candidate_evaluation_budget": 0,
                "candidate_evaluation_progress": None,
            })
            return False
        ratios = {
            "outer_steps": min(1.0, max(0, next_step_no - 1) / max_outer),
            "runtime": min(1.0, elapsed / (max_hours * 3600.0)),
            "llm_calls": min(1.0, llm_calls / max_llm),
        }
        if candidate_budget > 0:
            ratios["candidate_evaluations"] = min(
                1.0, candidate_count / candidate_budget
            )
        rate = candidate_count / elapsed if elapsed > 0 and candidate_count else 0.0
        eta = (
            max(0.0, (candidate_budget - candidate_count) / rate)
            if candidate_budget > candidate_count and rate > 0
            else 0.0 if candidate_budget > 0 else None
        )
        self.status.update({
            "max_outer_steps": max_outer,
            "max_runtime_hours": max_hours,
            "max_llm_calls": max_llm,
            "llm_calls": llm_calls,
            "scientific_budget_progress": {
                key: round(value, 6) for key, value in ratios.items()
            },
            "global_progress": round(max(ratios.values(), default=0.0), 6),
            "estimated_remaining_seconds": (
                round(eta, 1) if eta is not None else None
            ),
        })
        reason = None
        if next_step_no > max_outer:
            reason = "max_outer_steps_reached"
        elif elapsed >= max_hours * 3600.0:
            reason = "max_runtime_reached"
        elif llm_calls >= max_llm:
            reason = "max_llm_calls_reached"
        if reason is None:
            return False
        if self.status.get("stop_reason") != reason:
            self.status["stop_reason"] = reason
            await self.log(f"科学预算自动停止: {reason}", "info")
        self.running = False
        return True

    async def _stop_if_evaluation_budget_reached(
        self,
        *,
        refresh: bool = False,
    ) -> bool:
        budget = self._candidate_evaluation_budget()
        if budget <= 0:
            return False
        count = (
            await self._candidate_evaluation_count()
            if refresh
            else int(self.status.get("candidate_evaluations") or 0)
        )
        self.status.update({
            "candidate_evaluations": count,
            "candidate_evaluation_budget": budget,
            "candidate_evaluation_progress": round(count / budget, 6),
        })
        if count < budget:
            return False
        if self.status.get("stop_reason") != "candidate_evaluation_budget_reached":
            self.status["stop_reason"] = "candidate_evaluation_budget_reached"
            await self.log(
                f"随机研究预算完成: 候选评价 {count}/{budget}，worker 自动停止",
                "info",
            )
        self.running = False
        return True

    async def _stop_if_factor_target_reached(self) -> bool:
        target = self._target_factor_count()
        if target <= 0:
            return False
        count = await self._factor_count()
        self.status.update({
            "factor_count": count,
            "target_factor_count": target,
            "factor_target_progress": round(count / target, 6),
        })
        if count < target:
            return False
        self.status["stop_reason"] = "target_factor_count_reached"
        await self.log(
            f"研究目标完成: 当前协议因子库 {count}/{target}，worker 自动停止",
            "info",
        )
        self.running = False
        return True

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
        task_service = str(self.task_config.get("service_instance") or "").strip()
        if not service_accepts_task(self.task_config):
            self.running = False
            return {
                "ok": False,
                "msg": (
                    f"研究任务 {self.exp_id} 绑定服务 {task_service}，"
                    f"当前实例为 {SERVICE_INSTANCE}；"
                    "只有架构中立的统一主服务可以手动接管历史实例任务"
                ),
            }
        if SERVICE_ARCHITECTURE in {"two_layer", "three_layer"}:
            expected_layer3 = SERVICE_ARCHITECTURE == "three_layer"
            actual_layer3 = bool(self.task_config.get("layer3_enabled", False))
            if not (
                bool(self.task_config.get("layer1_enabled", False))
                and bool(self.task_config.get("layer2_enabled", False))
                and actual_layer3 == expected_layer3
            ):
                self.running = False
                return {
                    "ok": False,
                    "msg": (
                        f"任务架构与 {SERVICE_ARCHITECTURE} 服务不匹配"
                    ),
                }
        if SERVICE_ARCHITECTURE == "full_llm_three_layer":
            required = (
                self._full_llm_architecture()
                and self._scientific_governor_enabled()
                and self._layer1_enabled()
                and self._layer2_enabled()
                and self._layer3_enabled()
                and self._proposal_mode() == "llm"
            )
            if not required:
                self.running = False
                return {
                    "ok": False,
                    "msg": "任务不是可审计的全 LLM 三层架构，拒绝在 10013 启动",
                }
        self._runtime_identity = runtime_identity()
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
            "evaluation_queued": False,
            "evaluation_queue_seconds": None,
            "evaluation_queue_heartbeat_count": 0,
            "evaluation_started_at": None,
            "evaluation_elapsed_seconds": None,
            "evaluation_heartbeat_count": 0,
            "evaluation_soft_deadline_seconds": None,
            "evaluation_deadline_exceeded": False,
            "llm_active": False,
            "llm_started_at": None,
            "llm_elapsed_seconds": None,
            "llm_heartbeat_count": 0,
            "last_llm_duration_seconds": None,
            "proposal_mode": self._proposal_mode(),
            "memory_mode": self._memory_mode(),
            "experiment_arm": self.task_config.get("architecture_arm") or (
                "random"
                if self._proposal_mode() == "random"
                else "llm_memory" if self._memory_mode() == "adaptive" else "llm_cold"
            ),
            "three_layer": {
                "layer1_enabled": self._layer1_enabled(),
                "layer2_enabled": self._layer2_enabled(),
                "layer3_enabled": self._layer3_enabled(),
                "search_algorithms": list(self._search_algorithms())
                if self._layer1_enabled() else [],
                "full_llm_architecture": self._full_llm_architecture(),
                "scientific_governor_enabled": (
                    self._scientific_governor_enabled()
                ),
                "llm_roles": (
                    ["mechanism_scientist", "research_director", "scientific_governor"]
                    if self._full_llm_architecture()
                    else []
                ),
            },
            "runtime_identity": self._runtime_identity,
            "target_factor_count": self._target_factor_count(),
            "candidate_evaluation_budget": self._candidate_evaluation_budget(),
            "candidate_evaluations": 0,
            "candidate_evaluation_progress": 0.0,
            "budget_mode": (
                "unlimited" if self._continuous_operation() else "bounded"
            ),
            "service_instance": SERVICE_INSTANCE,
            "service_architecture": SERVICE_ARCHITECTURE or None,
            "target_mechanisms": list(self._target_mechanisms()),
            "factor_count": 0,
            "factor_target_progress": 0.0,
            "stop_reason": None,
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
            async with _PANEL_LOAD_SEMAPHORE:
                await asyncio.to_thread(panel.ensure_loaded)
            await self.log(f"面板就绪: {panel.summary()['rows']} 行 · {self.task_config.get('market', 'configured')}")

            self._set_phase("initializing_miner", progress=True)
            incumbent = await self._ensure_incumbent_v2()
            self.status["state"] = "running"
            cfg = await self._config_v2()

            # Unlimited campaigns must restore the real cumulative trial count;
            # otherwise restarts silently reset the multiple-testing burden.
            self.status["candidate_evaluations"] = (
                await self._candidate_evaluation_count()
            )

            if await self._stop_if_evaluation_budget_reached(refresh=True):
                return
            if await self._stop_if_factor_target_reached():
                return

            while self.running:
                if await self._stop_if_evaluation_budget_reached(refresh=True):
                    break
                step_no = await self._next_step_no()
                if await self._stop_if_scientific_budget_reached(
                    cfg,
                    next_step_no=step_no,
                ):
                    break
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
        layer3_enabled = self._layer3_enabled()
        provider = await self._provider("outer_provider") if layer3_enabled else None
        if layer3_enabled and provider is None:
            raise RuntimeError(
                "第三层 Governor 已启用，但 outer_provider 未配置；worker 熔断"
            )
        deliberate_random = self._proposal_mode() == "random"
        # Deliberate random baselines must not even build a feedback context;
        # otherwise provenance could imply that the random proposal learned
        # from historical scores despite the generator being outcome-agnostic.
        history = (
            []
            if deliberate_random or self._memory_mode() == "cold"
            else await self._version_history_v2()
        )

        # 第三层只治理搜索策略，不接触评价器或封存层。A-D 使用固定模板，
        # E 才允许 Governor LLM 提出单变量模板变更。
        inc_template = incumbent.harness_spec if isinstance(incumbent.harness_spec, dict) else DEFAULT_MINER_TEMPLATE
        scientific_directive = deepcopy(
            inc_template.get("_scientific_governor_directive") or {}
        )
        scientific_reflection: dict = {
            "decision": "not_applicable",
            "architecture_layer": 3,
        }
        if self._full_llm_architecture():
            if not self._scientific_governor_enabled():
                raise RuntimeError("全 LLM 三层任务未启用科学总督")
            governor_provider = await self._provider(
                "scientific_governor_provider"
            )
            if governor_provider is None:
                raise RuntimeError("第三层科学总督 provider 未配置；worker 熔断")
            interval = max(
                1,
                int(cfg.get("scientific_governor_interval_outer_steps", 3)),
            )
            directive_due = not scientific_directive or (step_no - 1) % interval == 0
            if directive_due:
                self._set_phase(
                    "outer_proposal",
                    current_operation="scientific_governor_llm",
                )
                (
                    scientific_directive,
                    _governor_note,
                    _governor_source,
                    scientific_reflection,
                ) = await self._run_llm_with_heartbeat(
                    propose_scientific_directive(
                        history,
                        governor_provider,
                        market=self.task_config.get("market", "us"),
                        portfolio_mode=self._portfolio_mode(),
                        allowed_mechanisms=tuple(
                            self._target_mechanisms()
                            or mechanisms_for_market(
                                self.task_config.get("market", "us")
                            )
                        ),
                        previous_directive=scientific_directive,
                        trace_context={
                            "experiment_id": self.exp_id,
                            "outer_step_no": step_no,
                            "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                            "miner_version_id": incumbent.id,
                            "runtime_identity": self._runtime_identity,
                            "experiment_arm": self.status.get("experiment_arm"),
                        },
                    ),
                    phase="outer_proposal",
                    operation="scientific_governor_llm",
                    current_seed=None,
                    current_task=None,
                )
            else:
                scientific_reflection = {
                    "decision": "reuse_active_directive",
                    "directive_id": scientific_directive.get("directive_id"),
                    "interval_outer_steps": interval,
                    "architecture_layer": 3,
                    "training_safe": True,
                }
        evaluated_before_governor = await self._candidate_evaluation_count()
        governor_warmup = int(cfg.get("governor_warmup_candidates", 0))
        if layer3_enabled and evaluated_before_governor < governor_warmup:
            cand_template = inc_template
            note = (
                "第三层进入确定性冷启动；完成一个完整候选 cohort 后再调用 "
                "Governor LLM"
            )
            source = "governor_warmup"
            proposal_reflection = {
                "decision": "deterministic_warmup",
                "evaluated_candidates": evaluated_before_governor,
                "required_candidates": governor_warmup,
                "architecture_layer": 2 if self._full_llm_architecture() else 3,
                "training_safe": True,
            }
        elif layer3_enabled:
            self._set_phase("outer_proposal", current_operation="governor_llm_or_fallback")
            cand_template, note, source, proposal_reflection = (
                await self._run_llm_with_heartbeat(
                    propose_template(
                        inc_template, history, provider,
                        market=self.task_config.get("market", "us"),
                        portfolio_mode=self._portfolio_mode(),
                        direction=self._signal_direction(),
                        direction_policy=self._direction_policy(),
                        deliberate_random=deliberate_random,
                        scientific_directive=scientific_directive,
                        trace_context={
                            "experiment_id": self.exp_id,
                            "outer_step_no": step_no,
                            "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                            "miner_version_id": incumbent.id,
                            "runtime_identity": self._runtime_identity,
                            "experiment_arm": self.status.get("experiment_arm"),
                            "architecture_layer": (
                                2 if self._full_llm_architecture() else 3
                            ),
                            "llm_role": (
                                "research_director"
                                if self._full_llm_architecture()
                                else "outer"
                            ),
                        },
                    ),
                    phase="outer_proposal",
                    operation="governor_llm",
                    current_seed=None,
                    current_task=None,
                )
            )
        else:
            cand_template = inc_template
            note = "第三层关闭；固定同一 MinerTemplate，仅累计同预算训练证据"
            source = "fixed_policy"
            proposal_reflection = {
                "decision": "layer3_disabled",
                "history_context_fingerprint": "",
                "architecture_layer": 2 if self._full_llm_architecture() else 3,
                "training_safe": True,
            }

        if self._full_llm_architecture():
            cand_template = deepcopy(cand_template)
            cand_template["_scientific_governor_directive"] = (
                scientific_directive
            )
            proposal_reflection = {
                **proposal_reflection,
                "research_director": deepcopy(proposal_reflection),
                "scientific_governor": scientific_reflection,
                "architecture_stack": [
                    "mechanism_scientist",
                    "research_director",
                    "scientific_governor",
                ],
            }

        # A changed Governor template requires a complete paired candidate and
        # incumbent block.  Never spend the tail of the campaign on an arm
        # that cannot reach the pre-registered acceptance test.
        candidate_budget = self._candidate_evaluation_budget()
        if layer3_enabled and cand_template != inc_template and candidate_budget > 0:
            evaluated = await self._candidate_evaluation_count()
            paired_block = int(cfg["n_seeds_per_candidate"]) * (
                int(cfg["inner_budget_per_outer_step"])
                + int(
                    cfg.get(
                        "incumbent_remeasure_budget",
                        cfg["inner_budget_per_outer_step"],
                    )
                )
            )
            if candidate_budget - evaluated < paired_block:
                cand_template = inc_template
                note = (
                    "剩余候选预算不足以完成 Governor 提案的候选/在位配对块；"
                    "本步降级为固定模板证据积累"
                )
                source = "budget_guard"
                proposal_reflection = {
                    **proposal_reflection,
                    "decision": "insufficient_paired_budget",
                    "evaluated_candidates": evaluated,
                    "remaining_candidates": candidate_budget - evaluated,
                    "required_paired_block": paired_block,
                }

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

        # A no-op proposal is evidence about the outer model, not a new arm.
        # Do not spend a full candidate-versus-incumbent comparison on two
        # identical templates.  The first no-op may still warm the baseline so
        # the next proposal receives real, training-safe feedback.
        if cand_template == inc_template:
            warmup_budget = int(cfg["baseline_warmup_budget"])
            warmup_results: list[dict] = []
            baseline_score = float(incumbent.meta_score or 0.0)
            baseline_report = dict(incumbent.feedback_summary or {})
            if (incumbent.meta_score is None or not layer3_enabled) and self.running:
                feedback_baseline = await self._feedback_baseline_v2(
                    [task["name"] for task in cfg["tasks"]],
                )
                for seed in range(int(cfg["n_seeds_per_candidate"])):
                    if not self.running:
                        break
                    warmup_results.append(
                        await self._mining_session_v2(
                            incumbent,
                            step_no,
                            warmup_budget,
                            cfg,
                            seed,
                            feedback_baseline,
                        )
                    )
                if warmup_results:
                    baseline_score = st.mean(
                        row["score"] for row in warmup_results
                    )
                    incumbent = await self._update_score(
                        incumbent.id,
                        baseline_score,
                    )
                    baseline_report = combine_seed_feedback(warmup_results)
                    async with SessionLocal() as session:
                        incumbent_db = await session.get(
                            MinerVersion,
                            incumbent.id,
                        )
                        incumbent_db.feedback_summary = baseline_report
                        await session.commit()
            async with SessionLocal() as session:
                candidate_db = await session.get(MinerVersion, cand.id)
                candidate_db.status = "rejected"
                candidate_db.meta_score = None
                candidate_db.feedback_summary = {
                    "decision": "no_op",
                    "reason": "candidate template equals incumbent",
                }
                session.add(OuterStep(
                    experiment_id=self.exp_id,
                    step_no=step_no,
                    candidate_id=cand.id,
                    incumbent_id=incumbent.id,
                    candidate_score=None,
                    incumbent_score=baseline_score,
                    accepted=False,
                    evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                    context_fingerprint=str(
                        proposal_reflection.get(
                            "history_context_fingerprint",
                            "",
                        )
                    ),
                    detail={
                        "protocol_version": EVALUATION_PROTOCOL_VERSION,
                        "mode": "v2",
                        "decision": "no_op_baseline_warmup",
                        "note": note,
                        "source": source,
                        "warmup_budget": warmup_budget,
                        "warmup_seed_count": len(warmup_results),
                        "incumbent_report": baseline_report,
                        "runtime_identity": self._runtime_identity,
                    },
                ))
                await session.commit()
            await self.log(
                f"[V2] 外层步 {step_no}: 候选模板无实质改动，"
                f"跳过 A/B；基线={baseline_score:.4f}",
                "info",
            )
            return incumbent, cfg

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
            if not self.running:
                break
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
        if not self.running or len(cand_seed_results) != n_seeds:
            async with SessionLocal() as session:
                candidate_db = await session.get(MinerVersion, cand.id)
                candidate_db.status = "rejected"
                candidate_db.meta_score = cand_mean
                candidate_db.feedback_summary = cand_report
                session.add(OuterStep(
                    experiment_id=self.exp_id,
                    step_no=step_no,
                    candidate_id=cand.id,
                    incumbent_id=incumbent.id,
                    candidate_score=cand_mean,
                    incumbent_score=incumbent.meta_score,
                    accepted=False,
                    evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                    detail={
                        "protocol_version": EVALUATION_PROTOCOL_VERSION,
                        "mode": "v2",
                        "decision": "incomplete_budget_stop",
                        "completed_candidate_seeds": len(cand_seed_results),
                        "required_candidate_seeds": n_seeds,
                        "stop_reason": self.status.get("stop_reason"),
                        "candidate_report": cand_report,
                        "runtime_identity": self._runtime_identity,
                    },
                ))
                await session.commit()
            await self.log(
                f"[V2] 外层步 {step_no}: 预算停止时仅完成 "
                f"{len(cand_seed_results)}/{n_seeds} 个候选 seed，"
                "不进行外层接受判定",
                "info",
            )
            return incumbent, cfg

        # ---- 在位者重测 ----
        remeasure_every = int(cfg["incumbent_remeasure_every"])
        remeasure_budget = int(cfg.get("incumbent_remeasure_budget", budget))
        inc_seed_results = []
        remeasure_required = (
            incumbent.meta_score is None or step_no % remeasure_every == 0
        )
        if remeasure_required:
            for seed in range(n_seeds):
                if not self.running:
                    break
                self._set_phase(
                    "incumbent_remeasure",
                    progress=True,
                    current_operation="seed",
                    current_seed=seed,
                    current_budget_index=0,
                    current_budget_total=remeasure_budget,
                )
                inc_seed_result = await self._mining_session_v2(
                    incumbent,
                    step_no,
                    remeasure_budget,
                    cfg,
                    seed,
                    feedback_baseline,
                )
                inc_seed_results.append(inc_seed_result)
            inc_scores = [row["score"] for row in inc_seed_results]
            inc_mean = st.mean(inc_scores) if inc_scores else 0.0
            inc_std = st.stdev(inc_scores) if len(inc_scores) >= 2 else 0.0
            inc_report = combine_seed_feedback(inc_seed_results)
            if len(inc_seed_results) == n_seeds:
                incumbent = await self._update_score(incumbent.id, inc_mean)
            await self.log(f"[V2]   在位重测: mean={inc_mean:.4f} std={inc_std:.4f} (n={len(inc_scores)})")
        else:
            inc_scores = []
            inc_mean = incumbent.meta_score or 0.0
            inc_std = 0.0
            inc_report = dict(incumbent.feedback_summary or {})

        if remeasure_required and len(inc_seed_results) != n_seeds:
            async with SessionLocal() as session:
                candidate_db = await session.get(MinerVersion, cand.id)
                candidate_db.status = "rejected"
                candidate_db.meta_score = cand_mean
                candidate_db.feedback_summary = cand_report
                session.add(OuterStep(
                    experiment_id=self.exp_id,
                    step_no=step_no,
                    candidate_id=cand.id,
                    incumbent_id=incumbent.id,
                    candidate_score=cand_mean,
                    incumbent_score=inc_mean,
                    accepted=False,
                    evaluation_protocol=EVALUATION_PROTOCOL_VERSION,
                    detail={
                        "protocol_version": EVALUATION_PROTOCOL_VERSION,
                        "mode": "v2",
                        "decision": "incomplete_paired_budget_stop",
                        "completed_candidate_seeds": len(cand_seed_results),
                        "completed_incumbent_seeds": len(inc_seed_results),
                        "required_seeds": n_seeds,
                        "stop_reason": self.status.get("stop_reason"),
                        "candidate_report": cand_report,
                        "incumbent_report": inc_report,
                        "runtime_identity": self._runtime_identity,
                    },
                ))
                await session.commit()
            await self.log(
                f"[V2] 外层步 {step_no}: 在位者配对仅完成 "
                f"{len(inc_seed_results)}/{n_seeds} seed，不进行接受判定",
                "info",
            )
            return incumbent, cfg

        # ---- 同协议单边统计门 ----
        test = (
            _paired_score_test(cand_scores, inc_scores)
            if len(cand_scores) == len(inc_scores) and len(cand_scores) >= 2
            else _one_sided_score_test(
                cand_scores,
                inc_scores,
                reference_mean=inc_mean,
            )
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
        if int(inc_report.get("attempts") or 0) > 0:
            diversity_non_degrading = bool(
                float(cand_report.get("mechanism_coverage") or 0.0) + 0.05
                >= float(inc_report.get("mechanism_coverage") or 0.0)
                and float(cand_report.get("mechanism_hhi") or 1.0)
                <= float(inc_report.get("mechanism_hhi") or 1.0) + 0.05
                and float(cand_report.get("structural_duplicate_rate") or 0.0)
                <= float(inc_report.get("structural_duplicate_rate") or 0.0) + 0.05
                and float(cand_report.get("behavior_duplicate_rate") or 0.0)
                <= float(inc_report.get("behavior_duplicate_rate") or 0.0) + 0.05
            )
        else:
            diversity_non_degrading = int(
                cand_report.get("distinct_mechanisms") or 0
            ) >= 3
        return_source_governance = self._return_source_governance()
        if return_source_governance["enabled"]:
            if int(inc_report.get("attempts") or 0) > 0:
                return_sources_non_degrading = bool(
                    float(
                        cand_report.get("scoped_behavior_duplicate_rate")
                        or 0.0
                    )
                    <= float(
                        inc_report.get("scoped_behavior_duplicate_rate")
                        or 0.0
                    ) + 0.05
                    and float(
                        cand_report.get("effective_return_sources") or 0.0
                    ) + 0.5
                    >= float(
                        inc_report.get("effective_return_sources") or 0.0
                    )
                )
            else:
                return_sources_non_degrading = int(
                    cand_report.get("return_source_clusters") or 0
                ) >= min(3, return_source_governance["required_sources"])
        else:
            return_sources_non_degrading = True
        accepted = bool(
            len(cand_scores) == n_seeds
            and len(cand_scores) >= 2
            and cand_mean > inc_mean
            and p_value < p_threshold
            and gate_non_degrading
            and pass_rate_non_degrading
            and diversity_non_degrading
            and return_sources_non_degrading
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
                "runtime_identity": self._runtime_identity,
                "experiment_arm": self.status.get("experiment_arm"),
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
                    "runtime_identity": self._runtime_identity,
                    "admission_safety": {
                        "gate_score_non_degrading": gate_non_degrading,
                        "pass_rate_non_degrading": pass_rate_non_degrading,
                        "diversity_non_degrading": diversity_non_degrading,
                        "return_sources_non_degrading": (
                            return_sources_non_degrading
                        ),
                        "return_source_governance": return_source_governance,
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
        provider = await self._provider("inner_provider") if self._layer2_enabled() else None
        if self._layer2_enabled() and provider is None:
            raise RuntimeError(
                "第二层 Researcher 已启用，但 inner_provider 未配置；worker 熔断"
            )
        tasks = cfg["tasks"]
        task_best_scores: dict[str, float] = {}
        session_envelopes: list[dict] = []

        # 固定种子确保可复现
        rng = random.Random(seed * 10000 + step_no * 100)
        session_start_node_id = await self._max_node_id()
        campaign_expressions = await self._experiment_expressions()
        proposal_queue: dict[int, tuple[dict, tuple[str, str, str, dict]]] = {}
        market = self.task_config.get("market", "us")

        async def prepare_slot(slot: int) -> dict:
            task = tasks[slot % len(tasks)]
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
            proposal_feedback_nodes = (
                feedback_nodes if self._memory_mode() == "adaptive" else []
            )
            family_schedule = self._mechanism_schedule(template, market)
            target_family = family_schedule[
                (seed * budget + slot) % len(family_schedule)
            ]
            base_node = max(
                (
                    node
                    for node in proposal_feedback_nodes
                    if node.get("status") == "ok"
                    and mechanism_from_item(node) == target_family
                    and int(
                        (node.get("proposal_meta") or {}).get("tree_depth", 0)
                    ) < int(cfg["max_tree_depth"])
                ),
                key=lambda node: float(node.get("public_score") or 0.0),
                default=None,
            )
            op = (
                "improve"
                if base_node and rng.random() >= float(cfg["draft_ratio"])
                else "draft"
            )
            search_seed = None
            if self._layer1_enabled():
                use_algorithm_inspiration = True
                if self._full_llm_architecture():
                    use_algorithm_inspiration = rng.random() < float(
                        cfg.get("algorithm_inspiration_share", 0.30)
                    )
                if use_algorithm_inspiration:
                    search_seed = propose_search_seed(
                        family=target_family,
                        fields=get_dsl_fields(self.task_config.get("market")),
                        feedback_nodes=feedback_nodes,
                        algorithms=self._search_algorithms(),
                        rng=rng,
                    )
            return {
                "slot": slot,
                "task": task,
                "feedback_nodes": proposal_feedback_nodes,
                "target_family": target_family,
                "base_node": base_node,
                "op": op,
                "request_id": f"step-{step_no}:seed-{seed}:slot-{slot + 1}",
                "rng": rng,
                "search_seed": search_seed,
            }

        for i in range(budget):
            if not self.running:
                break
            if i not in proposal_queue and await self._stop_if_scientific_budget_reached(
                cfg,
                next_step_no=step_no,
            ):
                break
            if await self._stop_if_evaluation_budget_reached():
                break
            if i not in proposal_queue:
                assignment = await prepare_slot(i)
                batch_size = min(
                    int(cfg["batch_candidates_per_call"]),
                    budget - i,
                )
                if self._layer1_enabled() and not self._layer2_enabled():
                    seed_proposal = assignment["search_seed"]
                    proposal_queue[i] = (
                        assignment,
                        (
                            seed_proposal.expression,
                            seed_proposal.hypothesis,
                            "search_pool",
                            {
                                **seed_proposal.metadata,
                                "reflection": "第一层独立候选；第二层关闭。",
                                "targeted_failures": [],
                                "expected_effect": "估计第一层算法组合的独立增量",
                                "change_axis": "new_draft",
                                "declared_family": assignment["target_family"],
                                "family_match": True,
                            },
                        ),
                    )
                elif (
                    provider is not None
                    and self._proposal_mode() == "llm"
                    and batch_size > 1
                ):
                    assignments = [assignment]
                    for slot in range(i + 1, i + batch_size):
                        assignments.append(await prepare_slot(slot))
                    proposals = await self._run_llm_with_heartbeat(
                        propose_batch(
                            template,
                            assignments,
                            provider,
                            fields=get_dsl_fields(self.task_config.get("market")),
                            trace_context={
                                "experiment_id": self.exp_id,
                                "outer_step_no": step_no,
                                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                                "miner_version_id": miner.id,
                                "seed": seed,
                                "batch_start_index": i + 1,
                                "runtime_identity": self._runtime_identity,
                                "experiment_arm": self.status.get("experiment_arm"),
                                "architecture_layer": (
                                    1 if self._full_llm_architecture() else 2
                                ),
                                "llm_role": (
                                    "mechanism_scientist"
                                    if self._full_llm_architecture()
                                    else "inner"
                                ),
                                "direct_expression_authority": (
                                    self._full_llm_architecture()
                                ),
                            },
                        ),
                        phase="inner_proposal",
                        operation="researcher_llm_batch",
                        current_seed=seed,
                        current_task=assignment["task"]["name"],
                        current_budget_index=i + 1,
                        current_budget_total=budget,
                    )
                    proposal_queue.update({
                        row["slot"]: (row, proposal)
                        for row, proposal in zip(assignments, proposals)
                    })
                else:
                    proposal = await self._run_llm_with_heartbeat(
                        propose(
                            template,
                            assignment["op"],
                            assignment["task"],
                            assignment["feedback_nodes"],
                            provider,
                            fields=get_dsl_fields(self.task_config.get("market")),
                            trace_context={
                                "experiment_id": self.exp_id,
                                "outer_step_no": step_no,
                                "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                                "miner_version_id": miner.id,
                                "task_name": assignment["task"]["name"],
                                "seed": seed,
                                "budget_index": i + 1,
                                "runtime_identity": self._runtime_identity,
                                "experiment_arm": self.status.get("experiment_arm"),
                                "cohort_id": assignment["request_id"],
                                "architecture_layer": (
                                    1 if self._full_llm_architecture() else 2
                                ),
                                "llm_role": (
                                    "mechanism_scientist"
                                    if self._full_llm_architecture()
                                    else "inner"
                                ),
                                "direct_expression_authority": (
                                    self._full_llm_architecture()
                                ),
                            },
                            rng=rng,
                            target_family=assignment["target_family"],
                            deliberate_random=self._proposal_mode() == "random",
                        ),
                        phase="inner_proposal",
                        operation="researcher_llm_single",
                        current_seed=seed,
                        current_task=assignment["task"]["name"],
                        current_budget_index=i + 1,
                        current_budget_total=budget,
                    )
                    proposal_queue[i] = (assignment, proposal)
            assignment, proposal = proposal_queue.pop(i)
            task = assignment["task"]
            proposal_feedback_nodes = assignment["feedback_nodes"]
            target_family = assignment["target_family"]
            base_node = assignment["base_node"]
            op = assignment["op"]
            expr, hypo, source, proposal_meta = proposal
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                progress=True,
                current_task=task["name"],
                current_operation="context_lookup",
                current_seed=seed,
                current_budget_index=i + 1,
                current_budget_total=budget,
            )
            self._set_phase(
                self.status.get("phase") or "candidate_mining",
                current_operation=f"proposal:{op}",
            )
            novelty_retries = 0
            while (
                source == "search_pool"
                and expr in campaign_expressions
                and novelty_retries < 32
            ):
                novelty_retries += 1
                replacement = propose_search_seed(
                    family=target_family,
                    fields=get_dsl_fields(self.task_config.get("market")),
                    feedback_nodes=self._merge_feedback_nodes(
                        (feedback_baseline or {}).get(task["name"], []),
                        await self._feedback_nodes_v2(
                            miner.id,
                            task["name"],
                            seed=seed,
                            min_node_id=session_start_node_id,
                        ),
                    ),
                    algorithms=self._search_algorithms(),
                    rng=rng,
                )
                assignment["search_seed"] = replacement
                expr = replacement.expression
                hypo = replacement.hypothesis
                proposal_meta = {
                    **replacement.metadata,
                    "reflection": "第一层重复候选重采样；第二层关闭。",
                    "targeted_failures": [],
                    "expected_effect": "保持第一层试验的表达式级新颖性",
                    "change_axis": "new_draft",
                    "declared_family": target_family,
                    "family_match": True,
                }
            while (
                source == "random"
                and expr in campaign_expressions
                and novelty_retries < 32
            ):
                novelty_retries += 1
                expr, hypo, source, proposal_meta = await propose(
                    template,
                    "draft",
                    task,
                    proposal_feedback_nodes,
                    provider,
                    fields=get_dsl_fields(
                        self.task_config.get("market")
                    ),
                    trace_context={
                        "experiment_id": self.exp_id,
                        "outer_step_no": step_no,
                        "evaluation_protocol": EVALUATION_PROTOCOL_VERSION,
                        "miner_version_id": miner.id,
                        "task_name": task["name"],
                        "seed": seed,
                        "budget_index": i + 1,
                        "campaign_novelty_retry": novelty_retries,
                        "runtime_identity": self._runtime_identity,
                        "experiment_arm": self.status.get("experiment_arm"),
                    },
                    rng=rng,
                    target_family=target_family,
                    deliberate_random=self._proposal_mode() == "random",
                )
            seed_meta = (
                assignment["search_seed"].metadata
                if assignment.get("search_seed") is not None
                else {}
            )
            proposal_meta = {
                **seed_meta,
                **(proposal_meta or {}),
                "campaign_novelty_retries": novelty_retries,
                "campaign_exact_duplicate": expr in campaign_expressions,
                "experiment_arm": self.status.get("experiment_arm"),
                "memory_mode": self._memory_mode(),
                "runtime_identity": self._runtime_identity,
                "cohort_id": f"step-{step_no}:seed-{seed}:slot-{i + 1}",
                "architecture_layer": (
                    1 if self._full_llm_architecture()
                    else (proposal_meta or {}).get("architecture_layer", 2)
                ),
                "llm_role": (
                    "mechanism_scientist"
                    if self._full_llm_architecture()
                    else "inner"
                ),
                "direct_expression_authority": self._full_llm_architecture(),
                "scientific_directive_id": (
                    (template.get("_scientific_governor_directive") or {}).get(
                        "directive_id"
                    )
                ),
                "tree_depth": (
                    int((base_node.get("proposal_meta") or {}).get("tree_depth", 0)) + 1
                    if op == "improve" and base_node
                    else 0
                ),
                "parent_change_axis": (
                    (proposal_meta or {}).get("change_axis")
                    if op == "improve"
                    else "new_draft"
                ),
            }
            campaign_expressions.add(expr)

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
                if source == "llm_rejected":
                    raise ValueError(
                        str(
                            (proposal_meta or {}).get("validation_error")
                            or "LLM 批量候选未通过语义验证"
                        )
                    )
                self._set_phase(
                    self.status.get("phase") or "candidate_mining",
                    current_operation="factor_evaluation",
                )
                metrics = await self._run_shared_evaluation(
                        expr, task["universe_n"], task["horizon"],
                        task.get("mode", DEFAULT_PORTFOLIO_MODE), task.get("direction", 1),
                        self._panel_glob(), task.get("cost_bps", 15),
                        self.task_config.get("market", "us"),
                        self._dynamic_evaluation_overrides(),
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
                    "training_return_path_signature": combined_training_signature(
                        metrics["public"],
                        metrics["gate"],
                    ),
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
                    node_id=node.id,
                    parent_node_id=node.parent_id,
                    expression=expr,
                    search_method=str(
                        (node.proposal_meta or {}).get("search_algorithm")
                        or node.source
                    ),
                    mechanism=str(
                        (node.proposal_meta or {}).get("target_family") or ""
                    ),
                    selected=bool(
                        (node.public_metrics.get("discovery") or {}).get("passed")
                    ),
                    failure_reason=node.error or "; ".join(
                        (node.public_metrics.get("discovery") or {}).get(
                            "failure_reasons", []
                        )[:4]
                    ),
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
            self.status["candidate_evaluations"] = (
                int(self.status.get("candidate_evaluations") or 0) + 1
            )
            self._touch_progress(
                current_operation="register_factor" if node.status == "ok" else "candidate_failed",
            )
            if node.status == "ok":
                min_icir = float(template.get("min_public_icir", 0.25))
                admission = await self._maybe_register_factor_v2(
                    node,
                    min_icir,
                    task=task,
                )
                if admission:
                    envelope = enrich_feedback_with_factor_admission(
                        envelope,
                        admission,
                    )
                    session_envelopes[-1] = envelope
                    async with SessionLocal() as s:
                        db_node = await s.get(Node, node.id)
                        if db_node is not None:
                            db_node.feedback_summary = envelope
                            await s.commit()
                if await self._stop_if_factor_target_reached():
                    break
                if i % 10 == 0:
                    await self.log(
                        f"[V2 s{seed}]  内层[{task['name']}] {op}/{source} "
                        f"learn={node.public_score:.3f} "
                        f"gate={float((node.public_metrics.get('discovery') or {}).get('gate_score') or 0.0):.3f} "
                        f"family={target_family} "
                        f"dir={int((node.public_metrics.get('discovery') or {}).get('selected_direction') or task.get('direction', 1)):+d} "
                        f"{expr[:60]}",
                        "debug",
                    )
            if await self._stop_if_evaluation_budget_reached():
                break

        return_source_governance = self._return_source_governance()
        score, score_detail = diversity_adjusted_score(
            task_best_scores,
            session_envelopes,
            self.task_config.get("market", "us"),
            return_source_weight=return_source_governance[
                "meta_score_weight"
            ],
            required_return_sources=return_source_governance[
                "required_sources"
            ],
        )
        score_detail["return_source_governance"] = return_source_governance
        summary = combine_seed_feedback([{
            "seed": seed,
            "score": score,
            "task_best_scores": task_best_scores,
            "envelopes": session_envelopes,
        }])
        summary["session_score_detail"] = score_detail
        return {
            "seed": seed,
            "score": score,
            "task_best_scores": task_best_scores,
            "envelopes": session_envelopes,
            "summary": summary,
            "score_detail": score_detail,
        }

    async def _maybe_register_factor_v2(
        self,
        node: Node,
        min_icir: float,
        *,
        task: dict | None = None,
    ) -> dict | None:
        pm = node.public_metrics
        discovery = pm.get("discovery") or {}
        if not discovery.get("passed") or (pm.get("icir") or 0) < min_icir:
            return None
        market = self.task_config.get("market", "us")
        portfolio_mode = self._portfolio_mode()
        governance = self._return_source_governance()
        task = dict(task or {})
        task_snapshot = {
            "name": str(task.get("name") or node.task_name),
            "market": market,
            "portfolio_mode": portfolio_mode,
            "universe_n": int(task.get("universe_n") or 0),
            "horizon": int(task.get("horizon") or pm.get("horizon") or 0),
            "cost_bps": float(task.get("cost_bps") or 0.0),
        }
        task_signature = "|".join(
            str(task_snapshot[key])
            for key in (
                "market",
                "portfolio_mode",
                "universe_n",
                "horizon",
                "cost_bps",
            )
        )
        semantic_audit = audit_expression_semantics(node.expression, market)
        if semantic_audit["errors"]:
            await self.log(
                f"  因子未入库: 字段语义审计失败 · {semantic_audit['errors'][0][:180]}",
                "warning",
            )
            return None
        async with SessionLocal() as s:
            exists = await s.scalar(select(Factor).where(
                Factor.expression == node.expression, Factor.experiment_id == self.exp_id))
            if exists:
                return None
            factor_filters = [
                Factor.evaluation_protocol == EVALUATION_PROTOCOL_VERSION
            ]
            if not governance["cross_experiment_admission"]:
                factor_filters.append(Factor.experiment_id == self.exp_id)
            existing_factors = list((await s.scalars(
                select(Factor).where(*factor_filters)
            )).all())
            nearest_structural: tuple[float, Factor] | None = None
            nearest_behavior: tuple[float, Factor] | None = None
            candidate_signature = pm.get("training_return_path_signature") or {}
            for existing_factor in existing_factors:
                existing_meta = dict(existing_factor.research_meta or {})
                existing_market = str(
                    existing_meta.get("market")
                    or (market if existing_factor.experiment_id == self.exp_id else "")
                )
                existing_mode = str(
                    existing_meta.get("portfolio_mode")
                    or (
                        portfolio_mode
                        if existing_factor.experiment_id == self.exp_id
                        else ""
                    )
                )
                if existing_market != market or existing_mode != portfolio_mode:
                    continue
                try:
                    structural = expression_similarity(
                        node.expression,
                        existing_factor.expression,
                    )
                except (SyntaxError, ValueError):
                    structural = 0.0
                if nearest_structural is None or structural > nearest_structural[0]:
                    nearest_structural = (structural, existing_factor)
                same_return_scope = (
                    existing_factor.task_name == node.task_name
                    if existing_factor.experiment_id == self.exp_id
                    else existing_meta.get("task_signature") == task_signature
                )
                if not same_return_scope:
                    continue
                correlation = return_path_correlation(
                    candidate_signature,
                    (existing_factor.public_metrics or {}).get(
                        "training_return_path_signature"
                    ),
                )
                if correlation is not None and (
                    nearest_behavior is None or correlation > nearest_behavior[0]
                ):
                    nearest_behavior = (correlation, existing_factor)
            admission = {
                "protocol": (
                    "factor_diversity_admission_v2"
                    if governance["enabled"]
                    else "factor_diversity_admission_v1"
                ),
                "return_source_governance": governance,
                "accepted": True,
                "mechanism_family": mechanism_from_item({
                    "expression": node.expression,
                    "hypothesis": node.hypothesis,
                    "proposal_meta": node.proposal_meta,
                }),
                "max_structural_similarity": round(
                    nearest_structural[0] if nearest_structural else 0.0,
                    6,
                ),
                "max_return_path_correlation": round(
                    nearest_behavior[0] if nearest_behavior else 0.0,
                    6,
                ),
                "structural_threshold": 0.84,
                "return_path_threshold": governance[
                    "correlation_threshold"
                ],
                "task_signature": task_signature,
                "task_snapshot": task_snapshot,
                "semantic_audit": semantic_audit,
            }
            reason = ""
            nearest_reference = None
            if nearest_structural and nearest_structural[0] >= 0.84:
                reason = (
                    "structural_duplicate_of_factor_"
                    f"{nearest_structural[1].id}"
                )
                nearest_reference = nearest_structural[1]
            elif (
                nearest_behavior
                and nearest_behavior[0] >= governance["correlation_threshold"]
            ):
                reason = (
                    "return_path_duplicate_of_factor_"
                    f"{nearest_behavior[1].id}"
                )
                nearest_reference = nearest_behavior[1]
            if nearest_reference is not None:
                admission["reference"] = {
                    "factor_id": nearest_reference.id,
                    "experiment_id": nearest_reference.experiment_id,
                    "task_name": nearest_reference.task_name,
                    "cross_experiment": (
                        nearest_reference.experiment_id != self.exp_id
                    ),
                }
            if reason:
                admission.update({"accepted": False, "reason": reason})
                db_node = await s.get(Node, node.id)
                db_node.proposal_meta = {
                    **(db_node.proposal_meta or {}),
                    "factor_admission": admission,
                }
                db_node.feedback_summary = {
                    **(db_node.feedback_summary or {}),
                    "factor_admission": admission,
                }
                await s.commit()
                await self.log(
                    f"  因子未入库: {reason} · "
                    f"structure={admission['max_structural_similarity']:.3f} "
                    f"return_corr={admission['max_return_path_correlation']:.3f}",
                    "info",
                )
                return admission
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
                    "mechanism_family": admission["mechanism_family"],
                    "factor_admission": admission,
                    "semantic_audit": semantic_audit,
                    "training_return_path_signature": candidate_signature,
                    "task_signature": task_signature,
                    "task_snapshot": task_snapshot,
                    "return_source_governance": governance,
                    "double_blind_review": {
                        "protocol": "factorfactory.double-blind-review/v1",
                        "method_review": "pending_optional_llm_shortlist_review",
                        "code_review": deterministic_code_review(
                            node.expression,
                            self.task_config.get("market", "us"),
                        ),
                        "promotion_policy": (
                            "research_record_allowed_formal_promotion_requires_both"
                        ),
                    },
                },
                fingerprint={
                    "miner_version_id": node.miner_version_id, "outer_step": node.outer_step_no,
                    "source": node.source,
                    "mechanism_family": admission["mechanism_family"],
                    **expression_fingerprint(node.expression),
                },
            ))
            await s.commit()
            return admission

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

    async def _experiment_expressions(self) -> set[str]:
        """Return exact same-protocol expressions for random-campaign dedupe."""
        async with SessionLocal() as session:
            values = await session.scalars(
                select(Node.expression).where(
                    Node.experiment_id == self.exp_id,
                    Node.evaluation_protocol
                    == EVALUATION_PROTOCOL_VERSION,
                )
            )
            return {
                str(value).strip()
                for value in values
                if str(value or "").strip()
            }

    async def _config_v2(self) -> dict:
        async with SessionLocal() as s:
            row = await s.get(Setting, "engine_config")
            cfg = {**DEFAULT_ENGINE_CONFIG_V2, **(row.value if row else {})}
            task_cfg = self.task_config.get("engine_config", {})
            local_tasks = task_cfg.get("tasks")
            cfg.update(task_cfg)
            positive_ints = (
                "inner_budget_per_outer_step",
                "n_seeds_per_candidate",
                "paired_cohorts_per_comparison",
                "incumbent_remeasure_every",
                "incumbent_remeasure_budget",
                "baseline_warmup_budget",
                "max_outer_steps",
                "max_llm_calls",
                "batch_candidates_per_call",
                "max_tree_depth",
                "governor_warmup_candidates",
            )
            for key in positive_ints:
                try:
                    cfg[key] = int(cfg[key])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"engine_config.{key} 必须为正整数") from exc
                if key == "governor_warmup_candidates":
                    if cfg[key] < 0:
                        raise ValueError(
                            "engine_config.governor_warmup_candidates 必须为非负整数"
                        )
                elif cfg[key] <= 0:
                    raise ValueError(f"engine_config.{key} 必须为正整数")
            try:
                cfg["scientific_governor_interval_outer_steps"] = int(
                    cfg.get("scientific_governor_interval_outer_steps", 3)
                )
                cfg["algorithm_inspiration_share"] = float(
                    cfg.get("algorithm_inspiration_share", 0.30)
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("全 LLM 三层控制参数非法") from exc
            if cfg["scientific_governor_interval_outer_steps"] <= 0:
                raise ValueError(
                    "engine_config.scientific_governor_interval_outer_steps 必须为正整数"
                )
            if not 0.0 <= cfg["algorithm_inspiration_share"] <= 1.0:
                raise ValueError(
                    "engine_config.algorithm_inspiration_share 必须在 0..1"
                )
            cfg["n_seeds_per_candidate"] = min(
                cfg["n_seeds_per_candidate"],
                cfg["paired_cohorts_per_comparison"],
            )
            try:
                cfg["max_runtime_hours"] = float(cfg["max_runtime_hours"])
                cfg["draft_ratio"] = float(cfg["draft_ratio"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("引擎小数控制参数非法") from exc
            if cfg["max_runtime_hours"] <= 0:
                raise ValueError("engine_config.max_runtime_hours 必须为正数")
            if not 0.0 <= cfg["draft_ratio"] <= 1.0:
                raise ValueError("engine_config.draft_ratio 必须在 0..1")
            cfg["memory_mode"] = self._memory_mode()
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
                deliberate_random=self._proposal_mode() == "random",
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
                    self._dynamic_evaluation_overrides(),
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
                    node_id=node.id,
                    parent_node_id=node.parent_id,
                    expression=expr,
                    search_method=str(
                        (node.proposal_meta or {}).get("search_algorithm")
                        or node.source
                    ),
                    mechanism=str(
                        (node.proposal_meta or {}).get("target_family") or ""
                    ),
                    selected=bool(
                        (node.public_metrics.get("discovery") or {}).get("passed")
                    ),
                    failure_reason=node.error or "; ".join(
                        (node.public_metrics.get("discovery") or {}).get(
                            "failure_reasons", []
                        )[:4]
                    ),
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
        # Random campaigns must remain isolated from global provider settings.
        # Returning None activates the deterministic, mechanism-targeted local
        # fallback for both the inner proposer and outer template proposer.
        if self._proposal_mode() in {"random", "search_pool"}:
            return None
        if role == "inner_provider" and not self._layer2_enabled():
            return None
        if role == "outer_provider" and not self._layer3_enabled():
            return None
        if (
            role == "scientific_governor_provider"
            and not self._scientific_governor_enabled()
        ):
            return None
        async with SessionLocal() as s:
            row = await s.get(Setting, "llm_providers")
            if not row:
                return None
            conf = row.value
            name = conf.get(role)
            if role == "scientific_governor_provider" and not name:
                # 10013 can run immediately with the audited outer provider;
                # operators may later assign a dedicated governor model.
                name = conf.get("outer_provider")
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
            "runtime_identity": runtime_identity(),
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


def _paired_score_test(
    candidate_scores: list[float],
    incumbent_scores: list[float],
) -> dict:
    """Exact one-sided paired sign-flip test for predeclared seed cohorts."""
    if len(candidate_scores) != len(incumbent_scores) or len(candidate_scores) < 2:
        return {
            "type": "invalid_paired_cohort",
            "p_value": 1.0,
            "candidate_n": len(candidate_scores),
            "incumbent_n": len(incumbent_scores),
        }
    differences = [
        float(candidate) - float(incumbent)
        for candidate, incumbent in zip(candidate_scores, incumbent_scores)
    ]
    observed = st.mean(differences)
    if observed <= 0:
        p_value = 1.0
    elif len(differences) <= 16:
        extreme = 0
        total = 1 << len(differences)
        for mask in range(total):
            permuted = st.mean(
                value if mask & (1 << index) else -value
                for index, value in enumerate(differences)
            )
            if permuted >= observed - 1e-12:
                extreme += 1
        p_value = extreme / total
    else:
        spread = st.stdev(differences)
        if spread <= 1e-12:
            p_value = 0.0
        else:
            t_stat = observed / (spread / math.sqrt(len(differences)))
            p_value = _student_t_survival(
                t_stat,
                float(len(differences) - 1),
            )
    return {
        "type": "paired_seed_sign_flip_one_sided",
        "candidate_n": len(candidate_scores),
        "incumbent_n": len(incumbent_scores),
        "candidate_mean": round(st.mean(candidate_scores), 6),
        "incumbent_mean": round(st.mean(incumbent_scores), 6),
        "difference": round(observed, 6),
        "paired_differences": [round(value, 6) for value in differences],
        "p_value": round(max(0.0, min(1.0, p_value)), 8),
    }


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
