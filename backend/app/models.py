from datetime import date, datetime

from sqlalchemy import JSON, Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from .config import EVALUATION_PROTOCOL_VERSION
from .db import Base


class Setting(Base):
    __tablename__ = "settings"
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JSON, default=dict)


class Experiment(Base):
    """研究任务 (一次独立实验运行): 挖掘产物按 experiment_id 隔离."""

    __tablename__ = "experiments"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="open")  # open/archived
    # 研究任务的完整、可复现配置；历史任务只读保留，不依赖当前环境变量重建。
    research_config: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class MinerVersion(Base):
    """内层 Miner 的一个版本 = 一份 HarnessSpec (声明式白名单)"""

    __tablename__ = "miner_versions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    version_no: Mapped[int] = mapped_column(Integer, index=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("miner_versions.id"), nullable=True)
    harness_spec: Mapped[dict] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(16), default="candidate")  # incumbent/candidate/rejected
    meta_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    proposal_note: Mapped[str] = mapped_column(Text, default="")
    evaluation_protocol: Mapped[str] = mapped_column(
        String(32), default=EVALUATION_PROTOCOL_VERSION, index=True
    )
    feedback_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    reflection: Mapped[dict] = mapped_column(JSON, default=dict)
    context_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class OuterStep(Base):
    __tablename__ = "outer_steps"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    step_no: Mapped[int] = mapped_column(Integer, index=True)
    candidate_id: Mapped[int] = mapped_column(ForeignKey("miner_versions.id"))
    incumbent_id: Mapped[int] = mapped_column(ForeignKey("miner_versions.id"))
    candidate_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    incumbent_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    evaluation_protocol: Mapped[str] = mapped_column(
        String(32), default=EVALUATION_PROTOCOL_VERSION, index=True
    )
    context_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    detail: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Node(Base):
    """内层搜索树节点 = 一次因子尝试"""

    __tablename__ = "nodes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    miner_version_id: Mapped[int] = mapped_column(ForeignKey("miner_versions.id"), index=True)
    outer_step_no: Mapped[int] = mapped_column(Integer, index=True, default=0)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("nodes.id"), nullable=True)
    op: Mapped[str] = mapped_column(String(16))  # draft/improve/debug
    expression: Mapped[str] = mapped_column(Text)
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok/error
    error: Mapped[str] = mapped_column(Text, default="")
    evaluation_protocol: Mapped[str] = mapped_column(
        String(32), default=EVALUATION_PROTOCOL_VERSION, index=True
    )
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    public_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    public_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_metrics: Mapped[dict] = mapped_column(JSON, default=dict)  # META_TRAIN, 不进提示词
    proposal_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    feedback_summary: Mapped[dict] = mapped_column(JSON, default=dict)
    source: Mapped[str] = mapped_column(String(16), default="llm")  # llm/random
    task_name: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Factor(Base):
    __tablename__ = "factors"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    name: Mapped[str] = mapped_column(String(128))
    expression: Mapped[str] = mapped_column(Text)  # 唯一性改为实验内注册时检查
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(32), default="public-leading", index=True)
    node_id: Mapped[int | None] = mapped_column(ForeignKey("nodes.id"), nullable=True)
    task_name: Mapped[str] = mapped_column(String(64), default="")
    public_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    evaluation_protocol: Mapped[str] = mapped_column(String(32), default="legacy_unoriented", index=True)
    lifecycle_stage: Mapped[str] = mapped_column(String(64), default="legacy_unreviewed", index=True)
    provenance_status: Mapped[str] = mapped_column(String(64), default="unverified", index=True)
    validation_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    eligibility: Mapped[dict] = mapped_column(JSON, default=dict)
    fingerprint: Mapped[dict] = mapped_column(JSON, default=dict)
    research_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    evaluated_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Trial(Base):
    """全局试验登记簿 (多重检验记账, append-only)"""

    __tablename__ = "trials"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    expression_hash: Mapped[str] = mapped_column(String(64), index=True)
    layer: Mapped[str] = mapped_column(String(24))
    task_name: Mapped[str] = mapped_column(String(64), default="")
    evaluation_protocol: Mapped[str] = mapped_column(
        String(32), default=EVALUATION_PROTOCOL_VERSION, index=True
    )
    statistic: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class LLMCallAudit(Base):
    """Secret-safe, append-only trace of the rendered LLM research context."""

    __tablename__ = "llm_call_audits"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    role: Mapped[str] = mapped_column(String(32), index=True)
    phase: Mapped[str] = mapped_column(String(64), default="")
    status: Mapped[str] = mapped_column(String(16), index=True)
    provider_name: Mapped[str] = mapped_column(String(128), default="")
    model: Mapped[str] = mapped_column(String(128), default="")
    evaluation_protocol: Mapped[str] = mapped_column(
        String(32), default=EVALUATION_PROTOCOL_VERSION, index=True
    )
    miner_version_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    outer_step_no: Mapped[int | None] = mapped_column(Integer, nullable=True)
    task_name: Mapped[str] = mapped_column(String(64), default="")
    prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
    feedback_fingerprint: Mapped[str] = mapped_column(String(64), default="")
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    user_prompt: Mapped[str] = mapped_column(Text, default="")
    response: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    trace_meta: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Backtest(Base):
    __tablename__ = "backtests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    experiment_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="done")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ScreenerRun(Base):
    """Append-only, task-scoped snapshot of one completed stock selection."""

    __tablename__ = "screener_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id"), index=True
    )
    schema_version: Mapped[str] = mapped_column(
        String(32), default="screener_run_v1", index=True
    )
    market: Mapped[str] = mapped_column(String(16), index=True)
    portfolio_mode: Mapped[str] = mapped_column(String(24))
    target_date: Mapped[date] = mapped_column(Date, index=True)
    requested_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    direction: Mapped[str] = mapped_column(String(16))
    panel_identity: Mapped[str] = mapped_column(Text, default="")
    request_spec: Mapped[dict] = mapped_column(JSON, default=dict)
    result_snapshot: Mapped[dict] = mapped_column(JSON, default=dict)
    factor_count: Mapped[int] = mapped_column(Integer, default=0)
    eligible_count: Mapped[int] = mapped_column(Integer, default=0)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    cache_hit: Mapped[bool] = mapped_column(Boolean, default=False)
    elapsed_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="done", index=True)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EngineEvent(Base):
    __tablename__ = "engine_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(8), default="info")
    message: Mapped[str] = mapped_column(Text)
    experiment_id: Mapped[int | None] = mapped_column(Integer, index=True, nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
