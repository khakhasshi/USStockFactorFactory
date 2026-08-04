from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

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
    public_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    public_metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    gate_metrics: Mapped[dict] = mapped_column(JSON, default=dict)  # META_TRAIN, 不进提示词
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
    fingerprint: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Trial(Base):
    """全局试验登记簿 (多重检验记账, append-only)"""

    __tablename__ = "trials"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[int] = mapped_column(Integer, index=True, default=1)
    expression_hash: Mapped[str] = mapped_column(String(64), index=True)
    layer: Mapped[str] = mapped_column(String(24))
    task_name: Mapped[str] = mapped_column(String(64), default="")
    statistic: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Backtest(Base):
    __tablename__ = "backtests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="done")
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class EngineEvent(Base):
    __tablename__ = "engine_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    level: Mapped[str] = mapped_column(String(8), default="info")
    message: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
