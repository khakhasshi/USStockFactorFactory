import logging
import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import DATABASE_URL


class Base(DeclarativeBase):
    pass


engine = create_async_engine(DATABASE_URL, pool_size=10, max_overflow=5)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
logger = logging.getLogger("database.migration")


async def init_db() -> None:
    from sqlalchemy import text

    from . import models  # noqa: F401  (注册表元数据)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        # 轻量迁移: 实验隔离列 + 放开表达式全局唯一 (改为实验内注册时检查)
        for tbl in ("miner_versions", "outer_steps", "nodes", "factors", "trials"):
            await conn.execute(text(
                f"ALTER TABLE {tbl} ADD COLUMN IF NOT EXISTS experiment_id INTEGER NOT NULL DEFAULT 1"
            ))
        await conn.execute(text(
            "ALTER TABLE factors DROP CONSTRAINT IF EXISTS factors_expression_key"
        ))
        await conn.execute(text(
            "ALTER TABLE experiments ADD COLUMN IF NOT EXISTS research_config JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        # SQLAlchemy's generic JSON type creates JSON on a brand-new PostgreSQL
        # database, while later migrations use JSONB containment operators.
        # Existing installations are already JSONB; this conversion is
        # lossless and idempotent for both bootstrap paths.
        await conn.execute(text(
            "ALTER TABLE experiments ALTER COLUMN research_config TYPE JSONB "
            "USING research_config::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE backtests ADD COLUMN IF NOT EXISTS experiment_id INTEGER NOT NULL DEFAULT 1"
        ))
        await conn.execute(text(
            "ALTER TABLE engine_events ADD COLUMN IF NOT EXISTS experiment_id INTEGER"
        ))
        await conn.execute(text(
            "ALTER TABLE engine_events ADD COLUMN IF NOT EXISTS payload JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE factors ADD COLUMN IF NOT EXISTS research_meta JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE factors ADD COLUMN IF NOT EXISTS evaluation_protocol VARCHAR(32) "
            "NOT NULL DEFAULT 'legacy_unoriented'"
        ))
        await conn.execute(text(
            "ALTER TABLE factors ADD COLUMN IF NOT EXISTS lifecycle_stage VARCHAR(64) "
            "NOT NULL DEFAULT 'legacy_unreviewed'"
        ))
        await conn.execute(text(
            "ALTER TABLE factors ADD COLUMN IF NOT EXISTS provenance_status VARCHAR(64) "
            "NOT NULL DEFAULT 'unverified'"
        ))
        # Some explicit re-audit states are longer than the original 32-char
        # columns.  Widening is lossless and keeps historical labels intact.
        await conn.execute(text(
            "ALTER TABLE factors ALTER COLUMN lifecycle_stage TYPE VARCHAR(64)"
        ))
        await conn.execute(text(
            "ALTER TABLE factors ALTER COLUMN provenance_status TYPE VARCHAR(64)"
        ))
        await conn.execute(text(
            "ALTER TABLE factors ADD COLUMN IF NOT EXISTS validation_metrics JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE factors ADD COLUMN IF NOT EXISTS eligibility JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE factors ADD COLUMN IF NOT EXISTS evaluated_at TIMESTAMP"
        ))
        await conn.execute(text(
            "ALTER TABLE miner_versions ADD COLUMN IF NOT EXISTS evaluation_protocol "
            "VARCHAR(32) NOT NULL DEFAULT 'legacy_unoriented'"
        ))
        await conn.execute(text(
            "ALTER TABLE miner_versions ADD COLUMN IF NOT EXISTS feedback_summary "
            "JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE miner_versions ADD COLUMN IF NOT EXISTS reflection "
            "JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE miner_versions ADD COLUMN IF NOT EXISTS context_fingerprint "
            "VARCHAR(64) NOT NULL DEFAULT ''"
        ))
        await conn.execute(text(
            "ALTER TABLE outer_steps ADD COLUMN IF NOT EXISTS evaluation_protocol "
            "VARCHAR(32) NOT NULL DEFAULT 'legacy_unoriented'"
        ))
        await conn.execute(text(
            "ALTER TABLE outer_steps ADD COLUMN IF NOT EXISTS context_fingerprint "
            "VARCHAR(64) NOT NULL DEFAULT ''"
        ))
        await conn.execute(text(
            "ALTER TABLE nodes ADD COLUMN IF NOT EXISTS evaluation_protocol "
            "VARCHAR(32) NOT NULL DEFAULT 'legacy_unoriented'"
        ))
        await conn.execute(text(
            "ALTER TABLE nodes ADD COLUMN IF NOT EXISTS seed INTEGER"
        ))
        await conn.execute(text(
            "ALTER TABLE nodes ADD COLUMN IF NOT EXISTS proposal_meta "
            "JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE nodes ADD COLUMN IF NOT EXISTS feedback_summary "
            "JSONB NOT NULL DEFAULT '{}'::jsonb"
        ))
        await conn.execute(text(
            "ALTER TABLE trials ADD COLUMN IF NOT EXISTS evaluation_protocol "
            "VARCHAR(32) NOT NULL DEFAULT 'legacy_unoriented'"
        ))
        for statement in (
            "ALTER TABLE trials ADD COLUMN IF NOT EXISTS node_id INTEGER",
            "ALTER TABLE trials ADD COLUMN IF NOT EXISTS parent_node_id INTEGER",
            "ALTER TABLE trials ADD COLUMN IF NOT EXISTS expression TEXT NOT NULL DEFAULT ''",
            "ALTER TABLE trials ADD COLUMN IF NOT EXISTS search_method VARCHAR(64) NOT NULL DEFAULT ''",
            "ALTER TABLE trials ADD COLUMN IF NOT EXISTS mechanism VARCHAR(64) NOT NULL DEFAULT ''",
            "ALTER TABLE trials ADD COLUMN IF NOT EXISTS selected BOOLEAN NOT NULL DEFAULT FALSE",
            "ALTER TABLE trials ADD COLUMN IF NOT EXISTS failure_reason TEXT NOT NULL DEFAULT ''",
            "CREATE INDEX IF NOT EXISTS ix_trials_node_id ON trials (node_id)",
            "CREATE INDEX IF NOT EXISTS ix_trials_search_method ON trials (search_method)",
            "CREATE INDEX IF NOT EXISTS ix_trials_mechanism ON trials (mechanism)",
            "CREATE INDEX IF NOT EXISTS ix_trials_selected ON trials (selected)",
        ):
            await conn.execute(text(statement))
        # Recover protocol lineage from immutable payloads without changing
        # historical scores or deleting interrupted work.
        await conn.execute(text(
            "UPDATE nodes SET evaluation_protocol = "
            "COALESCE(NULLIF(public_metrics->>'protocol_version', ''), 'legacy_unoriented') "
            "WHERE evaluation_protocol = 'legacy_unoriented' "
            "AND NULLIF(public_metrics->>'protocol_version', '') IS NOT NULL"
        ))
        await conn.execute(text(
            "UPDATE trials SET evaluation_protocol = "
            "COALESCE(NULLIF(statistic->>'protocol_version', ''), 'legacy_unoriented') "
            "WHERE evaluation_protocol = 'legacy_unoriented' "
            "AND NULLIF(statistic->>'protocol_version', '') IS NOT NULL"
        ))
        await conn.execute(text(
            "UPDATE outer_steps SET evaluation_protocol = "
            "COALESCE(NULLIF(detail->>'protocol_version', ''), 'legacy_unoriented') "
            "WHERE evaluation_protocol = 'legacy_unoriented' "
            "AND NULLIF(detail->>'protocol_version', '') IS NOT NULL"
        ))
        await conn.execute(text(
            "UPDATE miner_versions AS m SET evaluation_protocol = 'v4.0' "
            "WHERE m.evaluation_protocol = 'legacy_unoriented' "
            "AND EXISTS ("
            "  SELECT 1 FROM nodes n WHERE n.miner_version_id = m.id "
            "  AND n.evaluation_protocol = 'v4.0'"
            ") "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM nodes n WHERE n.miner_version_id = m.id "
            "  AND n.evaluation_protocol <> 'v4.0'"
            ")"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_factors_evaluation_protocol ON factors (evaluation_protocol)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_factors_lifecycle_stage ON factors (lifecycle_stage)"
        ))
        await conn.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_factors_provenance_status ON factors (provenance_status)"
        ))
        for statement in (
            "CREATE INDEX IF NOT EXISTS ix_miner_versions_evaluation_protocol "
            "ON miner_versions (evaluation_protocol)",
            "CREATE INDEX IF NOT EXISTS ix_outer_steps_evaluation_protocol "
            "ON outer_steps (evaluation_protocol)",
            "CREATE INDEX IF NOT EXISTS ix_nodes_evaluation_protocol "
            "ON nodes (evaluation_protocol)",
            "CREATE INDEX IF NOT EXISTS ix_trials_evaluation_protocol "
            "ON trials (evaluation_protocol)",
            "CREATE INDEX IF NOT EXISTS ix_nodes_feedback_lookup "
            "ON nodes (experiment_id, evaluation_protocol, miner_version_id, task_name, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_miner_versions_protocol_status "
            "ON miner_versions (experiment_id, evaluation_protocol, status, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_outer_steps_protocol_step "
            "ON outer_steps (experiment_id, evaluation_protocol, step_no)",
            "CREATE INDEX IF NOT EXISTS ix_trials_protocol_task "
            "ON trials (experiment_id, evaluation_protocol, task_name, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_llm_call_audits_experiment_id_desc "
            "ON llm_call_audits (experiment_id, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_screener_runs_experiment_id_desc "
            "ON screener_runs (experiment_id, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_screener_runs_experiment_date_desc "
            "ON screener_runs (experiment_id, target_date DESC, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_combination_experiments_experiment_id_desc "
            "ON combination_experiments (experiment_id, id DESC)",
            "CREATE INDEX IF NOT EXISTS ix_combination_experiments_status "
            "ON combination_experiments (status, id DESC)",
        ):
            await conn.execute(text(statement))
        await conn.execute(text(
            "UPDATE combination_experiments SET status = 'interrupted', "
            "error = CASE WHEN error = '' THEN 'service restarted during execution' ELSE error END, "
            "completed_at = COALESCE(completed_at, NOW()) "
            "WHERE status IN ('queued', 'running')"
        ))
        # Preserve contaminated historical expressions while preventing them
        # from appearing as validated research assets.
        await conn.execute(text(
            "UPDATE factors AS f "
            "SET provenance_status = 'invalid_historical_panel', "
            "    lifecycle_stage = 'invalid_provenance' "
            "FROM experiments AS e "
            "WHERE f.experiment_id = e.id "
            "  AND e.research_config ? 'provenance_warning' "
            "  AND f.evaluation_protocol = 'legacy_unoriented'"
        ))
        await conn.execute(text(
            "INSERT INTO experiments (id, name, description, status, research_config) VALUES "
            "(1, '实验1-初始双层挖掘', '2026-08 首轮: 旧评分函数(exp换手衰减, 无退化检测), 328因子/31外层步; 已冻结存档', 'archived', '{}'::jsonb) "
            "ON CONFLICT (id) DO NOTHING"
        ))
        await conn.execute(text(
            "SELECT setval('experiments_id_seq', (SELECT COALESCE(MAX(id),1) FROM experiments))"
        ))
    if os.environ.get("FF_SKIP_HISTORICAL_FEEDBACK_BACKFILL", "0") != "1":
        await _backfill_v4_feedback()


async def _backfill_v4_feedback() -> int:
    """Populate a bounded batch of safe envelopes for pre-migration nodes.

    Loading every historical JSON payload delayed each independent service
    startup by minutes.  A bounded, idempotent batch keeps startup predictable;
    subsequent starts continue from the remaining empty rows.
    """
    from sqlalchemy import select, text

    from .config import EVALUATION_PROTOCOL_VERSION
    from .feedback import build_feedback_envelope
    from .models import Experiment, Node

    async with SessionLocal() as session:
        nodes = (
            await session.scalars(
                select(Node)
                .where(
                    Node.evaluation_protocol == EVALUATION_PROTOCOL_VERSION,
                    text(
                        "COALESCE(nodes.feedback_summary::jsonb, '{}'::jsonb) "
                        "= '{}'::jsonb"
                    ),
                )
                .order_by(Node.id)
                .limit(500)
            )
        ).all()
        if not nodes:
            return 0
        experiments: dict[int, Experiment | None] = {}
        written = 0
        for node in nodes:
            if node.experiment_id not in experiments:
                experiments[node.experiment_id] = await session.get(
                    Experiment,
                    node.experiment_id,
                )
            experiment = experiments[node.experiment_id]
            config = dict(experiment.research_config or {}) if experiment else {}
            market = str(config.get("market") or "us")
            portfolio_mode = str(
                config.get("portfolio_mode")
                or ("long_only" if market == "ashare" else "long_short")
            )
            try:
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
                    market=market,
                    portfolio_mode=portfolio_mode,
                    direction=int(config.get("direction", 1)),
                    proposal_meta=node.proposal_meta,
                )
                written += 1
            except Exception as exc:  # noqa: BLE001 - preserve source history
                logger.warning(
                    "V4 节点 %s 反馈信封回填失败，原记录保持不变: %s",
                    node.id,
                    str(exc)[:300],
                )
        if written:
            await session.commit()
        return written


async def get_active_experiment_id() -> int:
    import os
    if override := os.environ.get("FF_EXPERIMENT_ID"):
        return int(override)
    from sqlalchemy import select

    from .config import SERVICE_ARCHITECTURE, SERVICE_INSTANCE
    from .models import Experiment, Setting

    async with SessionLocal() as s:
        key = active_experiment_setting_key()
        row = await s.get(Setting, key)
        if row:
            experiment_id = int(row.value.get("id", 1))
            if not SERVICE_ARCHITECTURE:
                return experiment_id
            experiment = await s.get(Experiment, experiment_id)
            if (
                experiment is not None
                and experiment.status == "open"
                and str(
                    (experiment.research_config or {}).get("service_instance")
                    or ""
                )
                == SERVICE_INSTANCE
            ):
                return experiment_id
        if SERVICE_ARCHITECTURE:
            rows = list(
                (
                    await s.scalars(
                        select(Experiment)
                        .where(Experiment.status == "open")
                        .order_by(Experiment.id)
                    )
                ).all()
            )
            for experiment in rows:
                if str(
                    (experiment.research_config or {}).get("service_instance")
                    or ""
                ) == SERVICE_INSTANCE:
                    return experiment.id
        return int(row.value.get("id", 1)) if row else 1


def active_experiment_setting_key() -> str:
    from .config import SERVICE_ARCHITECTURE, SERVICE_INSTANCE

    if SERVICE_ARCHITECTURE:
        return f"active_experiment:{SERVICE_INSTANCE}"[:64]
    return "active_experiment"
