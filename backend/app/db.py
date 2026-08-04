from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from .config import DATABASE_URL


class Base(DeclarativeBase):
    pass


engine = create_async_engine(DATABASE_URL, pool_size=10, max_overflow=5)
SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


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
            "INSERT INTO experiments (id, name, description, status) VALUES "
            "(1, '实验1-初始双层挖掘', '2026-08 首轮: 旧评分函数(exp换手衰减, 无退化检测), 328因子/31外层步; 已冻结存档', 'archived') "
            "ON CONFLICT (id) DO NOTHING"
        ))
        await conn.execute(text(
            "SELECT setval('experiments_id_seq', (SELECT COALESCE(MAX(id),1) FROM experiments))"
        ))


async def get_active_experiment_id() -> int:
    from .models import Setting

    async with SessionLocal() as s:
        row = await s.get(Setting, "active_experiment")
        return int(row.value.get("id", 1)) if row else 1
