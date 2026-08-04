import os

PORT = int(os.environ.get("FF_PORT", "9999"))
DATABASE_URL = os.environ.get(
    "FF_DATABASE_URL",
    "postgresql+asyncpg://jiangjingzhe@localhost:5432/factor_factory",
)
PANEL_GLOB = os.environ.get(
    "FF_PANEL_GLOB",
    "/Users/jiangjingzhe/Portfolios/MultiFactorUS/data_yfinance_research/"
    "processed/daily_panel/trade_year=*/data_0.parquet",
)

# 四级数据隔离边界 (ProtocolGeneration G1, 冻结)
LAYER_BOUNDS = {
    "INNER_PUBLIC": ("2010-06-01", "2019-12-31"),
    "META_TRAIN": ("2020-01-01", "2022-12-31"),
    "META_HOLDOUT": ("2023-01-01", "2024-12-31"),
    "FACTOR_VAULT": ("2025-01-01", "2026-07-31"),
}

# DSL 可见字段白名单 (仅前复权研究字段 + 量额)
DSL_FIELDS = ["open", "high", "low", "close", "vol", "amount"]

DEFAULT_HARNESS_SPEC = {
    "n_drafts": 4,            # 每轮先起草的多样化因子数
    "improve_bias": 0.65,     # 选择 improve 而非 draft 的概率
    "context_top_k": 5,       # 提示词中展示的历史最优尝试数
    "llm_temperature": 0.9,
    "anti_overfit_instruction": True,
    "min_public_icir": 0.25,  # 注册进因子库的 public 门槛
}

DEFAULT_ENGINE_CONFIG = {
    "inner_budget_per_outer_step": 10,   # 每个外层步的内层评估次数 (成本预算代理)
    "outer_accept_epsilon": 0.02,        # 外层接受门槛 (超出在位者的最小幅度)
    "incumbent_remeasure_every": 5,      # 每 N 步重测在位者 (noise band)
    "tasks": [
        {"name": "T1_liquid500_5d", "universe_n": 500, "horizon": 5, "cost_bps": 15},
        {"name": "T2_mid1500_10d", "universe_n": 1500, "horizon": 10, "cost_bps": 25},
        {"name": "T3_liquid500_20d", "universe_n": 500, "horizon": 20, "cost_bps": 15},
    ],
}
