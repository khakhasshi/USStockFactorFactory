import os

_MARKET = os.environ.get("FF_MARKET", "us")  # "us" 或 "ashare"
MARKET_LABEL = "A股" if _MARKET == "ashare" else "美股"

PORT = int(os.environ.get("FF_PORT", "10010"))
DEFAULT_PORTFOLIO_MODE = "long_only" if _MARKET == "ashare" else "long_short"
DATABASE_URL = os.environ.get(
    "FF_DATABASE_URL",
    "postgresql+asyncpg://jiangjingzhe@localhost:5432/factor_factory",
)
PANEL_GLOB = os.environ.get(
    "FF_PANEL_GLOB",
    "/Users/jiangjingzhe/Portfolios/MultiFactorUS/data_yfinance_research/"
    "processed/daily_panel/trade_year=*/data_0.parquet",
)
US_PANEL_GLOB = (
    "/Users/jiangjingzhe/Portfolios/MultiFactorUS/data_yfinance_research/"
    "processed/daily_panel/trade_year=*/data_0.parquet"
)
ASHARE_PANEL_GLOB = "/Users/jiangjingzhe/Portfolios/MultiFactorAshare/data/trade_year=*/data_0.parquet"


def default_panel_glob(market: str | None = None) -> str:
    """Use a market-specific default; never inherit the process boot market."""
    if market == "ashare":
        return ASHARE_PANEL_GLOB
    if market == "us":
        return US_PANEL_GLOB
    return PANEL_GLOB

# ---- 四级数据隔离边界 ----
if _MARKET == "ashare":
    LAYER_BOUNDS = {
        "INNER_PUBLIC": ("2010-01-01", "2019-12-31"),
        "META_TRAIN": ("2020-01-01", "2022-12-31"),
        "META_HOLDOUT": ("2023-01-01", "2024-12-31"),
        "FACTOR_VAULT": ("2025-01-01", "2026-08-04"),
    }
    # A股 DSL 字段: 价量 + 估值 + 市值 + 流动性 + 资金流向
    DSL_FIELDS = [
        # 价量
        "open", "high", "low", "close", "vol", "amount",
        # 估值
        "pe_ttm", "pb", "ps_ttm", "dv_ttm",
        # 市值
        "total_mv", "circ_mv",
        # 流动性
        "turnover_rate", "volume_ratio",
        # 资金流向
        "net_mf_amount",
        "buy_lg_amount", "sell_lg_amount",
        "buy_elg_amount", "sell_elg_amount",
        # 股本
        "float_share",
    ]
    ASHARE_DSL_FIELDS = DSL_FIELDS
    US_DSL_FIELDS = ["open", "high", "low", "close", "vol", "amount"]
else:
    LAYER_BOUNDS = {
        "INNER_PUBLIC": ("2010-06-01", "2019-12-31"),
        "META_TRAIN": ("2020-01-01", "2022-12-31"),
        "META_HOLDOUT": ("2023-01-01", "2024-12-31"),
        "FACTOR_VAULT": ("2025-01-01", "2026-07-31"),
    }
    DSL_FIELDS = ["open", "high", "low", "close", "vol", "amount"]
    US_DSL_FIELDS = DSL_FIELDS
    ASHARE_DSL_FIELDS = [
        "open", "high", "low", "close", "vol", "amount",
        "pe_ttm", "pb", "ps_ttm", "dv_ttm", "total_mv", "circ_mv",
        "turnover_rate", "volume_ratio", "net_mf_amount",
        "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount",
        "float_share",
    ]


def get_dsl_fields(market: str | None = None) -> list[str]:
    """Return fields for a task, independent of the process-wide default market."""
    return list(ASHARE_DSL_FIELDS if market == "ashare" else US_DSL_FIELDS if market == "us" else DSL_FIELDS)

# ---- 旧版 HarnessSpec (保留兼容, A组运行中) ----
DEFAULT_HARNESS_SPEC = {
    "n_drafts": 4,
    "improve_bias": 0.65,
    "context_top_k": 5,
    "llm_temperature": 0.9,
    "anti_overfit_instruction": True,
    "min_public_icir": 0.25,
}

# ---- 新版 MinerTemplate (B组: 外层可改写代码级对象) ----
# 外层 LLM 可以自由改写此模板中的任何文本字段;
# 只读边界由 MetaValidator 强制执行 (评估器/数据层/隔离边界不可触碰)
DEFAULT_MINER_TEMPLATE = {
    # === 外层可改写 ===
    "system_prompt": (
        "你是量化因子研究员。基于当前市场日线数据设计横截面选股因子表达式。\n"
        "可用字段: {fields} (前复权价格与量额)\n"
        "可用算子:\n{ops}\n"
        "规则: 只能用以上字段与算子; 窗口为 1..250 整数; 表达式一行;\n"
        "目标是最大化样本内 RankIC 的稳健性而非峰值;\n"
        "禁止只对特定时段有效的取巧构造。{anti}\n"
        "只回复 JSON: {{\"expression\": \"...\", \"hypothesis\": \"一句话经济学假设\"}}"
    ),
    "anti_overfit_instruction": (
        "特别要求: 避免过拟合——偏好简单、有经济含义、跨行业普适的结构。"
    ),
    "draft_strategy": (
        "从不同经济学机制出发提出新因子: "
        "动量(趋势跟随)、反转(均值回归)、波动(低波异象)、"
        "流动性(非流动性溢价)、量价关系(聪明钱流向)。"
        "每个因子陈述经济学假设, 优先使用低频窗口(20-120日)降低换手。"
    ),
    "improve_strategy": (
        "基于当前最优因子改进: "
        "1) 加权复合两个低相关因子; 2) 替换算子(如 ts_corr→ts_rank); "
        "3) 调整窗口长度; 4) 引入截面归一化(zscore/winsor/rank); "
        "5) 方向翻转(若IC符号与假设相反)。"
    ),
    "context_strategy": (
        "展示历史 top-{top_k} 高分因子(含 public score/ICIR/换手/表达式)。"
        "若存在多次失败(score<0.3)的因子, 归纳其失败模式为一句话警告。"
    ),
    "diversity_instruction": (
        "新因子必须与历史高分因子有不同经济学机制。"
    ),
    "scoring_weights": {
        "icir_weight": 0.45,
        "consistency_weight": 0.25,
        "turnover_weight": 0.30,  # 换手惩罚权重, 越高越偏好慢信号
    },
    "dsl_exploration_templates": [
        "{-}rank(ts_delta(close, {window}))",
        "ts_corr({field1}, {field2}, {window})",
        "{-}zscore(ts_std({field}, {window}))",
        "rank((close - ts_min(low, {window})) / (ts_max(high, {window}) - ts_min(low, {window})))",
        "ts_mean(abs(ts_delta(close,1))/(amount+1e-9), {window})",
    ],
    "min_public_icir": 0.25,
    "llm_temperature": 0.9,

    # === 只读元数据 (外层不可改写, 由系统注入) ===
    "_readonly": {
        "evaluator": "harness.py:evaluate() — 只读",
        "data_layer": "AsOfResearchView panel — 只读",
        "isolation_layers": "INNER_PUBLIC/META_TRAIN/META_HOLDOUT/FACTOR_VAULT — 只读",
        "acceptance_gate": "t-test p<0.10 across seeds — 只读",
        "fields": ["open", "high", "low", "close", "vol", "amount"],
    },
}

# ---- 新版引擎配置 (B组: 高预算 + 多种子) ----
DEFAULT_ENGINE_CONFIG_V2 = {
    "inner_budget_per_outer_step": 20,    # 快速验证: 20 (完整实验: 50)
    "n_seeds_per_candidate": 2,            # 快速验证: 2 (完整实验: 3)
    "outer_accept_p_value": 0.10,          # 配对 t 检验接受阈值
    "incumbent_remeasure_every": 3,        # 每 3 步重测在位者
    "incumbent_remeasure_budget": 30,      # 重测时用 30 次评估 (节省算力)
    "tasks": [
        {"name": "T1_liquid500_5d", "universe_n": 500, "horizon": 5, "cost_bps": 15, "mode": DEFAULT_PORTFOLIO_MODE},
        {"name": "T2_mid1500_10d", "universe_n": 1500, "horizon": 10, "cost_bps": 25, "mode": DEFAULT_PORTFOLIO_MODE},
        {"name": "T3_liquid500_20d", "universe_n": 500, "horizon": 20, "cost_bps": 15, "mode": DEFAULT_PORTFOLIO_MODE},
    ],
}

# ---- 旧版引擎配置 (A组兼容) ----
DEFAULT_ENGINE_CONFIG = {
    "inner_budget_per_outer_step": 10,
    "outer_accept_epsilon": 0.02,
    "incumbent_remeasure_every": 5,
    "tasks": [
        {"name": "T1_liquid500_5d", "universe_n": 500, "horizon": 5, "cost_bps": 15, "mode": DEFAULT_PORTFOLIO_MODE},
        {"name": "T2_mid1500_10d", "universe_n": 1500, "horizon": 10, "cost_bps": 25, "mode": DEFAULT_PORTFOLIO_MODE},
        {"name": "T3_liquid500_20d", "universe_n": 500, "horizon": 20, "cost_bps": 15, "mode": DEFAULT_PORTFOLIO_MODE},
    ],
}
