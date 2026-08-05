import ipaddress
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

_MARKET = os.environ.get("FF_MARKET", "us")  # "us" 或 "ashare"
MARKET_LABEL = "A股" if _MARKET == "ashare" else "美股"

HOST = os.environ.get("FF_HOST", "127.0.0.1").strip() or "127.0.0.1"
PORT = int(os.environ.get("FF_PORT", "10010"))
ALLOW_REMOTE_UNAUTHENTICATED = os.environ.get(
    "FF_ALLOW_REMOTE_UNAUTHENTICATED",
    "",
).strip().lower() in {"1", "true", "yes", "on"}
BACKTEST_ARTIFACT_ROOT = Path(
    os.environ.get(
        "FF_BACKTEST_ARTIFACT_ROOT",
        str(Path(__file__).resolve().parents[2] / "var" / "backtests"),
    )
).resolve()
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


def is_loopback_host(host: str = HOST) -> bool:
    """Return whether a bind address is local-only."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def default_panel_glob(market: str | None = None) -> str:
    """Use a market-specific default; never inherit the process boot market."""
    if market == "ashare":
        return ASHARE_PANEL_GLOB
    if market == "us":
        return US_PANEL_GLOB
    return PANEL_GLOB

# ---- 四级数据隔离边界 ----
MARKET_LAYER_BOUNDS = {
    "ashare": {
        "INNER_PUBLIC": ("2010-01-01", "2019-12-31"),
        "META_TRAIN": ("2020-01-01", "2022-12-31"),
        "META_HOLDOUT": ("2023-01-01", "2024-12-31"),
        "FACTOR_VAULT": ("2025-01-01", "2026-08-04"),
    },
    "us": {
        "INNER_PUBLIC": ("2010-06-01", "2019-12-31"),
        "META_TRAIN": ("2020-01-01", "2022-12-31"),
        "META_HOLDOUT": ("2023-01-01", "2024-12-31"),
        "FACTOR_VAULT": ("2025-01-01", "2026-08-04"),
    },
}
LAYER_BOUNDS = MARKET_LAYER_BOUNDS[_MARKET]


def get_layer_bounds(market: str | None = None) -> dict[str, tuple[str, str]]:
    """Return immutable chronological splits for one task market."""
    return dict(MARKET_LAYER_BOUNDS.get(market or _MARKET, MARKET_LAYER_BOUNDS["us"]))


if _MARKET == "ashare":
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


# ---- Evaluation Protocol V4 ----
# V4 keeps the mining score isolated from HOLDOUT/VAULT and adds a separate
# live-oriented audit rank.  PIT is deliberately outside this score at the
# user's request; the API keeps the panel's NON_PIT label visible separately.
EVALUATION_PROTOCOL_VERSION = "v4.0"
DEFAULT_EVALUATION_CONFIG = {
    "protocol_version": EVALUATION_PROTOCOL_VERSION,
    "top_fraction": 0.20,
    "tail_fraction": 0.20,
    "min_coverage": 0.70,
    "min_layer_days": 120,
    "min_research_score": 1.00,
    "min_oos_sharpe": 0.50,
    "min_stress_sharpe": 0.00,
    "min_return_hac_t": 1.2816,
    "max_hac_p_value": 0.10,
    "min_era_consistency": 0.60,
    "min_profitable_era_rate": 0.60,
    "min_monotonicity": 0.35,
    "min_cost_cushion_multiple": 1.50,
    "max_drawdown": 0.35,
    "max_market_beta_long_short": 0.35,
    "max_daily_turnover": {"ashare": 0.35, "us": 0.50},
    "base_cost_bps": {"ashare": 20.0, "us": 15.0},
    "stress_cost_bps": {"ashare": [10.0, 20.0, 35.0, 50.0], "us": [5.0, 15.0, 25.0, 40.0]},
    "borrow_cost_bps_annual": {"ashare": 0.0, "us": 300.0},
    "stress_borrow_cost_bps_annual": {"ashare": 0.0, "us": 600.0},
    "target_capital": {"ashare": 10_000_000.0, "us": 1_000_000.0},
    "max_adv_participation": 0.05,
    # Rank calibration targets are scale anchors, not promotion promises.
    "target_rank_sharpe": 1.50,
    "target_rank_ann_return": {"ashare": 0.10, "us": 0.12},
    "target_absolute_ann_return": {"ashare": 0.15, "us": 0.12},
    "target_cost_cushion_multiple": 3.00,
    "return_lcb_confidence": 0.90,
    # Pre-declared search budget.  Explicit audits use max(this, actual trials).
    "multiple_testing_trials": 1000,
    "multiple_testing_alpha": 0.10,
}


def evaluation_config(market: str, overrides: dict | None = None) -> dict:
    """Resolve market-specific V4 settings while preserving a serialisable snapshot."""
    if market not in {"ashare", "us"}:
        raise ValueError("market 必须是 ashare 或 us")
    src = DEFAULT_EVALUATION_CONFIG
    cfg = {
        "protocol_version": src["protocol_version"],
        "top_fraction": src["top_fraction"],
        "tail_fraction": src["tail_fraction"],
        "min_coverage": src["min_coverage"],
        "min_layer_days": src["min_layer_days"],
        "min_research_score": src["min_research_score"],
        "min_oos_sharpe": src["min_oos_sharpe"],
        "min_stress_sharpe": src["min_stress_sharpe"],
        "min_return_hac_t": src["min_return_hac_t"],
        "max_hac_p_value": src["max_hac_p_value"],
        "min_era_consistency": src["min_era_consistency"],
        "min_profitable_era_rate": src["min_profitable_era_rate"],
        "min_monotonicity": src["min_monotonicity"],
        "min_cost_cushion_multiple": src["min_cost_cushion_multiple"],
        "max_drawdown": src["max_drawdown"],
        "max_market_beta_long_short": src["max_market_beta_long_short"],
        "max_daily_turnover": src["max_daily_turnover"][market],
        "base_cost_bps": src["base_cost_bps"][market],
        "stress_cost_bps": list(src["stress_cost_bps"][market]),
        "borrow_cost_bps_annual": src["borrow_cost_bps_annual"][market],
        "stress_borrow_cost_bps_annual": src["stress_borrow_cost_bps_annual"][market],
        "target_capital": src["target_capital"][market],
        "max_adv_participation": src["max_adv_participation"],
        "target_rank_sharpe": src["target_rank_sharpe"],
        "target_rank_ann_return": src["target_rank_ann_return"][market],
        "target_absolute_ann_return": src["target_absolute_ann_return"][market],
        "target_cost_cushion_multiple": src["target_cost_cushion_multiple"],
        "return_lcb_confidence": src["return_lcb_confidence"],
        "multiple_testing_trials": src["multiple_testing_trials"],
        "multiple_testing_alpha": src["multiple_testing_alpha"],
    }
    cfg.update(overrides or {})
    for key in (
        "top_fraction",
        "tail_fraction",
        "min_coverage",
        "min_research_score",
        "min_oos_sharpe",
        "min_stress_sharpe",
        "min_return_hac_t",
        "max_hac_p_value",
        "min_era_consistency",
        "min_profitable_era_rate",
        "min_monotonicity",
        "min_cost_cushion_multiple",
        "max_drawdown",
        "max_market_beta_long_short",
        "max_daily_turnover",
        "base_cost_bps",
        "borrow_cost_bps_annual",
        "stress_borrow_cost_bps_annual",
        "target_capital",
        "max_adv_participation",
        "target_rank_sharpe",
        "target_rank_ann_return",
        "target_absolute_ann_return",
        "target_cost_cushion_multiple",
        "return_lcb_confidence",
        "multiple_testing_alpha",
    ):
        try:
            cfg[key] = float(cfg[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"evaluation_config.{key} 必须是数值") from exc
    try:
        cfg["min_layer_days"] = int(cfg["min_layer_days"])
        cfg["multiple_testing_trials"] = int(cfg["multiple_testing_trials"])
        cfg["stress_cost_bps"] = [float(value) for value in cfg["stress_cost_bps"]]
    except (TypeError, ValueError) as exc:
        raise ValueError("evaluation_config 的样本天数或压力成本格式错误") from exc
    if not 0 < cfg["top_fraction"] <= 0.5 or not 0 < cfg["tail_fraction"] <= 0.5:
        raise ValueError("top_fraction/tail_fraction 必须在 (0, 0.5] 内")
    if not 0 < cfg["min_coverage"] <= 1:
        raise ValueError("min_coverage 必须在 (0, 1] 内")
    if not 0 <= cfg["max_hac_p_value"] <= 1:
        raise ValueError("max_hac_p_value 必须在 [0, 1] 内")
    if not 0 <= cfg["min_era_consistency"] <= 1:
        raise ValueError("min_era_consistency 必须在 [0, 1] 内")
    if not 0 <= cfg["min_profitable_era_rate"] <= 1:
        raise ValueError("min_profitable_era_rate 必须在 [0, 1] 内")
    if not -1 <= cfg["min_monotonicity"] <= 1:
        raise ValueError("min_monotonicity 必须在 [-1, 1] 内")
    if not 0 < cfg["max_drawdown"] <= 1:
        raise ValueError("max_drawdown 必须在 (0, 1] 内")
    if cfg["max_daily_turnover"] <= 0:
        raise ValueError("max_daily_turnover 必须为正数")
    if cfg["target_capital"] <= 0:
        raise ValueError("target_capital 必须为正数")
    if not 0 < cfg["max_adv_participation"] <= 1:
        raise ValueError("max_adv_participation 必须在 (0, 1] 内")
    if cfg["min_layer_days"] < 30:
        raise ValueError("min_layer_days 不能少于 30")
    if not 0.50 < cfg["return_lcb_confidence"] < 1:
        raise ValueError("return_lcb_confidence 必须在 (0.5, 1) 内")
    if cfg["multiple_testing_trials"] < 1:
        raise ValueError("multiple_testing_trials 不能少于 1")
    if not 0 < cfg["multiple_testing_alpha"] < 0.5:
        raise ValueError("multiple_testing_alpha 必须在 (0, 0.5) 内")
    if not cfg["stress_cost_bps"] or any(value < 0 for value in cfg["stress_cost_bps"]):
        raise ValueError("stress_cost_bps 必须是非负数列表")
    for key in (
        "base_cost_bps",
        "borrow_cost_bps_annual",
        "stress_borrow_cost_bps_annual",
        "max_market_beta_long_short",
        "min_return_hac_t",
        "min_cost_cushion_multiple",
        "target_rank_sharpe",
        "target_rank_ann_return",
        "target_absolute_ann_return",
        "target_cost_cushion_multiple",
    ):
        if cfg[key] < 0:
            raise ValueError(f"{key} 不能为负数")
    cfg["protocol_version"] = EVALUATION_PROTOCOL_VERSION
    return cfg


def default_task_cost_bps(market: str, universe_n: int, horizon: int) -> float:
    """Market-aware research cost, with an illiquidity premium for broad pools."""
    if market not in {"ashare", "us"}:
        raise ValueError("market 必须是 ashare 或 us")
    base = 20.0 if market == "ashare" else 15.0
    illiquidity_premium = 10.0 if universe_n > 1000 or horizon == 10 else 0.0
    return base + illiquidity_premium


def resolve_engine_tasks(
    tasks: list[dict],
    market: str,
    portfolio_mode: str,
    direction: int,
    *,
    preserve_declared_costs: bool = False,
) -> list[dict]:
    """Resolve one task list without leaking another market's mode or costs."""
    resolved = []
    for task in tasks:
        row = dict(task)
        universe_n = int(row.get("universe_n", 500))
        horizon = int(row.get("horizon", 5))
        costs_by_market = row.get("cost_bps_by_market") or {}
        if market in costs_by_market:
            cost_bps = float(costs_by_market[market])
        elif preserve_declared_costs and row.get("cost_bps") is not None:
            cost_bps = float(row["cost_bps"])
        else:
            cost_bps = default_task_cost_bps(market, universe_n, horizon)
        row.update({
            "market": market,
            "mode": portfolio_mode,
            "direction": direction,
            "universe_n": universe_n,
            "horizon": horizon,
            "cost_bps": cost_bps,
            "cost_bps_by_market": {
                "ashare": float(
                    costs_by_market.get(
                        "ashare",
                        default_task_cost_bps("ashare", universe_n, horizon),
                    )
                ),
                "us": float(
                    costs_by_market.get(
                        "us",
                        default_task_cost_bps("us", universe_n, horizon),
                    )
                ),
            },
        })
        resolved.append(row)
    return resolved

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
        {"name": "T1_liquid500_5d", "universe_n": 500, "horizon": 5, "cost_bps": 15, "cost_bps_by_market": {"ashare": 20, "us": 15}, "mode": DEFAULT_PORTFOLIO_MODE},
        {"name": "T2_mid1500_10d", "universe_n": 1500, "horizon": 10, "cost_bps": 25, "cost_bps_by_market": {"ashare": 30, "us": 25}, "mode": DEFAULT_PORTFOLIO_MODE},
        {"name": "T3_liquid500_20d", "universe_n": 500, "horizon": 20, "cost_bps": 15, "cost_bps_by_market": {"ashare": 20, "us": 15}, "mode": DEFAULT_PORTFOLIO_MODE},
    ],
}

# ---- 旧版引擎配置 (A组兼容) ----
DEFAULT_ENGINE_CONFIG = {
    "inner_budget_per_outer_step": 10,
    "outer_accept_epsilon": 0.02,
    "incumbent_remeasure_every": 5,
    "tasks": [
        {"name": "T1_liquid500_5d", "universe_n": 500, "horizon": 5, "cost_bps": 15, "cost_bps_by_market": {"ashare": 20, "us": 15}, "mode": DEFAULT_PORTFOLIO_MODE},
        {"name": "T2_mid1500_10d", "universe_n": 1500, "horizon": 10, "cost_bps": 25, "cost_bps_by_market": {"ashare": 30, "us": 25}, "mode": DEFAULT_PORTFOLIO_MODE},
        {"name": "T3_liquid500_20d", "universe_n": 500, "horizon": 20, "cost_bps": 15, "cost_bps_by_market": {"ashare": 20, "us": 15}, "mode": DEFAULT_PORTFOLIO_MODE},
    ],
}
