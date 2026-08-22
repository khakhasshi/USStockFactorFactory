"""Clean-room internal mechanism map; ideas, never pre-approved factors."""

from __future__ import annotations

import hashlib
import json


CATALOG_PROTOCOL = "factorfactory.mechanism-lens/v1"

# Exactly 33 compact mechanisms.  Parameter grids are deliberately absent.
_SPECS = (
    ("trend_multi_scale", "多尺度趋势同向可能反映信息缓慢扩散", ["close"], ["multi_scale", "consensus"], ["震荡反复"], ["us", "ashare"]),
    ("trend_efficiency", "相同涨幅下路径更平滑者可能更可持续", ["close"], ["path_efficiency"], ["跳空主导"], ["us", "ashare"]),
    ("trend_acceleration", "趋势加速度可能刻画资金追逐的边际变化", ["close"], ["delta_of_return"], ["末端拥挤"], ["us", "ashare"]),
    ("short_reversal", "短期价格冲击后存在流动性补偿性反转", ["close"], ["negative_return"], ["信息型跳跃"], ["us", "ashare"]),
    ("drawdown_depth", "深度回撤可能携带过度反应或风险状态", ["close"], ["drawdown"], ["基本面断裂"], ["us", "ashare"]),
    ("drawdown_duration", "回撤持续时间比单点跌幅更能描述压力路径", ["close"], ["path_duration"], ["长熊市"], ["us", "ashare"]),
    ("recovery_geometry", "回撤后的恢复速度可能区分暂时冲击与永久损伤", ["close"], ["recovery_ratio"], ["幸存者偏差"], ["us", "ashare"]),
    ("downside_volatility", "下行波动可能比总波动更接近风险补偿", ["close"], ["negative_semivariance"], ["崩盘延续"], ["us", "ashare"]),
    ("up_down_asymmetry", "上下行波动不对称可能刻画投机和杠杆状态", ["close"], ["asymmetry"], ["低样本尾部"], ["us", "ashare"]),
    ("volatility_of_volatility", "波动率自身不稳定可能预示风险状态切换", ["close"], ["nested_volatility"], ["估计噪声"], ["us", "ashare"]),
    ("range_compression", "区间压缩后的价格发现可能出现方向性释放", ["high", "low", "close"], ["range", "compression"], ["假突破"], ["us", "ashare"]),
    ("atr_normalized_move", "以真实波幅归一的价格变化可跨标的比较冲击强度", ["high", "low", "close"], ["atr_normalize"], ["隔夜缺口"], ["us", "ashare"]),
    ("volume_persistence", "成交量持续性可能反映机构执行尚未结束", ["vol"], ["autocorrelation"], ["指数调仓"], ["us", "ashare"]),
    ("volume_burst", "异常放量可能代表信息到达或被迫交易", ["vol"], ["relative_volume"], ["一次性事件"], ["us", "ashare"]),
    ("volume_concentration", "成交集中度可区分稳定参与与少数爆发日", ["vol"], ["concentration"], ["停牌附近"], ["us", "ashare"]),
    ("volume_exhaustion", "急跌放量后成交压力衰减可能代表卖压耗尽", ["close", "vol"], ["shock", "path_decay"], ["系统去杠杆", "流动性冻结"], ["us", "ashare"]),
    ("price_volume_confirmation", "价格与成交同向持续可能提高趋势可信度", ["close", "vol"], ["interaction", "correlation"], ["拥挤末端"], ["us", "ashare"]),
    ("price_impact_decay", "单位成交额价格冲击的衰减可描述流动性恢复", ["close", "amount"], ["impact", "decay"], ["成交额口径漂移"], ["us", "ashare"]),
    ("amihud_illiquidity", "单位成交额绝对收益刻画流动性风险补偿", ["close", "amount"], ["impact_ratio"], ["微盘异常值"], ["us", "ashare"]),
    ("turnover_shock", "换手突变可能揭示持有人结构重新定价", ["vol", "amount"], ["relative_activity"], ["公司行动"], ["us", "ashare"]),
    ("overnight_gap", "隔夜与日内收益承载不同信息和交易者约束", ["open", "close"], ["overnight_decomposition"], ["复权错配"], ["us", "ashare"]),
    ("intraday_reversal", "开盘冲击在收盘前可能发生流动性反转", ["open", "close"], ["intraday_return"], ["趋势日"], ["us", "ashare"]),
    ("close_location", "收盘在日内区间的位置反映收盘前买卖压力", ["high", "low", "close"], ["range_location"], ["涨跌停"], ["us", "ashare"]),
    ("gap_fill_pressure", "缺口回补程度可能区分信息跳跃与暂时失衡", ["open", "high", "low", "close"], ["gap", "recovery"], ["连续跳空"], ["us", "ashare"]),
    ("state_gated_momentum", "动量仅在低噪声状态有效可降低条件性失效", ["close"], ["gate", "trend"], ["门控过拟合"], ["us", "ashare"]),
    ("state_gated_reversal", "反转在高冲击低持续状态可能更稳定", ["close", "vol"], ["gate", "reversal"], ["状态漂移"], ["us", "ashare"]),
    ("cross_scale_disagreement", "短长周期信号分歧可能代表状态转折", ["close"], ["multi_scale", "disagreement"], ["反复横盘"], ["us", "ashare"]),
    ("valuation_earnings_yield", "盈利收益率可能补偿基本面与融资风险", ["pe_ttm"], ["inverse_valuation"], ["亏损公司"], ["ashare"]),
    ("valuation_book_to_price", "账面市值比可能补偿困境和资产风险", ["pb"], ["inverse_valuation"], ["资产减值"], ["ashare"]),
    ("valuation_sales_yield", "销售收益率可用于尚未盈利公司的相对估值", ["ps_ttm"], ["inverse_valuation"], ["低毛利行业"], ["ashare"]),
    ("size_capacity", "规模溢价必须与容量和交易成本共同评价", ["total_mv", "amount"], ["size", "capacity_gate"], ["微盘不可交易"], ["ashare"]),
    ("capital_flow_imbalance", "大单净流可能刻画知情交易或暂时价格压力", ["net_mf_amount", "amount"], ["flow_normalize"], ["数据定义变化"], ["ashare"]),
    ("large_order_exhaustion", "大单买卖压力衰减可能领先价格压力结束", ["buy_lg_amount", "sell_lg_amount", "amount"], ["imbalance", "decay"], ["拆单规避"], ["ashare"]),
)


def catalog(market: str | None = None) -> dict:
    rows = [
        {
            "mechanism": mechanism,
            "hypothesis": hypothesis,
            "required_fields": fields,
            "formula_shapes": shapes,
            "failure_modes": failures,
            "market_suitability": markets,
            "known_return_source_clusters": [],
            "attempt_count": 0,
            "historical_success_rate": None,
        }
        for mechanism, hypothesis, fields, shapes, failures, markets in _SPECS
        if market is None or market in markets
    ]
    lenses = []
    for row in rows:
        mechanism = row["mechanism"]
        questions = (
            ("mechanism", f"如何用最少字段证伪：{row['hypothesis']}"),
            ("failure", f"候选在何种状态下会因{row['failure_modes'][0]}失效？"),
            ("path", f"{mechanism} 的路径、持续时间与单点幅度哪个更有解释力？"),
            ("transfer", f"{mechanism} 在不同市场制度和成本下方向是否保持？"),
        )
        for lens_type, question in questions:
            lenses.append(
                {
                    "lens_id": f"{mechanism}:{lens_type}",
                    "mechanism": mechanism,
                    "lens_type": lens_type,
                    "research_question": question,
                    "factor_expression": None,
                    "score_bonus": 0.0,
                }
            )
    # The source inspiration contained 119 lenses.  We retain that breadth as
    # falsifiable research questions, not 119 pre-approved expressions.
    lenses = lenses[: min(119, len(lenses))]
    fingerprint = hashlib.sha256(
        json.dumps({"mechanisms": rows, "lenses": lenses}, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "schema": CATALOG_PROTOCOL,
        "policy": "idea_map_only_no_score_bonus_no_bulk_grid",
        "count": len(rows),
        "lens_count": len(lenses),
        "fingerprint": fingerprint,
        "mechanisms": rows,
        "lenses": lenses,
    }


assert len(_SPECS) == 33
