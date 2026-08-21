"""Deterministic economic hypotheses for factor expressions.

These descriptions are intentionally framed as hypotheses inferred from DSL
structure.  They are presentation metadata, never evidence of causality or a
substitute for the report's frozen validation results.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..dsl.engine import expression_profile
from .diversity import MECHANISM_LABELS, infer_mechanism


_MECHANISM_EXPLANATIONS: dict[str, dict[str, Any]] = {
    "momentum": {
        "return_source": "价格趋势延续、信息缓慢扩散与机构分批建仓",
        "rationale": (
            "表达式主要刻画价格在一段窗口内的方向或相对强弱。若市场对新信息"
            "反应不充分，较强趋势可能继续；反向使用时则是在检验趋势拥挤后的反转。"
        ),
        "failure_modes": ["趋势快速反转", "高换手侵蚀微弱边际", "市场状态由趋势切换为震荡"],
    },
    "reversal": {
        "return_source": "短期过度反应、流动性冲击消退与均值回归",
        "rationale": (
            "表达式试图识别偏离近期常态的价格。若偏离来自非信息型订单冲击或"
            "投资者过度反应，价格可能向常态回归；若偏离来自永久信息，反转会失效。"
        ),
        "failure_modes": ["基本面信息造成永久重定价", "单边趋势延续", "抄底端出现尾部损失"],
    },
    "volatility": {
        "return_source": "波动、尾部风险、彩票偏好或不确定性溢价",
        "rationale": (
            "表达式以收益波动或高低价区间衡量不确定性。正向与反向方向分别检验"
            "高风险补偿和低波动异象，实际收益也可能来自规模、流动性或 Beta 暴露。"
        ),
        "failure_modes": ["波动制度突变", "低波动因子拥挤", "风险暴露而非独立 Alpha"],
    },
    "liquidity": {
        "return_source": "流动性风险补偿、交易拥挤与容量约束",
        "rationale": (
            "表达式依据成交量、成交额或换手刻画交易便利度。低流动性可能要求风险"
            "补偿，高流动性也可能代表信息关注；方向必须结合报告中的冻结定向理解。"
        ),
        "failure_modes": ["冲击成本高于纸面收益", "容量不足", "成交活跃度定义随市场变化"],
    },
    "volume_price_interaction": {
        "return_source": "订单需求冲击、量价确认或拥挤交易耗竭",
        "rationale": (
            "表达式同时使用价格与成交活跃度，试图区分有成交支持的价格变化和"
            "缺乏确认的波动。正向可能表示需求持续，反向则可能表示冲击后的回吐。"
        ),
        "failure_modes": ["放量来自被动再平衡而非信息", "成交额受价格尺度污染", "冲击衰减速度漂移"],
    },
    "gap_intraday": {
        "return_source": "隔夜信息定价与开盘后流动性再平衡",
        "rationale": (
            "表达式比较开盘、收盘或日内高低价，试图区分隔夜信息与日内交易阶段。"
            "收益可能来自隔夜反应不足，也可能来自开盘冲击的日内修复。"
        ),
        "failure_modes": ["开盘不可成交或滑点放大", "公司事件主导缺口", "隔夜与日内制度变化"],
    },
    "price_relationship": {
        "return_source": "价格序列共振、状态切换或短期关系修复",
        "rationale": (
            "表达式使用相关、排序或多窗口关系识别价格状态。它可能捕捉趋势同步、"
            "均值回归或状态切换，但经济方向不能仅凭相关符号断言。"
        ),
        "failure_modes": ["相关结构断裂", "窗口选择过拟合", "同一风险暴露造成伪相关"],
    },
    "valuation": {
        "return_source": "估值均值回归与承担基本面风险的补偿",
        "rationale": (
            "表达式依据估值字段比较价格与基本面锚。低估值可能获得风险补偿，"
            "也可能是基本面恶化的价值陷阱；方向以报告冻结结果为准。"
        ),
        "failure_modes": ["价值陷阱", "会计口径不可比", "行业和盈利周期暴露"],
    },
    "size": {
        "return_source": "小盘风险、关注不足、流动性与容量补偿",
        "rationale": (
            "表达式使用市值或流通规模，可能捕捉小盘风险补偿或大盘质量偏好。"
            "该收益来源通常与流动性、行业和指数成分暴露纠缠。"
        ),
        "failure_modes": ["小盘流动性枯竭", "指数制度变化", "容量和冲击成本约束"],
    },
    "capital_flow": {
        "return_source": "订单失衡、机构资金持续性或拥挤后的反转",
        "rationale": (
            "表达式使用大单或资金流字段近似需求压力。持续流入可能反映信息交易，"
            "极端流入也可能代表拥挤；正反方向都需要由冻结样本规则决定。"
        ),
        "failure_modes": ["资金流字段口径漂移", "拥挤反转", "大单分类不能代表真实投资者身份"],
    },
    "other": {
        "return_source": "表达式结构无法归入现有机制族",
        "rationale": (
            "仅凭当前字段与算子不能形成可靠的经济机制映射，应回到原始研究假设、"
            "暴露归因和分期结果判断，而不是从回测表现反推故事。"
        ),
        "failure_modes": ["机制不可识别", "数据挖掘偏差", "隐藏风险暴露"],
    },
}


def _clean_items(values: Iterable[Any] | None) -> list[str]:
    return sorted({str(value) for value in (values or []) if str(value).strip()})


def explain_factor_economics(
    expression: str,
    *,
    direction: int = 1,
    market: str = "ashare",
    portfolio_mode: str = "long_only",
    fields: Iterable[Any] | None = None,
    operators: Iterable[Any] | None = None,
    hypothesis: str = "",
    screening_only: bool = False,
) -> dict[str, Any]:
    """Return an auditable, non-causal interpretation of a frozen factor row."""
    profile: dict[str, Any] = {}
    try:
        profile = expression_profile(expression)
    except (SyntaxError, TypeError, ValueError):
        pass
    resolved_fields = _clean_items(fields) or _clean_items(profile.get("fields"))
    resolved_operators = _clean_items(operators) or _clean_items(
        profile.get("operators")
    )
    family = infer_mechanism(expression, hypothesis)
    explanation = _MECHANISM_EXPLANATIONS.get(
        family,
        _MECHANISM_EXPLANATIONS["other"],
    )
    oriented_direction = -1 if int(direction or 1) < 0 else 1
    if portfolio_mode == "long_short":
        direction_text = (
            "对报告中保存的完整 DSL 分值正向排序：高值端做多、低值端做空。"
            if oriented_direction > 0
            else "对报告中保存的完整 DSL 分值反向排序：低值端做多、高值端做空。"
        )
    else:
        direction_text = (
            "对报告中保存的完整 DSL 分值正向排序，纯多组合选择高值端。"
            if oriented_direction > 0
            else "对报告中保存的完整 DSL 分值反向排序，纯多组合选择低值端。"
        )
    direction_text += " 表达式内部负号已包含在完整 DSL 中，不应再次口头抵消。"

    warnings = list(explanation["failure_modes"])
    if screening_only:
        warnings.append("当前为全窗口筛选，排名本身不是独立样本外证据")
    if portfolio_mode == "long_only":
        warnings.append("纯多收益可能混入市场 Beta、行业和规模暴露")
    elif market == "us":
        warnings.append("美股空头结果仍受 locate、HTB、召回和真实借券费约束")
    if market == "ashare":
        warnings.append("A股实盘还受 T+1、涨跌停、停牌与整手成交约束")

    return {
        "classification": "structural_hypothesis",
        "mechanism_family": family,
        "mechanism_label": MECHANISM_LABELS.get(family, MECHANISM_LABELS["other"]),
        "possible_return_source": explanation["return_source"],
        "rationale": explanation["rationale"],
        "direction_interpretation": direction_text,
        "observed_structure": {
            "fields": resolved_fields,
            "operators": resolved_operators,
            "required_history": profile.get("required_history"),
        },
        "failure_modes": warnings,
        "disclaimer": (
            "这是基于 DSL 字段、算子和冻结方向生成的可能经济学解释，"
            "不是因果证明，也不改变榜单原有评价与实盘准入结论。"
        ),
    }
