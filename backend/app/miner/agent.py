"""内层 Factor Miner v2: 由 MinerTemplate 驱动 (prompt/策略/模板全部可被外层改写).

纪律: 提示词只包含训练安全的 V4 聚合反馈；最终封存层永不进入上下文.
"""

import json
import logging
import random

from ..config import DEFAULT_MINER_TEMPLATE, DSL_FIELDS, get_dsl_fields
from ..dsl.engine import OPERATORS_DOC, validate
from ..factors.diversity import (
    MECHANISM_LABELS,
    mechanism_compatible,
    mechanism_from_item,
    mechanisms_for_market,
    select_target_mechanism,
)
from ..factors.semantics import (
    audit_expression_semantics,
    render_field_contract,
)
from ..factors.similarity import expression_similarity
from ..feedback import (
    build_feedback_envelope,
    build_inner_feedback_context,
    ensure_training_safe,
)
from ..llm import client as llm
from ..observability import redact_text, redact_value

logger = logging.getLogger("miner")

_FIELDS = DSL_FIELDS
_WINDOWS = [3, 5, 10, 20, 40, 60, 120]
_CHANGE_AXES = {"window", "transform", "normalization", "combination", "new_draft"}

# ============================================================
# 随机回退 (无 LLM 时)
# ============================================================

def random_expression(
    templates: list[str] | None = None,
    fields: list[str] | None = None,
    rng: random.Random | None = None,
) -> str:
    """使用模板生成随机表达式。"""
    generator = rng or random
    f = generator.choice
    w = lambda: f(_WINDOWS)  # noqa: E731
    field = lambda: f(fields or _FIELDS)  # noqa: E731

    if templates and len(templates) >= 3:
        # 从模板列表随机选一个, 填参数
        tpl = f(templates)
        expr = tpl.replace("{window}", str(w()))
        expr = expr.replace("{field1}", field()).replace("{field2}", field())
        expr = expr.replace("{field}", field())
        expr = expr.replace(
            "{-}",
            "-" if generator.random() < 0.5 else "",
        )
        if validate(expr, fields or _FIELDS) is None:
            return expr

    # 内置回退模板
    builtin = [
        lambda: f"-rank(ts_delta(close, {w()}))",
        lambda: f"rank(ts_corr({field()}, {field()}, {w()}))",
        lambda: f"-zscore(ts_std({field()}, {w()}))",
        lambda: f"rank((close - ts_min(low, {w()})) / (ts_max(high, {w()}) - ts_min(low, {w()})))",
        lambda: f"zscore(ts_mean(amount, {f([3, 5])}) / ts_mean(amount, {f([40, 60])}))",
        lambda: f"-rank(ts_rank(close, {w()}))",
        lambda: f"rank(ts_delta(close, {f([40, 60, 120])})) - rank(ts_delta(close, {f([3, 5])}))",
        lambda: f"rank(ts_mean(abs(ts_delta(close,1))/(delay(close,1)+1e-9)/(amount+1e-9), {w()}))",
    ]
    return f(builtin)()


def _random_tree_expression_for_family(
    family: str,
    fields: list[str],
    generator: random.Random,
) -> str:
    """Build a shallow typed expression tree for one economic mechanism.

    This deliberately has no access to evaluation outcomes.  The types here
    are economic/measurement types (return, range, liquidity surprise, flow
    share), which prevents nonsensical raw-price/turnover arithmetic while
    still giving the random baseline a combinatorial search space.
    """
    field_set = set(fields)
    # A 10k-candidate campaign must not collapse into a few dozen templates
    # crossed with seven canonical windows.  The ranges stay interpretable
    # (roughly days, weeks, months and one trading year) while providing enough
    # independent horizon combinations for a real random-search baseline.
    fast = generator.choice(list(range(2, 21)))
    medium = generator.choice(list(range(10, 81, 5)))
    slow = generator.choice(list(range(40, 251, 10)))
    transform_window = generator.choice([fast, medium, slow])

    ret1 = "(ts_delta(close, 1)/(delay(close, 1)+1e-9))"
    ret_fast = f"(ts_delta(close, {fast})/(delay(close, {fast})+1e-9))"
    ret_medium = f"(ts_delta(close, {medium})/(delay(close, {medium})+1e-9))"
    ret_slow = f"(ts_delta(close, {slow})/(delay(close, {slow})+1e-9))"
    range1 = "((high-low)/(close+1e-9))"
    overnight = "((open-delay(close, 1))/(delay(close, 1)+1e-9))"
    intraday = "((close-open)/(open+1e-9))"
    close_location = "((close-low)/(high-low+1e-9))"
    breakout = f"((close-ts_min(low, {slow}))/(ts_max(high, {slow})-ts_min(low, {slow})+1e-9))"
    price_bases = [
        ret1, ret_fast, ret_medium, ret_slow, range1, overnight,
        intraday, close_location, breakout,
    ]

    def temporal(value: str) -> str:
        return generator.choice([
            value,
            f"ts_mean({value}, {transform_window})",
            f"ts_std({value}, {transform_window})",
            f"ts_delta({value}, {transform_window})",
            f"ts_rank({value}, {transform_window})",
            f"ts_mean(sign({value}), {transform_window})",
        ])

    def trend_temporal(value: str) -> str:
        """Direction-preserving transforms for momentum/reversal signals."""
        return generator.choice([
            value,
            f"ts_mean({value}, {transform_window})",
            f"ts_sum({value}, {transform_window})",
            f"ts_delta({value}, {transform_window})",
            f"ts_rank({value}, {transform_window})",
            f"ts_mean(sign({value}), {transform_window})",
        ])

    if family in {"momentum", "reversal"}:
        trend = generator.choice([
            ret_fast,
            ret_medium,
            ret_slow,
            f"ts_mean({ret1}, {medium})",
            f"ts_sum({ret1}, {medium})",
            f"(close/(ts_mean(close, {medium})+1e-9)-1)",
            f"(ts_mean({ret1}, {fast})-ts_mean({ret1}, {slow}))",
            f"(ts_rank(close, {slow})-ts_rank(close, {fast}))",
            f"(close/(ts_max(high, {slow})+1e-9)-1)",
            f"(close/(ts_min(low, {slow})+1e-9)-1)",
        ])
        comparator = generator.choice([
            f"ts_std({ret1}, {slow})",
            f"ts_mean(abs({ret1}), {slow})",
            f"abs(ts_mean({ret1}, {slow}))",
            f"ts_mean({range1}, {slow})",
            f"ts_std({trend}, {medium})",
        ])
        tree = generator.choice([
            trend_temporal(trend),
            f"({trend_temporal(trend)}/({comparator}+1e-9))",
            f"({trend_temporal(trend)}-ts_mean({ret1}, {slow}))",
            f"ts_mean(sign({ret1}), {medium})",
            f"ts_corr({trend}, delay({trend}, {fast}), {medium})",
            f"ts_rank({trend}, {slow})",
            f"(ts_mean({trend}, {fast})-ts_mean({trend}, {slow}))",
            f"ts_delta({trend}, {medium})",
            f"ts_corr({trend}, {range1}, {medium})",
        ])
        prefix = "-" if family == "reversal" else ""
        return f"{prefix}rank({tree})"

    if family == "volatility":
        base = generator.choice(price_bases)
        other = generator.choice(price_bases)
        tree = generator.choice([
            f"ts_std({base}, {transform_window})",
            f"ts_mean(abs({base}), {transform_window})",
            f"ts_max(abs({base}), {transform_window})",
            f"(ts_std({base}, {fast})/(ts_std({base}, {slow})+1e-9))",
            f"ts_delta(ts_std({base}, {medium}), {fast})",
            f"ts_corr(abs({ret1}), {range1}, {medium})",
            f"ts_std(ts_mean({base}, {fast}), {slow})",
            f"ts_mean(({base})*({base}), {medium})",
            f"(ts_max({base}, {slow})-ts_min({base}, {slow}))",
            f"ts_corr(abs({base}), abs({other}), {medium})",
            f"(ts_mean(abs({base}), {fast})/(ts_mean(abs({base}), {slow})+1e-9))",
            f"ts_rank(ts_std({base}, {medium}), {slow})",
        ])
        return f"rank({tree})"

    liquidity_bases = [
        f"(amount/(ts_mean(amount, {slow})+1e-9))",
        f"(vol/(ts_mean(vol, {slow})+1e-9))",
        "log(amount)",
        "log(vol)",
        "(amount/(vol+1e-9))",
    ]
    if "turnover_rate" in field_set:
        liquidity_bases.extend([
            "turnover_rate",
            f"(turnover_rate/(ts_mean(turnover_rate, {slow})+1e-9))",
        ])
    if "volume_ratio" in field_set:
        liquidity_bases.append("volume_ratio")

    if family == "liquidity":
        base = generator.choice(liquidity_bases)
        tree = generator.choice([
            temporal(base),
            f"ts_corr({base}, delay({base}, {fast}), {medium})",
            f"(ts_mean({base}, {fast})-ts_mean({base}, {slow}))",
            f"(ts_std({base}, {fast})/(ts_std({base}, {slow})+1e-9))",
        ])
        return f"rank({tree})"

    if family == "volume_price_interaction":
        price = generator.choice(price_bases)
        liquidity = generator.choice(liquidity_bases)
        price_signal = temporal(price)
        liquidity_signal = temporal(liquidity)
        tree = generator.choice([
            f"ts_corr({price}, {liquidity}, {medium})",
            f"ts_mean(({price})*({liquidity}), {medium})",
            f"ts_mean(sign({price})*({liquidity}), {medium})",
            f"(ts_mean({price}, {fast})/"
            f"(ts_std({liquidity}, {slow})+1e-9))",
            f"ts_delta(ts_corr({price}, {liquidity}, {medium}), {fast})",
            f"ts_mean(abs({price})/(amount+1e-9), {medium})",
            f"ts_corr({price_signal}, {liquidity_signal}, {medium})",
            f"ts_mean(({price_signal})*({liquidity_signal}), {medium})",
            f"(ts_mean({price}, {fast})*ts_mean({liquidity}, {slow}))",
            f"(ts_mean({price}, {fast})-ts_mean(sign({price})*({liquidity}), {slow}))",
            f"ts_corr(ts_rank({price}, {fast}), ts_rank({liquidity}, {fast}), {medium})",
            f"ts_delta(ts_mean(sign({price})*({liquidity}), {medium}), {fast})",
        ])
        return f"rank({tree})"

    if family == "gap_intraday":
        base = generator.choice([overnight, intraday, f"({overnight}-{intraday})"])
        tree = generator.choice([
            temporal(base),
            f"ts_corr({overnight}, {intraday}, {medium})",
            f"(ts_mean({overnight}, {fast})-ts_mean({intraday}, {slow}))",
            f"(ts_mean({base}, {medium})/(ts_std({base}, {slow})+1e-9))",
        ])
        return f"rank({tree})"

    if family == "price_relationship":
        left = generator.choice(price_bases)
        right = generator.choice([
            value for value in price_bases if value != left
        ])
        tree = generator.choice([
            f"ts_corr({left}, {right}, {medium})",
            f"ts_delta(ts_corr({left}, {right}, {slow}), {fast})",
            f"ts_corr(ts_rank({left}, {fast}), ts_rank({right}, {fast}), {medium})",
            f"ts_corr({left}, delay({right}, {fast}), {medium})",
        ])
        return f"rank({tree})"

    if family == "valuation":
        values = [name for name in ["pe_ttm", "pb", "ps_ttm", "dv_ttm"] if name in field_set]
        value = generator.choice(values)
        other = generator.choice(values)
        tree = generator.choice([
            temporal(value),
            f"({value}/(ts_mean({value}, {slow})+1e-9))",
            f"(zscore({value})-zscore({other}))",
            f"ts_corr({value}, delay({other}, {fast}), {medium})",
        ])
        return f"rank({tree})"

    if family == "size":
        sizes = [name for name in ["total_mv", "circ_mv", "float_share"] if name in field_set]
        size = generator.choice(sizes)
        other = generator.choice(sizes)
        left = generator.choice([
            f"log({size})",
            f"ts_rank(log({size}), {transform_window})",
            f"({size}/(ts_mean({size}, {slow})+1e-9))",
            f"ts_delta(log({size}), {medium})",
            f"ts_mean(log({size}), {medium})",
            f"ts_std(log({size}), {medium})",
        ])
        right_options = [
            f"log({other})",
            f"ts_rank(log({other}), {fast})",
            f"({other}/(ts_mean({other}, {slow})+1e-9))",
            f"ts_delta(log({other}), {fast})",
            f"ts_mean(log({other}), {slow})",
            f"ts_std(log({other}), {slow})",
        ]
        right = generator.choice([value for value in right_options if value != left] or right_options)
        tree = generator.choice([
            temporal(left),
            f"({size}/({other}+1e-9))",
            f"({size}/(ts_mean({size}, {slow})+1e-9))",
            f"ts_corr(log({size}), delay(log({other}), {fast}), {medium})",
            f"({left}-{right})",
            f"({left}/(abs({right})+1e-9))",
            f"ts_corr({left}, delay({right}, {fast}), {medium})",
            f"(ts_mean({left}, {fast})-ts_mean({left}, {slow}))",
            f"ts_delta({left}, {medium})",
            f"ts_rank({left}, {slow})",
            f"ts_std({left}, {medium})",
        ])
        return f"rank({tree})"

    if family == "capital_flow":
        flows = [
            name for name in [
                "net_mf_amount", "buy_lg_amount", "sell_lg_amount",
                "buy_elg_amount", "sell_elg_amount",
            ]
            if name in field_set
        ]
        flow = generator.choice(flows)
        normalized = f"({flow}/(amount+1e-9))"
        tree = generator.choice([
            temporal(normalized),
            f"(ts_mean({normalized}, {fast})-ts_mean({normalized}, {slow}))",
            f"ts_corr({normalized}, {ret1}, {medium})",
            f"(ts_mean({normalized}, {medium})/(ts_std({normalized}, {slow})+1e-9))",
        ])
        return f"rank({tree})"

    raise ValueError(f"不支持的随机收益机制: {family}")


def random_expression_for_family(
    family: str,
    fields: list[str],
    rng: random.Random | None = None,
) -> str:
    """Generate a valid mechanism-targeted fallback without an LLM.

    The random baseline is intended to be a real search baseline rather than a
    window sweep over one expression per mechanism.  Keep the grammar generic
    and outcome-agnostic: it samples several economically distinct shapes, then
    applies the same DSL and field-provenance checks as an LLM proposal.
    """
    generator = rng or random
    market = "ashare" if "pb" in fields else "us"
    field_set = set(fields)

    for _ in range(32):
        fast = generator.choice([2, 3, 5, 10, 15, 20])
        medium = generator.choice([10, 15, 20, 30, 40, 60])
        slow = generator.choice([40, 60, 90, 120, 180, 240])
        previous = "(delay(close, 1)+1e-9)"
        returns = f"(ts_delta(close, 1)/{previous})"
        lagged = f"(delay(close, {fast})+1e-9)"
        fast_return = f"(ts_delta(close, {fast})/{lagged})"

        candidates: dict[str, list] = {
            "momentum": [
                lambda: f"rank(ts_delta(close, {slow})/(delay(close, {slow})+1e-9))",
                lambda: f"rank(ts_mean({returns}, {medium}))",
                lambda: f"rank(ts_sum({returns}, {medium}))",
                lambda: f"rank(ts_mean({returns}, {fast})-ts_mean({returns}, {slow}))",
                lambda: f"rank(ts_rank(close, {slow})-ts_rank(close, {fast}))",
                lambda: f"rank(ts_mean({fast_return}, {medium}))",
            ],
            "reversal": [
                lambda: f"-rank({fast_return})",
                lambda: f"-rank(ts_mean({returns}, {fast}))",
                lambda: f"-rank(close/(ts_mean(close, {medium})+1e-9)-1)",
                lambda: f"-rank(ts_rank(close, {medium}))",
                lambda: f"-rank(ts_mean({returns}, {fast})-ts_mean({returns}, {slow}))",
                lambda: f"-rank(ts_delta(ts_mean(close, {fast}), {medium})/(delay(ts_mean(close, {fast}), {medium})+1e-9))",
            ],
            "volatility": [
                lambda: f"-rank(ts_std({returns}, {medium}))",
                lambda: f"rank(ts_std({returns}, {fast})-ts_std({returns}, {slow}))",
                lambda: f"-rank(ts_mean((high-low)/(close+1e-9), {medium}))",
                lambda: f"rank(ts_std((high-low)/(close+1e-9), {medium}))",
                lambda: f"-rank(ts_max(high, {medium})/(ts_min(low, {medium})+1e-9)-1)",
                lambda: f"rank(ts_mean(abs({returns}), {fast})/"
                        f"(ts_mean(abs({returns}), {slow})+1e-9))",
            ],
            "liquidity": [
                lambda: f"-rank(ts_mean(amount, {medium}))",
                lambda: f"rank(amount/(ts_mean(amount, {medium})+1e-9))",
                lambda: f"-rank(ts_mean(vol, {medium}))",
                lambda: f"rank(vol/(ts_mean(vol, {slow})+1e-9))",
                lambda: f"rank(ts_delta(log(amount), {medium}))",
                lambda: f"rank(ts_std(amount/(ts_mean(amount, {slow})+1e-9), {medium}))",
            ],
            "volume_price_interaction": [
                lambda: f"rank(ts_corr({returns}, amount/(ts_mean(amount, {slow})+1e-9), {medium}))",
                lambda: f"rank(ts_corr(abs({returns}), vol/(ts_mean(vol, {slow})+1e-9), {medium}))",
                lambda: f"rank(ts_mean(sign({returns})*amount/(ts_mean(amount, {slow})+1e-9), {medium}))",
                lambda: f"rank(ts_mean({returns}*vol/(ts_mean(vol, {slow})+1e-9), {medium}))",
                lambda: f"rank(ts_corr({fast_return}, log(amount), {medium}))",
                lambda: f"rank(ts_corr((high-low)/(close+1e-9), vol, {medium}))",
            ],
            "gap_intraday": [
                lambda: f"rank(ts_mean((open-delay(close, 1))/{previous}, {medium}))",
                lambda: f"rank(ts_mean((close-open)/(open+1e-9), {medium}))",
                lambda: f"rank(ts_corr((open-delay(close, 1))/{previous}, (close-open)/(open+1e-9), {medium}))",
                lambda: f"rank(ts_mean((open-delay(close, 1))/{previous}-(close-open)/(open+1e-9), {medium}))",
                lambda: f"rank((open-delay(close, 1))/{previous})",
                lambda: f"rank(ts_std((open-delay(close, 1))/{previous}, {medium}))",
            ],
            "price_relationship": [
                lambda: f"rank(ts_corr(high/(close+1e-9), low/(close+1e-9), {medium}))",
                lambda: f"rank(ts_corr(close/(open+1e-9), high/(low+1e-9), {medium}))",
                lambda: f"rank(ts_corr({returns}, (high-low)/(close+1e-9), {medium}))",
                lambda: f"rank(ts_corr(high/(low+1e-9), close/(delay(close, 1)+1e-9), {medium}))",
                lambda: f"rank(ts_corr(high/(close+1e-9), close/(low+1e-9), {medium}))",
                lambda: f"rank(ts_corr({returns}, delay({returns}, {fast}), {medium}))",
            ],
        }

        valuation_fields = [
            name for name in ["pe_ttm", "pb", "ps_ttm", "dv_ttm"]
            if name in field_set
        ]
        if valuation_fields:
            value = generator.choice(valuation_fields)
            other = generator.choice(valuation_fields)
            candidates["valuation"] = [
                lambda: f"-rank({value})",
                lambda: f"rank(ts_rank({value}, {medium}))",
                lambda: f"rank(ts_delta({value}, {medium}))",
                lambda: f"-rank(zscore({value})+zscore({other}))",
                lambda: f"rank({value}/(ts_mean({value}, {slow})+1e-9))",
            ]

        size_fields = [
            name for name in ["total_mv", "circ_mv", "float_share"]
            if name in field_set
        ]
        if size_fields:
            size = generator.choice(size_fields)
            other_size = generator.choice(size_fields)
            candidates["size"] = [
                lambda: f"-rank({size})",
                lambda: f"rank(ts_rank({size}, {medium}))",
                lambda: f"rank(ts_delta(log({size}), {medium}))",
                lambda: f"rank({size}/({other_size}+1e-9))",
                lambda: f"rank({size}/(ts_mean({size}, {slow})+1e-9))",
            ]

        flow_fields = [
            name for name in [
                "net_mf_amount", "buy_lg_amount", "sell_lg_amount",
                "buy_elg_amount", "sell_elg_amount",
            ]
            if name in field_set
        ]
        if flow_fields and "amount" in field_set:
            flow = generator.choice(flow_fields)
            signed_flow = (
                "(buy_lg_amount-sell_lg_amount)"
                if {"buy_lg_amount", "sell_lg_amount"}.issubset(field_set)
                else flow
            )
            candidates["capital_flow"] = [
                lambda: f"rank(ts_mean({flow}/(amount+1e-9), {medium}))",
                lambda: f"rank(ts_sum({signed_flow}, {medium})/(ts_mean(amount, {medium})+1e-9))",
                lambda: f"rank(ts_corr({flow}/(amount+1e-9), {returns}, {medium}))",
                lambda: f"rank(({flow}/(amount+1e-9))/"
                        f"(ts_mean({flow}/(amount+1e-9), {slow})+1e-9))",
                lambda: f"rank(ts_delta({signed_flow}/(amount+1e-9), {medium}))",
            ]

        builders = candidates.get(family) or []
        if not builders:
            break
        # Keep 10% simple templates as interpretable controls.  The remaining
        # draws use the larger compositional grammar so long campaigns spend
        # their budget on distinct economic structures instead of repeatedly
        # revisiting the same canonical window grid.
        expression = (
            _random_tree_expression_for_family(family, fields, generator)
            if generator.random() < 0.90
            else generator.choice(builders)()
        )
        if (
            validate(expression, fields) is None
            and not audit_expression_semantics(expression, market)["errors"]
            and mechanism_compatible(expression, family)
        ):
            return expression

    # Every supported family has a conservative deterministic fallback.  Do
    # not silently return a different mechanism, because that would corrupt
    # diversity accounting and the factor's research lineage.
    fallbacks = {
        "momentum": "rank(ts_delta(close, 60)/(delay(close, 60)+1e-9))",
        "reversal": "-rank(ts_delta(close, 5)/(delay(close, 5)+1e-9))",
        "volatility": "-rank(ts_std(ts_delta(close, 1)/(delay(close, 1)+1e-9), 40))",
        "liquidity": "-rank(ts_mean(amount, 40))",
        "volume_price_interaction": "rank(ts_corr(ts_delta(close, 1)/(delay(close, 1)+1e-9), amount, 40))",
        "gap_intraday": "rank(ts_mean((open-delay(close, 1))/(delay(close, 1)+1e-9), 40))",
        "price_relationship": "rank(ts_corr(high/(close+1e-9), low/(close+1e-9), 40))",
        "valuation": "-rank(pb)",
        "size": "-rank(total_mv)",
        "capital_flow": "rank(ts_mean(net_mf_amount/(amount+1e-9), 40))",
    }
    expression = fallbacks.get(family, "")
    if (
        expression
        and validate(expression, fields) is None
        and not audit_expression_semantics(expression, market)["errors"]
        and mechanism_compatible(expression, family)
    ):
        return expression
    raise ValueError(f"无法为收益机制 {family} 生成字段兼容的随机表达式")


def mutate_expression(
    expr: str,
    templates: list[str] | None = None,
    fields: list[str] | None = None,
    rng: random.Random | None = None,
) -> str:
    """轻量随机变异: 换窗口/翻方向/加 rank。"""
    generator = rng or random
    out = expr
    for old, new in [(str(a), str(b)) for a in _WINDOWS for b in _WINDOWS if a != b]:
        token = f", {old})"
        if token in out and generator.random() < 0.3:
            out = out.replace(token, f", {new})", 1)
            break
    if out == expr:
        out = f"-({expr})" if not expr.startswith("-") else expr[1:].strip("()") or expr
    return out if validate(out, fields or _FIELDS) is None else expr


# ============================================================
# 模板驱动的提示词构造
# ============================================================

def _build_system_prompt(
    template: dict,
    fields: list[str] | None = None,
    portfolio_mode: str = "long_short",
    market: str = "us",
    direction: int = 1,
    direction_policy: str = "both_train_select",
    target_family: str | None = None,
) -> str:
    """从模板组装 system prompt。约束块强制置顶 (外层不可稀释)。"""
    ops_doc = "\n".join(f"- {k}: {v}" for k, v in OPERATORS_DOC.items())
    anti = template.get("anti_overfit_instruction", "")
    sys_tpl = template.get("system_prompt", DEFAULT_MINER_TEMPLATE["system_prompt"])
    fields = fields or _FIELDS
    strategy_part = sys_tpl.format(fields=", ".join(fields), ops=ops_doc, anti=anti)
    family_instruction = (
        f"【本轮指定收益机制】{target_family}: "
        f"{MECHANISM_LABELS.get(target_family, target_family)}。"
        "必须返回完全相同的 mechanism_family；表达式字段/算子必须与该机制相容。\n"
        if target_family
        else ""
    )
    direction_instruction = (
        (
            "每个候选都在训练安全层同时评价 +1（高值偏多）与 "
            "-1（低值偏多）；系统计入双向试验惩罚后选优并冻结。"
            f"完全同分时优先 {direction:+d}"
        )
        if direction_policy == "both_train_select"
        else (
            "方向冻结为 +1：因子值越高，越偏向多头"
            if direction == 1
            else "方向冻结为 -1：因子值越低，越偏向多头；系统会反向排序"
        )
    )
    mode_policy = (
        "只能做多，按选中方向挑选股票，禁止依赖做空获利。"
        if portfolio_mode == "long_only"
        else "允许多空，按选中方向建立多头侧，反方向建立空头侧。"
    )

    # 强制约束块 (置顶, 外层改写不能削弱)
    constraints = (
        f"【研究任务约束】市场: {market}；持仓模式: {portfolio_mode}。"
        f"{mode_policy}\n"
        f"【信号方向】{direction_instruction}；"
        f"{'其余股票保持空仓，不建立空头。' if portfolio_mode == 'long_only' else '另一侧作为空头组合。'}\n"
        f"【硬约束 — 违反者无效】\n"
        f"可用字段 ({len(fields)}个): {', '.join(fields)}\n"
        "【字段语义与来源】\n"
        f"{render_field_contract(market, fields)}\n"
        "禁止把绝对前复权价格尺度直接与原始成交额 amount 混合；"
        "价格变化必须先归一化为收益率或振幅比例。\n"
        f"{family_instruction}"
        f"可用算子 ({len(OPERATORS_DOC)}个):\n{ops_doc}\n"
        f"窗口: 1..250 整数\n"
        "权威目标: 改善 V4.2 连续学习分的最弱组件，同时不得削弱硬门槛；"
        "费后收益/下置信界、HAC 置信度、跨期稳定性、分位单调性、"
        "压力成本和可实施性不能由高 ICIR 抵消。\n"
        "输出格式: 只回复 JSON: "
        "{\"expression\":\"...\",\"hypothesis\":\"...\","
        "\"mechanism_family\":\"...\","
        "\"change_axis\":\"window|transform|normalization|combination|new_draft\","
        "\"reflection\":\"从反馈提炼的经验与本次改变\","
        "\"targeted_failures\":[\"本次针对的失败原因\"],"
        "\"expected_effect\":\"预期改善的评价组件\"}\n"
        "以上五个键全部必填，不得省略 expected_effect；"
        "targeted_failures 必须是 JSON 数组。\n"
    )
    return constraints + "\n" + strategy_part


def _feedback_envelopes(
    nodes: list[dict],
    *,
    market: str,
    portfolio_mode: str,
    direction: int,
) -> list[dict]:
    envelopes = []
    for node in nodes:
        if node.get("feedback_summary"):
            envelopes.append(dict(node["feedback_summary"]))
            continue
        envelopes.append(build_feedback_envelope(
            node_id=node.get("id"),
            parent_id=node.get("parent_id"),
            task_name=str(node.get("task_name") or ""),
            expression=str(node.get("expression") or ""),
            hypothesis=str(node.get("hypothesis") or ""),
            source=str(node.get("source") or "unknown"),
            status=str(node.get("status") or "ok"),
            error=node.get("error"),
            public_score=node.get("public_score"),
            public_metrics=dict(node.get("public_metrics") or {}),
            evaluation_protocol=str(
                node.get("evaluation_protocol")
                or (node.get("public_metrics") or {}).get(
                    "protocol_version",
                    "legacy_unoriented",
                )
            ),
            market=market,
            portfolio_mode=portfolio_mode,
            direction=direction,
            proposal_meta=dict(node.get("proposal_meta") or {}),
        ))
    return envelopes


def _build_context_block(
    top_nodes: list[dict],
    template: dict,
    *,
    market: str = "us",
    portfolio_mode: str = "long_short",
    direction: int = 1,
) -> str:
    """Compatibility wrapper around the structured V4 feedback renderer."""
    context, _ = build_inner_feedback_context(
        _feedback_envelopes(
            top_nodes,
            market=market,
            portfolio_mode=portfolio_mode,
            direction=direction,
        ),
        template,
    )
    return context


# ============================================================
# 主提案接口
# ============================================================

async def propose(
    template_or_spec: dict,
    op: str,
    task: dict,
    top_nodes: list[dict],
    provider: dict | None,
    fields: list[str] | None = None,
    trace_context: dict | None = None,
    rng: random.Random | None = None,
    target_family: str | None = None,
    deliberate_random: bool = False,
) -> tuple[str, str, str, dict]:
    """返回 (expression, hypothesis, source, proposal_meta).

    template_or_spec: MinerTemplate (v2) 或 HarnessSpec (v1 兼容)
    LLM 失败/未配置时回退随机；deliberate_random 表示任务主动选择
    无 LLM 随机基线，不得把它记录成供应商故障或模型回退。
    """
    # 兼容 v1 spec 和 v2 template
    is_v2 = "draft_strategy" in template_or_spec
    template = template_or_spec if is_v2 else None
    portfolio_mode = task.get("mode", "long_short")
    market = task.get("market", "us")
    direction = int(task.get("direction", 1))
    direction_policy = str(
        task.get("direction_policy") or "both_train_select"
    )
    target_family = str(
        target_family
        or task.get("target_family")
        or select_target_mechanism(top_nodes, market, rng)
    )
    if target_family not in mechanisms_for_market(market):
        raise ValueError(f"未知或不可用于 {market} 的收益机制: {target_family}")
    fallback_reason = "provider_not_configured"
    text: str | None = None

    if provider:
        try:
            system = (
                _build_system_prompt(
                    template,
                    fields,
                    portfolio_mode,
                    market,
                    direction,
                    direction_policy,
                    target_family,
                )
                if template
                else _system_prompt_old(
                    template_or_spec,
                    fields,
                    portfolio_mode,
                    market,
                    direction,
                )
            )
            if template:
                context, feedback_snapshot = build_inner_feedback_context(
                    _feedback_envelopes(
                        top_nodes,
                        market=market,
                        portfolio_mode=portfolio_mode,
                        direction=direction,
                    ),
                    template,
                )
            else:
                context = _context_block_old(top_nodes, template_or_spec)
                feedback_snapshot = {
                    "schema_version": "legacy",
                    "context_fingerprint": "",
                }
            base = max(
                (
                    node
                    for node in top_nodes
                    if str(node.get("status") or "ok") == "ok"
                    and mechanism_from_item(node) == target_family
                ),
                key=lambda node: float(node.get("public_score") or 0.0),
                default=None,
            )
            dsl_examples = (
                "\n".join(
                    f"- {row}"
                    for row in template.get(
                        "dsl_exploration_templates",
                        [],
                    )[:10]
                )
                if template
                else ""
            )
            context_instruction = (
                template.get("context_strategy", "")
                if template
                else ""
            )
            scientific_directive = (
                redact_value(template.get("_scientific_governor_directive") or {})
                if template
                else {}
            )
            directive_instruction = (
                "\n第三层科学总督指令（只约束研究方向，不替代你的表达式判断）: "
                f"{scientific_directive}\n"
                "你拥有直接提出完整因子表达式的权力；算法种子如存在仅供参考。"
                if scientific_directive else ""
            )

            if op == "draft":
                draft_inst = template.get("draft_strategy", "提出与历史不同的新因子。") if template else "请提出一个与历史尝试思路不同的新因子。"
                div_inst = template.get("diversity_instruction", "") if template else ""
                user = (
                    f"任务: market={market}, portfolio_mode={portfolio_mode}, direction_policy={direction_policy}, tie_break_direction={direction}, target_family={target_family}, universe=流动性前{task['universe_n']}, 预测 horizon={task['horizon']} 交易日。\n"
                    f"评价反馈与历史经验:\n{context}\n\n"
                    f"上下文使用要求: {context_instruction}\n"
                    f"策略指令: {draft_inst}\n{div_inst}\n"
                    f"{directive_instruction}"
                    f"可探索的 DSL 结构样例（只作语法启发，不得机械复制）:\n{dsl_examples or '(无)'}"
                )
            else:  # improve
                impr_inst = template.get("improve_strategy", "改进当前最优因子。") if template else "请改进当前最优因子 (调整结构/窗口/复合), 保持简洁。"
                user = (
                    f"任务: market={market}, portfolio_mode={portfolio_mode}, direction_policy={direction_policy}, tie_break_direction={direction}, target_family={target_family}, universe=流动性前{task['universe_n']}, horizon={task['horizon']} 交易日。\n"
                    f"当前最优: {base['expression'] if base else '无'} "
                    f"(learning_score={base['public_score']:.3f} icir={base['public_metrics'].get('icir',0):+.2f})\n"
                    f"完整评价反馈与经验:\n{context}\n\n"
                    f"上下文使用要求: {context_instruction}\n"
                    f"改进策略: {impr_inst}\n"
                    f"{directive_instruction}"
                    f"可探索的 DSL 结构样例（只作语法启发，不得机械复制）:\n{dsl_examples or '(无)'}"
                )

            temp_val = float(template.get("llm_temperature", 0.9)) if template else float(template_or_spec.get("llm_temperature", 0.9))
            trace = {
                **(trace_context or {}),
                "role": str(
                    (trace_context or {}).get("llm_role") or "inner"
                ),
                "phase": f"proposal_{op}",
                "task_name": task.get("name"),
                "feedback_fingerprint": feedback_snapshot.get(
                    "context_fingerprint",
                    "",
                ),
                "feedback_schema": feedback_snapshot.get("schema_version"),
            }
            text = await llm.chat(
                provider,
                system,
                user,
                temp_val,
                trace=trace,
            )
            data = llm.extract_json(text)
            expr = str(data.get("expression", "")).strip()
            err = validate(expr, fields or _FIELDS)
            if err:
                raise llm.LLMError(f"表达式非法: {err} | {expr}")
            hypothesis = str(data.get("hypothesis") or "").strip()[:500]
            declared_family = str(data.get("mechanism_family") or "").strip()
            reflection = str(data.get("reflection") or "").strip()[:800]
            change_axis = str(data.get("change_axis") or "").strip()
            expected_effect = str(
                data.get("expected_effect") or ""
            ).strip()[:500]
            targeted = data.get("targeted_failures") or []
            if not isinstance(targeted, list):
                targeted = [str(targeted)]
            targeted = [
                str(item).strip()[:240]
                for item in targeted[:6]
                if str(item).strip()
            ]
            if not hypothesis or not reflection:
                raise llm.LLMError(
                    "LLM 输出缺少 hypothesis/reflection"
                )
            if op == "draft" and not change_axis:
                change_axis = "new_draft"
            if change_axis not in _CHANGE_AXES:
                raise llm.LLMError(
                    "change_axis 必须为 window/transform/normalization/"
                    "combination/new_draft"
                )
            if op == "improve" and change_axis == "new_draft":
                raise llm.LLMError("改进节点必须声明唯一改动轴")
            if declared_family != target_family:
                raise llm.LLMError(
                    f"mechanism_family 必须为本轮指定的 {target_family}"
                )
            if not mechanism_compatible(expr, target_family):
                raise llm.LLMError(
                    f"表达式字段/算子与收益机制 {target_family} 不相容"
                )
            semantic_audit = audit_expression_semantics(expr, market)
            if semantic_audit["errors"]:
                raise llm.LLMError("；".join(semantic_audit["errors"]))
            structural_similarity = max(
                (
                    expression_similarity(expr, str(node.get("expression") or ""))
                    for node in top_nodes
                    if node.get("expression")
                ),
                default=0.0,
            )
            if op == "draft" and structural_similarity >= 0.84:
                raise llm.LLMError(
                    f"新草稿结构相似度 {structural_similarity:.3f} 过高"
                )
            if top_nodes and not targeted:
                raise llm.LLMError(
                    "已有评价反馈时 targeted_failures 不能为空"
                )
            semantic_normalizations = []
            if not expected_effect:
                # expected_effect is useful explanatory metadata, but it is not
                # part of expression validity or the evaluator gate.  Some
                # otherwise complete DeepSeek responses omit only this final
                # key.  Preserve the substantive proposal and make the
                # deterministic repair explicit instead of silently replacing
                # it with a random expression.
                basis = targeted or [reflection]
                expected_effect = (
                    "模型未单列 expected_effect；依据其失败目标规范化补全："
                    + "；".join(basis)
                )[:500]
                semantic_normalizations.append(
                    "expected_effect_derived_from_targeted_failures"
                )
            proposal_meta = {
                "reflection": reflection,
                "change_axis": change_axis,
                "targeted_failures": targeted,
                "expected_effect": expected_effect,
                "semantic_normalizations": semantic_normalizations,
                "target_family": target_family,
                "declared_family": declared_family,
                "inferred_family": mechanism_from_item({
                    "expression": expr,
                    "hypothesis": hypothesis,
                }),
                "family_match": True,
                "semantic_audit": semantic_audit,
                "max_structural_similarity": round(structural_similarity, 6),
                "feedback_context_fingerprint": feedback_snapshot.get(
                    "context_fingerprint",
                    "",
                ),
                "feedback_protocol": feedback_snapshot.get(
                    "protocol_version",
                ),
            }
            ensure_training_safe(proposal_meta)
            await llm.mark_validation(
                text,
                accepted=True,
                trace_meta_updates={
                    "semantic_normalizations": semantic_normalizations,
                },
            )
            return (
                expr,
                hypothesis,
                "llm",
                proposal_meta,
            )

        except Exception as exc:
            fallback_reason = redact_text(exc, 300)
            await llm.mark_validation(
                text,
                accepted=False,
                error=fallback_reason,
            )
            logger.warning("LLM 回退随机: %s", fallback_reason[:200])

    # 回退随机
    tpls = template.get("dsl_exploration_templates") if template else None
    if deliberate_random:
        fallback_meta = {
            "reflection": "任务主动配置为无 LLM 随机基线；候选不读取模型反馈。",
            "targeted_failures": [],
            "expected_effect": "独立估计机制约束随机候选的训练层分布",
            "proposal_mode": "random",
            "random_reason": "configured_random_mode",
            "target_family": target_family,
            "declared_family": target_family,
            "family_match": True,
            "change_axis": "new_draft" if op == "draft" else "window",
        }
    else:
        fallback_meta = {
            "reflection": "LLM 不可用或输出无效；本轮使用确定性随机回退，不作为模型经验。",
            "targeted_failures": [],
            "expected_effect": "仅维持搜索连续性",
            "fallback_reason": fallback_reason,
            "target_family": target_family,
            "declared_family": target_family,
            "family_match": True,
            "change_axis": "new_draft" if op == "draft" else "window",
        }
    known_expressions = {
        str(node.get("expression") or "").strip()
        for node in top_nodes
        if str(node.get("expression") or "").strip()
    }

    def fresh_targeted_expression() -> tuple[str, int]:
        """Prefer a new AST shape/window before accepting a repeat."""
        last = ""
        for retry in range(32):
            last = random_expression_for_family(
                target_family, fields or _FIELDS, rng
            )
            if last not in known_expressions:
                return last, retry
        # A saturated tiny custom field set may have no unseen expression.  A
        # valid repeat is preferable to crashing a long-running campaign, but
        # the retry count remains observable in proposal metadata.
        return last, 32

    if op == "improve" and top_nodes:
        base = max(
            (
                node
                for node in top_nodes
                if str(node.get("status") or "ok") == "ok"
            ),
            key=lambda node: float(node.get("public_score") or 0.0),
            default=top_nodes[0],
        )
        use_local_mutation = (rng or random).random() < 0.35
        expression = (
            mutate_expression(base["expression"], tpls, fields, rng)
            if use_local_mutation
            else ""
        )
        semantic_audit = audit_expression_semantics(expression, market)
        novelty_retries = 0
        if (
            not expression
            or expression in known_expressions
            or semantic_audit["errors"]
            or not mechanism_compatible(expression, target_family)
        ):
            expression, novelty_retries = fresh_targeted_expression()
            semantic_audit = audit_expression_semantics(expression, market)
        fallback_meta.update({
            "semantic_audit": semantic_audit,
            "random_strategy": (
                "local_mutation" if use_local_mutation and novelty_retries == 0
                else "fresh_mechanism_grammar"
            ),
            "novelty_retries": novelty_retries,
            "inferred_family": mechanism_from_item({
                "expression": expression,
                "hypothesis": MECHANISM_LABELS[target_family],
            }),
        })
        return (
            expression,
            (
                f"机制约束随机变异：{MECHANISM_LABELS[target_family]}"
                if deliberate_random
                else "随机变异自当前最优"
            ),
            "random",
            fallback_meta,
        )
    expression, novelty_retries = fresh_targeted_expression()
    fallback_meta.update({
        "semantic_audit": audit_expression_semantics(expression, market),
        "random_strategy": "fresh_mechanism_grammar",
        "novelty_retries": novelty_retries,
        "inferred_family": mechanism_from_item({
            "expression": expression,
            "hypothesis": MECHANISM_LABELS[target_family],
        }),
    })
    return (
        expression,
        f"机制约束随机生成：{MECHANISM_LABELS[target_family]}",
        "random",
        fallback_meta,
    )


async def propose_batch(
    template: dict,
    requests: list[dict],
    provider: dict | None,
    *,
    fields: list[str],
    trace_context: dict | None = None,
) -> list[tuple[str, str, str, dict]]:
    """Propose several preassigned candidates in one audited LLM call.

    Invalid or missing members stay explicit ``llm_rejected`` records.  They
    are never silently replaced by random candidates, which preserves the
    scientific distinction between the LLM, cold-LLM, and random arms.
    """
    if not requests:
        return []
    if provider is None:
        return [
            await propose(
                template,
                row["op"],
                row["task"],
                row["feedback_nodes"],
                None,
                fields=fields,
                trace_context={**(trace_context or {}), "request_id": row["request_id"]},
                rng=row.get("rng"),
                target_family=row["target_family"],
            )
            for row in requests
        ]

    first_task = requests[0]["task"]
    system = _build_system_prompt(
        template,
        fields,
        first_task.get("mode", "long_short"),
        first_task.get("market", "us"),
        int(first_task.get("direction", 1)),
        str(first_task.get("direction_policy") or "both_train_select"),
        None,
    ) + (
        "\n【批量协议】仅返回 JSON 对象 {\"candidates\":[...]}；"
        "每个成员必须原样返回 request_id，且独立满足其指定机制。"
    )
    rendered = []
    snapshots = []
    for row in requests:
        task = row["task"]
        context, snapshot = build_inner_feedback_context(
            _feedback_envelopes(
                row["feedback_nodes"],
                market=task.get("market", "us"),
                portfolio_mode=task.get("mode", "long_short"),
                direction=int(task.get("direction", 1)),
            ),
            template,
        )
        snapshots.append(snapshot)
        base = row.get("base_node") or {}
        search_seed = row.get("search_seed")
        rendered.append({
            "request_id": row["request_id"],
            "task_name": task.get("name"),
            "market": task.get("market", "us"),
            "portfolio_mode": task.get("mode", "long_short"),
            "universe_n": task.get("universe_n"),
            "horizon": task.get("horizon"),
            "operation": row["op"],
            "target_family": row["target_family"],
            "base_expression": base.get("expression"),
            "layer1_seed_expression": (
                search_seed.expression if search_seed is not None else None
            ),
            "layer1_seed_hypothesis": (
                search_seed.hypothesis if search_seed is not None else None
            ),
            "layer1_search_algorithm": (
                search_seed.metadata.get("search_algorithm")
                if search_seed is not None else None
            ),
            "feedback": context[:4500],
        })
    user = (
        "你是因子机制科学家 LLM，拥有直接提出完整因子表达式的权力。"
        "算法种子若存在只作可选灵感，不是前置条件，也不要求复制；"
        "只在能依据训练安全反馈说明理由时做结构化起草、单轴变异或修复；不得访问或猜测"
        "META_HOLDOUT、FACTOR_VAULT 或冻结评级结果。为以下预注册请求各生成一个候选。"
        "每个候选必须包含 "
        "request_id, expression, hypothesis, mechanism_family, change_axis, "
        "reflection, targeted_failures, expected_effect。\n"
        + "\n第三层科学总督指令: "
        + json.dumps(
            redact_value(template.get("_scientific_governor_directive") or {}),
            ensure_ascii=False,
        )
        + "\n请求: "
        + json.dumps(rendered, ensure_ascii=False)
    )
    text: str | None = None
    try:
        text = await llm.chat(
            provider,
            system,
            user,
            float(template.get("llm_temperature", 0.9)),
            trace={
                **(trace_context or {}),
                "role": str(
                    (trace_context or {}).get("llm_role") or "inner"
                ),
                "phase": "proposal_batch",
                "task_name": "batch",
                "batch_size": len(requests),
                "feedback_fingerprints": [
                    row.get("context_fingerprint", "") for row in snapshots
                ],
            },
        )
        payload = llm.extract_json(text)
        candidates = payload.get("candidates") or []
        if not isinstance(candidates, list):
            raise llm.LLMError("batch candidates 必须是数组")
        by_id = {
            str(item.get("request_id")): item
            for item in candidates
            if isinstance(item, dict) and item.get("request_id") is not None
        }
        results: list[tuple[str, str, str, dict]] = []
        valid_count = 0
        errors: list[str] = []
        for row, snapshot in zip(requests, snapshots):
            request_id = str(row["request_id"])
            item = by_id.get(request_id)
            error = ""
            if item is None:
                error = "LLM 未返回该 request_id"
                item = {}
            expr = str(item.get("expression") or "").strip()
            hypothesis = str(item.get("hypothesis") or "").strip()[:500]
            reflection = str(item.get("reflection") or "").strip()[:800]
            declared = str(item.get("mechanism_family") or "").strip()
            change_axis = str(item.get("change_axis") or "").strip()
            targeted = item.get("targeted_failures") or []
            if not isinstance(targeted, list):
                targeted = [str(targeted)]
            targeted = [str(value).strip()[:240] for value in targeted[:6] if str(value).strip()]
            expected_effect = str(item.get("expected_effect") or "").strip()[:500]
            market = row["task"].get("market", "us")
            if not error:
                error = validate(expr, fields) or ""
            if not error and (not hypothesis or not reflection):
                error = "缺少 hypothesis/reflection"
            if not error and declared != row["target_family"]:
                error = f"mechanism_family 必须为 {row['target_family']}"
            if not error and change_axis not in _CHANGE_AXES:
                error = "change_axis 非法"
            if not error and row["op"] == "improve" and change_axis == "new_draft":
                error = "improve 必须使用单一改动轴"
            if not error and not mechanism_compatible(expr, row["target_family"]):
                error = "表达式与指定收益机制不相容"
            semantic_audit = audit_expression_semantics(expr, market) if expr else {"errors": []}
            if not error and semantic_audit["errors"]:
                error = "；".join(semantic_audit["errors"])
            seed_expression = (
                row["search_seed"].expression
                if row.get("search_seed") is not None else ""
            )
            seed_similarity = (
                expression_similarity(expr, seed_expression)
                if expr and seed_expression else None
            )
            direct_expression_authority = bool(
                (trace_context or {}).get("direct_expression_authority", False)
            )
            if (
                not error
                and seed_similarity is not None
                and seed_similarity < 0.20
                and not direct_expression_authority
            ):
                error = (
                    f"第二层偏离第一层种子过远: similarity={seed_similarity:.3f}"
                )
            similarity = max(
                (
                    expression_similarity(expr, str(node.get("expression") or ""))
                    for node in row["feedback_nodes"]
                    if node.get("expression")
                ),
                default=0.0,
            )
            if not error and row["op"] == "draft" and similarity >= 0.84:
                error = f"新草稿结构相似度 {similarity:.3f} 过高"
            if not error and row["feedback_nodes"] and not targeted:
                error = "已有反馈时 targeted_failures 不能为空"
            meta = {
                "request_id": request_id,
                "reflection": reflection,
                "targeted_failures": targeted,
                "expected_effect": expected_effect,
                "change_axis": change_axis or ("new_draft" if row["op"] == "draft" else ""),
                "target_family": row["target_family"],
                "declared_family": declared,
                "semantic_audit": semantic_audit,
                "max_structural_similarity": round(similarity, 6),
                "feedback_context_fingerprint": snapshot.get("context_fingerprint", ""),
                "batch_size": len(requests),
                "architecture_layer": int(
                    (trace_context or {}).get("architecture_layer", 2)
                ),
                "direct_expression_authority": direct_expression_authority,
                "layer1_seed_expression": (
                    row["search_seed"].expression
                    if row.get("search_seed") is not None else None
                ),
                "layer1_search_algorithm": (
                    row["search_seed"].metadata.get("search_algorithm")
                    if row.get("search_seed") is not None else None
                ),
                "layer2_changed_seed": (
                    expr != row["search_seed"].expression
                    if row.get("search_seed") is not None else None
                ),
                "layer1_seed_similarity": (
                    round(seed_similarity, 6)
                    if seed_similarity is not None else None
                ),
            }
            if error:
                meta["validation_error"] = redact_text(error, 500)
                results.append((expr, hypothesis, "llm_rejected", meta))
                errors.append(f"{request_id}: {error}")
            else:
                ensure_training_safe(meta)
                results.append((expr, hypothesis, "llm", meta))
                valid_count += 1
        await llm.mark_validation(
            text,
            accepted=valid_count == len(requests),
            error="; ".join(errors),
            trace_meta_updates={
                "valid_candidates": valid_count,
                "rejected_candidates": len(requests) - valid_count,
            },
        )
        return results
    except Exception as exc:
        error = redact_text(exc, 500)
        await llm.mark_validation(text, accepted=False, error=error)
        if text is None:
            # Provider/auth/network/balance failures are infrastructure state,
            # not rejected factor hypotheses.  Let the worker circuit-break
            # before it consumes any candidate-evaluation budget.
            raise RuntimeError(
                f"第二层 LLM provider 不可用: {error}"
            ) from exc
        return [
            (
                "",
                "",
                "llm_rejected",
                {
                    "request_id": str(row["request_id"]),
                    "target_family": row["target_family"],
                    "declared_family": "",
                    "change_axis": "",
                    "validation_error": error,
                    "batch_size": len(requests),
                },
            )
            for row in requests
        ]


# ============================================================
# 向后兼容: v1 旧版函数 (A组仍在用)
# ============================================================

_OLD_SYSTEM = """你是量化因子研究员。基于当前市场日线数据设计横截面选股因子表达式。
可用字段: {fields}；价格是前复权口径，amount 是原始成交额或 raw_close×vol 代理，禁止直接混用绝对价格尺度与 amount。
可用算子:
{ops}
规则: 只能用以上字段与算子; 窗口为 1..250 整数; 表达式一行; 目标是最大化样本内 RankIC 的稳健性而非峰值;
禁止只对特定时段有效的取巧构造。{anti}
只回复 JSON: {{"expression": "...", "hypothesis": "一句话经济学假设"}}"""

_ANTI = "\n特别要求: 避免过拟合——偏好简单、有经济含义、跨行业普适的结构。"


def _system_prompt_old(
    spec: dict,
    fields: list[str] | None = None,
    portfolio_mode: str = "long_short",
    market: str = "us",
    direction: int = 1,
) -> str:
    ops = "\n".join(f"- {k}: {v}" for k, v in OPERATORS_DOC.items())
    anti = _ANTI if spec.get("anti_overfit_instruction") else ""
    policy = (
        f"\n研究任务: 市场={market}, 持仓模式={portfolio_mode}, "
        f"信号方向={'高值偏多' if direction == 1 else '低值偏多'}({direction:+d})。"
    )
    policy += "只能做多，禁止依赖负向信号做空获利。" if portfolio_mode == "long_only" else "允许多空双侧组合。"
    return _OLD_SYSTEM.format(fields=", ".join(fields or _FIELDS), ops=ops, anti=anti) + policy


def _context_block_old(top_nodes: list[dict], spec: dict) -> str:
    if not top_nodes:
        return "(暂无历史尝试)"
    k = int(spec.get("context_top_k", 5))
    lines = []
    for n in top_nodes[:k]:
        lines.append(
            f"- learning_score={n['public_score']:.3f} icir={n['public_metrics'].get('icir')} "
            f"daily_turnover={n['public_metrics'].get('daily_turnover', n['public_metrics'].get('turnover'))} "
            f"expr: {n['expression']}"
        )
    return "\n".join(lines)
