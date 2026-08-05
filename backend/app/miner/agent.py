"""内层 Factor Miner v2: 由 MinerTemplate 驱动 (prompt/策略/模板全部可被外层改写).

纪律: 提示词只包含训练安全的 V4 聚合反馈；最终封存层永不进入上下文.
"""

import logging
import random

from ..config import DEFAULT_MINER_TEMPLATE, DSL_FIELDS, get_dsl_fields
from ..dsl.engine import OPERATORS_DOC, validate
from ..feedback import (
    build_feedback_envelope,
    build_inner_feedback_context,
    ensure_training_safe,
)
from ..llm import client as llm
from ..observability import redact_text

logger = logging.getLogger("miner")

_FIELDS = DSL_FIELDS
_WINDOWS = [3, 5, 10, 20, 40, 60, 120]

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
        lambda: f"rank(ts_mean(abs(ts_delta(close,1))/(amount+1e-9), {w()}))",
    ]
    return f(builtin)()


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
) -> str:
    """从模板组装 system prompt。约束块强制置顶 (外层不可稀释)。"""
    ops_doc = "\n".join(f"- {k}: {v}" for k, v in OPERATORS_DOC.items())
    anti = template.get("anti_overfit_instruction", "")
    sys_tpl = template.get("system_prompt", DEFAULT_MINER_TEMPLATE["system_prompt"])
    fields = fields or _FIELDS
    strategy_part = sys_tpl.format(fields=", ".join(fields), ops=ops_doc, anti=anti)
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
        f"可用算子 ({len(OPERATORS_DOC)}个):\n{ops_doc}\n"
        f"窗口: 1..250 整数\n"
        "权威目标: 改善 V4.2 连续学习分的最弱组件，同时不得削弱硬门槛；"
        "费后收益/下置信界、HAC 置信度、跨期稳定性、分位单调性、"
        "压力成本和可实施性不能由高 ICIR 抵消。\n"
        "输出格式: 只回复 JSON: "
        "{\"expression\":\"...\",\"hypothesis\":\"...\","
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
) -> tuple[str, str, str, dict]:
    """返回 (expression, hypothesis, source, proposal_meta).

    template_or_spec: MinerTemplate (v2) 或 HarnessSpec (v1 兼容)
    LLM 失败/未配置时回退随机.
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

            if op == "draft":
                draft_inst = template.get("draft_strategy", "提出与历史不同的新因子。") if template else "请提出一个与历史尝试思路不同的新因子。"
                div_inst = template.get("diversity_instruction", "") if template else ""
                user = (
                    f"任务: market={market}, portfolio_mode={portfolio_mode}, direction_policy={direction_policy}, tie_break_direction={direction}, universe=流动性前{task['universe_n']}, 预测 horizon={task['horizon']} 交易日。\n"
                    f"评价反馈与历史经验:\n{context}\n\n"
                    f"上下文使用要求: {context_instruction}\n"
                    f"策略指令: {draft_inst}\n{div_inst}\n"
                    f"可探索的 DSL 结构样例（只作语法启发，不得机械复制）:\n{dsl_examples or '(无)'}"
                )
            else:  # improve
                impr_inst = template.get("improve_strategy", "改进当前最优因子。") if template else "请改进当前最优因子 (调整结构/窗口/复合), 保持简洁。"
                user = (
                    f"任务: market={market}, portfolio_mode={portfolio_mode}, direction_policy={direction_policy}, tie_break_direction={direction}, universe=流动性前{task['universe_n']}, horizon={task['horizon']} 交易日。\n"
                    f"当前最优: {base['expression'] if base else '无'} "
                    f"(learning_score={base['public_score']:.3f} icir={base['public_metrics'].get('icir',0):+.2f})\n"
                    f"完整评价反馈与经验:\n{context}\n\n"
                    f"上下文使用要求: {context_instruction}\n"
                    f"改进策略: {impr_inst}\n"
                    f"可探索的 DSL 结构样例（只作语法启发，不得机械复制）:\n{dsl_examples or '(无)'}"
                )

            temp_val = float(template.get("llm_temperature", 0.9)) if template else float(template_or_spec.get("llm_temperature", 0.9))
            trace = {
                **(trace_context or {}),
                "role": "inner",
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
            reflection = str(data.get("reflection") or "").strip()[:800]
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
                "targeted_failures": targeted,
                "expected_effect": expected_effect,
                "semantic_normalizations": semantic_normalizations,
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
    fallback_meta = {
        "reflection": "LLM 不可用或输出无效；本轮使用确定性随机回退，不作为模型经验。",
        "targeted_failures": [],
        "expected_effect": "仅维持搜索连续性",
        "fallback_reason": fallback_reason,
    }
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
        return (
            mutate_expression(base["expression"], tpls, fields, rng),
            "随机变异自当前最优",
            "random",
            fallback_meta,
        )
    return (
        random_expression(tpls, fields, rng),
        "随机模板生成",
        "random",
        fallback_meta,
    )


# ============================================================
# 向后兼容: v1 旧版函数 (A组仍在用)
# ============================================================

_OLD_SYSTEM = """你是量化因子研究员。基于当前市场日线数据设计横截面选股因子表达式。
可用字段: {fields} (前复权价格与量额)
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
