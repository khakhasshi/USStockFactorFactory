"""内层 Factor Miner v2: 由 MinerTemplate 驱动 (prompt/策略/模板全部可被外层改写).

纪律: 提示词只包含 INNER_PUBLIC 层指标, gate 指标永不进入上下文.
"""

import logging
import random

from ..config import DEFAULT_MINER_TEMPLATE, DSL_FIELDS
from ..dsl.engine import OPERATORS_DOC, validate
from ..llm import client as llm

logger = logging.getLogger("miner")

_FIELDS = DSL_FIELDS
_WINDOWS = [3, 5, 10, 20, 40, 60, 120]

# ============================================================
# 随机回退 (无 LLM 时)
# ============================================================

def random_expression(templates: list[str] | None = None) -> str:
    """使用模板生成随机表达式。"""
    f = random.choice
    w = lambda: f(_WINDOWS)  # noqa: E731
    field = lambda: f(_FIELDS)  # noqa: E731

    if templates and len(templates) >= 3:
        # 从模板列表随机选一个, 填参数
        tpl = f(templates)
        expr = tpl.replace("{window}", str(w()))
        expr = expr.replace("{field1}", field()).replace("{field2}", field())
        expr = expr.replace("{field}", field())
        expr = expr.replace("{-}", "-" if random.random() < 0.5 else "")
        if validate(expr) is None:
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


def mutate_expression(expr: str, templates: list[str] | None = None) -> str:
    """轻量随机变异: 换窗口/翻方向/加 rank。"""
    out = expr
    for old, new in [(str(a), str(b)) for a in _WINDOWS for b in _WINDOWS if a != b]:
        token = f", {old})"
        if token in out and random.random() < 0.3:
            out = out.replace(token, f", {new})", 1)
            break
    if out == expr:
        out = f"-({expr})" if not expr.startswith("-") else expr[1:].strip("()") or expr
    return out if validate(out) is None else expr


# ============================================================
# 模板驱动的提示词构造
# ============================================================

def _build_system_prompt(template: dict) -> str:
    """从模板组装 system prompt。JSON 输出指令强制追加 (外层不可移除)。"""
    ops_doc = "\n".join(f"- {k}: {v}" for k, v in OPERATORS_DOC.items())
    anti = template.get("anti_overfit_instruction", "")
    sys_tpl = template.get("system_prompt", DEFAULT_MINER_TEMPLATE["system_prompt"])
    base = sys_tpl.format(fields=", ".join(_FIELDS), ops=ops_doc, anti=anti)
    # 强制追加 JSON 输出格式 (外层改写不能移除)
    if "只回复 JSON" not in base and "reply JSON" not in base.lower():
        base += '\n只回复 JSON: {"expression": "...", "hypothesis": "一句话经济学假设"}'
    return base


def _build_context_block(top_nodes: list[dict], template: dict) -> str:
    """按模板的 context_strategy 构造上下文。"""
    if not top_nodes:
        return "(暂无历史尝试)"

    top_k = min(len(top_nodes), 8)
    lines = []

    # 高分因子
    for i, n in enumerate(top_nodes[:top_k]):
        lines.append(
            f"- #{i+1} score={n['public_score']:.3f} icir={n['public_metrics'].get('icir',0):+.2f} "
            f"turnover={n['public_metrics'].get('turnover',0):.1%} expr: {n['expression']}"
        )

    # 失败模式摘要 (如果模板要求)
    ctx_strategy = template.get("context_strategy", "")
    if "失败" in ctx_strategy or "fail" in ctx_strategy.lower():
        low_nodes = [n for n in top_nodes if (n.get('public_score') or 0) < 0.3]
        if low_nodes:
            # 简单归纳: 统计常见算子
            from collections import Counter
            ops = Counter()
            for n in low_nodes[:20]:
                expr = n.get('expression', '')
                for op in ['ts_corr', 'ts_delta', 'ts_rank', 'ts_mean', 'ts_std', 'rank', 'zscore']:
                    if op in expr:
                        ops[op] += 1
            common_ops = [k for k, v in ops.most_common(2)]
            lines.append(f"⚠ 失败模式: {len(low_nodes)} 个低分尝试, 常见算子: {common_ops}")

    return "\n".join(lines)


# ============================================================
# 主提案接口
# ============================================================

async def propose(
    template_or_spec: dict,
    op: str,
    task: dict,
    top_nodes: list[dict],
    provider: dict | None,
) -> tuple[str, str, str]:
    """返回 (expression, hypothesis, source).

    template_or_spec: MinerTemplate (v2) 或 HarnessSpec (v1 兼容)
    LLM 失败/未配置时回退随机.
    """
    # 兼容 v1 spec 和 v2 template
    is_v2 = "draft_strategy" in template_or_spec
    template = template_or_spec if is_v2 else None

    if provider:
        try:
            system = _build_system_prompt(template) if template else _system_prompt_old(template_or_spec)
            context = _build_context_block(top_nodes, template) if template else _context_block_old(top_nodes, template_or_spec)

            if op == "draft":
                draft_inst = template.get("draft_strategy", "提出与历史不同的新因子。") if template else "请提出一个与历史尝试思路不同的新因子。"
                div_inst = template.get("diversity_instruction", "") if template else ""
                user = (
                    f"任务: universe=流动性前{task['universe_n']}, 预测 horizon={task['horizon']} 交易日。\n"
                    f"历史尝试:\n{context}\n\n"
                    f"策略指令: {draft_inst}\n{div_inst}"
                )
            else:  # improve
                base = top_nodes[0] if top_nodes else None
                impr_inst = template.get("improve_strategy", "改进当前最优因子。") if template else "请改进当前最优因子 (调整结构/窗口/复合), 保持简洁。"
                user = (
                    f"任务: universe=流动性前{task['universe_n']}, horizon={task['horizon']} 交易日。\n"
                    f"当前最优: {base['expression'] if base else '无'} "
                    f"(score={base['public_score']:.3f} icir={base['public_metrics'].get('icir',0):+.2f})\n"
                    f"其余尝试:\n{_build_context_block(top_nodes[1:], template) if template else _context_block_old(top_nodes[1:], template_or_spec)}\n\n"
                    f"改进策略: {impr_inst}"
                )

            temp_val = float(template.get("llm_temperature", 0.9)) if template else float(template_or_spec.get("llm_temperature", 0.9))
            text = await llm.chat(provider, system, user, temp_val)
            data = llm.extract_json(text)
            expr = str(data.get("expression", "")).strip()
            err = validate(expr)
            if err:
                raise llm.LLMError(f"表达式非法: {err} | {expr}")
            return expr, str(data.get("hypothesis", ""))[:500], "llm"

        except Exception as e:
            logger.warning("LLM 回退随机: %s", str(e)[:200])

    # 回退随机
    tpls = template.get("dsl_exploration_templates") if template else None
    if op == "improve" and top_nodes:
        return mutate_expression(top_nodes[0]["expression"], tpls), "随机变异自当前最优", "random"
    return random_expression(tpls), "随机模板生成", "random"


# ============================================================
# 向后兼容: v1 旧版函数 (A组仍在用)
# ============================================================

_OLD_SYSTEM = """你是量化因子研究员。基于美股日线数据设计横截面选股因子表达式。
可用字段: {fields} (前复权价格与量额)
可用算子:
{ops}
规则: 只能用以上字段与算子; 窗口为 1..250 整数; 表达式一行; 目标是最大化样本内 RankIC 的稳健性而非峰值;
禁止只对特定时段有效的取巧构造。{anti}
只回复 JSON: {{"expression": "...", "hypothesis": "一句话经济学假设"}}"""

_ANTI = "\n特别要求: 避免过拟合——偏好简单、有经济含义、跨行业普适的结构。"


def _system_prompt_old(spec: dict) -> str:
    ops = "\n".join(f"- {k}: {v}" for k, v in OPERATORS_DOC.items())
    anti = _ANTI if spec.get("anti_overfit_instruction") else ""
    return _OLD_SYSTEM.format(fields=", ".join(_FIELDS), ops=ops, anti=anti)


def _context_block_old(top_nodes: list[dict], spec: dict) -> str:
    if not top_nodes:
        return "(暂无历史尝试)"
    k = int(spec.get("context_top_k", 5))
    lines = []
    for n in top_nodes[:k]:
        lines.append(
            f"- score={n['public_score']:.3f} icir={n['public_metrics'].get('icir')} "
            f"turnover={n['public_metrics'].get('turnover')} expr: {n['expression']}"
        )
    return "\n".join(lines)
