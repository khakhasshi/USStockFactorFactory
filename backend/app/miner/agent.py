"""内层 Factor Miner: LLM 起草/改进因子表达式; 无可用 LLM 时回退随机生成 (系统可无钥运行).

纪律: 提示词只包含 INNER_PUBLIC 层指标, gate 指标永不进入上下文.
"""

import logging
import random

from ..config import DSL_FIELDS
from ..dsl.engine import OPERATORS_DOC, validate
from ..llm import client as llm

_FIELDS = DSL_FIELDS
_WINDOWS = [3, 5, 10, 20, 40, 60, 120]


def random_expression() -> str:
    f = random.choice
    w = lambda: f(_WINDOWS)  # noqa: E731
    templates = [
        lambda: f"-rank(ts_delta(close, {w()}))",
        lambda: f"rank(ts_corr({f(_FIELDS)}, {f(_FIELDS)}, {w()}))",
        lambda: f"-zscore(ts_mean(vol, {w()}) / ts_mean(vol, {f([60, 120])}))",
        lambda: f"rank((close - ts_min(low, {w()})) / (ts_max(high, {w()}) - ts_min(low, {w()})))",
        lambda: f"-rank(ts_std(close / delay(close, 1), {w()}))",
        lambda: f"zscore(ts_mean(amount, {f([3, 5])}) / ts_mean(amount, {f([40, 60])}))",
        lambda: f"-rank(ts_rank(close, {w()}))",
        lambda: f"rank(ts_delta(close, {f([40, 60, 120])})) - rank(ts_delta(close, {f([3, 5])}))",
    ]
    return f(templates)()


def mutate_expression(expr: str) -> str:
    """轻量随机变异: 换窗口/加 rank/翻方向."""
    out = expr
    for old, new in [(str(a), str(b)) for a in _WINDOWS for b in _WINDOWS if a != b]:
        token = f", {old})"
        if token in out and random.random() < 0.3:
            out = out.replace(token, f", {new})", 1)
            break
    if out == expr:
        out = f"-({expr})" if not expr.startswith("-") else expr[1:].strip("()") or expr
    return out if validate(out) is None else expr


_SYSTEM = """你是量化因子研究员。基于美股日线数据设计横截面选股因子表达式。
可用字段: {fields} (前复权价格与量额)
可用算子:
{ops}
规则: 只能用以上字段与算子; 窗口为 1..250 整数; 表达式一行; 目标是最大化样本内 RankIC 的稳健性而非峰值;
禁止只对特定时段有效的取巧构造。{anti}
只回复 JSON: {{"expression": "...", "hypothesis": "一句话经济学假设"}}"""

_ANTI = "\n特别要求: 避免过拟合——偏好简单、有经济含义、跨行业普适的结构。"


def _system_prompt(spec: dict) -> str:
    ops = "\n".join(f"- {k}: {v}" for k, v in OPERATORS_DOC.items())
    anti = _ANTI if spec.get("anti_overfit_instruction") else ""
    return _SYSTEM.format(fields=", ".join(_FIELDS), ops=ops, anti=anti)


def _context_block(top_nodes: list[dict], spec: dict) -> str:
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


async def propose(
    spec: dict,
    op: str,
    task: dict,
    top_nodes: list[dict],
    provider: dict | None,
) -> tuple[str, str, str]:
    """返回 (expression, hypothesis, source). LLM 失败/未配置时回退随机."""
    if provider:
        try:
            if op == "draft":
                user = (
                    f"任务: universe=流动性前{task['universe_n']}, 预测 horizon={task['horizon']} 交易日。\n"
                    f"历史最优尝试 (public 指标):\n{_context_block(top_nodes, spec)}\n"
                    f"请提出一个与历史尝试思路不同的新因子。"
                )
            else:
                base = top_nodes[0] if top_nodes else None
                user = (
                    f"任务: universe=流动性前{task['universe_n']}, horizon={task['horizon']} 交易日。\n"
                    f"当前最优: {base['expression'] if base else '无'} "
                    f"(score={base['public_score']:.3f})\n"
                    f"其余高分尝试:\n{_context_block(top_nodes[1:], spec)}\n"
                    f"请改进当前最优因子 (调整结构/窗口/复合), 保持简洁。"
                )
            text = await llm.chat(
                provider, _system_prompt(spec), user, float(spec.get("llm_temperature", 0.9))
            )
            data = llm.extract_json(text)
            expr = str(data.get("expression", "")).strip()
            err = validate(expr)
            if err:
                raise llm.LLMError(f"表达式非法: {err} | {expr}")
            return expr, str(data.get("hypothesis", ""))[:500], "llm"
        except Exception as e:  # noqa: BLE001  LLM 任意失败均降级, 循环不中断
            logging.getLogger("miner").warning("LLM 回退随机: %s", str(e)[:200])
    if op == "improve" and top_nodes:
        return mutate_expression(top_nodes[0]["expression"]), "随机变异自当前最优", "random"
    return random_expression(), "随机模板生成", "random"
