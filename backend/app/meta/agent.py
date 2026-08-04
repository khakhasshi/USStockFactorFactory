"""外层 Meta-Researcher: 在 HarnessSpec 白名单内提出内层配置变更.

P0-P2 阶段外层只能改声明式参数; 评估器/数据接口/门禁只读。
"""

import random

from ..llm import client as llm

# 白名单: 字段 -> (类型, 下界, 上界)
SPEC_BOUNDS = {
    "n_drafts": (int, 2, 8),
    "improve_bias": (float, 0.3, 0.9),
    "context_top_k": (int, 2, 10),
    "llm_temperature": (float, 0.3, 1.2),
    "anti_overfit_instruction": (bool, None, None),
    "min_public_icir": (float, 0.1, 0.6),
}

_SYSTEM = """你是自动化研究系统的元优化器。你优化的是"因子挖掘智能体"的超参数配置 (HarnessSpec), 不是因子本身。
可调字段与边界: {bounds}
你会看到历史配置与其 meta-score (跨任务挖出的头部因子的隐藏分位分数)。
只回复 JSON: {{"spec": {{完整新配置}}, "note": "一句话说明改动逻辑"}}"""


def _clamp(spec: dict) -> dict:
    out = {}
    for k, (typ, lo, hi) in SPEC_BOUNDS.items():
        v = spec.get(k)
        if typ is bool:
            out[k] = bool(v)
        elif v is None:
            continue
        else:
            out[k] = typ(min(max(typ(v), lo), hi))
    return out


def random_jitter(spec: dict) -> tuple[dict, str]:
    out = dict(spec)
    key = random.choice(list(SPEC_BOUNDS))
    typ, lo, hi = SPEC_BOUNDS[key]
    if typ is bool:
        out[key] = not out.get(key, True)
    elif typ is int:
        out[key] = random.randint(lo, hi)
    else:
        out[key] = round(random.uniform(lo, hi), 2)
    return _clamp({**spec, **out}), f"随机扰动 {key}"


async def propose_spec(
    incumbent_spec: dict, history: list[dict], provider: dict | None
) -> tuple[dict, str, str]:
    """返回 (new_spec, note, source)."""
    if provider:
        try:
            hist = "\n".join(
                f"- v{h['version_no']} score={h['meta_score']} spec={h['harness_spec']} ({h['status']})"
                for h in history[-8:]
            )
            user = f"在位配置: {incumbent_spec}\n历史:\n{hist or '(无)'}\n请提出下一个候选配置。"
            text = await llm.chat(
                provider, _SYSTEM.format(bounds=str({k: v[1:] for k, v in SPEC_BOUNDS.items()})), user, 0.7
            )
            data = llm.extract_json(text)
            new_spec = _clamp({**incumbent_spec, **data.get("spec", {})})
            if new_spec != incumbent_spec:
                return new_spec, str(data.get("note", ""))[:300], "llm"
        except Exception as e:  # noqa: BLE001
            import logging

            logging.getLogger("meta").warning("外层 LLM 回退随机: %s", str(e)[:200])
    spec, note = random_jitter(incumbent_spec)
    return spec, note, "random"
