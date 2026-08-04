"""外层 Meta-Researcher v2: 可改写 MinerTemplate (system prompt/搜索策略/上下文构造/评分偏好/DSL模板).

设计原则 (来自 AIDE² 论文经验):
- 外层优化对象 = Miner 的"源代码级"文本 (prompt + 策略 + 模板), 不是 6 个数字参数
- 安全边界: MetaValidator 确保外层不能触碰评估器/数据层/隔离边界
- 无 LLM 时回退随机 (系统可无钥运行)
"""

import json
import logging
import random
from copy import deepcopy

from ..config import DEFAULT_MINER_TEMPLATE
from ..llm import client as llm

logger = logging.getLogger("meta")

# ============================================================
# 安全边界: 外层不可修改的字段
# ============================================================
READONLY_KEYS = frozenset({
    "_readonly",           # 只读元数据块
    "min_public_icir",     # 入库门槛 (防止放水)
    "llm_temperature",     # 温度上限由人类 Gate 控制
})

FORBIDDEN_WORDS = [
    "FACTOR_VAULT", "META_HOLDOUT", "META_TRAIN", "INNER_PUBLIC",
    "trade_year", "trade_date == ", "era ==",
    "2025", "2026",  # 不得针对特定年份优化
    "import ", "exec(", "eval(", "subprocess", "os.",
]


def validate_template(template: dict) -> str | None:
    """安全检查: 模板是否试图触碰只读区域或注入后门。返回 None=通过, 否则返回错误信息。"""
    # 跳过 _readonly 元数据块 (系统注入, 外层不可改)
    for key in READONLY_KEYS:
        if key == "_readonly":
            continue  # 系统元数据, 允许存在
        if key in template and key != "min_public_icir" and key != "llm_temperature":
            if template[key] != DEFAULT_MINER_TEMPLATE.get(key):
                return f"禁止修改只读字段: {key}"

    # 检查所有文本字段是否包含禁用词 (跳过 _readonly)
    for key, val in template.items():
        if key == "_readonly":
            continue
        if isinstance(val, str):
            for word in FORBIDDEN_WORDS:
                if word.lower() in val.lower():
                    return f"模板字段 '{key}' 包含禁用词: {word}"
        elif isinstance(val, dict):
            for sub_key, sub_val in val.items():
                if isinstance(sub_val, str):
                    for word in FORBIDDEN_WORDS:
                        if word.lower() in sub_val.lower():
                            return f"模板字段 '{key}.{sub_key}' 包含禁用词: {word}"

    # 必须保留占位符
    for required in ["system_prompt", "draft_strategy", "improve_strategy"]:
        if required not in template or not isinstance(template[required], str):
            return f"缺少必填文本字段: {required}"

    return None


def clamp_template(raw: dict, base: dict | None = None) -> dict:
    """将 LLM 输出合并到基础模板, 做安全裁剪。自动修复 LLM 的类型错误 (字符串→dict/list)。"""
    import ast as _ast

    base = deepcopy(base or DEFAULT_MINER_TEMPLATE)
    for key in READONLY_KEYS:
        raw.pop(key, None)

    # 文本字段: 允许 LLM 改写, 但检查安全
    text_keys = [
        "system_prompt", "anti_overfit_instruction",
        "draft_strategy", "improve_strategy",
        "context_strategy", "diversity_instruction",
    ]
    for key in text_keys:
        if key in raw and isinstance(raw[key], str) and len(raw[key]) > 20:
            base[key] = raw[key][:3000]  # 长度限制

    # 评分权重: LLM 可能返回字符串, 自动修复
    if "scoring_weights" in raw:
        sw = raw["scoring_weights"]
        if isinstance(sw, str):
            try:
                sw = json.loads(sw)
            except (json.JSONDecodeError, TypeError):
                try:
                    sw = _ast.literal_eval(sw)
                except (ValueError, SyntaxError):
                    sw = None
        if isinstance(sw, dict):
            base["scoring_weights"] = {
                "icir_weight": max(0.1, min(0.8, float(sw.get("icir_weight", 0.45)))),
                "consistency_weight": max(0.05, min(0.5, float(sw.get("consistency_weight", 0.25)))),
                "turnover_weight": max(0.1, min(0.7, float(sw.get("turnover_weight", 0.30)))),
            }

    # DSL 模板: LLM 可能返回字符串, 自动修复
    if "dsl_exploration_templates" in raw:
        tpls = raw["dsl_exploration_templates"]
        if isinstance(tpls, str):
            try:
                tpls = json.loads(tpls)
            except (json.JSONDecodeError, TypeError):
                try:
                    tpls = _ast.literal_eval(tpls)
                except (ValueError, SyntaxError):
                    tpls = None
        if isinstance(tpls, list):
            templates = []
            for t in tpls:
                if isinstance(t, str) and 5 < len(t) < 200:
                    safe = True
                    for word in FORBIDDEN_WORDS:
                        if word.lower() in t.lower():
                            safe = False
                            break
                    if safe:
                        templates.append(t)
            if len(templates) >= 2:
                base["dsl_exploration_templates"] = templates[:15]

    # 微调参数 (允许小幅调整)
    if "min_public_icir" in raw:
        base["min_public_icir"] = max(0.1, min(0.5, float(raw["min_public_icir"])))
    if "llm_temperature" in raw:
        base["llm_temperature"] = max(0.3, min(1.2, float(raw["llm_temperature"])))

    return base


def random_jitter(template: dict) -> tuple[dict, str]:
    """随机微调模板 (无 LLM 时的回退)。"""
    t = deepcopy(template)
    action = random.choice([
        "swap_weight", "add_template", "modify_instruction",
    ])
    if action == "swap_weight":
        sw = t["scoring_weights"]
        keys = list(sw.keys())
        if len(keys) >= 2:
            a, b = random.sample(keys, 2)
            sw[a], sw[b] = round(random.uniform(0.1, 0.7), 2), round(random.uniform(0.1, 0.7), 2)
            return t, f"随机交换评分权重 {a}<->{b}"
    elif action == "add_template":
        tpl = random.choice([
            "-zscore(ts_mean({field}, {window}))",
            "rank(ts_sum(amount, {window}) / ts_sum(vol, {window}))",
            "ts_rank({field}, {window})",
        ])
        t.setdefault("dsl_exploration_templates", []).append(tpl)
        return t, f"随机新增 DSL 模板: {tpl}"
    elif action == "modify_instruction":
        additions = [
            " 优先使用长窗口(>=40日)降低换手。",
            " 探索量价交互: 将成交额纳入分母构造流动性度量。",
            " 思考: 这个因子在横截面上是否与已有因子正交?",
        ]
        t["draft_strategy"] = t.get("draft_strategy", "") + random.choice(additions)
        return t, "随机扩展 draft 策略指令"
    return t, "随机微调"


# ============================================================
# 外层 LLM 提案
# ============================================================

_SYSTEM_V2 = """你是自动化因子挖掘系统的元优化器 (Meta-Optimizer)。
你的优化对象不是因子本身, 而是"内层因子挖掘智能体 (Miner) 的源代码级配置"。

你可以自由改写 Miner 的以下组件 (纯文本/配置, 不涉及可执行代码):
1. **system_prompt**: 内层 LLM 的系统提示词 (经济学假设模板/防过拟合指令/输出格式)
2. **draft_strategy**: 起草新因子的策略描述 (探索方向/机制多样性/窗口偏好)
3. **improve_strategy**: 改进已有因子的策略描述 (复合方式/算子替换/归一化)
4. **context_strategy**: 如何向 LLM 展示历史因子 (top-k/聚类/失败模式摘要)
5. **diversity_instruction**: 多样性约束措辞
6. **scoring_weights**: 评分偏好权重 (icir/consistency/turnover 三者权重)
7. **dsl_exploration_templates**: 引导探索的 DSL 模板列表 (带占位符)

安全边界 (你绝不能触碰):
- 评估器代码 (只读) | 数据层 (只读) | 四级隔离边界 (只读)
- 不得在模板中引用特定年份/era/数据层名称
- 不得注入 Python 代码或文件系统操作

你会看到:
- 当前在位模板
- 历史模板版本与其 meta-score (跨任务/多种子 gate 分数均值)
- 最近失败提案 (可选)

只回复 JSON: {{"changed_fields": ["字段名列表"], "template": {{要修改的字段}}, "note": "改动逻辑与预期效果"}}
如果认为当前模板已足够好, 返回: {{"changed_fields": [], "note": "无需改动"}}"""


async def propose_template(
    incumbent_template: dict,
    history: list[dict],
    provider: dict | None,
) -> tuple[dict, str, str]:
    """返回 (new_template, note, source).

    history: [{"version_no": int, "meta_score": float, "status": str, "template_note": str}, ...]
    """
    if provider:
        try:
            # 构造历史上下文
            hist_lines = []
            for h in history[-12:]:
                hist_lines.append(
                    f"- v{h['version_no']} meta={h.get('meta_score','?'):.3f} "
                    f"status={h['status']} note={h.get('template_note','?')[:80]}"
                )
            hist_text = "\n".join(hist_lines) if hist_lines else "(无历史)"

            # 展示当前在位模板的关键字段
            current_summary = {
                "draft_strategy": incumbent_template.get("draft_strategy", "")[:200],
                "improve_strategy": incumbent_template.get("improve_strategy", "")[:200],
                "scoring_weights": incumbent_template.get("scoring_weights", {}),
                "dsl_templates_count": len(incumbent_template.get("dsl_exploration_templates", [])),
                "min_public_icir": incumbent_template.get("min_public_icir", 0.25),
            }

            user = (
                f"=== 当前在位模板 ===\n{current_summary}\n\n"
                f"=== 历史版本 ===\n{hist_text}\n\n"
                f"请分析当前模板的问题, 提出改进。"
                f"重点考虑: 1)探索方向是否过于单一 2)评分是否过于偏好高IC高换手 "
                f"3)指令是否足够具体 4)是否缺少多样性约束。"
            )

            logger.info("外层 LLM 调用中...")
            text = await llm.chat(provider, _SYSTEM_V2, user, 0.7)
            logger.info("外层 LLM 返回 %d 字符", len(text))
            data = llm.extract_json(text)

            changed = data.get("changed_fields", [])
            if not changed:
                logger.info("外层 LLM 判定无需改动")
                return deepcopy(incumbent_template), "外层 LLM 判定无需改动", "llm"

            raw_updates = data.get("template", {})
            logger.info("外层 LLM 拟改动 %d 个字段: %s", len(raw_updates), list(raw_updates.keys())[:10])
            new_template = clamp_template(raw_updates, incumbent_template)
            note = str(data.get("note", ""))[:300]

            # 安全检查
            err = validate_template(new_template)
            if err:
                logger.warning("外层模板安全检查拒绝: %s", err)
                return deepcopy(incumbent_template), f"安全检查拒绝: {err}", "llm_rejected"

            # 检查实质改动
            changed_keys = [k for k in new_template if new_template.get(k) != incumbent_template.get(k)]
            if changed_keys:
                logger.info("外层模板实质改动 %d 个字段: %s", len(changed_keys), changed_keys)
            else:
                logger.warning("外层模板无实质改动 (LLM返回了changed_fields但clamp后无变化)")

            return new_template, note, "llm"

        except Exception as e:
            logger.warning("外层 LLM 异常, 回退随机: %s", str(e)[:300])

    t, note = random_jitter(incumbent_template)
    return t, note, "random"


# ============================================================
# 向后兼容: 旧版 propose_spec (A组仍在用)
# ============================================================

SPEC_BOUNDS = {
    "n_drafts": (int, 2, 8),
    "improve_bias": (float, 0.3, 0.9),
    "context_top_k": (int, 2, 10),
    "llm_temperature": (float, 0.3, 1.2),
    "anti_overfit_instruction": (bool, None, None),
    "min_public_icir": (float, 0.1, 0.6),
}

_SPEC_SYSTEM = """你是自动化研究系统的元优化器。你优化的是"因子挖掘智能体"的超参数配置 (HarnessSpec), 不是因子本身。
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


def _random_jitter_old(spec: dict) -> tuple[dict, str]:
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
    """旧版接口: 返回 (new_spec, note, source)。A组兼容。"""
    if provider:
        try:
            hist = "\n".join(
                f"- v{h['version_no']} score={h['meta_score']} spec={h.get('harness_spec',h.get('template',{}))} ({h['status']})"
                for h in history[-8:]
            )
            user = f"在位配置: {incumbent_spec}\n历史:\n{hist or '(无)'}\n请提出下一个候选配置。"
            text = await llm.chat(
                provider, _SPEC_SYSTEM.format(bounds=str({k: v[1:] for k, v in SPEC_BOUNDS.items()})), user, 0.7
            )
            data = llm.extract_json(text)
            new_spec = _clamp({**incumbent_spec, **data.get("spec", {})})
            if new_spec != incumbent_spec:
                return new_spec, str(data.get("note", ""))[:300], "llm"
        except Exception as e:
            logger.warning("外层 LLM 回退随机: %s", str(e)[:200])
    spec, note = _random_jitter_old(incumbent_spec)
    return spec, note, "random"
