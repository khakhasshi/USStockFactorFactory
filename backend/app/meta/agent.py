"""外层 Meta-Researcher v2: 可改写 MinerTemplate (system prompt/搜索策略/上下文构造/评分偏好/DSL模板).

设计原则 (来自 AIDE² 论文经验):
- 外层优化对象 = Miner 的"源代码级"文本 (prompt + 策略 + 模板), 不是 6 个数字参数
- 安全边界: MetaValidator 确保外层不能触碰评估器/数据层/隔离边界
- 无 LLM 时回退随机 (系统可无钥运行)
"""

import json
import hashlib
import logging
import random
from copy import deepcopy

from ..config import DEFAULT_MINER_TEMPLATE, EVALUATION_PROTOCOL_VERSION
from ..feedback import (
    DEFAULT_CONTEXT_POLICY,
    ensure_training_safe,
    normalise_scoring_weights,
    resolve_context_policy,
)
from ..llm import client as llm
from ..observability import redact_text, redact_value

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
            base["scoring_weights"] = normalise_scoring_weights(base)

    if "context_policy" in raw and isinstance(raw["context_policy"], dict):
        candidate = {
            **DEFAULT_CONTEXT_POLICY,
            **raw["context_policy"],
        }
        base["context_policy"] = resolve_context_policy(
            {"context_policy": candidate}
        )

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
            t["scoring_weights"] = normalise_scoring_weights(t)
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
4. **context_strategy**: 如何解释和使用评价反馈
5. **context_policy**: 高分/近失/失败/错误样本的数量边界
6. **diversity_instruction**: 多样性约束措辞
7. **scoring_weights**: 只控制上下文示例优先级, 绝不改变权威评价分
8. **dsl_exploration_templates**: 会真实进入内层提示词的 DSL 结构样例

安全边界 (你绝不能触碰):
- 权威 V4.2 评估器与数据层只读；最终封存评价不可作为训练反馈
- 不得在模板中引用特定年份/era/数据层名称
- 不得注入 Python 代码或文件系统操作
- 所有改动必须引用历史报告中的证据；没有证据时应声明为探索性假设

你会看到:
- 当前在位模板
- 同一评价协议下的历史模板版本
- 跨任务/多种子的通过率、连续学习分、硬门槛分、方向分布、组件均值、失败原因与重复率
- 上轮提案假设、结果反思和下一步建议

分数与方向语义:
- 每个新候选在训练安全层同时测试 +1/-1，双向搜索已计入试验预算，选中方向随后冻结
- seed_score_mean/score_mean 是连续学习分，只用于失败归因和模板搜索
- gate_score_mean/gate_score_best 与 pass_rate 是独立硬门槛证据
- 学习分变高不等于通过准入，禁止据此宣称因子可实盘

只回复 JSON:
{{
  "reflection": {{
    "diagnosis": ["基于证据的问题"],
    "lessons_applied": ["本轮吸取的经验"],
    "evidence_used": ["引用的版本/任务/指标"],
    "hypothesis": "本次改动为何应改善评价"
  }},
  "changed_fields": ["字段名列表"],
  "template": {{要修改的字段}},
  "note": "改动逻辑、预期改善组件和停止条件"
}}
如果证据不足或当前模板已足够好, 返回 changed_fields=[] 并明确说明。"""


def _compact_feedback_report(report: dict | None) -> dict:
    report = dict(report or {})
    compact = {
        "score_semantics": report.get(
            "score_semantics",
            "continuous_failure_margin_v4.2",
        ),
        "attempts": report.get("attempts", 0),
        "errors": report.get("errors", 0),
        "pass_rate": report.get("pass_rate", 0.0),
        "seed_score_mean": report.get(
            "seed_score_mean",
            report.get("score_mean", 0.0),
        ),
        "seed_score_std": report.get(
            "seed_score_std",
            report.get("score_std", 0.0),
        ),
        "gate_score_mean": report.get("gate_score_mean", 0.0),
        "gate_score_best": report.get("gate_score_best", 0.0),
        "duplicate_rate": report.get("duplicate_rate", 0.0),
        "source_counts": report.get("source_counts", {}),
        "direction_counts": report.get("direction_counts", {}),
        "component_means": report.get("component_means", {}),
        "metric_means": {
            key: value
            for key, value in (report.get("metric_means") or {}).items()
            if key in {
                "icir",
                "portfolio_sharpe",
                "return_hac_t",
                "sharpe_lcb",
                "ann_return_lcb",
                "era_consistency",
                "profitable_era_rate",
                "monotonicity",
                "daily_turnover",
                "worst_stress_sharpe",
                "cost_cushion_multiple",
            }
        },
        "failure_reason_counts": dict(
            list((report.get("failure_reason_counts") or {}).items())[:8]
        ),
        "improvement_target_counts": dict(
            list(
                (report.get("improvement_target_counts") or {}).items()
            )[:6]
        ),
        "seeds": [
            {
                "seed": row.get("seed"),
                "meta_score": row.get("meta_score"),
                "task_best_scores": row.get("task_best_scores", {}),
            }
            for row in (report.get("seeds") or [])[:8]
            if isinstance(row, dict)
        ],
        "by_task": {
            task: {
                "attempts": row.get("attempts", 0),
                "pass_rate": row.get("pass_rate", 0.0),
                "score_mean": row.get("score_mean", 0.0),
                "score_best": row.get("score_best", 0.0),
                "gate_score_mean": row.get("gate_score_mean", 0.0),
                "gate_score_best": row.get("gate_score_best", 0.0),
                "errors": row.get("errors", 0),
                "duplicate_rate": row.get("duplicate_rate", 0.0),
                "direction_counts": row.get("direction_counts", {}),
                "failure_reason_counts": dict(
                    list(
                        (row.get("failure_reason_counts") or {}).items()
                    )[:4]
                ),
            }
            for task, row in (report.get("by_task") or {}).items()
        },
        "feedback_fingerprint": report.get("feedback_fingerprint"),
    }
    ensure_training_safe(compact)
    return compact


def _compact_template(template: dict | None) -> dict:
    template = dict(template or {})
    return {
        "system_prompt": str(template.get("system_prompt") or "")[:500],
        "anti_overfit_instruction": str(
            template.get("anti_overfit_instruction") or ""
        )[:300],
        "draft_strategy": str(template.get("draft_strategy") or "")[:500],
        "improve_strategy": str(
            template.get("improve_strategy") or ""
        )[:500],
        "context_strategy": str(
            template.get("context_strategy") or ""
        )[:400],
        "context_policy": resolve_context_policy(template),
        "diversity_instruction": str(
            template.get("diversity_instruction") or ""
        )[:300],
        "scoring_weights": normalise_scoring_weights(template),
        "dsl_exploration_templates": [
            str(item)[:200]
            for item in (
                template.get("dsl_exploration_templates") or []
            )[:8]
        ],
    }


def _as_text_list(value: object, limit: int) -> list[str]:
    if value is None:
        return []
    rows = value if isinstance(value, list) else [value]
    return [
        str(item)[:limit]
        for item in rows[:10]
        if str(item).strip()
    ]


def _clean_reflection(value: dict | None) -> dict:
    value = dict(value) if isinstance(value, dict) else {}
    result = {
        "diagnosis": _as_text_list(value.get("diagnosis"), 400)[:8],
        "lessons_applied": _as_text_list(
            value.get("lessons_applied"),
            400,
        )[:8],
        "evidence_used": _as_text_list(
            value.get("evidence_used"),
            300,
        )[:10],
        "hypothesis": str(value.get("hypothesis") or "")[:1000],
    }
    ensure_training_safe(result)
    return result


def _history_context(
    history: list[dict],
    protocol: str = EVALUATION_PROTOCOL_VERSION,
) -> tuple[str, str]:
    """Render only comparable history and return its context fingerprint."""
    comparable = [
        row
        for row in history
        if row.get("evaluation_protocol") == protocol
    ][-10:]
    payload = []
    for row in comparable:
        payload.append({
            "version": row.get("version_no"),
            "status": row.get("status"),
            "meta_score": row.get("meta_score"),
            "note": str(row.get("template_note") or "")[:300],
            "feedback": _compact_feedback_report(
                row.get("feedback_summary")
            ),
            "reflection": redact_value(row.get("reflection") or {}),
            "template_controls": _compact_template(row.get("template")),
            "context_fingerprint": row.get("context_fingerprint"),
        })
    ensure_training_safe(payload)
    text = (
        json.dumps(payload, ensure_ascii=False, indent=2)
        if payload
        else "(当前协议暂无已完成历史)"
    )
    fingerprint = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]
    return text, fingerprint


async def propose_template(
    incumbent_template: dict,
    history: list[dict],
    provider: dict | None,
    market: str = "us",
    portfolio_mode: str = "long_short",
    direction: int = 1,
    direction_policy: str = "both_train_select",
    trace_context: dict | None = None,
) -> tuple[dict, str, str, dict]:
    """返回 (new_template, note, source, proposal_reflection).

    Only rows tagged with the current protocol may become model feedback.
    """
    hist_text, history_fingerprint = _history_context(history)
    fallback_reason = "provider_not_configured"
    text: str | None = None
    if provider:
        try:
            current_summary = {
                **_compact_template(incumbent_template),
                "min_public_icir": incumbent_template.get("min_public_icir", 0.25),
            }

            user = (
                f"=== 研究任务硬约束 ===\n市场: {market}\n持仓模式: {portfolio_mode}\n"
                f"方向评价策略: {direction_policy}；"
                f"同分优先方向: {direction:+d}\n"
                "每个候选必须在训练安全层同时评价正向与反向，"
                "由系统选中并冻结方向；不得要求用隔离层翻号。\n"
                f"{'只能做多，评价只奖励正向收益和多头稳定性。' if portfolio_mode == 'long_only' else '允许多空，评价可同时使用多头和空头收益。'}\n\n"
                f"=== 当前在位模板 ===\n{current_summary}\n\n"
                f"=== 同协议历史与评价反馈 ===\n{hist_text}\n\n"
                "请先判断上轮假设得到支持、被证伪还是证据不足，再提出下一项最小可归因改动。"
                "重点检查任务间退化、种子方差、通过率、失败原因、重复率、随机回退率和最弱评价组件。"
            )

            logger.info("外层 LLM 调用中...")
            text = await llm.chat(
                provider,
                _SYSTEM_V2,
                user,
                0.7,
                trace={
                    **(trace_context or {}),
                    "role": "outer",
                    "phase": "template_proposal",
                    "feedback_fingerprint": history_fingerprint,
                },
            )
            logger.info("外层 LLM 返回 %d 字符", len(text))
            data = llm.extract_json(text)
            reflection = _clean_reflection(data.get("reflection"))
            reflection["history_context_fingerprint"] = history_fingerprint

            changed = data.get("changed_fields", [])
            if not changed:
                logger.info("外层 LLM 判定无需改动")
                await llm.mark_validation(text, accepted=True)
                return (
                    deepcopy(incumbent_template),
                    str(data.get("note") or "外层 LLM 判定无需改动")[:500],
                    "llm",
                    reflection,
                )

            if (
                not reflection["hypothesis"]
                or not reflection["diagnosis"]
            ):
                raise llm.LLMError(
                    "外层改动缺少可审计的 diagnosis/hypothesis"
                )
            raw_updates = data.get("template", {})
            logger.info("外层 LLM 拟改动 %d 个字段: %s", len(raw_updates), list(raw_updates.keys())[:10])
            new_template = clamp_template(raw_updates, incumbent_template)
            note = str(data.get("note", ""))[:500]

            # 安全检查
            err = validate_template(new_template)
            if err:
                logger.warning("外层模板安全检查拒绝: %s", err)
                reflection["diagnosis"].append(f"模板安全检查拒绝: {err}")
                await llm.mark_validation(
                    text,
                    accepted=False,
                    error=err,
                )
                return (
                    deepcopy(incumbent_template),
                    f"安全检查拒绝: {err}",
                    "llm_rejected",
                    reflection,
                )

            # 检查实质改动
            changed_keys = [k for k in new_template if new_template.get(k) != incumbent_template.get(k)]
            if changed_keys:
                logger.info("外层模板实质改动 %d 个字段: %s", len(changed_keys), changed_keys)
            else:
                logger.warning("外层模板无实质改动 (LLM返回了changed_fields但clamp后无变化)")

            reflection["changed_fields"] = changed_keys
            await llm.mark_validation(text, accepted=True)
            return new_template, note, "llm", reflection

        except Exception as exc:
            fallback_reason = redact_text(exc, 300)
            await llm.mark_validation(
                text,
                accepted=False,
                error=fallback_reason,
            )
            logger.warning(
                "外层 LLM 异常, 回退随机: %s",
                fallback_reason,
            )

    t, note = random_jitter(incumbent_template)
    return (
        t,
        note,
        "random",
        {
            "diagnosis": ["外层 LLM 不可用或输出无效"],
            "lessons_applied": [],
            "evidence_used": [history_fingerprint] if history_fingerprint else [],
            "hypothesis": "随机微调仅用于维持搜索连续性，不视为模型反思。",
            "fallback_reason": fallback_reason,
            "history_context_fingerprint": history_fingerprint,
        },
    )


_REFLECTION_SYSTEM = """你是因子研究外层审稿人。
你只能依据同一评价协议下的训练安全聚合报告，复盘模板改动是否有效。
不得臆测未提供的数据，不得要求查看最终封存评价，不得改写权威评价器。
只回复 JSON:
{
  "hypothesis_result": "supported|refuted|inconclusive",
  "lessons": [{"observation":"证据","interpretation":"解释","action":"下一轮动作"}],
  "avoid_patterns": ["应避免的重复失败"],
  "next_experiment": "一个最小、可归因的下一步实验",
  "stop_condition": "何时停止沿此方向搜索"
}"""


def _deterministic_outcome_reflection(
    comparison: dict,
    *,
    accepted: bool,
) -> dict:
    deltas = comparison.get("deltas") or {}
    failures = comparison.get("candidate_failures") or {}
    result = "supported" if accepted else "refuted"
    if abs(float(deltas.get("seed_score_mean") or 0.0)) < 1e-9:
        result = "inconclusive"
    top_failure = next(iter(failures), "没有可归因的失败样本")
    return {
        "hypothesis_result": result,
        "lessons": [{
            "observation": (
                f"seed_score_delta={deltas.get('seed_score_mean', 0)}, "
                f"gate_score_delta={deltas.get('gate_score_mean', 0)}, "
                f"pass_rate_delta={deltas.get('pass_rate', 0)}, "
                f"duplicate_delta={deltas.get('duplicate_rate', 0)}"
            ),
            "interpretation": (
                "候选模板通过统计门"
                if accepted
                else f"候选模板未通过统计门；主要失败为 {top_failure}"
            ),
            "action": (
                "保留改动并只验证一个相邻假设"
                if accepted
                else "撤销本次改动，针对主要失败做单变量实验"
            ),
        }],
        "avoid_patterns": list(failures)[:5],
        "next_experiment": (
            "围绕最弱评价组件做一个单字段模板改动，并冻结其他字段。"
        ),
        "stop_condition": "连续两轮同一失败原因占主导且组件无改善时停止该方向。",
        "source": "deterministic",
    }


async def reflect_on_outcome(
    *,
    proposal_reflection: dict,
    candidate_report: dict,
    incumbent_report: dict,
    comparison: dict,
    accepted: bool,
    p_value: float,
    provider: dict | None,
    trace_context: dict | None = None,
) -> tuple[dict, str]:
    """Persistable post-decision reflection that becomes next-step context."""
    safe_payload = {
        "proposal_reflection": proposal_reflection,
        "candidate_report": _compact_feedback_report(candidate_report),
        "incumbent_report": _compact_feedback_report(incumbent_report),
        "comparison": comparison,
        "decision": {
            "accepted": bool(accepted),
            "p_value": round(float(p_value), 6),
        },
    }
    ensure_training_safe(safe_payload)
    text: str | None = None
    if provider:
        try:
            user = json.dumps(safe_payload, ensure_ascii=False, indent=2)
            text = await llm.chat(
                provider,
                _REFLECTION_SYSTEM,
                user,
                0.2,
                trace={
                    **(trace_context or {}),
                    "role": "outer_reflection",
                    "phase": "post_decision_reflection",
                    "feedback_fingerprint": comparison.get(
                        "comparison_fingerprint",
                        "",
                    ),
                },
            )
            data = llm.extract_json(text)
            lessons = []
            raw_lessons = data.get("lessons") or []
            if not isinstance(raw_lessons, list):
                raw_lessons = [raw_lessons]
            for row in raw_lessons[:8]:
                if not isinstance(row, dict):
                    continue
                lessons.append({
                    "observation": str(row.get("observation") or "")[:500],
                    "interpretation": str(
                        row.get("interpretation") or ""
                    )[:500],
                    "action": str(row.get("action") or "")[:500],
                })
            hypothesis_result = str(
                data.get("hypothesis_result") or "inconclusive"
            ).lower()
            if hypothesis_result not in {
                "supported",
                "refuted",
                "inconclusive",
            }:
                hypothesis_result = "inconclusive"
            next_experiment = str(
                data.get("next_experiment") or ""
            )[:1000]
            stop_condition = str(
                data.get("stop_condition") or ""
            )[:800]
            if not lessons or not next_experiment or not stop_condition:
                raise llm.LLMError(
                    "外层结果反思缺少 lessons/next_experiment/stop_condition"
                )
            result = {
                "hypothesis_result": hypothesis_result,
                "lessons": lessons,
                "avoid_patterns": _as_text_list(
                    data.get("avoid_patterns"),
                    300,
                )[:8],
                "next_experiment": next_experiment,
                "stop_condition": stop_condition,
                "source": "llm",
                "comparison_fingerprint": comparison.get(
                    "comparison_fingerprint",
                ),
            }
            ensure_training_safe(result)
            await llm.mark_validation(text, accepted=True)
            return result, "llm"
        except Exception as exc:
            await llm.mark_validation(
                text,
                accepted=False,
                error=exc,
            )
            logger.warning("外层结果反思失败，使用确定性摘要: %s", str(exc)[:300])
    return _deterministic_outcome_reflection(
        comparison,
        accepted=accepted,
    ), "deterministic"


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
    text: str | None = None
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
                await llm.mark_validation(text, accepted=True)
                return new_spec, str(data.get("note", ""))[:300], "llm"
            await llm.mark_validation(
                text,
                accepted=False,
                error="旧版外层返回与在位配置相同",
            )
        except Exception as exc:
            await llm.mark_validation(
                text,
                accepted=False,
                error=exc,
            )
            logger.warning("外层 LLM 回退随机: %s", str(exc)[:200])
    spec, note = _random_jitter_old(incumbent_spec)
    return spec, note, "random"
