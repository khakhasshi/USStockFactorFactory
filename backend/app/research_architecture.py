"""Validated architecture templates for manually-created research tasks."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .search_pool import DEFAULT_SEARCH_ALGORITHMS, SUPPORTED_SEARCH_ALGORITHMS
from .qlib_native import resolve_qlib_task_integration


RESEARCH_ARCHITECTURE_SCHEMA = "factorfactory.research-architecture/v1"

_TEMPLATES: dict[str, dict[str, Any]] = {
    "random_only": {
        "label": "第一层结构化随机（无 LLM）",
        "description": "机制约束随机搜索；不调用 Researcher 或 Governor。",
        "layer1_enabled": True,
        "layer2_enabled": False,
        "layer3_enabled": False,
        "search_algorithms": ["structured_random"],
        "default_memory_mode": "cold",
    },
    "algorithm_pool_only": {
        "label": "第一层算法池（无 LLM）",
        "description": "结构、残差、ML 蒸馏、局部优化与高风险探索按科学配额运行；不调用 LLM。",
        "layer1_enabled": True,
        "layer2_enabled": False,
        "layer3_enabled": False,
        "search_algorithms": [*DEFAULT_SEARCH_ALGORITHMS, "qlib_alpha158_prior"],
        "qlib_integration": {"enabled": True},
        "default_memory_mode": "cold",
    },
    "random_researcher": {
        "label": "第一层随机 → 第二层 Researcher LLM",
        "description": "先生成结构化随机种子，再由 LLM 审查并做一次可归因改进。",
        "layer1_enabled": True,
        "layer2_enabled": True,
        "layer3_enabled": False,
        "search_algorithms": ["structured_random"],
        "default_memory_mode": "adaptive",
        "recommended": True,
    },
    "algorithm_pool_researcher": {
        "label": "第一层算法池 → 第二层 Researcher LLM",
        "description": "完整算法池提供种子；Researcher LLM 审查改进；无 Governor。",
        "layer1_enabled": True,
        "layer2_enabled": True,
        "layer3_enabled": False,
        "search_algorithms": [*DEFAULT_SEARCH_ALGORITHMS, "qlib_alpha158_prior"],
        "qlib_integration": {"enabled": True},
        "default_memory_mode": "adaptive",
    },
    "full_three_layer": {
        "label": "完整三层：算法池 → Researcher → Governor",
        "description": "完整第一层、连续记忆 Researcher 与低频 Governor。",
        "layer1_enabled": True,
        "layer2_enabled": True,
        "layer3_enabled": True,
        "search_algorithms": [*DEFAULT_SEARCH_ALGORITHMS, "qlib_alpha158_prior"],
        "qlib_integration": {"enabled": True},
        "default_memory_mode": "adaptive",
    },
    "full_llm_three_layer": {
        "label": "全 LLM 三层：机制科学家 → 研究主任 → 科学总督",
        "description": (
            "确定性算法池仅作为发现底座；三层研究决策均由 LLM 完成，"
            "并保留分层调用、决策和连续记忆审计。"
        ),
        "layer1_enabled": True,
        "layer2_enabled": True,
        "layer3_enabled": True,
        "full_llm_architecture": True,
        "scientific_governor_enabled": True,
        "search_algorithms": [*DEFAULT_SEARCH_ALGORITHMS, "qlib_alpha158_prior"],
        "qlib_integration": {"enabled": True},
        "default_memory_mode": "adaptive",
    },
    "direct_researcher": {
        "label": "直接 Researcher LLM（无第一层）",
        "description": "兼容直接 LLM 候选生成，用于与第一层种子架构对照。",
        "layer1_enabled": False,
        "layer2_enabled": True,
        "layer3_enabled": False,
        "search_algorithms": [],
        "default_memory_mode": "adaptive",
    },
}


def architecture_catalog() -> list[dict[str, Any]]:
    return [
        {
            "key": key,
            **deepcopy(value),
            "schema": RESEARCH_ARCHITECTURE_SCHEMA,
        }
        for key, value in _TEMPLATES.items()
    ]


def _as_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{field} 必须为布尔值")


def resolve_research_architecture(config: dict[str, Any] | None) -> dict[str, Any]:
    """Resolve a preset/custom/legacy task into one internally consistent plan."""
    raw = dict(config or {})
    legacy_direct_random = False
    requested = str(raw.get("architecture_template") or "").strip()
    explicit_layers = any(
        key in raw
        for key in (
            "layer1_enabled",
            "layer2_enabled",
            "layer3_enabled",
            "search_algorithms",
        )
    )

    if requested and requested not in {*_TEMPLATES, "custom", "legacy"}:
        raise ValueError(f"不支持的研究架构模板: {requested}")

    if requested in _TEMPLATES:
        template = deepcopy(_TEMPLATES[requested])
        layer1 = bool(template["layer1_enabled"])
        layer2 = bool(template["layer2_enabled"])
        layer3 = bool(template["layer3_enabled"])
        algorithms = list(template["search_algorithms"])
        memory_mode = str(
            raw.get("memory_mode") or template["default_memory_mode"]
        ).strip().lower()
        template_key = requested
    elif requested == "custom" or explicit_layers:
        layer1 = _as_bool(
            raw.get("layer1_enabled", True), field="layer1_enabled"
        )
        layer2 = _as_bool(
            raw.get("layer2_enabled", False), field="layer2_enabled"
        )
        layer3 = _as_bool(
            raw.get("layer3_enabled", False), field="layer3_enabled"
        )
        configured = raw.get("search_algorithms") or []
        if not isinstance(configured, (list, tuple)):
            raise ValueError("search_algorithms 必须为算法名称列表")
        algorithms = [
            str(value).strip() for value in configured if str(value).strip()
        ]
        memory_mode = str(raw.get("memory_mode") or "adaptive").strip().lower()
        template_key = "custom"
    else:
        # Existing API clients and historical tasks retain their old meaning.
        proposal_mode = str(raw.get("proposal_mode") or "llm").strip().lower()
        if proposal_mode not in {"llm", "random", "search_pool"}:
            raise ValueError("proposal_mode 必须为 llm、random 或 search_pool")
        layer1 = bool(raw.get("layer1_enabled", False))
        layer2 = bool(raw.get("layer2_enabled", proposal_mode == "llm"))
        layer3 = bool(raw.get("layer3_enabled", proposal_mode == "llm"))
        configured = raw.get("search_algorithms") or (
            DEFAULT_SEARCH_ALGORITHMS if layer1 else []
        )
        algorithms = list(configured)
        memory_mode = str(raw.get("memory_mode") or "adaptive").strip().lower()
        template_key = "legacy"
        legacy_direct_random = proposal_mode == "random" and not any(
            (layer1, layer2, layer3)
        )

    if memory_mode not in {"adaptive", "cold"}:
        raise ValueError("memory_mode 必须为 adaptive 或 cold")
    if layer3 and not layer2:
        raise ValueError("第三层 Governor 需要先启用第二层 Researcher LLM")
    if not any((layer1, layer2, layer3)) and not legacy_direct_random:
        raise ValueError("研究架构至少需要启用第一层搜索或第二层 LLM")

    allowed = set(SUPPORTED_SEARCH_ALGORITHMS)
    unknown = sorted(set(algorithms) - allowed)
    if unknown:
        raise ValueError(f"未知第一层搜索算法: {unknown}")
    if layer1 and not algorithms:
        raise ValueError("启用第一层时至少选择一种搜索算法")
    if not layer1:
        algorithms = []
    if not layer2:
        memory_mode = "cold"

    qlib_integration = resolve_qlib_task_integration(
        raw.get("qlib_integration") or (
            template.get("qlib_integration")
            if requested in _TEMPLATES
            else None
        ),
        layer1_enabled=layer1,
        alpha158_algorithm_selected="qlib_alpha158_prior" in algorithms,
        joint_algorithm_selected="qlib_joint_residual_distill" in algorithms,
    )
    if qlib_integration["alpha158_prior_enabled"] and "qlib_alpha158_prior" not in algorithms:
        algorithms.append("qlib_alpha158_prior")
    if not qlib_integration["alpha158_prior_enabled"]:
        algorithms = [name for name in algorithms if name != "qlib_alpha158_prior"]
    if qlib_integration["joint_model_enabled"] and "qlib_joint_residual_distill" not in algorithms:
        algorithms.append("qlib_joint_residual_distill")
    if not qlib_integration["joint_model_enabled"]:
        algorithms = [
            name for name in algorithms
            if name != "qlib_joint_residual_distill"
        ]

    proposal_mode = "llm" if layer2 else "search_pool" if layer1 else "random"
    resolved = {
        "architecture_schema": RESEARCH_ARCHITECTURE_SCHEMA,
        "architecture_template": template_key,
        "layer1_enabled": layer1,
        "layer2_enabled": layer2,
        "layer3_enabled": layer3,
        "search_algorithms": algorithms,
        "proposal_mode": proposal_mode,
        "memory_mode": memory_mode,
        "qlib_integration": qlib_integration,
    }
    if requested in _TEMPLATES and _TEMPLATES[requested].get(
        "full_llm_architecture", False
    ):
        resolved["full_llm_architecture"] = True
        resolved["scientific_governor_enabled"] = True
    return resolved
