import pytest

from backend.app.research_architecture import (
    RESEARCH_ARCHITECTURE_SCHEMA,
    architecture_catalog,
    resolve_research_architecture,
)
from backend.app.search_pool import DEFAULT_SEARCH_ALGORITHMS


def test_catalog_exposes_requested_random_to_researcher_template():
    catalog = {row["key"]: row for row in architecture_catalog()}
    template = catalog["random_researcher"]

    assert template["schema"] == RESEARCH_ARCHITECTURE_SCHEMA
    assert template["recommended"] is True
    assert template["layer1_enabled"] is True
    assert template["layer2_enabled"] is True
    assert template["layer3_enabled"] is False
    assert template["search_algorithms"] == ["structured_random"]


def test_preset_is_authoritative_and_derives_runtime_protocol():
    resolved = resolve_research_architecture({
        "architecture_template": "random_researcher",
        # A stale or forged client cannot silently rewrite preset semantics.
        "layer1_enabled": False,
        "layer2_enabled": False,
        "layer3_enabled": True,
        "search_algorithms": ["q_learning"],
        "proposal_mode": "random",
        "memory_mode": "cold",
    })

    qlib = resolved.pop("qlib_integration")
    assert qlib["enabled"] is False
    assert qlib["effective"] is False
    assert resolved == {
        "architecture_schema": RESEARCH_ARCHITECTURE_SCHEMA,
        "architecture_template": "random_researcher",
        "layer1_enabled": True,
        "layer2_enabled": True,
        "layer3_enabled": False,
        "search_algorithms": ["structured_random"],
        "proposal_mode": "llm",
        "memory_mode": "cold",
    }


def test_no_llm_and_full_three_layer_presets_are_explicit():
    no_llm = resolve_research_architecture({
        "architecture_template": "random_only",
        "memory_mode": "adaptive",
    })
    full = resolve_research_architecture({"architecture_template": "full_three_layer"})

    assert no_llm["proposal_mode"] == "search_pool"
    assert no_llm["memory_mode"] == "cold"
    assert no_llm["layer2_enabled"] is False
    assert full["search_algorithms"] == [
        *DEFAULT_SEARCH_ALGORITHMS,
        "qlib_alpha158_prior",
        "qlib_joint_residual_distill",
    ]
    assert full["qlib_integration"]["effective"] is True
    assert [full[f"layer{i}_enabled"] for i in (1, 2, 3)] == [True, True, True]


def test_full_llm_three_layer_has_three_llm_decision_layers():
    full = resolve_research_architecture({
        "architecture_template": "full_llm_three_layer"
    })
    assert full["proposal_mode"] == "llm"
    assert full["full_llm_architecture"] is True
    assert full["scientific_governor_enabled"] is True
    assert [full[f"layer{i}_enabled"] for i in (1, 2, 3)] == [True, True, True]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (
            {
                "architecture_template": "custom",
                "layer1_enabled": True,
                "layer2_enabled": False,
                "layer3_enabled": False,
                "search_algorithms": [],
            },
            "至少选择一种搜索算法",
        ),
        (
            {
                "architecture_template": "custom",
                "layer1_enabled": False,
                "layer2_enabled": False,
                "layer3_enabled": True,
            },
            "第三层 Governor",
        ),
        (
            {
                "architecture_template": "custom",
                "layer1_enabled": True,
                "layer2_enabled": False,
                "layer3_enabled": False,
                "search_algorithms": ["unknown"],
            },
            "未知第一层搜索算法",
        ),
        (
            {
                "architecture_template": "custom",
                "layer1_enabled": [],
                "layer2_enabled": False,
                "layer3_enabled": False,
            },
            "layer1_enabled 必须为布尔值",
        ),
    ],
)
def test_custom_architecture_rejects_ambiguous_or_invalid_plans(config, message):
    with pytest.raises(ValueError, match=message):
        resolve_research_architecture(config)


def test_legacy_direct_random_and_llm_clients_remain_compatible():
    random_task = resolve_research_architecture({"proposal_mode": "random"})
    llm_task = resolve_research_architecture({"proposal_mode": "llm"})

    assert random_task["architecture_template"] == "legacy"
    assert random_task["proposal_mode"] == "random"
    assert not any(random_task[f"layer{i}_enabled"] for i in (1, 2, 3))
    assert llm_task["proposal_mode"] == "llm"
    assert llm_task["layer2_enabled"] is True
    assert llm_task["layer3_enabled"] is True
