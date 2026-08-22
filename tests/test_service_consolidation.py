from backend.scripts.consolidate_ideal_services_to_10010 import (
    MIGRATION_PROTOCOL,
    TARGET_SERVICE,
    migrated_config,
)


def test_migration_preserves_architecture_and_adds_runtime_lineage():
    source = {
        "service_instance": "factorfactory-three-layer-ideal",
        "service_port": 10012,
        "architecture_profile": "ideal_three_layer_continuous_v1",
        "layer1_enabled": True,
        "layer2_enabled": True,
        "layer3_enabled": True,
        "engine_config": {"inner_budget_per_outer_step": 6},
    }
    result, changed = migrated_config(source, migrated_at="2026-08-22T10:00:00+00:00")

    assert changed is True
    assert result["service_instance"] == TARGET_SERVICE
    assert result["service_port"] == 10010
    assert result["layer3_enabled"] is True
    assert result["engine_config"] == source["engine_config"]
    assert result["runtime_lineage"][-1]["protocol"] == MIGRATION_PROTOCOL
    assert result["runtime_lineage"][-1]["source_service_port"] == 10012
    assert "runtime_lineage" not in source


def test_migration_is_idempotent_after_rebinding():
    source = {
        "service_instance": "factorfactory-two-layer-ideal",
        "service_port": 10011,
        "architecture_profile": "ideal_two_layer_continuous_v1",
    }
    migrated, changed = migrated_config(source, migrated_at="first")
    repeated, changed_again = migrated_config(migrated, migrated_at="second")

    assert changed is True
    assert changed_again is False
    assert repeated == migrated
