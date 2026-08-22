from app.config import service_accepts_task


def test_architecture_neutral_main_service_accepts_historical_bindings():
    config = {"service_instance": "factorfactory-full-llm-three-layer"}

    assert service_accepts_task(
        config,
        service_instance="factorfactory-10010",
        service_architecture="",
    )


def test_specialized_service_rejects_another_services_task():
    config = {"service_instance": "factorfactory-10010"}

    assert not service_accepts_task(
        config,
        service_instance="factorfactory-internal-research-demo",
        service_architecture="internal_research_hybrid",
    )


def test_specialized_service_accepts_matching_and_legacy_unbound_tasks():
    kwargs = {
        "service_instance": "factorfactory-internal-research-demo",
        "service_architecture": "internal_research_hybrid",
    }

    assert service_accepts_task(
        {"service_instance": "factorfactory-internal-research-demo"},
        **kwargs,
    )
    assert service_accepts_task({}, **kwargs)
