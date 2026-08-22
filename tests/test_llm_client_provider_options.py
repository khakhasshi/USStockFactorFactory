from backend.app.llm.client import _openai_payload


def test_openai_payload_forwards_explicit_non_thinking_mode():
    payload = _openai_payload(
        {
            "model": "deepseek-v4-flash",
            "thinking": {"type": "disabled"},
        },
        "system",
        "user",
        0.4,
    )

    assert payload["model"] == "deepseek-v4-flash"
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["temperature"] == 0.4


def test_openai_payload_does_not_invent_provider_extensions():
    payload = _openai_payload(
        {"model": "ordinary-openai-compatible-model"},
        "system",
        "user",
        0.8,
    )

    assert "thinking" not in payload
