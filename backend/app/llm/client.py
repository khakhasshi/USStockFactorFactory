"""统一 LLM 客户端: 支持 openai (chat/completions) 与 anthropic (messages) 两种接口格式.

Provider 配置存于 settings 表 key='llm_providers':
{"providers": [{"name","format":"openai|anthropic","base_url","api_key","model"}],
 "inner_provider": "...", "outer_provider": "..."}
"""

import json

import httpx

TIMEOUT = httpx.Timeout(120.0, connect=10.0)


class LLMError(Exception):
    pass


async def chat(provider: dict, system: str, user: str, temperature: float = 0.8) -> str:
    fmt = provider.get("format", "openai")
    base = provider.get("base_url", "").rstrip("/")
    key = provider.get("api_key", "")
    model = provider.get("model", "")
    if not (base and key and model):
        raise LLMError("provider 配置不完整 (base_url/api_key/model)")

    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        if fmt == "anthropic":
            url = f"{base}/v1/messages" if not base.endswith("/v1") else f"{base}/messages"
            resp = await client.post(
                url,
                headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                json={
                    "model": model,
                    "max_tokens": 1500,
                    "temperature": temperature,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            if resp.status_code != 200:
                raise LLMError(f"anthropic {resp.status_code}: {resp.text[:300]}")
            data = resp.json()
            return "".join(b.get("text", "") for b in data.get("content", []))
        # openai 格式
        url = f"{base}/chat/completions" if "/chat/completions" not in base else base
        resp = await client.post(
            url,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        if resp.status_code != 200:
            raise LLMError(f"openai {resp.status_code}: {resp.text[:300]}")
        return resp.json()["choices"][0]["message"]["content"]


def extract_json(text: str) -> dict:
    """从 LLM 回复中提取第一个 JSON 对象."""
    start = text.find("{")
    if start < 0:
        raise LLMError(f"回复中无 JSON: {text[:200]}")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise LLMError("JSON 未闭合")
