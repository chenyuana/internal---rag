from __future__ import annotations

import json

import httpx
from pydantic import SecretStr

from app.core.config import ModelEndpointSettings
from app.services.model_client import OpenAICompatibleModelClient, flatten_json_schema


def test_flatten_json_schema_expands_refs_and_lifts_required() -> None:
    schema = {
        "$defs": {
            "Item": {
                "properties": {
                    "id": {"type": "string", "title": "Id"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id"],
                "title": "Item",
                "type": "object",
            }
        },
        "properties": {
            "name": {"type": "string", "title": "Name"},
            "items": {"items": {"$ref": "#/$defs/Item"}, "type": "array"},
        },
        "required": ["name"],
        "title": "Root",
        "type": "object",
    }

    flattened = flatten_json_schema(schema)

    dumped = json.dumps(flattened)
    assert "$ref" not in dumped
    assert "$defs" not in dumped
    assert '"title"' not in dumped
    assert flattened["required"] == ["name", "items"]
    assert flattened["additionalProperties"] is False
    items_schema = flattened["properties"]["items"]["items"]
    assert items_schema["required"] == ["id", "tags"]
    assert items_schema["additionalProperties"] is False
    assert set(items_schema["properties"]) == {"id", "tags"}


async def test_model_probe_checks_configured_model_name() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "expected-model"}]})

    settings = ModelEndpointSettings(
        enabled=True,
        required=True,
        base_url="http://model.local/v1",
        api_key=SecretStr("secret"),
        model_name="expected-model",
        timeout_seconds=2,
    )
    client = OpenAICompatibleModelClient(
        settings,
        transport=httpx.MockTransport(handler),
    )
    try:
        result = await client.probe()
    finally:
        await client.close()

    assert result.ready is True


async def test_model_probe_rejects_wrong_model() -> None:
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, json={"data": [{"id": "other-model"}]})
    )
    settings = ModelEndpointSettings(
        enabled=True,
        required=True,
        base_url="http://model.local/v1",
        model_name="expected-model",
    )
    client = OpenAICompatibleModelClient(settings, transport=transport)
    try:
        result = await client.probe()
    finally:
        await client.close()

    assert result.ready is False
    assert result.detail == "configured_model_not_reported"


async def test_model_catalog_returns_unique_model_ids() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(
            200,
            json={"data": [{"id": "model-a"}, {"id": "model-b"}, {"id": "model-a"}]},
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=False,
        base_url="http://model.local/v1",
        model_name="model-a",
    )
    client = OpenAICompatibleModelClient(settings, transport=httpx.MockTransport(handler))
    try:
        models = await client.list_models()
    finally:
        await client.close()

    assert models == ["model-a", "model-b"]


async def test_model_chat_completion_requests_strict_json_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "expected-model"
        assert payload["temperature"] == 0
        assert payload["seed"] == 42
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["response_format"]["json_schema"]["strict"] is True
        assert payload["reasoning_effort"] == "none"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"answerability":"UNANSWERABLE"}',
                        }
                    }
                ]
            },
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=True,
        base_url="http://model.local/v1",
        model_name="expected-model",
        temperature=0,
        seed=42,
        max_output_tokens=2048,
    )
    client = OpenAICompatibleModelClient(
        settings,
        transport=httpx.MockTransport(handler),
    )
    try:
        content = await client.chat_completion(
            messages=[{"role": "user", "content": "question"}],
            response_schema={"type": "object"},
        )
    finally:
        await client.close()

    assert content == '{"answerability":"UNANSWERABLE"}'


async def test_model_chat_completion_omits_seed_when_unset() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert "seed" not in payload
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"answerability":"UNANSWERABLE"}',
                        }
                    }
                ]
            },
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=True,
        base_url="http://model.local/v1",
        model_name="expected-model",
    )
    client = OpenAICompatibleModelClient(
        settings,
        transport=httpx.MockTransport(handler),
    )
    try:
        content = await client.chat_completion(
            messages=[{"role": "user", "content": "question"}],
            response_schema={"type": "object"},
        )
    finally:
        await client.close()

    assert content == '{"answerability":"UNANSWERABLE"}'


async def test_model_chat_completion_retries_json_object_for_compatible_api() -> None:
    attempts: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        attempts.append(payload)
        if payload.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "unsupported response format"})
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answerability":"UNANSWERABLE"}'}}]},
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=False,
        base_url="https://api.example.com/v1",
        model_name="compatible-model",
        seed=42,
    )
    client = OpenAICompatibleModelClient(settings, transport=httpx.MockTransport(handler))
    try:
        content = await client.chat_completion(
            messages=[{"role": "user", "content": "question"}],
            response_schema={"type": "object"},
        )
    finally:
        await client.close()

    assert content == '{"answerability":"UNANSWERABLE"}'
    assert len(attempts) == 3
    assert attempts[1]["response_format"]["type"] == "json_schema"
    assert "reasoning_effort" not in attempts[1]
    assert "seed" not in attempts[1]
    assert attempts[2]["response_format"] == {"type": "json_object"}
    schema_instruction = attempts[2]["messages"][0]["content"]
    assert "JSON Schema" in schema_instruction
    assert 'JSON Schema: {"type":"object"}' in schema_instruction


async def test_model_chat_completion_keeps_schema_when_only_optional_parameters_fail() -> None:
    attempts: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        attempts.append(payload)
        if "reasoning_effort" in payload or "seed" in payload:
            return httpx.Response(400, json={"error": "unsupported parameter"})
        assert payload["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answerability":"UNANSWERABLE"}'}}]},
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=False,
        base_url="https://api.example.com/v1",
        model_name="compatible-model",
        seed=42,
    )
    client = OpenAICompatibleModelClient(settings, transport=httpx.MockTransport(handler))
    try:
        content = await client.chat_completion(
            messages=[{"role": "user", "content": "question"}],
            response_schema={"type": "object"},
        )
    finally:
        await client.close()

    assert content == '{"answerability":"UNANSWERABLE"}'
    assert len(attempts) == 2
    assert attempts[1]["response_format"]["type"] == "json_schema"


async def test_json_object_fallback_includes_required_structured_answer_fields() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "unsupported response format"})
        captured.update(payload)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    schema = {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
            "claims": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["answer"],
    }
    settings = ModelEndpointSettings(
        enabled=True,
        required=False,
        base_url="https://api.example.com/v1",
        model_name="compatible-model",
    )
    client = OpenAICompatibleModelClient(settings, transport=httpx.MockTransport(handler))
    try:
        await client.chat_completion(
            messages=[{"role": "user", "content": "question"}],
            response_schema=schema,
        )
    finally:
        await client.close()

    instruction = captured["messages"][0]["content"]
    assert "claims" in instruction
    assert "citation_ids" in instruction
    assert '"required":["answer","claims"]' in instruction


async def test_model_chat_completion_retries_one_empty_ollama_response() -> None:
    attempts: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        attempts.append(payload)
        if len(attempts) == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "", "reasoning_content": "thinking"},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=False,
        base_url="http://localhost:11434/v1",
        model_name="qwen",
        seed=42,
    )
    client = OpenAICompatibleModelClient(settings, transport=httpx.MockTransport(handler))
    try:
        content = await client.chat_completion(
            messages=[{"role": "user", "content": "question"}],
            response_schema={"type": "object"},
        )
    finally:
        await client.close()

    assert content == '{"ok":true}'
    assert len(attempts) == 2
    assert attempts[1]["think"] is False
    assert "seed" not in attempts[1]
    assert "without analysis" in attempts[1]["messages"][-1]["content"]


async def test_empty_compatible_api_response_reuses_json_object_fallback() -> None:
    attempts: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        attempts.append(payload)
        if payload.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "unsupported response format"})
        if "think" in payload:
            return httpx.Response(400, json={"error": "unsupported parameter: think"})
        recovery_requested = any(
            "previous attempt returned no final content" in message.get("content", "")
            for message in payload["messages"]
        )
        content = '{"ok":true}' if recovery_requested else ""
        return httpx.Response(
            200,
            json={"choices": [{"finish_reason": "stop", "message": {"content": content}}]},
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=False,
        base_url="https://api.example.com/v1",
        model_name="compatible-model",
        seed=42,
    )
    client = OpenAICompatibleModelClient(settings, transport=httpx.MockTransport(handler))
    try:
        content = await client.chat_completion(
            messages=[{"role": "user", "content": "question"}],
            response_schema={"type": "object"},
        )
    finally:
        await client.close()

    assert content == '{"ok":true}'
    assert len(attempts) == 6
    assert attempts[2]["response_format"] == {"type": "json_object"}
    assert attempts[5]["response_format"] == {"type": "json_object"}
    assert "think" not in attempts[5]
    assert "without analysis" in attempts[5]["messages"][-1]["content"]


async def test_deepseek_v4_disables_thinking_for_json_object_fallback() -> None:
    attempts: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        attempts.append(payload)
        if payload.get("response_format", {}).get("type") == "json_schema":
            return httpx.Response(400, json={"error": "unsupported response format"})
        thinking = payload.get("thinking")
        if thinking != {"type": "disabled"}:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "", "reasoning_content": "thinking"},
                        }
                    ]
                },
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"ok":true}'}}]},
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=False,
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-v4-flash",
        seed=42,
    )
    client = OpenAICompatibleModelClient(settings, transport=httpx.MockTransport(handler))
    try:
        content = await client.chat_completion(
            messages=[{"role": "user", "content": "question"}],
            response_schema={"type": "object"},
        )
    finally:
        await client.close()

    assert content == '{"ok":true}'
    assert len(attempts) == 3
    assert all(attempt["thinking"] == {"type": "disabled"} for attempt in attempts)
    assert all("think" not in attempt for attempt in attempts)
    assert all("reasoning_effort" not in attempt for attempt in attempts)
