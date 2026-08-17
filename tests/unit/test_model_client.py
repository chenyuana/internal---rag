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


async def test_model_chat_completion_requests_strict_json_schema() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["model"] == "expected-model"
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
        temperature=0.1,
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
