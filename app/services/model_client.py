from __future__ import annotations

import copy
import time
from typing import Any

import httpx

from app.core.config import ModelEndpointSettings
from app.core.exceptions import AppError
from app.services.probe import ProbeResult


def flatten_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Expand ``$ref``/``$defs`` and lift optional-with-defaults fields into
    ``required`` so small local models emit every field.

    pydantic ``model_json_schema()`` emits ``$ref`` references and omits
    optional fields from ``required``. Local models (e.g. qwen3.5 via Ollama)
    frequently drop referenced arrays under those schemas, producing JSON that
    fails the pydantic validator. Flattening the schema avoids that failure.
    """
    definitions = schema.get("$defs", {}) or {}

    def _resolve(value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                ref_name = value["$ref"].rsplit("/", 1)[-1]
                return _resolve(copy.deepcopy(definitions.get(ref_name, {})))
            return {key: _resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_resolve(item) for item in value]
        return value

    flattened = _resolve(schema)
    flattened.pop("$defs", None)

    def _strip_titles(value: Any) -> None:
        if isinstance(value, dict):
            value.pop("title", None)
            value.pop("$defs", None)
            for item in value.values():
                _strip_titles(item)
        elif isinstance(value, list):
            for item in value:
                _strip_titles(item)

    _strip_titles(flattened)

    def _lift_required(value: Any) -> None:
        """Recursively promote every declared field to ``required``.

        pydantic omits fields that carry defaults (e.g. ``claims``,
        ``citation_ids``) from ``required``. Small local models then drop
        those fields entirely, which breaks the pydantic validator. Requiring
        them makes the model emit every field.
        """
        if isinstance(value, list):
            for item in value:
                _lift_required(item)
            return
        if not isinstance(value, dict):
            return
        if isinstance(value.get("properties"), dict):
            properties = value["properties"]
            required = value.get("required", [])
            if not isinstance(required, list):
                required = []
            for key in properties:
                if key not in required:
                    required.append(key)
            value["required"] = required
            value["additionalProperties"] = False
            for prop in properties.values():
                _lift_required(prop)
        if isinstance(value.get("items"), dict):
            _lift_required(value["items"])

    _lift_required(flattened)

    return flattened



class OpenAICompatibleModelClient:
    """OpenAI-compatible model service adapter used by all configured local models."""

    def __init__(
        self,
        settings: ModelEndpointSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {}
        api_key = settings.api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.model_name = settings.model_name
        self.temperature = settings.temperature
        self.max_output_tokens = settings.max_output_tokens
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=settings.timeout_seconds,
            transport=transport,
            trust_env=False,
        )

    async def probe(self) -> ProbeResult:
        started = time.perf_counter()
        try:
            response = await self._client.get("models")
            response.raise_for_status()
            payload = response.json()
            model_ids = {
                item.get("id") for item in payload.get("data", []) if isinstance(item, dict)
            }
            if model_ids and self.model_name not in model_ids:
                return ProbeResult(
                    ready=False,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    detail="configured_model_not_reported",
                )
            return ProbeResult(
                ready=True,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            return ProbeResult(
                ready=False,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=type(exc).__name__,
            )

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        """Request strict JSON from a local OpenAI-compatible chat endpoint."""
        flattened_schema = flatten_json_schema(response_schema)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "reasoning_effort": "none",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_answer",
                    "strict": True,
                    "schema": flattened_schema,
                },
            },
        }
        try:
            response = await self._client.post("chat/completions", json=payload)
            response.raise_for_status()
            body = response.json()
            content = body["choices"][0]["message"]["content"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise AppError(
                code="ANSWER_MODEL_UNAVAILABLE",
                message="The local answer model is unavailable or returned an invalid response.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            )
            if text.strip():
                return text
        raise AppError(
            code="ANSWER_MODEL_INVALID_RESPONSE",
            message="The local answer model returned empty content.",
            status_code=502,
        )

    async def close(self) -> None:
        await self._client.aclose()
