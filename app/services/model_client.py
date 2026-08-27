from __future__ import annotations

import copy
import json
import time
from typing import Any, cast

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

    return cast(dict[str, Any], flattened)


class OpenAICompatibleModelClient:
    """Adapter for configured local or runtime OpenAI-compatible models."""

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
        self._uses_deepseek_v4_thinking = settings.model_name.startswith("deepseek-v4-")
        self.temperature = settings.temperature
        self.seed = settings.seed
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

    async def list_models(self) -> list[str]:
        """Return model ids from an OpenAI-compatible ``GET /models`` endpoint."""
        try:
            response = await self._client.get("models")
            response.raise_for_status()
            payload = response.json()
            data = payload.get("data", [])
            if not isinstance(data, list):
                raise TypeError("model list is not an array")
            return list(
                dict.fromkeys(
                    item["id"]
                    for item in data
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and item["id"].strip()
                )
            )
        except (httpx.HTTPError, ValueError, KeyError, TypeError) as exc:
            raise AppError(
                code="MODEL_CATALOG_UNAVAILABLE",
                message="The model service did not return a valid model catalog.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        """Request strict JSON from a local OpenAI-compatible chat endpoint."""
        flattened_schema = flatten_json_schema(response_schema)
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_answer",
                    "strict": True,
                    "schema": flattened_schema,
                },
            },
        }
        if self._uses_deepseek_v4_thinking:
            # DeepSeek V4 enables thinking by default. Its Chat Completions API
            # uses this object (rather than Ollama's ``think`` boolean) to turn
            # reasoning off, so the bounded output budget is reserved for the
            # required structured final answer.
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["reasoning_effort"] = "none"
        if self.seed is not None:
            payload["seed"] = self.seed
        try:
            response = await self._post_chat_payload(payload)
            body = response.json()
            message = body["choices"][0]["message"]
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise AppError(
                code="ANSWER_MODEL_UNAVAILABLE",
                message="The local answer model is unavailable or returned an invalid response.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc
        content = self._message_text(message)
        if content is not None:
            return content

        # A provider can return HTTP 200 with an empty final ``content`` (for
        # example, after spending the token budget on hidden reasoning). Retry
        # once with thinking explicitly disabled and without a fixed seed. The
        # normal compatibility path removes provider-specific parameters and
        # falls back from json_schema to json_object when necessary.
        recovery = copy.deepcopy(payload)
        recovery.pop("seed", None)
        if self._uses_deepseek_v4_thinking:
            recovery["thinking"] = {"type": "disabled"}
        else:
            recovery["think"] = False
        recovery_messages = copy.deepcopy(recovery["messages"])
        recovery_messages.append(
            {
                "role": "user",
                "content": (
                    "The previous attempt returned no final content. Return only "
                    "the required JSON object now, without analysis or explanation."
                ),
            }
        )
        recovery["messages"] = recovery_messages
        try:
            recovery_response = await self._post_chat_payload(recovery)
            recovery_body = recovery_response.json()
            recovery_message = recovery_body["choices"][0]["message"]
            recovered = self._message_text(recovery_message)
            if recovered is not None:
                return recovered
        except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
            raise AppError(
                code="ANSWER_MODEL_UNAVAILABLE",
                message="The selected answer model recovery request failed.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc
        raise AppError(
            code="ANSWER_MODEL_INVALID_RESPONSE",
            message="The selected answer model returned empty content.",
            status_code=502,
            details={
                "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
                "reasoning_content_present": bool(message.get("reasoning_content")),
                "empty_recovery_attempted": True,
            },
        )

    @staticmethod
    def _message_text(message: object) -> str | None:
        if not isinstance(message, dict):
            return None
        content = message.get("content")
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
        return None

    async def _post_chat_payload(self, payload: dict[str, Any]) -> httpx.Response:
        """Use strict schema first, then retry a portable strict variant.

        Some OpenAI-compatible APIs reject optional parameters such as
        ``reasoning_effort`` or ``seed`` while still supporting JSON Schema. Keep
        the schema on the first compatibility retry. Only then fall back to
        ``json_object``, with the complete schema embedded in the prompt so the
        provider still knows every required field. Never fall back to unconstrained
        plain text for a response that the application treats as verified JSON.
        """
        attempts = [payload]
        portable_strict = copy.deepcopy(payload)
        portable_strict.pop("reasoning_effort", None)
        portable_strict.pop("seed", None)
        portable_strict.pop("think", None)
        attempts.append(portable_strict)

        compatible = copy.deepcopy(portable_strict)
        schema = compatible["response_format"]["json_schema"]["schema"]
        compatible["response_format"] = {"type": "json_object"}
        schema_instruction = (
            "Return exactly one JSON object matching the following JSON Schema. "
            "Every property listed in required must be present. Do not omit arrays "
            "such as claims, missing_information, conflicts, or citation_ids. Do "
            "not add properties. JSON Schema: "
            + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
        messages = copy.deepcopy(compatible["messages"])
        if messages and messages[0].get("role") == "system":
            messages[0]["content"] = f"{messages[0]['content']}\n\n{schema_instruction}"
        else:
            messages.insert(0, {"role": "system", "content": schema_instruction})
        compatible["messages"] = messages
        attempts.append(compatible)
        last_response: httpx.Response | None = None
        for attempt in attempts:
            response = await self._client.post("chat/completions", json=attempt)
            last_response = response
            if response.status_code not in {400, 422}:
                response.raise_for_status()
                return response
        assert last_response is not None
        last_response.raise_for_status()
        return last_response

    async def close(self) -> None:
        await self._client.aclose()
