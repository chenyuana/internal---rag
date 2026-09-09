"""In-memory per-user runtime config for LLM-driven structure cleaning.

The ingestion console lets an operator set the model endpoint / model name /
API key used for ``llm_structure`` without editing env files. Like the answer
model connections, this config lives only in the Gateway process memory and is
lost on restart (re-enter it from the console).
"""

from __future__ import annotations

import threading
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr

_DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
_DEFAULT_MODEL = "deepseek-v4-flash"


class LlmStructureRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str = Field(default=_DEFAULT_BASE_URL, min_length=1, max_length=500)
    model: str = Field(default=_DEFAULT_MODEL, min_length=1, max_length=200)
    api_key: str = Field(default="", max_length=1000)
    max_tokens: int = Field(default=2048, ge=128, le=8192)
    page_batch_size: int = Field(default=3, ge=1, le=50)
    timeout_seconds: float = Field(default=180, gt=0)

    def public(self) -> dict[str, Any]:
        key = self.api_key
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key_configured": bool(key),
            "api_key_last4": key[-4:] if key else "",
            "max_tokens": self.max_tokens,
            "page_batch_size": self.page_batch_size,
            "timeout_seconds": self.timeout_seconds,
        }


class LlmStructureSettingsUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_url: str | None = Field(default=None, min_length=1, max_length=500)
    model: str | None = Field(default=None, min_length=1, max_length=200)
    api_key: SecretStr | None = None
    max_tokens: int | None = Field(default=None, ge=128, le=8192)
    page_batch_size: int | None = Field(default=None, ge=1, le=50)
    timeout_seconds: float | None = Field(default=None, gt=0)


class LlmStructureConfigStore:
    """Process-memory store keyed by user id."""

    def __init__(self) -> None:
        self._store: dict[str, LlmStructureRuntimeConfig] = {}
        self._lock = threading.Lock()

    def get(self, user_id: str) -> LlmStructureRuntimeConfig | None:
        with self._lock:
            return self._store.get(user_id)

    def update(
        self,
        user_id: str,
        payload: LlmStructureSettingsUpdate,
    ) -> LlmStructureRuntimeConfig:
        with self._lock:
            current = self._store.get(user_id) or LlmStructureRuntimeConfig()
            updates: dict[str, Any] = {}
            if payload.base_url is not None:
                updates["base_url"] = payload.base_url
            if payload.model is not None:
                updates["model"] = payload.model
            if payload.api_key is not None:
                updates["api_key"] = payload.api_key.get_secret_value()
            if payload.max_tokens is not None:
                updates["max_tokens"] = payload.max_tokens
            if payload.page_batch_size is not None:
                updates["page_batch_size"] = payload.page_batch_size
            if payload.timeout_seconds is not None:
                updates["timeout_seconds"] = payload.timeout_seconds
            merged = current.model_copy(update=updates)
            self._store[user_id] = merged
            return merged

    def clear_api_key(self, user_id: str) -> LlmStructureRuntimeConfig | None:
        with self._lock:
            current = self._store.get(user_id)
            if current is None:
                return None
            cleared = current.model_copy(update={"api_key": ""})
            self._store[user_id] = cleared
            return cleared


module_store = LlmStructureConfigStore()
