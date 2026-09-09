from __future__ import annotations

import asyncio
import ipaddress
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from app.core.config import ModelEndpointSettings
from app.core.exceptions import AppError
from app.schemas.models import (
    AnswerModelCatalog,
    AnswerModelSelection,
    AnswerModelSource,
    ModelConnectionCreate,
)
from app.services.model_client import OpenAICompatibleModelClient

CONFIGURED_SOURCE_ID = "configured-answer"


@dataclass(slots=True)
class _RuntimeSource:
    source_id: str
    owner_id: str
    name: str
    settings: ModelEndpointSettings
    models: list[str]


class OnDemandAnswerModel:
    """Open a short-lived client for each generation call.

    Runtime connections may be removed while another request is running. Keeping
    an immutable settings snapshot here makes the in-flight request independent
    and guarantees every HTTP client is closed after use.
    """

    def __init__(self, settings: ModelEndpointSettings) -> None:
        self.model_name = settings.model_name
        self._settings = settings

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        client = OpenAICompatibleModelClient(self._settings)
        try:
            return await client.chat_completion(
                messages=messages,
                response_schema=response_schema,
            )
        finally:
            await client.close()


class AnswerModelManager:
    """Discover configured models and keep user API connections in memory only."""

    def __init__(self, configured: ModelEndpointSettings) -> None:
        self._configured = configured
        self._runtime: dict[str, _RuntimeSource] = {}
        self._lock = asyncio.Lock()

    async def catalog(self, *, user_id: str) -> AnswerModelCatalog:
        sources: list[AnswerModelSource] = []
        if self._configured.enabled:
            models, detail = await self._discover(self._configured)
            if self._configured.model_name not in models:
                models.insert(0, self._configured.model_name)
            sources.append(
                AnswerModelSource(
                    source_id=CONFIGURED_SOURCE_ID,
                    name="本地模型" if _is_local_url(self._configured.base_url) else "默认模型服务",
                    source_type=(
                        "local" if _is_local_url(self._configured.base_url) else "api"
                    ),
                    base_url=self._configured.base_url,
                    models=_unique(models),
                    default_model=self._configured.model_name,
                    removable=False,
                    available=detail is None,
                    detail=detail,
                )
            )
        async with self._lock:
            runtime = [item for item in self._runtime.values() if item.owner_id == user_id]
        sources.extend(self._public_runtime(item) for item in runtime)
        return AnswerModelCatalog(
            default_source_id=(CONFIGURED_SOURCE_ID if self._configured.enabled else None),
            default_model=(self._configured.model_name if self._configured.enabled else None),
            sources=sources,
        )

    async def connect(
        self,
        request: ModelConnectionCreate,
        *,
        user_id: str,
    ) -> AnswerModelSource:
        base_url = _normalize_base_url(request.base_url)
        probe_settings = self._configured.model_copy(
            update={
                "enabled": True,
                "required": False,
                "base_url": base_url,
                "api_key": request.api_key,
                "model_name": "__discover__",
            }
        )
        models, detail = await self._discover(probe_settings)
        if not models:
            raise AppError(
                code="MODEL_CONNECTION_FAILED",
                message="模型 API 未返回可用模型，请检查地址、API Key 和网络连接。",
                status_code=400,
                details={"error_type": detail or "empty_model_list"},
            )
        source_id = f"api-{uuid4().hex[:12]}"
        source = _RuntimeSource(
            source_id=source_id,
            owner_id=user_id,
            name=request.name.strip(),
            settings=probe_settings.model_copy(update={"model_name": models[0]}),
            models=models,
        )
        async with self._lock:
            self._runtime[source_id] = source
        return self._public_runtime(source)

    async def disconnect(self, source_id: str, *, user_id: str) -> None:
        async with self._lock:
            source = self._runtime.get(source_id)
            if source is None or source.owner_id != user_id:
                raise AppError(
                    code="MODEL_CONNECTION_NOT_FOUND",
                    message="模型连接不存在或已失效。",
                    status_code=404,
                )
            del self._runtime[source_id]

    async def resolve(
        self,
        selection: AnswerModelSelection,
        *,
        user_id: str,
    ) -> OnDemandAnswerModel:
        if selection.source_id == CONFIGURED_SOURCE_ID:
            if not self._configured.enabled:
                raise AppError(
                    code="ANSWER_MODEL_DISABLED",
                    message="默认答案模型未启用。",
                    status_code=503,
                )
            update = {"model_name": selection.model_name}
            if selection.thinking is not None:
                update["thinking"] = selection.thinking
            settings = self._configured.model_copy(update=update)
            return OnDemandAnswerModel(settings)
        async with self._lock:
            source = self._runtime.get(selection.source_id)
        if source is None or source.owner_id != user_id:
            raise AppError(
                code="MODEL_CONNECTION_NOT_FOUND",
                message="模型连接不存在或已失效，请重新接入。",
                status_code=404,
            )
        if selection.model_name not in source.models:
            raise AppError(
                code="MODEL_NOT_AVAILABLE",
                message="所选模型不属于当前模型连接。",
                status_code=400,
            )
        update = {"model_name": selection.model_name}
        if selection.thinking is not None:
            update["thinking"] = selection.thinking
        return OnDemandAnswerModel(source.settings.model_copy(update=update))

    @staticmethod
    async def _discover(settings: ModelEndpointSettings) -> tuple[list[str], str | None]:
        client = OpenAICompatibleModelClient(settings)
        try:
            models = await client.list_models()
            return [model for model in models if _is_answer_model(model)], None
        except AppError as exc:
            error_type = exc.details.get("error_type") if exc.details else None
            return [], str(error_type or exc.code)
        finally:
            await client.close()

    @staticmethod
    def _public_runtime(source: _RuntimeSource) -> AnswerModelSource:
        return AnswerModelSource(
            source_id=source.source_id,
            name=source.name,
            source_type="local" if _is_local_url(source.settings.base_url) else "api",
            base_url=source.settings.base_url,
            models=source.models,
            default_model=source.settings.model_name,
            removable=True,
            available=True,
        )


def _normalize_base_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    parts = urlsplit(raw)
    if parts.scheme not in {"http", "https"} or not parts.hostname:
        raise AppError(
            code="INVALID_MODEL_BASE_URL",
            message="模型地址必须是有效的 http:// 或 https:// URL。",
            status_code=400,
        )
    if parts.username or parts.password or parts.query or parts.fragment:
        raise AppError(
            code="INVALID_MODEL_BASE_URL",
            message="模型地址不能包含账号、密码、查询参数或片段。",
            status_code=400,
        )
    # Cloud metadata endpoints must never be reachable through this feature.
    try:
        address = ipaddress.ip_address(parts.hostname)
    except ValueError:
        address = None
    if address is not None and address.is_link_local:
        raise AppError(
            code="INVALID_MODEL_BASE_URL",
            message="不允许连接链路本地地址。",
            status_code=400,
        )
    path = parts.path.rstrip("/")
    if not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return urlunsplit((parts.scheme, parts.netloc, path, "", ""))


def _is_local_url(value: str) -> bool:
    host = (urlsplit(value).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(item for item in values if item))


def _is_answer_model(model_name: str) -> bool:
    """Hide obvious embedding/reranking endpoints from the answer-model picker."""
    lowered = model_name.lower()
    return not any(
        marker in lowered
        for marker in ("embedding", "text-embed", "rerank", "reranker")
    )
