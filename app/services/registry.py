from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core.config import ModelEndpointSettings, Settings
from app.schemas.health import ServiceStatus
from app.services.answer_model_manager import AnswerModelManager
from app.services.model_client import OpenAICompatibleModelClient
from app.services.probe import ProbeResult
from app.services.ragflow_client import RagflowClient
from app.services.reranker_client import RerankerClient


@dataclass(slots=True)
class RegisteredModel:
    name: str
    settings: ModelEndpointSettings
    client: OpenAICompatibleModelClient | RerankerClient | None


class ServiceRegistry:
    def __init__(
        self,
        *,
        settings: Settings,
        ragflow_client: RagflowClient | None,
        models: list[RegisteredModel],
        answer_model_manager: AnswerModelManager,
    ) -> None:
        self._settings = settings
        self._ragflow_client = ragflow_client
        self._models = models
        self._answer_model_manager = answer_model_manager

    @classmethod
    def from_settings(cls, settings: Settings) -> ServiceRegistry:
        ragflow_client = RagflowClient(settings.ragflow) if settings.ragflow.enabled else None
        models = []
        for name, model_settings in (
            ("answer", settings.models.answer),
            ("verifier", settings.models.verifier),
            ("embedding", settings.models.embedding),
            ("reranker", settings.models.reranker),
        ):
            client: OpenAICompatibleModelClient | RerankerClient | None
            if not model_settings.enabled:
                client = None
            elif name == "reranker":
                client = RerankerClient(model_settings)
            else:
                client = OpenAICompatibleModelClient(model_settings)
            models.append(RegisteredModel(name=name, settings=model_settings, client=client))
        return cls(
            settings=settings,
            ragflow_client=ragflow_client,
            models=models,
            answer_model_manager=AnswerModelManager(settings.models.answer),
        )

    @property
    def ragflow_client(self) -> RagflowClient | None:
        return self._ragflow_client

    @property
    def reranker_client(self) -> RerankerClient | None:
        for model in self._models:
            if model.name == "reranker" and isinstance(model.client, RerankerClient):
                return model.client
        return None

    @property
    def answer_model_client(self) -> OpenAICompatibleModelClient | None:
        for model in self._models:
            if model.name == "answer" and isinstance(
                model.client,
                OpenAICompatibleModelClient,
            ):
                return model.client
        return None

    @property
    def verifier_model_client(self) -> OpenAICompatibleModelClient | None:
        for model in self._models:
            if model.name == "verifier" and isinstance(
                model.client,
                OpenAICompatibleModelClient,
            ):
                return model.client
        return None

    @property
    def answer_model_manager(self) -> AnswerModelManager:
        return self._answer_model_manager

    @staticmethod
    def _disabled(name: str, required: bool) -> ServiceStatus:
        return ServiceStatus(
            name=name,
            enabled=False,
            required=required,
            status="disabled",
        )

    @staticmethod
    def _to_status(
        name: str,
        *,
        enabled: bool,
        required: bool,
        result: ProbeResult,
    ) -> ServiceStatus:
        return ServiceStatus(
            name=name,
            enabled=enabled,
            required=required,
            status="ready" if result.ready else "unavailable",
            latency_ms=result.latency_ms,
            detail=result.detail,
        )

    async def probe_ragflow(self) -> ServiceStatus:
        config = self._settings.ragflow
        if not config.enabled or self._ragflow_client is None:
            return self._disabled("ragflow", config.required)
        result = await self._ragflow_client.probe()
        return self._to_status(
            "ragflow",
            enabled=True,
            required=config.required,
            result=result,
        )

    async def _probe_model(self, model: RegisteredModel) -> ServiceStatus:
        if not model.settings.enabled or model.client is None:
            return self._disabled(model.name, model.settings.required)
        result = await model.client.probe()
        return self._to_status(
            model.name,
            enabled=True,
            required=model.settings.required,
            result=result,
        )

    async def probe_models(self) -> list[ServiceStatus]:
        return list(await asyncio.gather(*(self._probe_model(item) for item in self._models)))

    async def probe_all(self) -> list[ServiceStatus]:
        ragflow, models = await asyncio.gather(self.probe_ragflow(), self.probe_models())
        return [ragflow, *models]

    async def close(self) -> None:
        close_calls = []
        if self._ragflow_client is not None:
            close_calls.append(self._ragflow_client.close())
        close_calls.extend(item.client.close() for item in self._models if item.client is not None)
        if close_calls:
            await asyncio.gather(*close_calls)
