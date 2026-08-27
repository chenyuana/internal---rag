from __future__ import annotations

from typing import Any

import pytest
from pydantic import SecretStr

from app.core.config import ModelEndpointSettings
from app.core.exceptions import AppError
from app.schemas.models import AnswerModelSelection, ModelConnectionCreate
from app.services import answer_model_manager as manager_module
from app.services.answer_model_manager import AnswerModelManager


class FakeClient:
    created_settings: list[ModelEndpointSettings] = []

    def __init__(self, settings: ModelEndpointSettings) -> None:
        self.settings = settings
        self.created_settings.append(settings)

    async def list_models(self) -> list[str]:
        return ["api-model-a", "api-model-b"]

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        assert messages
        assert response_schema
        return "{}"

    async def close(self) -> None:
        return None


def configured_settings(*, enabled: bool = True) -> ModelEndpointSettings:
    return ModelEndpointSettings(
        enabled=enabled,
        required=False,
        base_url="http://127.0.0.1:11434/v1",
        api_key=SecretStr("local-secret"),
        model_name="local-model",
        temperature=0,
        seed=42,
        max_output_tokens=2048,
    )


async def test_catalog_and_runtime_connection_are_owner_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    FakeClient.created_settings = []
    monkeypatch.setattr(manager_module, "OpenAICompatibleModelClient", FakeClient)
    manager = AnswerModelManager(configured_settings())

    source = await manager.connect(
        ModelConnectionCreate(
            name="企业 API",
            base_url="https://models.example.com",
            api_key=SecretStr("top-secret"),
        ),
        user_id="user-a",
    )

    assert source.base_url == "https://models.example.com/v1"
    assert source.models == ["api-model-a", "api-model-b"]
    owner_catalog = await manager.catalog(user_id="user-a")
    other_catalog = await manager.catalog(user_id="user-b")
    assert [item.source_id for item in owner_catalog.sources] == [
        "configured-answer",
        source.source_id,
    ]
    assert [item.source_id for item in other_catalog.sources] == ["configured-answer"]
    assert "top-secret" not in owner_catalog.model_dump_json()

    resolved = await manager.resolve(
        AnswerModelSelection(source_id=source.source_id, model_name="api-model-b"),
        user_id="user-a",
    )
    assert resolved.model_name == "api-model-b"

    with pytest.raises(AppError) as exc_info:
        await manager.resolve(
            AnswerModelSelection(source_id=source.source_id, model_name="api-model-a"),
            user_id="user-b",
        )
    assert exc_info.value.code == "MODEL_CONNECTION_NOT_FOUND"


async def test_disconnect_removes_runtime_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(manager_module, "OpenAICompatibleModelClient", FakeClient)
    manager = AnswerModelManager(configured_settings(enabled=False))
    source = await manager.connect(
        ModelConnectionCreate(name="API", base_url="http://localhost:8000/v1"),
        user_id="user-a",
    )

    await manager.disconnect(source.source_id, user_id="user-a")

    assert (await manager.catalog(user_id="user-a")).sources == []


async def test_link_local_model_endpoint_is_rejected() -> None:
    manager = AnswerModelManager(configured_settings(enabled=False))

    with pytest.raises(AppError) as exc_info:
        await manager.connect(
            ModelConnectionCreate(name="metadata", base_url="http://169.254.169.254"),
            user_id="user-a",
        )

    assert exc_info.value.code == "INVALID_MODEL_BASE_URL"
