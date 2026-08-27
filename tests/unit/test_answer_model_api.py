from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import ModelEndpointSettings
from app.services.answer_model_manager import AnswerModelManager


def test_answer_model_catalog_is_available_when_default_model_is_disabled(
    client: TestClient,
) -> None:
    response = client.get("/api/v1/admin/answer-models", headers={"X-User-ID": "user-a"})

    assert response.status_code == 200
    assert response.json() == {
        "default_source_id": None,
        "default_model": None,
        "sources": [],
    }


def test_runtime_model_connection_api_never_returns_key(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_discover(
        settings: ModelEndpointSettings,
    ) -> tuple[list[str], str | None]:
        assert settings.api_key.get_secret_value() == "top-secret"
        return ["model-a", "model-b"], None

    monkeypatch.setattr(AnswerModelManager, "_discover", staticmethod(fake_discover))
    response = client.post(
        "/api/v1/admin/model-connections",
        headers={"X-User-ID": "user-a"},
        json={
            "name": "企业 API",
            "base_url": "https://models.example.com/v1",
            "api_key": "top-secret",
        },
    )

    assert response.status_code == 201
    payload = response.json()
    assert payload["source"]["models"] == ["model-a", "model-b"]
    assert "top-secret" not in response.text
    source_id = payload["source"]["source_id"]

    other_user = client.get(
        "/api/v1/admin/answer-models",
        headers={"X-User-ID": "user-b"},
    )
    assert other_user.json()["sources"] == []

    removed = client.delete(
        f"/api/v1/admin/model-connections/{source_id}",
        headers={"X-User-ID": "user-a"},
    )
    assert removed.status_code == 204
