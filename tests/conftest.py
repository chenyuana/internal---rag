from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, load_settings
from app.main import create_app


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> Settings:
    monkeypatch.setenv("RAGFLOW_ENABLED", "false")
    monkeypatch.setenv("ANSWER_MODEL_ENABLED", "false")
    monkeypatch.setenv("VERIFIER_MODEL_ENABLED", "false")
    monkeypatch.setenv("EMBEDDING_MODEL_ENABLED", "false")
    monkeypatch.setenv("RERANKER_MODEL_ENABLED", "false")
    monkeypatch.setenv("MINERU_ENABLED", "false")
    monkeypatch.setenv("DOCLING_ENABLED", "false")
    monkeypatch.setenv("FIGURE_VLM_ENABLED", "false")
    monkeypatch.setenv("INGESTION_ROOT", str(tmp_path / "ingestion"))
    return load_settings(environment="development")


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client
