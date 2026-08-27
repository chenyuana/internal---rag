from __future__ import annotations

import logging
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, load_settings
from app.main import create_app


@pytest.fixture(scope="session", autouse=True)
def silence_ragas_analytics() -> None:
    """ragas 0.2.x starts an analytics thread even when tracking is disabled;
    neutralize it so its atexit logging does not pollute pytest output."""
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"
    logging.getLogger("ragas").setLevel(logging.CRITICAL)
    try:
        import ragas._analytics as analytics

        # The atexit handler captured the bound method at import time, so patch
        # the singleton instance attributes (instance lookup shadows the class).
        analytics._analytics_batcher.flush = lambda: None
        analytics._analytics_batcher.shutdown = lambda: None
    except ImportError:
        pass


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Settings:
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
