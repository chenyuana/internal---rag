from __future__ import annotations

from pathlib import Path

from app.core.config import load_settings


def test_environment_override_and_interpolation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "default.yaml").write_text(
        """
app:
  name: test-app
  version: 1.0.0
  environment: development
  host: 0.0.0.0
  port: ${TEST_PORT:-8000}
database: {url: "postgresql+asyncpg://user:pass@db/test"}
redis: {url: "redis://redis/0"}
object_storage:
  endpoint: "http://minio:9000"
  bucket: test
  access_key: key
  secret_key: secret
ragflow:
  base_url: "http://ragflow:9380"
models:
  answer: {base_url: "http://a/v1", model_name: answer}
  verifier: {base_url: "http://v/v1", model_name: verifier}
  embedding: {base_url: "http://e/v1", model_name: embedding}
  reranker: {base_url: "http://r/v1", model_name: reranker}
retrieval: {}
logging: {}
""",
        encoding="utf-8",
    )
    (tmp_path / "development.yaml").write_text(
        "app:\n  name: overridden\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_PORT", "8123")

    settings = load_settings(config_dir=tmp_path, environment="development")

    assert settings.app.name == "overridden"
    assert settings.app.port == 8123


def test_public_config_redacts_all_secrets(settings) -> None:
    payload = str(settings.public_view())

    assert "change-me-now" not in payload
    assert "postgresql+asyncpg" not in payload
    assert "api_key" not in payload


def test_answer_model_deterministic_sampling_defaults(settings) -> None:
    assert settings.models.answer.temperature == 0.0
    assert settings.models.answer.seed == 42
