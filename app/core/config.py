from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


class AppSettings(BaseModel):
    name: str = "internal-rag"
    version: str = "0.1.0"
    environment: Literal["development", "test", "production"] = "development"
    host: str = "0.0.0.0"
    port: int = Field(default=8080, ge=1, le=65535)


class DatabaseSettings(BaseModel):
    url: SecretStr


class RedisSettings(BaseModel):
    url: SecretStr


class ObjectStorageSettings(BaseModel):
    endpoint: str
    bucket: str
    access_key: SecretStr
    secret_key: SecretStr


class RagflowSettings(BaseModel):
    enabled: bool = False
    required: bool = False
    base_url: str
    api_key: SecretStr = SecretStr("")
    health_path: str = "/"
    timeout_seconds: float = Field(default=10, gt=0)
    publish_timeout_seconds: float = Field(default=120, gt=0)
    publish_batch_size: int = Field(default=16, ge=1, le=32)


class RemoteParserSettings(BaseModel):
    enabled: bool = False
    base_url: str = ""
    health_path: str = "/openapi.json"
    timeout_seconds: float = Field(default=1800, gt=0)
    backend: str = "pipeline"
    parse_method: str = "auto"
    # MinerU / Docling 每个 HTTP 请求处理的连续页数上限。越大吞吐越高，
    # 但会增加单次请求的解压与内存峰值；8GB 显存下建议 10-30。
    page_batch_size: int = Field(default=10, ge=1, le=200)


class FigureVisionSettings(BaseModel):
    enabled: bool = False
    base_url: str = "http://127.0.0.1:11434"
    health_path: str = "/api/version"
    model_name: str = "qwen3.5:9b-q4_K_M"
    timeout_seconds: float = Field(default=300, gt=0)
    max_output_tokens: int = Field(default=800, ge=128, le=4096)


class IngestionSettings(BaseModel):
    enabled: bool = True
    root_dir: Path = Path("D:/internal-rag/data/ingestion")
    max_upload_bytes: int = Field(default=200 * 1024 * 1024, ge=1024)
    allowed_extensions: list[str] = Field(
        default_factory=lambda: [".pdf", ".docx"]
    )
    preview_chunk_limit: int = Field(default=10, ge=1, le=100)
    worker_concurrency: int = Field(default=1, ge=1, le=1)
    mineru: RemoteParserSettings = Field(default_factory=RemoteParserSettings)
    docling: RemoteParserSettings = Field(
        default_factory=lambda: RemoteParserSettings(
            health_path="/openapi.json",
            parse_method="raw",
        )
    )
    figure_vlm: FigureVisionSettings = Field(default_factory=FigureVisionSettings)


class AccessControlSettings(BaseModel):
    mode: Literal["development_passthrough", "static", "deny_all"] = "deny_all"
    static_grants: dict[str, list[str]] = Field(default_factory=dict)


class ModelEndpointSettings(BaseModel):
    enabled: bool = False
    required: bool = False
    base_url: str
    api_key: SecretStr = SecretStr("")
    model_name: str
    timeout_seconds: float = Field(default=30, gt=0)
    operation_path: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=1024, ge=1)


class ModelsSettings(BaseModel):
    answer: ModelEndpointSettings
    verifier: ModelEndpointSettings
    embedding: ModelEndpointSettings
    reranker: ModelEndpointSettings


class RetrievalSettings(BaseModel):
    candidate_top_k: int = Field(default=30, ge=1)
    rerank_input_k: int = Field(default=30, ge=1)
    rerank_top_k: int = Field(default=6, ge=1)
    max_final_chunks: int = Field(default=8, ge=1)
    similarity_threshold: float = Field(default=0.25, ge=0, le=1)
    vector_similarity_weight: float = Field(default=0.3, ge=0, le=1)
    allow_rerank_fallback: bool = False
    enable_ragflow_keyword_extraction: bool = False
    enable_query_rewrite: bool = True
    enable_parent_context: bool = True
    enable_adjacent_context: bool = True
    max_adjacent_chunks: int = Field(default=1, ge=0)
    max_complex_subqueries: int = Field(default=12, ge=2, le=20)
    complex_candidates_per_subquery: int = Field(default=2, ge=1, le=10)
    complex_rerank_input_k: int = Field(default=12, ge=2, le=50)
    complex_similarity_threshold: float = Field(default=0.15, ge=0, le=1)
    complex_min_rerank_score: float = Field(default=0.1, ge=0, le=1)


class GenerationSettings(BaseModel):
    require_citations: bool = True
    reject_on_no_evidence: bool = True
    allow_partial_answer: bool = True
    json_repair_attempts: int = Field(default=1, ge=0, le=1)
    max_evidence_sentences: int = Field(default=16, ge=1, le=100)
    max_sentences_per_citation: int = Field(default=3, ge=1, le=10)
    min_evidence_overlap: float = Field(default=0.08, ge=0, le=1)
    answer_prompt_path: str = "prompts/answer_generation.txt"
    repair_prompt_path: str = "prompts/json_repair.txt"
    prompt_version: str = "answer-generation-v1"


class LoggingSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_logs: bool = Field(default=True, alias="json")
    store_full_query: bool = True
    store_full_context: bool = False
    store_model_raw_output: bool = False
    mask_sensitive_data: bool = True


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    app: AppSettings
    database: DatabaseSettings
    redis: RedisSettings
    object_storage: ObjectStorageSettings
    ragflow: RagflowSettings
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    access_control: AccessControlSettings = Field(default_factory=AccessControlSettings)
    models: ModelsSettings
    retrieval: RetrievalSettings
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    logging: LoggingSettings

    def public_view(self) -> dict[str, object]:
        """Expose operational settings while omitting credentials and connection secrets."""
        model_view = {
            name: {
                "enabled": endpoint.enabled,
                "required": endpoint.required,
                "base_url": endpoint.base_url,
                "model_name": endpoint.model_name,
                "timeout_seconds": endpoint.timeout_seconds,
                "operation_path": endpoint.operation_path,
            }
            for name, endpoint in (
                ("answer", self.models.answer),
                ("verifier", self.models.verifier),
                ("embedding", self.models.embedding),
                ("reranker", self.models.reranker),
            )
        }
        return {
            "app": self.app.model_dump(),
            "ragflow": {
                "enabled": self.ragflow.enabled,
                "required": self.ragflow.required,
                "base_url": self.ragflow.base_url,
                "health_path": self.ragflow.health_path,
                "timeout_seconds": self.ragflow.timeout_seconds,
                "publish_timeout_seconds": self.ragflow.publish_timeout_seconds,
                "publish_batch_size": self.ragflow.publish_batch_size,
            },
            "ingestion": {
                "enabled": self.ingestion.enabled,
                "root_dir": str(self.ingestion.root_dir),
                "max_upload_bytes": self.ingestion.max_upload_bytes,
                "allowed_extensions": self.ingestion.allowed_extensions,
                "preview_chunk_limit": self.ingestion.preview_chunk_limit,
                "worker_concurrency": self.ingestion.worker_concurrency,
                "mineru": {
                    "enabled": self.ingestion.mineru.enabled,
                    "base_url": self.ingestion.mineru.base_url,
                    "health_path": self.ingestion.mineru.health_path,
                    "timeout_seconds": self.ingestion.mineru.timeout_seconds,
                    "backend": self.ingestion.mineru.backend,
                    "parse_method": self.ingestion.mineru.parse_method,
                    "page_batch_size": self.ingestion.mineru.page_batch_size,
                },
                "docling": {
                    "enabled": self.ingestion.docling.enabled,
                    "base_url": self.ingestion.docling.base_url,
                    "health_path": self.ingestion.docling.health_path,
                    "timeout_seconds": self.ingestion.docling.timeout_seconds,
                    "parse_method": self.ingestion.docling.parse_method,
                },
                "figure_vlm": {
                    "enabled": self.ingestion.figure_vlm.enabled,
                    "base_url": self.ingestion.figure_vlm.base_url,
                    "health_path": self.ingestion.figure_vlm.health_path,
                    "model_name": self.ingestion.figure_vlm.model_name,
                    "timeout_seconds": self.ingestion.figure_vlm.timeout_seconds,
                    "max_output_tokens": self.ingestion.figure_vlm.max_output_tokens,
                },
            },
            "access_control": {"mode": self.access_control.mode},
            "models": model_view,
            "retrieval": self.retrieval.model_dump(),
            "generation": self.generation.model_dump(),
            "logging": self.logging.model_dump(by_alias=True),
        }


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_environment(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        variable, default = match.group(1), match.group(2)
        if variable in os.environ:
            return os.environ[variable]
        if default is not None:
            return default
        raise ValueError(f"Required environment variable is not set: {variable}")

    return ENV_PATTERN.sub(replace, value)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(content, dict):
        raise ValueError(f"Configuration root must be a mapping: {path}")
    return content


def load_settings(
    *,
    config_dir: Path | None = None,
    environment: str | None = None,
) -> Settings:
    """Load default YAML, merge the selected environment, expand env vars, and validate."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    selected_environment = environment or os.getenv("APP_ENV", "development")
    selected_dir = config_dir or Path(
        os.getenv("INTERNAL_RAG_CONFIG_DIR", str(PROJECT_ROOT / "config"))
    )
    default_config = _read_yaml(selected_dir / "default.yaml")
    environment_path = selected_dir / f"{selected_environment}.yaml"
    merged = (
        _deep_merge(default_config, _read_yaml(environment_path))
        if environment_path.exists()
        else default_config
    )
    expanded = _expand_environment(merged)
    return Settings.model_validate(expanded)
