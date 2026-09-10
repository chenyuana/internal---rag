from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}")


def _is_local_url(value: str) -> bool:
    """True when a model base_url points at a loopback host (local inference)."""
    from urllib.parse import urlsplit

    host = (urlsplit(value).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


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


class LlmStructureSettings(BaseModel):
    """Optional LLM-driven re-annotation of chunk structure metadata.

    ``split_blocks/build_chunks`` heuristically classify headings; for Federal
    Register amendment documents this mis-binds appendix content to the
    amended section and treats preamble ``Section 23.NNN <verb>...`` reference
    prose as clause headings. When enabled, this runs an OpenAI-compatible chat
    model over each batch of pages and rewrites ``section_path / article_id_* /
    title / article_aliases / keywords`` from reading-order boundary events.
    Body text is never rewritten. Any error falls back to the heuristic result.
    """

    enabled: bool = False
    base_url: str = "https://api.deepseek.com/v1"
    model: str = "deepseek-v4-flash"
    api_key: SecretStr = Field(default=SecretStr(""))
    timeout_seconds: float = Field(default=180, gt=0)
    page_batch_size: int = Field(default=3, ge=1, le=50)
    max_tokens: int = Field(default=2048, ge=128, le=8192)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)



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
    llm_structure: LlmStructureSettings = Field(default_factory=LlmStructureSettings)


class AccessControlSettings(BaseModel):
    mode: Literal["development_passthrough", "static", "deny_all"] = "deny_all"
    static_grants: dict[str, list[str]] = Field(default_factory=dict)


class ModelEndpointSettings(BaseModel):
    enabled: bool = False
    required: bool = False
    base_url: str
    api_key: SecretStr = SecretStr("")
    model_name: str
    thinking: bool = False
    timeout_seconds: float = Field(default=30, gt=0)
    operation_path: str | None = None
    temperature: float = Field(default=0, ge=0, le=2)
    seed: int | None = Field(default=None, ge=0)
    max_output_tokens: int = Field(default=1024, ge=1)


class ModelsSettings(BaseModel):
    answer: ModelEndpointSettings
    verifier: ModelEndpointSettings
    embedding: ModelEndpointSettings
    reranker: ModelEndpointSettings


class PlanningSettings(BaseModel):
    enabled: bool = True
    model_enabled: bool = False
    model_role: Literal["verifier"] = "verifier"
    timeout_seconds: float = Field(default=15, gt=0)
    max_subqueries: int = Field(default=8, ge=1, le=8)
    min_confidence: float = Field(default=0.75, ge=0, le=1)
    use_deterministic_fast_path: bool = True
    fallback_to_legacy: bool = True
    shadow_mode: bool = True
    diagnostics_enabled: bool = True
    prompt_path: str = "prompts/query_planning.txt"
    prompt_version: str = "query-planning-v1"


class TranslationSettings(BaseModel):
    enabled: bool = True
    model_enabled: bool = False
    trigger: Literal["low_confidence_only"] = "low_confidence_only"
    timeout_seconds: float = Field(default=10, gt=0)
    max_variants_per_cell: int = Field(default=1, ge=0, le=1)
    max_total_translations: int = Field(default=4, ge=0, le=8)
    min_original_results: int = Field(default=1, ge=0, le=20)
    min_coverage_confidence: float = Field(default=0.55, ge=0, le=1)
    preserve_original_query: bool = True
    prompt_path: str = "prompts/query_translation.txt"
    prompt_version: str = "query-translation-v1"


class RetrievalSettings(BaseModel):
    candidate_top_k: int = Field(default=30, ge=1)
    rerank_input_k: int = Field(default=12, ge=1)
    rerank_top_k: int = Field(default=6, ge=1)
    max_final_chunks: int = Field(default=8, ge=1)
    # 流程/汇总类问题（"完整操作流程包含哪些关键环节"）：需要覆盖从起点到
    # 终点的各环节章节，名额和重排输入都放宽，并按章节多样性选择；同名多
    # 规程（如省标+市标）时还要保证每份规程的流程章节都能进入（每份规程
    # 按文档配额轮转，名额需 ≥ 每份规程的章节数×规程数）。
    procedure_rerank_input_k: int = Field(default=22, ge=1, le=50)
    procedure_rerank_top_k: int = Field(default=16, ge=1, le=30)
    procedure_final_limit: int = Field(default=18, ge=1, le=30)
    similarity_threshold: float = Field(default=0.25, ge=0, le=1)
    vector_similarity_weight: float = Field(default=0.3, ge=0, le=1)
    allow_rerank_fallback: bool = False
    enable_ragflow_keyword_extraction: bool = False
    #: 零命中时启用“精确锚点确定性兜底”：提取条款号/标准号/版本/引号术语等
    #: 锚点，用 RAGFlow 关键词模式再检索一轮并标记 deterministic_fallback。
    enable_deterministic_fallback: bool = True
    enable_query_rewrite: bool = True
    enable_parent_context: bool = True
    enable_adjacent_context: bool = True
    max_adjacent_chunks: int = Field(default=1, ge=0)
    max_complex_subqueries: int = Field(default=12, ge=2, le=20)
    complex_candidates_per_subquery: int = Field(default=8, ge=1, le=12)
    complex_rerank_input_k: int = Field(default=20, ge=2, le=60)
    # 对比/多跳矩阵的最终证据名额（对比矩阵每格需要 1~3 个 chunk 承载
    # 各维度章节，如桥梁设备选型需同时含 4.3.2 无人机与 4.3.4 云台相机）。
    complex_final_limit: int = Field(default=12, ge=4, le=24)
    # 实体枚举题的答案常跨越同一文档的许多讨论段落，需给每个检索视角
    # 足够的重排输入，并允许更多去重后的章节进入生成。
    enumeration_rerank_input_k: int = Field(default=36, ge=3, le=100)
    enumeration_final_limit: int = Field(default=20, ge=4, le=40)
    complex_similarity_threshold: float = Field(default=0.15, ge=0, le=1)
    min_rerank_score: float = Field(default=0.1, ge=0, le=1)
    complex_min_rerank_score: float = Field(default=0.1, ge=0, le=1)
    # 问题显式限定"结合 A、B 两类…规程/规范"时（procedure/summary 类），只允许
    # 属于声明类别的文档进入生成证据；范围内规程缺失的细节如实写"未明确说明"，
    # 禁止用其它作物/领域规程内容顶替（如林地问题引用马铃薯规范）。
    enforce_declared_scope: bool = True
    max_parallel_ragflow_requests: int = Field(default=2, ge=1, le=8)
    per_cell_candidate_top_k: int = Field(default=12, ge=1, le=50)
    max_section_expansion_queries_per_cell: int = Field(default=2, ge=0, le=6)
    fusion_rrf_k: int = Field(default=60, ge=1, le=200)
    # 仅用于 weighted RRF 中 translated 排名的贡献，不与原始相似度相乘。
    translated_query_weight: float = Field(default=0.85, gt=0, le=1)


class GenerationSettings(BaseModel):
    cross_language_generation: bool = True
    require_citations: bool = True
    reject_on_no_evidence: bool = True
    allow_partial_answer: bool = True
    enforce_scope_consistency: bool = True
    enforce_claim_grounding: bool = True
    enforce_subject_grounding: bool = True
    enforce_claim_topic: bool = True
    # 对比矩阵完整性：已覆盖的每个（主体×维度）格子必须在答案中出现，
    # 缺失触发 repair（并允许第二轮回补），repair 后仍缺失则优雅降级。
    enforce_comparison_completeness: bool = True
    json_repair_attempts: int = Field(default=1, ge=0, le=2)
    max_evidence_sentences: int = Field(default=16, ge=1, le=100)
    enumeration_max_evidence_sentences: int = Field(default=30, ge=1, le=100)
    max_sentences_per_citation: int = Field(default=5, ge=1, le=10)
    min_evidence_overlap: float = Field(default=0.08, ge=0, le=1)
    answer_prompt_path: str = "prompts/answer_generation.txt"
    repair_prompt_path: str = "prompts/json_repair.txt"
    prompt_version: str = "answer-generation-v15"
    document_section_prompt_path: str = "prompts/document_section_generation.txt"
    document_section_prompt_version: str = "document-section-generation-v1"
    comparison_matrix_first: bool = True
    # 答案生成模式：deterministic=确定性构建优先（现状）；llm=直接由模型(API)生成。
    answer_mode: Literal["deterministic", "llm", "hybrid"] = "deterministic"
    # 确定性答案生成后，再让模型用通俗语言"再讲一遍"（只转述、不新增事实）。
    explain_enabled: bool = False
    explain_prompt_path: str = "prompts/answer_explanation.txt"
    comparison_summary_enabled: bool = True
    comparison_summary_timeout_seconds: float = Field(default=30, gt=0)
    comparison_summary_max_attempts: int = Field(default=1, ge=0, le=1)
    matrix_summary_prompt_path: str = "prompts/matrix_summary.txt"


class LoggingSettings(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_logs: bool = Field(default=True, alias="json")
    store_full_query: bool = True
    store_full_context: bool = False
    store_model_raw_output: bool = False
    mask_sensitive_data: bool = True


class EvaluationSettings(BaseModel):
    """Read-only access to reports produced by the sibling rag-evaluation project."""

    enabled: bool = True
    database_path: Path = Path("D:/internal-rag/temp/rag-eval-reports/runs.db")


class RagasSettings(BaseModel):
    """RAGAS calibration evaluation executed from the live monitor page.

    The judge LLM and embeddings default to local Ollama models.  When the
    ``api_model`` field is set, an OpenAI-compatible API endpoint is used for
    the judge LLM instead (and ``api_embedding_model`` optionally switches the
    embeddings too); leaving them empty keeps the fully-offline local path.
    """

    enabled: bool = True
    judge_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    # OpenAI-compatible API backend (empty = use local Ollama).
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = ""
    api_embedding_model: str = ""
    benchmark_path: Path = Path(
        "D:/internal-rag/source/rag-evaluation/datasets/benchmark.jsonl"
    )
    max_cases: int = Field(default=20, ge=1, le=100)
    # Runtime-only override of the derived backend (set by the monitor UI).
    # None = derive from ``api_model`` presence; never persisted to .env.
    active_backend: Literal["api", "local"] | None = None

    @property
    def llm_backend(self) -> str:
        if self.active_backend:
            return self.active_backend
        return "api" if self.api_model else "local"


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
    planning: PlanningSettings = Field(default_factory=PlanningSettings)
    translation: TranslationSettings = Field(default_factory=TranslationSettings)
    retrieval: RetrievalSettings
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    ragas: RagasSettings = Field(default_factory=RagasSettings)
    logging: LoggingSettings

    @model_validator(mode="after")
    def _auto_enable_remote_model_stages(self) -> Settings:
        """When an answer/verifier model endpoint is a remote (API) service,
        auto-enable the model-dependent stages (planning, translation,
        comparison summary) that default to off for the local-only setup.
        Explicit env overrides (PLANNING_MODEL_ENABLED, TRANSLATION_MODEL_ENABLED,
        COMPARISON_SUMMARY_ENABLED) take precedence over this auto behaviour."""
        remote_api = any(
            endpoint.enabled and not _is_local_url(endpoint.base_url)
            for endpoint in (self.models.answer, self.models.verifier)
        )
        if remote_api:
            if os.getenv("PLANNING_MODEL_ENABLED") is None:
                self.planning.model_enabled = True
            if os.getenv("TRANSLATION_MODEL_ENABLED") is None:
                self.translation.model_enabled = True
            if os.getenv("COMPARISON_SUMMARY_ENABLED") is None:
                self.generation.comparison_summary_enabled = True
        return self

    def public_view(self) -> dict[str, object]:
        """Expose operational settings while omitting credentials and connection secrets."""
        model_view = {
            name: {
                "enabled": endpoint.enabled,
                "required": endpoint.required,
                "base_url": endpoint.base_url,
                "model_name": endpoint.model_name,
                "thinking": endpoint.thinking,
                "max_output_tokens": endpoint.max_output_tokens,
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
            "planning": self.planning.model_dump(),
            "translation": self.translation.model_dump(),
            "retrieval": self.retrieval.model_dump(),
            "generation": self.generation.model_dump(),
            "ragas": {
                "enabled": self.ragas.enabled,
                "backend": self.ragas.llm_backend,
                "judge_model": self.ragas.judge_model,
                "embedding_model": self.ragas.embedding_model,
                "ollama_base_url": self.ragas.ollama_base_url,
                "api_base_url": self.ragas.api_base_url,
                "api_model": self.ragas.api_model,
                "api_embedding_model": self.ragas.api_embedding_model,
            },
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
