from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class IngestionJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    WARNING = "warning"
    NEEDS_REVIEW = "needs_review"
    FAILED = "failed"


class PublicationStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    PLANNED = "planned"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REWORK = "rework"


class IngestionReview(BaseModel):
    status: ReviewStatus = ReviewStatus.UNREVIEWED
    note: str | None = None
    reviewer: str | None = None
    started_at: datetime | None = None
    reviewed_at: datetime | None = None
    quality_gate_override: bool = False
    overridden_quality_gates: list[dict[str, Any]] = Field(default_factory=list)
    quality_gate_overridden_at: datetime | None = None
    updated_at: datetime | None = None


class IngestionProgress(BaseModel):
    stage: str = "queued"
    percent: int = Field(default=0, ge=0, le=100)
    message: str = "等待进入处理队列"
    current: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    indeterminate: bool = False


class RagflowPublication(BaseModel):
    status: PublicationStatus = PublicationStatus.NOT_REQUESTED
    stage: str = "not_requested"
    percent: int = Field(default=0, ge=0, le=100)
    message: str = "尚未生成发布计划"
    dataset_id: str | None = None
    document_id: str | None = None
    plan_sha256: str | None = None
    planned_chunk_count: int = 0
    published_chunk_count: int = 0
    skipped_chunk_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def normalize_legacy_progress(self) -> RagflowPublication:
        if self.status == PublicationStatus.NOT_REQUESTED or self.stage != "not_requested":
            return self
        completed = self.published_chunk_count + self.skipped_chunk_count
        if self.status == PublicationStatus.PUBLISHED:
            self.stage = "published"
            self.percent = 100
            self.message = (
                f"发布完成：新增 {self.published_chunk_count}，跳过 {self.skipped_chunk_count}"
            )
        elif self.status == PublicationStatus.PUBLISHING:
            self.stage = "publishing_chunks" if self.document_id else "uploading_document"
            self.percent = (
                min(99, 5 + int(completed / self.planned_chunk_count * 94))
                if self.planned_chunk_count
                else 2
            )
            self.message = (
                f"正在写入 Chunk：{completed}/{self.planned_chunk_count}"
                if self.document_id
                else "正在向 RAGFlow 上传原始文件"
            )
        elif self.status == PublicationStatus.PLANNED:
            self.stage = "planned"
            self.message = f"发布计划已生成，共 {self.planned_chunk_count} 个 Chunk"
        elif self.status == PublicationStatus.FAILED:
            self.stage = "failed"
            self.message = self.error_message or "发布失败"
        return self


class IngestionQualitySummary(BaseModel):
    route: str | None = None
    page_count: int = 0
    chunk_count: int = 0
    text_coverage: float = 0.0
    toc_page_count: int = 0
    warning_count: int = 0
    failure_count: int = 0


class IngestionPageReprocessRequest(BaseModel):
    mode: Literal["clean", "ocr"] = "clean"


class IngestionJob(BaseModel):
    job_id: str
    status: IngestionJobStatus
    source_name: str
    source_relative_path: str | None = None
    batch_id: str | None = None
    sequence_in_batch: int | None = Field(default=None, ge=0)
    source_sha256: str
    media_type: str
    size_bytes: int = Field(ge=0)
    knowledge_base_id: str | None = None
    created_by: str
    original_path: str
    # 该资料是否启用 LLM 智能结构清洗（章节/附录/条号识别）。默认关闭；开启时
    # 解析后会对 chunk 的 section_path/article_id/title/keywords 做 LLM 语义重标注。
    llm_structure_enabled: bool = False
    output_path: str | None = None
    progress: IngestionProgress = Field(default_factory=IngestionProgress)
    quality: IngestionQualitySummary = Field(default_factory=IngestionQualitySummary)
    review: IngestionReview = Field(default_factory=IngestionReview)
    publication: RagflowPublication = Field(default_factory=RagflowPublication)
    error_code: str | None = None
    error_message: str | None = None
    attempts: int = Field(default=0, ge=0)
    page_reprocess: dict[str, Any] | None = None
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def normalize_legacy_progress(self) -> IngestionJob:
        if self.progress.stage != "queued" or self.status == IngestionJobStatus.QUEUED:
            return self
        if self.status == IngestionJobStatus.RUNNING:
            self.progress = IngestionProgress(
                stage="parsing",
                percent=15,
                message="解析器正在处理文档",
                indeterminate=True,
            )
        elif self.status == IngestionJobStatus.FAILED:
            self.progress = IngestionProgress(
                stage="failed",
                percent=self.progress.percent,
                message=self.error_message or "处理失败",
            )
        else:
            messages = {
                IngestionJobStatus.COMPLETED: "解析与质量检查已完成",
                IngestionJobStatus.WARNING: "解析完成，请抽检警告项",
                IngestionJobStatus.NEEDS_REVIEW: "解析完成，质量门禁要求人工复核",
            }
            self.progress = IngestionProgress(
                stage=self.status.value,
                percent=100,
                message=messages[self.status],
                current=self.quality.page_count,
                total=self.quality.page_count,
            )
        return self


class IngestionPreview(BaseModel):
    job_id: str
    status: IngestionJobStatus
    source_name: str
    route: str | None
    quality: dict[str, Any]
    page_routes: dict[str, int]
    pages: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]]


class IngestionPageDetail(BaseModel):
    job_id: str
    source_name: str
    page_number: int
    page: dict[str, Any]
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    chunks: list[dict[str, Any]] = Field(default_factory=list)
    assets: list[dict[str, Any]] = Field(default_factory=list)
    parser_trace: list[dict[str, Any]] = Field(default_factory=list)
    quality_gates: list[dict[str, Any]] = Field(default_factory=list)
    page_reprocess: dict[str, Any] = Field(default_factory=dict)
    source_url: str


class IngestionChunkUpdateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)
    reason: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def normalize_manual_revision(self) -> IngestionChunkUpdateRequest:
        self.text = self.text.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not self.text:
            raise ValueError("Chunk text cannot be blank.")
        if self.reason is not None:
            self.reason = self.reason.strip() or None
        return self


class IngestionChunkUpdateResult(BaseModel):
    job_id: str
    chunk_id: str
    text: str
    content_type: str
    page_start: int
    page_end: int
    revision: int = Field(ge=1)
    updated_at: datetime


class IngestionReviewUpdateRequest(BaseModel):
    status: ReviewStatus
    note: str | None = Field(default=None, max_length=2000)
    quality_gate_override: bool = False

    @model_validator(mode="after")
    def normalize_note(self) -> IngestionReviewUpdateRequest:
        if self.note is not None:
            self.note = self.note.strip() or None
        return self


class ParserServiceStatus(BaseModel):
    name: str
    enabled: bool
    ready: bool
    base_url: str
    detail: str | None = None


class IngestionQueueStatus(BaseModel):
    worker_concurrency: int = 1
    active_job_id: str | None = None
    queued_job_ids: list[str] = Field(default_factory=list)


class IngestionPublishRequest(BaseModel):
    dataset_id: str | None = None
    dry_run: bool = True


class IngestionExportRequest(BaseModel):
    job_ids: list[str]


class RagflowPublishPlan(BaseModel):
    job_id: str
    dataset_id: str
    source_name: str
    source_sha256: str
    chunk_count: int
    plan_sha256: str
    sample_payloads: list[dict[str, Any]] = Field(default_factory=list)


class RagflowPublishResult(BaseModel):
    dry_run: bool
    plan: RagflowPublishPlan
    publication: RagflowPublication
