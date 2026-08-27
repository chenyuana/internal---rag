from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.chat import Answerability, CandidateClaim, ReferenceDocument
from app.schemas.models import AnswerModelSelection
from app.schemas.retrieval import Citation, RetrievalFilters


class DocumentSectionGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_title: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=2000)
    audience: str | None = Field(default=None, max_length=200)
    target_length: Literal["brief", "standard", "detailed"] = "standard"
    section_title: str = Field(min_length=1, max_length=140)
    section_guidance: str = Field(min_length=1, max_length=1000)
    knowledge_base_ids: list[str] = Field(min_length=1, max_length=50)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    model: AnswerModelSelection | None = None
    supplemental_documents: list[ReferenceDocument] = Field(
        default_factory=list,
        max_length=8,
    )


class DocumentSectionGenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    status: Answerability
    content: str
    claims: list[CandidateClaim] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    validation_degraded: bool = False
    validation_warnings: list[str] = Field(default_factory=list)
    #: 硬性质量门禁是否通过（必需要素覆盖、范围一致性等实质问题）。
    #: 字数不足不再计入，仅作为 advisory_warnings 提示。
    quality_passed: bool = False
    generated_char_count: int = Field(default=0, ge=0)
    #: 建议正文篇幅（软性目标，按证据量缩放；不作为硬门禁）。
    target_min_chars: int = Field(default=0, ge=0)
    #: 硬性门禁警告（必需要素缺失、范围不当收缩等），阻断正文写入草稿。
    coverage_warnings: list[str] = Field(default_factory=list)
    #: 软性建议（如篇幅偏短），不阻断；正文仍交付由作者人工决定。
    advisory_warnings: list[str] = Field(default_factory=list)
    prompt_version: str
    model_source_id: str | None = None
    model_name: str | None = None
    evidence_document_count: int = Field(default=0, ge=0)
    evidence_sentence_count: int = Field(default=0, ge=0)


class SectionGuidanceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=140)
    guidance: str | None = Field(default=None, max_length=1000)


class GuidanceSuggestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_title: str = Field(min_length=1, max_length=160)
    objective: str = Field(min_length=1, max_length=2000)
    audience: str | None = Field(default=None, max_length=200)
    #: 结构文件/格式模板摘录，供模型理解章节上下文（不参与事实引用）。
    template_preview: str | None = Field(default=None, max_length=4000)
    sections: list[SectionGuidanceItem] = Field(min_length=1, max_length=50)
    model: AnswerModelSelection | None = None


class GuidanceSuggestResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sections: list[SectionGuidanceItem]
