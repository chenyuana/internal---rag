from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.models import AnswerModelSelection
from app.schemas.retrieval import Citation, CoverageStatus, RetrievalFilters

Answerability: TypeAlias = Literal[
    "ANSWERABLE",
    "PARTIALLY_ANSWERABLE",
    "UNANSWERABLE",
    "CONFLICTED",
]


class EvidenceSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")

    citation_id: str
    chunk_id: str
    text: str = Field(min_length=1)
    document_name: str | None = None
    version: str | None = None


class EvidenceConflict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    requirement: str
    citation_ids: list[str] = Field(min_length=2)
    details: str


class EvidenceAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Answerability
    covered_requirements: list[str] = Field(default_factory=list)
    missing_requirements: list[str] = Field(default_factory=list)
    conflicting_citations: list[EvidenceConflict] = Field(default_factory=list)


class CandidateClaim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=100)
    claim: str = Field(min_length=1)
    citation_ids: list[str] = Field(default_factory=list)


class StructuredAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answerability: Answerability
    answer: str
    claims: list[CandidateClaim] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_claim_ids(self) -> StructuredAnswer:
        claim_ids = [claim.claim_id for claim in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("claim_id values must be unique")
        if self.answerability in {"ANSWERABLE", "PARTIALLY_ANSWERABLE"}:
            if not self.answer.strip():
                raise ValueError("answer is required for an answerable response")
            if not self.claims:
                raise ValueError("at least one claim is required for an answerable response")
        return self


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    knowledge_base_ids: list[str] = Field(min_length=1, max_length=50)
    document_ids: list[str] = Field(default_factory=list, max_length=100)
    filters: RetrievalFilters = Field(default_factory=RetrievalFilters)
    candidate_top_k: int | None = Field(default=None, ge=1, le=100)
    conversation_id: str | None = Field(default=None, max_length=100)
    model: AnswerModelSelection | None = None
    reference_document: ReferenceDocument | None = None


class ReferenceDocument(BaseModel):
    """A request-local document used as auxiliary evidence and presentation context."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255)
    text: str = Field(min_length=1, max_length=20_000)


class ReferenceDocumentParsed(ReferenceDocument):
    truncated: bool = False


class DiagnosticCell(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cell_id: str
    subject: str | None = None
    aspect: str
    status: CoverageStatus
    document_ids: list[str] = Field(default_factory=list)
    citation_ids: list[str] = Field(default_factory=list)
    evidence_record_count: int = Field(default=0, ge=0)


class ChatDiagnostics(BaseModel):
    """Safe evaluation telemetry; never includes chunk text or model raw output."""

    model_config = ConfigDict(extra="forbid")

    plan_version: str
    planner_source: str
    planned_cell_count: int = Field(ge=0)
    cells: list[DiagnosticCell] = Field(default_factory=list)
    subject_document_ids: dict[str, list[str]] = Field(default_factory=dict)
    retrieval_calls: int = Field(default=0, ge=0)
    translated_cell_ids: list[str] = Field(default_factory=list)
    model_calls: int = Field(default=0, ge=0)
    model_call_breakdown: dict[str, int] = Field(default_factory=dict)
    latencies_ms: dict[str, float] = Field(default_factory=dict)
    stage_counts: dict[str, int] = Field(default_factory=dict)


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    conversation_id: str
    status: Answerability
    answer: str
    claims: list[CandidateClaim]
    citations: list[Citation]
    missing_information: list[str]
    conflicts: list[str]
    evidence_assessment: EvidenceAssessment
    prompt_version: str
    json_repaired: bool = False
    validation_degraded: bool = False
    validation_warnings: list[str] = Field(default_factory=list)
    model_source_id: str | None = None
    model_name: str | None = None
    diagnostics: ChatDiagnostics | None = None
