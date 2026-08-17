from __future__ import annotations

from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.schemas.retrieval import Citation, RetrievalFilters

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
