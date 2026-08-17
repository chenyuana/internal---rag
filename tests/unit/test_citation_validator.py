from __future__ import annotations

import pytest

from app.schemas.chat import CandidateClaim, EvidenceSentence, StructuredAnswer
from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.citation_validator import CitationValidator


def chunk(*, status: str = "effective") -> SelectedChunk:
    return SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text="HEALTH_MONITOR_PERIOD_MS = 1000U",
        metadata=ChunkMetadata(status=status),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )


def answer(*, citation_id: str = "C1") -> StructuredAnswer:
    return StructuredAnswer(
        answerability="ANSWERABLE",
        answer=f"任务周期为1000毫秒。[{citation_id}]",
        claims=[
            CandidateClaim(
                claim_id="claim-1",
                claim="任务周期为1000毫秒",
                citation_ids=[citation_id],
            )
        ],
    )


def evidence() -> list[EvidenceSentence]:
    return [
        EvidenceSentence(
            citation_id="C1",
            chunk_id="chunk-1",
            text="HEALTH_MONITOR_PERIOD_MS = 1000U",
        )
    ]


def test_citation_validator_accepts_backend_mapped_citation() -> None:
    CitationValidator(require_citations=True).validate(
        answer(),
        evidence=evidence(),
        chunks=[chunk()],
        include_historical=False,
    )


def test_citation_validator_rejects_model_created_citation() -> None:
    with pytest.raises(ValueError, match="outside the current evidence"):
        CitationValidator(require_citations=True).validate(
            answer(citation_id="C99"),
            evidence=evidence(),
            chunks=[chunk()],
            include_historical=False,
        )


def test_citation_validator_rejects_inactive_document_by_default() -> None:
    with pytest.raises(ValueError, match="not from an effective document"):
        CitationValidator(require_citations=True).validate(
            answer(),
            evidence=evidence(),
            chunks=[chunk(status="superseded")],
            include_historical=False,
        )


def test_citation_validator_accepts_chunk_without_status_field() -> None:
    """Legacy published chunks carry no status field (old publisher did not
    write it). They must be treated as effective, matching the retrieval
    service's backward-compatible filter."""
    CitationValidator(require_citations=True).validate(
        answer(),
        evidence=evidence(),
        chunks=[chunk(status=None)],
        include_historical=False,
    )


def test_citation_validator_accepts_citationless_missing_semantic_claim() -> None:
    """A claim reporting missing data is useful output and must not fail the
    whole answer, provided its wording clearly signals absence."""
    answer_obj = StructuredAnswer(
        answerability="PARTIALLY_ANSWERABLE",
        answer="资料中未明确说明普通可见光巡检无人机的作业最佳时段要求。",
        claims=[
            CandidateClaim(
                claim_id="claim-1",
                claim="资料中未明确说明普通可见光巡检无人机的作业最佳时段要求。",
                citation_ids=[],
            )
        ],
    )
    CitationValidator(require_citations=True).validate(
        answer_obj,
        evidence=evidence(),
        chunks=[chunk()],
        include_historical=False,
    )


def test_citation_validator_rejects_citationless_fabricated_claim() -> None:
    """A citation-less claim that asserts concrete facts (no absence markers)
    must still be rejected."""
    answer_obj = StructuredAnswer(
        answerability="PARTIALLY_ANSWERABLE",
        answer="普通可见光巡检无人机的最佳作业时段是14:00至16:00。",
        claims=[
            CandidateClaim(
                claim_id="claim-1",
                claim="普通可见光巡检无人机的最佳作业时段是14:00至16:00。",
                citation_ids=[],
            )
        ],
    )
    with pytest.raises(ValueError, match="has no citation"):
        CitationValidator(require_citations=True).validate(
            answer_obj,
            evidence=evidence(),
            chunks=[chunk()],
            include_historical=False,
        )
