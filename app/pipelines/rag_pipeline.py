from __future__ import annotations

from collections.abc import Iterable
from typing import cast
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    EvidenceAssessment,
    StructuredAnswer,
)
from app.schemas.retrieval import Citation, RetrievalSearchRequest
from app.services.answer_generator import AnswerGenerator, AnswerModel
from app.services.citation_validator import CITATION_MARKER, CitationValidator
from app.services.evidence_extractor import EvidenceExtractor
from app.services.evidence_judge import EvidenceJudge
from app.services.model_client import OpenAICompatibleModelClient
from app.services.retrieval_service import RetrievalExecution, RetrievalService


class RagPipeline:
    """Run retrieval and the stage-four constrained generation gates."""

    def __init__(
        self,
        *,
        settings: Settings,
        retrieval: RetrievalService,
        answer_model: AnswerModel | None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._judge = EvidenceJudge(settings.generation)
        self._extractor = EvidenceExtractor(settings.generation)
        self._citation_validator = CitationValidator(
            require_citations=settings.generation.require_citations
        )
        self._generator = (
            AnswerGenerator(settings=settings.generation, model=answer_model)
            if answer_model is not None
            else None
        )

    @classmethod
    def from_services(
        cls,
        *,
        settings: Settings,
        retrieval: RetrievalService,
        answer_model: OpenAICompatibleModelClient | None,
    ) -> RagPipeline:
        return cls(
            settings=settings,
            retrieval=retrieval,
            answer_model=cast(AnswerModel | None, answer_model),
        )

    async def run(
        self,
        request: ChatCompletionRequest,
        *,
        user_id: str,
    ) -> ChatCompletionResponse:
        execution = await self._retrieval.execute(
            RetrievalSearchRequest(
                query=request.query,
                knowledge_base_ids=request.knowledge_base_ids,
                document_ids=request.document_ids,
                filters=request.filters,
                candidate_top_k=request.candidate_top_k,
            ),
            user_id=user_id,
        )
        assessment = self._judge.assess(
            execution.query,
            execution.selected_chunks,
            execution.coverage_matrix,
        )
        conversation_id = request.conversation_id or str(uuid4())
        if assessment.status in {"UNANSWERABLE", "CONFLICTED"}:
            return self._safe_response(
                execution=execution,
                assessment=assessment,
                conversation_id=conversation_id,
            )
        if self._generator is None:
            raise AppError(
                code="ANSWER_MODEL_DISABLED",
                message="The local answer model is required for constrained generation.",
                status_code=503,
            )

        evidence = self._extractor.extract(execution.query, execution.selected_chunks)

        def validate(answer: StructuredAnswer) -> None:
            self._validate_answerability(assessment, answer)
            self._citation_validator.validate(
                answer,
                evidence=evidence,
                chunks=execution.selected_chunks,
                include_historical=request.filters.include_historical,
            )

        generation = await self._generator.generate(
            question=request.query,
            assessment=assessment,
            evidence=evidence,
            semantic_validator=validate,
        )
        answer = generation.answer
        used_citations = set(CITATION_MARKER.findall(answer.answer))
        used_citations.update(
            citation_id for claim in answer.claims for citation_id in claim.citation_ids
        )
        return ChatCompletionResponse(
            request_id=execution.request_id,
            conversation_id=conversation_id,
            status=answer.answerability,
            answer=answer.answer,
            claims=answer.claims,
            citations=[
                citation
                for citation in execution.citations
                if citation.citation_id in used_citations
            ],
            missing_information=self._unique(
                [*assessment.missing_requirements, *answer.missing_information]
            ),
            conflicts=self._unique(
                [
                    *(item.details for item in assessment.conflicting_citations),
                    *answer.conflicts,
                ]
            ),
            evidence_assessment=assessment,
            prompt_version=self._settings.generation.prompt_version,
            json_repaired=generation.repaired,
        )

    def _safe_response(
        self,
        *,
        execution: RetrievalExecution,
        assessment: EvidenceAssessment,
        conversation_id: str,
    ) -> ChatCompletionResponse:
        if assessment.status == "CONFLICTED":
            answer = "检索到的有效资料存在冲突，无法在不裁决冲突的情况下给出确定答案。"
            citation_ids = {
                citation_id
                for conflict in assessment.conflicting_citations
                for citation_id in conflict.citation_ids
            }
            citations = [
                item for item in execution.citations if item.citation_id in citation_ids
            ]
        else:
            answer = "现有资料中未找到足够证据，无法可靠回答该问题。"
            citations = []
        return ChatCompletionResponse(
            request_id=execution.request_id,
            conversation_id=conversation_id,
            status=assessment.status,
            answer=answer,
            claims=[],
            citations=citations,
            missing_information=assessment.missing_requirements,
            conflicts=[item.details for item in assessment.conflicting_citations],
            evidence_assessment=assessment,
            prompt_version=self._settings.generation.prompt_version,
        )

    @staticmethod
    def _validate_answerability(
        assessment: EvidenceAssessment,
        answer: StructuredAnswer,
    ) -> None:
        if (
            assessment.status == "PARTIALLY_ANSWERABLE"
            and answer.answerability == "ANSWERABLE"
        ):
            raise ValueError("the model cannot upgrade partial evidence to ANSWERABLE")
        if answer.answerability == "PARTIALLY_ANSWERABLE":
            missing = set(assessment.missing_requirements)
            if not missing.issubset(answer.missing_information):
                raise ValueError(
                    "the model omitted requirements marked missing by the Evidence Judge"
                )

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))


def citations_by_id(citations: list[Citation]) -> dict[str, Citation]:
    """Small typed helper retained for the stage-five verifier."""
    return {citation.citation_id: citation for citation in citations}
