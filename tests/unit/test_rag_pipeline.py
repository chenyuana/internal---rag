from __future__ import annotations

import json
from typing import Any, cast

from app.core.config import Settings
from app.pipelines.rag_pipeline import RagPipeline
from app.schemas.chat import ChatCompletionRequest
from app.schemas.retrieval import (
    ChunkMetadata,
    Citation,
    RagflowRetrievalRequest,
    SelectedChunk,
)
from app.services.answer_generator import AnswerModel
from app.services.query_analyzer import QueryAnalyzer
from app.services.retrieval_service import RetrievalExecution, RetrievalService


class FakeRetrieval:
    def __init__(self, execution: RetrievalExecution) -> None:
        self.execution = execution

    async def execute(
        self,
        request: object,
        *,
        user_id: str,
    ) -> RetrievalExecution:
        assert user_id == "user-1"
        return self.execution


class FakeAnswerModel:
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        assert messages
        assert response_schema["type"] == "object"
        return json.dumps(
            {
                "answerability": "ANSWERABLE",
                "answer": "健康监控任务每1000毫秒执行一次。[C1]",
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "claim": "健康监控任务每1000毫秒执行一次",
                        "citation_ids": ["C1"],
                    }
                ],
                "missing_information": [],
                "conflicts": [],
            },
            ensure_ascii=False,
        )


def execution(*, with_evidence: bool) -> RetrievalExecution:
    query = QueryAnalyzer().analyze("健康监控任务的执行周期是多少？")
    chunks = (
        [
            SelectedChunk(
                citation_id="C1",
                chunk_id="chunk-1",
                document_id="DOC-1",
                dataset_id="kb-1",
                text="健康监控任务的执行周期为1000毫秒。",
                metadata=ChunkMetadata(
                    document_name="设计说明书",
                    version="V2.3",
                    source_path="技术资料/任务机",
                    chapter_path="第4章/4.2",
                    page_number=37,
                    status="effective",
                ),
                hybrid_score=0.9,
                vector_score=0.8,
                keyword_score=1,
            )
        ]
        if with_evidence
        else []
    )
    citations = (
        [
            Citation(
                citation_id="C1",
                chunk_id="chunk-1",
                document_id="DOC-1",
                document_name="设计说明书",
                version="V2.3",
                source_path="技术资料/任务机",
                chapter_path="第4章/4.2",
                page_number=37,
                quote="健康监控任务的执行周期为1000毫秒。",
            )
        ]
        if with_evidence
        else []
    )
    return RetrievalExecution(
        request_id="request-1",
        query=query,
        selected_chunks=chunks,
        citations=citations,
        candidate_count=len(chunks),
        reranker_used=False,
        reranker_fallback=True,
        ragflow_request=RagflowRetrievalRequest(
            question=query.normalized_query,
            dataset_ids=["kb-1"],
        ),
        stage_counts={"final_selected": len(chunks)},
        candidates=[],
    )


async def test_pipeline_refuses_without_evidence_before_model_call(
    settings: Settings,
) -> None:
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(RetrievalService, cast(Any, FakeRetrieval(execution(with_evidence=False)))),
        answer_model=None,
    )

    result = await pipeline.run(
        ChatCompletionRequest(
            query="健康监控任务的执行周期是多少？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert result.status == "UNANSWERABLE"
    assert result.claims == []
    assert result.citations == []


async def test_pipeline_returns_only_backend_mapped_citations(
    settings: Settings,
) -> None:
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(RetrievalService, cast(Any, FakeRetrieval(execution(with_evidence=True)))),
        answer_model=cast(AnswerModel, FakeAnswerModel()),
    )

    result = await pipeline.run(
        ChatCompletionRequest(
            query="健康监控任务的执行周期是多少？",
            knowledge_base_ids=["kb-1"],
            conversation_id="conversation-1",
        ),
        user_id="user-1",
    )

    assert result.status == "ANSWERABLE"
    assert result.conversation_id == "conversation-1"
    assert result.citations[0].document_name == "设计说明书"
    assert result.citations[0].page_number == 37
    assert result.claims[0].citation_ids == ["C1"]
