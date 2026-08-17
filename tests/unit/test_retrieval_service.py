from __future__ import annotations

from typing import Any, cast

from app.core.config import Settings
from app.schemas.retrieval import (
    ChunkMetadata,
    RagflowRetrievalRequest,
    RerankResult,
    RetrievalSearchRequest,
    RetrievedChunk,
)
from app.services.access_control import AccessControlService
from app.services.ragflow_client import RagflowClient
from app.services.reranker_client import RerankerClient
from app.services.retrieval_service import RetrievalService


class FakeRagflow:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.request: RagflowRetrievalRequest | None = None

    async def retrieve(
        self,
        request: RagflowRetrievalRequest,
    ) -> list[RetrievedChunk]:
        self.request = request
        return self.chunks


class QueryAwareRagflow:
    def __init__(self, results: dict[str, list[RetrievedChunk]]) -> None:
        self.results = results
        self.requests: list[RagflowRetrievalRequest] = []

    async def retrieve(
        self,
        request: RagflowRetrievalRequest,
    ) -> list[RetrievedChunk]:
        self.requests.append(request)
        return self.results.get(request.question, [])


class FailingSubqueryRagflow:
    def __init__(self, fallback: list[RetrievedChunk]) -> None:
        self.fallback = fallback
        self.calls: list[str] = []

    async def retrieve(
        self,
        request: RagflowRetrievalRequest,
    ) -> list[RetrievedChunk]:
        self.calls.append(request.question)
        if request.question == "甲设备 波段":
            from app.core.exceptions import AppError

            raise AppError(code="TEST_FAILURE", message="subquery failed", status_code=503)
        if request.question.startswith("对比甲设备、乙设备"):
            return self.fallback
        return []


class FakeReranker:
    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        assert query == "任务周期是多少？"
        assert documents == ["有效证据一", "有效证据二"]
        assert top_n == 2
        return [
            RerankResult(index=1, score=0.98),
            RerankResult(index=0, score=0.76),
        ]


class MatrixReranker:
    def __init__(self) -> None:
        self.queries: list[str] = []

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        self.queries.append(query)
        score = 0.9 if "成像波段" in query else 0.01
        return [
            RerankResult(index=index, score=score)
            for index in range(min(top_n, len(documents)))
        ]


def chunk(
    chunk_id: str,
    *,
    dataset_id: str = "kb-1",
    status: str = "effective",
    text: str,
    score: float,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        dataset_id=dataset_id,
        text=text,
        metadata=ChunkMetadata(
            document_name=f"{chunk_id}.pdf",
            version="V2.3",
            source_path="技术资料/任务机",
            chapter_path="第4章/4.2",
            page_number=37,
            status=status,
        ),
        hybrid_score=score,
        vector_score=score - 0.1,
        keyword_score=1.0,
    )


async def test_pipeline_filters_before_rerank_and_assigns_citations(
    settings: Settings,
) -> None:
    fake_ragflow = FakeRagflow(
        [
            chunk("c1", text="有效证据一", score=0.90),
            chunk("c2", text="有效证据二", score=0.80),
            chunk("old", status="superseded", text="旧版证据", score=0.99),
            chunk(
                "foreign",
                dataset_id="kb-forbidden",
                text="无权证据",
                score=0.95,
            ),
        ]
    )
    retrieval_settings = settings.retrieval.model_copy(
        update={"rerank_top_k": 2, "max_final_chunks": 2}
    )
    configured = settings.model_copy(update={"retrieval": retrieval_settings})
    service = RetrievalService(
        settings=configured,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=cast(RerankerClient, cast(Any, FakeReranker())),
        access_control=AccessControlService(configured.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(
            query="任务周期是多少？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )
    response = execution.search_response()
    debug = execution.debug_response()

    assert [item.chunk_id for item in response.chunks] == ["c2", "c1"]
    assert [item.citation_id for item in response.chunks] == ["C1", "C2"]
    assert response.citations[0].document_name == "c2.pdf"
    assert response.reranker_used is True
    assert fake_ragflow.request is not None
    assert fake_ragflow.request.dataset_ids == ["kb-1"]
    assert fake_ragflow.request.metadata_condition is None
    reasons = {item.chunk_id: item.filter_reason for item in debug.candidates}
    assert reasons["old"] == "inactive_document"
    assert reasons["foreign"] == "unauthorized_dataset"


async def test_legacy_chunk_without_status_is_treated_as_effective(
    settings: Settings,
) -> None:
    legacy = chunk("legacy", text="旧发布链路未写入状态字段。", score=0.9)
    legacy.metadata.status = None
    fake_ragflow = FakeRagflow([legacy])
    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(settings.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(query="状态字段兼容性？", knowledge_base_ids=["kb-1"]),
        user_id="user-1",
    )

    assert execution.selected_chunks[0].chunk_id == "legacy"


async def test_exact_article_match_bypasses_similarity_threshold(
    settings: Settings,
) -> None:
    fake_ragflow = FakeRagflow(
        [
            chunk(
                "semantic",
                text="其他条款的高相似内容。",
                score=0.90,
            ),
            chunk(
                "article",
                text="条号：21.1(1)\n第21.1条第1款的准确内容。",
                score=0.05,
            ),
        ]
    )
    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(settings.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(
            query="第21.1条1的内容是什么？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert execution.selected_chunks[0].chunk_id == "article"
    assert execution.stage_counts["article_exact_matches"] == 1
    assert fake_ragflow.request is not None
    assert fake_ragflow.request.keyword is True
    assert "精确条号" in fake_ragflow.request.question


async def test_comparison_runs_subqueries_and_reports_missing_coverage(
    settings: Settings,
) -> None:
    question = (
        "对比植被多光谱遥感无人机、普通可见光巡检无人机，"
        "在成像波段、作业最佳时段上有什么不同？"
    )
    fake_ragflow = QueryAwareRagflow(
        {
            "植被多光谱遥感无人机 成像波段": [
                chunk("multi-band", text="多光谱包含近红外波段。", score=0.9)
            ],
            "普通可见光巡检无人机 成像波段": [
                chunk("rgb-band", text="可见光相机获取彩色影像。", score=0.8)
            ],
            "植被多光谱遥感无人机 作业最佳时段": [
                chunk("multi-time", text="特定监测任务要求在指定物候期作业。", score=0.7)
            ],
            "普通可见光巡检无人机 作业最佳时段": [],
        }
    )
    retrieval_settings = settings.retrieval.model_copy(
        update={
            "rerank_top_k": 4,
            "max_final_chunks": 4,
            "allow_rerank_fallback": True,
        }
    )
    configured = settings.model_copy(update={"retrieval": retrieval_settings})
    service = RetrievalService(
        settings=configured,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(configured.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(query=question, knowledge_base_ids=["kb-1"]),
        user_id="user-1",
    )

    assert len(fake_ragflow.requests) == 4
    assert execution.query_plan is not None
    assert execution.query_plan.requires_multi_query is True
    assert [cell.status for cell in execution.coverage_matrix] == [
        "covered",
        "covered",
        "covered",
        "missing",
    ]
    assert len(execution.selected_chunks) == 3
    assert execution.stage_counts["covered_cells"] == 3


async def test_complex_reranker_uses_cell_query_and_rejects_low_scores(
    settings: Settings,
) -> None:
    fake_ragflow = QueryAwareRagflow(
        {
            "甲设备 成像波段": [chunk("a-band", text="甲波段。", score=0.8)],
            "乙设备 成像波段": [chunk("b-band", text="乙波段。", score=0.8)],
            "甲设备 作业时段": [chunk("a-time", text="甲时段。", score=0.8)],
            "乙设备 作业时段": [chunk("b-time", text="乙时段。", score=0.8)],
        }
    )
    reranker = MatrixReranker()
    retrieval_settings = settings.retrieval.model_copy(
        update={"complex_min_rerank_score": 0.1}
    )
    configured = settings.model_copy(update={"retrieval": retrieval_settings})
    service = RetrievalService(
        settings=configured,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=cast(RerankerClient, cast(Any, reranker)),
        access_control=AccessControlService(configured.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(
            query="对比甲设备、乙设备，在成像波段、作业时段上有什么不同？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert reranker.queries == [
        "甲设备 成像波段",
        "乙设备 成像波段",
        "甲设备 作业时段",
        "乙设备 作业时段",
    ]
    assert [cell.status for cell in execution.coverage_matrix] == [
        "covered",
        "covered",
        "missing",
        "missing",
    ]
    assert {item.chunk_id for item in execution.selected_chunks} == {"a-band", "b-band"}


async def test_complex_retrieval_falls_back_to_original_query(
    settings: Settings,
) -> None:
    fake_ragflow = FailingSubqueryRagflow(
        [chunk("fallback", text="原问题的单次检索证据。", score=0.9)]
    )
    retrieval_settings = settings.retrieval.model_copy(
        update={"allow_rerank_fallback": True}
    )
    configured = settings.model_copy(update={"retrieval": retrieval_settings})
    service = RetrievalService(
        settings=configured,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(configured.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(
            query="对比甲设备、乙设备，在波段上有什么不同？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert execution.stage_counts["multi_query_fallback"] == 1
    assert execution.selected_chunks[0].chunk_id == "fallback"
    assert fake_ragflow.calls == [
        "甲设备 波段",
        "对比甲设备、乙设备，在波段上有什么不同？",
    ]
