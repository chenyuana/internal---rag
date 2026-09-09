from __future__ import annotations

from typing import Any, cast

from app.core.config import Settings
from app.schemas.planning import PlannedCell, RetrievalQuery
from app.schemas.retrieval import (
    ChunkMetadata,
    NormalizedQuery,
    RagflowRetrievalRequest,
    RerankResult,
    RetrievalSearchRequest,
    RetrievedChunk,
    SelectedChunk,
)
from app.services.access_control import AccessControlService
from app.services.ragflow_client import RagflowClient
from app.services.reranker_client import RerankerClient
from app.services.retrieval_service import RetrievalExecution, RetrievalService


def test_cell_retrieval_query_does_not_append_taxonomy_keyword_bag() -> None:
    original = "地质灾害倾斜摄影测量成果验收标准要求"
    cell = PlannedCell(
        id="q1",
        subject="地质灾害倾斜摄影测量",
        aspect="成果验收",
        original_query=original,
        retrieval_queries=[
            RetrievalQuery(kind="original", text=original, generated_by="user")
        ],
    )

    assert RetrievalService._cell_retrieval_query(cell) == original
    secondary = RetrievalService._secondary_cell_retrieval_query(cell)
    assert secondary is not None
    assert secondary.startswith("地质灾害倾斜摄影测量 ")
    assert "表2" not in secondary
    assert "表3" not in secondary


def test_subject_discovery_query_uses_specialist_vocabulary() -> None:
    forestry = RetrievalService._subject_discovery_query("林地病虫害喷洒无人机")
    rice = RetrievalService._subject_discovery_query("水稻植保无人机")

    assert "林业 有害生物 喷洒 防治" in forestry
    assert "水稻 病虫害 施药 稻纵卷叶螟" in rice
    assert RetrievalService._subject_discovery_query("甲设备") == "甲设备"


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
        if request.question in {"甲设备", "乙设备"}:
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


class LowScoreReranker:
    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        return [
            RerankResult(index=index, score=0.09 - index * 0.01)
            for index in range(min(top_n, len(documents)))
        ]


def chunk(
    chunk_id: str,
    *,
    dataset_id: str = "kb-1",
    status: str = "effective",
    text: str,
    score: float,
    document_name: str | None = None,
    document_id: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id or f"doc-{chunk_id}",
        dataset_id=dataset_id,
        text=text,
        metadata=ChunkMetadata(
            document_name=document_name or f"{chunk_id}.pdf",
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


async def test_simple_retrieval_rejects_candidates_below_rerank_floor(
    settings: Settings,
) -> None:
    fake_ragflow = FakeRagflow(
        [
            chunk("generic-1", text="泛化的无人机巡检定义", score=0.8),
            chunk("generic-2", text="泛化的设备验收要求", score=0.7),
        ]
    )
    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=cast(RerankerClient, cast(Any, LowScoreReranker())),
        access_control=AccessControlService(settings.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(
            query="火星地下隧道巡检无人机的量子雷达验收阈值是多少？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert execution.selected_chunks == []
    assert execution.stage_counts["final_selected"] == 0
    assert all(
        item.filter_reason == "reranker_not_returned"
        for item in execution.candidates
    )


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
    assert fake_ragflow.request.keyword is False
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
            "植被多光谱遥感无人机": [
                chunk(
                    "multi-discovery",
                    text="植被多光谱遥感无人机规程。",
                    score=0.9,
                    document_name="《植被多光谱遥感无人机巡检技术规程》.pdf",
                    document_id="doc-multi",
                )
            ],
            "普通可见光巡检无人机": [
                chunk(
                    "rgb-discovery",
                    text="普通可见光巡检无人机规程。",
                    score=0.9,
                    document_name="《普通可见光巡检无人机作业规范》.pdf",
                    document_id="doc-rgb",
                )
            ],
            "植被多光谱遥感无人机 成像波段": [
                chunk(
                    "multi-band",
                    text="多光谱包含近红外波段。",
                    score=0.9,
                    document_name="《植被多光谱遥感无人机巡检技术规程》.pdf",
                    document_id="doc-multi",
                )
            ],
            "普通可见光巡检无人机 成像波段": [
                chunk(
                    "rgb-band",
                    text="可见光波段成像获取彩色影像。",
                    score=0.8,
                    document_name="《普通可见光巡检无人机作业规范》.pdf",
                    document_id="doc-rgb",
                )
            ],
            "植被多光谱遥感无人机 作业最佳时段": [
                chunk(
                    "multi-time",
                    text="最佳作业时段为指定物候期。",
                    score=0.7,
                    document_name="《植被多光谱遥感无人机巡检技术规程》.pdf",
                    document_id="doc-multi",
                )
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

    # V2 路径按主体各发现一次文档，再对四个单元格做 scoped 检索。
    assert len(fake_ragflow.requests) == 6
    scoped = [r for r in fake_ragflow.requests if r.document_ids]
    assert len(scoped) == 4
    assert execution.query_plan is not None
    assert execution.query_plan.requires_multi_query is True
    assert [cell.status for cell in execution.coverage_matrix] == [
        "covered",
        "covered",
        "covered",
        "not_specified",
    ]
    assert len(execution.selected_chunks) == 3
    assert execution.stage_counts["covered_cells"] == 3


async def test_section_family_expansion_does_not_enable_llm_keyword_extraction(
    settings: Settings,
) -> None:
    document_name = "《长大桥梁无人机精细化巡检技术规程》.pdf"
    fake_ragflow = QueryAwareRagflow(
        {
            "9.4 巡检成果": [
                chunk(
                    "family-child",
                    text="9.4.1 巡检报告应包含检测内容与方法。",
                    score=0.9,
                    document_name=document_name,
                    document_id="doc-heading",
                )
            ]
        }
    )
    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(settings.access_control),
    )
    heading = SelectedChunk(
        citation_id="C1",
        chunk_id="heading",
        document_id="doc-heading",
        dataset_id="kb-1",
        text="文档：长大桥梁\n章节：9 成果 / 9.4 巡检成果\n\n9.4 巡检成果",
        metadata=ChunkMetadata(document_name=document_name, status="effective"),
        hybrid_score=0.8,
        vector_score=0.7,
        keyword_score=0.9,
    )
    execution = RetrievalExecution(
        request_id="request-1",
        query=NormalizedQuery(
            original_query="成果验收",
            normalized_query="成果验收",
            query_type="comparison",
        ),
        selected_chunks=[
            heading,
            heading.model_copy(
                update={
                    "chunk_id": "heading-2",
                    "text": "文档：长大桥梁\n章节：8 成果 / 8.1 质量检查\n\n8.1 质量检查",
                }
            ),
            heading.model_copy(
                update={
                    "chunk_id": "heading-3",
                    "text": "文档：长大桥梁\n章节：7 成果 / 7.2 报告编制\n\n7.2 报告编制",
                }
            ),
        ],
        citations=[],
        candidate_count=1,
        reranker_used=False,
        reranker_fallback=False,
        ragflow_request=RagflowRetrievalRequest(
            question="成果验收",
            dataset_ids=["kb-1"],
        ),
        stage_counts={},
        candidates=[],
    )

    calls = await service._expand_section_family(
        execution,
        subject_document_ids=["doc-heading"],
        subject="长大桥梁无人机精细化巡检",
    )

    assert calls == 2
    assert len(fake_ragflow.requests) == 2
    assert fake_ragflow.requests[0].question == "9.4 巡检成果"
    assert [request.keyword for request in fake_ragflow.requests] == [False, False]
    assert [item.chunk_id for item in execution.selected_chunks] == [
        "heading",
        "heading-2",
        "heading-3",
        "family-child",
    ]


async def test_complex_reranker_uses_cell_query_and_rejects_low_scores(
    settings: Settings,
) -> None:
    fake_ragflow = QueryAwareRagflow(
        {
            "甲设备": [
                chunk(
                    "a-discovery",
                    text="甲设备规程。",
                    score=0.9,
                    document_name="《甲设备巡检技术规程》.pdf",
                    document_id="doc-a",
                )
            ],
            "乙设备": [
                chunk(
                    "b-discovery",
                    text="乙设备规程。",
                    score=0.9,
                    document_name="《乙设备巡检技术规程》.pdf",
                    document_id="doc-b",
                )
            ],
            "甲设备 成像波段": [
                chunk(
                    "a-band",
                    text="甲波段。",
                    score=0.8,
                    document_name="《甲设备巡检技术规程》.pdf",
                    document_id="doc-a",
                )
            ],
            "乙设备 成像波段": [
                chunk(
                    "b-band",
                    text="乙波段。",
                    score=0.8,
                    document_name="《乙设备巡检技术规程》.pdf",
                    document_id="doc-b",
                )
            ],
            "甲设备 作业时段": [
                chunk(
                    "a-time",
                    text="甲时段。",
                    score=0.8,
                    document_name="《甲设备巡检技术规程》.pdf",
                    document_id="doc-a",
                )
            ],
            "乙设备 作业时段": [
                chunk(
                    "b-time",
                    text="乙时段。",
                    score=0.8,
                    document_name="《乙设备巡检技术规程》.pdf",
                    document_id="doc-b",
                )
            ],
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
        "not_specified",
        "not_specified",
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
    assert fake_ragflow.calls[-1] == "对比甲设备、乙设备，在波段上有什么不同？"
    assert "甲设备" in fake_ragflow.calls


async def test_v2_comparison_resolves_subject_once_and_blocks_high_score_wrong_document(
    settings: Settings,
) -> None:
    fake_ragflow = QueryAwareRagflow(
        {
            "甲设备": [
                chunk(
                    "a-discovery",
                    text="甲设备规程目录。",
                    score=0.8,
                    document_name="《甲设备技术规程》",
                    document_id="doc-a",
                )
            ],
            "乙设备": [
                chunk(
                    "b-discovery",
                    text="乙设备规程目录。",
                    score=0.8,
                    document_name="《乙设备技术规程》",
                    document_id="doc-b",
                )
            ],
            "甲设备 成像波段": [
                chunk(
                    "wrong-high",
                    text="乙设备采用红外成像波段。",
                    score=0.99,
                    document_name="《乙设备技术规程》",
                    document_id="doc-b",
                ),
                chunk(
                    "a-band",
                    text="甲设备采用可见光成像波段。",
                    score=0.75,
                    document_name="《甲设备技术规程》",
                    document_id="doc-a",
                ),
            ],
            "乙设备 成像波段": [
                chunk(
                    "b-band",
                    text="乙设备采用红外成像波段。",
                    score=0.8,
                    document_name="《乙设备技术规程》",
                    document_id="doc-b",
                )
            ],
        }
    )
    configured = settings.model_copy(
        update={"planning": settings.planning.model_copy(update={"shadow_mode": False})}
    )
    service = RetrievalService(
        settings=configured,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(configured.access_control),
    )
    execution = await service.execute(
        RetrievalSearchRequest(
            query="比较甲设备和乙设备，在成像波段上有什么差异？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )
    assert {item.document_id for item in execution.selected_chunks} == {"doc-a", "doc-b"}
    assert "wrong-high" not in {item.chunk_id for item in execution.selected_chunks}
    assert execution.stage_counts["subject_discovery_calls"] == 2
    assert execution.stage_counts["coverage_cells"] == 2
    # 发现检索分数（0.8）明显领先时走 retrieval 通道；title 通道仅在
    # 检索分不分胜负时兜底。两种通道最终选中的文档一致。
    assert {item.source for item in execution.subject_resolutions} <= {"retrieval", "title"}


async def test_forestry_rice_comparison_resolves_two_specialist_documents(
    settings: Settings,
) -> None:
    class SpecialistComparisonRagflow:
        async def retrieve(
            self, request: RagflowRetrievalRequest
        ) -> list[RetrievedChunk]:
            if not request.document_ids:
                if "林业 有害生物 喷洒 防治" in request.question:
                    return [
                        chunk(
                            "forest-discovery",
                            text="林业有害生物喷洒防治作业要求。",
                            score=0.462,
                            document_name="20240705：《无人机喷洒防治林业有害生物技术规程》.pdf",
                            document_id="doc-forest",
                        ),
                        chunk(
                            "generic-forest",
                            text="通用植保无人机要求。",
                            score=0.39,
                            document_name="20240723：《植保无人机飞防农作物病虫害技术规范》.pdf",
                            document_id="doc-generic",
                        ),
                    ]
                if "水稻 病虫害 施药 稻纵卷叶螟" in request.question:
                    return [
                        chunk(
                            "rice-discovery",
                            text="无人机防治稻纵卷叶螟施药要求。",
                            score=0.426,
                            document_name="20240821：《无人机防治稻纵卷叶螟施药技术规范》.pdf",
                            document_id="doc-rice",
                        ),
                        chunk(
                            "generic-rice",
                            text="通用植保无人机要求。",
                            score=0.39,
                            document_name="20240723：《植保无人机飞防农作物病虫害技术规范》.pdf",
                            document_id="doc-generic",
                        ),
                    ]
                return []
            if request.document_ids == ["doc-forest"]:
                return [
                    chunk(
                        "forest-parameters",
                        text=(
                            "根据不同林地状况、防治要求、防治对象、为害程度确定施药量。"
                            "平原地区作业高度为2 m~5 m，山地与坡地为2 m~10 m。"
                        ),
                        score=0.9,
                        document_name="20240705：《无人机喷洒防治林业有害生物技术规程》.pdf",
                        document_id="doc-forest",
                    )
                ]
            if request.document_ids == ["doc-rice"]:
                return [
                    chunk(
                        "rice-parameters",
                        text=(
                            "亩均施药量为2 L~3 L，依据作物种植密度和稻叶受害级别而定。"
                            "作业高度宜在作物冠层上方2 m~3 m范围内选择。"
                        ),
                        score=0.9,
                        document_name="20240821：《无人机防治稻纵卷叶螟施药技术规范》.pdf",
                        document_id="doc-rice",
                    )
                ]
            return []

    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, SpecialistComparisonRagflow())),
        reranker=None,
        access_control=AccessControlService(settings.access_control),
    )
    execution = await service.execute(
        RetrievalSearchRequest(
            query=(
                "对比林地病虫害喷洒无人机、水稻植保无人机，"
                "二者亩均施药量、作业高度有哪些明显差异？差异形成的原因是什么？"
            ),
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    resolutions = {item.subject: item.document_ids for item in execution.subject_resolutions}
    assert resolutions["林地病虫害喷洒无人机"] == ["doc-forest"]
    assert resolutions["水稻植保无人机"] == ["doc-rice"]
    assert {item.document_id for item in execution.selected_chunks} == {
        "doc-forest",
        "doc-rice",
    }
    parameter_cells = [
        cell for cell in execution.coverage_matrix if cell.aspect != "差异形成原因"
    ]
    assert len(parameter_cells) == 4
    assert all(cell.status == "covered" for cell in parameter_cells)


async def test_process_document_supplement_keeps_low_score_archive_section(
    settings: Settings,
) -> None:
    document_name = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    archive = chunk(
        "archive",
        text=(
            f"文档：{document_name}\n"
            "章节：10 档案管理\n条号：10 页码：9\n\n"
            "原始影像、处理后的影像及异常变色木数据应归档保存3年。"
        ),
        score=0.05,
        document_name=document_name,
        document_id="doc-process",
    )
    fake_ragflow = FakeRagflow([archive])
    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(settings.access_control),
    )
    seed = chunk(
        "planning",
        text=(
            f"文档：{document_name}\n"
            "章节：1 范围\n条号：1 页码：5\n\n"
            "1 范围 本文件规定了航摄规划、作业准备、影像获取、数据处理、"
            "地面验证、档案管理等要求。"
        ),
        score=0.9,
        document_name=document_name,
        document_id="doc-process",
    )
    accepted = [seed]
    seen = {(seed.document_id, " ".join(seed.text.split()))}

    await service._supplement_process_documents(
        accepted,
        seen=seen,
        query=service._query_analyzer.analyze(
            "从航摄规划到成果归档的完整操作流程包含哪些关键环节？"
        ),
        retrieval_question="从航摄规划到成果归档的完整操作流程",
        allowed_datasets=["kb-1"],
        candidate_top_k=30,
        similarity_threshold=None,
        metadata_values={"status": "effective"},
    )

    assert fake_ragflow.request is not None
    assert fake_ragflow.request.document_ids == ["doc-process"]
    assert fake_ragflow.request.similarity_threshold == 0.0
    assert [item.chunk_id for item in accepted] == ["planning", "archive"]


def test_process_topic_filter_keeps_same_title_family_and_drops_generic_aerial_docs() -> None:
    query = (
        "采用无人机开展松材线虫病监测，从航摄规划到成果归档完整操作流程"
        "包含哪些关键环节？各环节核心工作是什么？"
    )
    pine_2024 = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    pine_2025 = "20250721：《无人机监测松材线虫病致死松树技术规程》.pdf"
    generic = "20210510：《低空数字航空摄影测量外业规范》(报批稿).pdf"
    candidates = [
        chunk(
            "pine-2024",
            text="5 航摄规划",
            score=0.7,
            document_name=pine_2024,
            document_id="doc-pine-2024",
        ),
        chunk(
            "pine-2025",
            text="4 航拍准备",
            score=0.7,
            document_name=pine_2025,
            document_id="doc-pine-2025",
        ),
        chunk(
            "generic-aerial",
            text="低空数字航空摄影测量外业工作",
            score=0.9,
            document_name=generic,
            document_id="doc-generic",
        ),
    ]

    selected = RetrievalService._filter_process_topic_documents(candidates, query)

    assert {item.document_id for item in selected} == {
        "doc-pine-2024",
        "doc-pine-2025",
    }


def test_process_topic_filter_preserves_candidates_without_clear_title_match() -> None:
    candidates = [
        chunk(
            "a",
            text="准备工作",
            score=0.8,
            document_name="甲作业规范.pdf",
            document_id="doc-a",
        ),
        chunk(
            "b",
            text="成果检查",
            score=0.7,
            document_name="乙成果要求.pdf",
            document_id="doc-b",
        ),
    ]

    assert RetrievalService._filter_process_topic_documents(
        candidates,
        "完整流程是什么？",
    ) == candidates


def test_chunk_section_parses_leading_section_header() -> None:
    """章节路径按顶层章归并，保证流程章级覆盖。"""
    item = chunk(
        "c1",
        text=(
            "文档：20241217：《无人机监测松材线虫病致死松树技术规程》.pdf\n"
            "章节：6 数据处理 / 6.2 飞行控制\n"
            "条号：6.2 页码：8\n\n"
            "6 数据处理\n"
            "6.2 飞行控制\n"
            "无人机进入任务区域进行作业飞行。"
        ),
        score=0.9,
    )
    assert RetrievalService._chunk_section(item) == "chapter:6"


def test_chunk_section_infers_top_level_chapter_from_legacy_subsection() -> None:
    item = chunk(
        "c1",
        text="6.1 飞行平台组装和调试\n检查连接、电池与通信。",
        score=0.9,
    )

    assert RetrievalService._chunk_section(item) == "chapter:6"


def test_chunk_section_empty_without_header() -> None:
    item = chunk("c1", text="无章节头的正文内容。", score=0.9)
    assert RetrievalService._chunk_section(item) == ""


def test_diverse_by_section_keeps_two_per_top_level_process_chapter() -> None:
    """同章保留两个子节代表，不同文档/章节仍独立计数。"""
    from app.schemas.retrieval import RetrievedChunk

    def item(chunk_id: str, document_id: str, text: str, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            dataset_id="kb-1",
            text=text,
            metadata=ChunkMetadata(status="effective"),
            hybrid_score=score,
            rerank_score=score,
        )

    ranked = [
        item("c1", "doc-a", "文档：A.pdf\n章节：5 航摄规划\n条号：5.1\n\n5.1 航摄范围", 0.95),
        item("c2", "doc-a", "文档：A.pdf\n章节：5 航摄规划\n条号：5.2\n\n5.2 航摄设计", 0.94),
        item("c3", "doc-a", "文档：A.pdf\n章节：7 影像获取\n条号：7.2\n\n7.2 飞行控制", 0.90),
        item("c4", "doc-b", "文档：B.pdf\n章节：5 航拍作业\n条号：5.1\n\n5.1 航拍作业", 0.80),
    ]
    selected = RetrievalService._diverse_by_section(ranked, limit=4)
    assert [item.chunk_id for item in selected] == ["c1", "c2", "c3", "c4"]


def test_diverse_by_section_keeps_headerless_chunks() -> None:
    """无章节头的 chunk 不去重，按分数顺序进入。"""
    ranked = [
        chunk("c1", text="文档：A.pdf\n章节：5 航摄规划\n条号：5.1\n\n5.1 航摄范围", score=0.95),
        chunk("c2", text="术语和定义内容。", score=0.90),
        chunk("c3", text="另一个无章节头内容。", score=0.85),
    ]
    selected = RetrievalService._diverse_by_section(ranked, limit=3)
    assert [item.chunk_id for item in selected] == ["c1", "c2", "c3"]


def test_balanced_rerank_input_rotates_across_same_named_documents() -> None:
    """同名多规程：rerank 输入按文档均衡轮转，两份规程的流程章节都能进入。"""
    from app.schemas.retrieval import RetrievedChunk

    def item(
        chunk_id: str,
        document_name: str,
        section: str,
        score: float,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            dataset_id="kb-1",
            text=(
                f"文档：{document_name}\n"
                f"章节：{section}\n"
                f"条号：{section}\n\n"
                f"{section} 内容"
            ),
            metadata=ChunkMetadata(document_name=document_name, status="effective"),
            hybrid_score=score,
            rerank_score=score,
        )

    maoming = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    province = "20250721：《无人机监测松材线虫病致死松树技术规程》.pdf"
    accepted = [
        item("m1", maoming, "5 航摄规划", 0.99),
        item("m2", maoming, "7 影像获取", 0.98),
        item("m3", maoming, "10 档案管理", 0.97),
        item("p1", province, "5 航拍作业", 0.60),
        item("p2", province, "6 数据处理", 0.59),
        item("p3", province, "8 档案管理", 0.58),
        item("x1", "20241018：《常绿果树…》.pdf", "图1", 0.95),
    ]

    balanced = RetrievalService._balanced_rerank_input(accepted, limit=6)

    ids = [item.chunk_id for item in balanced]
    assert "p1" in ids and "p2" in ids, "省标准流程章节必须进入 rerank 输入"
    assert len(ids) <= 6


def test_balanced_final_chunks_allocates_quota_to_similar_documents() -> None:
    """同名多规程：最终证据按文档配额分配，省标准流程章节不被裁剪。"""
    from app.schemas.retrieval import RetrievedChunk

    def item(
        chunk_id: str,
        document_name: str,
        section: str,
        score: float,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            dataset_id="kb-1",
            text=(
                f"文档：{document_name}\n"
                f"章节：{section}\n"
                f"条号：{section}\n\n"
                f"{section} 内容"
            ),
            metadata=ChunkMetadata(document_name=document_name, status="effective"),
            hybrid_score=score,
            rerank_score=score,
        )

    maoming = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    province = "20250721：《无人机监测松材线虫病致死松树技术规程》.pdf"
    rerank_input = [
        item("m1", maoming, "5 航摄规划", 0.99),
        item("m2", maoming, "7 影像获取", 0.98),
        item("m3", maoming, "10 档案管理", 0.97),
        item("p1", province, "5 航拍作业", 0.60),
        item("p2", province, "6 数据处理", 0.59),
        item("p3", province, "8 档案管理", 0.58),
    ]

    final_chunks = RetrievalService._balanced_final_chunks(rerank_input, limit=5)

    ids = [item.chunk_id for item in final_chunks]
    assert len(ids) == 5
    assert "p1" in ids and "p2" in ids, "省标准流程章节必须进入最终证据"
    assert "m1" in ids and "m2" in ids


def test_balanced_final_chunks_prefers_relevant_documents() -> None:
    """最终证据：高相关文档流程章节优先占满名额，无关文档不占名额。"""
    from app.schemas.retrieval import RetrievedChunk

    def item(
        chunk_id: str,
        document_name: str,
        section: str,
        score: float,
        *,
        declared: bool = False,
    ) -> RetrievedChunk:
        prefix = ""
        if declared:
            prefix = (
                f"文档：{document_name}\n章节：1 范围\n\n"
                "1 范围 本文件规定了施药作业、药后处理、废弃物处理等要求。\n"
            )
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            dataset_id="kb-1",
            text=(
                prefix
                + f"文档：{document_name}\n"
                f"章节：{section}\n"
                f"条号：{section}\n\n"
                f"{section} 内容"
            ),
            metadata=ChunkMetadata(document_name=document_name, status="effective"),
            hybrid_score=score,
            rerank_score=score,
        )

    main = "20240723：《植保无人机飞防农作物病虫害技术规程》.pdf"
    second = "20240628：《农业植保无人机安全作业规范》.pdf"
    unrelated = "20241018：《无人机河湖智能巡查要求》.pdf"
    rerank_input = [
        item("a1", main, "5 施药作业", 0.99, declared=True),
        item("a2", main, "6 药后处理", 0.98, declared=True),
        item("a3", main, "7 废弃物处理", 0.97, declared=True),
        item("b1", second, "5 作业要求", 0.90, declared=True),
        item("b2", second, "6 安全规范", 0.89, declared=True),
        item("x1", unrelated, "5 巡查方式", 0.60),
    ]

    final_chunks = RetrievalService._balanced_final_chunks(rerank_input, limit=5)

    ids = [item.chunk_id for item in final_chunks]
    assert {"a1", "a2", "a3", "b1", "b2"} <= set(ids), "高相关文档流程章节必须优先进入"
    assert "x1" not in ids, "无关文档不应占用名额"


def test_process_section_floor_guarantees_later_process_chapters() -> None:
    """流程环节保底：每文档每顶层流程章节至少保留一个最高分代表。

    高分章节（术语/基本要求）不能把数据处理/地面验证/档案管理等后半段
    环节挤出 rerank 输入与最终证据。
    """
    from app.schemas.retrieval import RetrievedChunk

    def item(
        chunk_id: str,
        document_id: str,
        section: str,
        score: float,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            dataset_id="kb-1",
            text=(
                f"文档：{document_id}\n章节：{section}\n条号：{section}\n\n{section} 内容"
            ),
            metadata=ChunkMetadata(document_name=document_id, status="effective"),
            hybrid_score=score,
            rerank_score=score,
        )

    doc = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    # 首个 chunk 携带"1 范围/本文件规定了…"声明，将该文档标记为规程类
    # （流程环节保底仅对规程文档生效，避免无关文档任意章节占用名额）。
    scope = RetrievedChunk(
        chunk_id="scope",
        document_id=doc,
        dataset_id="kb-1",
        text=(
            "文档：20241217：《无人机监测松材线虫病致死松树技术规程》.pdf\n"
            "章节：1 范围\n\n"
            "1 范围 本文件规定了航摄规划、作业准备、影像获取、数据处理、"
            "地面验证、档案管理等要求。"
        ),
        metadata=ChunkMetadata(document_name=doc, status="effective"),
        hybrid_score=0.99,
        rerank_score=0.99,
    )
    chunks = [
        scope,
        # 高分但非流程章节：不应挤占保底名额
        item("t1", doc, "3 术语和定义", 0.99),
        item("t2", doc, "4 基本要求", 0.98),
        # 流程前半段
        item("p1", doc, "5 航摄规划", 0.97),
        item("p2", doc, "6 作业准备", 0.96),
        item("p3", doc, "7 影像获取", 0.95),
        # 流程后半段：低分但必须有代表
        item("p4", doc, "8 数据处理", 0.55),
        item("p5", doc, "9 地面验证", 0.54),
        item("p6", doc, "10 档案管理", 0.53),
        # 无章节头 chunk 不参与保底
        item("h1", doc, "无章节头内容", 0.90),
    ]
    chunks[-1] = RetrievedChunk(
        chunk_id="h1",
        document_id=doc,
        dataset_id="kb-1",
        text="无章节头的封面/术语内容。",
        metadata=ChunkMetadata(document_name=doc, status="effective"),
        hybrid_score=0.90,
        rerank_score=0.90,
    )

    floor = RetrievalService._process_section_floor(chunks)
    ids = {item.chunk_id for item in floor}

    assert "p4" in ids, "8 数据处理必须有保底代表"
    assert "p5" in ids, "9 地面验证必须有保底代表"
    assert "p6" in ids, "10 档案管理必须有保底代表"
    assert "t1" not in ids, "术语等非流程章节不参与保底"
    assert "h1" not in ids, "无章节头 chunk 不参与保底"
    assert "p1" in ids and "p2" in ids and "p3" in ids

    # 无"本文件规定了"声明的文档（如巡检要求）不参与流程保底
    unrelated = RetrievedChunk(
        chunk_id="x1",
        document_id="doc-river",
        dataset_id="kb-1",
        text="文档：20241018：《无人机河湖智能巡查要求》.pdf\n章节：5 巡查方式\n\n巡查方式 内容",
        metadata=ChunkMetadata(
            document_name="20241018：《无人机河湖智能巡查要求》.pdf",
            status="effective",
        ),
        hybrid_score=0.60,
        rerank_score=0.60,
    )
    floor2 = RetrievalService._process_section_floor([*chunks, unrelated])
    assert "x1" not in {item.chunk_id for item in floor2}, "无流程声明文档不参与保底"


def test_balanced_final_chunks_keeps_later_process_chapters_with_floor() -> None:
    """同名多规程最终证据：流程后半段章节在保底逻辑下进入最终证据。"""
    from app.schemas.retrieval import RetrievedChunk

    def item(
        chunk_id: str,
        document_name: str,
        document_id: str,
        section: str,
        score: float,
    ) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            dataset_id="kb-1",
            text=(
                f"文档：{document_name}\n章节：{section}\n条号：{section}\n\n{section} 内容"
            ),
            metadata=ChunkMetadata(document_name=document_name, status="effective"),
            hybrid_score=score,
            rerank_score=score,
        )

    maoming = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    province = "20250721：《无人机监测松材线虫病致死松树技术规程》.pdf"
    rerank_input = [
        item("m1", maoming, "doc-m", "5 航摄规划", 0.99),
        item("m2", maoming, "doc-m", "6 作业准备", 0.98),
        item("m3", maoming, "doc-m", "8 数据处理", 0.50),
        item("m4", maoming, "doc-m", "9 地面验证", 0.49),
        item("m5", maoming, "doc-m", "10 档案管理", 0.48),
        item("p1", province, "doc-p", "4 航拍准备", 0.60),
        item("p2", province, "doc-p", "5 航拍作业", 0.59),
        item("p3", province, "doc-p", "8 档案管理", 0.50),
    ]

    final_chunks = RetrievalService._balanced_final_chunks(rerank_input, limit=8)

    ids = [item.chunk_id for item in final_chunks]
    assert "m3" in ids, "20241217 数据处理必须进入最终证据"
    assert "m4" in ids, "20241217 地面验证必须进入最终证据"
    assert "m5" in ids, "20241217 档案管理必须进入最终证据"
    assert "p3" in ids, "20250721 档案管理必须进入最终证据"


def test_filter_declared_scope_keeps_only_matching_documents() -> None:
    """域过滤：声明类别外的文档（如马铃薯规范）不进生成证据。"""
    from app.schemas.retrieval import RetrievedChunk

    def item(chunk_id: str, document_name: str) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            dataset_id="kb-1",
            text=f"内容 {chunk_id}",
            metadata=ChunkMetadata(document_name=document_name, status="effective"),
            hybrid_score=0.9,
        )

    forest = "20240705：《无人机喷洒防治林业有害生物技术规程》.pdf"
    pine = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    potato = "20250122：《基于植保无人机的马铃薯病虫害药剂防治作业规范》.pdf"
    chunks = [item("a1", forest), item("b1", pine), item("p1", potato)]

    kept, excluded = RetrievalService._filter_declared_scope(
        chunks,
        scope_cores=["林业有害生物", "松材线虫病"],
    )

    assert [c.chunk_id for c in kept] == ["a1", "b1"]
    assert excluded == {"p1"}


def test_supplement_target_names_prefers_declared_scope_over_highest_score() -> None:
    """补充检索目标：声明类别内的文档优先，分数最高的范围外文档不作为目标。"""
    from app.schemas.retrieval import RetrievedChunk

    def item(chunk_id: str, document_name: str, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            dataset_id="kb-1",
            text=f"内容 {chunk_id}",
            metadata=ChunkMetadata(document_name=document_name, status="effective"),
            hybrid_score=score,
        )

    forest = "20240705：《无人机喷洒防治林业有害生物技术规程》.pdf"
    pine = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    potato = "20250122：《基于植保无人机的马铃薯病虫害药剂防治作业规范》.pdf"
    accepted = [
        item("f1", forest, 0.80),
        item("p1", pine, 0.75),
        item("pot1", potato, 0.99),  # 分数最高但是范围外文档
    ]

    targets = RetrievalService._supplement_target_names(
        accepted,
        scope_cores=["林业有害生物", "松材线虫病"],
    )

    assert targets == {forest, pine}
    assert potato not in targets

    # 无声明类别时维持原逻辑：取分数最高主文档
    targets_plain = RetrievalService._supplement_target_names(accepted, scope_cores=None)
    assert targets_plain == {potato}


async def test_multi_hop_with_declared_subjects_scopes_retrieval_per_document(
    settings: Settings,
) -> None:
    """跨文档综合题（沙地播种双重要求）每个 cell 按声明主体限定文档。

    回归 2026-08-21 沙地题：q1（技术参数）与 q2（运营管理）此前无 subject，
    全库混合检索导致水稻水直播/低空旅游等无关文档进入证据。现在每个 cell
    带声明主体，检索层必须先做主体发现、再按解析出的 document_ids 限定
    检索，第三文档被 document_scope_mismatch 过滤掉。
    """
    fake_ragflow = QueryAwareRagflow(
        {
            # 主体发现：声明主体词各自只命中目标文档。
            "沙地无人机播种治沙作业": [
                chunk(
                    "seed-discover",
                    text="无人机播种治沙技术规程。",
                    score=0.45,
                    document_name="无人机播种治沙技术规程",
                    document_id="doc-seed",
                )
            ],
            "无人机通用服务管理要求": [
                chunk(
                    "svc-discover",
                    text="无人机应用服务通用规范。",
                    score=0.43,
                    document_name="无人机应用服务通用规范",
                    document_id="doc-svc",
                )
            ],
            # 单元格限定检索：q1 的技术参数检索返回正确文档 + 无关高分文档，
            # 但 document_ids 限定后只有正确文档能通过。
            "沙地无人机播种治沙作业 技术参数": [
                chunk(
                    "seed-param",
                    text="航高10m～50m，速度3m/s～8m/s。",
                    score=0.9,
                    document_name="无人机播种治沙技术规程",
                    document_id="doc-seed",
                ),
                chunk(
                    "rice-intruder",
                    text="水稻丸粒化种子播量参数。",
                    score=0.99,
                    document_name="水稻丸粒化种子无人机水直播技术规程",
                    document_id="doc-rice",
                ),
            ],
            "无人机通用服务管理要求 运营管理": [
                chunk(
                    "svc-manage",
                    text="应建立完善的运营管理制度。",
                    score=0.9,
                    document_name="无人机应用服务通用规范",
                    document_id="doc-svc",
                ),
                chunk(
                    "tourism-intruder",
                    text="低空旅游驾驶人员要求。",
                    score=0.95,
                    document_name="低空旅游服务规范",
                    document_id="doc-tourism",
                ),
            ],
        }
    )
    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(settings.access_control),
    )
    execution = await service.execute(
        RetrievalSearchRequest(
            query=(
                "沙地无人机播种治沙作业，结合播种技术规范与无人机通用服务管理要求，"
                "完整项目实施需要兼顾技术参数与运营管理哪些双重要求？"
            ),
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )
    selected_docs = {item.document_id for item in execution.selected_chunks}
    assert selected_docs == {"doc-seed", "doc-svc"}
    assert "rice-intruder" not in {item.chunk_id for item in execution.selected_chunks}
    assert "tourism-intruder" not in {item.chunk_id for item in execution.selected_chunks}
    assert execution.stage_counts["subject_discovery_calls"] == 2
    assert {item.source for item in execution.subject_resolutions} == {"retrieval"}


class AnchorAwareRagflow:
    """首次（原问题，含"精确条号"标记）返回空以触发兜底；兜底请求（锚点组合）返回命中。"""

    def __init__(self, fallback: list[RetrievedChunk]) -> None:
        self.fallback = fallback
        self.requests: list[RagflowRetrievalRequest] = []

    async def retrieve(
        self,
        request: RagflowRetrievalRequest,
    ) -> list[RetrievedChunk]:
        self.requests.append(request)
        if "精确条号" in request.question:
            return []
        return self.fallback


class FailingFallbackRagflow:
    """首次零命中；兜底请求抛 RAGFLOW_UNAVAILABLE，用于验证兜底失败被记录。"""

    def __init__(self) -> None:
        self.requests: list[RagflowRetrievalRequest] = []

    async def retrieve(
        self,
        request: RagflowRetrievalRequest,
    ) -> list[RetrievedChunk]:
        self.requests.append(request)
        if "精确条号" in request.question:
            return []
        from app.core.exceptions import AppError

        raise AppError(
            code="RAGFLOW_UNAVAILABLE",
            message="ragflow unavailable",
            status_code=503,
        )


class InventoryAwareRagflow:
    """支持检索 + 文档清单（分页：第一页返回全部，第二页为空）。"""

    def __init__(
        self,
        chunks: list[RetrievedChunk],
        documents: list[str] | None = None,
    ) -> None:
        self.chunks = chunks
        self.documents = documents or ["CCAR-25-R4.pdf", "AC-25.981.pdf", "CCAR-23-R3.pdf"]
        self.list_calls: list[tuple[str, int]] = []

    async def retrieve(
        self,
        request: RagflowRetrievalRequest,
    ) -> list[RetrievedChunk]:
        return self.chunks

    async def list_documents(
        self,
        dataset_id: str,
        page: int = 1,
        page_size: int = 1024,
    ) -> list[dict[str, str]]:
        self.list_calls.append((dataset_id, page))
        if page == 1:
            return [
                {"id": f"doc-{index}", "name": name}
                for index, name in enumerate(self.documents)
            ]
        return []


async def test_anchor_fallback_recovers_zero_hit_retrieval(settings: Settings) -> None:
    fake_ragflow = AnchorAwareRagflow(
        [chunk("anchor-1", text="第25.981条 燃油系统防火要求。", score=0.85)]
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
            query="CCAR-25 第25.981条 燃油系统防火要求？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )
    response = execution.search_response()

    assert execution.deterministic_fallback is True
    assert execution.stage_counts.get("deterministic_fallback") == 1
    assert len(fake_ragflow.requests) == 2
    assert fake_ragflow.requests[1].keyword is True
    assert [item.chunk_id for item in execution.selected_chunks] == ["anchor-1"]
    # API 响应与证据级来源标记
    assert response.deterministic_fallback is True
    assert response.chunks[0].retrieval_stage == "deterministic_fallback"
    assert execution.candidates[0].retrieval_stage == "deterministic_fallback"


async def test_anchor_fallback_not_triggered_when_first_round_has_hits(
    settings: Settings,
) -> None:
    fake_ragflow = FakeRagflow([chunk("c1", text="有效证据一", score=0.90)])
    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(settings.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(
            query="CCAR-25 第25.981条 燃油系统防火要求？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert execution.deterministic_fallback is False
    assert execution.stage_counts.get("deterministic_fallback") is None
    assert [item.chunk_id for item in execution.selected_chunks] == ["c1"]


async def test_anchor_fallback_failure_is_recorded_as_failure(settings: Settings) -> None:
    fake_ragflow = FailingFallbackRagflow()
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
            query="CCAR-25 第25.981条 燃油系统防火要求？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )
    response = execution.search_response()

    assert execution.deterministic_fallback is False
    assert [failure.stage for failure in response.failures] == ["deterministic_fallback"]
    assert response.failures[0].error_code == "RAGFLOW_UNAVAILABLE"
    assert response.failures[0].retryable is True
    assert response.failures[0].dataset_ids == ["kb-1"]


async def test_inventory_question_builds_document_boundary(settings: Settings) -> None:
    fake_ragflow = InventoryAwareRagflow([chunk("c1", text="有效证据一", score=0.90)])
    service = RetrievalService(
        settings=settings,
        ragflow=cast(RagflowClient, cast(Any, fake_ragflow)),
        reranker=None,
        access_control=AccessControlService(settings.access_control),
    )

    execution = await service.execute(
        RetrievalSearchRequest(
            query="CCAR-25 有哪些版本？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )
    response = execution.search_response()

    assert execution.query.requires_inventory is True
    assert execution.doc_inventory is not None
    assert execution.doc_inventory["total"] == 3
    assert execution.doc_inventory["by_dataset"] == {
        "kb-1": ["CCAR-25-R4.pdf", "AC-25.981.pdf", "CCAR-23-R3.pdf"]
    }
    assert fake_ragflow.list_calls == [("kb-1", 1)]
    assert response.doc_inventory == execution.doc_inventory
    assert response.chunks[0].retrieval_stage is None
