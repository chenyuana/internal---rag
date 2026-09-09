from __future__ import annotations

import json
from typing import Any, cast

from app.core.config import Settings
from app.pipelines.rag_pipeline import RagPipeline
from app.schemas.chat import ChatCompletionRequest, ReferenceDocument
from app.schemas.models import AnswerModelSelection
from app.schemas.planning import PlannedCell, QueryPlanV2, RetrievalQuery
from app.schemas.retrieval import (
    ChunkMetadata,
    Citation,
    CoverageCell,
    NormalizedQuery,
    QueryPlan,
    RagflowRetrievalRequest,
    SelectedChunk,
    SubQuery,
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


class FakeAnswerModelManager:
    def __init__(self) -> None:
        self.selection: AnswerModelSelection | None = None

    async def resolve(
        self,
        selection: AnswerModelSelection,
        *,
        user_id: str,
    ) -> FakeAnswerModel:
        assert user_id == "user-1"
        self.selection = selection
        return FakeAnswerModel()


class FakeInvalidJsonAnswerModel:
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        assert messages
        assert response_schema["type"] == "object"
        return '{"answerability":"ANSWERABLE","answer":"unterminated'


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


def test_pipeline_attaches_temporary_document_to_evidence_and_citations(
    settings: Settings,
) -> None:
    retrieval_execution = execution(with_evidence=False)
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(RetrievalService, cast(Any, FakeRetrieval(retrieval_execution))),
        answer_model=None,
    )

    pipeline._attach_auxiliary_document(
        retrieval_execution,
        ReferenceDocument(
            name="补充说明.md",
            text="健康监控任务的执行周期为1000毫秒。",
        ),
    )

    assert retrieval_execution.selected_chunks[0].citation_id == "C1"
    assert retrieval_execution.citations[0].document_name == "补充说明.md（临时辅助资料）"
    assert retrieval_execution.stage_counts["auxiliary_document_chunks"] == 1


async def test_pipeline_refuses_without_evidence_before_model_call(
    settings: Settings,
) -> None:
    no_evidence = execution(with_evidence=False)
    no_evidence.query_plan_v2 = QueryPlanV2(
        query_type="fact",
        aspects=["执行周期"],
        cells=[
            PlannedCell(
                id="q1",
                aspect="执行周期",
                original_query="健康监控任务的执行周期是多少？",
                retrieval_queries=[
                    RetrievalQuery(
                        kind="original",
                        text="健康监控任务的执行周期是多少？",
                        generated_by="user",
                    )
                ],
            )
        ],
        synthesis_mode="direct",
        planner_source="deterministic",
        confidence=1.0,
    )
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(RetrievalService, cast(Any, FakeRetrieval(no_evidence))),
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
    assert result.diagnostics is not None
    assert result.diagnostics.cells[0].status == "not_specified"


class FakeRepairAnswerModel:
    """First attempt violates cross-regulation scope; repair is clean."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        assert messages
        assert response_schema["type"] == "object"
        self.calls += 1
        if self.calls == 1:
            return json.dumps(
                {
                    "answerability": "ANSWERABLE",
                    "answer": "根据《运输类飞机适航标准》第 25.981 条，"
                    "FRM 必须满足附录 M 的要求。[C4]",
                    "claims": [
                        {
                            "claim_id": "claim-1",
                            "claim": "根据第 25.981 条，FRM 必须满足附录 M 的要求",
                            "citation_ids": ["C4"],
                        }
                    ],
                    "missing_information": [],
                    "conflicts": [],
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "answerability": "ANSWERABLE",
                "answer": "根据第 25.981 条，燃油箱内不得存在点火源。[C1]",
                "claims": [
                    {
                        "claim_id": "claim-1",
                        "claim": "根据第 25.981 条，燃油箱内不得存在点火源",
                        "citation_ids": ["C1"],
                    }
                ],
                "missing_information": [],
                "conflicts": [],
            },
            ensure_ascii=False,
        )


def scope_execution() -> RetrievalExecution:
    query = QueryAnalyzer().analyze("对于运输类飞机燃油箱点燃防护的要求是什么？")
    chunks = [
        SelectedChunk(
            citation_id="C1",
            chunk_id="chunk-1",
            document_id="DOC-25",
            dataset_id="kb-1",
            text=(
                "文档：CCAR-25-R4 运输类飞机适航标准.pdf\n"
                "章节：第 25.981 条 燃油箱点燃防护\n"
                "条号：25.981\n"
                "燃油箱内不得存在点火源，机队平均可燃性暴露时间不得超过3%。"
            ),
            metadata=ChunkMetadata(
                document_name="CCAR-25-R4《运输类飞机适航标准》",
                version="R4",
                status="effective",
            ),
            hybrid_score=0.9,
            vector_score=0.8,
            keyword_score=1,
        ),
        SelectedChunk(
            citation_id="C4",
            chunk_id="chunk-4",
            document_id="DOC-26",
            dataset_id="kb-1",
            text=(
                "文档：CCAR-26 运输类飞机的持续适航和安全改进规定.pdf\n"
                "章节：第 26.33 条 燃油箱可燃性\n"
                "条号：26.33\n"
                "设计更改 对于机队平均可燃性暴露水平超过7%的燃油箱，"
                "FRM必须满足附录M除M25.1外的所有要求。"
            ),
            metadata=ChunkMetadata(
                document_name="CCAR-26《运输类飞机的持续适航和安全改进规定》",
                version="R2",
                status="effective",
            ),
            hybrid_score=0.8,
            vector_score=0.7,
            keyword_score=1,
        ),
    ]
    return RetrievalExecution(
        request_id="request-scope",
        query=query,
        selected_chunks=chunks,
        citations=[
            Citation(
                citation_id="C1",
                chunk_id="chunk-1",
                document_id="DOC-25",
                document_name="CCAR-25-R4《运输类飞机适航标准》",
                version="R4",
                quote=chunks[0].text,
            ),
            Citation(
                citation_id="C4",
                chunk_id="chunk-4",
                document_id="DOC-26",
                document_name="CCAR-26《运输类飞机的持续适航和安全改进规定》",
                version="R2",
                quote=chunks[1].text,
            ),
        ],
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


async def test_pipeline_repairs_cross_regulation_scope_violation(
    settings: Settings,
) -> None:
    settings.generation.answer_mode = "llm"
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(
            RetrievalService,
            cast(Any, FakeRetrieval(scope_execution())),
        ),
        answer_model=cast(AnswerModel, FakeRepairAnswerModel()),
    )

    result = await pipeline.run(
        ChatCompletionRequest(
            query="对于运输类飞机燃油箱点燃防护的要求是什么？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert result.status == "ANSWERABLE"
    assert result.json_repaired is True
    assert [item.citation_id for item in result.citations] == ["C1"]
    assert "25.981" in result.answer


class FakeDegradingAnswerModel:
    """Both attempts keep one grounded claim and one cross-regulation claim."""

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
                "answer": (
                    "根据 CCAR-25-R4 第25.981条，燃油箱内不得存在点火源，"
                    "机队平均可燃性暴露时间不得超过3%。[C1]"
                    "根据 CCAR-25-R4 第25.981条，FRM必须满足附录M要求。[C4]"
                ),
                "claims": [
                    {
                        "claim_id": "grounded-claim",
                        "claim": (
                            "根据 CCAR-25-R4 第25.981条，燃油箱内不得存在点火源，"
                            "机队平均可燃性暴露时间不得超过3%"
                        ),
                        "citation_ids": ["C1"],
                    },
                    {
                        "claim_id": "cross-scope-claim",
                        # 条号只在总答案前文声明，复现真实模型输出；逐 Claim
                        # 校验仍必须继承该范围，不能让 C4 绕过。
                        "claim": "FRM必须满足附录M要求",
                        "citation_ids": ["C4"],
                    },
                ],
                "missing_information": [],
                "conflicts": [],
            },
            ensure_ascii=False,
        )


async def test_pipeline_drops_invalid_claim_after_failed_model_repair(
    settings: Settings,
) -> None:
    settings.generation.answer_mode = "llm"
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(
            RetrievalService,
            cast(Any, FakeRetrieval(scope_execution())),
        ),
        answer_model=cast(AnswerModel, FakeDegradingAnswerModel()),
    )

    result = await pipeline.run(
        ChatCompletionRequest(
            query="运输类飞机燃油箱点燃防护的要求是什么？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert result.status == "PARTIALLY_ANSWERABLE"
    assert result.validation_degraded is True
    assert result.json_repaired is True
    assert [claim.claim_id for claim in result.claims] == ["grounded-claim"]
    assert [citation.citation_id for citation in result.citations] == ["C1"]
    assert "3%" in result.answer
    assert "FRM" not in result.answer
    assert "cross-scope-claim" in " ".join(result.validation_warnings)


class FakeAllInvalidAnswerModel:
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
                "answer": "根据 CCAR-25-R4 第25.981条，暴露水平不得超过7%。[C1]",
                "claims": [
                    {
                        "claim_id": "wrong-number",
                        "claim": "根据 CCAR-25-R4 第25.981条，暴露水平不得超过7%",
                        "citation_ids": ["C1"],
                    }
                ],
                "missing_information": [],
                "conflicts": [],
            },
            ensure_ascii=False,
        )


async def test_pipeline_returns_safe_unanswerable_when_all_claims_fail(
    settings: Settings,
) -> None:
    settings.generation.answer_mode = "llm"
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(
            RetrievalService,
            cast(Any, FakeRetrieval(scope_execution())),
        ),
        answer_model=cast(AnswerModel, FakeAllInvalidAnswerModel()),
    )

    result = await pipeline.run(
        ChatCompletionRequest(
            query="运输类飞机燃油箱点燃防护的要求是什么？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert result.status == "UNANSWERABLE"
    assert result.validation_degraded is True
    assert result.claims == []
    assert result.citations == []
    assert "暂不能给出可靠结论" in result.answer


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


async def test_procedure_answer_validates_all_builder_chunks_not_sentence_budget(
    settings: Settings,
) -> None:
    """A later archive citation must not become a 500 when prompt evidence is capped."""
    question = (
        "采用无人机开展松材线虫病监测，从航摄规划到成果归档完整操作流程"
        "包含哪些关键环节？各环节核心工作是什么？"
    )
    document = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    bodies = [
        "5 航摄规划\n5.1 航摄范围\n根据疫情信息确定航摄范围。",
        "6 作业准备\n6.1 设备调试\n检查飞行平台、电池和通信。",
        "7 影像获取\n7.1 航拍作业\n执行航线并检查影像质量。",
        "8 数据处理\n8.1 影像处理\n制作正射影像并提取异常木。",
        "9 地面验证\n9.1 现场核查\n对异常木定位点开展核查。",
        "10 档案管理\n10.1 成果归档\n原始影像和处理成果归档保存。",
    ]
    chunks = [
        SelectedChunk(
            citation_id=f"C{index}",
            chunk_id=f"procedure-{index}",
            document_id="doc-pine",
            dataset_id="kb-1",
            text=f"文档：{document}\n\n{body}",
            metadata=ChunkMetadata(document_name=document, status="effective"),
            hybrid_score=0.9,
            vector_score=0.8,
            keyword_score=1.0,
        )
        for index, body in enumerate(bodies, start=1)
    ]
    query = QueryAnalyzer().analyze(question)
    procedure_execution = RetrievalExecution(
        request_id="request-procedure",
        query=query,
        selected_chunks=chunks,
        citations=[
            Citation(
                citation_id=chunk.citation_id,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                document_name=document,
                quote=chunk.text,
            )
            for chunk in chunks
        ],
        candidate_count=len(chunks),
        reranker_used=True,
        reranker_fallback=False,
        ragflow_request=RagflowRetrievalRequest(question=question, dataset_ids=["kb-1"]),
        stage_counts={"final_selected": len(chunks)},
        candidates=[],
    )
    capped_settings = settings.model_copy(
        update={
            "generation": settings.generation.model_copy(
                update={"max_evidence_sentences": 1}
            )
        }
    )
    pipeline = RagPipeline(
        settings=capped_settings,
        retrieval=cast(
            RetrievalService,
            cast(Any, FakeRetrieval(procedure_execution)),
        ),
        answer_model=cast(AnswerModel, FakeAnswerModel()),
    )

    result = await pipeline.run(
        ChatCompletionRequest(query=question, knowledge_base_ids=["kb-1"]),
        user_id="user-1",
    )

    assert result.status == "ANSWERABLE"
    assert "档案管理" in result.answer
    assert "[C6]" in result.answer
    assert [citation.citation_id for citation in result.citations] == [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
    ]


async def test_pipeline_uses_requested_runtime_answer_model(settings: Settings) -> None:
    manager = FakeAnswerModelManager()
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(RetrievalService, cast(Any, FakeRetrieval(execution(with_evidence=True)))),
        answer_model=None,
        answer_models=cast(Any, manager),
    )

    result = await pipeline.run(
        ChatCompletionRequest(
            query="健康监控任务的执行周期是多少？",
            knowledge_base_ids=["kb-1"],
            model=AnswerModelSelection(source_id="api-example", model_name="model-x"),
        ),
        user_id="user-1",
    )

    assert manager.selection == AnswerModelSelection(
        source_id="api-example",
        model_name="model-x",
    )
    assert result.model_source_id == "api-example"
    assert result.model_name == "model-x"


def comparison_execution() -> RetrievalExecution:
    question = "甲设备与乙设备在设备选型上有什么区别？"
    query = NormalizedQuery(
        original_query=question,
        normalized_query=question,
        query_type="comparison",
    )
    plan = QueryPlan(
        query_type="comparison",
        subjects=["甲设备", "乙设备"],
        aspects=["设备选型"],
        subqueries=[
            SubQuery(id="sq-1", query="甲设备 设备选型", subject="甲设备", aspect="设备选型"),
            SubQuery(id="sq-2", query="乙设备 设备选型", subject="乙设备", aspect="设备选型"),
        ],
        synthesis_mode="matrix",
    )
    chunks = [
        SelectedChunk(
            citation_id="C1",
            chunk_id="chunk-a",
            document_id="doc-a",
            dataset_id="kb-1",
            text="甲设备应选用续航时间不少于30分钟的无人机。",
            metadata=ChunkMetadata(document_name="甲设备技术规范.pdf", status="effective"),
            hybrid_score=0.9,
            vector_score=0.8,
            keyword_score=1,
            subquery_id="sq-1",
        ),
        SelectedChunk(
            citation_id="C2",
            chunk_id="chunk-b",
            document_id="doc-b",
            dataset_id="kb-1",
            text="乙设备应选用载荷能力不少于2千克的无人机。",
            metadata=ChunkMetadata(document_name="乙设备技术规范.pdf", status="effective"),
            hybrid_score=0.9,
            vector_score=0.8,
            keyword_score=1,
            subquery_id="sq-2",
        ),
    ]
    citations = [
        Citation(
            citation_id=item.citation_id,
            chunk_id=item.chunk_id,
            document_id=item.document_id,
            document_name=item.metadata.document_name,
            quote=item.text,
        )
        for item in chunks
    ]
    return RetrievalExecution(
        request_id="request-comparison",
        query=query,
        selected_chunks=chunks,
        citations=citations,
        candidate_count=2,
        reranker_used=True,
        reranker_fallback=False,
        ragflow_request=RagflowRetrievalRequest(question=question, dataset_ids=["kb-1"]),
        stage_counts={"final_selected": 2},
        candidates=[],
        query_plan=plan,
        coverage_matrix=[
            CoverageCell(
                subquery_id="sq-1",
                subject="甲设备",
                aspect="设备选型",
                status="covered",
                chunk_ids=["chunk-a"],
                citation_ids=["C1"],
            ),
            CoverageCell(
                subquery_id="sq-2",
                subject="乙设备",
                aspect="设备选型",
                status="covered",
                chunk_ids=["chunk-b"],
                citation_ids=["C2"],
            ),
        ],
    )


async def test_comparison_falls_back_to_matrix_when_model_json_is_invalid(
    settings: Settings,
) -> None:
    settings = settings.model_copy(
        update={
            "generation": settings.generation.model_copy(
                update={"comparison_summary_enabled": True}
            )
        }
    )
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(
            RetrievalService,
            cast(Any, FakeRetrieval(comparison_execution())),
        ),
        answer_model=cast(AnswerModel, FakeInvalidJsonAnswerModel()),
    )

    result = await pipeline.run(
        ChatCompletionRequest(
            query="甲设备与乙设备在设备选型上有什么区别？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    assert result.status == "ANSWERABLE"
    assert result.validation_degraded is True
    assert "| 对比项目 | 甲设备 | 乙设备 |" in result.answer
    assert [item.citation_id for item in result.citations] == ["C1", "C2"]
    assert "模型总结不可用" in " ".join(result.validation_warnings)
    assert result.diagnostics is not None
    assert result.diagnostics.model_call_breakdown == {"summary": 1}


async def test_single_subject_comparison_plan_does_not_500(
    settings: Settings,
) -> None:
    """回归（2026-08-24）：query_type=comparison 但 plan 只有 1 个主体时，
    矩阵构建必然失败（build 要求 subjects>=2）。此前 is_matrix_first 只看
    query_type，导致矩阵构建失败直接抛 500；现在 is_matrix_first 收紧为
    subjects>=2 且 aspects 非空，单主体题回退 extractive 路径，不抛 500。"""
    execution = comparison_execution()
    execution.query_plan = execution.query_plan.model_copy(
        update={"subjects": ["甲设备"]}
    )
    execution.coverage_matrix = [
        cell for cell in execution.coverage_matrix if cell.subject == "甲设备"
    ]
    pipeline = RagPipeline(
        settings=settings,
        retrieval=cast(
            RetrievalService,
            cast(Any, FakeRetrieval(execution)),
        ),
        answer_model=cast(AnswerModel, FakeAnswerModel()),
    )

    result = await pipeline.run(
        ChatCompletionRequest(
            query="甲设备在设备选型上有什么要求？",
            knowledge_base_ids=["kb-1"],
        ),
        user_id="user-1",
    )

    # 不抛 COMPARISON_MATRIX_BUILD_FAILED，返回正常答案（提取式或生成式）
    assert result.status in {"ANSWERABLE", "PARTIALLY_ANSWERABLE", "UNANSWERABLE"}
    assert result.diagnostics is not None
    assert result.diagnostics.stage_counts.get("matrix_build_failed", 0) in {0, 1}
