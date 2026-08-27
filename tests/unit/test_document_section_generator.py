from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.chat import StructuredAnswer
from app.schemas.documents import (
    DocumentSectionGenerateRequest,
    GuidanceSuggestRequest,
    SectionGuidanceItem,
)
from app.schemas.retrieval import (
    ChunkMetadata,
    Citation,
    RagflowRetrievalRequest,
    RetrievalSearchRequest,
    SelectedChunk,
)
from app.services.document_section_generator import DocumentSectionGenerator
from app.services.query_analyzer import QueryAnalyzer
from app.services.retrieval_service import RetrievalExecution


class FakeRetrieval:
    def __init__(
        self,
        *,
        with_evidence: bool = True,
        include_irrelevant: bool = False,
    ) -> None:
        self.with_evidence = with_evidence
        self.include_irrelevant = include_irrelevant
        self.request: RetrievalSearchRequest | None = None

    async def execute(
        self,
        request: RetrievalSearchRequest,
        *,
        user_id: str,
    ) -> RetrievalExecution:
        assert user_id == "author-1"
        self.request = request
        query = QueryAnalyzer().analyze(request.query)
        chunks = []
        citations = []
        if self.with_evidence:
            text = (
                "文档：中高风险无人直升机系统适航标准（试行）\n"
                "章节：UHS.1529 持续适航文件\n"
                "持续适航文件必须包括发动机、旋翼以及适用设备的持续适航资料。"
            )
            chunks.append(
                SelectedChunk(
                    citation_id="C1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    dataset_id="kb-1",
                    text=text,
                    metadata=ChunkMetadata(
                        document_name="中高风险无人直升机系统适航标准（试行）",
                        status="effective",
                    ),
                    hybrid_score=0.9,
                    vector_score=0.8,
                    keyword_score=1.0,
                )
            )
            citations.append(
                Citation(
                    citation_id="C1",
                    chunk_id="chunk-1",
                    document_id="doc-1",
                    document_name="中高风险无人直升机系统适航标准（试行）",
                    quote="持续适航文件必须包括发动机、旋翼以及适用设备的持续适航资料。",
                )
            )
            if self.include_irrelevant:
                chunks.append(
                    SelectedChunk(
                        citation_id="C2",
                        chunk_id="chunk-2",
                        document_id="doc-2",
                        dataset_id="kb-1",
                        text="文档：HB 9102 航空产品首件检验要求\n零部件特性核查记录。",
                        metadata=ChunkMetadata(
                            document_name="HB 9102 航空产品首件检验要求",
                            status="effective",
                        ),
                        hybrid_score=0.8,
                        vector_score=0.7,
                        keyword_score=0.9,
                    )
                )
                citations.append(
                    Citation(
                        citation_id="C2",
                        chunk_id="chunk-2",
                        document_id="doc-2",
                        document_name="HB 9102 航空产品首件检验要求",
                        quote="零部件特性核查记录。",
                    )
                )
        return RetrievalExecution(
            request_id="document-request-1",
            query=query,
            selected_chunks=chunks,
            citations=citations,
            candidate_count=len(chunks),
            reranker_used=False,
            reranker_fallback=False,
            ragflow_request=RagflowRetrievalRequest(
                question=request.query,
                dataset_ids=request.knowledge_base_ids,
            ),
            stage_counts={},
            candidates=[],
        )


class CapturingModel:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        self.messages = messages
        assert response_schema["type"] == "object"
        return json.dumps(
            {
                "answerability": "PARTIALLY_ANSWERABLE",
                "answer": (
                    "本文件用于建立无人机适航要求与符合性证据的统一边界。"
                    "对于中高风险无人直升机构型，持续适航资料应覆盖发动机、"
                    "旋翼及适用设备。[C1]"
                ),
                "claims": [
                    {
                        "claim_id": "control-claim",
                        "claim": "本文档用于建立适航要求与符合性证据的统一边界",
                        "citation_ids": [],
                    },
                    {
                        "claim_id": "claim-1",
                        "claim": (
                            "中高风险无人直升机持续适航资料应覆盖发动机、"
                            "旋翼及适用设备"
                        ),
                        "citation_ids": ["C1"],
                    }
                ],
                "missing_information": ["无人机系统通用术语定义"],
                "conflicts": [],
            },
            ensure_ascii=False,
        )


class MixedValidityModel:
    def __init__(self) -> None:
        self.calls = 0

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        self.calls += 1
        return json.dumps(
            {
                "answerability": "PARTIALLY_ANSWERABLE",
                "answer": "持续适航资料应覆盖发动机和旋翼。[C1]另有未经证实的要求。[C99]",
                "claims": [
                    {
                        "claim_id": "valid-claim",
                        "claim": "持续适航资料应覆盖发动机和旋翼",
                        "citation_ids": ["C1"],
                    },
                    {
                        "claim_id": "invalid-claim",
                        "claim": "另有未经证实的要求",
                        "citation_ids": ["C99"],
                    },
                ],
                "missing_information": [],
                "conflicts": [],
            },
            ensure_ascii=False,
        )


class GuidanceModel:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.messages: list[dict[str, str]] = []

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        self.messages = messages
        assert response_schema["type"] == "object"
        return json.dumps(self.payload, ensure_ascii=False)


def request() -> DocumentSectionGenerateRequest:
    return DocumentSectionGenerateRequest(
        document_title="无人机适航",
        objective="形成适航要求与符合性验证的完整技术文件",
        audience="适航审定与研发团队",
        section_title="编制说明",
        section_guidance="说明目的、适用对象、文档边界、术语和证据使用原则。",
        knowledge_base_ids=["kb-1"],
    )


async def test_generates_section_with_dedicated_prompt_and_citations(
    settings: Settings,
) -> None:
    retrieval = FakeRetrieval()
    model = CapturingModel()
    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=retrieval,  # type: ignore[arg-type]
        answer_model=model,
    )

    result = await generator.generate(request(), user_id="author-1")

    assert result.status == "PARTIALLY_ANSWERABLE"
    assert result.prompt_version == "document-section-generation-v1"
    assert result.citations[0].citation_id == "C1"
    assert result.evidence_document_count == 1
    assert "普通问答助手" in model.messages[0]["content"]
    payload = json.loads(model.messages[1]["content"])
    assert "文档目标" in payload["question"]
    assert "编制说明" in payload["question"]
    assert retrieval.request is not None
    assert retrieval.request.candidate_top_k == 40


async def test_overview_filters_tangential_non_uav_documents(
    settings: Settings,
) -> None:
    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=FakeRetrieval(include_irrelevant=True),  # type: ignore[arg-type]
        answer_model=CapturingModel(),
    )

    result = await generator.generate(request(), user_id="author-1")

    assert result.evidence_document_count == 1
    assert [citation.citation_id for citation in result.citations] == ["C1"]


async def test_returns_missing_information_without_calling_model_when_no_evidence(
    settings: Settings,
) -> None:
    retrieval = FakeRetrieval(with_evidence=False)
    model = CapturingModel()
    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=retrieval,  # type: ignore[arg-type]
        answer_model=model,
    )

    result = await generator.generate(request(), user_id="author-1")

    assert result.status == "UNANSWERABLE"
    assert result.content == ""
    assert result.missing_information
    assert model.messages == []


async def test_preserves_valid_claims_when_one_long_form_claim_fails_validation(
    settings: Settings,
) -> None:
    model = MixedValidityModel()
    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=FakeRetrieval(),  # type: ignore[arg-type]
        answer_model=model,
    )

    result = await generator.generate(request(), user_id="author-1")

    assert result.status == "PARTIALLY_ANSWERABLE"
    assert result.validation_degraded is True
    assert "持续适航资料应覆盖发动机和旋翼" in result.content
    assert "未经证实" not in result.content
    assert [claim.claim_id for claim in result.claims] == ["valid-claim"]
    assert result.citations[0].citation_id == "C1"
    assert any("invalid-claim" in warning for warning in result.validation_warnings)


def test_quality_gate_rejects_short_and_scope_narrowed_detailed_section() -> None:
    payload = request().model_copy(
        update={
            "target_length": "detailed",
            "section_title": "适用范围与无人机系统概述",
        }
    )
    content = (
        "本章节旨在界定适用于《大型货运无人机系统通用要求》的航空器构型。"
        "本文档所述无人机系统特指大型货运无人机。"
    )

    passed, char_count, target_min, warnings, advisory = (
        DocumentSectionGenerator._quality_gate(
            payload,
            content,
            evidence_sentence_count=24,
        )
    )

    assert passed is False
    assert char_count < target_min
    assert target_min == 800
    assert any("范围收缩" in warning for warning in warnings)
    assert any("任务场景" in warning for warning in warnings)
    assert any("建议篇幅" in warning for warning in advisory)


def test_quality_gate_rejects_one_line_certification_basis() -> None:
    payload = request().model_copy(
        update={
            "target_length": "detailed",
            "section_title": "适航审定依据与审定基础",
        }
    )

    passed, _, _, warnings, _ = DocumentSectionGenerator._quality_gate(
        payload,
        "申请人必须提交符合适用的适航指令的声明和清单。",
    )

    assert passed is False
    assert any("适航标准" in warning for warning in warnings)
    assert any("审定基础" in warning for warning in warnings)


def test_quality_gate_short_but_complete_content_only_advises() -> None:
    payload = request().model_copy(
        update={"section_title": "编制说明"}
    )
    content = (
        "编制目的：建立无人机适航要求与符合性证据的统一边界，用于指导各章节编制。"
        "适用对象：面向适航审定与研发团队。"
        "文档边界：范围覆盖无人机系统适航要求与验证方法。"
        "术语：本章术语沿用 CCAR 与型号合格审定常用定义。"
        "证据原则：所有事实结论均引用检索证据并标注引用编号。"
    )

    passed, char_count, target_min, warnings, advisory = (
        DocumentSectionGenerator._quality_gate(
            payload,
            content,
            evidence_sentence_count=24,
        )
    )

    assert passed is True
    assert warnings == []
    assert char_count < target_min
    assert any("建议篇幅" in warning for warning in advisory)


def test_quality_gate_table_like_section_ignores_char_count() -> None:
    payload = request().model_copy(
        update={"section_title": "符合性验证矩阵"}
    )
    content = (
        "要求：符合适航标准。来源：CCAR-21。符合性方法：分析。"
        "条件输入：试验数据。判据：全部满足。记录状态：已记录。"
    )

    passed, _, _, warnings, advisory = (
        DocumentSectionGenerator._quality_gate(
            payload,
            content,
            evidence_sentence_count=0,
        )
    )

    assert passed is True
    assert warnings == []
    assert advisory == []


def test_length_target_scales_with_evidence_and_section_type() -> None:
    narrative = request().model_copy(
        update={"target_length": "standard", "section_title": "编制说明"}
    )
    matrix = request().model_copy(
        update={"target_length": "standard", "section_title": "符合性验证矩阵"}
    )

    assert DocumentSectionGenerator._length_target(narrative, 24) == 550
    assert DocumentSectionGenerator._length_target(narrative, 6) == 275
    assert DocumentSectionGenerator._length_target(narrative, 0) == 200
    assert DocumentSectionGenerator._length_target(matrix, 0) == 550


async def test_suggest_guidances_maps_model_output_back_to_input_titles(
    settings: Settings,
) -> None:
    model = GuidanceModel(
        {
            "sections": [
                {"title": "范围", "guidance": "界定适用范围、适用对象与边界条件。"},
                {"title": "测试环境", "guidance": "明确测试场地、设备与环境条件。"},
                {"title": "多余章节", "guidance": "模型不应新增章节。"},
            ]
        }
    )
    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=FakeRetrieval(),  # type: ignore[arg-type]
        answer_model=model,
    )
    payload = GuidanceSuggestRequest(
        document_title="测试文档",
        objective="完成验收测试",
        sections=[
            SectionGuidanceItem(title="范围"),
            SectionGuidanceItem(title="测试环境"),
        ],
    )

    result = await generator.suggest_guidances(payload, user_id="author-1")

    assert [item.title for item in result.sections] == ["范围", "测试环境"]
    assert result.sections[0].guidance == "界定适用范围、适用对象与边界条件。"
    assert result.sections[1].guidance == "明确测试场地、设备与环境条件。"
    assert "章节列表" in model.messages[1]["content"]


async def test_suggest_guidances_keeps_existing_guidance_when_model_omits_title(
    settings: Settings,
) -> None:
    model = GuidanceModel({"sections": [{"title": "范围", "guidance": "模型生成的新要求"}]})
    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=FakeRetrieval(),  # type: ignore[arg-type]
        answer_model=model,
    )
    payload = GuidanceSuggestRequest(
        document_title="测试文档",
        objective="完成验收测试",
        sections=[
            SectionGuidanceItem(title="范围"),
            SectionGuidanceItem(title="测试环境", guidance="原有要求"),
        ],
    )

    result = await generator.suggest_guidances(payload, user_id="author-1")

    assert result.sections[0].guidance == "模型生成的新要求"
    assert result.sections[1].guidance == "原有要求"


async def test_suggest_guidances_rejects_invalid_model_json(
    settings: Settings,
) -> None:
    class BrokenModel:
        async def chat_completion(
            self,
            *,
            messages: list[dict[str, str]],
            response_schema: dict[str, Any],
        ) -> str:
            return "not-json"

    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=FakeRetrieval(),  # type: ignore[arg-type]
        answer_model=BrokenModel(),
    )
    payload = GuidanceSuggestRequest(
        document_title="t",
        objective="o",
        sections=[SectionGuidanceItem(title="范围")],
    )

    with pytest.raises(AppError) as excinfo:
        await generator.suggest_guidances(payload, user_id="author-1")

    assert excinfo.value.code == "GUIDANCE_SUGGEST_INVALID_JSON"


def test_scope_validation_excludes_long_form_heading_numbers() -> None:
    structured = DocumentSectionGenerator._scope_validation_answer(
        StructuredAnswer(
            answerability="PARTIALLY_ANSWERABLE",
            answer="1.1 编制目的\n1.2 适用对象\n事实结论。[C1]",
            claims=[
                {
                    "claim_id": "claim-1",
                    "claim": "事实结论",
                    "citation_ids": ["C1"],
                }
            ],
            missing_information=[],
            conflicts=[],
        )
    )

    assert "1.1" not in structured.answer
    assert "1.2" not in structured.answer
    assert structured.answer == "事实结论"
