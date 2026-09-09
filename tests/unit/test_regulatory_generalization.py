from __future__ import annotations

import json
from typing import Any, cast

import pytest

from app.core.config import Settings
from app.ingestion.pipeline import PageRecord, build_chunks, split_blocks
from app.ingestion.regulatory_structure import expand_regulatory_blocks, split_regulatory_text
from app.pipelines.rag_pipeline import RagPipeline
from app.schemas.chat import CandidateClaim, ChatCompletionRequest, StructuredAnswer
from app.schemas.planning import PlannedCell, RetrievalQuery
from app.schemas.retrieval import (
    ChunkMetadata,
    RagflowRetrievalRequest,
    RetrievalSearchRequest,
    RetrievedChunk,
    SelectedChunk,
)
from app.services.access_control import AccessControlService
from app.services.article_identity import owned_articles, owns_requested_article
from app.services.citation_service import CitationService
from app.services.evidence_extractor import EvidenceExtractor
from app.services.evidence_judge import EvidenceJudge
from app.services.evidence_scope_selector import EvidenceScopeSelector
from app.services.evidence_text import has_parameter_value, overlap_score
from app.services.grounding_validator import ClaimGroundingValidator, _extract_numerical_marks
from app.services.query_analyzer import QueryAnalyzer
from app.services.query_translator import QueryTranslator
from app.services.ragflow_client import RagflowClient
from app.services.regulation_answer_builder import RegulationAnswerBuilder
from app.services.regulation_context import amendment_scope_note
from app.services.retrieval_service import RetrievalExecution, RetrievalService


def selected(text: str, section: str | None = None, cid: str = "C1") -> SelectedChunk:
    return SelectedChunk(
        citation_id=cid, chunk_id=cid, document_id="source-a", dataset_id="kb-1", text=text,
        metadata=ChunkMetadata(section_id=section, document_name="regulation.pdf",
                               status="effective"),
        hybrid_score=0.8, vector_score=0.8, keyword_score=0.8,
    )


@pytest.mark.parametrize("number,title", [
    ("27.801", "Ditching"), ("25.1309", "Equipment, systems, and installations"),
    ("33.77", "Foreign object ingestion"),
])
def test_joined_headings_recover_chunk_ownership(number: str, title: str) -> None:
    text = f"12. Amend Sec. {number} to read as follows: Sec. {number} {title}.(a) Test body."
    page = PageRecord(page_number=7, raw_text="", rich_blocks=[
        {"block_type": "paragraph", "text": text, "bbox": [10, 20, 300, 90]},
    ])
    blocks = split_blocks("generic", [page])
    body = next(b for b in blocks if "Test body" in b.text)
    assert body.article_id_normalized == number + "(A)"
    assert body.section_path[-1] == f"Sec. {number} {title}."
    assert body.bbox == [10, 20, 300, 90]
    chunks = build_chunks("generic", blocks)
    assert any(c.article_id_normalized and c.article_id_normalized.startswith(number)
               for c in chunks)
    assert "Test body" in "\n".join(c.text for c in chunks)


@pytest.mark.parametrize("text", [
    "See Sec. 25.1309 for system requirements.",
    "Sec. 25.1309 is revised as proposed.",
    "The applicant suggested Sec. 25.1309 Equipment. This is a comment.",
    "14. Amend paragraph (b) as follows.",
    "The price is 25.1309 dollars.",
])
def test_prose_does_not_create_new_section(text: str) -> None:
    assert split_regulatory_text(text) == [text]
    assert not owned_articles(selected(text))


def test_table_and_excluded_blocks_are_not_split() -> None:
    text = "Sec. 91.205 Equipment.(a) Value"
    items = [{"block_type": "table", "text": text},
             {"block_type": "paragraph", "text": text, "exclusion_reason": "margin"}]
    assert expand_regulatory_blocks(items) == items


@pytest.mark.parametrize("number", ["21.137", "25.1309", "91.205"])
def test_inline_citation_cannot_impersonate_section(number: str) -> None:
    inline = selected(f"The installation must comply with Sec. {number}.", "99.12")
    own = selected(f"Sec. {number} Equipment.\n(a) A requirement.", number)
    assert not owns_requested_article(inline, [number])
    assert owns_requested_article(own, [number])
    assert not owns_requested_article(selected(f"See Sec. {number}."), [number])


def test_section_prefix_and_child_scope() -> None:
    assert not owns_requested_article(selected("text", "25.13090"), ["25.1309"])
    assert owns_requested_article(selected("text", "25.1309(B)(1)"), ["25.1309"])
    assert not owns_requested_article(selected("text", "25.1309(C)"), ["25.1309(B)"])


def test_inline_reference_cannot_bypass_similarity_threshold(settings: Settings) -> None:
    service = RetrievalService(
        settings=settings, ragflow=None, reranker=None,
        access_control=AccessControlService(settings.access_control),
    )
    inline = RetrievedChunk(
        chunk_id="inline", document_id="source-a", dataset_id="kb-1",
        text="See Sec. 25.1309 for another requirement.",
        metadata=ChunkMetadata(section_id="25.9999", status="effective"),
        hybrid_score=0.01,
    )
    reason = service._filter_reason(
        inline, allowed_datasets=["kb-1"], metadata_values={"status": "effective"},
        article_aliases=["25.1309"], article_ids=["25.1309"],
    )
    assert reason == "below_similarity_threshold"


def test_scope_keeps_multiple_sections_and_documents() -> None:
    a = selected("first", "25.1309(A)")
    b = selected("second", "25.1353", "C2").model_copy(update={"document_id": "source-b"})
    noise = selected("noise", "25.9999", "C3")
    query = QueryAnalyzer().analyze("25.1309和25.1353有什么要求？")
    assert EvidenceScopeSelector().select(query, [a, b, noise]) == [a, b]


@pytest.mark.parametrize("cn,en", [
    ("动态稳定性", "Dynamic stability"), ("验收标准", "acceptance criteria"),
    ("试验方法", "test method"), ("静态方向稳定性", "Static directional stability"),
])
def test_bilingual_terms_match_without_clause_mapping(cn: str, en: str) -> None:
    assert overlap_score(cn, en) == 1


@pytest.mark.parametrize("value", ["70 knots", "25,000 feet", "120 pounds", "4 percent",
                                         "2.4 million", "90 degrees", "60 minutes"])
def test_aviation_and_economic_values(value: str) -> None:
    assert has_parameter_value(value)


def test_metadata_question_does_not_return_unrelated_clause() -> None:
    assert RegulationAnswerBuilder().build(
        QueryAnalyzer().analyze("发布日期是什么？"),
        [selected("(a) A requirement.", "25.1309")],
    ) is None


def test_extractive_preserves_intro_later_letters_and_continuations() -> None:
    a = selected("Sec. 33.77 Test conditions.\nThis applies only to turbine engines.\n\n"
                 "(e) First condition.\n"
                 "(f) Next condition.", "33.77")
    b = selected("(f) Additional exception.\n(g) Last condition.", "33.77", "C2")
    answer = RegulationAnswerBuilder().build(
        QueryAnalyzer().analyze("33.77有哪些要求？"), [a, b],
    )
    assert answer is not None
    for text in ("only to turbine", "First condition", "Next condition",
                 "Additional exception", "Last condition"):
        assert text in answer.answer


def test_amendment_note_does_not_invent_complete_text() -> None:
    assert amendment_scope_note([selected("* * * * *\n(b) New text", "27.801")])
    assert amendment_scope_note([selected("(a) Full text", "27.801")]) is None


def test_wrong_section_is_not_answerable(settings: Settings) -> None:
    assessment = EvidenceJudge(settings.generation).assess(
        QueryAnalyzer().analyze("27.801的要求是什么？"),
        [selected("The equipment complies with 27.801.", "27.1309")],
    )
    assert assessment.status == "UNANSWERABLE"
    assert "核对条号" in assessment.missing_requirements[0]


@pytest.mark.parametrize("section", ["25.181", "27.181", "29.181"])
async def test_mixed_query_translation_retains_identifiers(
    settings: Settings, section: str,
) -> None:
    question = f"{section}动态稳定性有什么要求？"
    cell = PlannedCell(id="q1", subject=None, aspect=question, original_query=question,
                       retrieval_queries=[RetrievalQuery(kind="original", text=question,
                                                        generated_by="user")])
    translator = QueryTranslator(settings.translation.model_copy(update={"model_enabled": False}),
                                 None)
    result = await translator.translate(query=QueryAnalyzer().analyze(question), cell=cell,
                                        section_titles=[])
    assert result.decision.should_translate
    assert section in (result.decision.translated_query or "")
    assert "dynamic stability" in (result.decision.translated_query or "")


async def test_translator_rejects_changed_article(settings: Settings) -> None:
    class Model:
        async def chat_completion(self, **kwargs: Any) -> str:
            return json.dumps({"should_translate": True,
                               "translated_query": "25.999 动态稳定性 dynamic stability",
                               "reason": "translation"})
    question = "25.181动态稳定性要求"
    cell = PlannedCell(id="q1", aspect="动态稳定性", original_query=question,
                       retrieval_queries=[RetrievalQuery(kind="original", text=question,
                                                        generated_by="user")])
    translator = QueryTranslator(settings.translation.model_copy(update={"model_enabled": True}),
                                 Model())
    result = await translator.translate(query=QueryAnalyzer().analyze(question), cell=cell,
                                        section_titles=[])
    assert not result.decision.should_translate


async def test_single_query_translation_preserves_access_scope(settings: Settings) -> None:
    class Backend:
        requests: list[RagflowRetrievalRequest]

        def __init__(self) -> None:
            self.requests = []

        async def retrieve(self, request: RagflowRetrievalRequest) -> list[RetrievedChunk]:
            self.requests.append(request)
            if "dynamic stability" not in request.question:
                return []
            chunk = selected("Sec. 27.181 Dynamic stability.\n(b) Oscillations must be damped.",
                             "27.181")
            return [RetrievedChunk(**chunk.model_dump(exclude={"citation_id"}))]

    backend = Backend()
    service = RetrievalService(settings=settings, ragflow=cast(RagflowClient, backend),
                               reranker=None,
                               access_control=AccessControlService(settings.access_control))
    result = await service.execute(RetrievalSearchRequest(
        query="27.181动态稳定性要求是什么？", knowledge_base_ids=["kb-1"],
        document_ids=["source-a"],
    ), user_id="user-1")
    assert result.selected_chunks
    assert len(backend.requests) == 2
    assert all(r.document_ids == ["source-a"] and r.dataset_ids == ["kb-1"]
               for r in backend.requests)
    assert owned_articles(result.selected_chunks[0]) == {"27.181"}


@pytest.mark.parametrize("source,claim", [
    ("25,000 feet", "25000英尺"), ("70 knots", "70节"),
    ("4 percent", "4%"), ("60 minutes", "60分钟"),
])
def test_numeric_grounding_allows_translation_but_not_different_values(
    source: str, claim: str,
) -> None:
    assert _extract_numerical_marks(source) == _extract_numerical_marks(claim)
    assert _extract_numerical_marks(source)
    assert _extract_numerical_marks(source) != _extract_numerical_marks("9999英尺")


def test_numeric_grounding_rejects_reversed_bound() -> None:
    answer = StructuredAnswer(answerability="ANSWERABLE", answer="爬升梯度不超过4% [C1]",
                              claims=[CandidateClaim(claim_id="wrong-bound",
                                                     claim="爬升梯度不超过4%",
                                                     citation_ids=["C1"])])
    with pytest.raises(ValueError, match="bound"):
        ClaimGroundingValidator().validate(answer, chunks=[selected(
            "The climb gradient must be at least 4 percent.", "27.65",
        )])


def test_exact_clause_evidence_keeps_exceptions_under_sentence_budget(settings: Settings) -> None:
    body = "Sec. 27.801 Ditching.\n(a) Applies only to this configuration.\n"
    body += "\n".join(f"({i}) A subordinate requirement." for i in range(1, 10))
    body += "\n(b) This paragraph does not apply to the other configuration."
    evidence = EvidenceExtractor(settings.generation).extract(
        QueryAnalyzer().analyze("27.801有哪些要求？"), [selected(body, "27.801")],
    )
    assert len(evidence) == 1
    assert "does not apply" in evidence[0].text
    assert "(9)" in evidence[0].text


def test_appendix_resets_prior_section_ownership() -> None:
    page = PageRecord(page_number=1, raw_text="", cleaned_text=(
        "Sec. 27.801 Ditching.\n(a) A requirement.\n"
        "Appendix B to Part 27—Test Methods\nTest apparatus specifications."
    ))
    blocks = split_blocks("annex-document", [page])
    last = blocks[-1]
    assert last.article_id_normalized is None
    assert last.section_path == ["Appendix B to Part 27—Test Methods"]


@pytest.mark.parametrize("enabled", [True, False])
async def test_chinese_generation_uses_validated_path_and_can_be_disabled(
    settings: Settings, enabled: bool,
) -> None:
    query = QueryAnalyzer().analyze("27.65爬升梯度的要求是什么？")
    chunk = selected("Sec. 27.65 Climb gradient.\n"
                     "(b) The steady climb gradient must be at least 4 percent.", "27.65")
    chunks, citations = CitationService().build([
        RetrievedChunk(**chunk.model_dump(exclude={"citation_id"})),
    ])
    execution = RetrievalExecution(
        request_id="cross-language", query=query, selected_chunks=chunks, citations=citations,
        candidate_count=1, reranker_used=False, reranker_fallback=False,
        ragflow_request=RagflowRetrievalRequest(
            question=query.original_query, dataset_ids=["kb-1"],
        ),
        stage_counts={}, candidates=[],
    )

    class Retrieval:
        async def execute(self, *args: Any, **kwargs: Any) -> RetrievalExecution:
            return execution

    class Model:
        calls = 0

        async def chat_completion(self, **kwargs: Any) -> str:
            self.calls += 1
            return json.dumps({
                "answerability": "ANSWERABLE", "answer": "爬升梯度至少4%。[C1]",
                "claims": [{"claim_id": "climb", "claim": "爬升梯度至少4%",
                            "citation_ids": ["C1"]}],
                "missing_information": [], "conflicts": [],
            })

    settings.generation.cross_language_generation = enabled
    settings.generation.answer_mode = "deterministic"
    model = Model()
    pipeline = RagPipeline(settings=settings, retrieval=cast(RetrievalService, Retrieval()),
                           answer_model=model)
    result = await pipeline.run(ChatCompletionRequest(query=query.original_query,
                                                      knowledge_base_ids=["kb-1"]), user_id="test")
    assert result.status == "ANSWERABLE", result.validation_warnings
    assert result.citations
    if enabled:
        assert model.calls == 1
        assert "爬升梯度至少4%" in result.answer
    else:
        assert model.calls == 0
        assert "steady climb gradient" in result.answer
