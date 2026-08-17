from __future__ import annotations

from app.core.config import GenerationSettings
from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.evidence_judge import EvidenceJudge
from app.services.query_analyzer import QueryAnalyzer


def selected_chunk(citation_id: str, text: str) -> SelectedChunk:
    return SelectedChunk(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id=f"doc-{citation_id}",
        dataset_id="kb-1",
        text=text,
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )


def test_evidence_judge_rejects_no_evidence() -> None:
    query = QueryAnalyzer().analyze("健康监控任务的执行周期是多少？")

    result = EvidenceJudge(GenerationSettings()).assess(query, [])

    assert result.status == "UNANSWERABLE"
    assert result.covered_requirements == []
    assert result.missing_requirements


def test_evidence_judge_only_allows_the_covered_part() -> None:
    query = QueryAnalyzer().analyze(
        "健康监控任务的执行周期和失败返回值分别是什么？"
    )

    result = EvidenceJudge(GenerationSettings()).assess(
        query,
        [selected_chunk("C1", "健康监控任务的执行周期为1000毫秒。")],
    )

    assert result.status == "PARTIALLY_ANSWERABLE"
    assert result.covered_requirements == ["健康监控任务的执行周期"]
    assert result.missing_requirements == ["失败返回值"]


def test_evidence_judge_detects_conflicting_measurements() -> None:
    query = QueryAnalyzer().analyze("健康监控任务的执行周期是多少？")

    result = EvidenceJudge(GenerationSettings()).assess(
        query,
        [
            selected_chunk("C1", "健康监控任务的执行周期为1000毫秒。"),
            selected_chunk("C2", "健康监控任务的执行周期为2000毫秒。"),
        ],
    )

    assert result.status == "CONFLICTED"
    assert result.conflicting_citations[0].citation_ids == ["C1", "C2"]
