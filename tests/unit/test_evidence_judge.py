from __future__ import annotations

from app.core.config import GenerationSettings
from app.schemas.retrieval import ChunkMetadata, CoverageCell, SelectedChunk
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


def test_parameter_requirement_needs_numeric_value() -> None:
    # "作业高度"是参数类需求，仅命中定义句（无数值）不能算覆盖。
    query = QueryAnalyzer().analyze("无人机作业高度是多少？")

    result = EvidenceJudge(GenerationSettings()).assess(
        query,
        [selected_chunk("C1", "作业高度 无人机作业时喷头与靶标顶端的垂直距离。")],
    )

    assert result.status == "UNANSWERABLE"


def test_parameter_requirement_covered_with_numeric_value() -> None:
    query = QueryAnalyzer().analyze("无人机作业高度是多少？")

    result = EvidenceJudge(GenerationSettings()).assess(
        query,
        [selected_chunk("C1", "无人机作业高度为2 m~4 m。")],
    )

    assert result.status == "ANSWERABLE"


def test_non_parameter_requirement_keeps_legacy_coverage() -> None:
    # 非参数类需求（定义类）不应被数值检查收紧，仍按词汇重合判覆盖。
    query = QueryAnalyzer().analyze("什么是植保无人机？")

    result = EvidenceJudge(GenerationSettings()).assess(
        query,
        [
            selected_chunk(
                "C1",
                "植保无人机 配备液态农药喷洒系统用于农业生产植保作业的无人飞机。",
            )
        ],
    )

    assert result.status == "ANSWERABLE"


def test_parameter_cell_without_value_is_missing_in_matrix() -> None:
    query = QueryAnalyzer().analyze("无人机作业高度是多少？")
    chunk = selected_chunk("C1", "作业高度 无人机作业时喷头与靶标顶端的垂直距离。")
    coverage = [
        CoverageCell(
            subquery_id="q1",
            subject="无人机",
            aspect="作业高度",
            status="covered",
            chunk_ids=["chunk-C1"],
        )
    ]

    result = EvidenceJudge(GenerationSettings()).assess(query, [chunk], coverage)

    assert result.status == "UNANSWERABLE"
    assert result.missing_requirements == ["无人机：作业高度"]
