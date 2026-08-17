from __future__ import annotations

from app.core.config import GenerationSettings
from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.evidence_extractor import EvidenceExtractor
from app.services.query_analyzer import QueryAnalyzer


def test_evidence_extractor_preserves_exact_technical_token() -> None:
    query = QueryAnalyzer().analyze("HEALTH_MONITOR_PERIOD_MS 的值是多少？")
    chunk = SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text=(
            "该模块负责系统初始化。\n"
            "HEALTH_MONITOR_PERIOD_MS = 1000U。\n"
            "其他任务使用独立调度器。"
        ),
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    assert evidence[0].citation_id == "C1"
    assert evidence[0].text == "HEALTH_MONITOR_PERIOD_MS = 1000U。"
