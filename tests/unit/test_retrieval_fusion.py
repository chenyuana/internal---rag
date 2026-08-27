from __future__ import annotations

from app.schemas.retrieval import ChunkMetadata, RetrievedChunk
from app.services.retrieval_fusion import RetrievalFusion


def _chunk(chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-1",
        dataset_id="kb-1",
        text=chunk_id,
        metadata=ChunkMetadata(document_name="规程"),
        hybrid_score=score,
    )


def test_rrf_deduplicates_and_rewards_cross_query_hits() -> None:
    result = RetrievalFusion(rrf_k=60, translated_weight=0.85).fuse(
        [_chunk("original-only", 0.99), _chunk("both", 0.4)],
        [_chunk("both", 0.2), _chunk("translated-only", 0.95)],
    )
    assert [item.chunk_id for item in result] == [
        "both",
        "original-only",
        "translated-only",
    ]
    assert len([item for item in result if item.chunk_id == "both"]) == 1
