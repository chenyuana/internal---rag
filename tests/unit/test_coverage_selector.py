from app.schemas.retrieval import ChunkMetadata, QueryPlan, RetrievedChunk, SubQuery
from app.services.coverage_selector import CoverageSelector


def _chunk(chunk_id: str, document_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        dataset_id="kb-1",
        text=chunk_id,
        metadata=ChunkMetadata(document_name=f"{document_id}.pdf"),
        hybrid_score=score,
        rerank_score=score,
    )


def test_selector_reserves_slots_for_required_cells_and_deduplicates() -> None:
    plan = QueryPlan(
        query_type="comparison",
        subjects=["A", "B"],
        aspects=["波段", "时段"],
        subqueries=[
            SubQuery(id="q1", query="A 波段", subject="A", aspect="波段"),
            SubQuery(id="q2", query="B 波段", subject="B", aspect="波段"),
            SubQuery(id="q3", query="A 时段", subject="A", aspect="时段"),
            SubQuery(id="q4", query="B 时段", subject="B", aspect="时段"),
        ],
        synthesis_mode="matrix",
    )
    shared = _chunk("shared", "doc-a", 0.99)
    results = {
        "q1": [shared, _chunk("a-band", "doc-a", 0.8)],
        "q2": [shared, _chunk("b-band", "doc-b", 0.7)],
        "q3": [_chunk("a-time", "doc-a", 0.6)],
        "q4": [],
    }

    selection = CoverageSelector().select(plan, results, limit=4)

    assert len({item.chunk_id for item in selection.chunks}) == len(selection.chunks)
    assert {item.chunk_id for item in selection.chunks} == {
        "shared",
        "b-band",
        "a-time",
        "a-band",
    }
    status = {cell.subquery_id: cell.status for cell in selection.matrix}
    assert status == {"q1": "covered", "q2": "covered", "q3": "covered", "q4": "missing"}
