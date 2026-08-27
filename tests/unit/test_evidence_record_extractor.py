from __future__ import annotations

from app.schemas.planning import PlannedCell, QueryPlanV2, RetrievalQuery
from app.schemas.retrieval import ChunkMetadata, CoverageCell, SelectedChunk
from app.services.evidence_record_extractor import EvidenceRecordExtractor


def _plan() -> QueryPlanV2:
    query = "地质灾害倾斜摄影测量 成果验收"
    return QueryPlanV2(
        query_type="fact",
        aspects=["成果验收"],
        cells=[
            PlannedCell(
                id="q1",
                subject="地质灾害倾斜摄影测量",
                aspect="成果验收",
                original_query=query,
                retrieval_queries=[
                    RetrievalQuery(kind="original", text=query, generated_by="planner")
                ],
            )
        ],
        synthesis_mode="direct",
        planner_source="model",
        confidence=0.9,
    )


def test_extractor_converts_html_table_and_preserves_cell_identity() -> None:
    chunk = SelectedChunk(
        citation_id="C1",
        chunk_id="table-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text=(
            "<table><caption>成果精度检查</caption>"
            "<tr><th>项目</th><th>限差</th></tr>"
            "<tr><td>平面精度</td><td>0.20 m</td></tr></table>"
        ),
        metadata=ChunkMetadata(
            document_name="地质灾害摄影测量规程",
            section_id="A.3",
            requirement_type="acceptance",
        ),
        hybrid_score=0.8,
        vector_score=0.7,
        keyword_score=0.9,
    )
    records = EvidenceRecordExtractor().extract(
        plan=_plan(),
        chunks=[chunk],
        coverage_matrix=[
            CoverageCell(
                subquery_id="q1",
                subject="地质灾害倾斜摄影测量",
                aspect="成果验收",
                status="covered",
                chunk_ids=["table-1"],
                citation_ids=["C1"],
            )
        ],
    )
    assert len(records) == 1
    assert records[0].cell_id == "q1"
    assert records[0].requirement_type == "acceptance"
    assert "项目: 平面精度" in records[0].requirement_text
    assert "<table>" not in records[0].requirement_text
