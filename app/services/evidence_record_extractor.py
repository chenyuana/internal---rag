from __future__ import annotations

from typing import cast, get_args

from app.schemas.evidence import EvidenceRecord, RequirementType
from app.schemas.planning import QueryPlanV2
from app.schemas.retrieval import CoverageCell, SelectedChunk
from app.services.cell_evidence import dimension_keywords, substantive_lines
from app.services.requirement_taxonomy import requirement_type_for_line


class EvidenceRecordExtractor:
    """Extract cell-scoped evidence records from selected chunks."""

    def extract(
        self,
        *,
        plan: QueryPlanV2,
        chunks: list[SelectedChunk],
        coverage_matrix: list[CoverageCell],
    ) -> list[EvidenceRecord]:
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        cell_by_id = {cell.id: cell for cell in plan.cells}
        records: list[EvidenceRecord] = []
        effective_coverage = coverage_matrix
        if not effective_coverage and len(plan.cells) == 1 and chunks:
            single_cell = plan.cells[0]
            effective_coverage = [
                CoverageCell(
                    subquery_id=single_cell.id,
                    subject=single_cell.subject,
                    aspect=single_cell.aspect,
                    status="covered",
                    chunk_ids=[chunk.chunk_id for chunk in chunks],
                    citation_ids=[chunk.citation_id for chunk in chunks],
                )
            ]
        for coverage in effective_coverage:
            if coverage.status != "covered":
                continue
            planned_cell = cell_by_id.get(coverage.subquery_id)
            if planned_cell is None:
                continue
            keywords = dimension_keywords(planned_cell.aspect)
            for chunk_id in coverage.chunk_ids:
                chunk = chunk_by_id.get(chunk_id)
                if chunk is None:
                    continue
                lines = substantive_lines(chunk, keywords, planned_cell.aspect)
                for line_index, line in enumerate(lines, start=1):
                    requirement_type = self._requirement_type(chunk, line)
                    records.append(
                        EvidenceRecord(
                            evidence_id=(
                                f"{planned_cell.id}:{chunk.chunk_id}:{line_index}"
                            ),
                            cell_id=planned_cell.id,
                            subject=planned_cell.subject,
                            aspect=planned_cell.aspect,
                            document_id=chunk.document_id,
                            document_name=chunk.metadata.document_name or chunk.document_id,
                            section_id=(
                                chunk.metadata.section_id or chunk.metadata.chapter_path
                            ),
                            section_title=chunk.metadata.section_title,
                            requirement_type=requirement_type,
                            requirement_text=line,
                            citation_id=chunk.citation_id,
                            page_number=chunk.metadata.page_number,
                            retrieval_score=(
                                chunk.rerank_score
                                if chunk.rerank_score is not None
                                else chunk.hybrid_score
                            ),
                            confidence=0.9 if chunk.metadata.requirement_type else 0.75,
                        )
                    )
        return records

    @staticmethod
    def _requirement_type(chunk: SelectedChunk, line: str) -> RequirementType:
        configured = chunk.metadata.requirement_type
        allowed = set(get_args(RequirementType))
        if configured in allowed:
            return cast(RequirementType, configured)
        return requirement_type_for_line(line)
