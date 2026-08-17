from __future__ import annotations

from dataclasses import dataclass

from app.schemas.retrieval import CoverageCell, QueryPlan, RetrievedChunk


@dataclass(slots=True)
class CoverageSelection:
    chunks: list[RetrievedChunk]
    matrix: list[CoverageCell]


class CoverageSelector:
    """Select evidence by required-cell coverage before global relevance."""

    def select(
        self,
        plan: QueryPlan,
        results: dict[str, list[RetrievedChunk]],
        *,
        limit: int,
    ) -> CoverageSelection:
        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()

        # Round one reserves one slot for each required retrieval cell.
        for subquery in plan.subqueries:
            if len(selected) >= limit:
                break
            candidate = next(
                (
                    item
                    for item in results.get(subquery.id, [])
                    if item.chunk_id not in selected_ids
                ),
                None,
            )
            if candidate is None:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.chunk_id)

        # Remaining budget rewards relevance while mildly preferring new documents.
        remaining: list[tuple[float, RetrievedChunk]] = []
        selected_documents = {item.document_id for item in selected}
        for candidates in results.values():
            for candidate in candidates:
                if candidate.chunk_id in selected_ids:
                    continue
                score = self._score(candidate)
                if candidate.document_id not in selected_documents:
                    score += 0.05
                remaining.append((score, candidate))
        remaining.sort(key=lambda item: item[0], reverse=True)
        for _, candidate in remaining:
            if len(selected) >= limit:
                break
            if candidate.chunk_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.chunk_id)

        matrix = []
        for subquery in plan.subqueries:
            covered = [
                item
                for item in results.get(subquery.id, [])
                if item.chunk_id in selected_ids
            ]
            matrix.append(
                CoverageCell(
                    subquery_id=subquery.id,
                    subject=subquery.subject,
                    aspect=subquery.aspect,
                    status="covered" if covered else "missing",
                    chunk_ids=[item.chunk_id for item in covered],
                    reason=None if covered else "no_selected_evidence",
                )
            )
        return CoverageSelection(chunks=selected, matrix=matrix)

    @staticmethod
    def _score(chunk: RetrievedChunk) -> float:
        return chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score
