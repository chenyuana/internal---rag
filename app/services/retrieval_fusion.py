from __future__ import annotations

from app.schemas.retrieval import RetrievedChunk


class RetrievalFusion:
    """Weighted reciprocal-rank fusion without comparing raw cross-query scores."""

    def __init__(self, *, rrf_k: int = 60, translated_weight: float = 0.85) -> None:
        self._rrf_k = rrf_k
        self._translated_weight = translated_weight

    def fuse(
        self,
        original: list[RetrievedChunk],
        translated: list[RetrievedChunk] | None = None,
    ) -> list[RetrievedChunk]:
        scores: dict[tuple[str, str], float] = {}
        items: dict[tuple[str, str], RetrievedChunk] = {}
        for weight, candidates in (
            (1.0, original),
            (self._translated_weight, translated or []),
        ):
            for rank, candidate in enumerate(candidates, start=1):
                key = (candidate.document_id, candidate.chunk_id)
                scores[key] = scores.get(key, 0.0) + weight / (self._rrf_k + rank)
                current = items.get(key)
                if current is None or candidate.hybrid_score > current.hybrid_score:
                    items[key] = candidate
        return [
            items[key]
            for key in sorted(scores, key=lambda item: scores[item], reverse=True)
        ]
