from __future__ import annotations

from app.core.config import GenerationSettings
from app.schemas.chat import EvidenceSentence
from app.schemas.retrieval import NormalizedQuery, SelectedChunk
from app.services.evidence_text import overlap_score, split_sentences


class EvidenceExtractor:
    """Select request-relevant source sentences without asking a model to invent evidence."""

    def __init__(self, settings: GenerationSettings) -> None:
        self._settings = settings

    def extract(
        self,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
    ) -> list[EvidenceSentence]:
        ranked: list[tuple[float, int, int, EvidenceSentence]] = []
        for chunk_rank, chunk in enumerate(chunks):
            sentences = split_sentences(chunk.text)
            candidates = [
                (
                    self._sentence_score(query, sentence),
                    sentence_index,
                    sentence,
                )
                for sentence_index, sentence in enumerate(sentences)
            ]
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = [
                item
                for item in candidates
                if item[0] >= self._settings.min_evidence_overlap
            ][: self._settings.max_sentences_per_citation]
            if not selected and candidates:
                selected = candidates[:1]
            for score, sentence_index, sentence in selected:
                ranked.append(
                    (
                        score,
                        chunk_rank,
                        sentence_index,
                        EvidenceSentence(
                            citation_id=chunk.citation_id,
                            chunk_id=chunk.chunk_id,
                            text=sentence,
                        ),
                    )
                )
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        return [
            item[3]
            for item in ranked[: self._settings.max_evidence_sentences]
        ]

    @staticmethod
    def _sentence_score(query: NormalizedQuery, sentence: str) -> float:
        score = overlap_score(query.normalized_query, sentence)
        if query.exact_tokens and any(token in sentence for token in query.exact_tokens):
            score += 1
        return score
