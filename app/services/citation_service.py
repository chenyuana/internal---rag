from __future__ import annotations

from app.schemas.retrieval import Citation, RetrievedChunk, SelectedChunk


class CitationService:
    """Assign request-local citation IDs from backend-owned chunk metadata."""

    def build(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[SelectedChunk], list[Citation]]:
        selected: list[SelectedChunk] = []
        citations: list[Citation] = []
        for index, chunk in enumerate(chunks, start=1):
            citation_id = f"C{index}"
            selected.append(
                SelectedChunk(
                    citation_id=citation_id,
                    **chunk.model_dump(),
                )
            )
            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_name=chunk.metadata.document_name,
                    version=chunk.metadata.version,
                    source_path=chunk.metadata.source_path,
                    chapter_path=chunk.metadata.chapter_path,
                    page_number=chunk.metadata.page_number,
                    paragraph_index=chunk.metadata.paragraph_index,
                    quote=chunk.text,
                )
            )
        return selected, citations
