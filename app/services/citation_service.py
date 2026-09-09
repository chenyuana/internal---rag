from __future__ import annotations

import re

from app.schemas.retrieval import Citation, RetrievedChunk, SelectedChunk
from app.services.citation_location import published_page_number
from app.services.table_evidence import html_table_blocks


def _table_title(table_html: str) -> str | None:
    """Extract a caption for display without making the schema HTML-aware."""
    match = re.search(r"<caption\b[^>]*>(.*?)</caption>", table_html, re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    title = re.sub(r"<[^>]+>", "", match.group(1))
    title = " ".join(title.split()).strip()
    return title or None


class CitationService:
    """Assign request-local citation IDs from backend-owned chunk metadata."""

    def build(
        self,
        chunks: list[RetrievedChunk],
    ) -> tuple[list[SelectedChunk], list[Citation]]:
        selected: list[SelectedChunk] = []
        citations: list[Citation] = []
        for index, chunk in enumerate(chunks, start=1):
            if not chunk.metadata.page_number:
                recovered_page = published_page_number(chunk.text)
                if recovered_page:
                    chunk = chunk.model_copy(
                        update={
                            "metadata": chunk.metadata.model_copy(
                                update={"page_number": recovered_page}
                            )
                        }
                    )
            citation_id = f"C{index}"
            table_htmls = html_table_blocks(chunk.text)
            table_html = next(iter(table_htmls), None)
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
                    dataset_id=chunk.dataset_id,
                    document_name=chunk.metadata.document_name,
                    version=chunk.metadata.version,
                    source_path=chunk.metadata.source_path,
                    chapter_path=chunk.metadata.chapter_path,
                    page_number=chunk.metadata.page_number,
                    paragraph_index=chunk.metadata.paragraph_index,
                    quote=chunk.text,
                    table_html=table_html,
                    table_title=_table_title(table_html) if table_html else None,
                    table_htmls=table_htmls,
                )
            )
        return selected, citations
