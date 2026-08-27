from __future__ import annotations

import re
import unicodedata

from app.ingestion.regulations import extract_article_references
from app.schemas.retrieval import NormalizedQuery, SelectedChunk

_LEADING_DOC_RE = re.compile(r"^文档：\s*(.+)$", re.MULTILINE)
_LEADING_SECTION_RE = re.compile(r"^章节：\s*(.+)$", re.MULTILINE)
_LEADING_ARTICLE_RE = re.compile(r"^条号：\s*(\S+)\s*$", re.MULTILINE)
_ARTICLE_TITLE_RE = re.compile(
    r"第\s*[A-Za-z]?\d+(?:\.\d+)*\s*条\s*(.+)$",
    re.IGNORECASE,
)
_COMPARISON_MARKERS = ("比较", "对比", "区别", "差异", "分别", "异同")


def _header(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


def _section_title(text: str) -> str | None:
    section = _header(_LEADING_SECTION_RE, text)
    if section is None:
        return None
    # Chapter paths may contain unrelated parent headings before the article.
    leaf = section.rsplit("/", maxsplit=1)[-1].strip()
    match = _ARTICLE_TITLE_RE.search(leaf)
    title = match.group(1).strip() if match else leaf
    return title or None


class EvidenceScopeSelector:
    """Narrow generation evidence when the query identifies an exact article.

    Retrieval may legitimately return semantically similar requirements from
    several regulations. When an article number or a sufficiently specific
    article title is present in the query, generation should use all chunks of
    that article in its owning document, instead of mixing those regulations.
    Comparison questions deliberately keep the complete evidence set.
    """

    def select(
        self,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
    ) -> list[SelectedChunk]:
        if len(chunks) < 2 or any(
            marker in query.original_query for marker in _COMPARISON_MARKERS
        ):
            return chunks

        declared_articles = {
            reference.normalized_id
            for reference in extract_article_references(query.original_query)
        }
        query_text = _normalized_text(query.original_query)
        anchors: list[tuple[int, str, str]] = []
        for chunk in chunks:
            document = _header(_LEADING_DOC_RE, chunk.text)
            article = _header(_LEADING_ARTICLE_RE, chunk.text)
            if not document or not article:
                continue
            if article in declared_articles:
                anchors.append((10_000, document, article))
                continue
            title = _section_title(chunk.text)
            normalized_title = _normalized_text(title or "")
            if len(normalized_title) >= 4 and normalized_title in query_text:
                anchors.append((len(normalized_title), document, article))

        if not anchors:
            return chunks
        _, primary_document, primary_article = max(anchors, key=lambda item: item[0])
        scoped = [
            chunk
            for chunk in chunks
            if _header(_LEADING_DOC_RE, chunk.text) == primary_document
            and _header(_LEADING_ARTICLE_RE, chunk.text) == primary_article
        ]
        return scoped or chunks
