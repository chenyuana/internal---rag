from __future__ import annotations

import re

from app.ingestion.regulations import (
    ArticleReference,
    extract_article_references,
    text_contains_article_alias,
)
from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import NormalizedQuery, SelectedChunk

# Ingested regulation chunks carry a structured header produced by the parser:
#   文档：CCAR-25-R4 运输类飞机适航标准.pdf
#   章节：... / 第 25.981 条 燃油箱点燃防护
#   条号：25.981
#   页码：123-124
# The authoritative article of a chunk comes from the `条号` line, which is far
# more reliable than scanning the whole chunk text (a CCAR-26 chunk may mention
# "M25.1" or "25.981" in passing without being the article itself).
_LEADING_DOC_RE = re.compile(r"^文档：\s*(.+)$", re.MULTILINE)
_LEADING_ARTICLE_RE = re.compile(r"^条号：\s*(\S+)\s*$", re.MULTILINE)
# Regulation file identifiers, e.g. CCAR-25-R4, CCAR-26.
_CCAR_MARKER_RE = re.compile(
    r"CCAR[\s\-—–]*(\d+(?:[\s\-—–]*R\d+)?)",
    re.IGNORECASE,
)


def _leading_document(text: str) -> str | None:
    match = _LEADING_DOC_RE.search(text)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _leading_article(text: str) -> str | None:
    match = _LEADING_ARTICLE_RE.search(text)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _file_markers(text: str) -> set[str]:
    """Extract normalized regulation file markers from free text."""
    markers: set[str] = set()
    for match in _CCAR_MARKER_RE.finditer(text):
        parts = re.findall(r"\d+|[Rr]\d+", match.group(1))
        if parts:
            markers.add("CCAR-" + "-".join(parts).upper())
    return markers


def _file_marker_match(declared: str, candidate: str) -> bool:
    """Match a declared marker against a chunk marker, version-aware.

    ``CCAR-25`` matches ``CCAR-25-R4`` (declared without a revision), while a
    revision-pinned declaration ``CCAR-25-R4`` does not match ``CCAR-25``.
    """
    if declared == candidate:
        return True
    if declared.startswith(candidate + "-") or candidate.startswith(declared + "-"):
        return True
    return False


class ScopeConsistencyValidator:
    """Reject claims that attribute one document's content to another document
    or to an article of a different document.

    The rule is deliberately general rather than tied to any specific
    regulation pair. It checks two orthogonal scopes declared by the question
    and by the answer text:

    * **Article scope** — when a concrete article number is declared (e.g.
      ``25.981``), every cited chunk must belong to the document that contains
      that article, or itself reference a named article.
    * **File scope** — when a document marker is declared (e.g. ``CCAR-25-R4``),
      every cited chunk must belong to that document, or the claim must
      explicitly name the chunk's own document.

    A chunk that violates either scope without being explicitly attributed is
    treated as cross-document contamination and rejected.
    """

    def validate(
        self,
        answer: StructuredAnswer,
        *,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
    ) -> None:
        declared_articles = self._declared_articles(query.original_query, answer.answer)
        declared_files = _file_markers(query.original_query) | _file_markers(answer.answer)
        if not declared_articles and not declared_files:
            return
        chunk_articles = {
            chunk.chunk_id: _leading_article(chunk.text) for chunk in chunks
        }
        chunk_documents = {
            chunk.chunk_id: _leading_document(chunk.text) for chunk in chunks
        }
        chunk_files = {
            chunk.chunk_id: self._chunk_file_markers(chunk, chunk_documents.get(chunk.chunk_id))
            for chunk in chunks
        }
        primary_docs = self._primary_documents(
            chunks, declared_articles, chunk_articles, chunk_documents
        )
        # Article-level checks are only meaningful when at least one declared
        # article is actually present in the evidence pool. File-level checks
        # stay independent so a missing article anchor cannot disable them.
        article_enforceable = bool(declared_articles) and any(primary_docs.values())
        if not article_enforceable and not declared_files:
            return
        chunk_by_citation = {item.citation_id: item for item in chunks}
        for claim in answer.claims:
            self._validate_claim(
                claim,
                declared_articles=declared_articles,
                declared_files=declared_files,
                chunk_articles=chunk_articles,
                chunk_documents=chunk_documents,
                chunk_files=chunk_files,
                primary_docs=primary_docs,
                article_enforceable=article_enforceable,
                chunk_by_citation=chunk_by_citation,
            )

    def _validate_claim(
        self,
        claim: CandidateClaim,
        *,
        declared_articles: dict[str, set[str]],
        declared_files: set[str],
        chunk_articles: dict[str, str | None],
        chunk_documents: dict[str, str | None],
        chunk_files: dict[str, set[str]],
        primary_docs: dict[str, set[str]],
        article_enforceable: bool,
        chunk_by_citation: dict[str, SelectedChunk],
    ) -> None:
        claim_articles = self._merge_aliases(
            declared_articles, extract_article_references(claim.claim)
        )
        claim_files = _file_markers(claim.claim)
        for citation_id in claim.citation_ids:
            chunk = chunk_by_citation.get(citation_id)
            if chunk is None:
                continue
            article = chunk_articles.get(chunk.chunk_id)
            document = chunk_documents.get(chunk.chunk_id)
            if not self._article_scope_allowed(
                chunk,
                article=article,
                document=document,
                declared_articles=declared_articles,
                claim_articles=claim_articles,
                primary_docs=primary_docs,
                enforceable=article_enforceable,
            ):
                raise ValueError(
                    f"claim {claim.claim_id} cites {citation_id} from document "
                    f"'{document}' which does not belong to any article named in "
                    f"the claim ({', '.join(sorted(declared_articles))})"
                )
            if not self._file_scope_allowed(
                chunk_files.get(chunk.chunk_id, set()),
                declared_files=declared_files,
                claim_files=claim_files,
            ):
                raise ValueError(
                    f"claim {claim.claim_id} cites {citation_id} from a document "
                    f"outside the declared scope ({', '.join(sorted(declared_files))}) "
                    f"without naming that document"
                )

    @staticmethod
    def _article_scope_allowed(
        chunk: SelectedChunk,
        *,
        article: str | None,
        document: str | None,
        declared_articles: dict[str, set[str]],
        claim_articles: dict[str, set[str]],
        primary_docs: dict[str, set[str]],
        enforceable: bool,
    ) -> bool:
        if not enforceable:
            return True
        # 1) The chunk's own article is one of the declared articles.
        if article is not None and article in declared_articles:
            return True
        # 2) The chunk belongs to the same document as a declared article.
        if document is not None:
            if any(
                document in primary_docs.get(article_id, set())
                for article_id in declared_articles
            ):
                return True
        # 3) Explicit cross-reference: the claim names the chunk's own article.
        if article is not None and article not in declared_articles and article in claim_articles:
            return True
        # 4) Chunks with no authoritative header fall back to containing the
        #    declared article's aliases (legacy format).
        if article is None:
            if any(
                text_contains_article_alias(chunk.text, aliases)
                for aliases in declared_articles.values()
            ):
                return True
        # 5) Without any signal we cannot prove cross-document contamination.
        if article is None and document is None:
            return True
        return False

    @staticmethod
    def _file_scope_allowed(
        chunk_files: set[str],
        *,
        declared_files: set[str],
        claim_files: set[str],
    ) -> bool:
        if not declared_files:
            return True
        if not chunk_files:
            return True
        if any(
            _file_marker_match(declared, candidate)
            for declared in declared_files
            for candidate in chunk_files
        ):
            return True
        if any(
            _file_marker_match(declared, candidate)
            for declared in claim_files
            for candidate in chunk_files
        ):
            return True
        return False

    @staticmethod
    def _chunk_file_markers(chunk: SelectedChunk, leading_document: str | None) -> set[str]:
        if leading_document:
            return _file_markers(leading_document)
        return _file_markers(chunk.metadata.document_name or "")

    @staticmethod
    def _declared_articles(*texts: str) -> dict[str, set[str]]:
        aliases: dict[str, set[str]] = {}
        for text in texts:
            for reference in extract_article_references(text):
                aliases.setdefault(reference.normalized_id, set()).update(reference.aliases)
        return aliases

    @staticmethod
    def _merge_aliases(
        base: dict[str, set[str]],
        references: list[ArticleReference],
    ) -> dict[str, set[str]]:
        merged = {article_id: set(alias_set) for article_id, alias_set in base.items()}
        for reference in references:
            merged.setdefault(reference.normalized_id, set()).update(reference.aliases)
        return merged

    @staticmethod
    def _primary_documents(
        chunks: list[SelectedChunk],
        declared_articles: dict[str, set[str]],
        chunk_articles: dict[str, str | None],
        chunk_documents: dict[str, str | None],
    ) -> dict[str, set[str]]:
        primary: dict[str, set[str]] = {article_id: set() for article_id in declared_articles}
        for chunk in chunks:
            article = chunk_articles.get(chunk.chunk_id)
            if article is None or article not in declared_articles:
                continue
            document = chunk_documents.get(chunk.chunk_id)
            if document:
                primary[article].add(document)
        return primary
