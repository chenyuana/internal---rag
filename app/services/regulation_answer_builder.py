from __future__ import annotations

import re
import unicodedata

from app.ingestion.regulations import match_article_heading
from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import NormalizedQuery, SelectedChunk

_HEADER_RE = re.compile(r"^(文档|章节|条号|页码)：", re.MULTILINE)
_DOCUMENT_RE = re.compile(r"^文档：\s*(.+)$", re.MULTILINE)
_ARTICLE_RE = re.compile(r"^条号：\s*(\S+)\s*$", re.MULTILINE)
_SECTION_RE = re.compile(r"^章节：\s*(.+)$", re.MULTILINE)
_CLAUSE_RE = re.compile(r"(?m)^[ \t]*\(([a-z])\)[ \t]*")
_ARTICLE_TITLE_RE = re.compile(
    r"第\s*[A-Za-z]?\d+(?:\.\d+)*\s*条\s*(.+)$",
    re.IGNORECASE,
)
_REQUIREMENT_MARKERS = ("要求", "规定", "是什么", "有哪些", "如何")
_TOP_SECTION_RE = re.compile(r"(?m)^(?P<number>\d{1,2})\s+(?P<title>[^\n]+?)\s*$")
_SUBSECTION_RE = re.compile(
    r"(?m)^(?P<number>\d{1,2}(?:\.\d+)+)\s+(?P<title>[^\n]+?)\s*$"
)
_HTML_RE = re.compile(r"<[^>]+>")
_NON_PROCESS_TITLES = {
    "范围",
    "规范性引用文件",
    "术语和定义",
    "基本要求",
    "安全注意事项",
}


def _header(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(1).strip() if match else None


def _body(text: str) -> str:
    # Blank lines are valid source paragraph boundaries, not proof of a header.
    lines = text.splitlines()
    while lines and (_HEADER_RE.match(lines[0]) or not lines[0].strip()):
        lines.pop(0)
    return "\n".join(lines).strip()


def _trim_clause(text: str) -> str:
    lines: list[str] = []
    for line in text.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith("〔"):
            break
        lines.append(stripped)
    return "\n".join(line for line in lines if line).strip()


def _normalized_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).lower()
    return "".join(character for character in normalized if character.isalnum())


class RegulationAnswerBuilder:
    """Build a verbatim answer for exact regulation-article requirement queries.

    Logical qualifiers in regulations are easy for a small language model to
    broaden accidentally (for example changing "(b) does not apply" into
    "this article does not apply"). When evidence has already been narrowed to
    one structured article, returning its top-level clauses verbatim is safer
    and more complete than abstractive generation.
    """

    def build(
        self,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
    ) -> StructuredAnswer | None:
        if query.query_type == "procedure":
            procedure = self._build_procedure(query, chunks)
            if procedure is not None:
                return procedure
        if not chunks or not any(
            marker in query.original_query for marker in _REQUIREMENT_MARKERS
        ):
            return None
        if any(marker in query.original_query for marker in (
            "发布日期", "生效日期", "评论", "收益", "成本", "目的", "为什么",
        )):
            return None
        # 按来源（文档 + 条号/章节）分组，逐条提取要求原文（字母/数字分项或无分项）。
        # fact 题纯确定性，LLM 不参与正文。
        groups: dict[tuple[str, str, str], list[SelectedChunk]] = {}
        for chunk in sorted(chunks, key=lambda c: (
            c.document_id, c.metadata.page_number or 0, c.metadata.chunk_index or 0,
        )):
            document = _header(_DOCUMENT_RE, chunk.text) or chunk.metadata.document_name or ""
            article = (
                _header(_ARTICLE_RE, chunk.text)
                or chunk.metadata.section_id
                or _header(_SECTION_RE, chunk.text)
                or chunk.metadata.chapter_path
                or ""
            )
            groups.setdefault((chunk.document_id, document, article), []).append(chunk)

        claims: list[CandidateClaim] = []
        answer_parts: list[str] = []
        claim_index = 1
        for (_, document, article), group_chunks in groups.items():
            clauses: dict[str, tuple[str, str]] = {}
            for chunk in group_chunks:
                body = _body(chunk.text)
                matches = list(_CLAUSE_RE.finditer(body))
                if matches:
                    preamble = _trim_clause(body[:matches[0].start()])
                    preamble_lines = preamble.splitlines()
                    if preamble_lines and match_article_heading(preamble_lines[0]):
                        preamble = "\n".join(preamble_lines[1:]).strip()
                    if preamble:
                        clauses.setdefault(
                            f"intro-{chunk.citation_id}", (preamble, chunk.citation_id),
                        )
                    for index, match in enumerate(matches):
                        end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
                        clause = _trim_clause(body[match.start() : end])
                        if not clause:
                            continue
                        # Identically named paragraphs can span chunks. Keep
                        # distinct continuations instead of choosing the longest
                        # and silently deleting qualifications from the other.
                        key = f"{match.group(1)}-{chunk.citation_id}-{index}"
                        if not any(clause == value[0] for value in clauses.values()):
                            clauses[key] = (clause, chunk.citation_id)
                else:
                    clause = _trim_clause(body)
                    if clause:
                        clauses.setdefault(
                            f"plain-{chunk.citation_id}", (clause, chunk.citation_id),
                        )
            if not clauses:
                continue
            doc_label = document.removesuffix(".pdf") if document else ""
            article_label = article.rsplit("/", maxsplit=1)[-1].strip() if article else ""
            if doc_label and article_label:
                answer_parts.append(f"【{doc_label} · {article_label}】")
            elif doc_label or article_label:
                answer_parts.append(f"【{doc_label or article_label}】")
            for clause, citation_id in clauses.values():
                claims.append(
                    CandidateClaim(
                        claim_id=f"fact-{claim_index}",
                        claim=clause,
                        citation_ids=[citation_id],
                    )
                )
                claim_index += 1
                answer_parts.append(f"{clause} [{citation_id}]")

        if not claims:
            return None
        return StructuredAnswer(
            answerability="ANSWERABLE",
            answer="\n\n".join(answer_parts),
            claims=claims,
            missing_information=[],
            conflicts=[],
        )

    def _build_procedure(
        self,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
    ) -> StructuredAnswer | None:
        """Build a deterministic, section-ordered process answer.

        Complete-procedure questions are especially vulnerable to model
        omission. Published standards already encode their workflow in ordered
        top-level sections, so preserve that order and summarize each section
        from its own cited chunk instead of asking a model to reconstruct it.
        """
        by_document: dict[str, list[SelectedChunk]] = {}
        for chunk in chunks:
            name = chunk.metadata.document_name or ""
            if name:
                by_document.setdefault(name, []).append(chunk)
        rendered_documents: list[str] = []
        claims: list[CandidateClaim] = []
        claim_index = 1
        for document_name, document_chunks in by_document.items():
            sections = self._document_process_sections(query.original_query, document_chunks)
            if len(sections) < 2:
                continue
            lines = [f"【{document_name}】"]
            for step_index, (heading, summary, citation_id) in enumerate(
                sections,
                start=1,
            ):
                claim = f"{heading}：{summary}"
                claims.append(
                    CandidateClaim(
                        claim_id=f"procedure-{claim_index}",
                        claim=claim,
                        citation_ids=[citation_id],
                    )
                )
                claim_index += 1
                lines.append(f"{step_index}. {claim} [{citation_id}]")
            rendered_documents.append("\n".join(lines))
        if not rendered_documents or not claims:
            return None
        return StructuredAnswer(
            answerability="ANSWERABLE",
            answer="\n\n".join(rendered_documents),
            claims=claims,
            missing_information=[],
            conflicts=[],
        )

    @classmethod
    def _document_process_sections(
        cls,
        question: str,
        chunks: list[SelectedChunk],
    ) -> list[tuple[str, str, str]]:
        sections: dict[tuple[int, str], tuple[str, str]] = {}
        declared_titles = cls._declared_section_titles(chunks)
        for chunk in chunks:
            body = _body(chunk.text)
            matches = list(_TOP_SECTION_RE.finditer(body))
            for index, match in enumerate(matches):
                number = int(match.group("number"))
                title = match.group("title").strip()
                end = matches[index + 1].start() if index + 1 < len(matches) else len(body)
                segment = body[match.end() : end].strip()
                summary = cls._procedure_section_summary(segment)
                if not summary:
                    continue
                key = (number, title)
                existing = sections.get(key)
                if existing is None or len(summary) > len(existing[0]):
                    sections[key] = (summary, chunk.citation_id)
            # Published chunks can start at a subsection (``6.1``) and carry
            # the missing top-level heading only in the metadata prefix:
            # ``章节：6 作业准备 / 6.1 飞行平台组装和调试``.
            section_path = _header(_SECTION_RE, chunk.text)
            root = section_path.split("/", maxsplit=1)[0].strip() if section_path else ""
            root_match = re.fullmatch(r"(?P<number>\d{1,2})\s+(?P<title>.+)", root)
            if root_match is not None:
                number = int(root_match.group("number"))
                title = root_match.group("title").strip()
                first_later_heading = next(
                    (
                        match.start()
                        for match in matches
                        if int(match.group("number")) > number
                    ),
                    len(body),
                )
                summary = cls._procedure_section_summary(body[:first_later_heading])
                if summary:
                    key = (number, title)
                    existing = sections.get(key)
                    if existing is None or len(summary) > len(existing[0]):
                        sections[key] = (summary, chunk.citation_id)
            first_subsection = _SUBSECTION_RE.search(body)
            if first_subsection is not None:
                number = int(first_subsection.group("number").split(".", maxsplit=1)[0])
                title = declared_titles.get(number)
                if title:
                    first_later_heading = next(
                        (
                            match.start()
                            for match in matches
                            if int(match.group("number")) > number
                        ),
                        len(body),
                    )
                    summary = cls._procedure_section_summary(body[:first_later_heading])
                    if summary:
                        key = (number, title)
                        existing = sections.get(key)
                        if existing is None or len(summary) > len(existing[0]):
                            sections[key] = (summary, chunk.citation_id)
        ordered = sorted(sections.items(), key=lambda item: item[0][0])
        candidates = [
            (number, title, summary, citation_id)
            for (number, title), (summary, citation_id) in ordered
            if title not in _NON_PROCESS_TITLES
            and not title.startswith("附录")
        ]
        if not candidates:
            return []

        explicit_start = next(
            (
                index
                for index, (_, title, _, _) in enumerate(candidates)
                if len(title) >= 3 and title in question
            ),
            None,
        )
        start = explicit_start or 0
        selected: list[tuple[str, str, str]] = []
        for _, title, summary, citation_id in candidates[start:]:
            selected.append((title, summary, citation_id))
            if "归档" in question and ("档案" in title or "归档" in title):
                break
        return selected

    @classmethod
    def _declared_section_titles(
        cls,
        chunks: list[SelectedChunk],
    ) -> dict[int, str]:
        known: dict[str, int] = {}
        declarations: list[str] = []
        for chunk in chunks:
            body = _body(chunk.text)
            for match in _TOP_SECTION_RE.finditer(body):
                known[match.group("title").strip()] = int(match.group("number"))
            compact = "".join(body.split())
            declaration_match = re.search(r"本文件规定了(?P<items>[^。]+)", compact)
            if declaration_match is not None:
                declarations.append(declaration_match.group("items"))
        for declared_text in declarations:
            head, separator, tail = declared_text.rpartition("和")
            if separator and "、" in head:
                declared_text = f"{head}、{tail}"
            raw_items = [
                item.strip("，,；; ")
                for item in re.split(r"、|，|,|以及", declared_text)
                if item.strip("，,；; ")
            ]
            items: list[str] = []
            for item in raw_items:
                if "术语和定义" in item:
                    item = "术语和定义"
                items.append(item.removesuffix("等技术要求").strip())
            bases = [
                number - index
                for index, title in enumerate(items)
                if (number := known.get(title)) is not None
            ]
            if not bases:
                continue
            base = max(set(bases), key=bases.count)
            return {
                base + index: title
                for index, title in enumerate(items)
                if title
            }
        return {}

    @classmethod
    def _procedure_section_summary(cls, segment: str) -> str:
        clean = _HTML_RE.sub(" ", segment)
        matches = list(_SUBSECTION_RE.finditer(clean))
        parts: list[str] = []
        if matches:
            for index, match in enumerate(matches[:6]):
                end = matches[index + 1].start() if index + 1 < len(matches) else len(clean)
                content = cls._compact_text(clean[match.end() : end], limit=110)
                label = f"{match.group('number')} {match.group('title').strip()}"
                parts.append(f"{label}（{content}）" if content else label)
        else:
            content = cls._compact_text(clean, limit=240)
            if content:
                parts.append(content)
        return "；".join(parts)

    @staticmethod
    def _compact_text(text: str, *, limit: int) -> str:
        compact = " ".join(text.split())
        if not compact:
            return ""
        if len(compact) <= limit:
            return compact
        boundary = max(
            compact.rfind(marker, 0, limit)
            for marker in ("。", "；", ";", "，", ",")
        )
        if boundary >= max(30, limit // 2):
            return compact[: boundary + 1]
        return compact[:limit].rstrip() + "…"
