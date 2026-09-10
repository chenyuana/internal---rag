from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

from app.services.domain_terms import matched_concept_aliases, missing_concept_aliases

REGULATION_TARGET_CHARS = 800
REGULATION_MAX_CHARS = 1400
REGULATION_OVERLAP_CHARS = 100

_CHINESE_NUMERALS = "一二三四五六七八九十百千万零〇"
_LEVEL_TOKEN = rf"(?:[0-9]+|[A-Za-z]|[IVXivx]{{1,5}}|[{_CHINESE_NUMERALS}]+)"
_BARE_LEVEL_TOKEN = rf"[0-9{_CHINESE_NUMERALS}]+"
_BASE_TOKEN = (
    rf"(?:[A-Za-z]{{1,8}}\.)?\d+(?:\.\d+)*(?:[A-Za-z])?"
    rf"|[A-Za-z]?\d+(?:\.\d+)+(?:[A-Za-z])?"
    rf"|[{_CHINESE_NUMERALS}]+"
)
_EXPLICIT_LEVEL = (
    rf"(?:[（(]\s*{_LEVEL_TOKEN}\s*[）)]|"
    rf"第?\s*{_LEVEL_TOKEN}\s*(?:款|项|目))"
)
_MARKED_ARTICLE_RE = re.compile(
    rf"第\s*(?P<base>{_BASE_TOKEN})\s*条"
    rf"(?P<tail>\s*{_BARE_LEVEL_TOKEN}(?=$|[\s的规内要])|"
    rf"(?:\s*{_EXPLICIT_LEVEL})*)",
    re.IGNORECASE,
)
_BARE_ARTICLE_RE = re.compile(
    rf"(?<![A-Za-z0-9_.])(?P<base>"
    rf"(?:[A-Za-z]{{1,8}}\.\d+(?:\.\d+)*|[A-Za-z]?\d+(?:\.\d+)+(?:[A-Za-z])?)"
    rf")(?P<tail>(?:\s*[（(]\s*{_LEVEL_TOKEN}\s*[）)])*)",
    re.IGNORECASE,
)
_SECTION_ARTICLE_RE = re.compile(
    r"(?:§|Sec(?:tion)?\.?)\s*(?P<base>\d+\.\d+)"
    r"(?P<tail>(?:\s*\([A-Za-z0-9]+\))*)",
    re.IGNORECASE,
)
_WRAPPED_LEVEL_RE = re.compile(rf"[（(]\s*(?P<value>{_LEVEL_TOKEN})\s*[）)]")
_LABELED_LEVEL_RE = re.compile(
    rf"第?\s*(?P<value>{_LEVEL_TOKEN})\s*(?:款|项|目)",
    re.IGNORECASE,
)
_BARE_LEVEL_RE = re.compile(
    rf"^\s*(?P<value>{_BARE_LEVEL_TOKEN})(?=$|[\s的规内要])"
)
_ARTICLE_PREFIX_RE = re.compile(
    rf"^\s*(?:第\s*(?:{_BASE_TOKEN})\s*条|"
    rf"(?:[A-Za-z]{{1,8}}\.\d+(?:\.\d+)*|[A-Za-z]?\d+(?:\.\d+)+(?:[A-Za-z])?))",
    re.IGNORECASE,
)
_TECHNICAL_TOKEN_RE = re.compile(
    r"\b(?:[A-Za-z][A-Za-z0-9_.-]{1,30}|\d+(?:\.\d+)+[A-Za-z]?)\b"
)
_CHINESE_PHRASE_RE = re.compile(r"[\u3400-\u9fff]{2,20}")
_QUESTION_SUFFIX_RE = re.compile(r"[?？。；;:：]+$")
_APPENDIX_REFERENCE_RE = re.compile(r"\bAppendix\s+[A-Z0-9]+\b", re.IGNORECASE)
_PART_REFERENCE_RE = re.compile(r"\bPart\s+[IVX0-9]+\b", re.IGNORECASE)
_LIMIT_LANGUAGE_RE = re.compile(
    r"\b(?:must\s+not|may\s+not|shall\s+not|may\s+not\s+exceed|"
    r"shall\s+not\s+exceed|not\s+exceed|no\s+more\s+than|at\s+most)\b",
    re.IGNORECASE,
)
_ACCEPTANCE_HEADING_RE = re.compile(
    r"\b(?:requirements?|acceptance\s+criteria|pass(?:ing)?\s+criteria)\b",
    re.IGNORECASE,
)
_STRUCTURAL_TERMS = frozenset(
    {
        "appendix", "annex", "article", "chapter", "part", "section", "sec",
        "test", "procedure", "to",
    }
)
# A block's own caption/lead-in ("... are as follows:") names its subject far
# more precisely than the enclosing section title, which is usually the
# broader system-level heading ("Sec. 25.397 Control system loads." introducing
# a table of "Limit pilot forces and torques").  Both the chunker and the
# keyword builder must recognize it identically.
_LEAD_IN_CUE_RE = re.compile(
    r"(?:as\s+follows|listed\s+below|shown\s+below|as\s+listed|"
    r"(?:the\s+)?following\s+(?:apply|applies|table|tables|list|values?|data|"
    r"information|figures?|examples?|quantit(?:y|ies)))\s*[:.]?\s*$"
    r"|(?:如下|下述|下列|以下)\s*[:：]?\s*$",
    re.IGNORECASE,
)
_LEAD_IN_MAX_CHARS = 300
# Only the leading context lines are inspected: the caption of a chunk sits
# directly above its body, after the "section: ..." context header.
_LEAD_IN_SCAN_LINES = 4
_LEAD_IN_TOKEN_LIMIT = 6
# Function words and amendment verbs.  "By revising Sec. 25.603 to read as
# follows:" must contribute its subject, never "revising" or the cross-reference
# to a neighbouring section.
_ENGLISH_STOPWORDS = frozenset(
    {
        "a", "about", "above", "accordance", "add", "added", "adding", "all",
        "also", "amend", "amended", "amending", "amendment", "amendatory", "an",
        "and", "any", "apply", "applies", "are", "as", "at", "authority", "be",
        "been", "being", "below", "by", "citation", "citations", "continue",
        "continued", "continues", "data", "delete", "deleted", "deleting",
        "each", "either", "figure", "figures", "following", "follows", "for",
        "from", "given", "here", "hereby", "if", "in", "incorporate",
        "incorporated", "information", "insert", "inserted", "inserting",
        "into", "is", "it", "its", "lead", "list", "listed", "made", "may",
        "more", "must", "new", "no", "not", "of", "on", "or", "other",
        "provided", "pursuant", "read", "redesignate", "redesignated",
        "redesignating", "revise", "revised", "revising", "revision", "same",
        "shall", "should", "shown", "such", "table", "tables", "than", "that",
        "the", "their", "then", "therein", "thereof", "these", "this", "those",
        "to", "under", "use", "used", "using", "value", "values", "was",
        "were", "when", "where", "whereas", "which", "who", "whom", "will",
        "with", "would",
    }
)
_REFERENCE_TOKEN_RE = re.compile(r"^(?:[A-Za-z]{1,8}\.)?\d+(?:\.\d+)*[A-Za-z]?$")


@dataclass(frozen=True, slots=True)
class ArticleReference:
    raw: str
    base_id: str
    normalized_id: str
    parent_id: str | None
    levels: tuple[str, ...]
    aliases: tuple[str, ...]


def _normalized_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()


def _normalize_token(token: str) -> str:
    normalized = re.sub(r"\s+", "", _normalized_text(token))
    return normalized.upper() if re.search(r"[A-Za-z]", normalized) else normalized


def _levels_from_tail(tail: str) -> tuple[str, ...]:
    if not tail:
        return ()
    normalized = _normalized_text(tail)
    wrapped = [
        _normalize_token(match.group("value"))
        for match in _WRAPPED_LEVEL_RE.finditer(normalized)
    ]
    labeled = [
        _normalize_token(match.group("value"))
        for match in _LABELED_LEVEL_RE.finditer(normalized)
    ]
    if wrapped:
        return tuple(wrapped)
    if labeled:
        return tuple(labeled)
    bare = _BARE_LEVEL_RE.match(normalized)
    return (_normalize_token(bare.group("value")),) if bare else ()


def _aliases(raw: str, base_id: str, levels: tuple[str, ...]) -> tuple[str, ...]:
    normalized_id = base_id + "".join(f"({level})" for level in levels)
    values = [
        _normalized_text(raw),
        base_id,
        f"第{base_id}条",
        normalized_id,
        f"第{base_id}条" + "".join(f"({level})" for level in levels),
    ]
    if len(levels) == 1:
        values.extend(
            [
                f"第{base_id}条{levels[0]}",
                f"第{base_id}条第{levels[0]}款",
                f"{base_id}条{levels[0]}",
            ]
        )
    return tuple(dict.fromkeys(value for value in values if value))


def _reference(match: re.Match[str]) -> ArticleReference:
    raw = match.group(0)
    base_id = _normalize_token(match.group("base"))
    levels = _levels_from_tail(match.group("tail") or "")
    normalized_id = base_id + "".join(f"({level})" for level in levels)
    parent_id = None
    if levels:
        parent_id = base_id + "".join(f"({level})" for level in levels[:-1])
    return ArticleReference(
        raw=_normalized_text(raw),
        base_id=base_id,
        normalized_id=normalized_id,
        parent_id=parent_id,
        levels=levels,
        aliases=_aliases(raw, base_id, levels),
    )


def extract_article_references(text: str) -> list[ArticleReference]:
    """Extract marked and bare regulation identifiers while preserving hierarchy."""
    normalized = _normalized_text(text)
    references: list[ArticleReference] = []
    occupied: list[tuple[int, int]] = []
    for pattern in (_MARKED_ARTICLE_RE, _BARE_ARTICLE_RE):
        for match in pattern.finditer(normalized):
            if any(match.start() < end and match.end() > start for start, end in occupied):
                continue
            reference = _reference(match)
            if reference.normalized_id in {item.normalized_id for item in references}:
                continue
            references.append(reference)
            occupied.append(match.span())
    return references


def match_article_heading(text: str) -> ArticleReference | None:
    """Return an article only when the identifier starts the logical line."""
    normalized = _normalized_text(text)
    section = _SECTION_ARTICLE_RE.match(normalized)
    if section is not None and normalized[section.end():].strip():
        remainder = normalized[section.end():].lstrip(" .:—–-")
        if re.match(
            r"(?:as|is|are|was|were|would|should|must|requires?|applies|has|had|"
            r"by|to|and|of|in|for|the)\b", remainder, re.I,
        ):
            return None
        return _reference(section)
    marked = _MARKED_ARTICLE_RE.match(normalized)
    if marked is not None:
        return _reference(marked)
    bare = _BARE_ARTICLE_RE.match(normalized)
    if bare is None or not normalized[bare.end() :].strip():
        return None
    references = extract_article_references(normalized)
    return references[0] if references else None


def article_aliases_from_query(text: str) -> list[str]:
    aliases: list[str] = []
    for reference in extract_article_references(text):
        aliases.extend(reference.aliases)
    return list(dict.fromkeys(aliases))


def extract_keywords(
    title: str,
    section_path: Iterable[str],
    article_aliases: Iterable[str],
    *,
    text: str = "",
    table_rows: Iterable[Iterable[str]] | None = None,
    limit: int = 20,
) -> list[str]:
    """Build controlled, chunk-specific search anchors.

    Structural labels remain in the published source context, but are not
    repeated as high-weight keywords for every child chunk. This avoids
    ranking a whole appendix as a single indistinguishable topic.

    Priority under the budget: answer-type anchors, the block's own caption,
    the table's own field/row identifiers, bilingual glossary counterparts,
    clause citations, then the leaf title's terms.  The Part/Chapter banner is
    never expanded into terms.
    """

    context = [str(value) for value in section_path if str(value).strip()]
    candidates: list[str] = []
    acceptance = bool(
        _LIMIT_LANGUAGE_RE.search(text)
        and _ACCEPTANCE_HEADING_RE.search(title + "\n" + text)
    )
    if acceptance:
        # Put answer-type anchors first so they survive the bounded keyword
        # budget even when a chunk contains many domain concepts.
        candidates.extend(["通过标准", "合格判据", "验收标准", "限制条件"])
    # The caption that introduces the body ("Limit pilot forces and torques ...
    # are as follows:") is the chunk's own subject.  Under a broad section
    # title ("Control system loads") it is the only text that names what the
    # body actually contains, so it must survive the budget.
    lead_in = table_lead_in_line(text)
    if lead_in is not None:
        candidates.extend(_lead_in_terms(lead_in, _LEAD_IN_TOKEN_LIMIT))
    # A table's own structure names its subject even when the caption is thin:
    # "Proposal No. / Notice No. / Federal Register Citation" or "Control /
    # Maximum forces or torques" say what the rows mean, the first column
    # identifies the rows, and identifier-shaped cells carry the codes a
    # question is likely to name.
    candidates.extend(table_anchor_terms(table_rows))
    # Indexed content already holds the body verbatim, so a glossary term the
    # text itself spells out is worth nothing as an anchor; only the
    # counterpart form (for example 副翼 for "aileron") bridges a query that
    # the source language cannot match.
    candidates.extend(missing_concept_aliases(text))
    candidates.extend(_collapse_article_aliases(article_aliases))
    for value in [*context, title]:
        normalized = _QUESTION_SUFFIX_RE.sub("", _normalized_text(value))
        normalized = _ARTICLE_PREFIX_RE.sub("", normalized).strip(" -—:：")
        if normalized and not normalized.isascii():
            candidates.append(normalized)
        # Keep compact structural references (for example ``Appendix F``) and
        # the Chinese label of every level; only the *terms* of the leading
        # banner are dropped below.
        candidates.extend(_APPENDIX_REFERENCE_RE.findall(value))
        candidates.extend(_PART_REFERENCE_RE.findall(value))
        candidates.extend(_CHINESE_PHRASE_RE.findall(normalized))
    # A Part/Chapter banner ("PART 25 - AIRWORTHINESS STANDARDS: TRANSPORT
    # CATEGORY AIRPLANES") repeats on every chunk of the Part.  Its terms carry
    # no discriminating power and used to consume a third of the budget, so
    # they are taken from the leaf context and the title only; the banner still
    # reaches retrieval through ``section_path`` in the published context.
    for value in [*(context[1:] if len(context) > 1 else []), title]:
        candidates.extend(_technical_tokens(value))
    return _bounded_unique(candidates, limit)


def _technical_tokens(value: str, max_tokens: int | None = None) -> list[str]:
    """Split a label into search terms, dropping stopwords and citations."""
    tokens: list[str] = []
    seen: set[str] = set()
    for token in _TECHNICAL_TOKEN_RE.findall(value):
        key = token.casefold()
        if key in _STRUCTURAL_TERMS or key in _ENGLISH_STOPWORDS or len(token) < 2:
            continue
        if key in seen:
            continue
        seen.add(key)
        tokens.append(token)
        if max_tokens is not None and len(tokens) >= max_tokens:
            break
    return tokens


def _lead_in_terms(line: str, max_tokens: int | None = None) -> list[str]:
    """Terms of a caption, excluding the cross-references it may quote.

    "2-54. By revising the lead-in of Sec. 25.603 to read as follows:" must not
    publish ``25.603`` as a keyword of the section it precedes: that would
    associate a chunk with a neighbouring clause it merely points at.
    """
    tokens: list[str] = []
    for token in _technical_tokens(line):
        if _REFERENCE_TOKEN_RE.match(token):
            continue
        tokens.append(token)
        if max_tokens is not None and len(tokens) >= max_tokens:
            break
    return tokens


def _collapse_article_aliases(aliases: Iterable[str]) -> list[str]:
    """Keep one ASCII and one Chinese form per distinct citation.

    ``_aliases`` emits up to four Chinese spellings of the same clause
    ("第25.397条", "第25.397条第C款", ...).  Publishing all of them left no
    budget for the chunk's subject matter.
    """
    buckets: dict[str, list[str]] = {}
    for alias in aliases:
        value = str(alias).strip()
        if not value:
            continue
        core = re.sub(r"[第条款项目\s()（）]", "", _normalized_text(value)).casefold()
        buckets.setdefault(core, []).append(value)
    kept: list[str] = []
    for values in buckets.values():
        for form in (
            next((item for item in values if item.isascii()), None),
            next((item for item in values if not item.isascii()), None),
        ):
            if form and form not in kept:
                kept.append(form)
    return kept


_TABLE_FIELD_LIMIT = 4
_TABLE_VALUE_LIMIT = 2
_TABLE_IDENTIFIER_LIMIT = 3
_TABLE_VALUE_MAX_CHARS = 24
# Column labels that carry no subject on their own.
_TABLE_GENERIC_FIELDS = frozenset(
    {
        "no", "no.", "nos", "nos.", "number", "number.", "item", "items",
        "remarks", "remark", "do", "do.", "id", "section", "sec", "§",
        "序号", "项目", "名称", "备注", "说明", "内容",
    }
)
# A continued table repeats a banner row ("DISTRIBUTION TABLE—Continued", split
# by OCR into "...—COn" + "tinued") where a header row would be. It names the
# table, not its columns.
_TABLE_CONTINUATION_RE = re.compile(r"continu", re.IGNORECASE)
# Codes a question is likely to name: "23.33", "25.1305(a)", "75-10", "Pt. 23".
# The leading component must look like a part/clause number (no leading zero),
# which keeps measurements such as "0.403" or "0.99508" out.
_IDENTIFIER_VALUE_RE = re.compile(
    r"^(?:(?:Pt|Part|Sec|§)\.?\s*[1-9]\d{0,2}"
    r"|[1-9]\d{0,2}(?:\.\d+)+(?:\([A-Za-z0-9]+\))*"
    r"|[1-9]\d{0,2}-\d+)$",
    re.IGNORECASE,
)
_TABLE_CELL_MARKER_RE = re.compile(
    r"(?:[\*\u2020\u2021]+|(?<=[A-Za-z\u3400-\u9fff])\d{1,2})\s*$"
)


def _clean_cell(value: object) -> str:
    return " ".join(str(value).split()).strip()


def _clean_field(cell: str) -> str:
    """Normalize a column label.

    OCR sometimes merges a label with the value that followed it, so a label
    that contains the semantic "=" separator is cut at it; footnote markers
    ("torques2", "Wheel*") are not part of the label.  A label carrying a long
    qualifier ("Maximum forces or torques for design weight, weight equal to or
    less than 5,000 pounds") keeps its head, which is the searchable part.
    """
    field = _TABLE_CELL_MARKER_RE.sub("", cell).split("=")[0]
    field = re.split(r"[,;（]", field, maxsplit=1)[0].strip().rstrip(":;")
    if len(field) > 40:
        for pattern in (r"\s+[(（]", r"[,;]"):
            head = re.split(pattern, field, maxsplit=1)[0].strip()
            if len(head) <= 40:
                field = head
                break
        else:
            field = field[:40].rsplit(" ", 1)[0].strip() or field[:40]
    return field


def table_anchor_terms(rows: Iterable[Iterable[str]] | None) -> list[str]:
    """Search anchors taken from a table's own structure.

    ``table_to_semantic_text`` renders the first row as field names, so that is
    the header the anchors describe.  Bare cell values are numbers without a
    subject and are published only when they are identifier-shaped.
    """
    table = [
        [_clean_cell(cell) for cell in row]
        for row in (rows or [])
        if row is not None
    ]
    table = [row for row in table if any(row)]
    if not table:
        return []

    anchors: list[str] = []
    header_row = "".join(table[0])
    if _TABLE_CONTINUATION_RE.search(re.sub(r"[^A-Za-z]", "", header_row)):
        header_row = ""
    if header_row:
        for cell in table[0][: _TABLE_FIELD_LIMIT + 3]:
            field = _clean_field(cell)
            if not 1 < len(field) <= 40 or field.casefold() in _TABLE_GENERIC_FIELDS:
                continue
            if not re.search(r"[A-Za-z\u3400-\u9fff]", field):
                continue
            anchors.append(field)
            if len(anchors) >= _TABLE_FIELD_LIMIT:
                break

    values: list[str] = []
    for row in table[1:]:
        first = _TABLE_CELL_MARKER_RE.sub("", row[0] if row else "").strip(" .;:")
        if (
            1 < len(first) <= _TABLE_VALUE_MAX_CHARS
            and re.search(r"[A-Za-z\u3400-\u9fff]", first)
        ):
            values.append(first)
            if len(values) >= _TABLE_VALUE_LIMIT:
                break

    identifiers: list[str] = []
    for row in table[1:13]:
        for cell in row:
            if _IDENTIFIER_VALUE_RE.fullmatch(cell) and cell not in identifiers:
                identifiers.append(cell)
                if len(identifiers) >= _TABLE_IDENTIFIER_LIMIT:
                    break
        if len(identifiers) >= _TABLE_IDENTIFIER_LIMIT:
            break
    return [*anchors, *values, *identifiers]


def _bounded_unique(candidates: Iterable[str], limit: int) -> list[str]:
    """Deduplicate case-insensitively, preserving the first spelling."""
    kept: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        token = str(value).strip()
        if not 1 < len(token) <= 40:
            continue
        key = token.casefold() if token.isascii() else token
        if key in seen:
            continue
        seen.add(key)
        kept.append(token)
        if len(kept) >= limit:
            break
    return kept


def table_lead_in_line(text: str, *, max_chars: int = _LEAD_IN_MAX_CHARS) -> str | None:
    """Return the caption that introduces a block body, if the text has one.

    PDF extraction emits a table's lead-in sentence as an ordinary paragraph,
    so the chunker uses this to keep the caption with the table it introduces
    and the keyword builder uses it to anchor that table's real subject.
    """
    if not text:
        return None
    for line in text.splitlines()[:_LEAD_IN_SCAN_LINES]:
        stripped = line.strip()
        if not stripped or len(stripped) > max_chars:
            continue
        if _LEAD_IN_CUE_RE.search(stripped):
            return stripped
    return None


def build_question_aliases(
    article: ArticleReference | None,
    title: str,
    *,
    text: str = "",
    limit: int = 5,
) -> list[str]:
    """Generate factual templates only; no new regulatory claims are introduced."""
    questions: list[str] = []
    if article is not None:
        questions.extend(
            [
                f"{article.normalized_id}规定了什么？",
                f"第{article.base_id}条的内容是什么？",
            ]
        )
        if article.levels:
            questions.append(f"第{article.base_id}条第{article.levels[0]}款有什么要求？")
    cleaned_title = _ARTICLE_PREFIX_RE.sub("", _normalized_text(title)).strip(" -—:：")
    if cleaned_title and not cleaned_title.isascii():
        questions.extend(
            [
                f"{cleaned_title}有哪些要求？",
                f"如何理解{cleaned_title}？",
            ]
        )
    if _LIMIT_LANGUAGE_RE.search(text) and _ACCEPTANCE_HEADING_RE.search(title + "\n" + text):
        concept_aliases = matched_concept_aliases(text)
        chinese_subject = next(
            (value for value in concept_aliases if _CHINESE_PHRASE_RE.fullmatch(value)),
            None,
        )
        subject = chinese_subject or "该项测试"
        questions.extend(
            [
                f"{subject}的通过标准是什么？",
                f"{subject}的合格判据是什么？",
            ]
        )
    return list(dict.fromkeys(questions))[:limit]


def text_contains_article_alias(text: str, aliases: Iterable[str]) -> bool:
    compact = re.sub(r"\s+", "", _normalized_text(text)).upper()
    return any(
        re.sub(r"\s+", "", _normalized_text(alias)).upper() in compact
        for alias in aliases
        if alias
    )
