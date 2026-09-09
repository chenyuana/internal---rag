from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass

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
    limit: int = 16,
) -> list[str]:
    """Build deterministic search keywords without calling a generative model."""
    candidates: list[str] = []
    for value in [*section_path, title]:
        normalized = _QUESTION_SUFFIX_RE.sub("", _normalized_text(value))
        normalized = _ARTICLE_PREFIX_RE.sub("", normalized).strip(" -—:：")
        if normalized:
            candidates.append(normalized)
        candidates.extend(_TECHNICAL_TOKEN_RE.findall(value))
        candidates.extend(_CHINESE_PHRASE_RE.findall(normalized))
    candidates.extend(article_aliases)
    return list(dict.fromkeys(value for value in candidates if 1 < len(value) <= 40))[:limit]


def build_question_aliases(
    article: ArticleReference | None,
    title: str,
    *,
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
    if cleaned_title:
        questions.extend(
            [
                f"{cleaned_title}有哪些要求？",
                f"如何理解{cleaned_title}？",
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
