"""Evidence scope diagnostics derived from source text, never topic-to-law guesses."""
from __future__ import annotations

import re
from typing import TypedDict

from app.schemas.retrieval import NormalizedQuery, SelectedChunk
from app.services.article_identity import owned_articles, owns_requested_article

_OMISSION = re.compile(r"(?:\\?\*[ \t]*){3,}")
_AMENDMENT = re.compile(
    r"\b(?:amend|revise|revising|redesignating|adding)\b.{0,100}"
    r"\b(?:paragraph|section|Sec\.)", re.I | re.S,
)


class RegulationContext(TypedDict):
    requested_sections: list[str]
    evidence_sections: list[str]
    unverified_sections: list[str]
    amendment_excerpt_citations: list[str]
    instructions: str


def regulation_context(query: NormalizedQuery, chunks: list[SelectedChunk]) -> RegulationContext:
    requested = query.article_ids
    owned = sorted({section for chunk in chunks for section in owned_articles(chunk)})
    missing = [section for section in requested
               if not any(owns_requested_article(c, [section]) for c in chunks)]
    partial = [c.citation_id for c in chunks
               if owned_articles(c) and (_OMISSION.search(c.text) or _AMENDMENT.search(c.text))]
    return {
        "requested_sections": requested,
        "evidence_sections": owned,
        "unverified_sections": missing,
        "amendment_excerpt_citations": partial,
        "instructions": (
            "条号与题目主题必须分别核对。unverified_sections 只表示本次证据未证实，"
            "不得推断条款不存在，也不得按主题静默修改条号。"
            "amendment_excerpt_citations 包含修订或省略信号，仅支持本次展示内容；"
            "不能把星号省略部分补写成完整条款。区分评论建议、说明和最终条文。"
        ),
    }


def amendment_scope_note(chunks: list[SelectedChunk]) -> str | None:
    citations = [c.citation_id for c in chunks
                 if owned_articles(c) and (_OMISSION.search(c.text) or _AMENDMENT.search(c.text))]
    if not citations:
        return None
    return (
        "依据范围：以下来源含修订或省略标记，回答仅覆盖已展示内容，"
        "不代表该条款的完整要求。 " + " ".join(f"[{cid}]" for cid in dict.fromkeys(citations))
    )
