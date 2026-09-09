"""Resolve section ownership separately from references occurring in the body."""
from __future__ import annotations

import re

from app.ingestion.regulations import extract_article_references, match_article_heading
from app.schemas.retrieval import RetrievedChunk, SelectedChunk


def owned_articles(chunk: RetrievedChunk | SelectedChunk) -> set[str]:
    # Explicit metadata and publication headers are authoritative ownership.
    header = re.search(r"^条号：[ \t]*(\S+)[ \t]*$", chunk.text, re.M)
    values = [chunk.metadata.section_id or "", header[1] if header else ""]
    ids = {
        ref.normalized_id
        for value in values
        for ref in extract_article_references(value)
    }
    if ids:
        return ids
    section = re.search(r"^章节：[ \t]*(.+)$", chunk.text, re.M)
    titles = [chunk.metadata.section_title, chunk.metadata.chapter_path,
              section[1] if section else None]
    for title in titles:
        if title:
            for leaf in title.split(" / "):
                ref = match_article_heading(leaf)
                if ref:
                    ids.add(ref.normalized_id)
    if ids:
        return ids
    # Legacy untagged chunks: only a leading heading, never an arbitrary mention.
    body = re.sub(r"^(?:文档|章节|条号|页码|图表)：[^\n]*\n?", "", chunk.text, flags=re.M)
    first = body.strip().splitlines()
    if first:
        ref = match_article_heading(first[0])
        if ref:
            ids.add(ref.normalized_id)
    return ids


def owns_requested_article(chunk: RetrievedChunk | SelectedChunk, ids: list[str]) -> bool:
    owned = owned_articles(chunk)
    return any(
        actual == requested or actual.startswith(requested + "(")
        for actual in owned for requested in ids
    )
