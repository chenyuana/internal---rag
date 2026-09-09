"""LLM-driven structure annotation for regulation documents.

The heuristic heading classification in ``pipeline.split_blocks`` /
``build_chunks`` is tuned for standards/regulation bodies and mis-binds two
Federal-Register patterns:

* a ``Sec. NNN ...`` running / amended-section header that stays active while
  an appendix (``Amend Appendix F`` / ``A. Redesignate`` / ``B. Add a new Part
  II ... Appendix F ... Test Method``) is introduced, so appendix content is
  published under the wrong section (``23.1587(H)(1)`` instead of ``Appendix F``);
* preamble discussion paragraphs that open with ``Section 23.NNN <verb> ...``
  (e.g. ``Section 23.571(d) still requires the damage tolerance option ...``)
  and are mistaken for clause headings, attaching a fake
  ``article_id`` (``23.571(D)``) to prose.

This module runs **after** a parser has built a ``ParsedDocument`` and re-annotates
each chunk's ``section_path / article_id_* / title / article_aliases / keywords``
from LLM-provided reading-order boundary events. It never rewrites body text
(preserving numbers/citations) and falls back to the heuristic metadata on any
error, so it can be enabled without risk.
"""

# ruff: noqa: E501 — prompt templates exceed the line limit by design.
from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx

from app.core.config import LlmStructureSettings
from app.ingestion.regulations import extract_keywords

_JSON_ONLY = re.compile(r"^\s*[\[{]", re.MULTILINE)


def _find_json(text: str) -> str | None:
    """Return the first JSON array/object embedded in the model output."""
    start = _JSON_ONLY.search(text)
    if not start:
        return None
    # Try the whole tail, then progressively trim trailing junk.
    tail = text[start.start():]
    for trim in range(0, 200):
        candidate = tail[: len(tail) - trim] if trim else tail
        try:
            json.loads(candidate)
            return candidate
        except ValueError:
            continue
    return None


@dataclass(slots=True)
class StructureEvent:
    page: int
    seq: int
    kind: str            # annex | article | clause | chapter | body | table | figure
    section_path: list[str] = field(default_factory=list)
    article_id: str | None = None
    title: str | None = None
    line: str = ""


def _coerce_event(raw: dict[str, Any], page: int, seq: int) -> StructureEvent:
    sp = raw.get("section_path") or []
    if isinstance(sp, str):
        sp = [sp]
    return StructureEvent(
        page=int(raw.get("page", page)),
        seq=seq,
        kind=str(raw.get("kind") or "body").casefold(),
        section_path=[str(v) for v in sp if str(v).strip()],
        article_id=(str(raw["article_id"]) if raw.get("article_id") else None),
        title=(str(raw["title"]) if raw.get("title") else None),
        line=str(raw.get("line") or "")[:80],
    )


def _event_position(event: StructureEvent) -> tuple[int, int]:
    return (event.page, event.seq)


def _chunk_position(chunk: Any, order: int) -> tuple[int, int]:
    page = getattr(chunk, "page_start", None) or 0
    return (int(page), order)


def apply_events(
    chunks: Sequence[Any],
    events: Sequence[StructureEvent],
    *,
    set_attr: Callable[[Any, str, Any], None] | None = None,
) -> int:
    """Re-annotate chunks from reading-order boundary events.

    For each chunk (in document order) the active section is the last
    **non-body** boundary event on or before its position. ``body`` events
    (preamble reference prose) do not open a section, so the chunk keeps the
    enclosing section instead of inheriting a fake article id.
    Returns the number of chunks whose metadata changed.
    """
    if not events:
        return 0
    ordered = sorted((e for e in events if e.kind != "body"), key=_event_position)
    box = {"idx": -1, "active": None}
    changed = 0
    for order, chunk in enumerate(chunks):
        pos = _chunk_position(chunk, order)
        while box["idx"] + 1 < len(ordered) and _event_position(ordered[box["idx"] + 1]) <= pos:
            box["idx"] += 1
            box["active"] = ordered[box["idx"]]
        active = box["active"]
        section = list(active.section_path) if active else []
        title = (active.title if active else None) or (section[-1] if section else "Document preamble")
        article = active.article_id if active else None
        article_aliases: list[str] = []
        keywords = extract_keywords(title, section, article_aliases)
        updated = (
            getattr(chunk, "section_path", None) != section
            or getattr(chunk, "title", None) != title
            or getattr(chunk, "article_id_normalized", None) != article
            or getattr(chunk, "keywords", None) != keywords
        )
        if updated:
            changed += 1
        assign = set_attr or setattr
        assign(chunk, "section_path", section)
        assign(chunk, "title", title)
        assign(chunk, "article_id_raw", article)
        assign(chunk, "article_id_normalized", article)
        assign(chunk, "article_aliases", article_aliases)
        assign(chunk, "keywords", keywords)
    return changed


def build_prompt(page_texts: Sequence[tuple[int, str]]) -> str:
    parts: list[str] = []
    for page, text in page_texts:
        parts.append(f"[PAGE {page}]\n{text}")
    body = "\n\n".join(parts)
    return (
        "You annotate a US Federal Register amendment document for RAG chunking. "
        "Given OCR pages, output JSON ONLY (a JSON array, no prose). "
        "For every event emit: {\"page\":int,\"kind\":\"annex|article|clause|chapter|"
        "body|table|figure\",\"section_path\":[str,...],\"article_id\":str|null,"
        "\"title\":\"<short heading title>\",\"line\":\"<first ~50 chars>\"}, in reading order.\n"
        "Rules:\n"
        "1. A short section header is a CLAUSE: \"Sec. 23.856 Thermal/acoustic insulation "
        "materials.\" -> article_id=23.856. \"Sec. 23.1587 Performance information.\" -> "
        "article_id=23.1587.\n"
        "2. Amendment/appendix boundary lines open an ANNEX: \"63. Amend Appendix F to Part 23 as "
        "follows:\" (section_path=[\"Appendix F\"]), \"A. Redesignate the existing text as Part I ...\" "
        "([\"Appendix F\",\"Part I\"]), \"B. Add a new Part II. Appendix F to Part 23--Test Method ...\" "
        "([\"Appendix F\",\"Part II\"]); article_id=null.\n"
        "3. A line that starts with \"Section 23.NNN ...\" or \"Sec. 23.NNN ...\" followed by a verb "
        "(still/also/does/requires/specifies/cannot/is/are/may/will/shall/states/...) is PROSE in the "
        "preamble, NOT a heading: kind=body, section_path=[], article_id=null.\n"
        "4. Ordinary body/definitions/test-apparatus/(a)/(b)/(c) items are kind=body.\n"
        "5. Chapter/PART/subpart headers are kind=chapter.\n"
        "Emit events for every header/appendix/article/chapter boundary AND for the "
        "Section-reference prose lines of rule 3. Do NOT rewrite body text; keep page numbers intact.\n\n"
        f"{body}"
    )


class LlmStructureAnnotator:
    """Re-annotate chunk structure metadata via an OpenAI-compatible chat API."""

    def __init__(self, settings: LlmStructureSettings) -> None:
        self._settings = settings
        headers = {}
        if settings.api_key.get_secret_value():
            headers["Authorization"] = "Bearer " + settings.api_key.get_secret_value()
        self._client = httpx.Client(
            base_url=settings.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=settings.timeout_seconds,
            trust_env=False,
        )

    def close(self) -> None:
        self._client.close()

    def _call(self, page_texts: Sequence[tuple[int, str]]) -> list[StructureEvent]:
        if not self._settings.api_key.get_secret_value():
            raise RuntimeError("llm_structure requires an API key")
        system = (
            "You are a deterministic document structure annotator for RAG chunking. "
            "Output ONLY a JSON array of boundary events. Do not reason or explain; "
            "emit the JSON immediately. Never rewrite the source text."
        )
        body = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": build_prompt(page_texts)},
            ],
            "temperature": self._settings.temperature,
            "max_tokens": self._settings.max_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
        resp = self._request(body, thinking_disabled=True)
        if resp.status_code in (400, 422):
            # Some OpenAI-compatible backends reject the `thinking` flag; retry plain.
            body.pop("thinking", None)
            resp = self._request(body, thinking_disabled=False)
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        raw = _find_json(content)
        if raw is None:
            raise ValueError("LLM structure output contained no JSON")
        payload = json.loads(raw)
        if isinstance(payload, dict):
            payload = payload.get("events") or payload.get("result") or []
        if not isinstance(payload, list):
            raise ValueError("LLM structure output is not a list")
        events: list[StructureEvent] = []
        for seq, item in enumerate(payload):
            if not isinstance(item, dict):
                continue
            events.append(_coerce_event(item, int(item.get("page", 0) or 0), seq))
        return events

    def _request(self, body: dict[str, Any], *, thinking_disabled: bool) -> httpx.Response:
        payload = dict(body)
        if thinking_disabled:
            payload["thinking"] = {"type": "disabled"}
        return self._client.post("chat/completions", json=payload)

    def annotate_document(self, document: Any) -> list[str]:
        """Given a ParsedDocument, re-annotate all chunks. Returns warnings."""
        warnings: list[str] = []
        pages = getattr(document, "pages", None) or []
        chunks = getattr(document, "chunks", None) or []
        if not pages or not chunks:
            return ["llm_structure: no pages/chunks to annotate"]
        page_texts: list[tuple[int, str]] = []
        for page in pages:
            pn = int(getattr(page, "page_number", 0) or 0)
            text = getattr(page, "cleaned_text", None) or getattr(page, "text", "") or ""
            if text:
                page_texts.append((pn, text))
        page_texts.sort(key=lambda item: item[0])
        events: list[StructureEvent] = []
        failed = False
        batch = self._settings.page_batch_size
        for start in range(0, len(page_texts), max(1, batch)):
            group = page_texts[start:start + max(1, batch)]
            try:
                events.extend(self._call(group))
            except Exception as exc:  # noqa: BLE001 - fall back to heuristic on any failure
                failed = True
                warnings.append(f"llm_structure: page batch {group[0][0]}-{group[-1][0]} "
                                f"failed ({type(exc).__name__}); kept heuristic metadata")
        if failed:
            warnings.append("llm_structure: atomic fallback kept all heuristic metadata")
            return warnings
        applied = apply_events(list(chunks), events)
        warnings.append(f"llm_structure: re-annotated {applied}/{len(chunks)} chunks")
        return warnings


def build_annotator(settings: LlmStructureSettings) -> LlmStructureAnnotator | None:
    if not settings.enabled:
        return None
    return LlmStructureAnnotator(settings)
