"""Recover explicit regulation headings without document-specific OCR corrections.

Only split at source-supported headings. Never infer section numbers from topic
names, amendment instruction numbers, or inline references. Bounding boxes remain
the original source block's bounds; no synthetic line coordinates are invented.
"""
from __future__ import annotations

import re
from typing import Any

_SECTION = re.compile(
    r"(?:§[ \t]*|Sec(?:tion)?\.[ \t]*)(?P<id>\d+\.\d+[A-Za-z]?)"
    r"[ \t]+(?P<title>[A-Z][^\n.]{1,160}\.)"
)
_PROSE = re.compile(
    r"\b(?:must|shall|should|would|requires?|applies|is|are|was|were|"
    r"proposed|revised|amended|adopted)\b", re.I,
)
_INSTRUCTION = re.compile(r"\bto read as follows:\s*$", re.I)


def split_regulatory_text(text: str) -> list[str]:
    """Separate explicit section titles joined to their body by OCR.

An embedded heading requires a line boundary or an amendment instruction ending
in 'to read as follows:'. This deliberately leaves ambiguous prose untouched.
"""
    spans: list[tuple[int, int]] = []
    for match in _SECTION.finditer(text):
        prefix = text[:match.start()]
        line_prefix = prefix.rsplit("\n", 1)[-1]
        if line_prefix.strip() and not _INSTRUCTION.search(line_prefix):
            continue
        if _PROSE.search(match["title"]):
            continue
        spans.append(match.span())
    if not spans:
        return [text]
    pieces: list[str] = []
    cursor = 0
    for start, end in spans:
        pieces.extend([text[cursor:start].strip(), text[start:end].strip()])
        cursor = end
    pieces.append(text[cursor:].strip())
    return [piece for piece in pieces if piece]


def expand_regulatory_blocks(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for item in items:
        if item.get("block_type") not in {None, "", "paragraph", "heading"}:
            expanded.append(item)
            continue
        if item.get("exclusion_reason"):
            expanded.append(item)
            continue
        pieces = split_regulatory_text(str(item.get("text") or ""))
        if len(pieces) <= 1:
            expanded.append(item)
            continue
        expanded.extend({**item, "text": piece, "block_type": "paragraph"} for piece in pieces)
    return expanded
