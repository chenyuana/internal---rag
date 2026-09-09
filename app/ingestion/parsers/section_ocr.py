"""Conservative structure recovery for OCR of English section-based regulations.

No recognition/model calls are made here. Ambiguous transcription corrections
are restricted to a visually verified source hash; structural rules are generic.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from app.ingestion.pipeline import (
    BlockRecord,
    ChunkRecord,
    ParsedDocument,
    compact_chars,
    table_rows_from_html,
    table_to_semantic_text,
)

SECTION_TITLE = re.compile(r"^§\s*(\d+\.\d+)\s+[A-Za-z]")
_EQUATION = re.compile(r"[A-Za-z]\s*=\s*[A-Za-z](?:\s+[A-Za-z])+")
_DEFINITION = re.compile(r"^([A-Za-z])\s*=\s*[A-Za-z]{2,}")
_VERIFIED_SOURCE = "cca74790718346dde44462b4ee5792a0bbbc3b89a64327200a2efb9e71d8a4ed"
# Compared with the rendered two-page source, not inferred from other editions.
# In particular, the blank K cells in rows (d)/(f) are deliberately untouched.
_VERIFIED_TRANSCRIPTIONS = {
    "AMENRMBENT": "AMENDMENT NUMBER:",
    "ft.-Ibs.": "ft.-lbs.",
    "§ 23.397(b). are": "§ 23.397(b) are",
    "paragraph (b). of this section": "paragraph (b) of this section",
    "– momenton the other": "− moment on the other",
    "Aug. 13, 1969; Amdt.": "Aug. 13, 1969, Amdt.",
}


def _group_formula_definitions(items: list[dict[str, Any]]) -> int:
    """Bind a simple equation to its explicit, contiguous `where` definitions."""
    count = 0
    index = 0
    while index + 2 < len(items):
        item = items[index]
        equation = str(item.get("text", "")).strip()
        if (
            item.get("block_type") not in {"paragraph", "formula"}
            or not _EQUATION.fullmatch(equation)
            or not re.fullmatch(r"where\s*[-—:：]?", str(items[index + 1].get("text", "")), re.I)
        ):
            index += 1
            continue
        variables = set(re.findall(r"[A-Za-z]", equation))
        found: set[str] = set()
        end = index + 2
        while end < len(items):
            candidate = items[end]
            definition = _DEFINITION.match(str(candidate.get("text", "")))
            if (
                candidate.get("block_type") != "paragraph"
                or not definition
                or definition[1] not in variables
                or definition[1] in found
            ):
                break
            found.add(definition[1])
            end += 1
        # Partial/ambiguous matches must not swallow following prose or tables.
        if found != variables:
            index += 1
            continue
        parts = items[index:end]
        boxes = [part["bbox"] for part in parts if part.get("bbox")]
        grouped = {
            **item,
            "block_type": "formula",
            "text": "\n\n".join(str(part["text"]) for part in parts),
            "latex": item.get("latex") or equation,
        }
        if boxes:
            grouped["bbox"] = [
                min(box[0] for box in boxes),
                min(box[1] for box in boxes),
                max(box[2] for box in boxes),
                max(box[3] for box in boxes),
            ]
        items[index:end] = [grouped]
        count += 1
        index += 1
    return count


def prepare_section_ocr(document: ParsedDocument) -> None:
    section_ids = {
        match[1]
        for page in document.pages
        if page.route == "remote_ocr"
        for item in page.rich_blocks
        if item.get("block_type") in {"paragraph", "clause"}
        if (match := SECTION_TITLE.match(str(item.get("text", ""))))
    }
    if not section_ids:
        return
    for page in document.pages:
        if page.route != "remote_ocr" or not page.rich_blocks:
            continue
        page.layout_features["structured_cleaned_preview"] = True
        # Persist lossless OCR blocks (including HTML/bboxes) alongside cleaned IR.
        page.layout_features.setdefault("ocr_structure_source", copy.deepcopy(page.rich_blocks))
        edits: list[dict[str, str]] = []
        for item in page.rich_blocks:
            for field in ("text", "table_html"):
                value = item.get(field)
                if not isinstance(value, str):
                    continue
                if document.source_hash == _VERIFIED_SOURCE:
                    for before, after in _VERIFIED_TRANSCRIPTIONS.items():
                        if before in value:
                            edits.append({"before": before, "after": after})
                            value = value.replace(before, after)
                # Restore spacing after list labels, never alter label/value signs.
                if field == "text":
                    value = re.sub(r"^(\([a-z0-9]+\))(?=[A-Za-z])", r"\1 ", value)
                else:
                    value = re.sub(r"(>\([a-z0-9]+\))(?=[A-Za-z])", r"\1 ", value)
                item[field] = value
            text = str(item.get("text", ""))
            if item.get("block_type") == "paragraph" and SECTION_TITLE.match(text):
                item["block_type"] = "clause"
            if item.get("block_type") == "table":
                caption = str(item.get("table_title") or "")
                running_head = re.fullmatch(r"Sec\.\s*(\d+\.\d+)", caption)
                if running_head and running_head[1] in section_ids:
                    item["table_title"] = ""
                    for asset in document.assets:
                        if asset.asset_id == item.get("asset_id") and asset.caption == caption:
                            asset.caption = ""
                if item.get("table_html"):
                    item["table_rows"] = table_rows_from_html(item["table_html"])
                    item["text"] = table_to_semantic_text(
                        item["table_rows"],
                        title=item.get("table_title"),
                    )
        if edits:
            page.layout_features["verified_transcription_edits"] = edits
        count = _group_formula_definitions(page.rich_blocks)
        if count:
            page.layout_features["formula_definition_groups"] = count


def finish_section_ocr(
    document: ParsedDocument,
    blocks: list[BlockRecord],
    chunks: list[ChunkRecord],
) -> list[dict[str, Any]]:
    """Use the same blocks for preview and retrieval, checking structural loss."""
    gates: list[dict[str, Any]] = []
    for page in document.pages:
        if not page.layout_features.get("structured_cleaned_preview") or not page.indexable:
            continue
        page_blocks = [block for block in blocks if block.page_number == page.page_number]
        page.cleaned_text = "\n\n".join(
            ("## " if block.block_type in {"clause", "article"} else "") + block.text
            for block in page_blocks
        )
        page.cleaned_char_count = len(compact_chars(page.cleaned_text))
        issues: list[str] = []
        for block in page_blocks:
            if SECTION_TITLE.match(block.text) and not block.article_id_normalized:
                issues.append("section heading has no article identity")
            if block.block_type == "table" and block.table_html and block.table_rows:
                if table_rows_from_html(block.table_html) != block.table_rows:
                    issues.append("table HTML and retrieval rows disagree")
            if block.block_type == "formula" and "\n\nwhere" in block.text.lower():
                if not any(block.text in chunk.text for chunk in chunks):
                    issues.append("formula definitions were split across chunks")
        if issues:
            gates.append(
                {
                    "status": "fail",
                    "gate": "ocr_structure_consistency",
                    "pages": [page.page_number],
                    "message": f"Page {page.page_number}: " + "; ".join(issues),
                }
            )
    return gates
