"""Evidence-preserving OCR for scanned English regulations.

Marginalia and uncertain annotations are retained in the IR outside search.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
import zipfile
from collections.abc import Callable
from dataclasses import fields, replace
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
import pdfplumber
import pypdfium2 as pdfium
from pypdf import PdfReader

from app.ingestion.parsers.base import ParserPlugin
from app.ingestion.parsers.figure_links import link_figures, restore_figure_names
from app.ingestion.parsers.header_recovery import recover_header_pages
from app.ingestion.parsers.remote import MinerUClient, RemoteParseResult, clean_inline_latex
from app.ingestion.parsers.table_assets import restore_table_assets
from app.ingestion.parsers.table_recovery import (
    crop_table_pdf,
    recover_two_columns,
    structure_issue,
)
from app.ingestion.parsers.verified_tables import verified_table
from app.ingestion.pipeline import (
    AssetRecord,
    BlockRecord,
    ChunkRecord,
    PageRecord,
    ParsedDocument,
    compact_chars,
    split_structured_table_block,
    stable_id,
    table_rows_from_html,
    table_to_semantic_text,
)
from app.ingestion.regulations import match_article_heading
from app.ingestion.regulatory_structure import expand_regulatory_blocks

# 23-48-FINAL RULE Amendment27805.pdf (gust-load lateral mass ratio, §23.443).
# MinerU mis-read the ``K^2 / l_vt`` denominator as ``K^2 / l_vt^2``; the
# rendered source page shows ``l_vt`` (no superscript).  Scoped to immutable
# PDF bytes so no other document inherits the correction.
AMEND27805_SHA256 = "71223230ad3283f5f942eab39f46bfeeb18be2c27db9b49262e0c4c48a4bdd4c"


def repair_formula_latex(blocks: list[BlockRecord]) -> None:
    """Source-verified formula LaTeX corrections against the rendered pages."""
    for block in blocks:
        if block.block_type != "formula" or not block.latex:
            continue
        if r"\frac{K^{2}}{l_{vt}^{2}}" in block.latex:
            block.latex = block.latex.replace(
                r"\frac{K^{2}}{l_{vt}^{2}}", r"\frac{K^{2}}{l_{vt}}"
            )


def _crop_formula_png(path: Path, page_number: int, bbox: list[float]) -> bytes:
    """Render a page and crop a formula region (0-1000 normalized bbox) to PNG.

    Used only when MinerU returned no crop for a formula block that otherwise
    carries bbox geometry, so the visual evidence is still available for review.
    """
    with pdfium.PdfDocument(path) as document:
        page = document[page_number - 1]
        try:
            bitmap = page.render(scale=3.0)
            try:
                image = bitmap.to_pil()
                box = (
                    image.width * max(0.0, bbox[0] - 2) / 1000,
                    image.height * max(0.0, bbox[1] - 2) / 1000,
                    image.width * min(1000.0, bbox[2] + 2) / 1000,
                    image.height * min(1000.0, bbox[3] + 2) / 1000,
                )
                crop = image.crop(box).convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()
    output = BytesIO()
    crop.save(output, format="PNG")
    return output.getvalue()


def recover_formula_image_assets(
    path: Path,
    pages: list[PageRecord],
    assets: list[AssetRecord],
) -> None:
    """Create formula image crops (from page bbox) for formula blocks lacking one.

    MinerU sometimes returns only the LaTeX for several display equations on a
    page while cropping just one of them (here page 10's ``L_vt``).  Restore the
    visual crop for the others so the review asset panel and the published chunk
    carry the same evidence.
    """
    existing = {asset.asset_id for asset in assets}
    for page in pages:
        for item in page.rich_blocks:
            if str(item.get("block_type", "")).casefold() != "formula":
                continue
            if not item.get("latex") or item.get("asset_id"):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            try:
                values = [float(value) for value in bbox[:4]]
                content = _crop_formula_png(path, page.page_number, values)
            except (OSError, ValueError, RuntimeError):
                continue
            asset_id = stable_id(
                "formula-crop-v1",
                page.page_number,
                round(values[0], 1),
                round(values[1], 1),
                round(values[2], 1),
                round(values[3], 1),
            )
            if asset_id in existing:
                continue
            assets.append(
                AssetRecord(
                    asset_id=asset_id,
                    page_number=page.page_number,
                    asset_type="formula",
                    bbox=values,
                    filename=f"{asset_id}.png",
                    mime_type="image/png",
                    content=content,
                    caption=str(item.get("latex") or ""),
                )
            )
            item["asset_id"] = asset_id
            item["asset_ids"] = [asset_id]
            existing.add(asset_id)


SCAN_PRODUCER_RE = re.compile(
    r"Pdf-It|Pdf\.Capture|\b(?:Capture|Scanner|Scan|Kofax|Nuance|ABBYY|IRIS|OCR)\b", re.I
)
_REGULATORY_RE = re.compile(
    r"\b(?:Federal\s*(?:Aviation|Register)|Title\s*14|PART\s*(?:23|25|91)|"
    r"Docket\s*No\.?|Civil\s*Air\s*Regulations|airworthiness|Aeronautics)\b",
    re.I,
)
_HEADING_PATTERNS = (
    ("title", r"Title\s+\d+\s*[—–-]\s*.+"),
    ("chapter", r"Chapter\s+[IVXLC\d]+\s*[—–-]\s*.+"),
    ("docket", r"\[?(?:Regulatory\s+)?Docket\s+No\.?\s+\d+.*"),
    ("part", r"PART\s+\d+\s*[—–-]\s*.+"),
    ("subpart", r"Subpart\s+[A-Z]\s*[—–-]+\s*.+"),
    ("appendix", r"Appendix\s+[A-Z\d]+\s*(?:[—–-]|to\b).+"),
    ("section", r"(?:§|Sec(?:tion)?\.)\s*[A-Z]?\d+\.\d+\s+.+"),
)
_RANK = {
    "title": 0,
    "chapter": 1,
    "docket": 2,
    "part": 3,
    "subpart": 4,
    "appendix": 4,
    "group": 5,
    "section": 6,
}


def _item_bbox(item: dict[str, Any]) -> tuple[float, float, float, float] | None:
    bbox = item.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    try:
        values = tuple(float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values  # type: ignore[return-value]


def _horizontal_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
    first = _item_bbox(left)
    second = _item_bbox(right)
    if first is None or second is None:
        return 0.0
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    return overlap / max(1.0, min(first[2] - first[0], second[2] - second[0]))


def _open_ended_text(text: str) -> bool:
    normalized = " ".join(text.split()).rstrip()
    return bool(
        re.search(
            r"\b(?:to|and|or|from|between|with|of|for|must)$",
            normalized,
            re.I,
        )
    )


def _continuation_text(text: str) -> bool:
    normalized = text.strip()
    if not normalized or english_heading(normalized):
        return False
    # A lower-case or numeric start is a strong signal that this is the
    # right-column continuation of a line ending in "to", "and", etc.
    return bool(re.match(r"^(?:\d|[a-z])", normalized))


def _noncontent_item(item: dict[str, Any]) -> bool:
    return item.get("block_type") == "annotation" or item.get("source_type") in {
        "header",
        "footer",
        "page_number",
        "discarded",
        "aside_text",
    }


def _identity_index(items: list[dict[str, Any]], target: dict[str, Any]) -> int:
    """Return an item's position without confusing equal-but-distinct blocks."""
    return next((index for index, item in enumerate(items) if item is target), -1)


def _repair_reading_order(
    items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Repair high-confidence OCR flow inversions without guessing prose.

    MinerU normally returns content-list order, but scanned multi-column pages
    can place a wide image or a right-column continuation before an earlier
    block. Only pages with a wide image anchor are eligible here; the original
    item order remains available through each item's ``source_ordinal`` and
    ``raw_ocr_item``.
    """
    result = list(items)
    geometric = [_item_bbox(item) for item in result]
    valid = [bbox for bbox in geometric if bbox is not None]
    if len(valid) < 4:
        return result, {"status": "unchanged", "moved_count": 0}
    page_width = max(bbox[2] for bbox in valid) - min(bbox[0] for bbox in valid)
    page_height = max(bbox[3] for bbox in valid) - min(bbox[1] for bbox in valid)
    if page_width <= 0 or page_height <= 0:
        return result, {"status": "unchanged", "moved_count": 0}

    moves: list[dict[str, Any]] = []

    # A continuation can safely cross a column boundary only when it lands at
    # the top of that column, or immediately below a wide figure that occupies
    # the column.  This prevents ordinary lower-column paragraphs from being
    # pulled ahead merely because they begin with a lower-case word.
    page_top = min(bbox[1] for bbox in valid)
    text_widths = [
        bbox[2] - bbox[0]
        for item, bbox in zip(result, [_item_bbox(value) for value in result], strict=False)
        if bbox is not None and item.get("block_type") != "figure"
    ]
    median_text_width = sorted(text_widths)[len(text_widths) // 2] if text_widths else 0.0
    wide_figure_bboxes = [
        bbox
        for item in result
        if item.get("block_type") == "figure"
        and (bbox := _item_bbox(item)) is not None
        and bbox[2] - bbox[0] >= max(page_width * 0.28, median_text_width * 1.45)
    ]
    if not wide_figure_bboxes:
        return result, {"status": "unchanged", "moved_count": 0}

    def _is_continuation_anchor(
        candidate_bbox: tuple[float, float, float, float],
        previous_bbox: tuple[float, float, float, float],
    ) -> bool:
        near_column_top = candidate_bbox[1] <= page_top + max(60.0, page_height * 0.18)
        below_wide_figure = any(
            figure_bbox[2] >= candidate_bbox[0] - page_width * 0.04
            and figure_bbox[0] <= candidate_bbox[2] + page_width * 0.04
            and figure_bbox[3]
            <= candidate_bbox[1]
            <= figure_bbox[3] + max(24.0, page_height * 0.04)
            for figure_bbox in wide_figure_bboxes
        )
        return (
            candidate_bbox[1] < previous_bbox[1] - max(30.0, page_height * 0.05)
            and (near_column_top or below_wide_figure)
        )

    # 1) Reconnect a wrapped continuation that landed after a later block.
    # The candidate must move to the right and be visibly near the top of the
    # next column; this avoids reordering ordinary paragraphs.
    for item in list(result):
        index = _identity_index(result, item)
        if index < 0:
            continue
        previous_bbox = _item_bbox(item)
        text = str(item.get("text") or "")
        if _noncontent_item(item) or previous_bbox is None or not _open_ended_text(text):
            continue
        candidates: list[tuple[int, tuple[float, float, float, float]]] = []
        for candidate_index in range(index + 1, len(result)):
            candidate = result[candidate_index]
            candidate_bbox = _item_bbox(candidate)
            if (
                _noncontent_item(candidate)
                or candidate_bbox is None
                or not _continuation_text(str(candidate.get("text") or ""))
            ):
                continue
            if candidate_bbox[0] <= previous_bbox[0] + page_width * 0.08:
                continue
            if candidate_bbox[1] > previous_bbox[1] + max(80.0, page_height * 0.22):
                continue
            candidates.append((candidate_index, candidate_bbox))
        if not candidates:
            continue
        candidate_index, candidate_bbox = min(
            candidates,
            key=lambda value: (value[1][0] - previous_bbox[0], value[1][1]),
        )
        if not _is_continuation_anchor(candidate_bbox, previous_bbox):
            continue
        # Prefer the closest preceding open-ended block.  A later-column
        # continuation such as ``be decreased ...`` may otherwise attach to
        # an earlier list item ending in ``and`` instead of the nearer item
        # ending in ``must``.
        if any(
            _open_ended_text(str(result[later].get("text") or ""))
            and _item_bbox(result[later]) is not None
            and _item_bbox(result[later])[0] > previous_bbox[0] - page_width * 0.03
            and _item_bbox(result[later])[0] < candidate_bbox[0]
            and _item_bbox(result[later])[1]
            <= candidate_bbox[1] + max(120.0, page_height * 0.35)
            for later in range(index + 1, candidate_index)
        ):
            continue
        candidate = result.pop(candidate_index)
        insert_at = _identity_index(result, item) + 1
        result.insert(insert_at, candidate)
        moves.append(
            {
                "kind": "wrapped_continuation",
                "from_index": candidate_index,
                "to_after": insert_at - 1,
                "text": str(candidate.get("text") or "")[:80],
            }
        )

    # 2) Correct inversions inside one visual column.  This is deliberately
    # limited to blocks with strong horizontal overlap; blocks in adjacent
    # columns are handled by the continuation and figure rules below.
    for _ in range(len(result)):
        inversion = None
        for left_index, left in enumerate(result):
            left_bbox = _item_bbox(left)
            if _noncontent_item(left) or left.get("block_type") == "figure" or left_bbox is None:
                continue
            for right_index in range(left_index + 1, len(result)):
                right = result[right_index]
                right_bbox = _item_bbox(right)
                if (
                    _noncontent_item(right)
                    or right.get("block_type") == "figure"
                    or right_bbox is None
                    or right_bbox[1] >= left_bbox[1] - 6
                    or _horizontal_overlap(left, right) < 0.55
                ):
                    continue
                inversion = (left_index, right_index, right)
                break
            if inversion:
                break
        if inversion is None:
            break
        left_index, right_index, right = inversion
        result.pop(right_index)
        result.insert(left_index, right)
        moves.append(
            {
                "kind": "same_column_y_order",
                "from_index": right_index,
                "to_index": left_index,
                "text": str(right.get("text") or "")[:80],
            }
        )

    # 3) A section heading must precede same-column body blocks that are
    # geometrically below it, even if the OCR payload returned those blocks
    # first.  Do not cross unrelated columns or move a heading without a
    # clear horizontal overlap.
    for item in list(result):
        text = str(item.get("text") or "")
        if _noncontent_item(item) or not english_heading(text):
            continue
        heading_bbox = _item_bbox(item)
        if heading_bbox is None:
            continue
        current_index = _identity_index(result, item)
        if current_index < 0:
            continue
        same_column = []
        for candidate_index, candidate in enumerate(result):
            candidate_bbox = _item_bbox(candidate)
            if (
                candidate is not item
                and candidate_bbox is not None
                and not _noncontent_item(candidate)
                and candidate.get("block_type") != "figure"
                and _horizontal_overlap(item, candidate) >= 0.35
            ):
                same_column.append((candidate_index, candidate, candidate_bbox))
        below = [
            (index, candidate)
            for index, candidate, bbox in same_column
            if bbox[1] >= heading_bbox[1]
        ]
        above = [
            (index, candidate)
            for index, candidate, bbox in same_column
            if bbox[3] <= heading_bbox[1] + 4
        ]
        if not below or not any(index < current_index for index, _ in below):
            continue
        # Remove the heading and any same-column body blocks that were emitted
        # before it.  Reinsert them after the geometrically preceding content,
        # preserving their original relative order.
        move_indices = {index for index, _ in below if index < current_index}
        move_indices.add(current_index)
        moved = [candidate for index, candidate in enumerate(result) if index in move_indices]
        remaining = [
            candidate for index, candidate in enumerate(result) if index not in move_indices
        ]
        anchors = [
            remaining.index(candidate)
            for _, candidate in above
            if candidate in remaining
        ]
        insert_at = max(anchors) + 1 if anchors else 0
        body_after_heading = [candidate for candidate in moved if candidate is not item]
        result = remaining[:insert_at] + [item, *body_after_heading] + remaining[insert_at:]
        moves.append(
            {
                "kind": "heading_before_body",
                "from_index": current_index,
                "to_index": insert_at,
                "text": text[:80],
            }
        )

    # 4) Wide figures are layout anchors.  If text occupying the figure's
    # horizontal span is returned after it despite sitting above its top edge,
    # move the figure after that text.  Small margin images are intentionally
    # untouched because their order is not reliable enough to infer.
    text_widths = [
        bbox[2] - bbox[0]
        for item, bbox in zip(result, [_item_bbox(value) for value in result], strict=False)
        if bbox is not None and item.get("block_type") != "figure"
    ]
    median_text_width = sorted(text_widths)[len(text_widths) // 2] if text_widths else 0.0
    for figure in list(result):
        if figure.get("block_type") != "figure":
            continue
        figure_bbox = _item_bbox(figure)
        if figure_bbox is None:
            continue
        figure_width = figure_bbox[2] - figure_bbox[0]
        if figure_width < max(page_width * 0.28, median_text_width * 1.45):
            continue
        figure_index = _identity_index(result, figure)
        if figure_index < 0:
            continue
        related = []
        for candidate_index, candidate in enumerate(result):
            if candidate is figure:
                continue
            candidate_bbox = _item_bbox(candidate)
            if (
                _noncontent_item(candidate)
                or candidate_bbox is None
                or _horizontal_overlap(figure, candidate) < 0.25
            ):
                continue
            related.append((candidate_index, candidate, candidate_bbox))
        # A figure can span two columns.  Blocks physically above it must not
        # be left after a block below it just because the image was returned
        # early by the OCR service.
        below_indices = [
            index
            for index, _, bbox in related
            if bbox[1] > figure_bbox[1] + 4
        ]
        above_after = [
            (index, candidate)
            for index, candidate, bbox in related
            if bbox[3] <= figure_bbox[1] + 4
            and index > (min(below_indices) if below_indices else figure_index)
        ]
        if above_after and below_indices:
            moving_indices = {index for index, _ in above_after}
            moving = [
                candidate for index, candidate in enumerate(result) if index in moving_indices
            ]
            remaining = [
                candidate for index, candidate in enumerate(result) if index not in moving_indices
            ]
            first_below = min(
                remaining.index(candidate)
                for index, candidate, _ in related
                if index in below_indices and candidate in remaining
            )
            result = remaining[:first_below] + moving + remaining[first_below:]
            moves.append(
                {
                    "kind": "figure_flow_preceding",
                    "from_indices": sorted(moving_indices),
                    "to_before": first_below,
                    "text": f"{len(moving)} block(s) moved before wide figure",
                }
            )
            figure_index = _identity_index(result, figure)
            related = [
                (candidate_index, candidate, _item_bbox(candidate))
                for candidate_index, candidate in enumerate(result)
                if candidate is not figure
                and not _noncontent_item(candidate)
                and _item_bbox(candidate) is not None
                and _horizontal_overlap(figure, candidate) >= 0.25
            ]
        above_indices = [
            index
            for index, _, bbox in related
            if bbox[3] <= figure_bbox[1] + 4
        ]
        below_indices = [
            index
            for index, _, bbox in related
            if bbox[1] > figure_bbox[1] + 4
        ]
        if not above_indices:
            continue
        if not (
            max(above_indices) > figure_index
            or (below_indices and min(below_indices) < figure_index)
        ):
            continue
        insert_at = max(above_indices)
        result.pop(figure_index)
        if insert_at > figure_index:
            insert_at -= 1
        result.insert(insert_at + 1, figure)
        moves.append(
            {
                "kind": "wide_figure_anchor",
                "from_index": figure_index,
                "to_after": insert_at,
                "text": str(figure.get("caption") or "Source image")[:80],
            }
        )

    return result, {
        "status": "repaired" if moves else "unchanged",
        "moved_count": len(moves),
        "moves": moves,
    }


def _image_heavy(path: Path) -> bool:
    """Measure displayed image area in PDF points, not raster pixels."""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages[:3]:
            area = float(page.width * page.height)
            if area and any(
                max(0, min(page.width, image["x1"]) - max(0, image["x0"]))
                * max(0, min(page.height, image["bottom"]) - max(0, image["top"]))
                / area
                >= 0.8
                for image in page.images
            ):
                return True
    return False


def is_scan_regulatory_pdf(path: Path, *, pdf_reader: Any | None = None) -> bool:
    """Require English regulatory content AND scanning evidence.

    Image-only/unknown documents retain the existing Hybrid PDF OCR route.
    """
    try:
        reader = pdf_reader if pdf_reader is not None else PdfReader(path)
        producer = str((reader.metadata or {}).get("/Producer", ""))
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:3])
        letters = len(re.findall(r"[A-Za-z]", text))
        cjk = len(re.findall(r"[\u3400-\u9fff]", text))
        if letters < 20 or cjk / max(letters + cjk, 1) >= 0.1:
            return False
        if not _REGULATORY_RE.search(text):
            return False
        return bool(SCAN_PRODUCER_RE.search(producer)) or _image_heavy(path)
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def english_heading(text: str) -> tuple[str, str] | None:
    title = unicodedata.normalize("NFKC", " ".join(text.split()))
    if not title or len(title) > 180:
        return None
    # OCR can drop the separator between an explicit section number and its
    # title. Only repair a leading marker followed by a title-case word;
    # inline references, paragraph suffixes and lower-case prose stay intact.
    title = re.sub(
        r"^(§\s*[A-Z]?\d+\.\d+)(?=[A-Z][a-z]{2,}\b)",
        r"\1 ",
        title,
    )
    for kind, pattern in _HEADING_PATTERNS:
        if re.fullmatch(pattern, title, re.I):
            return kind, title
    return None


def _annotation_reason(item: dict[str, Any], body_top: float | None) -> str | None:
    """Quarantine uncertain marginal material, never arbitrary misspelled prose.

    Content-list bounding boxes are normalized to 0..1000. A short heading-like
    block ABOVE a regulatory heading is a review candidate, not proof of handwriting.
    """
    if item.get("layout_recovery") == "body_misclassified_as_header":
        return None
    if item.get("source_type") in {"aside_text", "header", "footer", "page_number", "discarded"}:
        return "mineru_excluded"
    bbox = item.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    text = item.get("text", "")
    if english_heading(text):
        return None
    if (
        body_top is not None
        and bbox[3] <= min(body_top, 100)
        and len(text) < 160
        and (item.get("text_level") or item["block_type"] == "figure")
    ):
        return "suspected_top_annotation"
    if item["block_type"] == "figure" and (bbox[2] < 65 or bbox[0] > 950):
        return "suspected_margin_annotation"
    return None


class ScannedRegulatoryPdfParser(ParserPlugin):
    name = "scan-regulatory-pdf"
    version = "10"
    supported_extensions = frozenset({".pdf"})

    def __init__(self, mineru: MinerUClient) -> None:
        self.mineru = mineru

    def _page_blocks_from(
        self,
        result: RemoteParseResult,
        page_number: int,
        assets: list[AssetRecord],
    ) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        remote = [
            *result.page_blocks.get(page_number, []),
            *result.excluded_blocks.get(page_number, []),
        ]
        for ordinal, block in enumerate(remote):
            asset_id = None
            if block.image_content:
                asset_id = stable_id(
                    "scan-asset",
                    page_number,
                    ordinal,
                    hashlib.sha256(block.image_content).hexdigest(),
                )
                assets.append(
                    AssetRecord(
                        asset_id=asset_id,
                        page_number=page_number,
                        asset_type=block.block_type,
                        bbox=block.bbox or [0, 0, 0, 0],
                        filename=block.image_filename or f"{asset_id}.bin",
                        mime_type=block.image_mime_type or "application/octet-stream",
                        content=block.image_content,
                        caption=block.caption,
                    )
                )
            items.append(
                {
                    "block_type": block.block_type,
                    "text": block.text,
                    "bbox": block.bbox,
                    "source_type": block.source_type,
                    "text_level": block.text_level,
                    "raw_ocr_item": block.raw_item,
                    "table_html": block.table_html,
                    "caption": block.caption,
                    "latex": block.latex,
                    "asset_id": asset_id,
                    "asset_ids": [asset_id] if asset_id else [],
                    "source_ordinal": ordinal,
                }
            )
        return items

    def parse(
        self,
        path: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParsedDocument:
        if not self.mineru.enabled:
            raise RuntimeError("Scanned regulatory PDF requires enabled MinerU OCR.")
        reader = PdfReader(path)
        numbers = list(range(1, len(reader.pages) + 1))
        result = self.mineru.parse_pages(
            path,
            numbers,
            parse_method="ocr",
            progress_callback=progress_callback,
        )
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        assets: list[AssetRecord] = []
        pages: list[PageRecord] = []
        fallback_pages: list[int] = []
        for number in numbers:
            items = self._page_blocks_from(result, number, assets)
            source_items = list(items)
            for ordinal, item in enumerate(source_items):
                item.setdefault("source_ordinal", ordinal)
            items, reading_order = _repair_reading_order(items)
            raw = result.page_texts.get(number, "")
            if not result.page_blocks.get(number) and raw.strip():
                fallback_pages.append(number)
                items = [
                    {"block_type": "paragraph", "text": part}
                    for part in re.split(r"\n\s*\n", raw)
                    if part.strip()
                ] + items
            tops = [b["bbox"][1] for b in items if b.get("bbox") and english_heading(b["text"])]
            for item in items:
                item["exclusion_reason"] = _annotation_reason(item, min(tops) if tops else None)
            raw = "\n\n".join(b["text"] for b in source_items if b["text"]) or raw
            clean = "\n\n".join(b["text"] for b in items if not b["exclusion_reason"] and b["text"])
            body_assets = [
                b["asset_id"] for b in items if not b["exclusion_reason"] and b.get("asset_id")
            ]
            layout_features = {"ocr_blocks": items}
            if reading_order["status"] == "repaired":
                layout_features["reading_order_repair"] = reading_order
            pages.append(
                PageRecord(
                    page_number=number,
                    raw_text=raw,
                    cleaned_text=clean,
                    raw_char_count=len(compact_chars(raw)),
                    cleaned_char_count=len(compact_chars(clean)),
                    route="scan_regulatory_ocr",
                    rich_blocks=items,
                    indexable=bool(clean.strip() or body_assets),
                    page_type="figure" if body_assets and not clean.strip() else "text",
                    image_count=len(body_assets),
                    layout_features=layout_features,
                )
            )
        recover_header_pages(pages, source_hash)
        self._recover_tables(path, pages, source_hash=source_hash)
        return self._finish(path, source_hash, pages, assets, result.warnings, fallback_pages)

    def _finish(
        self,
        path: Path,
        source_hash: str,
        pages: list[PageRecord],
        assets: list[AssetRecord],
        warnings: list[str],
        fallback_pages: list[int],
    ) -> ParsedDocument:
        document_id = stable_id("document", source_hash)
        restore_table_assets(path, source_hash, pages, assets)
        restore_figure_names(source_hash, pages, assets)
        recover_formula_image_assets(path, pages, assets)
        blocks = self._build_blocks(document_id, pages)
        if source_hash == AMEND27805_SHA256:
            repair_formula_latex(blocks)
        self._render_pages(pages, blocks, assets)
        chunks = self._build_chunks(document_id, blocks)
        link_figures(document_id, chunks)
        qa = self._quality(pages, blocks, chunks, assets, warnings, fallback_pages)
        cleaned = "\n".join(page.cleaned_text for page in pages)
        return ParsedDocument(
            source_path=path,
            source_hash=source_hash,
            document_id=document_id,
            normalized_text_hash=hashlib.sha256(
                unicodedata.normalize("NFKC", cleaned).encode("utf-8")
            ).hexdigest(),
            pages=pages,
            route="scan_regulatory_ocr",
            repeated_margin_lines=[],
            removed_margin_line_count=0,
            blocks=blocks,
            chunks=chunks,
            qa=qa,
            assets=assets,
        )

    def reprocess_page(
        self,
        path: Path,
        previous: dict[str, Any],
        asset_root: Path,
        page_number: int,
        mode: str,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParsedDocument:
        """Replace one page, then rebuild hierarchy and chunks for the whole document."""
        if mode not in {"clean", "ocr"}:
            raise ValueError("Unknown page reprocessing mode")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hash != previous.get("source", {}).get("sha256"):
            raise ValueError("Source hash differs from cached document")
        allowed = {f.name for f in fields(PageRecord)}
        baseline_path = asset_root / "baseline.json"
        baseline = (
            json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else {}
        )
        raw_pages = {p["page_number"]: p.get("raw_text", "") for p in baseline.get("pages", [])}
        pages = []
        for p in previous["pages"]:
            values = {k: copy.deepcopy(v) for k, v in p.items() if k in allowed}
            values["raw_text"] = raw_pages.get(p["page_number"], p.get("cleaned_text", ""))
            values["rich_blocks"] = values.get("layout_features", {}).get("ocr_blocks", [])
            if (
                not values["rich_blocks"]
                and p["page_number"] != page_number
                and p.get("rich_block_count")
            ):
                raise ValueError("Cached structured page blocks missing; full processing required")
            pages.append(PageRecord(**values))
        if [p.page_number for p in pages] != list(range(1, len(PdfReader(path).pages) + 1)):
            raise ValueError("Cached pages are incomplete or out of order")
        target = next((p for p in pages if p.page_number == page_number), None)
        if target is None:
            raise ValueError("Page does not exist")
        effective = mode
        reading_order: dict[str, Any] = {"status": "unchanged", "moved_count": 0}
        source_items: list[dict[str, Any]] = []
        if mode == "clean" and (
            not target.rich_blocks
            or any(not isinstance(i.get("raw_ocr_item"), dict) for i in target.rich_blocks)
        ):
            effective = "ocr"
        assets = []
        asset_fields = {f.name for f in fields(AssetRecord)} - {"content"}
        for record in previous.get("assets", []):
            if effective == "ocr" and record["page_number"] == page_number:
                continue
            file = (asset_root / record["relative_path"]).resolve()
            if not file.is_relative_to(asset_root.resolve()):
                raise ValueError("Cached asset is outside output directory")
            assets.append(
                AssetRecord(
                    **{k: v for k, v in record.items() if k in asset_fields},
                    content=file.read_bytes(),
                )
            )
        warnings = list(previous.get("qa", {}).get("remote_warnings", []))
        fallback = list(previous.get("qa", {}).get("text_only_fallback_pages", []))
        if effective == "ocr":
            result = self.mineru.parse_pages(
                path,
                [page_number],
                parse_method="ocr",
                progress_callback=progress_callback,
            )
            items = self._page_blocks_from(result, page_number, assets)
            if not result.page_blocks.get(page_number):
                raise ValueError(
                    "Single-page OCR returned no structured content; old output retained"
                )
            warnings.extend(result.warnings)
            fallback = [n for n in fallback if n != page_number]
        else:
            items = target.rich_blocks
            source_items = list(items)
            for ordinal, item in enumerate(source_items):
                item.setdefault("source_ordinal", ordinal)
            for item in items:
                raw = item["raw_ocr_item"]
                if item.get("table_review_status") == "recovered":
                    # Preserve independently validated table geometry and its evidence.
                    continue
                for key in ("table_review_status", "table_structure_issue", "table_recovery_error"):
                    item.pop(key, None)
                item["text"] = self.mineru._item_text(raw)
                if raw.get("type") in {
                    "header",
                    "footer",
                    "page_number",
                    "discarded",
                    "aside_text",
                }:
                    item["text"] = str(raw.get("text", ""))
                if raw.get("type") == "table":
                    item["table_html"] = clean_inline_latex(str(raw.get("table_body", ""))) or None
            if progress_callback:
                progress_callback(1, 1)
        if not source_items:
            source_items = list(items)
            for ordinal, item in enumerate(source_items):
                item.setdefault("source_ordinal", ordinal)
        items, reading_order = _repair_reading_order(items)
        tops = [i["bbox"][1] for i in items if i.get("bbox") and english_heading(i["text"])]
        for item in items:
            item["exclusion_reason"] = _annotation_reason(item, min(tops) if tops else None)
        target.rich_blocks = items
        if effective == "ocr" or not target.raw_text:
            target.raw_text = "\n\n".join(i["text"] for i in source_items if i["text"])
        body = [i for i in items if not i.get("exclusion_reason")]
        text = "\n\n".join(i["text"] for i in body if i["text"])
        target.raw_char_count = len(compact_chars(target.raw_text))
        target.cleaned_char_count = len(compact_chars(text))
        target.image_count = sum(bool(i.get("asset_ids")) for i in body)
        target.indexable = bool(text.strip() or target.image_count)
        target.page_type = "figure" if target.image_count and not text.strip() else "text"
        target.layout_features = {"ocr_blocks": items}
        if reading_order["status"] == "repaired":
            target.layout_features["reading_order_repair"] = reading_order
        recover_header_pages([target], source_hash)
        self._recover_tables(
            path, [target], allow_ocr=effective == "ocr", source_hash=source_hash
        )
        document = self._finish(path, source_hash, pages, assets, warnings, fallback)
        document.qa["page_reprocess"] = {
            "page_number": page_number,
            "requested_mode": mode,
            "effective_mode": effective,
            "previous_parser": previous.get("parser"),
            "chunks_rebuilt": True,
        }
        return document

    def _recover_tables(
        self, path: Path, pages: list[PageRecord], *, allow_ocr: bool = True,
        source_hash: str | None = None,
    ) -> None:
        attempts = 0
        for page in pages:
            for item in page.rich_blocks:
                if item.get("exclusion_reason"):
                    continue
                markup = item.get("table_html")
                issue = structure_issue(markup) if markup else None
                if not issue:
                    continue
                item["raw_table_html"] = markup
                item["raw_table_text"] = item["text"]
                item["table_structure_issue"] = issue
                item["table_review_status"] = "needs_review"
                repaired = verified_table(source_hash, page.page_number, markup)
                method = "source_visual_review" if repaired else None
                # Only the supported fragmented-header shape is auto-recovered.
                # Other suspect layouts remain quarantined, never guessed.
                if (
                    allow_ocr
                    and not repaired
                    and issue == "fragmented_spanning_header"
                    and item.get("bbox")
                    and attempts < 4
                ):
                    attempts += 1
                    try:
                        crop = crop_table_pdf(path, page.page_number, item["bbox"])
                        lines = self.mineru.ocr_table_lines(crop)
                        item["table_recovery_lines"] = lines
                        repaired = recover_two_columns(markup, lines)
                    except (
                        OSError,
                        ValueError,
                        RuntimeError,
                        httpx.HTTPError,
                        zipfile.BadZipFile,
                    ) as exc:
                        item["table_recovery_error"] = f"{type(exc).__name__}: {exc}"
                if repaired:
                    item["table_html"] = repaired
                    item["text"] = item["text"].replace(markup, repaired, 1)
                    item["table_review_status"] = "recovered"
                    item["table_recovery_method"] = (
                        method or "crop_ocr_geometry_and_token_agreement"
                    )
                else:
                    item["table_html"] = None
                    item["text"] = (
                        f"Table on page {page.page_number} requires structural review; "
                        "unverified cell/value pairs withheld. See source image and raw_table_html."
                    )

    @staticmethod
    def _render_pages(
        pages: list[PageRecord],
        blocks: list[BlockRecord],
        assets: list[AssetRecord],
    ) -> None:
        """Preserve headings, formulas, HTML and image links in the review Markdown."""
        by_id = {b.block_id: b for b in blocks}
        asset_map = {a.asset_id: a for a in assets}
        for page in pages:
            preview: list[str] = []
            for item in page.rich_blocks:
                if item.get("exclusion_reason"):
                    continue
                block = by_id.get(item.get("block_id"))
                if block is None:
                    continue
                if block.block_type == "heading":
                    preview.append("#" * min(len(block.section_path), 6) + " " + block.text)
                elif block.block_type == "formula" and block.latex:
                    preview.append("$$\n" + block.latex + "\n$$")
                elif item.get("text"):
                    preview.append(item["text"])
                for asset_id in block.asset_ids:
                    asset = asset_map[asset_id]
                    extension = Path(asset.filename).suffix.casefold() or ".bin"
                    preview.append(
                        f"![Source image, page {page.page_number}](assets/{asset_id}{extension})"
                    )
            page.cleaned_text = "\n\n".join(preview)

    def _build_blocks(self, document_id: str, pages: list[PageRecord]) -> list[BlockRecord]:
        blocks: list[BlockRecord] = []
        stack: list[tuple[int, str]] = []
        for page in pages:
            page.rich_blocks = expand_regulatory_blocks(page.rich_blocks)
            for ordinal, item in enumerate(page.rich_blocks):
                text = str(item.get("text") or "").strip()
                block_type = str(item.get("block_type") or "paragraph")
                heading = english_heading(text) if block_type in {"paragraph", "heading"} else None
                if heading is None and block_type in {"paragraph", "heading"}:
                    # Recover a lost appendix separator only with matching section
                    # numbering immediately afterwards; never split arbitrary prose.
                    joined = re.fullmatch(r"APPENDIX\s+([A-Z])([A-Z].{3,150})", text)
                    following = next((
                        str(candidate.get("text", "")).strip()
                        for candidate in page.rich_blocks[ordinal + 1:]
                        if not candidate.get("exclusion_reason") and candidate.get("text")
                    ), "")
                    if joined and text.isupper() and re.match(
                        rf"{joined[1]}\d+\.\d+\b", following
                    ):
                        heading = ("appendix", f"APPENDIX {joined[1]}—{joined[2]}")
                if item.get("exclusion_reason"):
                    block_type, heading = "annotation", None
                elif heading is None and item.get("text_level") and text.isupper():
                    if len(text) <= 100 and block_type == "paragraph":
                        heading = ("group", text)
                if heading:
                    kind, title = heading
                    rank = _RANK[kind]
                    stack = [(r, t) for r, t in stack if r < rank]
                    stack.append((rank, title))
                    text, block_type = title, "heading"
                table_html = item.get("table_html")
                rows = table_rows_from_html(table_html) if table_html else []
                if rows:
                    # The remote text also carries captions/footnotes outside
                    # table_html; do not lose them during semantic rendering.
                    remainder = text.replace(table_html, "", 1).strip()
                    text = table_to_semantic_text(rows, title=item.get("caption"))
                    if remainder and remainder not in text:
                        text += "\n" + remainder
                text = text or item.get("latex") or item.get("caption") or ""
                if not text and item.get("asset_ids") and block_type != "annotation":
                    text = (
                        f"Source image on page {page.page_number}; no OCR transcription available."
                    )
                if not text and not item.get("asset_ids"):
                    continue
                block_id = stable_id(document_id, page.page_number, ordinal, text)
                item["block_id"] = block_id
                article = next((
                    reference for _, title in reversed(stack)
                    if (reference := match_article_heading(title)) is not None
                ), None)
                blocks.append(
                    BlockRecord(
                        block_id=block_id,
                        block_type=block_type,
                        text=text,
                        page_number=page.page_number,
                        section_path=[t for _, t in stack],
                        article_id_raw=article.raw if article else None,
                        article_id_normalized=article.normalized_id if article else None,
                        article_parent_id=article.parent_id if article else None,
                        article_aliases=list(article.aliases) if article else [],
                        bbox=item.get("bbox"),
                        table_html=table_html,
                        table_rows=rows,
                        table_header_rows=1 if rows else 0,
                        table_id=block_id if block_type == "table" else None,
                        table_title=item.get("caption"),
                        latex=item.get("latex"),
                        asset_id=item.get("asset_id"),
                        asset_ids=item.get("asset_ids", []),
                    )
                )
        return blocks

    def _build_chunks(self, document_id: str, blocks: list[BlockRecord]) -> list[ChunkRecord]:
        chunks: list[ChunkRecord] = []
        pending: list[BlockRecord] = []

        def flush() -> None:
            if not pending:
                return
            first = pending[0]
            text = "\n\n".join(b.text for b in pending if b.text)
            if not text:
                pending.clear()
                return
            tables = [b for b in pending if b.block_type == "table"]
            chunks.append(
                ChunkRecord(
                    chunk_id=stable_id(document_id, "chunk", len(chunks), text),
                    parent_chunk_id=stable_id(document_id, "section", *first.section_path),
                    title=(
                        first.table_title or f"Source image on page {first.page_number}"
                        if first.block_type == "figure"
                        else first.section_path[-1] if first.section_path else "Document preamble"
                    ),
                    text=text,
                    page_start=first.page_number,
                    page_end=pending[-1].page_number,
                    section_path=list(first.section_path),
                    article_id_raw=first.article_id_raw,
                    article_id_normalized=first.article_id_normalized,
                    article_parent_id=first.article_parent_id,
                    article_aliases=list(first.article_aliases),
                    block_ids=[b.block_id for b in pending],
                    asset_ids=list(dict.fromkeys(a for b in pending for a in b.asset_ids)),
                    formula_latex=[b.latex for b in pending if b.latex],
                    table_html=[b.table_html for b in tables if b.table_html],
                    table_ids=[b.table_id for b in tables if b.table_id],
                    table_rows=[row for b in tables for row in b.table_rows],
                    content_type=(
                        "table" if tables else "figure" if first.block_type == "figure" else "text"
                    ),
                )
            )
            pending.clear()

        pieces: list[BlockRecord] = []
        for block in blocks:
            if block.block_type == "table" and block.table_rows and len(block.text) > 1600:
                parts = list(split_structured_table_block(block, 1600)) or [block]
                semantic = table_to_semantic_text(block.table_rows, title=block.table_title)
                remainder = block.text[len(semantic) :] if block.text.startswith(semantic) else ""
                if remainder:
                    parts[-1] = replace(parts[-1], text=parts[-1].text + remainder)
                pieces.extend(parts)
            elif (
                block.block_type not in {"table", "annotation", "figure"}
                and len(block.text) > 1600
            ):
                start = 0
                while start < len(block.text):
                    end = min(start + 1600, len(block.text))
                    if end < len(block.text):
                        space = block.text.rfind(" ", start + 800, end)
                        if space > start:
                            end = space
                    pieces.append(replace(block, text=block.text[start:end].strip()))
                    start = end
            else:
                pieces.append(block)
        for block in pieces:
            if block.block_type == "annotation":
                continue
            if pending and (
                block.section_path != pending[0].section_path
                or block.block_type in {"heading", "table", "figure"}
                or sum(len(b.text) + 2 for b in pending) + len(block.text) > 1600
            ):
                flush()
            pending.append(block)
            if block.block_type in {"table", "figure"} or sum(len(b.text) for b in pending) >= 1600:
                flush()
        flush()
        return chunks

    def _quality(
        self,
        pages: list[PageRecord],
        blocks: list[BlockRecord],
        chunks: list[ChunkRecord],
        assets: list[AssetRecord],
        warnings: list[str],
        fallback_pages: list[int],
    ) -> dict[str, Any]:
        gates: list[dict[str, str]] = []

        def gate(name: str, status: str, message: str) -> None:
            gates.append({"gate": name, "status": status, "message": message})

        empty = [p.page_number for p in pages if not p.indexable]
        no_text = [p.page_number for p in pages if not p.cleaned_char_count]
        image_only = [p.page_number for p in pages if p.page_type == "figure"]
        if not chunks:
            gate("nonempty_chunks", "fail", "OCR did not produce searchable content.")
        if empty:
            gate("page_text_coverage", "fail", f"Pages without searchable text: {empty}")
        if image_only:
            gate(
                "image_only_pages",
                "warn",
                f"Image assets preserved on pages {image_only}; chart text requires review.",
            )
        for page in pages:
            restored = sum(bool(i.get("layout_recovery")) for i in page.rich_blocks)
            if restored:
                gate("header_classification_recovery", "warn",
                     f"Page {page.page_number}: restored {restored} body blocks "
                     "mislabeled as headers; "
                     "compare source layout. Original OCR labels retained.")
        if warnings:
            gate("remote_warnings", "warn", "; ".join(warnings))
        if fallback_pages:
            gate("structured_content", "warn", f"Text-only fallback pages: {fallback_pages}")
        for page in pages:
            reading_order = page.layout_features.get("reading_order_repair", {})
            if reading_order.get("status") == "repaired":
                gate(
                    "reading_order_repair",
                    "warn",
                    f"Page {page.page_number}: repaired {reading_order.get('moved_count', 0)} "
                    "high-confidence OCR flow inversion(s); compare source layout.",
                )
            for item in page.rich_blocks:
                status = item.get("table_review_status")
                if status:
                    gate(
                        "table_structure_recovery",
                        "pass" if status == "recovered" else "fail",
                        f"Page {page.page_number}: {status}; "
                        f"{item['table_structure_issue']}; original evidence retained.",
                    )
                if item.get("table_structure_issue") == "multiple_labeled_rows":
                    gate(
                        "table_row_alignment",
                        "warn",
                        f"Page {page.page_number}: " + (
                            "source-specific visual correction applied; original evidence retained."
                            if item.get("table_recovery_method")
                            == "source_visual_review_2026-08-31"
                            else "merged labeled rows require source review."
                        ),
                    )
        annotations = [
            dict(item, page_number=p.page_number)
            for p in pages
            for item in p.rich_blocks
            if item.get("exclusion_reason")
        ]
        uncertain = [a for a in annotations if a["exclusion_reason"] != "mineru_excluded"]
        if uncertain:
            gate(
                "annotation_review",
                "warn",
                f"{len(uncertain)} suspected marginal annotations retained outside search; review.",
            )
        represented = {bid for c in chunks for bid in c.block_ids}
        missing = [
            b.block_id
            for b in blocks
            if b.text and b.block_type != "annotation" and b.block_id not in represented
        ]
        if missing:
            gate("block_chunk_coverage", "fail", f"Unindexed content blocks: {missing}")
        known = {a.asset_id for a in assets}
        if any(a not in known for b in blocks for a in b.asset_ids):
            gate("asset_references", "fail", "A block references a missing asset.")
        for block in blocks:
            if block.block_type == "table":
                if not block.table_html:
                    gate(
                        "table_structure",
                        "warn",
                        f"Page {block.page_number}: table has text but no HTML.",
                    )
                elif len(block.table_rows) <= 2 and re.search(
                    r"\([a-z]\).+\([a-z]\)", block.text, re.I | re.S
                ):
                    gate(
                        "table_row_alignment",
                        "warn",
                        f"Page {block.page_number}: multiple labeled rows may be merged; "
                        "compare numeric cells against the source, do not auto-split.",
                    )
        from app.ingestion.parsers.table_review import table_review_gate

        manual_table_gate = table_review_gate(pages, {})
        if manual_table_gate:
            gates.append(manual_table_gate)
        if not gates:
            gate("scan_structure", "pass", "Page, block and asset coverage checks passed.")
        return {
            "parser_name": self.name,
            "parser_version": self.version,
            "page_count": len(pages),
            "text_page_count": len(pages) - len(no_text),
            "text_coverage": (len(pages) - len(no_text)) / max(len(pages), 1),
            "content_coverage": (len(pages) - len(empty)) / max(len(pages), 1),
            "image_only_pages": image_only,
            "chunk_count": len(chunks),
            "heading_count": sum(b.block_type == "heading" for b in blocks),
            "asset_count": len(assets),
            "quality_gates": gates,
            "table_review_status": "review_required" if manual_table_gate else "not_required",
            "table_review_pages": manual_table_gate["pages"] if manual_table_gate else [],
            "annotations": annotations,
            "missing_text_pages": no_text,
            "missing_content_pages": empty,
            "remote_warnings": warnings,
            "text_only_fallback_pages": fallback_pages,
        }
