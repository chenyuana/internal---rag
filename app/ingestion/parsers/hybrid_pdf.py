from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import statistics
import tempfile
import unicodedata
from collections.abc import Callable
from dataclasses import fields
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream

from app.ingestion.parsers.figure_vision import FigureVisionClient
from app.ingestion.parsers.native_pdf import NativePdfParser
from app.ingestion.parsers.remote import (
    DoclingClient,
    MinerUClient,
    RemoteParseResult,
    clean_inline_latex,
    formula_latex,
)
from app.ingestion.parsers.section_ocr import finish_section_ocr, prepare_section_ocr
from app.ingestion.parsers.table_recovery import crop_table_pdf
from app.ingestion.parsers.table_review import table_review_gate
from app.ingestion.parsers.vector_pdf import VectorPdfParser
from app.ingestion.parsers.verified_tables import verified_hybrid_rows
from app.ingestion.pipeline import (
    AssetRecord,
    PageRecord,
    ParsedDocument,
    analyze_complex_table_fidelity,
    build_chunks,
    clean_page,
    compact_chars,
    detect_repeated_margin_lines,
    filter_federal_register_rich_blocks,
    filter_repeated_margin_rich_blocks,
    find_unpublished_clause_blocks,
    is_glyph_name_garbage,
    is_probable_toc_page,
    merge_cross_page_tables,
    normalize_table_html,
    repair_invalid_unicode,
    split_blocks,
    split_merged_or_rows,
    table_rows_from_html,
    table_to_html,
    table_to_semantic_text,
)

# A repeated raster mark that is no larger than half an inch on the page is
# overwhelmingly likely to be a watermark, logo, or distribution stamp.  It
# is removed only from the temporary PDF uploaded to OCR, never from the
# source document or the vector parsing path.
REPEATED_IMAGE_WATERMARK_MAX_EDGE_PT = 36.0
REPEATED_IMAGE_WATERMARK_MIN_PAGES = 3
REPEATED_IMAGE_WATERMARK_MIN_PAGE_FRACTION = 0.5

# When whole-page OCR blurs a tiny raster table, re-OCR the cropped region at
# higher resolution and adopt it only if its word/number multiset matches the
# whole-page result — i.e. the crop re-arranged the same content without
# inventing, dropping, or duplicating any token. Larger tables that the crop
# fragments fail this gate and keep the whole-page result.
#
# The comparison is deliberately insensitive to separator punctuation: "1.0-1.5"
# and "1.0–1.5" are the same range, and "3 2" vs "3" / "2" is the same pair of
# digits. Only the multiset of words and numbers matters, so a crop that fuses
# or splits cells still passes while one that duplicates/loses a value fails.
_TABLE_TOKEN_RE = re.compile(r"[a-z]+|\d+(?:\.\d+)?%?")


def _table_token_multiset(rows: list[list[str]]) -> dict[str, int]:
    text = " ".join(cell for row in rows for cell in row)
    text = unicodedata.normalize("NFKC", text).casefold()
    # Collapse every dash-like separator (hyphen, en/em dash, minus) to a space
    # so range endpoints are compared as distinct numbers, not as signed
    # negative tokens. Also drop underscores so snake_case words compare as
    # separate words.
    for dash in ("–", "—", "−", "-", "_"):
        text = text.replace(dash, " ")
    counts: dict[str, int] = {}
    for token in _TABLE_TOKEN_RE.findall(text):
        compact = token.strip()
        if compact:
            counts[compact] = counts.get(compact, 0) + 1
    return counts


def _multiply_pdf_matrices(
    left: tuple[float, float, float, float, float, float],
    right: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    """Concatenate two PDF affine transformation matrices."""

    a, b, c, d, e, f = left
    A, B, C, D, E, F = right
    return (
        a * A + c * B,
        b * A + d * B,
        a * C + c * D,
        b * C + d * D,
        a * E + c * F + e,
        b * E + d * F + f,
    )


def _repeated_small_image_draw_names(
    reader: PdfReader,
) -> dict[int, set[str]]:
    """Find small image XObjects reused as document-wide raster watermarks.

    The hash is taken from the decoded image payload, while the displayed size
    comes from the page content stream's graphics matrix.  Requiring both a
    small physical size and reuse on most pages keeps ordinary figures and
    one-off icons out of this rule.
    """

    draws: list[tuple[int, str, str]] = []
    page_count = len(reader.pages)
    for page_number, page in enumerate(reader.pages, start=1):
        image_hashes: dict[str, str] = {}
        try:
            for image in page.images:
                # pypdf appends an inferred extension (e.g. ``Xi121.png``)
                # to the XObject resource name used by the content stream.
                name = "/" + image.name.rsplit(".", 1)[0]
                image_hashes[name] = hashlib.sha256(image.data).hexdigest()
            stream = ContentStream(page.get_contents(), reader)
        except Exception:
            continue

        current = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        stack: list[tuple[float, float, float, float, float, float]] = []
        for operands, operator in stream.operations:
            if operator == b"q":
                stack.append(current)
                continue
            if operator == b"Q":
                if stack:
                    current = stack.pop()
                continue
            if operator == b"cm" and len(operands) == 6:
                try:
                    matrix = tuple(float(value) for value in operands)
                except (TypeError, ValueError):
                    continue
                current = _multiply_pdf_matrices(current, matrix)  # type: ignore[arg-type]
                continue
            if operator != b"Do" or not operands:
                continue
            name = str(operands[0])
            image_hash = image_hashes.get(name)
            if image_hash is None:
                continue
            x_edge = math.hypot(current[0], current[1])
            y_edge = math.hypot(current[2], current[3])
            if max(x_edge, y_edge) <= REPEATED_IMAGE_WATERMARK_MAX_EDGE_PT:
                draws.append((page_number, name, image_hash))

    required_pages = max(
        REPEATED_IMAGE_WATERMARK_MIN_PAGES,
        math.ceil(page_count * REPEATED_IMAGE_WATERMARK_MIN_PAGE_FRACTION),
    )
    pages_by_hash: dict[str, set[int]] = {}
    for page_number, _name, image_hash in draws:
        pages_by_hash.setdefault(image_hash, set()).add(page_number)
    watermark_hashes = {
        image_hash
        for image_hash, pages in pages_by_hash.items()
        if len(pages) >= required_pages
    }
    names_by_page: dict[int, set[str]] = {}
    for page_number, name, image_hash in draws:
        if image_hash in watermark_hashes:
            names_by_page.setdefault(page_number, set()).add(name)
    return names_by_page


def _without_image_draws(
    operations: list[tuple[list[Any], bytes]],
    names: set[str],
) -> list[tuple[list[Any], bytes]]:
    """Drop only ``Do`` image-painting instructions for selected XObjects."""

    return [
        (operands, operator)
        for operands, operator in operations
        if not (operator == b"Do" and operands and str(operands[0]) in names)
    ]


def build_ocr_sanitized_pdf(source_path: Path, output_path: Path) -> dict[int, int]:
    """Write an OCR-only copy without high-frequency, tiny raster marks.

    Returns the number of suppressed paint operations per page.  A caller that
    receives an empty mapping can keep using the original PDF.
    """

    reader = PdfReader(source_path)
    names_by_page = _repeated_small_image_draw_names(reader)
    if not names_by_page:
        return {}

    writer = PdfWriter()
    removed_by_page: dict[int, int] = {}
    for page_number, original_page in enumerate(reader.pages, start=1):
        writer.add_page(original_page)
        names = names_by_page.get(page_number)
        if not names:
            continue
        page = writer.pages[-1]
        stream = ContentStream(page.get_contents(), writer)
        filtered = _without_image_draws(stream.operations, names)
        removed = len(stream.operations) - len(filtered)
        if removed:
            stream.operations = filtered
            page.replace_contents(stream)
            removed_by_page[page_number] = removed

    if not removed_by_page:
        return {}
    with output_path.open("wb") as stream:
        writer.write(stream)
    return removed_by_page


class HybridPdfParser:
    name = "hybrid-pdf"
    version = "5"
    supported_extensions = frozenset({".pdf"})

    def __init__(
        self,
        native: NativePdfParser,
        mineru: MinerUClient,
        docling: DoclingClient,
        figure_vlm: FigureVisionClient,
    ) -> None:
        self._native = native
        self._vector = VectorPdfParser()
        self.mineru = mineru
        self.docling = docling
        self.figure_vlm = figure_vlm

    def _merge_vector_pages(
        self,
        document: ParsedDocument,
        *,
        table_targets: set[int],
        figure_targets: set[int],
        formula_targets: set[int],
    ) -> tuple[set[int], set[int], set[int], set[int], set[int], list[str]]:
        results = self._vector.parse_pages(
            document.source_path,
            table_pages=table_targets,
            figure_pages=figure_targets,
            formula_pages=formula_targets,
        )
        merged_tables: set[int] = set()
        preserved_figures: set[int] = set()
        scanned_text_pages: set[int] = set()
        figure_review_pages: set[int] = set()
        blank_pages: set[int] = set()
        warnings: list[str] = []
        for page_number, result in results.items():
            page = document.pages[page_number - 1]
            page.page_type = result.page_type
            page.image_count = result.image_count
            page.image_coverage = result.image_coverage
            page.rich_blocks.extend(
                {
                    "block_type": block.block_type,
                    "text": block.text,
                    "bbox": block.bbox,
                    "confidence": block.confidence,
                    "table_html": block.table_html,
                    "table_title": block.table_title,
                    "table_rows": block.table_rows,
                    "table_merged_cell_count": block.table_merged_cell_count,
                    "table_header_only": block.table_header_only,
                    "asset_id": block.asset_id,
                    "asset_ids": list(block.asset_ids),
                }
                for block in result.blocks
            )
            document.assets.extend(
                AssetRecord(
                    asset_id=asset.asset_id,
                    page_number=asset.page_number,
                    asset_type=asset.asset_type,
                    bbox=asset.bbox,
                    filename=asset.filename,
                    mime_type=asset.mime_type,
                    content=asset.content,
                    caption=asset.caption,
                    description=asset.description,
                )
                for asset in result.assets
            )
            if result.table_count:
                page.route = "vector_layout"
                if "矢量网格检测到结构化表格" not in page.table_hints:
                    page.table_hints.append("矢量网格检测到结构化表格")
                merged_tables.add(page_number)
            if any(
                asset.asset_type in {"figure_page", "formula_page", "scan_page"}
                for asset in result.assets
            ):
                preserved_figures.add(page_number)
            if result.page_type == "scanned_text":
                scanned_text_pages.add(page_number)
            elif result.page_type == "blank":
                page.route = "blank_excluded"
                page.indexable = False
                blank_pages.add(page_number)
            elif result.page_type == "figure":
                page.route = "figure_review"
                figure_review_pages.add(page_number)
            elif result.page_type == "table_figure":
                # A table_figure page carries both a table and images. Only
                # route it through VLM review when it actually produced a
                # cropped figure asset (a real illustration worth describing).
                # Pages whose "images" are tiny decorative elements (e.g. a
                # logo or a caption rule at ~0.5% coverage) only produce a
                # whole-page figure_page asset and a structured table; running
                # the VLM on those just blocks the queue for no retrieval
                # value.
                has_figure_asset = any(
                    asset.asset_type == "figure"
                    for asset in result.assets
                )
                if has_figure_asset:
                    figure_review_pages.add(page_number)
            elif result.page_type == "formula":
                page.route = "formula_review"
            warnings.extend(
                f"page {page_number}: {warning}" for warning in result.warnings
            )
        return (
            merged_tables,
            preserved_figures,
            scanned_text_pages,
            figure_review_pages,
            blank_pages,
            warnings,
        )

    @staticmethod
    def _separate_table_caption(
        caption: str,
        rows: list[list[str]],
    ) -> tuple[str, str]:
        """Detach a following prose heading that OCR attached to a table."""

        value = str(caption or "").strip()
        if not value or not rows:
            return value, ""
        if re.match(r"^(?:table|tab\.|表)\s*[A-Za-z0-9]", value, re.IGNORECASE):
            return value, ""
        first_row = " ".join(str(cell) for cell in rows[0] if str(cell).strip())
        words = re.findall(r"[A-Za-z]{3,}", value.casefold())
        title_words = set(re.findall(r"[A-Za-z]{3,}", first_row.casefold()))
        overlap = sum(word in title_words for word in words)
        title_case_heading = (
            2 <= len(words) <= 14
            and value[:1].isupper()
            and not value.endswith((".", ";"))
        )
        if title_case_heading and overlap < 2:
            return "", value
        return value, ""

    def _describe_figures(
        self,
        document: ParsedDocument,
        page_numbers: set[int],
    ) -> tuple[set[int], list[str]]:
        described_pages: set[int] = set()
        warnings: list[str] = []
        if not page_numbers or not self.figure_vlm.enabled:
            return described_pages, warnings
        for page_number in sorted(page_numbers):
            page_assets = [
                asset
                for asset in document.assets
                if asset.page_number == page_number and asset.asset_type == "figure"
            ]
            if not page_assets:
                page_assets = [
                    asset
                    for asset in document.assets
                    if asset.page_number == page_number
                    and asset.asset_type == "figure_page"
                ]
            page_described = False
            for asset in page_assets:
                try:
                    description = self.figure_vlm.describe(
                        asset.content,
                        page_number=page_number,
                        caption=asset.caption,
                    )
                except Exception as exc:
                    warnings.append(
                        f"Figure VLM failed for page {page_number}, "
                        f"asset {asset.asset_id}: {type(exc).__name__}: {exc}"
                    )
                    continue
                asset.description = description
                page_described = True
                for block in document.pages[page_number - 1].rich_blocks:
                    if block.get("asset_id") == asset.asset_id:
                        block["text"] = "\n".join(
                            value
                            for value in (asset.caption, description)
                            if value.strip()
                        )
            if page_described:
                described_pages.add(page_number)
        return described_pages, warnings

    @staticmethod
    def _figure_dominant_pages(
        document: Any,
        figure_pages: set[int],
    ) -> set[int]:
        """Return text-layer-corrupted pages that are *real* figures.

        A text-layer-corrupted page (PostScript glyph names / ToUnicode
        conflict) is usually routed to OCR. But when the page is essentially a
        raster chart — preflight flagged it as a figure AND its text is just a
        caption (CCAR-23-R3 p164/167/168/172: "图A2 在速度V_C时确定n_4系数的
        曲线图") — OCR returns an empty content list and the chart is dropped.
        Such pages should be emitted as figure assets instead.

        Heuristic: the page is a figure AND has little recoverable text
        (<= 120 chars). Text-dominant corrupted pages (formulas, body text)
        keep going to OCR.
        """
        text_layer_figures: set[int] = set()
        for page in document.pages:
            if page.page_number not in figure_pages:
                continue
            if page.route != "text_layer_review":
                continue
            # A page whose text layer is flagged as low-quality OCR is a scan
            # of *text* (a cover, a 分送 page), not a pure chart. OCR can
            # recover it, so it must NOT be treated as a figure-dominant page —
            # otherwise the short-text heuristic below would pull it back out of
            # the OCR route. Only clean short text (a real chart caption) means
            # "image-only page that OCR would return empty for".
            if page.text_layer_corruption.get("low_quality_ocr"):
                continue
            compact = "".join(
                (page.cleaned_text or page.raw_text or "").split()
            )
            if len(compact) <= 120:
                text_layer_figures.add(page.page_number)
        return text_layer_figures

    @staticmethod
    def _merge_pages(
        document: ParsedDocument,
        result: RemoteParseResult,
        targets: set[int],
        *,
        route: str,
    ) -> set[int]:
        merged: set[int] = set()
        for page in document.pages:
            if page.page_number not in targets:
                continue
            remote_blocks = result.page_blocks.get(page.page_number, [])
            if remote_blocks:
                text_parts: list[str] = []
                for remote_block in remote_blocks:
                    if remote_block.block_type == "table" and remote_block.table_html:
                        table_html = normalize_table_html(remote_block.table_html)
                        rows = table_rows_from_html(table_html)
                        if rows:
                            rows = verified_hybrid_rows(
                                document.source_hash,
                                page.page_number,
                                rows,
                            )
                            # General (document-agnostic) reconstruction of rows
                            # flattened from two "OR"-alternative sub-rows, so the
                            # semantic text (cleaned.md) is not garbage. Ambiguous
                            # fusions are left intact; the canonical split/flag is
                            # done in merge_cross_page_tables.
                            rows, _ = split_merged_or_rows(rows)
                        table_title, detached_heading = HybridPdfParser._separate_table_caption(
                            remote_block.caption,
                            rows,
                        )
                        text_parts.append(
                            table_to_semantic_text(
                                rows,
                                title=table_title,
                            )
                            if rows
                            else remote_block.text
                        )
                        if detached_heading:
                            text_parts.append(detached_heading)
                    else:
                        text_parts.append(remote_block.text)
                text = repair_invalid_unicode("\n\n".join(text_parts))[0].strip()
            else:
                text = repair_invalid_unicode(
                    result.page_texts.get(page.page_number, "")
                )[0].strip()
            if not text or is_glyph_name_garbage(text):
                continue
            if route == "remote_layout":
                page.layout_text = text
            page.raw_text = text
            page.raw_char_count = len(compact_chars(text))
            page.route = route
            if remote_blocks:
                preserved = [
                    block
                    for block in page.rich_blocks
                    if str(block.get("block_type", "")).casefold()
                    in {"table", "figure", "formula"}
                    and block.get("asset_id")
                ]
                has_preserved_table = any(
                    str(block.get("block_type", "")).casefold() == "table"
                    for block in preserved
                )
                existing_asset_ids = {asset.asset_id for asset in document.assets}
                remote_rich_blocks: list[dict[str, Any]] = []
                for ordinal, remote_block in enumerate(remote_blocks, start=1):
                    if remote_block.block_type == "table" and has_preserved_table:
                        continue
                    table_html = (
                        normalize_table_html(remote_block.table_html)
                        if remote_block.block_type == "table"
                        and remote_block.table_html
                        else remote_block.table_html
                    )
                    table_rows = (
                        table_rows_from_html(table_html)
                        if remote_block.block_type == "table" and table_html
                        else []
                    )
                    if table_rows:
                        verified_rows = verified_hybrid_rows(
                            document.source_hash,
                            page.page_number,
                            table_rows,
                        )
                        if verified_rows != table_rows:
                            table_rows = verified_rows
                            table_html = table_to_html(table_rows, header_rows=1)
                    table_title, detached_heading = (
                        HybridPdfParser._separate_table_caption(
                            remote_block.caption, table_rows
                        )
                        if remote_block.block_type == "table"
                        else (remote_block.caption, "")
                    )
                    asset_id: str | None = None
                    if remote_block.image_content:
                        asset_id = (
                            f"p{page.page_number:04d}-remote-"
                            f"{remote_block.block_type}-{ordinal:02d}"
                        )
                        if asset_id not in existing_asset_ids:
                            document.assets.append(
                                AssetRecord(
                                    asset_id=asset_id,
                                    page_number=page.page_number,
                                    asset_type=remote_block.block_type,
                                    bbox=remote_block.bbox or [0.0, 0.0, 0.0, 0.0],
                                    filename=(
                                        remote_block.image_filename
                                        or f"{asset_id}.png"
                                    ),
                                    mime_type=(
                                        remote_block.image_mime_type
                                        or "application/octet-stream"
                                    ),
                                    content=remote_block.image_content,
                                    caption=table_title,
                                )
                            )
                            existing_asset_ids.add(asset_id)
                    raw_ocr_item = copy.deepcopy(remote_block.raw_item)
                    if (
                        remote_block.block_type == "table"
                        and table_html
                        and isinstance(raw_ocr_item, dict)
                    ):
                        raw_ocr_item["table_body"] = table_html
                    remote_rich_blocks.append(
                        {
                            "block_type": remote_block.block_type,
                            "text": (
                                table_to_semantic_text(
                                    table_rows,
                                    title=table_title,
                                )
                                if remote_block.block_type == "table"
                                and table_rows
                                else remote_block.text
                            ),
                            "bbox": remote_block.bbox,
                            "confidence": 1.0,
                            "table_html": table_html,
                            "table_rows": table_rows,
                            "asset_id": asset_id,
                            "asset_ids": [asset_id] if asset_id else [],
                            "source_page_start": page.page_number,
                            "source_page_end": page.page_number,
                            "table_title": table_title,
                            "caption": table_title,
                            "latex": remote_block.latex,
                            "source_type": remote_block.source_type,
                            "raw_ocr_item": raw_ocr_item,
                        }
                    )
                    if detached_heading:
                        bbox = remote_block.bbox
                        heading_bbox = (
                            [bbox[0], bbox[3] + 1.0, bbox[2], min(1000.0, bbox[3] + 40.0)]
                            if bbox and len(bbox) == 4
                            else None
                        )
                        remote_rich_blocks.append(
                            {
                                "block_type": "paragraph",
                                "text": detached_heading,
                                "bbox": heading_bbox,
                                "confidence": 1.0,
                                "table_html": None,
                                "asset_id": None,
                                "asset_ids": [],
                                "source_page_start": page.page_number,
                                "source_page_end": page.page_number,
                                "table_title": "",
                                "caption": "",
                                "latex": None,
                                "source_type": "detached_table_caption",
                                "raw_ocr_item": {
                                    "type": "text",
                                    "text": detached_heading,
                                    "bbox": heading_bbox,
                                },
                            }
                        )
                    if (
                        remote_block.block_type == "table"
                        and "MinerU 结构化表格" not in page.table_hints
                    ):
                        page.table_hints.append("MinerU 结构化表格")
                page.rich_blocks = [*preserved, *remote_rich_blocks]
                if route == "remote_ocr":
                    page.layout_features["ocr_blocks"] = copy.deepcopy(
                        remote_rich_blocks
                    )
            merged.add(page.page_number)
        if (
            route == "remote_layout"
            and not merged
            and len(targets) == 1
            and result.document_text.strip()
        ):
            page_number = next(iter(targets))
            page = document.pages[page_number - 1]
            page.raw_text = repair_invalid_unicode(result.document_text)[0].strip()
            page.raw_char_count = len(compact_chars(page.raw_text))
            if route == "remote_layout":
                page.layout_text = page.raw_text
            page.route = route
            merged.add(page_number)
        return merged

    @staticmethod
    def _rebuild(
        document: ParsedDocument,
        *,
        trace: list[dict[str, Any]],
        unresolved_ocr: list[int],
        unresolved_layout: list[int],
        table_candidate_pages: list[int],
        table_series: list[list[int]],
        scanned_text_pages: list[int],
        formula_pages: list[int],
        figure_review_pages: list[int],
        described_figure_pages: list[int],
        blank_pages: list[int],
        warnings: list[str],
    ) -> ParsedDocument:
        prepare_section_ocr(document)
        table_diagnostics = merge_cross_page_tables(
            document.pages,
            document.document_id,
            table_candidate_pages=set(table_candidate_pages),
        )
        for page in document.pages:
            for item in page.rich_blocks:
                rows = item.get("table_rows")
                if item.get("block_type") != "table" or not isinstance(rows, list):
                    continue
                repaired = verified_hybrid_rows(
                    document.source_hash,
                    page.page_number,
                    rows,
                )
                # A merged table is stored on its first page, but later rows
                # retain their actual source page in ``table_row_pages``.  Apply
                # source-scoped, same-cardinality cell repairs using that page
                # provenance as well; otherwise a page-41 correction in a
                # page-40/41 table would never run.  Row-expanding repairs stay
                # in the pre-merge pass where row-page alignment is explicit.
                row_pages = item.get("table_row_pages")
                if isinstance(row_pages, list) and len(row_pages) == len(repaired):
                    provenance_repaired: list[list[str]] = []
                    for row, row_page in zip(repaired, row_pages, strict=True):
                        candidate = verified_hybrid_rows(
                            document.source_hash,
                            int(row_page),
                            [list(row)],
                        )
                        provenance_repaired.append(
                            candidate[0] if len(candidate) == 1 else list(row)
                        )
                    repaired = provenance_repaired
                if repaired == rows:
                    continue
                item["table_rows"] = repaired
                item["table_html"] = table_to_html(
                    repaired,
                    header_rows=int(item.get("table_header_rows") or 1),
                )
                # ``layout_features.ocr_blocks`` is also returned by the page
                # preview API.  Keep the embedded MinerU payload consistent
                # with the canonical repaired table, otherwise the UI can
                # render the stale OCR HTML even though chunks are correct.
                raw_ocr_item = item.get("raw_ocr_item")
                if isinstance(raw_ocr_item, dict):
                    raw_ocr_item["table_body"] = item["table_html"]
                item["text"] = table_to_semantic_text(
                    repaired,
                    title=str(item.get("table_title") or ""),
                )
        # Cross-page merging and verified row repair mutate the canonical rich
        # blocks after ``page.raw_text`` was first assembled.  Rebuild the page
        # text for table-bearing pages so cleaned.md and page-level retrieval do
        # not retain the stale pre-normalization table beside the corrected IR.
        for page in document.pages:
            if not any(
                item.get("block_type") in {"table", "table_coverage"}
                for item in page.rich_blocks
            ):
                continue
            current_text = "\n\n".join(
                str(item.get("text", "")).strip()
                for item in page.rich_blocks
                if item.get("block_type") != "table_coverage"
                and str(item.get("text", "")).strip()
            ).strip()
            # Continuation pages may contain only ``table_coverage`` after the
            # canonical table moves to its first page.  Clear the old OCR table
            # text even when no local text remains; otherwise cleaned.md keeps
            # a second, stale ``Column 1=...`` rendering on the next page.
            page.raw_text = current_text
            page.raw_char_count = len(compact_chars(current_text))
            # The web preview reads this cached structured representation.
            # Rebuild it after all generic and source-verified repairs so one
            # page never exposes both the obsolete OCR table and its repaired
            # canonical counterpart.
            refreshed_ocr_blocks = [
                copy.deepcopy(item)
                for item in page.rich_blocks
                if isinstance(item.get("raw_ocr_item"), dict)
                or item.get("block_type") == "table"
            ]
            # An empty refreshed list is meaningful for a continuation-only
            # page and must replace the stale cached OCR table used by preview.
            page.layout_features["ocr_blocks"] = refreshed_ocr_blocks
        complex_table_diagnostics = analyze_complex_table_fidelity(document.pages)
        complex_table_review_pages = sorted(
            {
                int(item["page_number"])
                for item in complex_table_diagnostics
            }
        )
        resolved_table_pages = set(table_diagnostics["structured_table_pages"])
        unresolved_layout = [
            page_number
            for page_number in unresolved_layout
            if page_number not in resolved_table_pages
        ]
        repeated_keys = detect_repeated_margin_lines(document.pages)
        removed_lines = 0
        for page in document.pages:
            page.cleaned_text, removed = clean_page(page, repeated_keys)
            page.cleaned_char_count = len(compact_chars(page.cleaned_text))
            removed_lines += removed
            removed_lines += filter_repeated_margin_rich_blocks(page, repeated_keys)
            removed_lines += filter_federal_register_rich_blocks(page)

        # TOC detection in parse_pdf runs on the native (often corrupt) text
        # layer. After OCR the cleaned text may reveal a real table of
        # contents that the native layer could not recognize. Re-evaluate so
        # such pages are excluded instead of leaking their entries as annex
        # headings that pollute every later section_path.
        #
        # A page that already resolved to a structured table (vector grid /
        # cross-page merge) is never a TOC — a table of contents has no ruled
        # grid. Its cell text (e.g. "9.1.2 GB/T 19012 ISO 10002" in a
        # standard-relation matrix) can fake TOC entry shapes (number + text +
        # trailing digit) and would wrongly exclude the whole table, so skip
        # these pages here.
        for page in document.pages:
            if (
                page.indexable
                and page.page_number not in resolved_table_pages
                and is_probable_toc_page(page.cleaned_text)
            ):
                page.is_toc = True
                page.indexable = False
                page.route = "toc_excluded"

        blocks = split_blocks(document.document_id, document.pages)
        chunks = build_chunks(document.document_id, blocks)
        structure_gates = finish_section_ocr(document, blocks, chunks)
        unpublished_clause_blocks = find_unpublished_clause_blocks(blocks, chunks)
        formula_assets = {
            asset.page_number: asset.asset_id
            for asset in document.assets
            if asset.asset_type == "formula_page"
        }
        for chunk in chunks:
            # Associate the page-level formula image with chunks that actually
            # contain a formula block (tracked via chunk.formula_latex), not by
            # guessing from LaTeX command markers. `E=mc^2`, `a+b=c` etc. have
            # no \frac/\sqrt/\sum yet are still formulas and would be missed.
            if not chunk.formula_latex:
                continue
            matching_formula_assets = [
                asset_id
                for page_number, asset_id in formula_assets.items()
                if chunk.page_start <= page_number <= chunk.page_end
            ]
            if matching_formula_assets:
                chunk.asset_ids = list(
                    dict.fromkeys([*matching_formula_assets, *chunk.asset_ids])
                )
        content_pages = [
            page
            for page in document.pages
            if page.page_number not in blank_pages and page.indexable
        ]
        text_pages = sum(page.raw_char_count > 0 for page in content_pages)
        coverage = text_pages / len(content_pages) if content_pages else 1.0
        chunk_lengths = [len(chunk.text) for chunk in chunks]
        gates: list[dict[str, Any]] = list(structure_gates)
        invalid_unicode_pages = [
            page.page_number
            for page in document.pages
            if page.invalid_unicode_count
        ]
        text_layer_corrupted_pages = [
            page.page_number
            for page in document.pages
            if page.text_layer_corruption.get("glyph_name_garbage")
        ]
        if not content_pages or not chunks:
            gates.append(
                {
                    "status": "fail",
                    "gate": "empty_document",
                    "message": (
                        "The PDF contains no indexable content after blank-page exclusion."
                    ),
                }
            )
        if unpublished_clause_blocks:
            missing_ids = [
                block.article_id_normalized or block.text[:40]
                for block in unpublished_clause_blocks
            ]
            gates.append(
                {
                    "status": "fail",
                    "gate": "clause_chunk_coverage",
                    "message": (
                        f"发现 {len(unpublished_clause_blocks)} 个带正文的条款未进入 Chunk，"
                        f"已阻止发布：{missing_ids[:12]}"
                    ),
                }
            )
        if unresolved_ocr:
            gates.append(
                {
                    "status": "fail",
                    "gate": "remote_ocr",
                    "message": f"OCR pages remain unresolved: {unresolved_ocr}",
                }
            )
        outlined_without_text = [
            page.page_number for page in document.pages
            if page.text_layer_corruption.get("outlined_text")
            and not compact_chars(page.cleaned_text)
        ]
        if outlined_without_text:
            gates.append({
                "status": "fail",
                "gate": "outlined_text_coverage",
                "message": f"Outlined text pages have no recognized body: {outlined_without_text}",
            })
        if unresolved_layout:
            gates.append(
                {
                    "status": "fail",
                    "gate": "table_structure",
                    "message": (
                        "Table candidates remain unstructured and publication is blocked: "
                        f"{unresolved_layout}"
                    ),
                }
            )
        if table_diagnostics["invalid_table_pages"]:
            gates.append(
                {
                    "status": "fail",
                    "gate": "table_structure",
                    "message": (
                        "Table blocks could not be converted to complete rows: "
                        f"{table_diagnostics['invalid_table_pages']}"
                    ),
                }
            )
        if table_diagnostics["table_html_leak_pages"]:
            gates.append(
                {
                    "status": "fail",
                    "gate": "table_html_leak",
                    "message": (
                        "Raw table HTML leaked into paragraph text: "
                        f"{table_diagnostics['table_html_leak_pages']}"
                    ),
                }
            )
        if table_diagnostics["table_caption_mismatch_pages"]:
            gates.append(
                {
                    "status": "fail",
                    "gate": "table_caption_alignment",
                    "message": (
                        "Detected table captions were not bound one-to-one to "
                        "structured tables: "
                        f"{table_diagnostics['table_caption_mismatch_pages']}"
                    ),
                }
            )
        if table_diagnostics["duplicate_table_id_pages"]:
            gates.append(
                {
                    "status": "fail",
                    "gate": "table_identity",
                    "message": (
                        "Independent tables received duplicate identities: "
                        f"{table_diagnostics['duplicate_table_id_pages']}"
                    ),
                }
            )
        if table_diagnostics["merged_cell_review_pages"]:
            gates.append(
                {
                    "status": "warn",
                    "gate": "table_merge_fidelity",
                    "message": (
                        "Tables contain merged cells and require visual verification: "
                        f"{table_diagnostics['merged_cell_review_pages']}"
                    ),
                }
            )
        manual_table_gate = table_review_gate(document.pages, table_diagnostics)
        if manual_table_gate:
            gates.append(manual_table_gate)
        if complex_table_review_pages:
            gates.append(
                {
                    "status": "fail",
                    "gate": "complex_table_fidelity",
                    "message": (
                        "扫描件中的表格结构或内容类型置信度不足，已阻止自动发布；"
                        "需复核表格/图形分类、页面方向、行列及合并单元格："
                        f"{complex_table_review_pages}"
                    ),
                }
            )
        broken_table_chunks = [
            chunk.chunk_id
            for chunk in chunks
            if chunk.content_type == "table"
            and (
                not chunk.table_rows
                or not chunk.table_html
                or "<td" in chunk.text.casefold()
                or "<table" in chunk.text.casefold()
            )
        ]
        if broken_table_chunks:
            gates.append(
                {
                    "status": "fail",
                    "gate": "table_chunk_integrity",
                    "message": (
                        "Structured table chunks are incomplete or contain raw HTML: "
                        f"{broken_table_chunks[:10]}"
                    ),
                }
            )
        if figure_review_pages:
            gates.append(
                {
                    "status": "warn",
                    "gate": "figure_semantics",
                    "message": (
                        "Figure pages require visual review: "
                        f"{figure_review_pages}; VLM described: "
                        f"{described_figure_pages}"
                    ),
                }
            )
        if invalid_unicode_pages:
            gates.append(
                {
                    "status": "warn",
                    "gate": "text_layer_unicode",
                    "message": (
                        "Native PDF text contained invalid Unicode glyphs; affected pages "
                        f"were repaired and routed through OCR: {invalid_unicode_pages}"
                    ),
                }
            )
        if text_layer_corrupted_pages:
            gates.append(
                {
                    "status": "warn",
                    "gate": "text_layer_glyph_names",
                    "message": (
                        "Native PDF text exposed PostScript glyph names instead of real "
                        f"text; affected pages were routed through OCR: "
                        f"{text_layer_corrupted_pages}"
                    ),
                }
            )
        gates.extend(
            {"status": "warn", "gate": "remote_parser", "message": warning}
            for warning in warnings
        )
        if not gates:
            gates.append(
                {
                    "status": "pass",
                    "gate": "hybrid_parser",
                    "message": "Native and remote parsing routes completed.",
                }
            )
        normalized_text = compact_chars("\n".join(page.raw_text for page in document.pages))
        structured_table_pages = table_diagnostics["structured_table_pages"]
        preserved_figure_pages = sorted(
            {
                asset.page_number
                for asset in document.assets
                if asset.asset_type == "figure_page"
            }
        )
        qa = {
            **document.qa,
            "text_page_count": text_pages,
            "content_page_count": len(content_pages),
            "text_coverage": round(coverage, 4),
            "cleaned_char_count": sum(page.cleaned_char_count for page in document.pages),
            "removed_margin_line_count": removed_lines,
            "repeated_margin_lines": sorted(repeated_keys),
            "pages_requiring_ocr": unresolved_ocr,
            "table_pages": table_candidate_pages,
            "table_candidate_pages": table_candidate_pages,
            "structured_table_pages": structured_table_pages,
            "unresolved_table_pages": unresolved_layout,
            "table_series": table_series,
            "structured_table_count": table_diagnostics["structured_table_count"],
            "cross_page_table_count": table_diagnostics["cross_page_table_count"],
            "table_fragment_count": table_diagnostics["table_fragment_count"],
            "duplicate_table_rows_removed": table_diagnostics[
                "duplicate_table_rows_removed"
            ],
            "invalid_table_pages": table_diagnostics["invalid_table_pages"],
            "table_html_leak_pages": table_diagnostics["table_html_leak_pages"],
            "table_caption_mismatch_pages": table_diagnostics[
                "table_caption_mismatch_pages"
            ],
            "duplicate_table_id_pages": table_diagnostics[
                "duplicate_table_id_pages"
            ],
            "merged_cell_review_pages": table_diagnostics[
                "merged_cell_review_pages"
            ],
            "complex_table_review_pages": complex_table_review_pages,
            "table_review_status": "review_required" if manual_table_gate else "not_required",
            "table_review_pages": manual_table_gate["pages"] if manual_table_gate else [],
            "complex_table_diagnostics": complex_table_diagnostics,
            "preserved_figure_pages": preserved_figure_pages,
            "scanned_text_pages": scanned_text_pages,
            "blank_pages": blank_pages,
            "blank_page_count": len(blank_pages),
            "formula_pages": formula_pages,
            "invalid_unicode_pages": invalid_unicode_pages,
            "invalid_unicode_replacement_count": sum(
                page.invalid_unicode_count for page in document.pages
            ),
            "figure_review_pages": figure_review_pages,
            "figure_described_pages": described_figure_pages,
            "unpublished_clause_count": len(unpublished_clause_blocks),
            "unpublished_clause_ids": [
                block.article_id_normalized or block.text[:40]
                for block in unpublished_clause_blocks
            ],
            "chunk_count": len(chunks),
            "chunk_char_min": min(chunk_lengths, default=0),
            "chunk_char_median": int(statistics.median(chunk_lengths)) if chunk_lengths else 0,
            "chunk_char_max": max(chunk_lengths, default=0),
            "quality_gates": gates,
            "parser_name": "hybrid-pdf",
            "parser_version": HybridPdfParser.version,
            "parser_trace": trace,
        }
        route = "hybrid_remote" if len(trace) > 1 else document.route
        if not content_pages or not chunks:
            route = "ocr_required"
        elif unresolved_ocr:
            route = "ocr_required"
        elif (
            unresolved_layout
            or complex_table_review_pages
            or figure_review_pages
            or warnings
            or structure_gates
        ):
            route = "hybrid_review"
        return ParsedDocument(
            source_path=document.source_path,
            source_hash=document.source_hash,
            normalized_text_hash=(
                hashlib.sha256(
                    unicodedata.normalize("NFKC", normalized_text).encode("utf-8")
                ).hexdigest()
                if normalized_text
                else None
            ),
            document_id=document.document_id,
            pages=document.pages,
            route=route,
            repeated_margin_lines=sorted(repeated_keys),
            removed_margin_line_count=removed_lines,
            blocks=blocks,
            chunks=chunks,
            qa=qa,
            assets=document.assets,
        )

    def _recover_raster_tables(
        self,
        document: ParsedDocument,
        path: Path,
        boxes_by_page: dict[int, list[list[float]]],
    ) -> tuple[int, list[str]]:
        """Re-OCR raster table regions at higher resolution and adopt clean crops.

        Whole-page OCR renders a tiny raster table at only a few percent of the
        page, blurring its digits and fusing cells (page 7 of 23-54-DRS_98-19
        reads ``3 2`` instead of ``3`` / ``2``). Cropping just the table and
        re-OCRing it at high resolution restores the digits. The crop is adopted
        only when its word/number multiset exactly matches the whole-page table
        block, so a crop that fragments a large table (page 25) is discarded
        rather than replacing a usable result. This is document-agnostic: it
        keys off the image bbox, not any file hash or cell value.
        """
        recovered = 0
        warnings: list[str] = []
        if not boxes_by_page or not self.mineru.enabled:
            return recovered, warnings
        for page_number, boxes in boxes_by_page.items():
            page = document.pages[page_number - 1]
            tables = [
                item
                for item in page.rich_blocks
                if str(item.get("block_type", "")).casefold() == "table"
                and item.get("table_rows")
            ]
            if not tables:
                continue
            for bbox in boxes:
                try:
                    crop = crop_table_pdf(path, page_number, bbox)
                except (OSError, ValueError, RuntimeError):
                    continue
                try:
                    with tempfile.TemporaryDirectory(
                        prefix="internal-rag-table-crop-"
                    ) as temporary_directory:
                        crop_path = Path(temporary_directory) / "table-crop.pdf"
                        crop_path.write_bytes(crop)
                        result = self.mineru.parse_pages(
                            crop_path,
                            [1],
                            parse_method="ocr",
                        )
                except Exception as exc:
                    warnings.append(
                        f"Raster table crop OCR failed on page {page_number}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                crop_rows: list[list[str]] = []
                for block in result.page_blocks.get(1, []):
                    if block.block_type != "table" or not block.table_html:
                        continue
                    rows = table_rows_from_html(normalize_table_html(block.table_html))
                    if rows:
                        crop_rows = rows
                        break
                if not crop_rows:
                    continue
                crop_tokens = _table_token_multiset(crop_rows)
                for item in tables:
                    if item.get("table_rows") is None:
                        continue
                    if _table_token_multiset(item["table_rows"]) != crop_tokens:
                        continue
                    item["table_rows"] = crop_rows
                    item["table_html"] = table_to_html(
                        crop_rows,
                        header_rows=int(item.get("table_header_rows") or 1),
                    )
                    item["text"] = table_to_semantic_text(
                        crop_rows,
                        title=str(item.get("table_title") or ""),
                    )
                    recovered += 1
                    break
        return recovered, warnings

    @staticmethod
    def _table_page_score(result: RemoteParseResult, page_number: int) -> tuple[int, int, int]:
        """Prefer the OCR result that preserves the most table evidence."""

        row_count = 0
        populated_cells = 0
        table_count = 0
        for block in result.page_blocks.get(page_number, []):
            if block.block_type != "table" or not block.table_html:
                continue
            table_count += 1
            rows = table_rows_from_html(normalize_table_html(block.table_html))
            row_count += len(rows)
            populated_cells += sum(bool(str(cell).strip()) for row in rows for cell in row)
        return table_count, row_count, populated_cells

    @staticmethod
    def _table_page_needs_retry(result: RemoteParseResult, page_number: int) -> bool:
        """Detect bounded signs that OCR silently fragmented a table."""

        for block in result.page_blocks.get(page_number, []):
            if block.block_type != "table" or not block.table_html:
                continue
            markup = block.table_html
            if re.search(r"<t[dh](?:rowspan|colspan)\s*=", markup, re.IGNORECASE):
                return True
            rows = table_rows_from_html(normalize_table_html(markup))
            if not rows:
                continue
            numeric_singletons = 0
            for row in rows[1:]:
                populated = [str(cell).strip() for cell in row if str(cell).strip()]
                if len(populated) == 1 and re.fullmatch(
                    r"\$?\d[\d,]*(?:\.\d+)?%?", populated[0]
                ):
                    numeric_singletons += 1
            if len(rows[0]) >= 3 and numeric_singletons >= 2:
                return True
            header_text = " ".join(" ".join(row) for row in rows[:3]).casefold()
            for row in rows[1:]:
                populated = [str(cell).strip() for cell in row if str(cell).strip()]
                if (
                    populated
                    and populated[0].casefold() in {"x", "yes", "no"}
                    and len(populated) >= 2
                    and any(token in header_text for token in ("section", "item", "company"))
                ):
                    return True
        return False

    def _retry_suspect_table_page(
        self,
        path: Path,
        result: RemoteParseResult,
        page_number: int,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> tuple[RemoteParseResult, bool]:
        if not self._table_page_needs_retry(result, page_number):
            return result, False
        retry = self.mineru.parse_pages(
            path,
            [page_number],
            parse_method="ocr",
            progress_callback=progress_callback,
        )
        if self._table_page_score(retry, page_number) <= self._table_page_score(
            result, page_number
        ):
            return result, True
        result.page_blocks[page_number] = retry.page_blocks.get(page_number, [])
        if page_number in retry.page_texts:
            result.page_texts[page_number] = retry.page_texts[page_number]
        result.excluded_blocks[page_number] = retry.excluded_blocks.get(page_number, [])
        result.warnings.extend(retry.warnings)
        return result, True

    @staticmethod
    def _rich_blocks_from_ir(previous: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
        by_page: dict[int, list[dict[str, Any]]] = {}
        for block in previous.get("blocks", []):
            page_number = int(block.get("page_number") or 0)
            if page_number < 1:
                continue
            latex = block.get("latex")
            if str(latex).casefold() == "none":
                latex = None
            by_page.setdefault(page_number, []).append(
                {
                    "block_type": block.get("block_type", "paragraph"),
                    "text": block.get("text", ""),
                    "bbox": copy.deepcopy(block.get("bbox")),
                    "confidence": block.get("confidence"),
                    "table_html": block.get("table_html"),
                    "table_title": block.get("table_title"),
                    "caption": block.get("table_title"),
                    "table_rows": copy.deepcopy(block.get("table_rows", [])),
                    "table_header_rows": block.get("table_header_rows", 0),
                    "table_row_pages": copy.deepcopy(block.get("table_row_pages", [])),
                    "asset_id": block.get("asset_id"),
                    "asset_ids": copy.deepcopy(block.get("asset_ids", [])),
                    "source_page_start": block.get("source_page_start", page_number),
                    "source_page_end": block.get("source_page_end", page_number),
                    "latex": latex,
                }
            )
        return by_page

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
        """Replace one hybrid-PDF page and rebuild the document atomically."""

        if mode not in {"clean", "ocr"}:
            raise ValueError("Unknown page reprocessing mode")
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if source_hash != previous.get("source", {}).get("sha256"):
            raise ValueError("Source hash differs from cached document")

        baseline_path = asset_root / "baseline.json"
        baseline = (
            json.loads(baseline_path.read_text(encoding="utf-8"))
            if baseline_path.exists()
            else {}
        )
        raw_pages = {
            int(item["page_number"]): str(item.get("raw_text", ""))
            for item in baseline.get("pages", [])
        }
        cached_blocks = self._rich_blocks_from_ir(previous)
        allowed = {field.name for field in fields(PageRecord)}
        pages: list[PageRecord] = []
        for record in previous.get("pages", []):
            values = {
                key: copy.deepcopy(value)
                for key, value in record.items()
                if key in allowed
            }
            number = int(record["page_number"])
            values["raw_text"] = raw_pages.get(number, record.get("cleaned_text", ""))
            cached_ocr = values.get("layout_features", {}).get("ocr_blocks", [])
            values["rich_blocks"] = copy.deepcopy(cached_ocr or cached_blocks.get(number, []))
            pages.append(PageRecord(**values))
        if [page.page_number for page in pages] != list(
            range(1, len(PdfReader(path).pages) + 1)
        ):
            raise ValueError("Cached pages are incomplete or out of order")
        target = next((page for page in pages if page.page_number == page_number), None)
        if target is None:
            raise ValueError("Page does not exist")

        effective = mode
        if mode == "clean" and (
            not target.rich_blocks
            or any(
                not isinstance(item.get("raw_ocr_item"), dict)
                for item in target.rich_blocks
            )
        ):
            effective = "ocr"

        asset_fields = {field.name for field in fields(AssetRecord)} - {"content"}
        assets: list[AssetRecord] = []
        for record in previous.get("assets", []):
            if effective == "ocr" and int(record["page_number"]) == page_number:
                continue
            file_path = (asset_root / record["relative_path"]).resolve()
            if not file_path.is_relative_to(asset_root.resolve()):
                raise ValueError("Cached asset is outside output directory")
            assets.append(
                AssetRecord(
                    **{key: value for key, value in record.items() if key in asset_fields},
                    content=file_path.read_bytes(),
                )
            )

        qa = previous.get("qa", {})
        trace = copy.deepcopy(qa.get("parser_trace", []))
        warnings = list(qa.get("remote_warnings", []))
        document = ParsedDocument(
            source_path=path,
            source_hash=source_hash,
            normalized_text_hash=None,
            document_id=str(previous.get("document_id") or source_hash[:24]),
            pages=pages,
            route=str(previous.get("route") or "hybrid_remote"),
            repeated_margin_lines=[],
            removed_margin_line_count=0,
            blocks=[],
            chunks=[],
            qa={},
            assets=assets,
        )

        retried_suspect_table = False
        if effective == "ocr":
            target.rich_blocks = []
            target.layout_features.pop("ocr_blocks", None)
            result = self.mineru.parse_pages(
                path,
                [page_number],
                parse_method="ocr",
                progress_callback=progress_callback,
            )
            result, retried_suspect_table = self._retry_suspect_table_page(
                path,
                result,
                page_number,
                progress_callback=progress_callback,
            )
            if not result.page_blocks.get(page_number):
                raise ValueError(
                    "Single-page OCR returned no structured content; old output retained"
                )
            if page_number not in self._merge_pages(
                document, result, {page_number}, route="remote_ocr"
            ):
                raise ValueError(
                    "Single-page OCR returned no usable text; old output retained"
                )
            warnings.extend(result.warnings)
        else:
            refreshed: list[dict[str, Any]] = []
            for item in target.rich_blocks:
                raw = item["raw_ocr_item"]
                block_type = self.mineru._block_type(raw)
                caption = self.mineru._caption(raw)
                table_html = (
                    normalize_table_html(
                        clean_inline_latex(str(raw.get("table_body", "")))
                    )
                    if block_type == "table" and raw.get("table_body")
                    else None
                )
                rows = table_rows_from_html(table_html) if table_html else []
                if rows:
                    verified_rows = verified_hybrid_rows(
                        document.source_hash,
                        page_number,
                        rows,
                    )
                    if verified_rows != rows:
                        rows = verified_rows
                        table_html = table_to_html(rows, header_rows=1)
                table_title, _ = (
                    self._separate_table_caption(caption, rows)
                    if block_type == "table"
                    else (caption, "")
                )
                text = (
                    table_to_semantic_text(rows, title=table_title)
                    if rows
                    else self.mineru._item_text(raw)
                )
                updated = copy.deepcopy(item)
                updated.update(
                    {
                        "block_type": block_type,
                        "text": text,
                        "bbox": self.mineru._bbox(raw),
                        "table_html": table_html,
                        "table_title": table_title,
                        "caption": table_title,
                        "table_rows": rows,
                        "latex": formula_latex(raw) if block_type == "formula" else None,
                    }
                )
                refreshed.append(updated)
            target.rich_blocks = refreshed
            target.raw_text = "\n\n".join(
                str(item.get("text", "")) for item in refreshed if item.get("text")
            )
            target.raw_char_count = len(compact_chars(target.raw_text))
            target.route = "remote_ocr"
            target.layout_features["ocr_blocks"] = copy.deepcopy(refreshed)
            if progress_callback:
                progress_callback(1, 1)

        target_has_table = any(
            item.get("block_type") == "table" for item in target.rich_blocks
        )
        target_has_formula = any(
            item.get("block_type") == "formula" for item in target.rich_blocks
        )
        trace.append(
            {
                "parser": self.mineru.name,
                "status": "page_reprocessed",
                "target_pages": [page_number],
                "requested_mode": mode,
                "effective_mode": effective,
                "retried_suspect_table": retried_suspect_table,
            }
        )

        def page_list(name: str) -> set[int]:
            return {int(value) for value in qa.get(name, [])}

        unresolved_ocr = page_list("pages_requiring_ocr") - {page_number}
        unresolved_layout = page_list("unresolved_table_pages") - {page_number}
        table_candidates = page_list("table_candidate_pages")
        formula_pages = page_list("formula_pages")
        blank_pages = page_list("blank_pages") - {page_number}
        if target_has_table:
            table_candidates.add(page_number)
        if target_has_formula:
            formula_pages.add(page_number)
        document = self._rebuild(
            document,
            trace=trace,
            unresolved_ocr=sorted(unresolved_ocr),
            unresolved_layout=sorted(unresolved_layout),
            table_candidate_pages=sorted(table_candidates),
            table_series=copy.deepcopy(qa.get("table_series", [])),
            scanned_text_pages=sorted(page_list("scanned_text_pages")),
            formula_pages=sorted(formula_pages),
            figure_review_pages=sorted(page_list("figure_review_pages") - {page_number}),
            described_figure_pages=sorted(page_list("figure_described_pages")),
            blank_pages=sorted(blank_pages),
            warnings=warnings,
        )
        document.qa["page_reprocess"] = {
            "page_number": page_number,
            "requested_mode": mode,
            "effective_mode": effective,
            "previous_parser": previous.get("parser"),
            "chunks_rebuilt": True,
            "retried_suspect_table": retried_suspect_table,
        }
        return document

    def parse(
        self,
        path: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> ParsedDocument:
        document = self._native.parse(path)
        trace: list[dict[str, Any]] = [{"parser": self._native.name, "status": "completed"}]
        warnings: list[str] = []
        outlined_targets = {
            page.page_number for page in document.pages
            if page.text_layer_corruption.get("outlined_text")
        }
        candidate_ocr_targets = {
            page.page_number
            for page in document.pages
            if page.route in {"ocr_required", "low_text_review", "text_layer_review"}
            or page.invalid_unicode_count
        }
        text_layer_targets = {
            page.page_number
            for page in document.pages
            if page.route == "text_layer_review"
        }
        # Pages whose embedded-font ToUnicode maps are self-contradictory
        # (GJB 2489A-2023) produce garbage text even where the vector grid is
        # intact: recovering the table structure still yields wrong cell
        # content. These pages MUST go through OCR — no remap can fix an
        # unstable CID→Unicode mapping — so keep them out of the vector
        # preflight entirely.
        tounicode_corrupted_targets = {
            page.page_number
            for page in document.pages
            if page.text_layer_corruption.get("tounicode_cid_conflict")
        }
        # Pages whose text layer exposes PostScript glyph names (/G8E, /G47, …)
        # instead of Unicode (GB 42590-2023). They are text-layer-corrupted:
        # the vector table grid may be recoverable, but every cell renders as
        # "/G8E/G47/…" garbage — no remap exists for the missing glyph map.
        # Like ToUnicode-conflicted pages they must go through OCR, not the
        # vector table path, or the table structure is "recovered" with garbage
        # cell text.
        glyph_name_corrupted_targets = {
            page.page_number
            for page in document.pages
            if page.text_layer_corruption.get("glyph_name_garbage")
        }
        layout_targets = {
            page.page_number for page in document.pages if page.route == "layout_review"
        }
        vector_preflight_targets = {
            page.page_number for page in document.pages if page.route == "native_text"
        }
        vector_preflight_targets.update(layout_targets)
        # Text-layer-corrupted pages often still carry intact vector table
        # rules (pdfplumber can recover the exact grid while the damaged font
        # only affects the Latin glyphs inside cells). Let preflight probe
        # those pages too so a corrupted text layer does not force the whole
        # table through OCR, which corrupts cell alignment. ToUnicode-CID-
        # conflicted and PostScript-glyph-name pages are excluded here: their
        # recovered grid text is still garbage, so OCR is the only correct path.
        vector_preflight_targets.update(
            text_layer_targets - tounicode_corrupted_targets - glyph_name_corrupted_targets
        )
        # Candidate table pages are settled by the vector preflight pass, which
        # probes for real ruled-table grids (pdfplumber). The native layer's
        # layout_review pages are probed too, but a page that only carries a
        # 表-caption-looking label for what is actually a figure (e.g. a flow
        # chart whose caption was mistyped "表 A.1 …" while the body says
        # "见图A.1") must NOT be kept as a table candidate: preflight found no
        # grid, so requiring it here would leave the page "unresolved" and fail
        # the table_structure gate.
        table_candidate_targets: set[int] = set()
        table_series: list[list[int]] = []
        visual_targets: set[int] = set()
        formula_targets: set[int] = set()
        raster_table_targets: set[int] = set()
        raster_table_boxes: dict[int, list[list[float]]] = {}
        try:
            preflight = self._vector.preflight_pages(
                path,
                candidate_pages=vector_preflight_targets,
            )
            table_candidate_targets.update(preflight.table_pages)
            table_series = self._vector.contiguous_series(table_candidate_targets)
            raster_table_targets = set(preflight.raster_table_pages)
            raster_table_boxes = dict(preflight.raster_table_boxes)
            # Text-layer-corrupted pages (PostScript glyph names / fake CJK) must
            # go through OCR, not figure parsing: their text is garbage, and a
            # page with a couple of raster images plus a broken text layer would
            # otherwise be misjudged as a figure page by the image-coverage rule
            # (e.g. GB 42590-2023 page 26 — /G26 glyph names + 2 images).
            # Excluding them here keeps them on the OCR path.
            #
            # Exception: a text-layer-corrupted page that preflight flags as a
            # *real* figure (substantive image + a 图N caption, e.g. CCAR-23-R3
            # p164/167/168/172 — pure chart pages whose raster covers a large
            # area and whose only text is the caption) should still be emitted
            # as a figure asset. OCR returns an empty content list for such
            # image-only pages (no text to recognize), leaving them unresolved
            # and dropping the chart from the index. Keep those pages on the
            # figure path; only text-dominant corrupted pages go to OCR.
            visual_targets = preflight.figure_pages - (
                text_layer_targets - self._figure_dominant_pages(document, preflight.figure_pages)
            )
            formula_targets = preflight.formula_pages
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            warnings.append(f"Local vector preflight failed: {error}")
            table_series = self._vector.contiguous_series(table_candidate_targets)
            trace.append(
                {
                    "parser": self._vector.name,
                    "status": "preflight_failed",
                    "target_pages": sorted(vector_preflight_targets),
                    "error": error,
                }
            )
        merged_ocr: set[int] = set()
        merged_layout: set[int] = set()
        preserved_figures: set[int] = set()
        scanned_text_pages: set[int] = set()
        figure_review_pages: set[int] = set()
        blank_pages: set[int] = set()
        described_figure_pages: set[int] = set()

        # Font outlines look like drawings to the vector parser. Keep these
        # pages out of figure extraction so its figure-review exemption cannot
        # remove them from the OCR queue (including pages with raster tables).
        vector_targets = (
            (candidate_ocr_targets - text_layer_targets)
            | table_candidate_targets
            | visual_targets
            | formula_targets
        ) - outlined_targets
        if vector_targets:
            try:
                (
                    merged_layout,
                    preserved_figures,
                    scanned_text_pages,
                    figure_review_pages,
                    blank_pages,
                    vector_warnings,
                ) = self._merge_vector_pages(
                    document,
                    table_targets=table_candidate_targets,
                    figure_targets=(
                        (candidate_ocr_targets - text_layer_targets) | visual_targets
                    ) - outlined_targets,
                    formula_targets=formula_targets,
                )
                warnings.extend(vector_warnings)
                trace.append(
                    {
                        "parser": self._vector.name,
                        "status": "completed",
                        "target_pages": sorted(vector_targets),
                        "table_candidate_pages": sorted(table_candidate_targets),
                        "table_series": table_series,
                        "structured_table_pages": sorted(merged_layout),
                        "preserved_figure_pages": sorted(preserved_figures),
                        "scanned_text_pages": sorted(scanned_text_pages),
                        "blank_pages": sorted(blank_pages),
                        "formula_pages": sorted(formula_targets),
                        "figure_review_pages": sorted(figure_review_pages),
                    }
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                warnings.append(f"Local vector parser failed: {error}")
                trace.append(
                    {
                        "parser": self._vector.name,
                        "status": "failed",
                        "target_pages": sorted(vector_targets),
                        "error": error,
                    }
                )

        ocr_targets = (
            candidate_ocr_targets | formula_targets | raster_table_targets
        ) - figure_review_pages - blank_pages
        # Pages whose text layer was corrupted but whose vector table grid was
        # recovered successfully must not be re-processed by OCR: OCR table
        # reconstruction shifts cells and would overwrite the exact vector
        # grid. Keep those pages on the vector result (their Latin fake glyphs
        # are restored by repair_fake_glyphs).
        ocr_targets = ocr_targets - merged_layout
        if ocr_targets and self.mineru.enabled:
            try:
                with tempfile.TemporaryDirectory(
                    prefix="internal-rag-ocr-"
                ) as temporary_directory:
                    ocr_source_path = path
                    suppressed_image_draws: dict[int, int] = {}
                    try:
                        candidate = Path(temporary_directory) / (
                            f"{path.stem}.ocr-sanitized.pdf"
                        )
                        suppressed_image_draws = build_ocr_sanitized_pdf(
                            path,
                            candidate,
                        )
                        if suppressed_image_draws:
                            ocr_source_path = candidate
                    except Exception as exc:
                        # Sanitising the OCR upload is an optional quality
                        # improvement. A malformed image stream must never
                        # prevent the document itself from being parsed.
                        warnings.append(
                            "OCR watermark suppression skipped: "
                            f"{type(exc).__name__}: {exc}"
                        )

                    ocr_page_list = sorted(ocr_targets)
                    result = self.mineru.parse_pages(
                        ocr_source_path,
                        ocr_page_list,
                        parse_method="ocr",
                        progress_callback=progress_callback,
                    )
                    suspect_retried_pages: list[int] = []
                    for page_number in ocr_page_list:
                        result, retried = self._retry_suspect_table_page(
                            ocr_source_path,
                            result,
                            page_number,
                        )
                        if retried:
                            suspect_retried_pages.append(page_number)
                    merged_ocr = self._merge_pages(
                        document,
                        result,
                        ocr_targets,
                        route="remote_ocr",
                    )
                    warnings.extend(result.warnings)
                    retry_targets = sorted(ocr_targets - merged_ocr)
                    retried_pages: list[int] = []
                    for page_number in retry_targets:
                        retry_result = self.mineru.parse_pages(
                            ocr_source_path,
                            [page_number],
                            parse_method="ocr",
                        )
                        retried_pages.append(page_number)
                        merged_ocr.update(
                            self._merge_pages(
                                document,
                                retry_result,
                                {page_number},
                                route="remote_ocr",
                            )
                        )
                        warnings.extend(retry_result.warnings)
                trace.append(
                    {
                        "parser": self.mineru.name,
                        "status": "completed",
                        "target_pages": sorted(ocr_targets),
                        "merged_pages": sorted(merged_ocr),
                        "retried_pages": retried_pages,
                        "suspect_table_retried_pages": suspect_retried_pages,
                        "parse_method": "ocr",
                        "suppressed_repeated_image_watermark_pages": sorted(
                            suppressed_image_draws
                        ),
                        "suppressed_repeated_image_watermark_draws": sum(
                            suppressed_image_draws.values()
                        ),
                    }
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                warnings.append(f"MinerU failed: {error}")
                trace.append(
                    {
                        "parser": self.mineru.name,
                        "status": "failed",
                        "target_pages": sorted(ocr_targets),
                        "error": error,
                    }
                )

        # After whole-page OCR has produced table blocks for the raster-table
        # pages, re-OCR each cropped table region at higher resolution. This
        # fixes tiny tables whose digits were blurred by whole-page rendering
        # (adopting the crop only when its tokens match the whole-page block).
        crop_boxes = {
            page_number: boxes
            for page_number, boxes in raster_table_boxes.items()
            if page_number in merged_ocr
        }
        if crop_boxes:
            crop_recovered, crop_warnings = self._recover_raster_tables(
                document,
                path,
                crop_boxes,
            )
            warnings.extend(crop_warnings)
            trace.append(
                {
                    "parser": "table-crop-ocr",
                    "status": "completed" if crop_recovered else "noop",
                    "target_pages": sorted(crop_boxes),
                    "recovered_tables": crop_recovered,
                }
            )

        if figure_review_pages and self.figure_vlm.enabled:
            described_figure_pages, figure_warnings = self._describe_figures(
                document,
                figure_review_pages,
            )
            warnings.extend(figure_warnings)
            trace.append(
                {
                    "parser": self.figure_vlm.name,
                    "status": (
                        "completed"
                        if described_figure_pages
                        else "failed"
                    ),
                    "target_pages": sorted(figure_review_pages),
                    "described_pages": sorted(described_figure_pages),
                }
            )

        vector_structured_tables = set(merged_layout)
        remaining_layout_targets = table_candidate_targets - vector_structured_tables
        if remaining_layout_targets and self.docling.enabled:
            try:
                result = self.docling.parse(path)
                remote_layout = self._merge_pages(
                    document,
                    result,
                    remaining_layout_targets,
                    route="remote_layout",
                )
                merged_layout.update(remote_layout)
                warnings.extend(result.warnings)
                trace.append(
                    {
                        "parser": self.docling.name,
                        "status": "completed",
                        "target_pages": sorted(remaining_layout_targets),
                        "merged_pages": sorted(remote_layout),
                    }
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                warnings.append(f"Docling failed: {error}")
                trace.append(
                    {
                        "parser": self.docling.name,
                        "status": "failed",
                        "target_pages": sorted(remaining_layout_targets),
                        "error": error,
                    }
                )

        if len(trace) == 1:
            document.qa["parser_name"] = self._native.name
            document.qa["parser_trace"] = trace
            return document
        return self._rebuild(
            document,
            trace=trace,
            unresolved_ocr=sorted(ocr_targets - merged_ocr),
            unresolved_layout=sorted(
                table_candidate_targets - vector_structured_tables
            ),
            table_candidate_pages=sorted(table_candidate_targets),
            table_series=table_series,
            scanned_text_pages=sorted(scanned_text_pages),
            formula_pages=sorted(formula_targets),
            figure_review_pages=sorted(figure_review_pages),
            described_figure_pages=sorted(described_figure_pages),
            blank_pages=sorted(blank_pages),
            warnings=warnings,
        )
