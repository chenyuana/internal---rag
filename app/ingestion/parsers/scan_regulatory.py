"""Evidence-preserving OCR for scanned English regulations.

Marginalia and uncertain annotations are retained in the IR outside search.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pdfplumber
from pypdf import PdfReader

from app.ingestion.parsers.base import ParserPlugin
from app.ingestion.parsers.remote import MinerUClient, RemoteParseResult
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
    ("section", r"§\s*[A-Z]?\d+\.\d+\s+.+"),
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
    for kind, pattern in _HEADING_PATTERNS:
        if re.fullmatch(pattern, title, re.I):
            return kind, title
    return None


def _annotation_reason(item: dict[str, Any], body_top: float | None) -> str | None:
    """Quarantine uncertain marginal material, never arbitrary misspelled prose.

    Content-list bounding boxes are normalized to 0..1000. A short heading-like
    block ABOVE a regulatory heading is a review candidate, not proof of handwriting.
    """
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
    version = "2"
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
                    "table_html": block.table_html,
                    "caption": block.caption,
                    "latex": block.latex,
                    "asset_id": asset_id,
                    "asset_ids": [asset_id] if asset_id else [],
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
        document_id = stable_id("document", source_hash)
        assets: list[AssetRecord] = []
        pages: list[PageRecord] = []
        fallback_pages: list[int] = []
        for number in numbers:
            items = self._page_blocks_from(result, number, assets)
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
            raw = "\n\n".join(b["text"] for b in items if b["text"]) or raw
            clean = "\n\n".join(b["text"] for b in items if not b["exclusion_reason"] and b["text"])
            body_assets = [
                b["asset_id"] for b in items if not b["exclusion_reason"] and b.get("asset_id")
            ]
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
                    layout_features={"ocr_blocks": items},
                )
            )
        blocks = self._build_blocks(document_id, pages)
        self._render_pages(pages, blocks, assets)
        chunks = self._build_chunks(document_id, blocks)
        qa = self._quality(pages, blocks, chunks, assets, result.warnings, fallback_pages)
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
            for ordinal, item in enumerate(page.rich_blocks):
                text = str(item.get("text") or "").strip()
                block_type = str(item.get("block_type") or "paragraph")
                heading = english_heading(text) if block_type in {"paragraph", "heading"} else None
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
                blocks.append(
                    BlockRecord(
                        block_id=block_id,
                        block_type=block_type,
                        text=text,
                        page_number=page.page_number,
                        section_path=[t for _, t in stack],
                        bbox=item.get("bbox"),
                        table_html=table_html,
                        table_rows=rows,
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
                    title=first.section_path[-1] if first.section_path else "Document preamble",
                    text=text,
                    page_start=first.page_number,
                    page_end=pending[-1].page_number,
                    section_path=list(first.section_path),
                    block_ids=[b.block_id for b in pending],
                    asset_ids=list(dict.fromkeys(a for b in pending for a in b.asset_ids)),
                    formula_latex=[b.latex for b in pending if b.latex],
                    table_html=[b.table_html for b in tables if b.table_html],
                    table_ids=[b.table_id for b in tables if b.table_id],
                    table_rows=[row for b in tables for row in b.table_rows],
                    content_type="table" if tables else "text",
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
            elif block.block_type not in {"table", "annotation"} and len(block.text) > 1600:
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
                or block.block_type in {"heading", "table"}
                or sum(len(b.text) + 2 for b in pending) + len(block.text) > 1600
            ):
                flush()
            pending.append(block)
            if block.block_type == "table" or sum(len(b.text) for b in pending) >= 1600:
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
        if warnings:
            gate("remote_warnings", "warn", "; ".join(warnings))
        if fallback_pages:
            gate("structured_content", "warn", f"Text-only fallback pages: {fallback_pages}")
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
            "annotations": annotations,
            "missing_text_pages": no_text,
            "missing_content_pages": empty,
            "remote_warnings": warnings,
            "text_only_fallback_pages": fallback_pages,
        }
