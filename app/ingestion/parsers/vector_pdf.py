from __future__ import annotations

import html
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import pdfplumber
from PIL import Image

from app.ingestion.pipeline import (
    FAKE_GLYPH_TO_ASCII,
    _looks_like_term_heading,
    repair_fake_glyphs,
    table_to_markdown,
)

logging.getLogger("pdfminer").setLevel(logging.WARNING)


TABLE_CAPTION_RE = re.compile(
    r"^(?:表\s*|Table\s+)(?P<reference>(?:[A-Za-z]\s*[.\-]?\s*)?"
    r"(?:\d+(?:[.\-]\d+)*|[一二三四五六七八九十]+))"
    r"(?:\s*[.、:：\-—])?\s*.*$",
    re.IGNORECASE,
)
TABLE_CAPTION_NARRATIVE_RE = re.compile(
    r"^表\s*(?:[A-Za-z]\s*[.\-]?\s*)?(?:\d+(?:[.\-]\d+)*|"
    r"[一二三四五六七八九十]+)\s*为.*(?:内容|规定|如下)[。.]?$",
    re.IGNORECASE,
)
# A figure caption is a "图 N 描述…" label at the start of a line (or preceded
# by a newline). Matching on line position avoids false positives from inline
# references like "结果如图1所示".
FIGURE_CAPTION_RE = re.compile(
    r"(?:^|\n)\s*图\s*[0-9A-Za-z一二三四五六七八九十]+\s*\S"
)
# Unnumbered figure captions ("图 驾驶员操纵方向舵的最大力") carry no number
# after 图 and so never match FIGURE_CAPTION_RE. Require a CJK description after
# "图 " so a clause reference like "图 23.443条" is not captured.
FIGURE_CAPTION_UNNUMBERED_RE = re.compile(
    r"(?:^|\n)\s*图\s+[一-鿿]"
)
# Lines that are body text rather than figure captions: revision dates,
# clause headings, sub-item markers, page footers, and table fragments. The
# _caption_near fallback skips them so a figure's text is never mixed with
# the surrounding paragraph (CCAR-23-R3 p52 mixed "图 驾驶员操纵方向舵的
# 最大力" with the clause body below it).
CAPTION_BODY_LINE_RE = re.compile(
    r"^\s*(?:"
    r"[\[（(]"                      # 修订日期 [..] / 子项标记 (a)
    r"|第\s*\d"                     # 条款标题 第23.443条
    r"|\d+[\.、]"                   # 编号列表项 1. / 1、
    r"|[－\-—–]"                     # 页脚破折号
    r"|CCAR"                        # CCAR 页脚
    r"|同[上下]"                     # 表格碎片 同上/同下
    r"|正常类、实用类、特技类和通勤类飞机适航规定"  # 重复页眉
    r")"
    r"|[，。；：！？、]"               # 正文续行/碎片（含 CJK 句读）
)
# Formula function names that are unambiguous math signals even on a page with
# a single "=" (fraction-style formulas split across lines, e.g. floor/ceil).
FORMULA_FUNCTION_RE = re.compile(
    r"\b(?:floor|ceil|sqrt|log|ln|exp|sin|cos|tan|arcsin|arccos|arctan|"
    r"sinh|cosh|tanh|sum|prod|int|lim|max|min|avg|mean|median)\b",
    re.IGNORECASE,
)

# A standards document cover page: ICS/CCS classification codes, a 发布/实施
# date line, and a standard number (e.g. YD/T××××). Such pages carry
# decorative ruled frames that pdfplumber mistakes for table grids.
_COVER_CLASS_RE = re.compile(r"ICS\s*[\d.]*(?:\s+\d+)*\s*CCS\s*\w+")
_COVER_STD_NUMBER_RE = re.compile(
    r"(?:YD|DB|GB|GA|CY|JR|LD|LY|MH|NY|QX|SB|SL|TB|WB|DL|NB|HG|HJ|CJ|JG|JC|QB|FZ|JT|JTG)\s*\d*/\s*T?\s*[×Xx0-9]"
)
_COVER_DATE_RE = re.compile(r"[发布实施]{2}")


def _looks_like_cover_page(compact_text: str) -> bool:
    if not compact_text or len(compact_text) > 600:
        return False
    has_class = bool(_COVER_CLASS_RE.search(compact_text))
    has_std = bool(_COVER_STD_NUMBER_RE.search(compact_text))
    has_date = bool(_COVER_DATE_RE.search(compact_text))
    return has_class and has_std and has_date


def normalize_cell(value: object) -> str:
    if value is None:
        return ""
    normalized = " ".join(str(value).replace("\x00", "").split())
    # Only touch cells that actually contain damaged-font fake glyphs; leave
    # normal cells (full-width punctuation, math symbols) byte-identical so
    # existing pipeline expectations are unchanged.
    if any(ch in FAKE_GLYPH_TO_ASCII for ch in normalized):
        normalized = repair_fake_glyphs(normalized)
    return re.sub(
        r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])",
        "",
        normalized,
    )


def table_to_html(rows: Sequence[Sequence[object | None]]) -> str:
    raw_rows = [list(row) for row in rows if any(cell is not None for cell in row)]
    if not raw_rows:
        return ""
    width = max(len(row) for row in raw_rows)
    padded = [row + [None] * (width - len(row)) for row in raw_rows]
    body = []
    for row_index, row in enumerate(padded):
        tag = "th" if row_index == 0 else "td"
        cells: list[str] = []
        column = 0
        while column < width:
            cell = row[column]
            if cell is None:
                column += 1
                continue
            colspan = 1
            while column + colspan < width and row[column + colspan] is None:
                colspan += 1
            span = f' colspan="{colspan}"' if colspan > 1 else ""
            cells.append(
                f"<{tag}{span}>{html.escape(normalize_cell(cell))}</{tag}>"
            )
            column += colspan
        body.append(f"<tr>{''.join(cells)}</tr>")
    return "<table>" + "".join(body) + "</table>"


def table_to_text(rows: Sequence[Sequence[object | None]]) -> str:
    normalized = [[normalize_cell(cell) for cell in row] for row in rows]
    normalized = [row for row in normalized if any(row)]
    lines = []
    for row in normalized:
        populated = [cell for cell in row if cell]
        lines.append(populated[0] if len(populated) == 1 else " | ".join(row))
    return "\n".join(lines)


@dataclass(slots=True)
class VectorAsset:
    asset_id: str
    page_number: int
    asset_type: str
    bbox: list[float]
    filename: str
    mime_type: str
    content: bytes
    caption: str = ""
    description: str = ""


@dataclass(slots=True)
class VectorBlock:
    block_type: str
    text: str
    bbox: list[float] | None = None
    table_html: str | None = None
    table_title: str | None = None
    table_rows: list[list[str]] | None = None
    table_merged_cell_count: int = 0
    table_header_only: bool = False
    asset_id: str | None = None
    asset_ids: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass(slots=True)
class VectorPageResult:
    page_number: int
    blocks: list[VectorBlock] = field(default_factory=list)
    assets: list[VectorAsset] = field(default_factory=list)
    table_count: int = 0
    page_type: str = "text"
    image_count: int = 0
    image_coverage: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class VectorPreflightResult:
    table_pages: set[int] = field(default_factory=set)
    table_series: list[list[int]] = field(default_factory=set)
    figure_pages: set[int] = field(default_factory=set)
    formula_pages: set[int] = field(default_factory=set)
    # Native-text pages that carry an embedded *raster* table (the PDF marks it
    # as a structure ``Figure`` content item but its cells live only in pixels,
    # so vector grid detection finds no ruled grid and the table is otherwise
    # dropped). These must be routed to remote OCR to recover the rows; treating
    # them as ordinary figures only keeps a cropped image with no cell values.
    raster_table_pages: set[int] = field(default_factory=set)
    # Per raster-table page, the image bboxes (0..1000 normalized, reading
    # order) classified as tables. The hybrid parser crops each region and
    # re-OCRs it at higher resolution so small tables' digits are not blurred.
    raster_table_boxes: dict[int, list[list[float]]] = field(default_factory=dict)


class VectorPdfParser:
    """Target-page PDF layout specialist.

    This parser intentionally operates only on pages selected by the native
    routing pass. It preserves vector-table row/column structure and renders a
    visual page asset for figure-heavy pages without rerunning the whole PDF.
    """

    name = "vector-pdf"
    version = "1"

    @staticmethod
    def _bbox(value: Any) -> list[float]:
        return [round(float(coordinate), 2) for coordinate in value]

    @staticmethod
    def _looks_like_record_form(
        rows: list[list[str]],
        *,
        title: str,
    ) -> bool:
        """Detect blank record forms (记录表/登记表/申报表).

        A record form is a table whose only populated row is the header; the
        data rows are empty fields waiting to be filled. Their field labels are
        the retrieval payload, and the whole-page image is kept for visual
        verification. We match a title keyword first (strong signal), then fall
        back to "header-only" content (a header row followed by rows that are
        almost entirely empty).
        """
        normalized_title = "".join(title.split())
        if any(
            keyword in normalized_title
            for keyword in ("记录表", "登记表", "申报表", "记录单", "检查表", "调查表")
        ):
            return True
        if len(rows) < 2:
            return False
        populated = sum(
            1
            for row in rows[1:]
            for cell in row
            if str(cell or "").strip()
        )
        data_cells = sum(len(row) for row in rows[1:])
        return data_cells > 0 and populated / data_cells <= 0.02

    @staticmethod
    def _asset_id(page_number: int, kind: str, ordinal: int) -> str:
        return f"p{page_number:04d}-{kind}-{ordinal:02d}"

    @staticmethod
    def _render_page(page: Any, *, resolution: int = 160) -> bytes:
        image = page.to_image(resolution=resolution, antialias=True).original
        stream = BytesIO()
        image.save(stream, format="PNG", optimize=True)
        return stream.getvalue()

    # Figure crops are rendered at a higher resolution than the whole-page
    # asset so small / flat diagrams (e.g. GB/T 19000 concept-relation trees,
    # only ~60-75pt tall) stay readable to the Figure VLM. 160dpi would give a
    # ~143px-tall crop that the VLM misjudges as "no actual graphic content".
    _FIGURE_RENDER_RESOLUTION = 300

    @staticmethod
    def _crop_png(
        page_png: bytes,
        *,
        page_width: float,
        page_height: float,
        bbox: list[float],
        origin: tuple[float, float] = (0.0, 0.0),
    ) -> bytes:
        """Crop ``bbox`` out of a rendered page image.

        ``bbox`` is expressed in the same coordinate system as the PDF page
        objects (pdfplumber reports words/images relative to the page *MediaBox*
        origin, which for some documents is NOT (0,0)). The rendered PNG, on the
        other hand, is normalized to the visible cropbox origin by
        ``Page.to_image()``. When the MediaBox origin is offset, cropping with
        the raw bbox coordinates misplaces the window and clips the figure.

        Pass ``origin`` = the page's visible-origin offset (``cropbox[0]``,
        ``cropbox[1]``) so the bbox is translated into the rendered image's
        coordinate space before scaling. For the common origin-(0,0) case this
        is a no-op.
        """
        origin_x, origin_y = origin
        with Image.open(BytesIO(page_png)) as image:
            scale_x = image.width / page_width
            scale_y = image.height / page_height
            left, top, right, bottom = bbox
            crop = image.crop(
                (
                    max(0, round((left - origin_x) * scale_x)),
                    max(0, round((top - origin_y) * scale_y)),
                    min(image.width, round((right - origin_x) * scale_x)),
                    min(image.height, round((bottom - origin_y) * scale_y)),
                )
            )
            stream = BytesIO()
            crop.save(stream, format="PNG", optimize=True)
            return stream.getvalue()

    @staticmethod
    def _page_origin(page: Any) -> tuple[float, float]:
        """Return the visible-origin offset of ``page``.

        pdfplumber reports object coordinates (images, words, bboxes) relative
        to the page *MediaBox* origin, which is not necessarily (0,0) — some
        documents (e.g. GB/T 19000-2016) shift the MediaBox so the visible page
        starts at a negative x / positive y. ``Page.to_image()`` normalizes the
        rendered PNG to the cropbox origin, so crops must subtract this offset
        to align bbox coordinates with pixels.
        """
        try:
            cropbox = page.cropbox
            return float(cropbox[0]), float(cropbox[1])
        except (AttributeError, TypeError, ValueError):
            return 0.0, 0.0

    @classmethod
    def _figure_crop_png(
        cls,
        page: Any,
        *,
        bbox: list[float],
    ) -> bytes:
        """Render and crop a figure region at higher resolution.

        Small / flat diagrams (concept-relation trees ~60-75pt tall) lose their
        structure at the 160dpi whole-page render, so the Figure VLM misreads
        them ("no actual graphic content"). Render the page at 300dpi for the
        figure crop so the tree connectors stay legible.
        """
        page_png = cls._render_page(page, resolution=cls._FIGURE_RENDER_RESOLUTION)
        return cls._crop_png(
            page_png,
            page_width=float(page.width),
            page_height=float(page.height),
            bbox=bbox,
            origin=cls._page_origin(page),
        )

    @staticmethod
    def _caption_near(page: Any, bbox: list[float]) -> str:
        left, top, right, bottom = bbox
        bands = (
            (
                max(0.0, left - 20),
                bottom,
                min(float(page.width), right + 20),
                min(float(page.height), bottom + 120),
            ),
            (
                max(0.0, left - 20),
                max(0.0, top - 30),
                min(float(page.width), right + 20),
                top,
            ),
        )
        for band in bands:
            if band[2] <= band[0] or band[3] <= band[1]:
                continue
            try:
                text = page.crop(band).extract_text() or ""
            except Exception:
                continue
            if not text.strip():
                continue
            lines = [
                " ".join(raw_line.split())
                for raw_line in text.splitlines()
                if raw_line.strip()
            ]
            # Prefer a real numbered figure caption line ("图2 物流无人机…")
            # inside the crop band over nearby inline symbol explanations
            # ("a 起降过程 b …"). Match at line starts so "结果如图2所示"
            # references are ignored.
            for line in lines:
                if FIGURE_CAPTION_RE.search(line):
                    return line[:240]
            # Unnumbered captions ("图 驾驶员操纵方向舵的最大力") have no
            # number after 图; treat them as real captions too, but require a
            # CJK description so clause references ("图 23.443条") are skipped.
            for line in lines:
                if FIGURE_CAPTION_UNNUMBERED_RE.search(line):
                    return line[:240]
            # Fallback: only a single short descriptive line (a symbol legend,
            # an axis label, an annex title) is a usable caption hint. Multi-line
            # or body-looking content (revision dates, clause headings, sub-item
            # markers, page footers, pure-CJK table-cell fragments) must never be
            # returned, otherwise the figure's text mixes with the surrounding
            # paragraph (CCAR-23-R3 p52 mixed "图 驾驶员操纵方向舵的最大力"
            # with the clause body; p163 table cells fused into the figure text).
            meaningful = [
                line
                for line in lines
                if not CAPTION_BODY_LINE_RE.search(line)
                # A long pure-CJK line with no Latin/digits/punctuation is a
                # table-cell prose fragment ("择大于这些最小值的速度作为"), not
                # a caption. Real symbol legends carry Latin/digit symbols
                # ("a 起降过程 b 航线飞行 …"); keep those.
                and not (
                    len(line) >= 6
                    and not re.search(r"[A-Za-z0-9]", line)
                    and not re.search(r"[：:；;，。]", line)
                )
            ]
            if len(meaningful) == 1 and len(meaningful[0]) <= 80:
                return meaningful[0][:240]
        return ""

    @staticmethod
    def _cluster_positions(
        values: Sequence[float],
        *,
        tolerance: float = 1.5,
    ) -> list[float]:
        groups: list[list[float]] = []
        for value in sorted(values):
            if not groups or value - groups[-1][-1] > tolerance:
                groups.append([value])
            else:
                groups[-1].append(value)
        return [sum(group) / len(group) for group in groups]

    @classmethod
    def _text_column_boundaries(
        cls,
        page: Any,
        *,
        top: float,
        bottom: float,
        left: float,
        right: float,
    ) -> list[float] | None:
        """Infer column boundaries from text x-coordinates when a ruled table
        has no vertical rules (GJB 2489A-2023 appendix D p37: the font/size
        table is delimited only by horizontal rules).

        Words are clustered by their ``x0`` inside the table bbox; each cluster
        center becomes a column boundary. Requires at least 3 columns and a
        reasonable minimum cluster size so a lone stray word does not create a
        phantom column.
        """
        try:
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=False,
            )
        except (AttributeError, TypeError, ValueError):
            return None
        x_positions: list[float] = []
        for word in words:
            x0 = float(word["x0"])
            x1 = float(word["x1"])
            wtop = float(word["top"])
            wbottom = float(word["bottom"])
            if x0 < left or x1 > right or wtop < top or wbottom > bottom:
                continue
            x_positions.append(x0)
        if not x_positions:
            return None
        x_positions.sort()
        clusters: list[list[float]] = []
        for x in x_positions:
            if clusters and x - clusters[-1][-1] <= 12:
                clusters[-1].append(x)
            else:
                clusters.append([x])
        meaningful = [c for c in clusters if len(c) >= 2]
        if len(meaningful) < 3:
            return None
        boundaries = [left]
        for cluster in meaningful:
            boundaries.append(sum(cluster) / len(cluster))
        boundaries.append(right)
        if len(boundaries) > 20:
            return None
        return boundaries

    @classmethod
    def _line_blocks(cls, page: Any) -> list[VectorBlock]:
        """Extract coordinate-bearing text lines without applying table masks."""

        try:
            words = page.extract_words(
                keep_blank_chars=False,
                use_text_flow=False,
            )
        except (AttributeError, TypeError, ValueError):
            return []
        line_groups: list[list[dict[str, Any]]] = []
        for word in sorted(words, key=lambda item: (item["top"], item["x0"])):
            if (
                not line_groups
                or abs(float(word["top"]) - float(line_groups[-1][0]["top"])) > 3
            ):
                line_groups.append([word])
            else:
                line_groups[-1].append(word)

        # Some PDFs set clause numbers with a floating dot line that sits ~7pt
        # below the digits ("515" above ".." is really "5.1.5" — GB 46750-2025).
        # The dots form their own short punctuation-only line right below the
        # digits. Merge them into the previous line so the clause number keeps
        # its dots; the interleaving happens when the block text is built.
        merged_groups: list[list[dict[str, Any]]] = []
        for group in line_groups:
            group_text = "".join(
                str(item.get("text", "")) for item in group
            ).strip()
            group_compact = re.sub(r"\s+", "", group_text)
            # A floating line of dots / commas / colons / digits (<=4 chars).
            cur_is_floating_dots = bool(
                group_compact
                and len(group_compact) <= 4
                and re.fullmatch(r"[.,，。、:：;；\-—–\d]{1,4}", group_compact)
            )
            if merged_groups and cur_is_floating_dots:
                previous = merged_groups[-1]
                gap = float(group[0]["top"]) - float(previous[-1]["top"])
                if 0 <= gap <= 8:
                    previous.extend(group)
                    continue
            merged_groups.append(group)

        blocks: list[VectorBlock] = []
        for group in merged_groups:
            ordered = sorted(group, key=lambda item: float(item["x0"]))
            text = " ".join(
                normalize_cell(item.get("text", "")) for item in ordered
            ).strip()
            # A bare clause number whose dots sank to a separate line produces
            # "515 .. 民用…" (digits then dots at line start) for numeric
            # clauses, or "A3 . 激活…" / "A31 .. 数据…" for letter-prefixed
            # annex clauses. The dots sit on their own word after a space
            # (unlike a normal "6.2" number, whose dot has no space around it).
            # Interleave the dots between the digits:
            #   "515" + ".." -> "5.1.5"
            #   "A3"  + "."  -> "A.3"
            #   "A31" + ".." -> "A.3.1"
            match = re.match(
                r"^([A-Za-z]?)(\d{1,6})\s+([.,，。]{1,4})\s*(.*)$",
                text,
            )
            if match:
                prefix, digits, dots, rest = (
                    match.group(1),
                    match.group(2),
                    match.group(3),
                    match.group(4),
                )
                parts = list(digits)
                # First place dots between adjacent digits (515 + 2 -> 5.1.5).
                for i in range(len(parts) - 1):
                    if i < len(dots):
                        parts[i] += "."
                merged = "".join(parts)
                # Extra dots (more than digit gaps) go between the letter
                # prefix and the first digit (A3 + 1 dot -> A.3).
                leftover = max(0, len(dots) - (len(parts) - 1))
                if leftover and prefix:
                    merged = prefix + "." * leftover + merged
                else:
                    merged = prefix + merged
                text = merged + " " + rest if rest else merged
            text = re.sub(
                r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])",
                "",
                text,
            )
            compact = re.sub(r"\s+", "", text)
            if not compact or re.fullmatch(r"[—\-–]*\d+[—\-–]*", compact):
                continue
            blocks.append(
                VectorBlock(
                    block_type="paragraph",
                    text=text,
                    bbox=cls._bbox(
                        (
                            min(float(item["x0"]) for item in ordered),
                            min(float(item["top"]) for item in ordered),
                            max(float(item["x1"]) for item in ordered),
                            max(float(item["bottom"]) for item in ordered),
                        )
                    ),
                )
            )

        # Terminology entries in standards such as GB/T 19000 lay out the term
        # number ("3.5.10") on its own line, then the term name + English name
        # ("愿景 vision") on the next line. Merge the bare number into the
        # following term heading so split_blocks sees "3.5.10 愿景 vision" as
        # one numbered term heading (and can nest it under "3.5 有关体系的术语").
        # A lenient term-heading test accepts "客体 objectentityitem ; ;" where
        # the English name's semicolons sank to a lower line.
        def _looks_like_term_line(text: str) -> bool:
            if _looks_like_term_heading(text):
                return True
            compact = re.sub(r"\s+", "", text)
            return bool(
                re.search(r"[㐀-鿿]{2,}", compact)
                and re.search(r"[A-Za-z]{3,}", compact)
            )

        merged_blocks: list[VectorBlock] = []
        for block in blocks:
            if (
                merged_blocks
                and re.fullmatch(r"\d+(?:\.\d+)+", block.text.strip())
                and _looks_like_term_line(merged_blocks[-1].text)
            ):
                # The term heading came before the number (reading order by
                # bbox top can place the number's bbox below the heading when
                # the number is indented); merge the number to the front.
                merged_blocks[-1].text = (
                    block.text.strip() + " " + merged_blocks[-1].text
                )
                continue
            if (
                merged_blocks
                and re.fullmatch(r"\d+(?:\.\d+)+", merged_blocks[-1].text.strip())
                and _looks_like_term_line(block.text)
            ):
                # Number first, then term heading: "3.5.10" + "愿景 vision".
                merged_blocks[-1].text = (
                    merged_blocks[-1].text.strip() + " " + block.text
                )
                continue
            merged_blocks.append(block)
        return merged_blocks

    @classmethod
    def _table_caption_blocks(cls, page: Any) -> list[VectorBlock]:
        """Return real table captions, excluding prose such as ``表A.1为…内容``."""

        captions = [
            block
            for block in cls._line_blocks(page)
            if TABLE_CAPTION_RE.match(block.text)
            and not TABLE_CAPTION_NARRATIVE_RE.match(block.text)
        ]
        # Some forms repeat the same reference in an introductory sentence and
        # in the real centered caption. Keep the lower/nearest caption.
        deduplicated: list[VectorBlock] = []
        for caption in captions:
            match = TABLE_CAPTION_RE.match(caption.text)
            reference = (
                re.sub(r"\s+", "", match.group("reference")).casefold()
                if match
                else caption.text.casefold()
            )
            replacement = next(
                (
                    index
                    for index, old in enumerate(deduplicated)
                    if (
                        (old_match := TABLE_CAPTION_RE.match(old.text))
                        and re.sub(r"\s+", "", old_match.group("reference")).casefold()
                        == reference
                        and old.bbox
                        and caption.bbox
                        and abs(caption.bbox[1] - old.bbox[1]) <= 40
                    )
                ),
                None,
            )
            if replacement is None:
                deduplicated.append(caption)
            else:
                deduplicated[replacement] = caption
        return sorted(
            deduplicated,
            key=lambda block: (
                block.bbox[1] if block.bbox else 0.0,
                block.bbox[0] if block.bbox else 0.0,
            ),
        )

    @staticmethod
    def _long_horizontal_positions(page: Any) -> list[float]:
        segments: list[tuple[float, float, float]] = []
        page_width = float(page.width)
        try:
            edges = page.edges
        except (AttributeError, TypeError, ValueError):
            return []
        for edge in edges:
            x0 = float(edge["x0"])
            x1 = float(edge["x1"])
            top = float(edge["top"])
            bottom = float(edge["bottom"])
            if abs(top - bottom) <= 1.0 and x1 - x0 >= 20:
                segments.append(((top + bottom) / 2, x0, x1))
        merged: list[tuple[float, float, float]] = []
        for top, x0, x1 in sorted(segments):
            if merged and top - merged[-1][0] <= 1.5:
                old_top, old_x0, old_x1 = merged[-1]
                merged[-1] = (
                    (old_top + top) / 2,
                    min(old_x0, x0),
                    max(old_x1, x1),
                )
            else:
                merged.append((top, x0, x1))
        return [top for top, x0, x1 in merged if x1 - x0 >= page_width * 0.25]

    @classmethod
    def _extract_captioned_tables(
        cls,
        page: Any,
    ) -> list[tuple[list[float], list[list[object | None]], str]]:
        """Bind captions by geometry and split grids that contain two tables.

        PDF table detectors often merge nearby ruled tables and do not include
        captions in the returned grid.  Using the first page-level caption for
        every table corrupts table identity.  Caption coordinates and long
        horizontal rules provide deterministic boundaries for re-cropping.
        """

        base_tables = sorted(cls._extract_best_tables(page), key=lambda item: item[0][1])
        captions = cls._table_caption_blocks(page)
        horizontal = cls._long_horizontal_positions(page)
        results: list[tuple[list[float], list[list[object | None]], str]] = []
        used_captions: set[int] = set()

        for bbox, rows in base_tables:
            left, top, right, bottom = bbox
            relevant: list[tuple[int, VectorBlock]] = []
            for index, caption in enumerate(captions):
                if index in used_captions or not caption.bbox:
                    continue
                caption_left, caption_top, caption_right, _ = caption.bbox
                overlaps_x = min(right, caption_right) - max(left, caption_left) > 0
                if overlaps_x and top - 80 <= caption_top <= bottom:
                    relevant.append((index, caption))

            if not relevant:
                results.append((bbox, rows, ""))
                continue

            relevant.sort(key=lambda item: item[1].bbox[1] if item[1].bbox else 0.0)
            if (
                len(relevant) == 1
                and relevant[0][1].bbox is not None
                and relevant[0][1].bbox[3] <= top + 1
            ):
                used_captions.add(relevant[0][0])
                results.append((bbox, rows, relevant[0][1].text))
                continue
            recovered = 0
            for position, (caption_index, caption) in enumerate(relevant):
                assert caption.bbox is not None
                next_top = (
                    relevant[position + 1][1].bbox[1]
                    if position + 1 < len(relevant)
                    and relevant[position + 1][1].bbox is not None
                    else None
                )
                start_candidates = [
                    value
                    for value in horizontal
                    if max(top - 1.5, caption.bbox[3] - 1) <= value <= bottom
                ]
                segment_top = start_candidates[0] if start_candidates else top
                if next_top is None:
                    segment_bottom = bottom
                else:
                    end_candidates = [
                        value
                        for value in horizontal
                        if segment_top < value < next_top
                    ]
                    segment_bottom = end_candidates[-1] if end_candidates else next_top
                if segment_bottom - segment_top < 18:
                    continue
                crop_bbox = (
                    left,
                    max(top, segment_top - 0.75),
                    right,
                    min(bottom, segment_bottom + 0.75),
                )
                try:
                    candidates = cls._extract_best_tables(page.crop(crop_bbox))
                except (AttributeError, TypeError, ValueError):
                    candidates = []
                if not candidates:
                    continue
                candidate = max(candidates, key=lambda item: cls._table_quality([item]))
                results.append((candidate[0], candidate[1], caption.text))
                used_captions.add(caption_index)
                recovered += 1

            if not recovered:
                title = relevant[0][1].text
                used_captions.add(relevant[0][0])
                results.append((bbox, rows, title))

        ordered = sorted(results, key=lambda item: (item[0][1], item[0][0]))
        if ordered:
            last_bbox, last_rows, last_title = ordered[-1]
            extended_bbox, extended_rows = cls._extend_open_table_tail(
                page,
                last_bbox,
                last_rows,
            )
            ordered[-1] = (extended_bbox, extended_rows, last_title)
        return ordered

    @classmethod
    def _expanded_ruled_tables(
        cls,
        page: Any,
    ) -> list[tuple[list[float], list[list[object | None]]]]:
        """Recover ruled tables whose merged cells hide their outer columns.

        pdfplumber's default detector can start at the first continuous
        internal vertical rule. Regulatory tables often omit vertical rules
        across merged cells, which silently drops row labels and rightmost
        columns. Long horizontal rules still reveal the complete table bounds,
        so rebuild an explicit grid from those rules and every internal
        vertical segment.
        """

        page_width = float(page.width)
        page_height = float(page.height)
        horizontal: list[tuple[float, float, float]] = []
        for edge in page.edges:
            x0 = float(edge["x0"])
            x1 = float(edge["x1"])
            top = float(edge["top"])
            bottom = float(edge["bottom"])
            if abs(top - bottom) > 1.0 or x1 - x0 < 20:
                continue
            if top < page_height * 0.1 or top > page_height * 0.9:
                continue
            horizontal.append((top, x0, x1))
        if len(horizontal) < 4:
            return []

        merged_horizontal: list[tuple[float, float, float]] = []
        for top, x0, x1 in sorted(horizontal):
            if merged_horizontal and top - merged_horizontal[-1][0] <= 1.5:
                previous_top, previous_x0, previous_x1 = merged_horizontal[-1]
                merged_horizontal[-1] = (
                    (previous_top + top) / 2,
                    min(previous_x0, x0),
                    max(previous_x1, x1),
                )
            else:
                merged_horizontal.append((top, x0, x1))
        merged_horizontal = [
            item
            for item in merged_horizontal
            if item[2] - item[1] >= page_width * 0.25
        ]

        horizontal_groups: list[list[tuple[float, float, float]]] = []
        for item in merged_horizontal:
            if (
                not horizontal_groups
                or item[0] - horizontal_groups[-1][-1][0] > 90
            ):
                horizontal_groups.append([item])
            else:
                horizontal_groups[-1].append(item)

        tables: list[tuple[list[float], list[list[object | None]]]] = []
        for group in horizontal_groups:
            if len(group) < 4 or group[-1][0] - group[0][0] < 30:
                continue
            top = group[0][0]
            bottom = group[-1][0]
            left = min(item[1] for item in group)
            right = max(item[2] for item in group)
            vertical_positions = []
            for edge in page.edges:
                x0 = float(edge["x0"])
                x1 = float(edge["x1"])
                edge_top = float(edge["top"])
                edge_bottom = float(edge["bottom"])
                if abs(x1 - x0) > 1.0 or edge_bottom - edge_top < 10:
                    continue
                if edge_bottom < top or edge_top > bottom:
                    continue
                x = (x0 + x1) / 2
                if left + 4 < x < right - 4:
                    vertical_positions.append(x)
            if vertical_positions:
                boundaries = [
                    left,
                    *cls._cluster_positions(vertical_positions),
                    right,
                ]
            else:
                # No vertical rules: the table is delimited only by horizontal
                # rules (GJB 2489A-2023 appendix D). Recover column boundaries
                # from text x-coordinates so the grid is still structured.
                boundaries = cls._text_column_boundaries(
                    page,
                    top=top,
                    bottom=bottom,
                    left=left,
                    right=right,
                ) or [left, right]
            if len(boundaries) < 3 or len(boundaries) > 20:
                continue
            settings = {
                "vertical_strategy": "explicit",
                "explicit_vertical_lines": boundaries,
                "horizontal_strategy": "explicit",
                "explicit_horizontal_lines": [item[0] for item in group],
            }
            try:
                rows = page.extract_table(settings) or []
            except Exception:
                continue
            if len(rows) < 2 or sum(bool(normalize_cell(cell)) for row in rows for cell in row) < 4:
                continue
            tables.append(
                (
                    cls._bbox((left, top, right, bottom)),
                    rows,
                )
            )
        return tables

    @staticmethod
    def _table_quality(
        tables: Sequence[tuple[list[float], Sequence[Sequence[object | None]]]],
    ) -> int:
        """Score content without rewarding fragments as separate tables.

        A tall merged row can create a large gap between horizontal rules.
        The explicit-grid fallback may then return two fragments around that
        row. Applying the column-width bonus to every fragment made those
        incomplete fragments outrank one complete detected table.
        """

        max_width = max(
            (
                max((len(row) for row in rows), default=0)
                for _, rows in tables
            ),
            default=0,
        )
        populated = sum(
            bool(normalize_cell(cell))
            for _, rows in tables
            for row in rows
            for cell in row
        )
        return max_width * 10 + populated

    @staticmethod
    def _valid_table(
        table: tuple[list[float], Sequence[Sequence[object | None]]],
    ) -> bool:
        bbox, rows = table
        width = max((len(row) for row in rows), default=0)
        populated = sum(
            bool(normalize_cell(cell))
            for row in rows
            for cell in row
        )
        table_width = max(0.0, bbox[2] - bbox[0])
        table_height = max(0.0, bbox[3] - bbox[1])
        if (
            len(rows) >= 2
            and width >= 2
            and populated >= max(4, width)
            and table_width >= 40
            and table_height >= 18
        ):
            return True
        # A ruled regulation table can be intentionally single-column: a
        # centred heading followed by a wide, structured amendment entry.  PDF
        # extractors represent its internal dotted leader / tab stop as text,
        # not as a vertical rule, so requiring two extracted columns drops the
        # entire boxed table back into prose.  Accept only a substantial wide
        # frame with multiple populated rows and a non-trivial body cell. This
        # still excludes narrow decoration and the short single-column appendix
        # headings that are not tables.
        if (
            len(rows) >= 2
            and width == 1
            and populated >= 2
            and table_width >= 300
            and table_height >= 80
            and max((len(normalize_cell(cell)) for row in rows for cell in row), default=0)
            >= 24
        ):
            return True
        # A single-row table header with many columns is a real table too
        # (e.g. the "数据包格式" schematic in GB 46750-2025: one header row of
        # 数据类型/版本号/数据长度/…/数据内容项N). Accept it only when it is
        # wide enough and every column is populated, so decorative cover frames
        # and lone single-column appendix titles are still rejected.
        return bool(
            len(rows) == 1
            and width >= 4
            and populated >= width
            and table_width >= 200
            and table_height >= 18
        )

    @staticmethod
    def _captioned_table_count(
        tables: Sequence[tuple[list[float], Sequence[Sequence[object | None]]]],
    ) -> int:
        count = 0
        for _, rows in tables:
            if not rows:
                continue
            first_row = " ".join(
                normalize_cell(cell)
                for cell in rows[0]
                if normalize_cell(cell)
            )
            if re.match(
                r"^表\s*(?:[A-Za-z]\s*[.\-]?\s*)?(?:\d+(?:[.\-]\d+)*|"
                r"[一二三四五六七八九十]+)",
                first_row,
                re.IGNORECASE,
            ):
                count += 1
        return count

    @classmethod
    def _choose_best_tables(
        cls,
        detected: list[tuple[list[float], list[list[object | None]]]],
        expanded: list[tuple[list[float], list[list[object | None]]]],
    ) -> list[tuple[list[float], list[list[object | None]]]]:
        """Keep explicitly captioned tables separate before using grid score.

        The expanded-grid fallback is intentionally generous around broken
        rules. On pages containing several nearby tables, that generosity can
        join ``表 1`` and ``表 2`` into one oversized grid. The default detector
        already separates those tables and retains their title rows, which are
        stronger evidence than a small populated-cell score advantage.
        """

        detected_captions = cls._captioned_table_count(detected)
        expanded_captions = cls._captioned_table_count(expanded)
        if detected_captions > expanded_captions:
            return detected
        if len(detected) > len(expanded):
            return detected
        detected_width = max(
            (max((len(row) for row in rows), default=0) for _, rows in detected),
            default=0,
        )
        expanded_width = max(
            (max((len(row) for row in rows), default=0) for _, rows in expanded),
            default=0,
        )
        detected_merged = sum(
            cell is None for _, rows in detected for row in rows for cell in row
        )
        expanded_merged = sum(
            cell is None for _, rows in expanded for row in rows for cell in row
        )
        if (
            len(detected) == len(expanded)
            and detected_width == expanded_width
            and detected_merged > expanded_merged
        ):
            return detected
        return (
            expanded
            if cls._table_quality(expanded) > cls._table_quality(detected)
            else detected
        )

    @classmethod
    def _is_split_header(cls, page: Any, table: Any) -> bool:
        """Accept a short ruled header only with a matching next-page body grid.

        Check actual cell boundaries, not a filename or column count alone.
        Standalone one-row boxes must remain excluded.
        """
        try:
            rows = table.extract() or []
            cells = table.rows[0].cells
            if (
                len(rows) != 1 or len(rows[0]) < 2
                or any(not normalize_cell(c) for c in rows[0])
                or any(not re.search(r"[A-Za-z\u3400-\u9fff]", normalize_cell(c))
                       or re.match(r"^[\d$€£¥±+\-]", normalize_cell(c)) for c in rows[0])
                or float(table.bbox[1]) < float(page.height) * 0.8
                or page.page_number >= len(page.pdf.pages)
                or any(cell is None for cell in cells)
            ):
                return False
            following = page.pdf.pages[page.page_number]
            for candidate in following.find_tables():
                body = candidate.extract() or []
                if (not cls._valid_table((cls._bbox(candidate.bbox), body))
                        or float(candidate.bbox[1]) > float(following.height) * 0.15
                        or len(body[0]) != len(rows[0])
                        or not all(re.search(r"\d", normalize_cell(c)) for c in body[0])):
                    continue
                next_cells = candidate.rows[0].cells
                if len(next_cells) != len(cells) or any(c is None for c in next_cells):
                    continue
                if all(
                    abs(float(a[i]) / page.width - float(b[i]) / following.width) <= 0.01
                    for a, b in zip(cells, next_cells, strict=True) for i in (0, 2)
                ):
                    return True
        except (AttributeError, IndexError, TypeError, ValueError):
            return False
        return False

    @staticmethod
    def _cluster_positions(values: Sequence[float], *, tolerance: float = 3.0) -> list[float]:
        """Collapse duplicate PDF rule coordinates into one grid position."""

        clusters: list[list[float]] = []
        for value in sorted(float(item) for item in values):
            if clusters and value - clusters[-1][-1] <= tolerance:
                clusters[-1].append(value)
            else:
                clusters.append([value])
        return [sum(cluster) / len(cluster) for cluster in clusters]

    @classmethod
    def _grid_columns(
        cls,
        page: Any,
        bbox: Sequence[float],
        width: int,
    ) -> list[float]:
        """Return the vertical grid boundaries for a ruled table."""

        left, _top, right, _bottom = (float(value) for value in bbox)
        positions: list[float] = []
        for edge in getattr(page, "edges", []) or []:
            try:
                x0, x1 = float(edge["x0"]), float(edge["x1"])
                edge_top, edge_bottom = float(edge["top"]), float(edge["bottom"])
            except (KeyError, TypeError, ValueError):
                continue
            if abs(x1 - x0) > 1.5 or edge_bottom - edge_top < 20:
                continue
            if left - 5 <= x0 <= right + 5:
                positions.append(x0)
        clustered = cls._cluster_positions(positions)
        if len(clustered) >= width + 1:
            # Prefer the outermost set that spans the detected table width. A
            # page may carry an unrelated narrow ruled box beside the table.
            in_range = [value for value in clustered if left - 5 <= value <= right + 5]
            if len(in_range) >= width + 1:
                return in_range[: width + 1]
        step = (right - left) / max(1, width)
        return [left + step * index for index in range(width + 1)]

    @classmethod
    def _open_table_tail(
        cls,
        page: Any,
        bbox: Sequence[float],
        width: int,
    ) -> tuple[list[float], list[float]] | None:
        """Find a table row whose bottom border continues past this page.

        A PDF table can be cut at a page boundary without a complete rectangle
        on either page. In that case pdfplumber returns the completed rows only
        and leaves the final row as ordinary text. The continued vertical rules
        are stronger evidence than text alone, so use them to recover that
        open row conservatively.
        """

        if width < 2:
            return None
        page_height = float(getattr(page, "height", 0.0) or 0.0)
        if page_height <= 0:
            return None
        left, _top, right, bottom = (float(value) for value in bbox)
        columns = cls._grid_columns(page, bbox, width)
        if len(columns) != width + 1:
            return None
        continuations: list[tuple[float, float]] = []
        for column in columns:
            segments: list[tuple[float, float]] = []
            for edge in getattr(page, "edges", []) or []:
                try:
                    x0, x1 = float(edge["x0"]), float(edge["x1"])
                    edge_top, edge_bottom = float(edge["top"]), float(edge["bottom"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    abs(x1 - x0) <= 1.5
                    and abs(x0 - column) <= 3.0
                    and bottom - 4 <= edge_top <= bottom + 4
                    and edge_bottom > bottom + 40
                ):
                    segments.append((edge_top, edge_bottom))
            if not segments:
                continue
            continuations.append(max(segments, key=lambda value: value[1]))
        if len(continuations) < width + 1:
            return None
        tail_top = min(value[0] for value in continuations)
        tail_bottom = max(value[1] for value in continuations)
        # Do not consume an ordinary paragraph block halfway down a page. An
        # open row should reach the footer margin, as the source grid does.
        if tail_bottom < page_height * 0.85:
            return None
        if tail_top < bottom - 5:
            return None
        return [left, tail_top, right, tail_bottom], columns

    @classmethod
    def _continuation_grid(
        cls,
        page: Any,
        columns: Sequence[float],
    ) -> list[float] | None:
        """Find a top-of-page grid fragment matching ``columns``."""

        if len(columns) < 3:
            return None
        left, right = float(columns[0]), float(columns[-1])
        page_height = float(getattr(page, "height", 0.0) or 0.0)
        if page_height <= 0:
            return None
        matched = 0
        vertical_bottom = 0.0
        for column in columns:
            segments: list[float] = []
            for edge in getattr(page, "edges", []) or []:
                try:
                    x0, x1 = float(edge["x0"]), float(edge["x1"])
                    edge_top, edge_bottom = float(edge["top"]), float(edge["bottom"])
                except (KeyError, TypeError, ValueError):
                    continue
                if (
                    abs(x1 - x0) <= 1.5
                    and abs(x0 - column) <= 3.0
                    and edge_top <= 55
                    and edge_bottom - edge_top >= 35
                ):
                    segments.append(edge_bottom)
            if segments:
                matched += 1
                vertical_bottom = max(vertical_bottom, max(segments))
        if matched != len(columns):
            return None
        horizontal: list[float] = []
        segmented_horizontal: list[tuple[float, float, float]] = []
        for edge in getattr(page, "edges", []) or []:
            try:
                x0, x1 = float(edge["x0"]), float(edge["x1"])
                edge_top, edge_bottom = float(edge["top"]), float(edge["bottom"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                abs(edge_bottom - edge_top) <= 1.5
                and edge_top >= 20
                and edge_top <= page_height * 0.65
            ):
                y = (edge_top + edge_bottom) / 2
                if (
                    x0 <= left + 3
                    and x1 >= right - 3
                    and x1 - x0 >= (right - left) * 0.8
                ):
                    horizontal.append(y)
                # Some PDF generators draw a row boundary as one segment per
                # cell rather than a single full-width rule. Keep those
                # segments so a continuation crop stops at the first row
                # boundary instead of swallowing the next row.
                if x1 > left and x0 < right and x1 - x0 >= 8:
                    segmented_horizontal.append(
                        (y, max(left, x0), min(right, x1))
                    )
        if segmented_horizontal:
            by_y: list[list[tuple[float, float, float]]] = []
            for segment in sorted(segmented_horizontal):
                if not by_y or segment[0] - by_y[-1][-1][0] > 1.5:
                    by_y.append([segment])
                else:
                    by_y[-1].append(segment)
            for group in by_y:
                coverage = 0.0
                cursor: float | None = None
                for start, end in sorted((item[1], item[2]) for item in group):
                    if cursor is None:
                        cursor = end
                        coverage = end - start
                    elif start >= cursor:
                        coverage += end - start
                        cursor = end
                    elif end > cursor:
                        coverage += end - cursor
                        cursor = end
                if coverage >= (right - left) * 0.85:
                    horizontal.append(sum(item[0] for item in group) / len(group))
        if not horizontal:
            return None
        top_candidates: list[float] = []
        for edge in getattr(page, "edges", []) or []:
            try:
                x0, x1 = float(edge["x0"]), float(edge["x1"])
                edge_top = float(edge["top"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                abs(x1 - x0) <= 1.5
                and abs(x0 - left) <= 3.0
                and edge_top <= 55
            ):
                top_candidates.append(edge_top)
        if not top_candidates:
            return None
        top = min(top_candidates)
        # The first horizontal rule after the top boundary closes the
        # continuation row. If no internal rule is available, retain the old
        # behavior and use the last full-width rule.
        row_boundaries = sorted(value for value in horizontal if value > top + 5)
        bottom = row_boundaries[0] if row_boundaries else max(horizontal)
        bottom = min(bottom, vertical_bottom)
        if bottom <= 30:
            return None
        return [left, min(55.0, max(0.0, top)), right, bottom]

    @classmethod
    def _grid_row_text(
        cls,
        page: Any,
        bbox: Sequence[float],
        columns: Sequence[float],
    ) -> list[str]:
        """Extract one row by cell boundaries, preserving wrapped text."""

        top, bottom = float(bbox[1]), float(bbox[3])
        cells: list[str] = []
        for left, right in zip(columns, columns[1:], strict=False):
            text = cls._smart_cell_text(
                page,
                [float(left) + 0.5, top + 0.5, float(right) - 0.5, bottom - 0.5],
            )
            cells.append(normalize_cell(text))
        return cells

    @classmethod
    def _extend_open_table_tail(
        cls,
        page: Any,
        bbox: Sequence[float],
        rows: list[list[object | None]],
    ) -> tuple[list[float], list[list[object | None]]]:
        """Append the unfinished bottom row when the next page continues it."""

        if not rows:
            return list(bbox), rows
        width = max((len(row) for row in rows), default=0)
        tail = cls._open_table_tail(page, bbox, width)
        if tail is None:
            return list(bbox), rows
        tail_bbox, columns = tail
        next_page_number = int(getattr(page, "page_number", 0) or 0) + 1
        pages = getattr(getattr(page, "pdf", None), "pages", []) or []
        if next_page_number < 1 or next_page_number > len(pages):
            return list(bbox), rows
        next_page = pages[next_page_number - 1]
        if cls._continuation_grid(next_page, columns) is None:
            return list(bbox), rows
        row = cls._grid_row_text(page, tail_bbox, columns)
        if not any(cell.strip() for cell in row):
            return list(bbox), rows
        extended_bbox = [
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(tail_bbox[3]),
        ]
        return extended_bbox, [*rows, row]

    @classmethod
    def _extract_open_table_continuation(
        cls,
        page: Any,
    ) -> tuple[list[float], list[list[object | None]], str] | None:
        """Extract a continuation row when the page has no closed table grid."""

        page_number = int(getattr(page, "page_number", 0) or 0)
        pages = getattr(getattr(page, "pdf", None), "pages", []) or []
        if page_number <= 1 or page_number > len(pages):
            return None
        previous = pages[page_number - 2]
        previous_tables = cls._extract_best_tables(previous)
        if not previous_tables:
            return None
        previous_bbox, previous_rows = max(
            previous_tables,
            key=lambda item: item[0][3],
        )
        width = max((len(row) for row in previous_rows), default=0)
        tail = cls._open_table_tail(previous, previous_bbox, width)
        if tail is None:
            return None
        _tail_bbox, columns = tail
        continuation_bbox = cls._continuation_grid(page, columns)
        if continuation_bbox is None:
            return None
        row = cls._grid_row_text(page, continuation_bbox, columns)
        if not any(cell.strip() for cell in row):
            return None
        return continuation_bbox, [row], ""

    @classmethod
    def _extract_following_table_fragments(
        cls,
        page: Any,
        continuation_bbox: Sequence[float],
        width: int,
    ) -> list[tuple[list[float], list[list[object | None]], str]]:
        """Recover ordinary rows drawn below an open continuation row.

        A page can start with the tail of a row from the previous page and
        immediately follow it with a complete one-row table fragment. The
        normal table validator intentionally rejects isolated one-row grids;
        here the continuation geometry provides the missing context, so only
        same-width rows below that geometry are accepted.
        """

        if width < 2:
            return []
        results: list[tuple[list[float], list[list[object | None]], str]] = []
        try:
            tables = page.find_tables()
        except (AttributeError, TypeError, ValueError):
            return results
        continuation_bottom = float(continuation_bbox[3])
        for table in tables:
            try:
                bbox = cls._bbox(table.bbox)
                rows = cls._rebuild_cells(page, table)
            except (AttributeError, TypeError, ValueError):
                continue
            if bbox[1] < continuation_bottom - 3 or len(rows) != 1:
                continue
            row = rows[0]
            if len(row) != width or sum(bool(normalize_cell(cell)) for cell in row) < 2:
                continue
            if bbox[3] - bbox[1] < 12 or bbox[2] - bbox[0] < 40:
                continue
            results.append((bbox, rows, ""))
        return sorted(results, key=lambda item: (item[0][1], item[0][0]))

    @classmethod
    def _extract_best_tables(
        cls,
        page: Any,
    ) -> list[tuple[list[float], list[list[object | None]]]]:
        detected: list[tuple[list[float], list[list[object | None]]]] = []
        try:
            for table in page.find_tables():
                candidate = (cls._bbox(table.bbox), cls._rebuild_cells(page, table))
                if cls._valid_table(candidate) or cls._is_split_header(page, table):
                    detected.append(candidate)
        except (AttributeError, TypeError, ValueError):
            detected = []

        try:
            expanded = cls._expanded_ruled_tables(page)
        except (AttributeError, TypeError, ValueError):
            expanded = []
        expanded = [table for table in expanded if cls._valid_table(table)]
        return cls._choose_best_tables(detected, expanded)

    @staticmethod
    def _smart_cell_text(page: Any, bbox: list[float]) -> str:
        """Rebuild one table cell from character coordinates.

        Some PDFs typeset cells with wide inter-character spacing so that
        pdfplumber's default ``y_tolerance=3`` splits one visual line into
        several (punctuation/units/number chars sit ~6pt lower and get moved
        to the next line). That corrupts the order, e.g. "起飞方式 | … |
        滑跑/弹射/手抛/垂直/其他" becomes "滑跑 弹射 … 其他 : / / / /".

        This rebuild groups chars by ``top``, then merges candidate lines whose
        gap is small and whose next line looks like a floating subscript
        (punctuation / number / unit). Chars are assigned to a cell by their
        center point so neighbouring cells do not leak in.
        """
        if bbox is None or len(bbox) < 4:
            return ""
        left, top, right, bottom = bbox
        chars: list[Any] = []
        for ch in page.chars:
            cx = (float(ch["x0"]) + float(ch["x1"])) / 2
            cy = (float(ch["top"]) + float(ch["bottom"])) / 2
            if left <= cx <= right and top <= cy <= bottom:
                # Browser PDFs sometimes draw the equality stroke of <= / >=
                # separately from the text glyph. Preserve that source mark,
                # but never interpret long underlines or cell borders as math.
                value = str(ch["text"])
                if value in {"<", ">"}:
                    size = float(ch.get("size", 12))
                    width = float(ch["x1"]) - float(ch["x0"])
                    for mark in getattr(page, "rects", []) or []:
                        if (
                            0 < float(mark["bottom"]) - float(mark["top"]) <= size * 0.12
                            and 0 <= float(mark["top"]) - float(ch["bottom"]) <= size * 0.15
                            and width * 0.8 <= float(mark["x1"]) - float(mark["x0"]) <= width * 2
                            and abs(float(mark["x0"]) - float(ch["x0"])) <= width * 0.5
                            and float(mark["x1"]) >= float(ch["x1"]) - width * 0.1
                        ):
                            ch = {**ch, "text": "≤" if value == "<" else "≥"}
                            break
                chars.append(ch)
        if not chars:
            return ""
        chars.sort(key=lambda ch: (float(ch["top"]), float(ch["x0"])))
        candidate_lines: list[list[Any]] = []
        current: list[Any] = []
        last_top: float | None = None
        for ch in chars:
            ch_top = float(ch["top"])
            if last_top is None or abs(ch_top - last_top) > 3:
                if current:
                    candidate_lines.append(current)
                current = [ch]
            else:
                current.append(ch)
            last_top = ch_top
        if current:
            candidate_lines.append(current)

        merged: list[list[Any]] = []
        for line in candidate_lines:
            if merged:
                previous = merged[-1]
                gap = float(line[0]["top"]) - float(previous[-1]["top"])
                prev_text = "".join(str(ch["text"]) for ch in previous)
                cur_text = "".join(str(ch["text"]) for ch in line)
                cur_is_floating = len(cur_text) <= 4 or not any(
                    "一" <= str(ch["text"]) <= "鿿" for ch in line
                )
                prev_ends_punct = bool(prev_text and prev_text[-1] in "，,：:")
                if gap <= 7 and (cur_is_floating or prev_ends_punct):
                    previous.extend(line)
                    continue
            merged.append(line)
        result: list[str] = []
        for line in merged:
            line.sort(key=lambda ch: float(ch["x0"]))
            result.append("".join(str(ch["text"]) for ch in line))
        return "\n".join(result)

    @classmethod
    def _rebuild_cells(
        cls,
        page: Any,
        table: Any,
    ) -> list[list[object | None]]:
        """Re-extract table cells, rebuilding text where pdfplumber's default
        line grouping corrupts the order. Merged cells (``None`` or a bbox that
        spans many rows) keep pdfplumber's original value so their text is not
        re-assembled from neighbouring rows.
        """
        try:
            rows = cast(list[list[object | None]], table.extract() or [])
        except (AttributeError, TypeError, ValueError):
            rows = []
        if not rows:
            return rows
        try:
            table_rows = table.rows
        except (AttributeError, TypeError, ValueError):
            return rows
        rebuilt: list[list[object | None]] = []
        for row_index, row in enumerate(rows):
            new_row: list[object | None] = []
            for col_index, cell in enumerate(row):
                if cell is None:
                    new_row.append(None)
                    continue
                if row_index >= len(table_rows):
                    new_row.append(cell)
                    continue
                try:
                    cells = table_rows[row_index].cells
                except (AttributeError, TypeError, ValueError):
                    new_row.append(cell)
                    continue
                if col_index >= len(cells) or cells[col_index] is None:
                    new_row.append(cell)
                    continue
                bbox = cells[col_index]
                if len(bbox) < 4:
                    new_row.append(cell)
                    continue
                height = float(bbox[3]) - float(bbox[1])
                # Tall bbox -> a vertically merged cell; keep pdfplumber's text
                # so we do not pull in characters from the following rows.
                if height > 60:
                    new_row.append(cell)
                else:
                    new_row.append(cls._smart_cell_text(page, bbox))
            rebuilt.append(new_row)
        return rebuilt

    @staticmethod
    def _has_math_font(page: Any) -> bool:
        """Detect pages typeset with math fonts whose sub/superscripts are
        invisible to plain text extraction."""
        try:
            fontnames = {char.get("fontname", "") for char in page.chars}
        except (AttributeError, TypeError, ValueError):
            return False
        if not fontnames:
            return False
        math_hints = ("cambriamath", "cambria math", "math", "stix", "xits")
        for fontname in fontnames:
            lowered = fontname.casefold()
            if any(hint in lowered for hint in math_hints):
                return True
        return False

    @staticmethod
    def contiguous_series(page_numbers: set[int]) -> list[list[int]]:
        series: list[list[int]] = []
        for page_number in sorted(page_numbers):
            if not series or page_number != series[-1][-1] + 1:
                series.append([page_number])
            else:
                series[-1].append(page_number)
        return series

    @classmethod
    def _outside_table_blocks(
        cls,
        page: Any,
        table_bboxes: Sequence[list[float]],
    ) -> list[VectorBlock]:
        """Preserve captions/body text outside table bounds without duplication."""
        blocks: list[VectorBlock] = []
        for block in cls._line_blocks(page):
            if not block.bbox:
                continue
            center_x = (block.bbox[0] + block.bbox[2]) / 2
            center_y = (block.bbox[1] + block.bbox[3]) / 2
            if any(
                left - 1 <= center_x <= right + 1
                and table_top - 1 <= center_y <= table_bottom + 1
                for left, table_top, right, table_bottom in table_bboxes
            ):
                continue
            blocks.append(block)
        return blocks

    @staticmethod
    def _visual_coverage(page: Any) -> tuple[int, float, float]:
        page_area = max(1.0, float(page.width) * float(page.height))
        coverages = [
            max(0.0, float(image["x1"]) - float(image["x0"]))
            * max(0.0, float(image["bottom"]) - float(image["top"]))
            / page_area
            for image in page.images
        ]
        return (
            len(coverages),
            min(1.0, sum(coverages)),
            max(coverages, default=0.0),
        )

    @staticmethod
    def _is_fullpage_background_image(page: Any, image: dict[str, Any]) -> bool:
        """Detect a full-page decorative background image that is not content.

        Some standards PDFs place a white / near-blank JPEG over the whole page
        as a background texture (GJB 2489A-2023 places a ~1241x1755 JPEG at
        bbox=(0,0)-(595,842) on every page). Such images have coverage ~1.0 but
        are almost entirely white, so they carry no retrieval value and must not
        trigger figure routing — otherwise every text page is demoted to
        figure_review and its body text is never published as chunks.

        Heuristic: the image covers nearly the whole page AND its encoded byte
        density is very low for its pixel count (a JPEG whose stream is a few
        KB for a megapixel image can only encode a flat / sparse fill). Real
        photos / scans compress to far more bytes per pixel.
        """
        try:
            page_area = max(1.0, float(page.width) * float(page.height))
            img_w = float(image["x1"]) - float(image["x0"])
            img_h = float(image["bottom"]) - float(image["top"])
            coverage = (img_w * img_h) / page_area
            if coverage < 0.9:
                return False
            src_size = image.get("srcsize")
            if not isinstance(src_size, (list, tuple)) or len(src_size) < 2:
                return False
            src_w, src_h = float(src_size[0]), float(src_size[1])
            if src_w * src_h <= 0:
                return False
            stream = image.get("stream")
            if stream is None:
                return False
            try:
                data = stream.get_data()
            except (AttributeError, TypeError, ValueError):
                return False
            bytes_per_pixel = len(data) / (src_w * src_h)
            # A flat/blank fill compresses to < 0.05 B/px; real content (photos,
            # scans, diagrams) is denser. imagemask glyphs are handled elsewhere
            # and never reach this test with a whole-page bbox.
            return bytes_per_pixel < 0.05
        except (AttributeError, TypeError, ValueError):
            return False

    @staticmethod
    def _has_substantive_images(page: Any) -> bool:
        """Return whether the page has any non-background, content-bearing image.

        A full-page white background JPEG (see ``_is_fullpage_background_image``)
        is decorative; ignoring it prevents text pages with such a background
        from being routed to figure review. Small icons / logos are also not
        content figures, but they are already filtered by the coverage cut-offs
        in the caller.
        """
        for image in getattr(page, "images", []) or []:
            if VectorPdfParser._is_fullpage_background_image(page, image):
                continue
            return True
        return False


    @staticmethod
    def _has_body_vector_content(page: Any) -> bool:
        """Whether a page has vector content in the body (not only margins).

        Blank spacer pages carry a running header and page number as tiny vector
        elements in the top/bottom margins (e.g. a blank page between sections).
        Those are page furniture, not figure content.  A real diagram occupies
        the page body, i.e. outside the narrow header/footer marginal bands.
        """
        height = float(getattr(page, "height", 0.0) or 0.0)
        if height <= 0:
            return False
        lower, upper = height * 0.08, height * 0.92
        for key in ("lines", "rects", "curves"):
            for item in getattr(page, key, None) or []:
                top = float(getattr(item, "top", 0.0))
                bottom = float(getattr(item, "bottom", top))
                if lower < (top + bottom) / 2.0 < upper:
                    return True
        return False

    @staticmethod
    def _boxed_text_diagram_region(page: Any) -> list[float] | None:
        """Detect a text-based structure diagram inside a single outer frame.

        Some standards render structure/organization diagrams as one large
        outer rectangle (rect) filled with text items and a few short connector
        lines, with no image objects at all. Unlike a pure-vector flow chart
        (many box rects + arrow curves), these diagrams carry only 1-2 rects
        and a handful of lines, so ``_flowchart_regions`` rejects them. Example:
        "图1 III 类手册的构成" in HB 8730-2023 — a single outer frame around
        cover/record/preface/appendix item names.

        Return the outer frame bbox when the page matches that profile.
        """
        try:
            rects = list(getattr(page, "rects", []) or [])
            curves = list(getattr(page, "curves", []) or [])
            lines = list(getattr(page, "lines", []) or [])
            images = list(getattr(page, "images", []) or [])
        except (AttributeError, TypeError, ValueError):
            return None
        if images:
            # Real raster figures are handled by the normal image pipeline.
            return None
        # The diagram is one dominant outer frame plus at most a couple of
        # small inner rules. If a page has many boxes/connectors it is a real
        # flow chart and is handled by _flowchart_regions instead.
        if len(rects) != 1:
            return None
        if len(curves) > 3 or len(lines) > 10:
            return None
        rect = rects[0]
        x0, top, x1, bottom = (
            float(rect["x0"]),
            float(rect["top"]),
            float(rect["x1"]),
            float(rect["bottom"]),
        )
        width = x1 - x0
        height = bottom - top
        page_w = float(getattr(page, "width", 0) or 0)
        page_h = float(getattr(page, "height", 0) or 0)
        # The outer frame must be a substantial area of the page and a real
        # box (not a thin underline/rule or a tiny icon frame).
        if width < 60 or height < 60:
            return None
        if page_w > 0 and page_h > 0:
            area_ratio = (width * height) / (page_w * page_h)
            if area_ratio < 0.03 or area_ratio > 0.9:
                return None
        # Text items inside the frame: a structure diagram labels each part.
        try:
            frame_text = page.crop((x0, top, x1, bottom)).extract_text() or ""
        except Exception:
            frame_text = ""
        compact = frame_text.replace(" ", "").replace("\n", "")
        if len(compact) < 6:
            return None
        # At least three distinct label lines inside the frame (封面/更改记录/
        # 正文 …) confirm this is a diagram, not a lone emphasized sentence.
        label_lines = [
            line for line in frame_text.splitlines() if len(line.replace(" ", "")) >= 2
        ]
        if len(label_lines) < 3:
            return None
        return [x0, top, x1, bottom]

    @staticmethod
    def _flowchart_regions(page: Any) -> list[list[float]]:
        """Detect flow-chart style diagram regions and return their outer bounds.

        Some standards render flow charts as a set of box rectangles (rects),
        connecting arrows (curves) and 1-bit ``imagemask`` glyphs for the text
        inside each box, with no extractable text in the diagram area. Treating
        each imagemask as a separate figure would slice one diagram into dozens
        of tiny crops. Instead, when the page matches that profile, merge all
        flow-box rules into one outer bounding rectangle — the complete diagram.
        """
        try:
            image_count = len(page.images)
            imagemask_count = sum(
                1 for image in page.images if image.get("imagemask")
            )
            rects = list(getattr(page, "rects", []) or [])
            curves = list(getattr(page, "curves", []) or [])
        except (AttributeError, TypeError, ValueError):
            return []
        # Pure vector flow charts carry no image objects at all: box rectangles
        # (rects) + connecting arrows (curves) + text inside each box. Allow
        # image_count == 0 when the rule/arrow profile is unmistakably a flow
        # chart (many boxes and connectors).
        is_pure_vector_flow = (
            image_count == 0
            and len(rects) >= 6
            and len(curves) >= 4
        )
        if image_count == 0 and not is_pure_vector_flow:
            return []
        # Flow chart profile A: most images are text masks, plus box rules and
        # connecting arrows, and little extractable text in the diagram area.
        is_imagemask_flow = (
            image_count > 0
            and imagemask_count >= max(4, image_count * 0.6)
        )
        # Flow chart profile B: some PDFs render every box as a small regular
        # image (not an imagemask) plus many box rules (rects) and connecting
        # arrows (curves). These are still one logical flow diagram and must not
        # be sliced per-image into dozens of figure assets that each trigger a
        # VLM call. Recognise them by a large rule/arrow count and images that
        # are all small decorative tiles.
        small_image_ratio = (
            sum(
                1
                for image in page.images
                if (float(image["x1"]) - float(image["x0"])) <= 200
                and (float(image["bottom"]) - float(image["top"])) <= 60
            )
            / image_count
            if image_count > 0
            else 0.0
        )
        is_tile_flow = (
            len(rects) >= 10
            and len(curves) >= 10
            and image_count >= 4
            and small_image_ratio >= 0.8
        )
        if not (is_imagemask_flow or is_tile_flow or is_pure_vector_flow):
            return []
        if len(rects) < 3 or len(curves) < 3:
            return []
        # Unique rect bounds (pdfplumber may report each rule twice).
        seen: set[tuple[float, float, float, float]] = set()
        bounds: list[tuple[float, float, float, float]] = []
        for rect in rects:
            key = (
                round(float(rect["x0"]), 1),
                round(float(rect["top"]), 1),
                round(float(rect["x1"]), 1),
                round(float(rect["bottom"]), 1),
            )
            if key not in seen:
                seen.add(key)
                bounds.append(
                    (
                        float(rect["x0"]),
                        float(rect["top"]),
                        float(rect["x1"]),
                        float(rect["bottom"]),
                    )
                )
        if not bounds:
            return []
        # The whole flow chart is a single logical diagram: merge every box
        # rule (title box + main flow + output box) into one outer bounding
        # rectangle. Splitting title/body/output into separate crops is not
        # useful for retrieval — the reader wants the complete diagram.
        outer_x0 = min(b[0] for b in bounds)
        outer_top = min(b[1] for b in bounds)
        outer_x1 = max(b[2] for b in bounds)
        outer_bottom = max(b[3] for b in bounds)
        return [[outer_x0, outer_top, outer_x1, outer_bottom]]

    def preflight_pages(
        self,
        path: Path,
        *,
        candidate_pages: set[int],
    ) -> VectorPreflightResult:
        """Find table, visual, and formula pages hidden by native extraction.

        Native PDF extraction can return a small amount of valid caption text
        for an otherwise image-only chart, or flatten vector equations into a
        misleading text stream. Those pages therefore never reach the normal
        low-text/layout routes. This inexpensive pass inspects PDF primitives
        without rendering pages.
        """

        result = VectorPreflightResult()
        if not candidate_pages:
            return result
        with pdfplumber.open(path) as pdf:
            for page_number in sorted(candidate_pages):
                if page_number < 1 or page_number > len(pdf.pages):
                    continue
                page = pdf.pages[page_number - 1]
                page_text = (page.extract_text() or "").strip()
                compact_text = page_text.replace(" ", "")
                # A standards cover page (ICS/CCS classification + 发布/实施 +
                # a standard number like YD/T) carries decorative ruled frames,
                # not a real table. pdfplumber can mistake the title frame for
                # a table grid; exclude it so the cover is not parsed as a
                # table chunk.
                is_cover_page = _looks_like_cover_page(compact_text)
                has_structured_table = (
                    False
                    if is_cover_page
                    else bool(self._extract_best_tables(page))
                )
                if has_structured_table:
                    result.table_pages.add(page_number)
                image_count, image_coverage, largest_image_coverage = (
                    self._visual_coverage(page)
                )
                # A full-page white/blank JPEG background (see
                # _is_fullpage_background_image) is decorative, not content.
                # Count only substantive images so text pages with such a
                # background are not routed to figure review (GJB 2489A-2023
                # carries a ~1241x1755 white JPEG on every page).
                substantive_image_count = sum(
                    1
                    for image in getattr(page, "images", []) or []
                    if not self._is_fullpage_background_image(page, image)
                )
                # A figure caption at the start of a line ("图1 物流无人机…")
                # is a strong signal that the page contains a real illustration,
                # even when the rendered image falls just below the coverage
                # threshold (e.g. a 0.1154 coverage diagram vs the 0.12 cut-off).
                has_figure_caption = bool(
                    FIGURE_CAPTION_RE.search(page_text)
                )
                # A pure-vector flow chart (box rects + arrow curves, no image
                # objects) with a "图N …" caption is a real figure even though
                # image_count is 0. Render/emit it as a whole-diagram figure.
                pure_vector_flow = bool(
                    self._flowchart_regions(page)
                )
                # A text-based structure diagram (one outer frame + label text +
                # a "图N …" caption, no image objects) is also a real figure,
                # e.g. "图1 III 类手册的构成" in HB 8730-2023. Such pages carry
                # only 1 rect and a few lines, so they never match the flow-chart
                # profile; without this signal they fall through to native_text
                # and the diagram is never emitted as a figure asset.
                boxed_text_diagram = bool(
                    self._boxed_text_diagram_region(page)
                )
                # Some regulations introduce a graph inline ("shown in the
                # following figure") rather than below a conventional "Figure"
                # caption.  Its raster is often small relative to a text-heavy
                # page, so image coverage alone misses it.
                has_inline_figure_reference = bool(
                    re.search(
                        r"\b(?:shown in (?:the )?following figure|following figure shows)\b",
                        page_text,
                        re.IGNORECASE,
                    )
                )
                # Native-text regulations sometimes render a *table* as a raster
                # image on an otherwise text-layer page (e.g. an FAA NPRM page
                # carrying a bird-weight table, "TABLE 1", "TABLE 2", "TABLE 3"
                # grids). Its cells exist only in pixels, so the vector-grid /
                # figure path can never recover the rows — only remote OCR can.
                #
                # The PDF itself tags these as structure ``Figure`` content
                # items (pdfplumber exposes ``image.tag``), while an untagged
                # logo, stamp, or background does not carry that marker. A real
                # illustration, in contrast, has an explicit figure signal: a
                # 图N caption, an inline "shown in the following figure"
                # reference, a flow chart, or a boxed diagram. When a page
                # carries a structurally-tagged ``Figure`` raster and NONE of
                # those explicit figure signals, it is overwhelmingly a raster
                # table on a text page — route it to OCR (``raster_table_pages``)
                # regardless of coverage, so the rows are recovered instead of
                # a cropped image with no cell values. Coverage must not win
                # here: a text page can stack several wide-but-short raster
                # tables (e.g. page 25 of 23-54-DRS_98-19 carries TABLE 2,
                # TABLE 3 and the 33.77(e) grid at ~39% combined coverage).
                has_structured_figure_content = any(
                    not self._is_fullpage_background_image(page, image)
                    and str(image.get("tag") or "").rstrip() == "Figure"
                    for image in getattr(page, "images", []) or []
                )
                explicit_figure_signal = bool(
                    has_figure_caption
                    or (
                        has_inline_figure_reference
                        and image_coverage >= 0.025
                    )
                    or (has_figure_caption and pure_vector_flow)
                    or (has_figure_caption and boxed_text_diagram)
                )
                if has_structured_figure_content and not explicit_figure_signal:
                    result.raster_table_pages.add(page_number)
                    page_width = float(getattr(page, "width", 0) or 0)
                    page_height = float(getattr(page, "height", 0) or 0)
                    boxes: list[list[float]] = []
                    for image in getattr(page, "images", []) or []:
                        if self._is_fullpage_background_image(page, image):
                            continue
                        if str(image.get("tag") or "").rstrip() != "Figure":
                            continue
                        if page_width <= 0 or page_height <= 0:
                            continue
                        # crop_table_pdf expects a 0..1000 normalized bbox;
                        # pdfplumber reports image corners in page points.
                        boxes.append([
                            round(float(image["x0"]) / page_width * 1000, 2),
                            round(float(image["top"]) / page_height * 1000, 2),
                            round(float(image["x1"]) / page_width * 1000, 2),
                            round(float(image["bottom"]) / page_height * 1000, 2),
                        ])
                    result.raster_table_boxes[page_number] = boxes
                elif (
                    substantive_image_count
                    and (
                        largest_image_coverage >= 0.12
                        or (
                            image_coverage >= 0.15
                            and len(compact_text) < 300
                        )
                        or has_figure_caption
                        or (
                            has_inline_figure_reference
                            and image_coverage >= 0.025
                        )
                    )
                ) or (
                    has_figure_caption and pure_vector_flow
                ) or (
                    has_figure_caption and boxed_text_diagram
                ):
                    result.figure_pages.add(page_number)

                nonempty_lines = [
                    line.strip() for line in page_text.splitlines() if line.strip()
                ]
                short_line_count = sum(
                    len(line.replace(" ", "")) <= 5 for line in nonempty_lines
                )
                graphic_count = len(page.lines) + len(page.curves)
                equation_count = page_text.count("=")
                # Fraction-style formulas split across lines (numerator line,
                # horizontal rule, denominator line) can carry only a single "="
                # while exposing function names (floor/ceil) and isolated short
                # fragments ("( )", "30", "2R"). Those are strong math signals
                # even when the >=4 "=" heuristic does not fire. Small inline
                # images (logos, decorations) do not negate the formula signal.
                has_formula_function = bool(FORMULA_FUNCTION_RE.search(page_text))
                # Fraction fragments ("( )", "30", "2R", "F") can appear anywhere
                # on the page, not just near the top (e.g. after a long table).
                paren_fragment_line = any(
                    "(" in line and ")" in line and len(line.replace(" ", "")) <= 6
                    for line in nonempty_lines
                )
                fraction_fragments = short_line_count >= 3 and paren_fragment_line
                if (
                    (
                        not image_count
                        and equation_count >= 4
                        and graphic_count >= 20
                        and short_line_count >= 4
                        and len(compact_text) < 900
                    )
                    or (
                        has_formula_function
                        and equation_count >= 1
                        and fraction_fragments
                    )
                ):
                    result.formula_pages.add(page_number)
                # Math typesetting fonts (e.g. CambriaMath) are a strong signal
                # that a page contains real formulas whose subscripts/superscripts
                # are lost by plain text extraction. Route those pages to OCR so
                # the visual model can preserve the baseline offsets. A page
                # whose vector table grid is fully recoverable (e.g. a blank
                # record form carrying a CambriaMath formula in its footer note)
                # is a table first: keeping it on the table path preserves the
                # exact grid instead of degrading it through OCR.
                elif self._has_math_font(page) and not has_structured_table:
                    result.formula_pages.add(page_number)
            # A table whose vertical rules continue below the last detected
            # row can have an otherwise ordinary-looking next page: there is
            # no closed rectangle for ``find_tables`` to return. Promote that
            # page to a table candidate so ``parse_pages`` can recover the
            # open row from its matching top-of-page grid.
            changed = True
            while changed:
                changed = False
                for page_number in sorted(result.table_pages):
                    next_page_number = page_number + 1
                    if next_page_number not in candidate_pages:
                        continue
                    if next_page_number > len(pdf.pages):
                        continue
                    previous = pdf.pages[page_number - 1]
                    following = pdf.pages[next_page_number - 1]
                    previous_tables = self._extract_best_tables(previous)
                    if not previous_tables:
                        continue
                    previous_bbox, previous_rows = max(
                        previous_tables,
                        key=lambda item: item[0][3],
                    )
                    width = max((len(row) for row in previous_rows), default=0)
                    tail = self._open_table_tail(previous, previous_bbox, width)
                    if tail is None:
                        continue
                    _tail_bbox, columns = tail
                    if (
                        next_page_number not in result.table_pages
                        and self._continuation_grid(following, columns) is not None
                    ):
                        result.table_pages.add(next_page_number)
                        changed = True
        result.table_series = self.contiguous_series(result.table_pages)
        return result

    def parse_pages(
        self,
        path: Path,
        *,
        table_pages: set[int],
        figure_pages: set[int],
        formula_pages: set[int] | None = None,
    ) -> dict[int, VectorPageResult]:
        formula_pages = formula_pages or set()
        targets = sorted(table_pages | figure_pages | formula_pages)
        results: dict[int, VectorPageResult] = {}
        if not targets:
            return results

        with pdfplumber.open(path) as pdf:
            for page_number in targets:
                if page_number < 1 or page_number > len(pdf.pages):
                    continue
                page = pdf.pages[page_number - 1]
                result = VectorPageResult(page_number=page_number)
                table_assets: list[tuple[str, list[float], str]] = []
                page_text = (page.extract_text() or "").strip()
                image_boxes = [
                    self._bbox(
                        (
                            image["x0"],
                            image["top"],
                            image["x1"],
                            image["bottom"],
                        )
                    )
                    for image in page.images
                ]
                page_area = max(1.0, float(page.width) * float(page.height))
                image_coverages = [
                    max(0.0, bbox[2] - bbox[0])
                    * max(0.0, bbox[3] - bbox[1])
                    / page_area
                    for bbox in image_boxes
                ]
                result.image_count = len(image_boxes)
                result.image_coverage = round(
                    min(1.0, sum(image_coverages)),
                    4,
                )
                largest_image_coverage = max(image_coverages, default=0.0)
                has_visual_content = bool(
                    page_text
                    or page.images
                    or page.lines
                    or page.rects
                    or page.curves
                )
                # Table pages stay tables even when they also carry a math font
                # (e.g. CambriaMath) in a footer formula note. Treating them as
                # formula pages would demote the exact vector grid to OCR and
                # split a blank record form into noisy fragments.
                if page_number in formula_pages and page_number in table_pages:
                    result.page_type = "table"
                elif page_number in formula_pages:
                    result.page_type = "formula"
                elif (
                    len(page_text.replace(" ", "")) < 80
                    and len(image_boxes) == 1
                    and largest_image_coverage >= 0.7
                ):
                    result.page_type = "scanned_text"
                elif page_number in table_pages and image_boxes:
                    result.page_type = "table_figure"
                elif page_number in table_pages:
                    result.page_type = "table"
                elif image_boxes and (
                    len(page_text.replace(" ", "")) < 120
                    or result.image_coverage >= 0.15
                ):
                    result.page_type = "figure"
                elif has_visual_content and (
                    page_text
                    or image_boxes
                    or self._has_body_vector_content(page)
                ):
                    result.page_type = "figure"
                else:
                    result.page_type = "blank"

                if page_number in table_pages:
                    # A page whose text carries a 记录表/登记表/申报表 caption
                    # (e.g. "A.2 空气中化学毒物采样记录表") is a blank record
                    # form. Its field labels are the retrieval payload and a
                    # whole-page image is attached for visual verification.
                    page_is_record_form = bool(
                        re.search(
                            r"(?:记录表|登记表|申报表|记录单|检查表|调查表)",
                            page_text,
                        )
                    )
                    try:
                        captioned_tables = self._extract_captioned_tables(page)
                    except Exception as exc:
                        captioned_tables = []
                        result.warnings.append(
                            f"Vector table detection failed: {type(exc).__name__}: {exc}"
                        )
                    if not captioned_tables:
                        continuation = self._extract_open_table_continuation(page)
                        if continuation is not None:
                            captioned_tables = [continuation]
                            continuation_bbox, continuation_rows, _ = continuation
                            captioned_tables.extend(
                                self._extract_following_table_fragments(
                                    page,
                                    continuation_bbox,
                                    max(
                                        (len(row) for row in continuation_rows),
                                        default=0,
                                    ),
                                )
                            )
                    for ordinal, (table_bbox, rows, table_title) in enumerate(
                        captioned_tables,
                        start=1,
                    ):
                        normalized_rows = [
                            [normalize_cell(cell) for cell in row] for row in rows
                        ]
                        table_html = table_to_html(normalized_rows)
                        text = table_to_markdown(normalized_rows, title=table_title or None)
                        if not table_html or not text:
                            continue
                        asset_id = self._asset_id(page_number, "table", ordinal)
                        block_asset_ids: list[str] = [asset_id]
                        # Blank record forms (记录表/登记表/申报表) carry a header
                        # row plus empty data rows. Their field labels are the
                        # retrieval payload, but a whole-page image lets users
                        # visually confirm the original form layout. Attach the
                        # page image to the table block so the published chunk
                        # carries both the searchable grid and the original page.
                        # The whole-page form image is placed first so it becomes
                        # the chunk's image_base64 in the publisher.
                        if page_is_record_form or self._looks_like_record_form(
                            normalized_rows,
                            title=table_title or text,
                        ):
                            form_page_id = self._asset_id(page_number, "form", 1)
                            block_asset_ids.insert(0, form_page_id)
                            if not any(
                                asset.asset_id == form_page_id
                                for asset in result.assets
                            ):
                                result.assets.append(
                                    VectorAsset(
                                        asset_id=form_page_id,
                                        page_number=page_number,
                                        asset_type="form_page",
                                        bbox=[
                                            0.0,
                                            0.0,
                                            float(page.width),
                                            float(page.height),
                                        ],
                                        filename=f"page-{page_number:04d}.png",
                                        mime_type="image/png",
                                        content=self._render_page(page),
                                        caption=f"原 PDF 第 {page_number} 页记录表版面",
                                    )
                                )
                        result.blocks.append(
                            VectorBlock(
                                block_type="table",
                                text=text,
                                bbox=table_bbox,
                                table_html=table_html,
                                table_title=(
                                    repair_fake_glyphs(table_title)
                                    if table_title
                                    else None
                                ),
                                table_rows=normalized_rows,
                                table_header_only=len(normalized_rows) == 1,
                                table_merged_cell_count=sum(
                                    cell is None for row in rows for cell in row
                                ),
                                asset_id=asset_id,
                                asset_ids=block_asset_ids,
                            )
                        )
                        table_assets.append(
                            (
                                asset_id,
                                table_bbox,
                                (table_title or text.splitlines()[0])[:160],
                            )
                        )
                        result.table_count += 1
                    if result.table_count:
                        result.blocks.extend(
                            self._outside_table_blocks(
                                page,
                                [
                                    table_bbox
                                    for table_bbox, _, _ in captioned_tables
                                ],
                            )
                        )
                        result.blocks.sort(
                            key=lambda block: (
                                block.bbox[1] if block.bbox else 0.0,
                                block.bbox[0] if block.bbox else 0.0,
                            )
                        )

                if (
                    result.page_type != "blank"
                    and (
                        (page_number in figure_pages and has_visual_content)
                        or page_number in formula_pages
                        or result.table_count
                    )
                ):
                    try:
                        page_png = self._render_page(page)
                    except Exception as exc:
                        result.warnings.append(
                            f"Page rendering failed: {type(exc).__name__}: {exc}"
                        )
                    else:
                        page_asset_id = self._asset_id(page_number, "page", 1)
                        # A pure table page (page_type == "table", including a
                        # blank record form whose math-font formula note is
                        # preserved inside the table) does not need a whole-page
                        # figure asset that duplicates the table content.
                        is_pure_table = (
                            result.page_type == "table"
                            and page_number not in figure_pages
                        )
                        if (
                            not is_pure_table
                            and (
                                page_number in figure_pages
                                or page_number in formula_pages
                                or result.page_type == "table_figure"
                            )
                        ):
                            if result.page_type == "scanned_text":
                                page_asset_type = "scan_page"
                            elif result.page_type == "formula":
                                page_asset_type = "formula_page"
                            else:
                                page_asset_type = "figure_page"
                            result.assets.append(
                                VectorAsset(
                                    asset_id=page_asset_id,
                                    page_number=page_number,
                                    asset_type=page_asset_type,
                                    bbox=[
                                        0.0,
                                        0.0,
                                        float(page.width),
                                        float(page.height),
                                    ],
                                    filename=f"page-{page_number:04d}.png",
                                    mime_type="image/png",
                                    content=page_png,
                                    caption=f"原 PDF 第 {page_number} 页版面",
                                )
                            )
                            if result.page_type == "formula":
                                result.blocks.append(
                                    VectorBlock(
                                        block_type="formula",
                                        text=f"第 {page_number} 页公式原版图像",
                                        bbox=[
                                            0.0,
                                            0.0,
                                            float(page.width),
                                            float(page.height),
                                        ],
                                        asset_id=page_asset_id,
                                    )
                                )
                        for asset_id, bbox, caption in table_assets:
                            result.assets.append(
                                VectorAsset(
                                    asset_id=asset_id,
                                    page_number=page_number,
                                    asset_type="table",
                                    bbox=bbox,
                                    filename=f"{asset_id}.png",
                                    mime_type="image/png",
                                    content=self._crop_png(
                                        page_png,
                                        page_width=float(page.width),
                                        page_height=float(page.height),
                                        bbox=bbox,
                                        origin=self._page_origin(page),
                                    ),
                                    caption=caption,
                                )
                            )
                        figure_assets: list[VectorAsset] = []
                        if (
                            (
                                page_number in figure_pages
                                or result.page_type == "table_figure"
                            )
                            and result.page_type in {"figure", "table_figure"}
                        ):
                            # Flow-chart style diagrams (many 1-bit text-mask
                            # images inside box rules + arrows) must be cropped
                            # as whole diagram regions, not sliced per imagemask.
                            flow_regions = self._flowchart_regions(page)
                            boxed_region = self._boxed_text_diagram_region(page)
                            if flow_regions:
                                for ordinal, bbox in enumerate(flow_regions, start=1):
                                    caption = self._caption_near(page, bbox)
                                    asset_id = self._asset_id(
                                        page_number,
                                        "figure",
                                        ordinal,
                                    )
                                    figure_asset = VectorAsset(
                                        asset_id=asset_id,
                                        page_number=page_number,
                                        asset_type="figure",
                                        bbox=bbox,
                                        filename=f"{asset_id}.png",
                                        mime_type="image/png",
                                        content=self._figure_crop_png(
                                            page,
                                            bbox=bbox,
                                        ),
                                        caption=(
                                            caption
                                            or f"原 PDF 第 {page_number} 页流程图 {ordinal}"
                                        ),
                                    )
                                    figure_assets.append(figure_asset)
                                    result.blocks.append(
                                        VectorBlock(
                                            block_type="figure",
                                            text=figure_asset.caption,
                                            bbox=bbox,
                                            asset_id=asset_id,
                                        )
                                    )
                            elif boxed_region:
                                # A text-based structure diagram (one outer frame
                                # + label text + caption) is emitted as a single
                                # whole-frame figure, mirroring the flow-chart
                                # handling. Without an image object there is no
                                # per-image box to crop; crop the outer frame.
                                ordinal = 1
                                caption = self._caption_near(page, boxed_region)
                                asset_id = self._asset_id(
                                    page_number,
                                    "figure",
                                    ordinal,
                                )
                                figure_asset = VectorAsset(
                                    asset_id=asset_id,
                                    page_number=page_number,
                                    asset_type="figure",
                                    bbox=boxed_region,
                                    filename=f"{asset_id}.png",
                                    mime_type="image/png",
                                    content=self._figure_crop_png(
                                        page,
                                        bbox=boxed_region,
                                    ),
                                    caption=(
                                        caption
                                        or f"原 PDF 第 {page_number} 页结构图 {ordinal}"
                                    ),
                                )
                                figure_assets.append(figure_asset)
                                result.blocks.append(
                                    VectorBlock(
                                        block_type="figure",
                                        text=figure_asset.caption,
                                        bbox=boxed_region,
                                        asset_id=asset_id,
                                    )
                                )
                            else:
                                for ordinal, bbox in enumerate(image_boxes, start=1):
                                    coverage = image_coverages[ordinal - 1]
                                    if coverage < 0.005 or coverage >= 0.7:
                                        continue
                                    # Thin horizontal decorative elements
                                    # (underlines, caption rules, divider bars)
                                    # are not real figures: a short band whose
                                    # height is far smaller than its width
                                    # (e.g. a 117x25 caption rule). Skip them so
                                    # they do not become standalone figure
                                    # chunks with a bogus VLM description.
                                    img_w = max(0.0, float(bbox[2]) - float(bbox[0]))
                                    img_h = max(0.0, float(bbox[3]) - float(bbox[1]))
                                    if (
                                        img_h < 35
                                        and img_w > 0
                                        and img_h / img_w < 0.3
                                    ):
                                        continue
                                    asset_id = self._asset_id(
                                        page_number,
                                        "figure",
                                        ordinal,
                                    )
                                    caption = self._caption_near(page, bbox)
                                    figure_asset = VectorAsset(
                                        asset_id=asset_id,
                                        page_number=page_number,
                                        asset_type="figure",
                                        bbox=bbox,
                                        filename=f"{asset_id}.png",
                                        mime_type="image/png",
                                        content=self._figure_crop_png(
                                            page,
                                            bbox=bbox,
                                        ),
                                        caption=(
                                            caption
                                            or f"原 PDF 第 {page_number} 页图表 {ordinal}"
                                        ),
                                    )
                                    figure_assets.append(figure_asset)
                                    result.blocks.append(
                                        VectorBlock(
                                            block_type="figure",
                                            text=figure_asset.caption,
                                            bbox=bbox,
                                            asset_id=asset_id,
                                        )
                                    )
                            result.assets.extend(figure_assets)
                        if (
                            (
                                page_number in figure_pages
                                or result.page_type == "table_figure"
                            )
                            and result.page_type in {"figure", "table_figure"}
                            and not figure_assets
                        ):
                            result.blocks.append(
                                VectorBlock(
                                    block_type="figure",
                                    text=f"原 PDF 第 {page_number} 页图表。",
                                    bbox=[
                                        0.0,
                                        0.0,
                                        float(page.width),
                                        float(page.height),
                                    ],
                                    asset_id=page_asset_id,
                                )
                            )
                        # Figure pages also carry body text (e.g. "5 高水效小麦
                        # 品种鉴定" under a flow-chart figure). Emit it as
                        # coordinate-bearing paragraph blocks so split_blocks can
                        # interleave paragraphs and figure blocks in true reading
                        # order. Without bboxes, split_blocks appends the figure
                        # at the end of the page's text and misattributes it to
                        # the last clause.
                        if (
                            page_number in figure_pages
                            and result.page_type == "figure"
                        ):
                            result.blocks.extend(
                                self._line_blocks(page)
                            )
                            # A text-based structure diagram (single outer frame
                            # + label text) labels each part inside the frame
                            # (封面/更改记录/正文/附录 …). Those labels are part
                            # of the diagram, not body paragraphs: emitting them
                            # as standalone blocks lets a frame label such as
                            # "附录" hijack the clause stack and misroute the
                            # surrounding section content. Drop paragraph blocks
                            # whose bbox lies fully inside the outer frame; the
                            # whole diagram is already emitted as a figure asset.
                            if boxed_region:
                                bx0, btop, bx1, bbottom = boxed_region
                                result.blocks = [
                                    block
                                    for block in result.blocks
                                    if block.block_type != "paragraph"
                                    or not block.bbox
                                    or not (
                                        float(block.bbox[0]) >= bx0 - 1
                                        and float(block.bbox[1]) >= btop - 1
                                        and float(block.bbox[2]) <= bx1 + 1
                                        and float(block.bbox[3]) <= bbottom + 1
                                    )
                                ]
                            result.blocks.sort(
                                key=lambda block: (
                                    block.bbox[1] if block.bbox else 0.0,
                                    block.bbox[0] if block.bbox else 0.0,
                                )
                            )
                results[page_number] = result
        return results
