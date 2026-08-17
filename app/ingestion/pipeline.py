from __future__ import annotations

# Canonical preprocessing engine used by the ingestion service and CLI experiment.
import argparse
import hashlib
import html as html_lib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from pypdf import PdfReader
from pypdf import __version__ as pypdf_version

from app.ingestion.regulations import (
    REGULATION_MAX_CHARS,
    REGULATION_OVERLAP_CHARS,
    REGULATION_TARGET_CHARS,
    build_question_aliases,
    extract_keywords,
    match_article_heading,
)

TEXT_PAGE_MIN_CHARS = 30
REPEATED_MARGIN_FRACTION = 0.25
TARGET_CHUNK_CHARS = REGULATION_TARGET_CHARS
MAX_CHUNK_CHARS = REGULATION_MAX_CHARS
CHUNK_OVERLAP_CHARS = REGULATION_OVERLAP_CHARS

PAGE_NUMBER_RE = re.compile(r"^[—\-–\s]*\d+[—\-–\s]*$")
# Annex decoration: appendix pages sometimes repeat a lone uppercase letter
# (e.g. "A") at the top of every page. After line normalization two of these
# collapse into "A A" which is not a heading and would otherwise be emitted
# as junk text under the previous clause. Match a lone letter OR the "A A"
# pair so both the raw single-letter line and the normalized pair are removed.
ANNEX_DECOR_RE = re.compile(r"^(?:[A-Z]|[A-Z]\s+[A-Z])$")
# Annex nature marker lines: "(资料性)" / "(规范性)" that follow "附录 X".
# They are part of the annex heading and must stay on their own line.
ANNEX_MARKER_RE = re.compile(r"^[（(]\s*(?:资料性|规范性|附录性)\s*[）)]$")
# Blank record forms (记录表/申报表) carry placeholder noise that vector text
# extraction emits as standalone short paragraphs: a right-margin standards
# number (DB4401/T), an isolated dash, or the unfilled "第页，共页" page-count
# placeholder. These carry no retrieval value and would otherwise pollute the
# form chunk with meaningless fragments.
FORM_STANDARD_NUMBER_RE = re.compile(
    r"^[A-Z]{2,}\s*\d+\s*[／/]\s*[A-Z](?:\s*[—\-–]?\s*\d*)?$"
)
FORM_DASH_ONLY_RE = re.compile(r"^[—\-–]{1,3}$")
FORM_PAGE_PLACEHOLDER_RE = re.compile(r"^.{0,12}[第编]?\d*[页]\s*[，,]\s*共\s*\d*[页]?$")
# Damaged fonts map "GB/T" to CJK fake glyphs such as 犌犅/犜—. These repeat as
# headers on every page and must be removed from indexable text.
FAKE_GB_HEADER_RE = re.compile(r"^犌犅[／/]犜.*$")
TABLE_TITLE_RE = re.compile(r"^\s*表\s*[A-Za-z0-9一二三四五六七八九十.-]+\s+.+")
# Prose references to a table like "表 A.6 的要求" / "表A.5 的规定" are NOT
# table titles — they are sentences that mention a table. Treating them as
# titles creates phantom expected captions that fail the caption-alignment
# gate (GB 46761-2025 page 17). Exclude these from table-hint detection.
TABLE_TITLE_PROSE_RE = re.compile(
    r"^表\s*(?:[A-Za-z]\s*[.\-]?\s*)?(?:\d+(?:[.\-]\d+)*|"
    r"[一二三四五六七八九十]+)\s*的\s*(?:要求|规定|参数|内容|格式|结构)[。.]?$",
    re.IGNORECASE,
)
# A "表 N …" whose text after the number starts with a verb / digit / English
# is a cell reference, not a table title. Fragmented text layers (AC-23-AA-2022-01
# p28) split "ASTM F3230-17 表3 替换…" into isolated lines, and detect_table_hints
# rejoins "表" + "3" + "替换" into a phantom title. Genuine table titles are
# noun phrases ("表 3 空域编码规则"). Reject verb/digit/English heads.
_TABLE_TITLE_CONTENT_RE = re.compile(
    r"^表\s*(?:[A-Za-z]\s*[.\-]?\s*)?(?:\d+(?:[.\-]\d+)*|"
    r"[一二三四五六七八九十]+)\s*"
    r"(?:[0-9A-Za-z]|[替换列出规定给根据位于以作对应详见为至从第其])"
)
TOC_TITLE_RE = re.compile(
    r"^\s*(?:目\s*录|目\s*次|contents)\s*$", re.IGNORECASE
)
TOC_DOTTED_ENTRY_RE = re.compile(r"(?:\.{3,}|…{2,}|·{3,})\s*\d+\s*$")
# OCR TOC rows sometimes interleave the leader with other noise characters
# before the page number (e.g. "A27.2 格式 … ● ¨ ● ¨ ● 216"). The page
# number must still end the line so table cells whose leader is followed by a
# bracket (e.g. "…………（") are not misclassified as TOC entries.
TOC_DOTTED_NOISY_RE = re.compile(r"(?:[.…·¨●]\s*){3,}\s*\d+\s*$")
# Some TOC layouts place the page number BEFORE the leader (e.g.
# "5.1 功能 2………"). The line starts with a numbered entry, contains a page
# number, and ends with a dot/ellipsis leader.
TOC_DOTTED_AFTER_ENTRY_RE = re.compile(
    r"^\s*(?:第.+[章节条]|[A-Z]?\d+(?:\.\d+)*\s+\S.*)\s+\d+\s*[.…·]{3,}\s*$"
)
# Some TOC layouts strip the page number entirely, leaving "5.1 检测流程……"
# or "范围……" with the leader at the end of the line and no trailing digit.
TOC_DOTTED_NO_PAGE_RE = re.compile(
    r"^\s*(?:[A-Z]?\d+(?:\.\d+)*\s*\S.*|[一-鿿].*)\s*[.…·]{3,}\s*$"
)
TOC_NUMBERED_ENTRY_RE = re.compile(
    r"^\s*(?:第.+[章节条]|[A-Z]?\d+(?:\.\d+)+\s+.+)\s+\d+\s*$"
)
LIST_ITEM_RE = re.compile(
    r"^\s*(?:[（(][一二三四五六七八九十百0-9a-zA-Z]+[）)]|"
    r"[一二三四五六七八九十百0-9a-zA-Z]+[、.)）])"
)

# A numbered regulation line can be either a structural heading (for example,
# "4.1 基本要求") or the complete body of a short clause (for example,
# "6.1.2 ……应符合……。").  The latter must remain publishable even when no
# following paragraph belongs to the same section.
CLAUSE_ASSERTION_RE = re.compile(
    r"(?:应当|必须|不得|不应|严禁|禁止|应|宜|可|须|需|符合|具备|具有|"
    r"负责|分为|包括|以下简称|用于|提供|支持|允许)"
)
CLAUSE_SENTENCE_END_RE = re.compile(r"[。！？；.!?;][\"'”’）)]*$")

HEADING_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "chapter",
        re.compile(r"^\s*第\s*[一二三四五六七八九十百零〇0-9]+\s*章(?:\s+|$)"),
    ),
    (
        "article",
        re.compile(
            r"^\s*第\s*(?:[A-Za-z]{0,5}\s*)?"
            r"(?:\d+(?:\.\d+)*|[一二三四五六七八九十百零〇]+)\s*条(?:\s+|$)"
        ),
    ),
    (
        "annex",
        # A real annex heading carries a label: "附录 A", "附录B", "附录 一",
        # optionally followed by a nature marker "(资料性附录)". A bare "附录"
        # (no label) is a body-text reference — either "…详见附录" or a pypdf
        # line-fragment of a sentence such as "附录 B 给出了…" that splits into
        # "附录" + "B" + "给出了". Requiring the label and rejecting a label
        # directly followed by prose (a verb/subject such as "给出了") prevents
        # body sentences from hijacking the annex clause stack
        # (GB/T 19001-2016 p8).
        re.compile(
            r"^\s*(?:附录|附件)\s*(?=[A-Za-z0-9一二三四五六七八九十百零〇]+)"
        ),
    ),
    (
        "clause",
        re.compile(
            r"^\s*(?:(?:[A-Z]{1,6}\.)\d+(?:\.\d+){0,3}|"
            r"\d+(?:\.\d+){1,4})\s+"
        ),
    ),
    (
        # Top-level numbered clauses in standards documents where the PDF
        # lays out the number and the CJK title on separate lines (e.g. a
        # lone "6" above "追溯方法"). After line normalization they appear
        # as "6 追溯方法". These must switch the active clause so trailing
        # content on the next page does not inherit the previous clause.
        # Titles must not begin with a measure word (个/次/年/天/月/条/种/类...)
        # so "3 次重复" or "2 个" are not misclassified as headings.
        # Titles may carry Roman numerals / Latin letters / parentheses
        # (e.g. "6 III类手册编制", "6 操作员手册(III－1)编制"), so the
        # CJK-only constraint from before is relaxed to allow them.
        "clause",
        re.compile(
            r"^\s*\d{1,2}\s+"
            r"(?!个|次|年|天|月|条|名|种|类|份|处|倍|元|米|cm|mm|kg|℃|°)"
            r"[0-9IVXLCMivxlcm()（）\-—–－．A-Za-z一-鿿]{2,20}$"
        ),
    ),
)

# PDF fonts with broken Unicode mappings sometimes expose PostScript glyph names
# such as /G21, /G22, ... instead of real text. pypdf extracts these literally.
GLYPH_NAME_RE = re.compile(r"/G[0-9A-F]{2,}")


@dataclass(slots=True)
class PageRecord:
    page_number: int
    raw_text: str
    cleaned_text: str = ""
    layout_text: str | None = None
    raw_char_count: int = 0
    cleaned_char_count: int = 0
    invalid_unicode_count: int = 0
    text_layer_corruption: dict[str, Any] = field(default_factory=dict)
    table_hints: list[str] = field(default_factory=list)
    route: str = "native_text"
    is_toc: bool = False
    indexable: bool = True
    rich_blocks: list[dict[str, Any]] = field(default_factory=list)
    page_type: str = "text"
    image_count: int = 0
    image_coverage: float = 0.0
    # First-layer layout features preserved for downstream scoring (e.g. TOC
    # detection, visual-review triage). Kept lossless; nothing is deleted here.
    layout_features: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BlockRecord:
    block_id: str
    block_type: str
    text: str
    page_number: int
    section_path: list[str]
    bbox: list[float] | None = None
    confidence: float | None = None
    article_id_raw: str | None = None
    article_id_normalized: str | None = None
    article_parent_id: str | None = None
    article_aliases: list[str] = field(default_factory=list)
    table_html: str | None = None
    asset_id: str | None = None
    asset_ids: list[str] = field(default_factory=list)
    source_page_start: int | None = None
    source_page_end: int | None = None
    table_id: str | None = None
    table_title: str | None = None
    table_rows: list[list[str]] = field(default_factory=list)
    table_header_rows: int = 0
    table_row_pages: list[int] = field(default_factory=list)
    # Raw LaTeX for formula blocks, preserved for faithful rendering and as
    # chunk metadata while `text` carries the retrieval-oriented cleaned form.
    latex: str | None = None


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    parent_chunk_id: str
    title: str
    text: str
    page_start: int
    page_end: int
    section_path: list[str]
    block_ids: list[str]
    article_id_raw: str | None = None
    article_id_normalized: str | None = None
    article_parent_id: str | None = None
    article_aliases: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    question_aliases: list[str] = field(default_factory=list)
    asset_ids: list[str] = field(default_factory=list)
    table_html: list[str] = field(default_factory=list)
    content_type: str = "text"
    table_ids: list[str] = field(default_factory=list)
    table_rows: list[list[str]] = field(default_factory=list)
    # Raw LaTeX for formula blocks in this chunk. The cleaned `text` drives
    # retrieval; the LaTeX is published as metadata for faithful rendering.
    formula_latex: list[str] = field(default_factory=list)


@dataclass(slots=True)
class AssetRecord:
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
class ParsedDocument:
    source_path: Path
    source_hash: str
    normalized_text_hash: str | None
    document_id: str
    pages: list[PageRecord]
    route: str
    repeated_margin_lines: list[str]
    removed_margin_line_count: int
    blocks: list[BlockRecord]
    chunks: list[ChunkRecord]
    qa: dict[str, Any]
    assets: list[AssetRecord] = field(default_factory=list)


@dataclass(slots=True)
class _HtmlCell:
    text: str
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False


class _TableHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[_HtmlCell]] = []
        self._row: list[_HtmlCell] | None = None
        self._cell: _HtmlCell | None = None
        self._cell_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized = tag.casefold()
        if normalized == "tr":
            self._row = []
        elif normalized in {"th", "td"} and self._row is not None:
            values = dict(attrs)
            try:
                rowspan = max(1, int(values.get("rowspan") or 1))
            except ValueError:
                rowspan = 1
            try:
                colspan = max(1, int(values.get("colspan") or 1))
            except ValueError:
                colspan = 1
            self._cell = _HtmlCell(
                text="",
                rowspan=rowspan,
                colspan=colspan,
                is_header=normalized == "th",
            )
            self._cell_parts = []
        elif normalized == "br" and self._cell is not None:
            self._cell_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in {"th", "td"} and self._cell is not None:
            self._cell.text = " ".join("".join(self._cell_parts).split())
            if self._row is not None:
                self._row.append(self._cell)
            self._cell = None
            self._cell_parts = []
        elif normalized == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _normalize_table_cell(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())


def _rectangular_rows(rows: list[list[str]]) -> list[list[str]]:
    normalized = [
        [_normalize_table_cell(cell) for cell in row]
        for row in rows
        if any(_normalize_table_cell(cell) for cell in row)
    ]
    width = max((len(row) for row in normalized), default=0)
    return [row + [""] * (width - len(row)) for row in normalized]


def collapse_repeated_full_width_header(
    rows: list[list[str]],
) -> list[list[str]]:
    """Restore a single merged header that OCR expanded across every column.

    Layout OCR sometimes emits a table title once per column when it misses a
    ``colspan``.  Only the first row is eligible: it must be at least four
    columns wide, fully populated, and contain the exact same text in every
    cell.  Body rows are deliberately untouched because repeated values there
    can be meaningful data.
    """

    rectangular = _rectangular_rows(rows)
    if not rectangular:
        return rectangular
    header = rectangular[0]
    if (
        len(header) >= 4
        and all(header)
        and len(set(header)) == 1
    ):
        rectangular[0] = [header[0], *([""] * (len(header) - 1))]
    return rectangular


def table_rows_from_html(
    table_html: str,
    *,
    collapse_repeated_header: bool = True,
) -> list[list[str]]:
    parser = _TableHtmlParser()
    try:
        parser.feed(table_html)
        parser.close()
    except (ValueError, TypeError):
        return []
    expanded: list[list[str]] = []
    active: dict[int, tuple[int, str]] = {}
    for raw_row in parser.rows:
        row_values: dict[int, str] = {}
        next_active: dict[int, tuple[int, str]] = {}
        for column, (remaining, value) in active.items():
            row_values[column] = value
            if remaining > 1:
                next_active[column] = (remaining - 1, value)
        column = 0
        for cell in raw_row:
            while column in row_values:
                column += 1
            for offset in range(cell.colspan):
                target = column + offset
                row_values[target] = cell.text
                if cell.rowspan > 1:
                    next_active[target] = (cell.rowspan - 1, cell.text)
            column += cell.colspan
        active = next_active
        if row_values:
            width = max(row_values) + 1
            expanded.append([row_values.get(index, "") for index in range(width)])
    rows = _rectangular_rows(expanded)
    return collapse_repeated_full_width_header(rows) if collapse_repeated_header else rows


def table_rows_from_text(text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.count("|") < 1:
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells if cell):
            continue
        if any(cells):
            rows.append(cells)
    return _rectangular_rows(rows)


def table_to_html(rows: list[list[str]], *, header_rows: int = 1) -> str:
    rectangular = _rectangular_rows(rows)
    if not rectangular:
        return ""
    body: list[str] = []
    for row_index, row in enumerate(rectangular):
        tag = "th" if row_index < header_rows else "td"
        if row_index < header_rows and row[0] and not any(row[1:]):
            cells = f'<{tag} colspan="{len(row)}">{html_lib.escape(row[0])}</{tag}>'
        else:
            cells = "".join(
                f"<{tag}>{html_lib.escape(cell)}</{tag}>"
                for cell in row
            )
        body.append(f"<tr>{cells}</tr>")
    return "<table>" + "".join(body) + "</table>"


def normalize_table_html(table_html: str, *, header_rows: int = 1) -> str:
    """Repair only the expanded-full-width-header pattern in OCR table HTML."""

    raw_rows = table_rows_from_html(table_html, collapse_repeated_header=False)
    collapsed_rows = collapse_repeated_full_width_header(raw_rows)
    if raw_rows == collapsed_rows:
        return table_html
    return table_to_html(collapsed_rows, header_rows=header_rows)


def table_to_text(
    rows: list[list[str]],
    *,
    title: str | None = None,
) -> str:
    lines = []
    for row in _rectangular_rows(rows):
        populated = [cell for cell in row if cell]
        lines.append(populated[0] if len(populated) == 1 else " | ".join(row))
    return "\n".join([*(title.strip() for title in [title] if title and title.strip()), *lines])


def _md_cell(value: str) -> str:
    """Render one Markdown table cell, escaping pipes and folding newlines."""
    return re.sub(r"\s+", " ", str(value)).replace("|", "\\|").strip()


def table_to_markdown(
    rows: list[list[str]],
    *,
    title: str | None = None,
) -> str:
    """Render a table as a Markdown grid so the original row/column layout is
    preserved and readable in Markdown viewers.

    Rows are rectangularised (merged cells fill trailing empty columns), the
    first row is treated as a header, and a ``| --- | --- |`` separator row
    follows so the grid renders as a real Markdown table.  Empty cells (from
    vertical merges) stay empty so the column alignment matches the source PDF.
    """
    rectangular = _rectangular_rows(rows)
    if not rectangular:
        return ""
    width = max(len(row) for row in rectangular)
    header = rectangular[0]
    body = rectangular[1:]

    def render(row: list[str]) -> str:
        cells = [_md_cell(cell) for cell in row]
        if len(cells) < width:
            cells += [""] * (width - len(cells))
        return "| " + " | ".join(cells) + " |"

    lines = [render(header), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(render(row) for row in body)
    if title and title.strip():
        normalized_title = unicodedata.normalize("NFKC", title.strip())
        return f"{normalized_title}\n" + "\n".join(lines)
    return "\n".join(lines)


def table_to_semantic_text(
    rows: list[list[str]],
    *,
    title: str | None = None,
    header_rows: int = 1,
) -> str:
    """Render a table as retrieval-oriented field/value statements.

    Markdown and HTML are useful for visual fidelity, but embedding models work
    more reliably when each value carries its column meaning in plain text.
    When OCR did not recover usable headers, fall back to neutral column
    numbers instead of inventing semantic labels.
    """

    rectangular = _rectangular_rows(rows)
    if not rectangular:
        return ""
    width = len(rectangular[0])
    header_count = min(max(1, header_rows), len(rectangular))
    normalized_title = _normalize_table_cell(title)
    column_labels: list[str] = []
    for column in range(width):
        parts: list[str] = []
        for row in rectangular[:header_count]:
            value = _normalize_table_cell(row[column])
            if not value or value == normalized_title or value in parts:
                continue
            parts.append(value)
        column_labels.append(" / ".join(parts))
    meaningful_labels = sum(bool(label) for label in column_labels)
    unique_labels = {label for label in column_labels if label}
    use_labels = (
        meaningful_labels >= min(2, width)
        and len(unique_labels) == meaningful_labels
    )

    lines = [normalized_title] if normalized_title else []
    if len(rectangular) == header_count:
        fields = [label for label in column_labels if label]
        if fields:
            lines.append("字段：" + "；".join(fields))
        else:
            values = [value for value in rectangular[0] if value]
            if values:
                lines.append("字段：" + "；".join(values))
        return "\n".join(lines)

    record_number = 0
    for row in rectangular[header_count:]:
        facts: list[str] = []
        column = 0
        while column < len(row):
            value = _normalize_table_cell(row[column])
            if not value:
                column += 1
                continue
            if use_labels:
                label = column_labels[column] or f"第{column + 1}列"
                facts.append(f"{label}={value}")
                column += 1
                continue
            end_column = column
            while (
                end_column + 1 < len(row)
                and _normalize_table_cell(row[end_column + 1]) == value
            ):
                end_column += 1
            label = (
                f"第{column + 1}列"
                if end_column == column
                else f"第{column + 1}-{end_column + 1}列"
            )
            facts.append(f"{label}={value}")
            column = end_column + 1
        if not facts:
            continue
        record_number += 1
        lines.append(f"记录{record_number}：" + "；".join(facts))
    return "\n".join(lines)


TABLE_TITLE_ROW_RE = re.compile(
    r"^表\s*(?:[A-Za-z]\s*[.\-]?\s*)?(?:\d+(?:[.\-]\d+)*|[一二三四五六七八九十]+)"
    r"(?:\s*[.、:：\-—])?\s*.*$",
    re.IGNORECASE,
)
TABLE_NOTE_ROW_RE = re.compile(
    r"^[（(]\s*(?:[A-Za-z]|\d+)\s*[）)]",
    re.IGNORECASE,
)
TABLE_REFERENCE_RE = re.compile(
    r"^表\s*((?:[A-Za-z]\s*[.\-]?\s*)?(?:\d+(?:[.\-]\d+)*|"
    r"[一二三四五六七八九十]+))",
    re.IGNORECASE,
)


def _table_title_from_row(row: list[str]) -> str:
    title = " ".join(cell for cell in row if cell).strip()
    return title if TABLE_TITLE_ROW_RE.match(title) else ""


def _table_reference(title: str) -> str:
    # NFKC first so full-width digits/letters (表１, 表２) and half-width forms
    # (表1, 表2) normalize to the same reference.
    normalized_title = unicodedata.normalize("NFKC", title.strip())
    match = TABLE_REFERENCE_RE.match(normalized_title)
    if not match:
        return ""
    # Normalize so "表 A.1", "表A1", "表 A.1（续）" all map to "表a1".
    raw = re.sub(r"\s+", "", match.group(1)).casefold()
    return "表" + raw.replace(".", "")


def _semantic_table_hints(hints: list[str]) -> list[str]:
    return [
        hint.strip()
        for hint in hints
        if _table_reference(hint)
        and not TABLE_TITLE_PROSE_RE.match(hint.strip())
        and not _TABLE_TITLE_CONTENT_RE.match(hint.strip())
    ]


def _looks_like_table_note(row: list[str]) -> bool:
    """Identify prose inserted between tables, not an in-table ``注：`` row."""
    text = " ".join(cell for cell in row if cell).strip()
    return len(text) >= 12 and bool(TABLE_NOTE_ROW_RE.match(text))


def _drop_empty_table_columns(rows: list[list[str]]) -> list[list[str]]:
    rectangular = _rectangular_rows(rows)
    if not rectangular:
        return []
    keep = [
        column
        for column in range(len(rectangular[0]))
        if any(row[column] for row in rectangular)
    ]
    return [[row[column] for column in keep] for row in rectangular]


def _repair_sparse_numeric_columns(rows: list[list[str]]) -> list[list[str]]:
    """Join OCR-split numeric columns without rewriting legitimate grids."""

    repaired = _drop_empty_table_columns(rows)
    while repaired and len(repaired[0]) >= 10:
        candidate: int | None = None
        best_score = 0
        row_limit = max(6, math.ceil(len(repaired) * 0.4))
        for column in range(1, len(repaired[0])):
            values = [row[column] for row in repaired if row[column]]
            if not values or len(values) > row_limit:
                continue
            if max(len(value) for value in values) > 2:
                continue
            numeric_joins = sum(
                bool(re.search(r"\d[.]?$", row[column - 1]))
                and bool(re.fullmatch(r"\d", row[column]))
                for row in repaired
                if row[column]
            )
            if numeric_joins >= 2 and numeric_joins > best_score:
                candidate = column
                best_score = numeric_joins
        if candidate is None:
            break
        for row in repaired:
            if row[candidate]:
                row[candidate - 1] = (
                    f"{row[candidate - 1]}{row[candidate]}"
                    if row[candidate - 1]
                    else row[candidate]
                )
            del row[candidate]
        repaired = _drop_empty_table_columns(repaired)
    return repaired


def _split_embedded_tables(
    rows: list[list[str]],
    *,
    title: str,
    page_number: int,
) -> tuple[list[tuple[str, list[list[str]], list[int]]], list[str]]:
    """Split several captioned tables accidentally returned as one page grid."""

    segments: list[tuple[str, list[list[str]], list[int]]] = []
    notes: list[str] = []
    current_title = title
    current_rows: list[list[str]] = []

    def finish_segment() -> None:
        nonlocal current_rows
        raw_row_count = len(current_rows)
        while len(current_rows) > 2 and _looks_like_table_note(current_rows[-1]):
            notes.insert(0, " ".join(cell for cell in current_rows.pop() if cell))
        normalized = _repair_sparse_numeric_columns(current_rows)
        if len(normalized) >= 2 and len(normalized[0]) >= 2:
            segments.append(
                (
                    current_title,
                    normalized,
                    [page_number] * len(normalized),
                )
            )
        elif (
            len(normalized) == 1
            and len(normalized[0]) >= 2
            and raw_row_count >= 1
        ):
            # Blank record form: the PDF grid carried a header row plus only
            # empty data rows (e.g. "采样检测项目 | 仪器设备 | 空气收集器 |
            # 采样体积 | 无人机飞行参数" with three blank rows). The empty
            # rows are dropped by rectangularisation, leaving one header row
            # that still carries all the field labels. Preserve it as a valid
            # single-row table instead of discarding the whole form. A bare
            # single-column appendix heading ("附录 A") is still rejected by
            # the width >= 2 guard.
            segments.append(
                (
                    current_title,
                    normalized,
                    [page_number] * len(normalized),
                )
            )
        current_rows = []

    for row in rows:
        embedded_title = _table_title_from_row(row)
        if embedded_title:
            if current_rows:
                finish_segment()
            current_title = embedded_title
            continue
        current_rows.append(row)
    if current_rows:
        finish_segment()
    return segments, notes


def _table_title_key(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    normalized = re.sub(r"[（(]?\s*续\s*[）)]?", "", normalized)
    return re.sub(r"[\s:：,，。;；\-—_]+", "", normalized)


def _row_fingerprint(row: list[str]) -> str:
    return "|".join(
        re.sub(r"[\s:：,，。;；\-—_]+", "", cell).casefold()
        for cell in row
    )


def _sequence_overlap(previous: list[str], current: list[str]) -> int:
    limit = min(len(previous), len(current))
    for size in range(limit, 0, -1):
        if previous[-size:] == current[:size]:
            return size
    return 0


def _rich_block_bbox_key(item: dict[str, Any]) -> tuple[float, float]:
    """Sort rich blocks by (top, left) so table blocks re-inserted by
    merge_cross_page_tables land back in their original reading position
    instead of being appended after later page content."""
    bbox = item.get("bbox")
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 2:
        return (float("inf"), float("inf"))
    try:
        return (float(bbox[1]), float(bbox[0]))
    except (TypeError, ValueError):
        return (float("inf"), float("inf"))


def merge_cross_page_tables(
    pages: list[PageRecord],
    document_id: str,
    *,
    table_candidate_pages: set[int] | None = None,
) -> dict[str, Any]:
    """Canonicalize table blocks and merge continuation fragments across pages.

    ``table_candidate_pages`` (optional) limits which pages' caption hints are
    treated as *expected* captions. A page that only carries a 表-caption-like
    label for a figure (e.g. a flow chart whose caption was mistyped "表 A.1
    …" while the body says "见图A.1") is not a real table candidate: its hint
    must not fail the caption-alignment gate.
    """

    pages_by_number = {page.page_number: page for page in pages}
    fragments: list[dict[str, Any]] = []
    invalid_pages: set[int] = set()
    html_leak_pages: set[int] = set()
    merged_cell_review_pages: set[int] = set()
    expected_caption_refs: dict[int, set[str]] = {
        page.page_number: (
            {
                reference
                for hint in _semantic_table_hints(page.table_hints)
                if (reference := _table_reference(hint))
            }
            if table_candidate_pages is None
            or page.page_number in table_candidate_pages
            else set()
        )
        for page in pages
    }
    actual_caption_refs: dict[int, set[str]] = {
        page.page_number: set() for page in pages
    }
    for page in pages:
        retained: list[dict[str, Any]] = []
        hint_titles = _semantic_table_hints(page.table_hints)
        used_hint_refs: set[str] = set()
        for ordinal, item in enumerate(page.rich_blocks):
            block_type = str(item.get("block_type", "")).casefold()
            text = str(item.get("text", "")).strip()
            if block_type == "table_coverage":
                # A cross-page table continuation page carries a table_coverage
                # marker. Its source table title satisfies the expected caption
                # reference on this page even though the table body lives on
                # the starting page.
                coverage_title = str(item.get("table_title") or "").strip()
                if reference := _table_reference(coverage_title):
                    actual_caption_refs[page.page_number].add(reference)
                retained.append(item)
                continue
            if block_type != "table":
                if "<table" in text.casefold() or "<td" in text.casefold():
                    html_leak_pages.add(page.page_number)
                retained.append(item)
                continue
            rows = item.get("table_rows")
            if not isinstance(rows, list) or not rows:
                table_html = str(item.get("table_html", "") or "")
                rows = (
                    table_rows_from_html(table_html)
                    if table_html
                    else table_rows_from_text(text)
                )
            raw_row_count = len(rows)
            rows = _rectangular_rows(
                [
                    [str(cell) for cell in row]
                    for row in rows
                    if isinstance(row, list)
                ]
            )
            header_only_form = (
                len(rows) == 1
                and len(rows[0]) >= 2
                and raw_row_count >= 2
            )
            # A single-row multi-column header is a real table too (e.g. the
            # "数据包格式" schematic in GB 46750-2025: 1 row x 8 columns of
            # 数据类型/版本号/数据长度/…/数据内容项N). Accept it when it is
            # wide enough; decorative cover frames / narrow single-column titles
            # are still rejected below.
            single_row_schematic = (
                len(rows) == 1
                and len(rows[0]) >= 4
                and all(str(cell).strip() for cell in rows[0])
            )
            if (len(rows) < 2 and not header_only_form and not single_row_schematic) or (
                rows and len(rows[0]) < 2
            ):
                invalid_pages.add(page.page_number)
                retained.append(item)
                continue
            title = str(item.get("table_title") or item.get("caption") or "").strip()
            explicit_reference = _table_reference(title)
            if explicit_reference:
                used_hint_refs.add(explicit_reference)
            elif hint_titles:
                # A single-row multi-column header with no explicit title is
                # usually a data-format schematic / illustration (e.g. the
                # "数据包格式" layout in GB 46750-2025 — one row of column
                # headers), not a numbered data table. Binding a page-level
                # "表N …" hint to it produces a confusing one-row "表N" chunk
                # (user sees "表1" with a single header row). Leave it titleless
                # so the real numbered table keeps the hint.
                is_titleless_schematic = (
                    len(rows) == 1
                    and len(rows[0]) >= 4
                    and all(str(cell).strip() for cell in rows[0])
                )
                if not is_titleless_schematic:
                    title = next(
                        (
                            hint
                            for hint in hint_titles
                            if _table_reference(hint) not in used_hint_refs
                        ),
                        "",
                    )
                    if title:
                        used_hint_refs.add(_table_reference(title))
            merged_cell_count = int(item.get("table_merged_cell_count") or 0)
            source_table_html = str(item.get("table_html") or "")
            if merged_cell_count and not re.search(
                r"\b(?:rowspan|colspan)\s*=",
                source_table_html,
                re.IGNORECASE,
            ):
                merged_cell_review_pages.add(page.page_number)
            asset_ids = [
                str(value)
                for value in item.get("asset_ids", [])
                if str(value).strip()
            ]
            asset_id = str(item.get("asset_id", "") or "").strip()
            if asset_id:
                asset_ids.append(asset_id)
            segments, notes = _split_embedded_tables(
                rows,
                title=title,
                page_number=page.page_number,
            )
            if not segments:
                invalid_pages.add(page.page_number)
                retained.append(item)
                continue
            for segment_title, _, _ in segments:
                if reference := _table_reference(segment_title):
                    actual_caption_refs[page.page_number].add(reference)
            retained.extend(
                {
                    "block_type": "paragraph",
                    "text": note,
                    "bbox": item.get("bbox"),
                    "confidence": 1.0,
                    "source_page_start": page.page_number,
                    "source_page_end": page.page_number,
                }
                for note in notes
            )
            fragments.extend(
                {
                    "page_start": page.page_number,
                    "page_end": page.page_number,
                    "title": segment_title,
                    "title_key": _table_title_key(segment_title),
                    "rows": segment_rows,
                    "row_pages": row_pages,
                    "header_rows": max(
                        1,
                        int(item.get("table_header_rows") or 1),
                    ),
                    "asset_ids": list(dict.fromkeys(asset_ids)),
                    "bbox": item.get("bbox"),
                    "ordinal": ordinal,
                    "merged_cell_count": merged_cell_count,
                    "table_html": (
                        source_table_html
                        if len(segments) == 1
                        else ""
                    ),
                }
                for segment_title, segment_rows, row_pages in segments
            )
        page.rich_blocks = retained

    series: list[dict[str, Any]] = []
    duplicate_rows_removed = 0
    fragments.sort(
        key=lambda fragment: (
            fragment["page_start"],
            fragment["bbox"][1] if fragment["bbox"] else 0.0,
            fragment["ordinal"],
        )
    )
    for fragment in fragments:
        current_fingerprints = [_row_fingerprint(row) for row in fragment["rows"]]
        previous = series[-1] if series else None
        merge = False
        overlap = 0
        if previous is not None and fragment["page_start"] == previous["page_end"] + 1:
            previous_fingerprints = [
                _row_fingerprint(row) for row in previous["rows"]
            ]
            same_width = len(previous["rows"][0]) == len(fragment["rows"][0])
            same_title = bool(
                fragment["title_key"]
                and fragment["title_key"] == previous["title_key"]
            )
            same_header = (
                same_width
                and current_fingerprints
                and previous_fingerprints
                and current_fingerprints[0] == previous_fingerprints[0]
            )
            continuation = "续" in fragment["title"] or "续" in previous["title"]
            common_prefix = 0
            if same_header:
                for old_row, new_row in zip(
                    previous_fingerprints[:6],
                    current_fingerprints[:6],
                    strict=False,
                ):
                    if old_row != new_row:
                        break
                    common_prefix += 1
            header_offset = common_prefix
            current_data = current_fingerprints[header_offset:]
            overlap = _sequence_overlap(previous_fingerprints, current_data)
            contained = bool(current_data) and any(
                previous_fingerprints[index : index + len(current_data)]
                == current_data
                for index in range(
                    max(0, len(previous_fingerprints) - len(current_data) + 1)
                )
            )
            merge = same_width and (
                same_title
                or continuation
                or (
                    same_header
                    and (
                        not previous["title_key"]
                        or not fragment["title_key"]
                    )
                )
                or overlap > 0
                or contained
                or (not fragment["title_key"] and not previous["title_key"])
            )
            if contained:
                overlap = len(current_data)
        elif previous is not None and fragment["page_start"] == previous["page_end"]:
            # Same-page adjacent table regions that share a blank/identical
            # title belong to the same blank record form (e.g. the upper field
            # block and the lower data grid of one horizontal 记录表 split by
            # pdfplumber's grid detection). Merge them into one table block so
            # the form is published as a single chunk instead of fragments.
            # The narrower region is right-padded to the wider grid's width so
            # the combined rows stay rectangular for Markdown/HTML rendering.
            previous_bbox = previous.get("bbox")
            fragment_bbox = fragment.get("bbox")
            adjacent = bool(
                previous_bbox
                and fragment_bbox
                and len(previous_bbox) >= 4
                and len(fragment_bbox) >= 4
                and abs(float(fragment_bbox[0]) - float(previous_bbox[0])) <= 6
                and abs(float(fragment_bbox[2]) - float(previous_bbox[2])) <= 6
                and 0 <= float(fragment_bbox[1]) - float(previous_bbox[3]) <= 40
            )
            same_blank_title = bool(
                not fragment["title_key"] and not previous["title_key"]
            )
            merge = adjacent and same_blank_title
            if merge:
                overlap = 0
                header_offset = 0
        if not merge:
            series.append(fragment)
            continue
        previous["page_end"] = fragment["page_end"]
        previous["asset_ids"] = list(
            dict.fromkeys([*previous["asset_ids"], *fragment["asset_ids"]])
        )
        previous["merged_cell_count"] = int(
            previous.get("merged_cell_count") or 0
        ) + int(fragment.get("merged_cell_count") or 0)
        previous["table_html"] = ""
        if not previous["title"] and fragment["title"]:
            previous["title"] = fragment["title"]
            previous["title_key"] = fragment["title_key"]
        repeated_header_rows = header_offset
        previous["header_rows"] = max(
            int(previous["header_rows"]),
            repeated_header_rows,
        )
        start = repeated_header_rows + overlap
        duplicate_rows_removed += start
        previous["rows"].extend(fragment["rows"][start:])
        previous["row_pages"].extend(fragment["row_pages"][start:])
        # Same-page merges join regions of different widths (e.g. a 2-column
        # field block above a 10-column data grid). Rectangularise right after
        # the merge so width checks against the NEXT fragment use the real
        # merged width; otherwise a later cross-page fragment with the same
        # width as the first (narrow) row would be wrongly treated as a
        # continuation of this form.
        if len({len(row) for row in previous["rows"]}) > 1:
            previous["rows"] = _rectangular_rows(previous["rows"])
            if len(previous["row_pages"]) < len(previous["rows"]):
                previous["row_pages"].extend(
                    [previous["page_start"]]
                    * (len(previous["rows"]) - len(previous["row_pages"]))
                )

    cross_page_tables = 0
    structured_pages: set[int] = set()
    duplicate_table_id_pages: set[int] = set()
    table_ids_seen: dict[str, int] = {}
    for table in series:
        page_start = int(table["page_start"])
        page_end = int(table["page_end"])
        if page_end > page_start:
            cross_page_tables += 1
        structured_pages.update(range(page_start, page_end + 1))
        table_id = stable_id(
            document_id,
            "table",
            page_start,
            page_end,
            table["title"],
            _row_fingerprint(table["rows"][0]),
            table["ordinal"],
            table["bbox"],
        )
        if table_id in table_ids_seen:
            duplicate_table_id_pages.update(
                {table_ids_seen[table_id], page_start}
            )
        else:
            table_ids_seen[table_id] = page_start
        rows = _rectangular_rows(table["rows"])
        table_header_rows = int(table["header_rows"])
        table_block = {
            "block_type": "table",
            "text": table_to_semantic_text(
                rows,
                title=table["title"],
                header_rows=table_header_rows,
            ),
            "bbox": table["bbox"],
            "confidence": 1.0,
            "table_html": table.get("table_html")
            or table_to_html(rows, header_rows=int(table["header_rows"])),
            "asset_id": next(iter(table["asset_ids"]), None),
            "asset_ids": table["asset_ids"],
            "source_page_start": page_start,
            "source_page_end": page_end,
            "table_id": table_id,
            "table_title": table["title"],
            "table_rows": rows,
            "table_header_rows": table_header_rows,
            "table_row_pages": table["row_pages"],
            "table_merged_cell_count": int(table.get("merged_cell_count") or 0),
        }
        start_page = pages_by_number.get(page_start)
        if start_page is None:
            invalid_pages.add(page_start)
            continue
        start_page.rich_blocks.append(table_block)
        for continuation_page_number in range(page_start + 1, page_end + 1):
            continuation_page = pages_by_number.get(continuation_page_number)
            if continuation_page is None:
                invalid_pages.add(continuation_page_number)
                continue
            continuation_page.rich_blocks.append(
                {
                    "block_type": "table_coverage",
                    "text": "",
                    "source_page_start": page_start,
                    "source_page_end": page_end,
                    "table_id": table_id,
                    "table_title": table["title"],
                }
            )

    # Re-inserting merged table blocks at the end of each page's rich_blocks
    # (above) can push them after later content that shares the page. Re-sort
    # every page's rich blocks by bbox position so section attribution in
    # split_blocks follows the true reading order.
    for page in pages:
        page.rich_blocks.sort(key=_rich_block_bbox_key)

    caption_mismatch_pages = sorted(
        page_number
        for page_number, expected in expected_caption_refs.items()
        if expected and not expected.issubset(actual_caption_refs[page_number])
    )
    return {
        "table_fragment_count": len(fragments),
        "structured_table_count": len(series),
        "cross_page_table_count": cross_page_tables,
        "structured_table_pages": sorted(structured_pages),
        "invalid_table_pages": sorted(invalid_pages),
        "table_html_leak_pages": sorted(html_leak_pages),
        "duplicate_table_rows_removed": duplicate_rows_removed,
        "table_caption_mismatch_pages": caption_mismatch_pages,
        "duplicate_table_id_pages": sorted(duplicate_table_id_pages),
        "merged_cell_review_pages": sorted(merged_cell_review_pages),
    }


def analyze_complex_table_fidelity(
    pages: list[PageRecord],
) -> list[dict[str, Any]]:
    """Identify OCR tables whose apparent structure is unsafe to publish.

    The rule uses document-independent layout signals: scanned input, remote
    OCR, a very wide grid, portrait-oriented table bounds, missing column
    headers, sparsity, and heavy use of spans.  It intentionally contains no
    standard number, page number, title text, or fixed expected column count.
    """

    diagnostics: list[dict[str, Any]] = []
    for page in pages:
        for table_ordinal, item in enumerate(
            (
                block
                for block in page.rich_blocks
                if str(block.get("block_type", "")).casefold() == "table"
            ),
            start=1,
        ):
            raw_rows = item.get("table_rows")
            rows = (
                _rectangular_rows(
                    [
                        [str(cell) for cell in row]
                        for row in raw_rows
                        if isinstance(row, list)
                    ]
                )
                if isinstance(raw_rows, list)
                else []
            )
            if not rows:
                table_html = str(item.get("table_html") or "")
                rows = table_rows_from_html(table_html) if table_html else []
            if not rows:
                continue
            row_count = len(rows)
            column_count = max(len(row) for row in rows)
            slot_count = row_count * column_count
            nonempty_count = sum(
                bool(_normalize_table_cell(cell))
                for row in rows
                for cell in row
            )
            nonempty_ratio = nonempty_count / slot_count if slot_count else 0.0
            first_row_nonempty = sum(
                bool(_normalize_table_cell(cell)) for cell in rows[0]
            )
            table_html = str(item.get("table_html") or "")
            span_attribute_count = len(
                re.findall(
                    r"\b(?:rowspan|colspan)\s*=",
                    table_html,
                    re.IGNORECASE,
                )
            )
            bbox = item.get("bbox")
            portrait_table_bounds = False
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                try:
                    box_width = abs(float(bbox[2]) - float(bbox[0]))
                    box_height = abs(float(bbox[3]) - float(bbox[1]))
                    portrait_table_bounds = box_height > box_width * 1.2
                except (TypeError, ValueError):
                    pass
            scanned_page = (
                page.page_type in {"scan", "scanned", "scanned_text"}
                or page.image_coverage >= 0.8
            )
            remote_ocr = page.route == "remote_ocr"
            reasons: list[str] = []
            risk_score = 0
            if scanned_page and remote_ocr and column_count >= 12:
                reasons.append("scanned_wide_grid")
                risk_score += 2
            if scanned_page and column_count >= 8 and portrait_table_bounds:
                reasons.append("suspected_rotated_table")
                risk_score += 3
            if column_count >= 8 and row_count >= 4 and first_row_nonempty <= 1:
                reasons.append("missing_column_header_schema")
                risk_score += 2
            if column_count >= 8 and slot_count >= 48 and nonempty_ratio < 0.4:
                reasons.append("sparse_cell_matrix")
                risk_score += 1
            if scanned_page and span_attribute_count >= max(8, row_count):
                reasons.append("span_heavy_ocr")
                risk_score += 1
            table_title = str(
                item.get("table_title") or item.get("caption") or ""
            ).strip()
            if re.match(
                r"^图\s*(?:[A-Za-z]\s*[.\-]?\s*)?\d+(?:[.\-]\d+)*\s+",
                table_title,
                re.IGNORECASE,
            ):
                reasons.append("figure_caption_on_table_block")
                risk_score += 4
            if not (scanned_page and remote_ocr and risk_score >= 4):
                continue
            diagnostics.append(
                {
                    "page_number": page.page_number,
                    "table_ordinal": table_ordinal,
                    "table_id": str(item.get("table_id") or ""),
                    "table_title": table_title,
                    "risk_level": "high",
                    "risk_score": risk_score,
                    "row_count": row_count,
                    "column_count": column_count,
                    "nonempty_ratio": round(nonempty_ratio, 4),
                    "span_attribute_count": span_attribute_count,
                    "reasons": reasons,
                    "recommended_route": "table_schema_review",
                }
            )
    return diagnostics


def compact_chars(text: str) -> str:
    return re.sub(r"\s+", "", text)


# Damaged CID fonts in some GB/T PDFs map Latin ASCII glyphs to CJK "fake
# glyph" codepoints in U+7280..U+72D8. The mapping is stable per damaged font,
# so the intended ASCII letters can be restored (e.g. "Technical", "multi",
# "GB", "RSD", "s(x)", "Mz", "borne" all decode from fake-glyph sequences).
FAKE_GLYPH_TO_ASCII: dict[str, str] = {
    chr(0x7283): "A", chr(0x7285): "B", chr(0x7286): "C", chr(0x7287): "D",
    chr(0x728C): "G", chr(0x728E): "H", chr(0x7290): "I", chr(0x7293): "K",
    chr(0x7294): "L", chr(0x7295): "M", chr(0x7296): "N", chr(0x7298): "P",
    chr(0x729A): "R", chr(0x729B): "S", chr(0x729C): "T", chr(0x729D): "U",
    chr(0x729E): "V", chr(0x72A1): "X", chr(0x72A2): "Y", chr(0x72A3): "Z",
    chr(0x72AA): "a", chr(0x72AB): "b", chr(0x72AE): "c", chr(0x72B1): "d",
    chr(0x72B2): "e", chr(0x72B3): "f", chr(0x72B5): "g", chr(0x72BA): "h",
    chr(0x72BB): "i", chr(0x72BC): "j", chr(0x72BE): "l", chr(0x72BF): "m",
    chr(0x72C0): "n", chr(0x72C5): "o", chr(0x72C6): "p", chr(0x72C9): "r",
    chr(0x72CA): "s", chr(0x72CB): "t", chr(0x72CC): "u", chr(0x72CF): "v",
    chr(0x72D1): "w", chr(0x72D3): "x", chr(0x72D4): "y", chr(0x72D5): "z",
    chr(0x72D8): "|",
    # Some PDFs map the clause-number dot (4.1.2) to a Private Use Area
    # codepoint instead of U+002E/U+FF0E (e.g. ４<U+1001B0>１<U+1001B0>２ for
    # "4.1.2"). Restore it to a real dot so heading_kind/_clause_number can
    # recognize the numbering (NY/T 4616-2025 pages 3-6).
    chr(0x1001B0): ".",
    # Math symbols rendered in an embedded SymbolMT font land in the Private
    # Use Area (Adobe Symbol encoding + 0xF000) instead of their real Unicode
    # (MH/T 4063.1-2026 appendix A/B grid-encoding formulas, pages 17-22).
    # Without remapping, "104°19′19.304″−102°" extracts as
    # "1041919.304−102".
    chr(0xF0B0): "°",  # degree
    chr(0xF0A2): "′",  # prime (arc minute)
    chr(0xF0A3): "″",  # double prime (arc second)
    chr(0xF0B2): "−",  # minus sign
    chr(0xF0B4): "×",  # multiply sign
    chr(0xF0EA): "⌈",  # left ceiling
    chr(0xF0FA): "⌉",  # right ceiling
    chr(0xF0EB): "⌊",  # left floor
    chr(0xF0FB): "⌋",  # right floor
    chr(0xF03E): ">",  # greater-than
    chr(0xF044): "Δ",  # capital Delta (Symbol 0x44)
}


def repair_fake_glyphs(text: str) -> str:
    """Restore Latin letters encoded as damaged-font fake CJK glyphs.

    Applies NFKC first so full-width digits/symbols (３％, （）) become
    half-width, then maps fake-glyph codepoints back to their ASCII letters.
    """
    if not text:
        return text
    nfkc = unicodedata.normalize("NFKC", text)
    return "".join(FAKE_GLYPH_TO_ASCII.get(ch, ch) for ch in nfkc)


def is_glyph_name_garbage(text: str) -> bool:
    """Detect PDF text layers that expose PostScript glyph names instead of Unicode.

    Some subsetted fonts map every glyph to an internal name like ``/G21``,
    ``/G22``, ... pypdf then extracts strings such as
    ``/G21/G22/G23`` rather than real text. The strings are valid Unicode but
    semantically garbage and must be replaced by OCR.
    """

    compact = compact_chars(text)
    total = len(compact)
    if total < 20:
        return False

    glyph_matches = list(GLYPH_NAME_RE.finditer(compact))
    if not glyph_matches:
        return False

    glyph_name_chars = sum(len(match.group(0)) for match in glyph_matches)
    glyph_name_ratio = glyph_name_chars / total

    # A very high glyph-name density is an unambiguous corruption signal even
    # when the damaged font also maps some glyphs to CJK look-alike characters
    # (e.g. U+72AA "犪") that fall inside the CJK range.
    if glyph_name_ratio > 0.3:
        return True

    if glyph_name_ratio <= 0.05:
        return False

    counter = Counter(compact)
    top_ratio = counter.most_common(1)[0][1] / total if counter else 0.0
    if top_ratio <= 0.18:
        return False

    cjk = sum(1 for ch in compact if "一" <= ch <= "鿿")
    if cjk / total >= 0.05:
        return False

    return len(glyph_matches) >= 5


def is_fake_cjk_garbage(text: str) -> bool:
    """Detect pages dominated by CJK Extension-A fake glyphs.

    Some damaged fonts map Latin letters and digits to CJK Extension-A
    codepoints (U+7280..U+72CF) such as ``犲`` (U+72B2) instead of real
    glyphs, WITHOUT exposing ``/GXX`` PostScript names. ``is_glyph_name_garbage``
    misses these because there are no ``/G`` tokens. This detector flags a
    page when the majority of its CJK characters fall in the Extension-A fake
    range.
    """
    compact = compact_chars(text)
    if not compact or len(compact) < 20:
        return False
    total = len(compact)
    cjk_total = sum(1 for ch in compact if 0x4E00 <= ord(ch) <= 0x9FFF)
    if cjk_total / total < 0.3:
        return False
    fake = sum(1 for ch in compact if 0x7280 <= ord(ch) <= 0x72CF)
    return fake / cjk_total > 0.5


def has_fake_glyph_corruption(text: str) -> bool:
    """Detect pages whose Latin text/formulas were mapped to CJK fake glyphs.

    Beyond whole-page corruption, some documents mix normal Chinese body text
    with damaged English terms (continuous runs of fake glyphs) or formula
    symbols (scattered fake glyphs). Either signal routes the page to OCR.
    """
    compact = compact_chars(text)
    if not compact or len(compact) < 20:
        return False
    total = len(compact)
    if total == 0:
        return False
    fake = sum(1 for ch in compact if 0x7280 <= ord(ch) <= 0x72CF)
    fake_ratio = fake / total

    # Continuous run of fake glyphs ⇒ a damaged English word (e.g. 犅狅狉狀犲).
    run = 0
    max_run = 0
    for ch in compact:
        if 0x7280 <= ord(ch) <= 0x72CF:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0

    return fake_ratio > 0.03 or max_run >= 4


_TOUNICODE_BFCHAR_RE = re.compile(r"<([0-9A-Fa-f]{2,6})> <([0-9A-Fa-f]{2,8})>")


def has_tounicode_cid_conflict(pdf_page: Any) -> bool:
    """Detect pages whose embedded font ToUnicode maps are self-contradictory.

    Some documents (GJB 2489A-2023) subset a font per text run with an
    Identity-H encoding, and the *same CID* is mapped to different Unicode
    codepoints inside one ToUnicode CMap (or across the page's fonts). The
    rendered page is correct (the glyph outlines are fine) but the extracted
    text layer is garbage — the same visual word appears as several different
    CJK strings ("五号宋体" vs "号宋体" vs "匹号宋体").

    A self-contradictory ToUnicode map (one CID → two different codepoints) is
    an unambiguous corruption signal: well-formed embedded fonts map each CID
    to exactly one Unicode value. When detected, the page must be routed to
    OCR — no fixed character-remap table can repair a per-run unstable mapping.

    A *single* conflict is not enough: many well-formed PDFs have one or two
    CID→space quirks in an otherwise clean CMap (CCAR-23-R3 p164 etc. — one
    SimSun CID maps to both a CJK char and a space). Require a material
    conflict density (>=2 conflicts and >=0.5% of bfchar entries) so only
    genuinely corrupted text layers (GJB 2489A) are flagged.
    """
    try:
        resources = pdf_page.get("/Resources", {})
        fonts = resources.get("/Font", {})
        if not fonts:
            return False
        total_conflicts = 0
        total_entries = 0
        for font_ref in fonts.values():
            try:
                font = font_ref.get_object()
            except (AttributeError, TypeError, ValueError):
                continue
            if font.get("/Subtype") != "/Type0":
                continue
            to_unicode = font.get("/ToUnicode")
            if to_unicode is None:
                continue
            try:
                stream = to_unicode.get_object().get_data()
            except (AttributeError, TypeError, ValueError):
                continue
            try:
                cmap_text = stream.decode("latin-1", errors="replace")
            except (UnicodeDecodeError, AttributeError):
                continue
            cid_to_unicode: dict[str, str] = {}
            entries = _TOUNICODE_BFCHAR_RE.findall(cmap_text)
            total_entries += len(entries)
            for cid, uni in entries:
                if cid in cid_to_unicode and cid_to_unicode[cid] != uni:
                    total_conflicts += 1
                cid_to_unicode[cid] = uni
        if total_entries == 0:
            return False
        ratio = total_conflicts / total_entries
        return total_conflicts >= 2 and ratio >= 0.005
    except (AttributeError, TypeError, ValueError):
        pass
    return False


# A damaged OCR text layer embedded under a scan (e.g. CCAR-27-R2) shows the
# same "digit . digit" pattern broken into "digit 。 digit" — OCR renders the
# decimal point / dot separator with a full-width CJK period. A genuine CJK
# period never sits between two digits, so a sustained density of this pattern
# is an unambiguous low-quality-OCR signal: the text is not reliable enough to
# build clause structure from, and the page should be re-OCR'd from the scan.
_OCR_PERIOD_BETWEEN_DIGITS_RE = re.compile(r"\d[。．]\s*\d")
# A secondary low-quality-OCR signal: a CJK body page whose every character is
# isolated by spaces ("如 果 申 请 人 证 实") is characteristic of an OCR text
# layer laid over a scan, where each glyph was recognized and spaced apart.
# Genuine typeset Chinese never inserts inter-character spaces. This signal is
# only trusted when the page also carries a large scan image (image_ratio),
# because some clean PDFs do space CJK for layout (table cells, cover pages).
# The ratio counts "汉字 汉字" adjacent pairs — the exact OCR artifact. Header
# spaces in clean PDFs ("C C A R  2 3 R 3") are not between two CJK glyphs and
# so do not inflate it.
_OCR_CJK_PAIR_SPACE_RE = re.compile(r"[一-鿿] [一-鿿]")
_OCR_CJK_SPACE_RATIO = 0.15
_OCR_CJK_SPACE_MIN_N = 100
_OCR_CJK_SPACE_IMAGE_MIN_RATIO = 0.5


def has_low_quality_ocr_text(
    text: str,
    *,
    image_ratio: float = 0.0,
) -> bool:
    """Detect pages whose text layer is an embedded low-quality OCR scan.

    CCAR-27-R2 and similar "scan + hidden OCR text layer" PDFs extract
    readable-looking Unicode that fails every existing corruption detector
    (no PostScript glyph names, no fake CJK, no ToUnicode conflict) yet is
    full of OCR substitutions: ``27。 673`` instead of ``27.673``, ``0。 335``
    instead of ``0.335``, ``zO17`` instead of ``2017``. Such text must not be
    trusted for clause structure; route the page through OCR so the visual
    model reads the scan itself.

    Two complementary signals:

    - ``digit 。 digit`` patterns. A damaged scan's OCR layer breaks the decimal
      point / clause-number dot into a full-width period dozens of times per
      page (CCAR-27 peaks at 20 per page); a clean document has at most 1-2
      incidental full-width periods next to digits (e.g. an annex id "C23．1").
      Require BOTH an absolute count (>=3) and a density per 10k chars (>=8):
      a short clean page with one "C23．1" would otherwise trip a pure density
      threshold.
    - For pages without that pattern (e.g. a paragraph with no decimals), a
      CJK page whose text is word-spaced AND carries a large scan image
      (image_ratio >= 0.5) is also an OCR text layer. Clean figure pages also
      have large images but their text is not word-spaced, so they stay put.
    """
    compact = compact_chars(text)
    if not compact or len(compact) < 60:
        return False
    matches = _OCR_PERIOD_BETWEEN_DIGITS_RE.findall(compact)
    if len(matches) >= 3:
        density = len(matches) / len(compact) * 10000
        # CCAR-27 sits at ~73-89 per 10k; clean documents are < 3. A threshold
        # of 8 leaves ample margin against incidental half/full-width mixes.
        if density >= 8.0:
            return True
    if image_ratio < _OCR_CJK_SPACE_IMAGE_MIN_RATIO:
        return False
    cjk = sum(1 for ch in compact if "一" <= ch <= "鿿")
    # A scan's OCR layer is overwhelmingly CJK body text, but a page can be a
    # mostly-English/table scan (HIRF frequency tables, "10kHz-2MHz 50 50").
    # Require some CJK presence (>=10%) — enough to rule out pure-English
    # documents (SAE AS9102C) while still catching table-heavy scans.
    if cjk / len(compact) < 0.1:
        return False
    if len(compact) < _OCR_CJK_SPACE_MIN_N:
        return False
    # Count "汉字 汉字" adjacent pairs — the inter-character space an OCR text
    # layer leaves between every CJK glyph. Clean pages keep CJK glyphs
    # unspaced (header "C C A R 23 R 3" never matches this pattern).
    cjk_pair_spaces = len(_OCR_CJK_PAIR_SPACE_RE.findall(text))
    if cjk_pair_spaces / cjk < _OCR_CJK_SPACE_RATIO:
        return False
    return True


WATERMARK_IMAGE_MAX_EDGE = 400


def is_watermark_only_page(pdf_page: Any, raw_text: str) -> bool:
    """Detect pages that contain only a small watermark image and no real text.

    Some documents place a small logo or "internal use" watermark on otherwise
    blank pages (e.g. chapter dividers). pypdf extracts no text from these
    pages, but the vector parser may still classify them as ``figure`` because
    a tiny image object is present. They should be excluded from indexing.
    """

    if compact_chars(raw_text):
        return False
    resources = pdf_page.get("/Resources", {})
    xobjects = resources.get("/XObject", {})
    if not xobjects:
        return False

    image_count = 0
    large_image_count = 0
    for xobj in xobjects.values():
        try:
            obj = xobj.get_object()
            if obj.get("/Subtype") != "/Image":
                continue
            image_count += 1
            width = int(obj.get("/Width", 0) or 0)
            height = int(obj.get("/Height", 0) or 0)
            if width > WATERMARK_IMAGE_MAX_EDGE or height > WATERMARK_IMAGE_MAX_EDGE:
                large_image_count += 1
        except Exception:
            continue

    return image_count > 0 and large_image_count == 0


def repair_invalid_unicode(text: str) -> tuple[str, int]:
    """Replace lone UTF-16 surrogates while preserving valid surrogate pairs.

    Some PDF text layers expose mathematical glyphs as lone high surrogates.
    Such strings can be manipulated in Python but cannot be encoded as UTF-8,
    which previously caused the entire ingestion job to fail before OCR could
    repair the affected page.
    """

    repaired: list[str] = []
    replacement_count = 0
    index = 0
    while index < len(text):
        codepoint = ord(text[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(text):
                low = ord(text[index + 1])
                if 0xDC00 <= low <= 0xDFFF:
                    repaired.append(
                        chr(0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00))
                    )
                    index += 2
                    continue
            repaired.append("\ufffd")
            replacement_count += 1
        elif 0xDC00 <= codepoint <= 0xDFFF:
            repaired.append("\ufffd")
            replacement_count += 1
        else:
            repaired.append(text[index])
        index += 1
    return "".join(repaired), replacement_count


def stable_id(*parts: object, size: int = 20) -> str:
    material = "\x1f".join(repair_invalid_unicode(str(part))[0] for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:size]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_line(text: str) -> str:
    text, _ = repair_invalid_unicode(text)
    text = unicodedata.normalize("NFKC", text).strip()
    text = re.sub(
        r"(?<![章节条款项目])(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])",
        "",
        text,
    )
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,，。；;:：!?！？、])", r"\1", text)
    return text


def detect_table_hints(text: str) -> list[str]:
    """Detect table titles even when PDF text extraction splits the title."""

    lines = [normalize_line(line) for line in text.splitlines() if line.strip()]
    hints: list[str] = []
    for index, line in enumerate(lines):
        if TABLE_TITLE_RE.match(line):
            if not TABLE_TITLE_PROSE_RE.match(line) and not _TABLE_TITLE_CONTENT_RE.match(
                line
            ):
                hints.append(line)
            continue
        if line != "表":
            continue
        for width in (2, 3):
            candidate = " ".join(lines[index : index + width])
            if TABLE_TITLE_RE.match(candidate) and not TABLE_TITLE_PROSE_RE.match(
                candidate
            ) and not _TABLE_TITLE_CONTENT_RE.match(candidate):
                hints.append(candidate)
                break
    return list(dict.fromkeys(hints))


def normalized_margin_key(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def filter_repeated_margin_rich_blocks(
    page: PageRecord,
    repeated_margin_keys: set[str],
) -> int:
    """Remove header/footer paragraphs that vector parsing re-extracted.

    Vector blocks come from ``pdfplumber.extract_words`` which is a separate
    text source from ``page.raw_text``. Repeating headers such as
    ``DB3205/T 1145—2024`` therefore bypass ``clean_page`` and survive into
    chunks as noise. This filters rich_blocks that sit in the top margin band
    (top y < 12% of a typical A4 page height) and whose normalized key is a
    known repeated margin line.
    """
    if not repeated_margin_keys or not page.rich_blocks:
        return 0
    removed = 0
    kept: list[dict[str, Any]] = []
    # A4 height is 842pt; use a fixed band rather than block-relative math.
    top_margin_max = 842 * 0.12
    for item in page.rich_blocks:
        text = str(item.get("text", "")).strip()
        bbox = item.get("bbox")
        block_type = str(item.get("block_type", "")).casefold()
        key = normalized_margin_key(text) if text else ""
        y_top: float | None = None
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 2:
            try:
                y_top = float(bbox[1])
            except (TypeError, ValueError):
                y_top = None
        # Repeating headers in the top margin band (or blocks without a bbox
        # whose text matches a known repeated margin line) are noise.
        is_fake_gb_header = bool(
            text and FAKE_GB_HEADER_RE.match(text) and len(text) <= 40
        )
        if (
            text
            and block_type == "paragraph"
            and (
                is_fake_gb_header
                or key in repeated_margin_keys
                and (
                    y_top is not None
                    and y_top <= top_margin_max
                    or y_top is None
                    and len(text) <= 40
                )
            )
        ):
            removed += 1
            continue
        kept.append(item)
    page.rich_blocks = kept
    return removed


def heading_kind(text: str) -> str | None:
    for kind, pattern in HEADING_PATTERNS:
        if pattern.search(text):
            if kind == "annex" and _looks_like_annex_prose(text):
                # "附录 B 给出了…" / "附录A 对…" are body sentences that merely
                # mention an annex, not annex headings. Treating them as annex
                # clears the clause stack and hijacks every later section_path
                # (GB/T 19001-2016 p8). Only a label + optional nature marker
                # is a heading; anything more is prose.
                continue
            return kind
    reference = match_article_heading(text)
    if reference is not None:
        return "article" if normalize_line(text).startswith("第") else "clause"
    return None


# Body sentences that mention an annex (e.g. "附录 B 给出了…", "附录A 对…的
# 要求", "见附录 A") are not annex headings. A genuine annex heading is the
# label plus optional nature marker "(资料性)" / "(规范性)" and nothing else —
# or a short title like "附录A 术语和定义" where the trailing phrase is a pure
# noun title, not a verb/subject sentence. We detect prose by common annex
# sentence verbs and demonstrative fragments that never appear in a heading.
_ANNEX_PROSE_RE = re.compile(
    r"(?:给出|给了|列出|规定|提供|为|包含|用|要求|以作|其对|详见|关于|对应|附注|"
    r"应符合|的内容|对\s*$|所述|指明|要求于|见附录|请参见)"
)


def _looks_like_annex_prose(text: str) -> bool:
    label = re.match(
        r"^\s*(?:附录|附件)\s*[A-Za-z0-9一二三四五六七八九十百零〇]+",
        text,
    )
    if not label:
        return False
    rest = text[label.end():]
    compact = re.sub(r"\s+", "", rest)
    if not compact:
        return False
    # A nature marker "(资料性附录)"/"(规范性)" alone is still a heading.
    if re.fullmatch(r"[（(](?:资料性|规范性)[^（()]*[）)]", compact):
        return False
    return bool(_ANNEX_PROSE_RE.search(compact))


# Leading clause number of a numbered heading, e.g. "4 一般要求" -> "4",
# "4.8.1 低气压" -> "4.8.1". Used to derive the clause hierarchy level.
_CLAUSE_NUMBER_RE = re.compile(r"^\s*(\d+(?:\.\d+)*)\s+")


def _clause_number(text: str) -> str | None:
    match = _CLAUSE_NUMBER_RE.match(text)
    return match.group(1) if match else None


# Titles of unnumbered sub-clauses whose number is missing in the PDF layout,
# e.g. "基本要求" under "5 高水效小麦品种鉴定" is really "5.1 基本要求". They are
# pure-CJK noun phrases: short, no terminal punctuation, no English (term
# definitions carry their English name), no list markers, and not table
# captions / figure captions / formula guides.
_UNNUMBERED_SUBHEADING_MIN = 2
_UNNUMBERED_SUBHEADING_MAX = 20


def _looks_like_unnumbered_subheading(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    if not compact or not (
        _UNNUMBERED_SUBHEADING_MIN <= len(compact) <= _UNNUMBERED_SUBHEADING_MAX
    ):
        return False
    # Must be a pure CJK noun phrase: no ASCII letters, digits, or punctuation.
    if re.search(r"[A-Za-z0-9]", compact):
        return False
    if re.search(r"[。！？；;:：，,、（）()〔〕《》「」【】\-\—–]", compact):
        return False
    # Not a numbered/list item.
    if re.match(r"^[（(][一二三四五六七八九十0-9a-zA-Z]+[）)]", compact):
        return False
    if re.match(r"^表\s*[0-9A-Za-z一二三四五六七八九十]", compact):
        return False
    # Figure captions ("图1 xxx" / "图A.1 xxx") are not sub-clause headings.
    if re.match(r"^图\s*[0-9A-Za-z一二三四五六七八九十]+", compact):
        return False
    # Formula guides / common non-heading fragments.
    if compact in {"式中", "其中", "注", "说明", "来源"}:
        return False
    if re.match(r"^(注|式中|其中|来源)", compact):
        return False
    # Body-sentence fragments from a damaged text layer must not become
    # sub-clause headings. pypdf sometimes splits a prose sentence into short
    # lines ("附录 B 给出了 SAC/TC 151 制定…" -> "附录" + "B" + "给出了" +
    # "制定…") and each fragment would otherwise be numbered as a fake
    # sub-clause (0.4.3 给出了 / 0.4.4 应关系见 — GB/T 19001-2016 p8).
    # Real unnumbered sub-clause titles are noun phrases; they do not start
    # with a sentence verb / conjunction / demonstrative. Note the prefixes are
    # multi-char phrases, not bare single chars like "可"/"见"/"应", which
    # appear in legitimate noun titles ("可见光相机…", "应用…", "应急…").
    if re.match(
        r"^(?:给出|列出|规定|提供|满足|符合|采用|使用|用于|作为|包括|包含|"
        r"按照|根据|本标准|本附录|见图|见表|附录|附件|可见附录|详见|应符合|"
        r"应关系见|对应关系见|之间的关系|之间的对应|内容之间的)",
        compact,
    ):
        return False
    return bool(re.search(r"[一-鿿]", compact))


# A term-definition heading: "术语名 英文术语名" (e.g. "太阳反射波段多光谱
# 遥感 multispectral remote sensing"). In GB/T 1.1 terminology clauses each
# term is numbered 3.1/3.2/… and the term name + English name sit on their own
# line, followed by the definition. They must be kept on their own line and
# numbered like sub-clause headings.
_TERM_HEADING_RE = re.compile(
    r"^[一-鿿]{2,20}\s+[A-Za-z][A-Za-z0-9\s;（）()，,&\-]{1,60}$"
)


def _looks_like_term_heading(text: str) -> bool:
    line = re.sub(r"\s+", " ", text).strip()
    if not line or len(line) > 80:
        return False
    if not _TERM_HEADING_RE.match(line):
        return False
    # Must not be a sentence: no terminal punctuation, no leading label.
    if re.search(r"[。！？；;]$", line):
        return False
    return True


def is_probable_toc_page(text: str) -> bool:
    lines = [normalize_line(line) for line in text.splitlines() if line.strip()]
    if any(TOC_TITLE_RE.fullmatch(line) for line in lines[:8]):
        return True
    dotted_entries = sum(
        bool(
            TOC_DOTTED_ENTRY_RE.search(line)
            or TOC_DOTTED_NOISY_RE.search(line)
            or TOC_DOTTED_AFTER_ENTRY_RE.match(line)
            or TOC_DOTTED_NO_PAGE_RE.match(line)
        )
        for line in lines
    )
    numbered_entries = sum(bool(TOC_NUMBERED_ENTRY_RE.match(line)) for line in lines)
    return dotted_entries >= 3 or numbered_entries >= 6


def compute_toc_score(
    text: str,
    *,
    page_number: int,
    total_pages: int,
    layout_features: dict[str, Any] | None = None,
) -> float:
    """Second-layer TOC scoring that combines layout/statistical features
    instead of relying on a single regex. Returns a score in [0, 1].

    Features:
    * title_score: page begins with 目录/目次/Contents (or nearby)
    * leader_ratio: fraction of lines with leader dots / ellipsis
    * ending_digit_ratio: fraction of lines ending with a page number
    * numbered_hierarchy: mixture of ``1``, ``1.1``, ``1.1.1`` numbering
    * position_score: page lies in the front third of the document
    * alignment: repeated page-number ending suggests right-aligned column
    """
    lines = [normalize_line(line) for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    title_score = 0.0
    for line in lines[:8]:
        if TOC_TITLE_RE.fullmatch(line):
            title_score = 1.0
            break
    else:
        title_score = 0.5 if any("目录" in line or "目次" in line for line in lines[:8]) else 0.0

    leader_matches = sum(
        1
        for line in lines
        if TOC_DOTTED_ENTRY_RE.search(line)
        or TOC_DOTTED_NOISY_RE.search(line)
        or TOC_DOTTED_AFTER_ENTRY_RE.match(line)
        or TOC_DOTTED_NO_PAGE_RE.match(line)
    )
    leader_ratio = leader_matches / len(lines)

    ending_digit = sum(1 for line in lines if re.search(r"\d+\s*$", line))
    ending_digit_ratio = ending_digit / len(lines)

    # Numbered hierarchy: presence of both top-level and dotted numbering.
    numbered = sum(1 for line in lines if TOC_NUMBERED_ENTRY_RE.match(line))
    has_subnumbering = any(
        re.match(r"^\s*\d+(\.\d+)+\s+\S", line) for line in lines
    )
    hierarchy_score = min(
        1.0, (numbered / max(1, len(lines))) * 2.0 + (0.5 if has_subnumbering else 0.0)
    )

    position_ratio = (page_number - 1) / max(1, total_pages)
    position_score = 1.0 if position_ratio <= 0.3 else max(0.0, 1.0 - (position_ratio - 0.3) / 0.5)

    # Font feature: TOC pages often have uniform, smaller font sizes.
    font_score = 0.0
    if layout_features:
        font_sizes = layout_features.get("font_sizes") or []
        if len(font_sizes) >= 2:
            spread = max(font_sizes) - min(font_sizes)
            font_score = 0.6 if spread <= 4.0 else 0.2

    return round(
        title_score * 0.20
        + leader_ratio * 0.35
        + ending_digit_ratio * 0.15
        + hierarchy_score * 0.10
        + position_score * 0.10
        + font_score * 0.10,
        4,
    )


def detect_repeated_margin_lines(pages: list[PageRecord]) -> set[str]:
    candidates: Counter[str] = Counter()
    eligible_pages = 0
    for page in pages:
        lines = [line.strip() for line in page.raw_text.splitlines() if line.strip()]
        if not lines:
            continue
        eligible_pages += 1
        margin_lines = lines[:3] + lines[-3:]
        page_keys = {
            normalized_margin_key(line)
            for line in margin_lines
            if 1 < len(normalized_margin_key(line)) <= 80
        }
        candidates.update(page_keys)

    minimum_hits = max(3, math.ceil(eligible_pages * REPEATED_MARGIN_FRACTION))
    return {key for key, count in candidates.items() if count >= minimum_hits}


def clean_page(page: PageRecord, repeated_margin_keys: set[str]) -> tuple[str, int]:
    kept: list[str] = []
    removed = 0
    raw_lines = [line for line in page.raw_text.splitlines() if line.strip()]
    for index, raw_line in enumerate(raw_lines):
        line = normalize_line(raw_line)
        # Damaged-font fake glyphs (犌犅 → GB, 附录犃 → 附录A) survive into the
        # cleaned text when a page was routed to native/vector content. Restore
        # them here so the final chunks carry readable Latin letters.
        line = repair_fake_glyphs(line)
        key = normalized_margin_key(line)
        is_margin = index < 3 or index >= max(0, len(raw_lines) - 3)
        if PAGE_NUMBER_RE.fullmatch(line):
            removed += 1
            continue
        if is_margin and ANNEX_DECOR_RE.fullmatch(line):
            removed += 1
            continue
        if is_margin and key in repeated_margin_keys:
            removed += 1
            continue
        if line:
            kept.append(line)

    paragraphs: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            joined = buffer[0]
            for part in buffer[1:]:
                separator = (
                    " "
                    if re.search(r"[A-Za-z0-9)]$", joined)
                    and re.match(r"^[A-Za-z0-9(]", part)
                    else ""
                )
                joined += separator + part
            paragraphs.append(joined)
            buffer.clear()

    annex_title_pending = False
    for line in kept:
        kind = heading_kind(line)
        if kind or TABLE_TITLE_RE.match(line):
            flush()
            paragraphs.append(line)
            annex_title_pending = False
            continue
        if ANNEX_MARKER_RE.fullmatch(line):
            # Annex nature markers ("(资料性)"/"(规范性)") are part of the
            # annex heading. Keeping them on their own line prevents them from
            # gluing the annex title to the following body text (e.g.
            # "(资料性)产量预测模型MultimodalNet使用方法遥感产量预测模型…").
            # The line that follows the marker is the annex title and is also
            # kept on its own line.
            flush()
            paragraphs.append(line)
            annex_title_pending = True
            continue
        if annex_title_pending:
            # Annex title line (no punctuation): keep it on its own line so it
            # does not glue to the following body paragraph.
            annex_title_pending = False
            flush()
            paragraphs.append(line)
            continue
        if _looks_like_unnumbered_subheading(line):
            # An unnumbered sub-clause heading (e.g. "无人机影像数据获取及
            # 预处理" that is really "5.1 …") must stay on its own line so the
            # later split_blocks pass can number it. Otherwise clean_page glues
            # it to the following body paragraph.
            flush()
            paragraphs.append(line)
            continue
        if _looks_like_term_heading(line):
            # A term-definition heading ("术语名 英文术语名") in a terminology
            # clause (3 术语和定义) must stay on its own line so split_blocks
            # can number it 3.1/3.2/…. Otherwise it is glued to the definition.
            flush()
            paragraphs.append(line)
            continue
        if LIST_ITEM_RE.match(line):
            flush()
            buffer.append(line)
            if re.search(r"[。！？；;:]$", line):
                flush()
            continue
        buffer.append(line)
        if re.search(r"[。！？；;:]$", line):
            flush()
    flush()
    return "\n".join(paragraphs), removed


def split_blocks(
    document_id: str,
    pages: list[PageRecord],
) -> list[BlockRecord]:
    blocks: list[BlockRecord] = []
    chapter: str | None = None
    annex: str | None = None
    article_reference = None
    # Clause number hierarchy (e.g. "4 一般要求" -> ["4 一般要求"],
    # then "4.1 功能" -> ["4 一般要求", "4.1 功能"], then
    # "4.8.1 低气压" -> ["4 一般要求", "4.8 环境适应性", "4.8.1 低气压"]).
    # Keeps the top-level clause ("4 一般要求") as the parent of its
    # sub-clauses so section_path carries the full hierarchy instead of
    # only the current clause.
    clause_stack: list[str] = []
    # Unnumbered sub-clause headings (e.g. "试验要求" that is really "5.3
    # 试验要求") buffer here so a deeper numbered clause like "5.3.1" can fill
    # its missing "5.3" parent from the PDF text.
    pending_unnumbered: list[str] = []
    # Parent stack for unnumbered sub-headings: the full clause prefix rooted
    # at the last real numbered clause, so consecutive unnumbered sub-headings
    # under the SAME parent number 5.1/5.2/5.3 (or 5.6.1/5.6.2) instead of
    # nesting under each other (5.1.1/5.1.2). Reset whenever a numbered
    # clause/annex/chapter switches the active clause.
    unnumbered_parent_stack: list[str] = []

    for page in pages:
        if not page.indexable:
            continue
        has_structured_table = any(
            item.get("block_type") in {"table", "table_coverage"}
            for item in page.rich_blocks
        )
        native_parts = [
            part.strip() for part in page.cleaned_text.splitlines() if part.strip()
        ]
        # Remote OCR / vector pages carry their content as rich_blocks (with
        # bbox and normalized punctuation). cleaned_text is then a duplicate
        # rendering of the same content (half-width punctuation) that would be
        # emitted twice. Only fall back to cleaned_text when the page has no
        # substantive rich content blocks.
        substantive_rich = any(
            str(item.get("block_type", "")).casefold() in {"paragraph", "table"}
            and str(item.get("text", "")).strip()
            for item in page.rich_blocks
        )
        if has_structured_table or substantive_rich:
            # Rich blocks already preserve the reading order and cell values.
            # Keeping bbox-less native text would duplicate content and
            # misclassify cell values such as "23.2100" as clause headings.
            native_parts = []
        # When a page already has structured tables, the whole-page figure_page
        # block (rendered page image + VLM caption) duplicates the table
        # content and should not be emitted as a standalone block.
        table_titles: set[str] = set()
        figure_captions: set[str] = set()
        for item in page.rich_blocks:
            block_type = str(item.get("block_type", "")).casefold()
            if block_type == "table":
                title = str(
                    item.get("table_title")
                    or (str(item.get("text", "")).split("\n")[0] if item.get("text") else "")
                ).strip()
                if title:
                    table_titles.add(_table_title_key(title))
            elif block_type == "figure":
                caption = str(item.get("text", "")).split("\n")[0].strip()
                if caption:
                    figure_captions.add(_table_title_key(caption))
        page_rich_blocks: list[dict[str, Any]] = []
        for item in page.rich_blocks:
            block_type = str(item.get("block_type", "")).casefold()
            if has_structured_table and block_type == "figure":
                bbox = item.get("bbox")
                is_full_page = (
                    isinstance(bbox, (list, tuple))
                    and len(bbox) >= 4
                    and abs(float(bbox[2]) - float(bbox[0])) > 400
                    and abs(float(bbox[3]) - float(bbox[1])) > 500
                )
                if is_full_page:
                    continue
            if (
                block_type == "paragraph"
                and table_titles
                and _table_title_key(str(item.get("text", ""))) in table_titles
            ):
                # The paragraph is a caption line that already exists as a
                # structured table title; skip to avoid duplication.
                continue
            if (
                block_type == "paragraph"
                and figure_captions
                and _table_title_key(str(item.get("text", ""))) in figure_captions
            ):
                # The paragraph duplicates a figure block's caption (e.g. the
                # caption line re-extracted by _line_blocks on a figure page).
                # Keep only the figure block which carries the image asset.
                continue
            # Blank record forms (记录表) carry short placeholder paragraphs
            # that vector extraction re-emits as standalone blocks: a
            # right-margin standard number (DB4401/T), an isolated dash, or the
            # unfilled "第页，共页" page-count placeholder. They add no retrieval
            # value and would fragment the form chunk with noise.
            if (
                block_type == "paragraph"
                and has_structured_table
            ):
                block_text = str(item.get("text", "")).strip()
                if (
                    block_text
                    and (
                        FORM_STANDARD_NUMBER_RE.fullmatch(block_text)
                        or FORM_DASH_ONLY_RE.fullmatch(block_text)
                        or FORM_PAGE_PLACEHOLDER_RE.fullmatch(block_text)
                    )
                ):
                    continue
            # A cross-page table continuation title ("表1 数据包内容续") and its
            # trailing "( )" residue are left over after merge_cross_page_tables
            # folded the continuation body into the merged table block on the
            # starting page. They carry no table data on their own and would
            # publish as an empty "表1 数据包内容续( )" noise chunk.
            if (
                block_type == "paragraph"
                and table_titles
            ):
                block_text = str(item.get("text", "")).strip()
                is_continuation_title = bool(
                    block_text
                    and "续" in block_text
                    and TABLE_TITLE_ROW_RE.match(block_text)
                )
                is_empty_residue = bool(
                    block_text
                    and len(block_text) <= 4
                    and re.fullmatch(r"[（(]?\s*[）)]?", block_text)
                )
                if (
                    is_continuation_title
                    or is_empty_residue
                ):
                    # Only drop when this page has any structured table already
                    # (the continuation body lives in the merged table).
                    continue
            page_rich_blocks.append(item)
        page_parts: list[dict[str, Any]] = [
            {
                "block_type": None,
                "text": text,
                "bbox": None,
                "confidence": None,
                "table_html": None,
                "asset_id": None,
                "asset_ids": [],
                "source_page_start": None,
                "source_page_end": None,
                "table_id": None,
                "table_title": None,
                "table_rows": [],
                "table_header_rows": 0,
                "table_row_pages": [],
            }
            for text in native_parts
        ]
        page_parts.extend(page_rich_blocks)
        # Pages whose rich blocks carry real bboxes (vector/figure/table pages)
        # are sorted by (top, left) so figure and paragraph blocks interleave in
        # true reading order. Native text parts (no bbox) keep their relative
        # order before the rich blocks.
        if any(
            isinstance(item.get("bbox"), (list, tuple)) and len(item["bbox"]) >= 2
            for item in page_rich_blocks
        ):
            page_parts.sort(
                key=lambda item: (
                    (
                        float(item["bbox"][1])
                        if isinstance(item.get("bbox"), (list, tuple))
                        and len(item["bbox"]) >= 2
                        else 0.0
                    ),
                    (
                        float(item["bbox"][0])
                        if isinstance(item.get("bbox"), (list, tuple))
                        and len(item["bbox"]) >= 2
                        else 0.0
                    ),
                )
            )
        for local_index, item in enumerate(page_parts):
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            kind = heading_kind(text)
            if kind == "chapter":
                chapter, annex = text, None
                article_reference = None
                clause_stack = []
                pending_unnumbered = []
                unnumbered_parent_stack = []
            elif kind in {"article", "clause"}:
                # Numbered clauses form a hierarchy by their number prefix:
                # "5" is the parent of "5.1.1", which is the sibling of
                # "5.1.2". Rebuild the clause stack so only true ancestors
                # (number prefixes) of the current heading survive. A bare
                # sibling like 5.1.2 must replace 5.1.1, not nest under it.
                heading_number = _clause_number(text)
                if heading_number is not None:
                    kept: list[str] = []
                    for existing in clause_stack:
                        existing_number = _clause_number(existing)
                        if existing_number is None:
                            continue
                        if (
                            heading_number.startswith(existing_number + ".")
                            or heading_number == existing_number
                        ):
                            kept.append(existing)
                    clause_stack = kept + [text]
                else:
                    clause_stack = [text]
                pending_unnumbered = []
                # Unnumbered sub-headings below this clause are siblings whose
                # parent is this clause (e.g. 5.6 -> 5.6.1/5.6.2), not the
                # top-level clause (5 -> 5.1). Keep the ancestor prefix so a
                # deeper sub-heading can restore the correct parent.
                unnumbered_parent_stack = clause_stack[:-1] + [text]
                article_reference = match_article_heading(text)
            elif kind == "annex":
                annex, chapter = text, None
                article_reference = None
                clause_stack = []
                pending_unnumbered = []
                unnumbered_parent_stack = []
            elif (
                not kind
                and clause_stack
                and _looks_like_term_heading(text)
            ):
                # A term-definition heading ("术语名 英文术语名") inside a
                # terminology clause ("3 术语和定义") is really "3.1 太阳反射
                # 波段多光谱遥感 multispectral remote sensing". Number it in
                # order under the active clause, like an unnumbered sub-heading.
                # The parent is the last real numbered clause
                # (unnumbered_parent_stack), so consecutive terms under
                # "3 术语和定义" number 3.1/3.2/3.3 (not 3.1.1/3.1.2).
                base_clause = (
                    unnumbered_parent_stack[-1]
                    if unnumbered_parent_stack
                    else clause_stack[-1]
                )
                base_number = _clause_number(base_clause)
                if base_number and "." not in base_number:
                    sub_index = len(pending_unnumbered) + 1
                    text = f"{base_number}.{sub_index} {text}"
                    pending_unnumbered.append(text)
                    clause_stack = unnumbered_parent_stack + [text]
                    kind = "clause"
                    article_reference = match_article_heading(text)
            elif (
                not kind
                and clause_stack
                and _looks_like_unnumbered_subheading(text)
            ):
                # The PDF omits the number on sub-clause headings such as
                # "试验要求" under "5 高水效小麦品种鉴定" (really "5.3
                # 试验要求"). Number them in order under the active clause:
                # 基本要求->5.1, 种子要求->5.2, 试验要求->5.3. The synthesized
                # heading becomes the active clause so its body and any deeper
                # numbered clause (5.3.1) nest under it. Only promote when the
                # next block is a substantive body paragraph — otherwise a table
                # cell (e.g. "大豆苗情等级") or isolated fragment is wrongly
                # treated as a heading.
                next_text = (
                    str(page_parts[local_index + 1].get("text", "")).strip()
                    if local_index + 1 < len(page_parts)
                    else ""
                )
                # A real sub-heading is followed by a body paragraph. A table
                # cell (e.g. header "大豆苗情等级" followed by
                # "好(1级)较好(2级)…遥感长势指数区间 0.81~1.00…") is NOT a body
                # paragraph: it carries parenthesised "N级" level markers and
                # numeric ranges. Reject those so table content is not promoted.
                next_compact = re.sub(r"\s+", "", next_text)
                next_is_table_row = bool(
                    re.search(r"[（(]\s*\d+\s*级", next_compact)
                    or re.search(r"\d[\d.]*\s*[~～\-—]\s*\d", next_compact)
                )
                next_is_body = (
                    len(next_compact) >= 12 and not next_is_table_row
                )
                # A sub-clause heading may be the last line of a page, with its
                # body paragraph starting on the next page (e.g. "种子质量" at
                # the bottom of page 6, followed by "种子调拨按GB/T 15776…" on
                # page 7). In that case there is no next block on this page, so
                # treat it as a heading when it is the final block of the page.
                is_last_block_on_page = local_index + 1 >= len(page_parts)
                # A candidate sub-heading may be a pure section header whose
                # "body" consists of deeper numbered clauses (e.g. "试验要求"
                # followed by "5.3.1 田间布置" — 试验要求 is really "5.3 试验
                # 要求"). Promote it when the next block is a deeper numbered
                # clause under the same base number.
                # The parent is the last real numbered clause
                # (unnumbered_parent_stack), not the top-level clause: a PDF
                # omits the number on sub-headings under a numbered sub-clause
                # too, e.g. "可见光相机的指标要求如下" under "5.6 任务载荷" is
                # really "5.6.1 可见光相机的指标要求如下" — numbering it off the
                # top-level "5" would wrongly produce "5.1". For a top-level
                # clause ("5 高水效小麦品种鉴定") the parent is that clause, so
                # "基本要求" still infers "5.1" and siblings infer 5.2/5.3.
                base_clause = (
                    unnumbered_parent_stack[-1]
                    if unnumbered_parent_stack
                    else clause_stack[-1]
                )
                base_number = _clause_number(base_clause)
                next_number = _clause_number(next_text)
                next_is_deeper_clause = bool(
                    next_number
                    and base_number
                    and next_number.startswith(base_number + ".")
                )
                if (
                    base_number
                    and (
                        next_is_body
                        or next_is_deeper_clause
                        or is_last_block_on_page
                    )
                ):
                    sub_index = len(pending_unnumbered) + 1
                    text = f"{base_number}.{sub_index} {text}"
                    pending_unnumbered.append(text)
                    clause_stack = unnumbered_parent_stack + [text]
                    kind = "clause"
                    article_reference = match_article_heading(text)

            section_path = [
                value
                for value in (annex, chapter, *clause_stack)
                if value
            ]
            block_type = (
                str(item.get("block_type") or "")
                or kind
                or ("table_hint" if TABLE_TITLE_RE.match(text) else "paragraph")
            )
            block_id = stable_id(document_id, page.page_number, local_index, text)
            blocks.append(
                BlockRecord(
                    block_id=block_id,
                    block_type=block_type,
                    text=text,
                    page_number=page.page_number,
                    section_path=section_path,
                    article_id_raw=(
                        article_reference.raw if article_reference is not None else None
                    ),
                    article_id_normalized=(
                        article_reference.normalized_id
                        if article_reference is not None
                        else None
                    ),
                    article_parent_id=(
                        article_reference.parent_id if article_reference is not None else None
                    ),
                    article_aliases=(
                        list(article_reference.aliases)
                        if article_reference is not None
                        else []
                    ),
                    bbox=item.get("bbox"),
                    confidence=item.get("confidence"),
                    table_html=item.get("table_html"),
                    asset_id=item.get("asset_id"),
                    asset_ids=[
                        str(value)
                        for value in item.get("asset_ids", [])
                        if str(value).strip()
                    ],
                    source_page_start=item.get("source_page_start"),
                    source_page_end=item.get("source_page_end"),
                    table_id=item.get("table_id"),
                    table_title=item.get("table_title"),
                    table_rows=[
                        [str(cell) for cell in row]
                        for row in (item.get("table_rows") or [])
                        if isinstance(row, list)
                    ],
                    table_header_rows=int(item.get("table_header_rows") or 0),
                    table_row_pages=[
                        int(value)
                        for value in item.get("table_row_pages", [])
                        if isinstance(value, int)
                    ],
                    latex=(
                        str(item.get("latex", "")).strip()
                        if str(item.get("latex", "")).strip()
                        else None
                    ),
                )
            )
    return blocks


def is_flattened_table_noise(text: str) -> bool:
    """Detect numeric streams created when a two-dimensional table is flattened."""

    tokens = text.split()
    if len(tokens) < 12 or len(text) < 80:
        return False
    numeric = sum(
        bool(re.fullmatch(r"[-+±]?\d+(?:\.\d+)?(?:%|℃|°C)?", token))
        for token in tokens
    )
    return numeric / len(tokens) >= 0.65


def split_long_text(text: str, limit: int, overlap: int) -> Iterable[str]:
    if len(text) <= limit:
        yield text
        return
    start = 0
    while start < len(text):
        end = min(len(text), start + limit)
        if end < len(text):
            punctuation = max(
                text.rfind(mark, start, end) for mark in (*"。；！？", "\n")
            )
            if punctuation > start + limit // 2:
                end = punctuation + 1
        yield text[start:end]
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def split_table_rows(text: str, limit: int) -> Iterable[str]:
    """Split a structured table at row boundaries and repeat its column header."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return
    if len(text) <= limit:
        yield text
        return
    header_count = min(4, len(lines))
    header = lines[:header_count]
    if len("\n".join(header)) > limit:
        yield from split_long_text(text, limit, 0)
        return
    pending = list(header)
    for row in lines[header_count:]:
        projected = "\n".join([*pending, row])
        if len(projected) <= limit:
            pending.append(row)
            continue
        if len(pending) > header_count:
            yield "\n".join(pending)
        pending = [*header, row]
        if len("\n".join(pending)) > limit:
            yield from split_long_text(row, limit, 0)
            pending = list(header)
    if len(pending) > header_count:
        yield "\n".join(pending)


def split_structured_table_block(
    block: BlockRecord,
    limit: int,
) -> Iterable[BlockRecord]:
    rows = _rectangular_rows(block.table_rows)
    if not rows:
        yield block
        return
    header_count = min(max(1, block.table_header_rows), len(rows))
    header = rows[:header_count]
    row_pages = list(block.table_row_pages)
    if len(row_pages) < len(rows):
        row_pages.extend(
            [block.source_page_start or block.page_number]
            * (len(rows) - len(row_pages))
        )
    if len(block.text) <= limit:
        yield replace(
            block,
            table_rows=rows,
            table_html=block.table_html
            or table_to_html(rows, header_rows=header_count),
        )
        return

    pending_rows = list(header)
    pending_pages = list(row_pages[:header_count])

    def materialize(
        part_rows: list[list[str]],
        part_pages: list[int],
    ) -> BlockRecord:
        data_pages = part_pages[header_count:] or part_pages
        page_start = min(data_pages, default=block.source_page_start or block.page_number)
        page_end = max(data_pages, default=block.source_page_end or block.page_number)
        return replace(
            block,
            text=table_to_semantic_text(
                part_rows,
                title=block.table_title,
                header_rows=header_count,
            ),
            table_html=table_to_html(part_rows, header_rows=header_count),
            table_rows=part_rows,
            table_row_pages=part_pages,
            source_page_start=page_start,
            source_page_end=page_end,
        )

    for row, page_number in zip(
        rows[header_count:],
        row_pages[header_count:],
        strict=False,
    ):
        projected = [*pending_rows, row]
        projected_text = table_to_semantic_text(
            projected,
            title=block.table_title,
            header_rows=header_count,
        )
        if len(projected_text) <= limit or len(pending_rows) == header_count:
            pending_rows.append(row)
            pending_pages.append(page_number)
            continue
        yield materialize(pending_rows, pending_pages)
        pending_rows = [*header, row]
        pending_pages = [*row_pages[:header_count], page_number]
    if len(pending_rows) > header_count:
        yield materialize(pending_rows, pending_pages)


def chunk_from_blocks(
    *,
    document_id: str,
    parent_chunk_id: str,
    title: str,
    section_key: tuple[str, ...],
    blocks: list[BlockRecord],
    ordinal: int,
) -> ChunkRecord:
    article_id_raw = next(
        (block.article_id_raw for block in blocks if block.article_id_raw),
        None,
    )
    article_id_normalized = next(
        (block.article_id_normalized for block in blocks if block.article_id_normalized),
        None,
    )
    article_parent_id = next(
        (block.article_parent_id for block in blocks if block.article_parent_id),
        None,
    )
    article_aliases = list(
        dict.fromkeys(alias for block in blocks for alias in block.article_aliases)
    )
    asset_ids = list(
        dict.fromkeys(
            asset_id
            for block in blocks
            for asset_id in [*block.asset_ids, *([block.asset_id] if block.asset_id else [])]
            if asset_id
        )
    )
    table_html = list(
        dict.fromkeys(
            block.table_html
            for block in blocks
            if block.block_type == "table" and block.table_html
        )
    )
    table_ids = list(
        dict.fromkeys(
            block.table_id
            for block in blocks
            if block.block_type == "table" and block.table_id
        )
    )
    table_rows = [
        row
        for block in blocks
        if block.block_type == "table"
        for row in block.table_rows
    ]
    formula_latex = list(
        dict.fromkeys(
            block.latex
            for block in blocks
            if block.block_type == "formula" and block.latex
        )
    )
    body = "\n".join(block.text for block in blocks)
    context = list(dict.fromkeys(value for value in section_key if value))
    # A standalone one-line clause is both the section label and its complete
    # body.  Keep it once in the published text rather than duplicating it as
    # both context and body.
    body_is_last_context = bool(
        context
        and len(blocks) == 1
        and normalize_line(body) == normalize_line(context[-1])
    )
    # When the first content block repeats the last context element (a heading
    # block whose text equals the section label), drop the duplicated heading
    # from the body so the title does not appear twice in the chunk.
    if context and not body_is_last_context:
        first_block_text = str(blocks[0].text).strip() if blocks else ""
        if first_block_text and normalize_line(first_block_text) == normalize_line(
            context[-1]
        ):
            body = "\n".join(block.text for block in blocks[1:])
    text = (
        "\n".join(context)
        if body_is_last_context
        else "\n".join([*context, body])
        if context
        else body
    )
    article = match_article_heading(title)
    return ChunkRecord(
        chunk_id=stable_id(
            document_id,
            "chunk",
            parent_chunk_id,
            ordinal,
            blocks[0].block_id,
        ),
        parent_chunk_id=parent_chunk_id,
        title=title,
        text=text,
        page_start=min(
            block.source_page_start or block.page_number
            for block in blocks
        ),
        page_end=max(
            block.source_page_end or block.page_number
            for block in blocks
        ),
        section_path=list(section_key),
        block_ids=[block.block_id for block in blocks],
        article_id_raw=article_id_raw,
        article_id_normalized=article_id_normalized,
        article_parent_id=article_parent_id,
        article_aliases=article_aliases,
        keywords=extract_keywords(title, section_key, article_aliases),
        question_aliases=build_question_aliases(article, title),
        asset_ids=asset_ids,
        table_html=table_html,
        content_type=(
            "table"
            if table_ids
            else (
                "figure"
                if any(
                    block.block_type == "figure" for block in blocks
                )
                else "text"
            )
        ),
        table_ids=table_ids,
        table_rows=table_rows,
        formula_latex=formula_latex,
    )


def clause_heading_contains_body(block: BlockRecord) -> bool:
    """Return True when an article/clause heading also contains substantive text."""

    if block.block_type not in {"article", "clause"}:
        return False
    reference = match_article_heading(block.text)
    if reference is None:
        return False
    normalized = normalize_line(block.text)
    prefix = normalize_line(reference.raw)
    if not normalized.startswith(prefix):
        return False
    remainder = normalized[len(prefix) :].strip(" \t-—:：")
    compact_remainder = compact_chars(remainder)
    if not compact_remainder:
        return False
    return bool(
        CLAUSE_SENTENCE_END_RE.search(remainder)
        or CLAUSE_ASSERTION_RE.search(remainder)
        or len(compact_remainder) >= 24
    )


def find_unpublished_clause_blocks(
    blocks: list[BlockRecord],
    chunks: list[ChunkRecord],
) -> list[BlockRecord]:
    """Find substantive one-line clauses that disappeared before publication."""

    chunk_text = "\n".join(normalize_line(chunk.text) for chunk in chunks)
    return [
        block
        for block in blocks
        if clause_heading_contains_body(block)
        and normalize_line(block.text) not in chunk_text
    ]


def build_chunks(document_id: str, blocks: list[BlockRecord]) -> list[ChunkRecord]:
    groups: list[tuple[tuple[str, ...], str | None, list[BlockRecord]]] = []
    for block in blocks:
        section_key = tuple(block.section_path)
        article_key = block.article_id_normalized
        if (
            not groups
            or groups[-1][0] != section_key
            or groups[-1][1] != article_key
        ):
            groups.append((section_key, article_key, []))
        groups[-1][2].append(block)

    chunks: list[ChunkRecord] = []
    heading_types = {kind for kind, _ in HEADING_PATTERNS}
    heading_types.update({"article", "clause", "document_heading"})
    for section_key, article_key, group in groups:
        content_blocks = [
            block for block in group if block.block_type not in heading_types
        ]
        if not content_blocks:
            # Some standards put an entire normative requirement on the same
            # line as its clause number.  Such a block has no following
            # paragraph, so treating every clause as a title used to discard it
            # here.  Preserve only substantive inline clauses; short structural
            # headings continue to be omitted as standalone chunks.
            content_blocks = [
                block for block in group if clause_heading_contains_body(block)
            ]
            if not content_blocks:
                continue
        parent_chunk_id = stable_id(
            document_id,
            "parent",
            *section_key,
            article_key or "no-article",
        )
        title = section_key[-1] if section_key else "文档前言"
        context_chars = sum(len(value) + 1 for value in section_key)
        body_max_chars = max(200, MAX_CHUNK_CHARS - context_chars)
        body_target_chars = max(160, TARGET_CHUNK_CHARS - context_chars)
        pending: list[BlockRecord] = []
        pending_chars = 0

        for block in content_blocks:
            if block.block_type == "table":
                if pending:
                    chunks.append(
                        chunk_from_blocks(
                            document_id=document_id,
                            parent_chunk_id=parent_chunk_id,
                            title=title,
                            section_key=section_key,
                            blocks=pending,
                            ordinal=len(chunks),
                        )
                    )
                    pending = []
                    pending_chars = 0
                table_parts = (
                    split_structured_table_block(block, body_max_chars)
                    if block.table_rows
                    else [
                        replace(block, text=part)
                        for part in split_table_rows(block.text, body_max_chars)
                    ]
                )
                for part_block in table_parts:
                    chunks.append(
                        chunk_from_blocks(
                            document_id=document_id,
                            parent_chunk_id=parent_chunk_id,
                            title=title,
                            section_key=section_key,
                            blocks=[part_block],
                            ordinal=len(chunks),
                        )
                    )
                continue
            if block.block_type == "figure":
                # Each figure block carries its own cropped image asset (and,
                # when the local VLM is enabled, a structured description).
                # Emit it as a standalone chunk so every illustration on a page
                # is published with its own image_base64 + positions instead of
                # only the first one being attached to a merged text chunk.
                if pending:
                    chunks.append(
                        chunk_from_blocks(
                            document_id=document_id,
                            parent_chunk_id=parent_chunk_id,
                            title=title,
                            section_key=section_key,
                            blocks=pending,
                            ordinal=len(chunks),
                        )
                    )
                    pending = []
                    pending_chars = 0
                chunks.append(
                    chunk_from_blocks(
                        document_id=document_id,
                        parent_chunk_id=parent_chunk_id,
                        title=title,
                        section_key=section_key,
                        blocks=[block],
                        ordinal=len(chunks),
                    )
                )
                continue
            if block.block_type == "formula":
                # A formula is an indivisible unit: splitting it mid-formula
                # would break its mathematical meaning. Emit it whole, bound to
                # whatever context ("式中"/前置说明) is already pending, even
                # if the combined text runs slightly over the char budget.
                if pending:
                    pending.append(block)
                    pending_chars += len(block.text) + 1
                    chunks.append(
                        chunk_from_blocks(
                            document_id=document_id,
                            parent_chunk_id=parent_chunk_id,
                            title=title,
                            section_key=section_key,
                            blocks=pending,
                            ordinal=len(chunks),
                        )
                    )
                    pending = []
                    pending_chars = 0
                else:
                    chunks.append(
                        chunk_from_blocks(
                            document_id=document_id,
                            parent_chunk_id=parent_chunk_id,
                            title=title,
                            section_key=section_key,
                            blocks=[block],
                            ordinal=len(chunks),
                        )
                    )
                continue
            if len(block.text) > body_max_chars:
                if pending:
                    chunks.append(
                        chunk_from_blocks(
                            document_id=document_id,
                            parent_chunk_id=parent_chunk_id,
                            title=title,
                            section_key=section_key,
                            blocks=pending,
                            ordinal=len(chunks),
                        )
                    )
                    pending = []
                    pending_chars = 0
                parts = split_long_text(
                    block.text,
                    body_max_chars,
                    min(CHUNK_OVERLAP_CHARS, body_max_chars // 4),
                )
                for part_index, part in enumerate(parts):
                    part_block = replace(
                        block,
                        block_id=f"{block.block_id}:{part_index}",
                        text=part,
                    )
                    chunks.append(
                        chunk_from_blocks(
                            document_id=document_id,
                            parent_chunk_id=parent_chunk_id,
                            title=title,
                            section_key=section_key,
                            blocks=[part_block],
                            ordinal=len(chunks),
                        )
                    )
                continue
            projected = pending_chars + len(block.text) + (1 if pending else 0)
            if pending and projected > body_max_chars:
                chunks.append(
                    chunk_from_blocks(
                        document_id=document_id,
                        parent_chunk_id=parent_chunk_id,
                        title=title,
                        section_key=section_key,
                        blocks=pending,
                        ordinal=len(chunks),
                    )
                )
                pending = []
                pending_chars = 0
            pending.append(block)
            pending_chars += len(block.text) + (1 if len(pending) > 1 else 0)
            if pending_chars >= body_target_chars:
                chunks.append(
                    chunk_from_blocks(
                        document_id=document_id,
                        parent_chunk_id=parent_chunk_id,
                        title=title,
                        section_key=section_key,
                        blocks=pending,
                        ordinal=len(chunks),
                    )
                )
                pending = []
                pending_chars = 0
        if pending:
            chunks.append(
                chunk_from_blocks(
                    document_id=document_id,
                    parent_chunk_id=parent_chunk_id,
                    title=title,
                    section_key=section_key,
                    blocks=pending,
                    ordinal=len(chunks),
                )
            )
    return chunks


def extract_layout_text(page: Any) -> str | None:
    try:
        text = page.extract_text(extraction_mode="layout") or ""
        return repair_invalid_unicode(text)[0] or None
    except (TypeError, ValueError):
        return None


def determine_document_route(
    text_coverage: float,
    has_tables: bool,
    text_layer_warning: bool = False,
) -> str:
    if text_coverage == 0:
        return "ocr_required"
    if text_coverage < 0.98 or text_layer_warning:
        return "hybrid_review"
    if has_tables:
        return "layout_review"
    return "native_text"


def extract_page_layout_features(pdf_page: Any, raw_text: str) -> dict[str, Any]:
    """Collect first-layer layout features from a pypdf page without deleting
    any content. These are consumed later by feature-based scoring (TOC,
    figure, formula) and visual-review triage."""
    features: dict[str, Any] = {
        "page_width": 0.0,
        "page_height": 0.0,
        "image_count": 0,
        "image_ratio": 0.0,
        "image_sizes": [],
        "char_count": 0,
        "line_count": 0,
        "font_sizes": [],
        "dominant_font_size": 0.0,
    }
    try:
        mediabox = pdf_page.mediabox
        features["page_width"] = round(float(mediabox.width), 2)
        features["page_height"] = round(float(mediabox.height), 2)
    except (AttributeError, TypeError, ValueError):
        pass
    page_area = max(1.0, features["page_width"] * features["page_height"])

    image_area = 0.0
    image_sizes: list[dict[str, float]] = []
    try:
        images = pdf_page.images
        for image in images:
            w = h = 0.0
            # pypdf's ImageFile carries a decoded PIL image; its size reflects
            # the source pixels even when the PDF image dict lacks explicit
            # width/height. Older code read image.width/height, which ImageFile
            # does not expose — every image silently became 0x0 and image_ratio
            # stayed 0 for scanned PDFs (CCAR-27-R2).
            try:
                pil_image = getattr(image, "image", None)
                if pil_image is not None:
                    w, h = float(pil_image.size[0]), float(pil_image.size[1])
                else:
                    w = float(image.width)
                    h = float(image.height)
            except (AttributeError, TypeError, ValueError):
                w = h = 0.0
            image_sizes.append({"width": w, "height": h})
            image_area += w * h
    except (AttributeError, TypeError, ValueError):
        pass
    except Exception:
        # Reading image dimensions decompresses the image stream. Some PDFs
        # embed streams larger than pypdf's ZLIB_MAX_OUTPUT_LENGTH (75 MB),
        # which raises pypdf LimitReachedError (a PyPdfError, not a
        # ValueError). Treat an oversized stream like a missing image rather
        # than failing the whole document (HB 9102-2008: 54 MB / 15 pages).
        pass
    features["image_count"] = len(image_sizes)
    features["image_sizes"] = image_sizes[:10]
    features["image_ratio"] = round(min(1.0, image_area / page_area), 4)

    compact = compact_chars(raw_text)
    features["char_count"] = len(compact)
    features["line_count"] = max(
        0, len([line for line in raw_text.splitlines() if line.strip()])
    )

    font_sizes: list[float] = []
    try:
        # Best-effort font-size sampling via the text visitor.
        samples: list[float] = []

        def _visitor(_text: str, _cm: Any, tm: Any, _font: Any, _fs: Any) -> None:
            try:
                scale = float(_fs)
                samples.append(round(scale, 1))
            except (TypeError, ValueError):
                pass

        pdf_page.extract_text(visitor_text=_visitor)
        if samples:
            from collections import Counter

            counter = Counter(samples)
            font_sizes = [size for size, _ in counter.most_common(5)]
            features["dominant_font_size"] = float(counter.most_common(1)[0][0])
    except (AttributeError, TypeError, ValueError):
        pass
    features["font_sizes"] = font_sizes
    return features


def parse_pdf(path: Path) -> ParsedDocument:
    source_hash = sha256_file(path)
    document_id = stable_id("document", source_hash)
    reader = PdfReader(path)
    pages: list[PageRecord] = []

    for page_number, pdf_page in enumerate(reader.pages, start=1):
        raw_text, invalid_unicode_count = repair_invalid_unicode(
            pdf_page.extract_text() or ""
        )
        compact_count = len(compact_chars(raw_text))
        table_hints = detect_table_hints(raw_text)
        if (
            not table_hints
            and raw_text
            and (
                is_flattened_table_noise(raw_text)
                or any(
                    is_flattened_table_noise(normalize_line(line))
                    for line in raw_text.splitlines()
                )
            )
        ):
            table_hints.append("检测到密集数字表格")
        layout_text = extract_layout_text(pdf_page) if table_hints else None
        layout_features = extract_page_layout_features(pdf_page, raw_text)
        tounicode_cid_conflict = has_tounicode_cid_conflict(pdf_page)
        glyph_name_garbage = is_glyph_name_garbage(raw_text)
        low_quality_ocr = has_low_quality_ocr_text(
            raw_text,
            image_ratio=float(layout_features.get("image_ratio", 0.0)),
        )
        text_layer_corrupted = (
            glyph_name_garbage
            or is_fake_cjk_garbage(raw_text)
            or has_fake_glyph_corruption(raw_text)
            or tounicode_cid_conflict
            or low_quality_ocr
        )
        watermark_only = is_watermark_only_page(pdf_page, raw_text)
        if watermark_only:
            page_route = "watermark_excluded"
        elif compact_count == 0:
            page_route = "ocr_required"
        elif compact_count < TEXT_PAGE_MIN_CHARS:
            page_route = "low_text_review"
        elif text_layer_corrupted:
            page_route = "text_layer_review"
        else:
            page_route = "native_text"
        if table_hints and page_route == "native_text":
            page_route = "layout_review"
        pages.append(
            PageRecord(
                page_number=page_number,
                raw_text=raw_text,
                layout_text=layout_text,
                raw_char_count=compact_count,
                invalid_unicode_count=invalid_unicode_count,
                text_layer_corruption={
                    "glyph_name_garbage": glyph_name_garbage,
                    "tounicode_cid_conflict": tounicode_cid_conflict,
                    "low_quality_ocr": low_quality_ocr,
                },
                table_hints=table_hints,
                route=page_route,
                page_type=("table" if table_hints and not text_layer_corrupted else "text"),
                indexable=(page_route not in {"watermark_excluded"}),
                image_count=layout_features["image_count"],
                image_coverage=layout_features["image_ratio"],
                layout_features=layout_features,
            )
        )

    for page in pages:
        # Second-layer feature scoring: combine layout/statistical signals.
        toc_score = compute_toc_score(
            page.raw_text,
            page_number=page.page_number,
            total_pages=len(pages),
            layout_features=page.layout_features,
        )
        page.layout_features["toc_score"] = toc_score
        if is_probable_toc_page(page.raw_text) or toc_score >= 0.85:
            page.is_toc = True
            page.indexable = False
            page.route = "toc_excluded"

    text_pages = sum(page.raw_char_count >= TEXT_PAGE_MIN_CHARS for page in pages)
    coverage = text_pages / len(pages) if pages else 0.0
    has_tables = any(page.table_hints for page in pages)
    raw_chars = sum(page.raw_char_count for page in pages)
    normalized_raw_chars = sum(
        len(compact_chars(unicodedata.normalize("NFKC", page.raw_text))) for page in pages
    )
    normalization_expansion_ratio = (
        normalized_raw_chars / raw_chars if raw_chars else 0.0
    )
    text_layer_warning = normalization_expansion_ratio > 1.02
    invalid_unicode_pages = [
        page.page_number for page in pages if page.invalid_unicode_count
    ]
    text_layer_corrupted_pages = [
        page.page_number
        for page in pages
        if page.text_layer_corruption.get("glyph_name_garbage")
    ]
    text_layer_warning = (
        text_layer_warning
        or bool(invalid_unicode_pages)
        or bool(text_layer_corrupted_pages)
    )
    route = determine_document_route(coverage, has_tables, text_layer_warning)
    repeated_keys = detect_repeated_margin_lines(pages)
    removed_line_count = 0

    for page in pages:
        page.cleaned_text, removed = clean_page(page, repeated_keys)
        page.cleaned_char_count = len(compact_chars(page.cleaned_text))
        removed_line_count += removed

    blocks = split_blocks(document_id, pages) if coverage else []
    chunks = build_chunks(document_id, blocks)
    unpublished_clause_blocks = find_unpublished_clause_blocks(blocks, chunks)
    cleaned_chars = sum(page.cleaned_char_count for page in pages)
    chunk_lengths = [len(chunk.text) for chunk in chunks]
    heading_types = {kind for kind, _ in HEADING_PATTERNS}
    heading_types.update({"article", "clause"})
    headings = [block for block in blocks if block.block_type in heading_types]
    article_ids = {
        block.article_id_normalized
        for block in blocks
        if block.article_id_normalized
    }
    block_articles = {
        block.block_id: block.article_id_normalized
        for block in blocks
        if block.article_id_normalized
    }
    hard_boundary_violations = sum(
        len(
            {
                block_articles[block_id]
                for block_id in chunk.block_ids
                if block_id in block_articles
            }
        )
        > 1
        for chunk in chunks
    )
    table_pages = [page.page_number for page in pages if page.table_hints]
    ocr_pages = [page.page_number for page in pages if page.route == "ocr_required"]
    low_text_pages = [
        page.page_number for page in pages if page.route == "low_text_review"
    ]
    toc_pages = [page.page_number for page in pages if page.is_toc]
    watermark_pages = [
        page.page_number for page in pages if page.route == "watermark_excluded"
    ]

    quality_gates: list[dict[str, str]] = []
    if unpublished_clause_blocks:
        missing_ids = [
            block.article_id_normalized or block.text[:40]
            for block in unpublished_clause_blocks
        ]
        quality_gates.append(
            {
                "status": "fail",
                "gate": "clause_chunk_coverage",
                "message": (
                    f"发现 {len(unpublished_clause_blocks)} 个带正文的条款未进入 Chunk，"
                    f"已阻止发布：{missing_ids[:12]}"
                ),
            }
        )
    if route == "ocr_required":
        quality_gates.append(
            {
                "status": "fail",
                "gate": "native_text_coverage",
                "message": "纯扫描件必须进入 OCR/MinerU，禁止提交空文本到向量索引。",
            }
        )
    elif coverage < 0.98:
        quality_gates.append(
            {
                "status": "warn",
                "gate": "page_text_coverage",
                "message": "部分页面缺少有效文本，应逐页 OCR 或版面解析。",
            }
        )
    if text_layer_warning:
        quality_gates.append(
            {
                "status": "warn",
                "gate": "text_layer_quality",
                "message": "文本层归一化膨胀异常，疑似旧式编码或低质量 OCR，应抽检。",
            }
        )
    if invalid_unicode_pages:
        quality_gates.append(
            {
                "status": "warn",
                "gate": "text_layer_unicode",
                "message": (
                    "PDF 文本层包含非法 Unicode 数学/特殊字体编码，已安全替换并将相关页"
                    f"交给 OCR 复核：{invalid_unicode_pages}"
                ),
            }
        )
    if text_layer_corrupted_pages:
        quality_gates.append(
            {
                "status": "warn",
                "gate": "text_layer_glyph_names",
                "message": (
                    "PDF 文本层暴露 PostScript 字形名而非真实文字，已将相关页"
                    f"交给 OCR 复核：{text_layer_corrupted_pages}"
                ),
            }
        )
    if table_pages:
        quality_gates.append(
            {
                "status": "warn",
                "gate": "table_structure",
                "message": "检测到表格提示；需由 MinerU/Docling 生成结构化表格后再放行。",
            }
        )
    if coverage and not headings:
        quality_gates.append(
            {
                "status": "warn",
                "gate": "section_structure",
                "message": "未识别到章/条标题，需人工抽检或扩展标题规则。",
            }
        )
    if not quality_gates:
        quality_gates.append(
            {
                "status": "pass",
                "gate": "baseline",
                "message": "原生文本覆盖率和章节结构通过基线门禁。",
            }
        )

    normalized_text = compact_chars("\n".join(page.raw_text for page in pages))
    normalized_text_hash = (
        hashlib.sha256(normalized_text.encode("utf-8")).hexdigest() if normalized_text else None
    )
    qa = {
        "page_count": len(pages),
        "text_page_count": text_pages,
        "text_coverage": round(coverage, 4),
        "raw_char_count": raw_chars,
        "normalized_raw_char_count": normalized_raw_chars,
        "normalization_expansion_ratio": round(normalization_expansion_ratio, 4),
        "invalid_unicode_pages": invalid_unicode_pages,
        "invalid_unicode_replacement_count": sum(
            page.invalid_unicode_count for page in pages
        ),
        "text_layer_corrupted_pages": text_layer_corrupted_pages,
        "cleaned_char_count": cleaned_chars,
        "cleaned_char_retention": (
            round(cleaned_chars / normalized_raw_chars, 4)
            if normalized_raw_chars
            else 0.0
        ),
        "removed_margin_line_count": removed_line_count,
        "repeated_margin_lines": sorted(repeated_keys),
        "heading_count": len(headings),
        "article_count": len(article_ids),
        "article_alias_count": sum(len(chunk.article_aliases) for chunk in chunks),
        "article_hard_boundary_violations": hard_boundary_violations,
        "unpublished_clause_count": len(unpublished_clause_blocks),
        "unpublished_clause_ids": [
            block.article_id_normalized or block.text[:40]
            for block in unpublished_clause_blocks
        ],
        "table_pages": table_pages,
        "pages_requiring_ocr": ocr_pages,
        "low_text_pages": low_text_pages,
        "toc_pages": toc_pages,
        "toc_scores": {
            page.page_number: page.layout_features.get("toc_score", 0.0)
            for page in pages
        },
        "watermark_pages": watermark_pages,
        "chunk_count": len(chunks),
        "chunk_char_min": min(chunk_lengths, default=0),
        "chunk_char_median": int(statistics.median(chunk_lengths)) if chunk_lengths else 0,
        "chunk_char_max": max(chunk_lengths, default=0),
        "quality_gates": quality_gates,
    }
    return ParsedDocument(
        source_path=path,
        source_hash=source_hash,
        normalized_text_hash=normalized_text_hash,
        document_id=document_id,
        pages=pages,
        route=route,
        repeated_margin_lines=sorted(repeated_keys),
        removed_margin_line_count=removed_line_count,
        blocks=blocks,
        chunks=chunks,
        qa=qa,
    )


def safe_folder_name(path: Path, source_hash: str) -> str:
    stem = re.sub(r'[<>:"/\\|?*]', "_", path.stem).strip(" .")
    return f"{stem[:72]}__{source_hash[:8]}"


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_document(
    output_root: Path,
    document: ParsedDocument,
    *,
    display_name: str | None = None,
) -> dict[str, Any]:
    display_path = Path(display_name) if display_name else document.source_path
    document_folder = output_root / safe_folder_name(display_path, document.source_hash)
    document_folder.mkdir(parents=True, exist_ok=True)
    asset_folder = document_folder / "assets"
    asset_manifest: list[dict[str, Any]] = []
    if document.assets:
        asset_folder.mkdir(parents=True, exist_ok=True)
        for asset in document.assets:
            extension = Path(asset.filename).suffix.casefold() or ".bin"
            stored_name = f"{asset.asset_id}{extension}"
            stored_path = asset_folder / stored_name
            stored_path.write_bytes(asset.content)
            asset_manifest.append(
                {
                    "asset_id": asset.asset_id,
                    "page_number": asset.page_number,
                    "asset_type": asset.asset_type,
                    "bbox": asset.bbox,
                    "filename": asset.filename,
                    "mime_type": asset.mime_type,
                    "caption": asset.caption,
                    "description": asset.description,
                    "relative_path": f"assets/{stored_name}",
                }
            )
    parser_name = str(document.qa.get("parser_name", "native-pdf"))
    parser_metadata = {
        "name": parser_name,
        "version": pypdf_version if parser_name == "native-pdf" else "1",
        "trace": document.qa.get("parser_trace", []),
    }
    blocks_by_page: dict[int, list[BlockRecord]] = defaultdict(list)
    chunks_by_page: dict[int, list[ChunkRecord]] = defaultdict(list)
    for block in document.blocks:
        page_start = block.source_page_start or block.page_number
        page_end = block.source_page_end or block.page_number
        for page_number in range(page_start, page_end + 1):
            blocks_by_page[page_number].append(block)
    for chunk in document.chunks:
        for page_number in range(chunk.page_start, chunk.page_end + 1):
            chunks_by_page[page_number].append(chunk)
    baseline = {
        "document_id": document.document_id,
        "source_path": str(document.source_path),
        "source_name": display_path.name,
        "source_sha256": document.source_hash,
        "parser": parser_metadata,
        "pages": [
            {
                "page_number": page.page_number,
                "raw_text": page.raw_text,
                "raw_char_count": page.raw_char_count,
                "invalid_unicode_count": page.invalid_unicode_count,
            }
            for page in document.pages
        ],
    }
    document_ir = {
        "schema_version": "document-ir/0.2",
        "document_id": document.document_id,
        "source": {
            "path": str(document.source_path),
            "name": display_path.name,
            "sha256": document.source_hash,
            "normalized_text_sha256": document.normalized_text_hash,
        },
        "parser": parser_metadata,
        "route": document.route,
        "pages": [
            {
                "page_number": page.page_number,
                "route": page.route,
                "raw_char_count": page.raw_char_count,
                "cleaned_char_count": page.cleaned_char_count,
                "invalid_unicode_count": page.invalid_unicode_count,
                "cleaned_text": page.cleaned_text,
                "layout_text": page.layout_text,
                "table_hints": page.table_hints,
                "is_toc": page.is_toc,
                "indexable": page.indexable,
                "block_count": len(blocks_by_page[page.page_number]),
                "chunk_count": len(chunks_by_page[page.page_number]),
                "article_ids": list(
                    dict.fromkeys(
                        block.article_id_normalized
                        for block in blocks_by_page[page.page_number]
                        if block.article_id_normalized
                    )
                ),
                "rich_block_count": len(page.rich_blocks),
                "asset_ids": [
                    asset["asset_id"]
                    for asset in asset_manifest
                    if asset["page_number"] == page.page_number
                ],
                "page_type": page.page_type,
                "image_count": page.image_count,
                "image_coverage": page.image_coverage,
                "layout_features": page.layout_features,
            }
            for page in document.pages
        ],
        "blocks": [asdict(block) for block in document.blocks],
        "chunks": [asdict(chunk) for chunk in document.chunks],
        "assets": asset_manifest,
        "qa": document.qa,
    }
    write_json(document_folder / "baseline.json", baseline)
    write_json(document_folder / "document_ir.json", document_ir)
    write_json(document_folder / "qa.json", document.qa)
    (document_folder / "cleaned.md").write_text(
        "\n\n".join(
            f"<!-- page: {page.page_number} -->\n\n{page.cleaned_text}"
            for page in document.pages
            if page.cleaned_text and page.indexable
        ),
        encoding="utf-8",
    )
    with (document_folder / "chunks.jsonl").open("w", encoding="utf-8") as stream:
        for chunk in document.chunks:
            stream.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    return {
        "document_id": document.document_id,
        "source_name": display_path.name,
        "source_path": str(document.source_path),
        "source_sha256": document.source_hash,
        "normalized_text_sha256": document.normalized_text_hash,
        "route": document.route,
        "output_folder": str(document_folder),
        "qa": document.qa,
    }


def duplicate_groups(manifest_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for document in manifest_documents:
        normalized_hash = document.get("normalized_text_sha256")
        if normalized_hash:
            groups[str(normalized_hash)].append(str(document["source_name"]))
    return [
        {"normalized_text_sha256": digest, "files": names}
        for digest, names in groups.items()
        if len(names) > 1
    ]


def write_report(output_root: Path, documents: list[dict[str, Any]]) -> None:
    route_counts = Counter(str(document["route"]) for document in documents)
    duplicates = duplicate_groups(documents)
    lines = [
        "# 规章文档预处理实验报告",
        "",
        f"- 生成时间：{datetime.now(UTC).isoformat()}",
        f"- 文档数：{len(documents)}",
        f"- 路由统计：{dict(route_counts)}",
        f"- 归一化文本重复组：{len(duplicates)}",
        "",
        "| 文档 | 路由 | 页数 | 文本覆盖率 | 标题数 | 目录页 | 表格页 | OCR/低文本页 | 块数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for document in documents:
        qa = document["qa"]
        lines.append(
            "| {name} | {route} | {pages} | {coverage:.1%} | {headings} | "
            "{toc} | {tables} | {ocr} | {chunks} |".format(
                name=str(document["source_name"]).replace("|", "\\|"),
                route=document["route"],
                pages=qa["page_count"],
                coverage=qa["text_coverage"],
                headings=qa["heading_count"],
                toc=len(qa["toc_pages"]),
                tables=len(qa["table_pages"]),
                ocr=(
                    f"{len(qa['pages_requiring_ocr'])}/"
                    f"{len(qa['low_text_pages'])}"
                ),
                chunks=qa["chunk_count"],
            )
        )
    if duplicates:
        lines.extend(["", "## 重复内容候选", ""])
        for group in duplicates:
            lines.append(f"- {'；'.join(group['files'])}")
    lines.extend(
        [
            "",
            "## 结论",
            "",
            "- `ocr_required` 文档必须先走 MinerU/OCR，不进入 RAGFlow 向量化。",
            "- `layout_review`/`hybrid_review` 文档的表格或缺字页需要版面解析后再放行。",
            "- `native_text` 文档可直接使用本实验生成的章节块继续做检索对照。",
        ]
    )
    (output_root / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_inputs(folder: Path | None, explicit_inputs: list[Path]) -> list[Path]:
    paths = list(explicit_inputs)
    if folder:
        paths.extend(sorted(folder.glob("*.pdf")))
    unique_paths: dict[str, Path] = {}
    for path in paths:
        resolved = path.resolve()
        if resolved.suffix.casefold() == ".pdf":
            unique_paths[str(resolved).casefold()] = resolved
    return list(unique_paths.values())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regulation PDF preprocessing experiment")
    parser.add_argument("--folder", type=Path, help="Folder containing PDF files")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Explicit PDF path; may be repeated",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args.folder, args.input)
    if not inputs:
        raise SystemExit("No PDF input files found.")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_documents: list[dict[str, Any]] = []
    for index, path in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] {path.name}", flush=True)
        document = parse_pdf(path)
        manifest_documents.append(write_document(args.output, document))

    manifest = {
        "schema_version": "preprocess-run/0.1",
        "generated_at": datetime.now(UTC).isoformat(),
        "parser": {"name": "pypdf", "version": pypdf_version},
        "documents": manifest_documents,
        "duplicate_groups": duplicate_groups(manifest_documents),
    }
    write_json(args.output / "manifest.json", manifest)
    write_report(args.output, manifest_documents)
    print(f"Output: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
