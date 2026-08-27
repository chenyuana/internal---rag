from __future__ import annotations

import re
from html.parser import HTMLParser

HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
MARKDOWN_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
_TABLE_TAG_ESCAPE_RE = re.compile(
    r"\\(?=</?(?:table|caption|thead|tbody|tfoot|tr|th|td)\b)",
    re.IGNORECASE,
)
_FORM_PLACEHOLDER_RE = re.compile(r"^(?:第\s*页(?:共\s*页)?|页|第\s*页共\s*)$")


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[tuple[str, list[tuple[list[str], bool]]]] = []
        self._in_table = False
        self._in_caption = False
        self._in_cell = False
        self._cell_is_header = False
        self._caption_parts: list[str] = []
        self._cell_parts: list[str] = []
        self._row: list[str] = []
        self._row_has_header = False
        self._rows: list[tuple[list[str], bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        tag = tag.lower()
        if tag == "table":
            self._in_table = True
            self._caption_parts = []
            self._rows = []
        elif self._in_table and tag == "caption":
            self._in_caption = True
        elif self._in_table and tag == "tr":
            self._row = []
            self._row_has_header = False
        elif self._in_table and tag in {"th", "td"}:
            self._in_cell = True
            self._cell_is_header = tag == "th"
            self._cell_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"th", "td"} and self._in_cell:
            self._row.append(_clean("".join(self._cell_parts)))
            self._row_has_header = self._row_has_header or self._cell_is_header
            self._in_cell = False
        elif tag == "tr" and self._in_table and any(self._row):
            self._rows.append((self._row, self._row_has_header))
        elif tag == "caption":
            self._in_caption = False
        elif tag == "table" and self._in_table:
            self.tables.append((_clean("".join(self._caption_parts)), self._rows))
            self._in_table = False

    def handle_data(self, data: str) -> None:
        if self._in_caption:
            self._caption_parts.append(data)
        if self._in_cell:
            self._cell_parts.append(data)


def _clean(value: str) -> str:
    return " ".join(value.split()).strip()


def _normalize_table_markup(text: str) -> str:
    """Accept the backslash-prefixed HTML emitted by some RAGFlow chunks.

    A few ingestion paths preserve escaped tags as ``\\<table>``.  The
    backslash is transport noise, not part of the table, so remove it only
    immediately before a known table tag.  We deliberately do not unescape
    arbitrary HTML or backslashes in cell text.
    """
    return _TABLE_TAG_ESCAPE_RE.sub("", text)


def _render_rows(caption: str, rows: list[tuple[list[str], bool]]) -> list[str]:
    if len(rows) < 2:
        return []
    header_index = next((index for index, (_, is_header) in enumerate(rows) if is_header), 0)
    headers = rows[header_index][0]
    if not headers:
        return []
    rendered: list[str] = []
    for values, _ in rows[header_index + 1 :]:
        pairs = [
            f"{header}: {value}"
            for header, value in zip(headers, values, strict=False)
            if header and value
        ]
        if not pairs:
            continue
        prefix = f"{caption}；" if caption else ""
        rendered.append(prefix + "；".join(pairs))
    return rendered


def _is_blank_form(rows: list[tuple[list[str], bool]]) -> bool:
    """Identify an empty inspection form, not a filled result table.

    A.3/A.4 contain labels such as ``项目名称：`` and page placeholders but no
    measured value.  They are useful as deliverable-format evidence, yet must
    not satisfy a question asking for acceptance limits.
    """
    labels = {
        "项目名称",
        "测区编号",
        "作业时间",
        "成果类型",
        "飞行架次",
        "无人机型号",
        "检测方式",
        "像控点密度",
        "测量员",
        "检查者",
        "复核者",
        "测量日期",
        "检查日期",
        "复核日期",
    }
    for values, _ in rows:
        for raw in values:
            value = _clean(raw).strip(" ：:")
            if not value or value in labels or _FORM_PLACEHOLDER_RE.match(value):
                continue
            # Header names in an unfilled form (e.g. 实测距离、差值、超限比例)
            # are not values.  Numeric values, project names, dates and filled
            # notes do not match this bounded header set and count as data.
            if value in {
                "序号",
                "要素类型",
                "实测距离",
                "图上距离",
                "差值",
                "检查边数量",
                "超限数量",
                "超限比例",
            }:
                continue
            return False
    return True


def html_table_records(text: str) -> list[str]:
    parser = _TableParser()
    parser.feed(_normalize_table_markup(text))
    parser.close()
    records: list[str] = []
    for caption, rows in parser.tables:
        if _is_blank_form(rows):
            continue
        records.extend(_render_rows(caption, rows))
    return records


def html_table_blocks(text: str) -> list[str]:
    """Return complete table blocks, including captions and blank cells.

    ``table_records`` is intentionally lossy and is used for retrieval.  UI
    citations need the opposite: the original table layout must remain
    inspectable, including empty form fields.  This helper returns only the
    table markup and never the surrounding chunk prose.
    """
    return HTML_TABLE_RE.findall(_normalize_table_markup(text))


def markdown_table_records(text: str) -> list[str]:
    records: list[str] = []
    lines = text.splitlines()
    index = 0
    while index + 2 < len(lines):
        header_line = lines[index].strip()
        separator_line = lines[index + 1].strip()
        if not (header_line.startswith("|") and separator_line.startswith("|")):
            index += 1
            continue
        headers = [item.strip() for item in header_line.strip("|").split("|")]
        separators = [item.strip() for item in separator_line.strip("|").split("|")]
        if len(headers) != len(separators) or not all(
            MARKDOWN_SEPARATOR_RE.match(item) for item in separators
        ):
            index += 1
            continue
        index += 2
        while index < len(lines) and lines[index].strip().startswith("|"):
            values = [item.strip() for item in lines[index].strip().strip("|").split("|")]
            pairs = [
                f"{header}: {value}"
                for header, value in zip(headers, values, strict=False)
                if header and value
            ]
            if pairs:
                records.append("；".join(pairs))
            index += 1
    return records


def table_records(text: str) -> list[str]:
    """Return semantic key-value records without exposing raw table markup."""
    return [*html_table_records(text), *markdown_table_records(text)]


def without_table_markup(text: str) -> str:
    without_html = HTML_TABLE_RE.sub("\n", text)
    return "\n".join(
        line for line in without_html.splitlines() if not line.strip().startswith("|")
    )
