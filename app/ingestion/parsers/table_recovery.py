"""Conservative, evidence-checked recovery of damaged two-column OCR tables."""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from collections import Counter
from html import escape
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import ImageOps

from app.ingestion.pipeline import (
    collapsed_table_rows,
    table_rows_from_html,
)


class _Cells(HTMLParser):
    def __init__(self, markup: str) -> None:
        super().__init__()
        self.cells: list[str] = []
        self.current: list[str] | None = None
        self.feed(markup)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"td", "th"}:
            self.current = []

    def handle_data(self, data: str) -> None:
        if self.current is not None:
            self.current.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self.current is not None:
            self.cells.append(" ".join(self.current))
            self.current = None


def structure_issue(markup: str) -> str | None:
    rows = table_rows_from_html(markup)
    if not rows:
        return None
    # One data row whose every cell carries a whole column: the printed row
    # separators are gone (Docket 21-44 p. 12 arrived as a header row plus one
    # data row holding the altitude, vapour-pressure, humidity and density
    # columns run together).  The caller rebuilds it from column agreement or, if
    # that is not provable, keeps it out of the index.
    if collapsed_table_rows(rows):
        return "collapsed_row"
    # Repeated dot leaders within one label indicate distinct printed rows.
    # Numeric cells beside them may have lost all row separators. Flag the
    # ambiguity; never guess how to partition a digit string into values.
    for row in rows[1:]:
        if (
            len(row) >= 2
            and len(re.findall(r"\S[^.]*\.{2,}", row[0])) >= 2
            and sum(bool(re.fullmatch(r"\d{3,}", c.strip())) for c in row[1:]) >= 1
        ):
            return "merged_numeric_rows"
    if len(rows) <= 2 and re.search(r"\([a-z]\).+\([a-z]\)", " ".join(_Cells(markup).cells), re.I):
        return "multiple_labeled_rows"
    header = [c for c in rows[0] if c]
    duplicate_header = len(set(header)) < len(header)
    header_value = any(re.search(r"[+±−-]\s*\d+(?:\.\d+)?\s*%", c) for c in header)
    fragmented = any(sum(bool(c.strip()) for c in row) == 1 for row in rows[1:])
    if duplicate_header and header_value and fragmented:
        return "fragmented_spanning_header"
    return None


def crop_table_pdf(path: Path, page_number: int, bbox: list[float]) -> bytes:
    """Render only source pixels; never trust or modify the PDF's hidden OCR layer."""
    if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 1000):
        raise ValueError("Invalid table horizontal bounds")
    if not (0 <= bbox[1] < bbox[3] <= 1000):
        raise ValueError("Invalid table vertical bounds")
    with pdfium.PdfDocument(path) as document:
        page = document[page_number - 1]
        try:
            width, height = page.get_size()
            # Bound memory on oversized source pages; add a small OCR-box margin.
            scale = min(6, (24_000_000 / (width * height)) ** 0.5)
            bitmap = page.render(scale=scale)
            try:
                full = bitmap.to_pil()
                box = (
                    full.width * max(0, bbox[0] - 1) / 1000,
                    full.height * max(0, bbox[1] - 1) / 1000,
                    full.width * min(1000, bbox[2] + 3) / 1000,
                    full.height * min(1000, bbox[3] + 2) / 1000,
                )
                crop = full.crop(box).convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()
    padded = ImageOps.expand(crop, border=60, fill="white")
    output = BytesIO()
    padded.save(output, format="PDF", resolution=144)
    return output.getvalue()


def _clean_label(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip()
    clean = re.sub(r"[\s._-]+$", "", text)
    if re.fullmatch(r"(?:[A-Za-z]\.)+[A-Za-z]", clean):
        clean += "."
    return clean


def _tokens(text: str) -> Counter[str]:
    text = unicodedata.normalize("NFKC", text).replace("−", "-").lower()
    return Counter(
        re.sub(r"\s+", "", token)
        for token in re.findall(r"[a-z]+|[+±-]?\s*\d+(?:\.\d+)?\s*%?", text.replace("_", " "))
    )


def recover_two_columns(markup: str, lines: list[dict[str, Any]]) -> str | None:
    """Require geometry AND full lexical/numeric agreement with independent OCR.

    Deliberately declines multi-column, low-confidence, crossing, incomplete or
    conflicting crops. No values or file/page identifiers are production rules.
    """
    if len(lines) < 6 or len(lines) > 200:
        return None
    if any(
        not isinstance(line.get("text"), str)
        or not isinstance(line.get("bbox"), list)
        or len(line["bbox"]) != 4
        or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in line["bbox"])
        or line["bbox"][2] <= line["bbox"][0]
        or line["bbox"][3] <= line["bbox"][1]
        or not isinstance(line.get("score"), (int, float))
        or not math.isfinite(line["score"])
        or line["score"] < 0.88
        for line in lines
    ):
        return None

    def center(line: dict[str, Any]) -> float:
        return (line["bbox"][1] + line["bbox"][3]) / 2

    ordered = sorted(lines, key=center)
    height = statistics.median(line["bbox"][3] - line["bbox"][1] for line in lines)
    headers = [line for line in ordered if center(line) - center(ordered[0]) < height * 0.5]
    if len(headers) != 2 or any(re.search(r"\d", line["text"]) for line in headers):
        return None
    headers.sort(key=lambda line: line["bbox"][0])
    divider = (headers[0]["bbox"][2] + headers[1]["bbox"][0]) / 2
    body = [line for line in ordered if line not in headers]
    left = [line for line in body if line["bbox"][0] < divider]
    right = [line for line in body if line["bbox"][0] >= divider]
    if not left or len(right) < 2 or any(not re.search(r"\d", line["text"]) for line in right):
        return None
    if max(line["bbox"][2] for line in left) >= min(line["bbox"][0] for line in right):
        return None
    if any(center(b) - center(a) < height * 0.75 for a, b in zip(right, right[1:], strict=False)):
        return None
    rows = [[h["text"].strip() for h in headers]]
    for value in right:
        # Labels can wrap ABOVE their aligned value, but not below it.
        labels = [line for line in left if center(line) <= center(value) + height * 0.3]
        if not labels or abs(center(labels[-1]) - center(value)) > height * 0.3:
            return None
        left = [line for line in left if line not in labels]
        label = " ".join(_clean_label(line["text"]) for line in labels)
        rows.append([label, unicodedata.normalize("NFKC", value["text"]).strip()])
    if left:
        return None
    original = " ".join(_Cells(markup).cells)
    recovered = " ".join(cell for row in rows for cell in row)
    if _tokens(original) != _tokens(recovered):
        return None
    return (
        "<table>"
        + "".join(
            "<tr>" + "".join(f"<{tag}>{escape(cell)}</{tag}>" for cell in row) + "</tr>"
            for i, row in enumerate(rows)
            for tag in ["th" if i == 0 else "td"]
        )
        + "</table>"
    )
