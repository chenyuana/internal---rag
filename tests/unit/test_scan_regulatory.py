from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from pypdf import PdfWriter

from app.ingestion.parsers.remote import RemoteContentBlock, RemoteParseResult
from app.ingestion.parsers.scan_regulatory import (
    ScannedRegulatoryPdfParser,
    english_heading,
    is_scan_regulatory_pdf,
)
from app.ingestion.pipeline import PageRecord


class FakeImage:
    def __init__(self, size: tuple[int, int] | None = None) -> None:
        self.image = None
        self._size = size

    @property
    def size(self) -> tuple[int, int]:
        return self._size or (0, 0)


class FakePage:
    def __init__(self, text: str = "", images: list[FakeImage] | None = None) -> None:
        self._text = text
        self.images = images or []

    def extract_text(self) -> str:
        return self._text


class FakeReader:
    def __init__(
        self,
        producer: str = "",
        pages: list[FakePage] | None = None,
    ) -> None:
        self.metadata = {"/Producer": producer} if producer else None
        self.pages = pages or [FakePage()]


# ---------------------------------------------------------------- 判定
def test_is_scan_regulatory_pdf_detects_scan_producer() -> None:
    path = Path("x.pdf")
    reader = FakeReader(
        producer="Pdf-It version 5.091",
        pages=[FakePage("Federal Aviation Regulations Title 14 PART 23")],
    )
    assert is_scan_regulatory_pdf(path, pdf_reader=reader) is True


def test_is_scan_regulatory_pdf_detects_marker_with_image() -> None:
    path = Path("x.pdf")
    reader = FakeReader(
        producer="",
        pages=[FakePage("Federal Aviation Regulations Title 14 PART 23", images=[FakeImage()])],
    )
    with patch("app.ingestion.parsers.scan_regulatory._image_heavy", return_value=True):
        assert is_scan_regulatory_pdf(path, pdf_reader=reader) is True


def test_is_scan_regulatory_pdf_ignores_clean_or_non_regulatory() -> None:
    path = Path("x.pdf")
    reader = FakeReader(
        producer="Microsoft Print to PDF",
        pages=[FakePage("This is a normal english document without regulatory markers.")],
    )
    assert is_scan_regulatory_pdf(path, pdf_reader=reader) is False


# ---------------------------------------------------------------- 噪音剔除
@pytest.mark.parametrize(
    "producer,text",
    [
        ("ABBYY FineReader", "中文技术标准" * 10),
        ("Pdf-It", "A far better product brochure with a small logo"),
        ("Scanner", "A regular English business letter with no regulations"),
        ("Pdf-It", ""),
    ],
)
def test_scan_producer_is_not_sufficient(producer: str, text: str) -> None:
    assert not is_scan_regulatory_pdf(
        Path("x.pdf"), pdf_reader=FakeReader(producer, [FakePage(text)])
    )


def test_small_logo_is_not_a_scan() -> None:
    reader = FakeReader("", [FakePage("Federal Aviation Regulations Title 14")])
    with patch("app.ingestion.parsers.scan_regulatory._image_heavy", return_value=False):
        assert not is_scan_regulatory_pdf(Path("x.pdf"), pdf_reader=reader)


# ---------------------------------------------------------------- 英文标题
def test_english_heading_detects_regulatory_headings() -> None:
    assert english_heading("PART 23—AIRWORTHINESS STAND-ARDS: NORMAL") == (
        "part",
        "PART 23—AIRWORTHINESS STAND-ARDS: NORMAL",
    )
    assert english_heading("Title 14—AERONAUTICS AND SPACE") == (
        "title",
        "Title 14—AERONAUTICS AND SPACE",
    )
    assert english_heading("[Regulatory Docket No. 4080]") == (
        "docket",
        "[Regulatory Docket No. 4080]",
    )
    assert english_heading("§ 23.629 Flutter prevention") is not None
    # 非标题正文
    assert english_heading("This amendment adds Part 23 [New] to the Federal Aviation") is None


# ---------------------------------------------------------------- 解析器构建
def _parser(mineru: Any = None) -> ScannedRegulatoryPdfParser:
    return ScannedRegulatoryPdfParser(mineru)


def test_build_blocks_uses_english_headings_as_section_path() -> None:
    parser = _parser()
    pages = [
        PageRecord(
            page_number=1,
            raw_text="",
            cleaned_text="",
            route="scan_regulatory_ocr",
            rich_blocks=[
                {"block_type": "paragraph", "text": "Title 14—AERONAUTICS AND SPACE"},
                {"block_type": "paragraph", "text": "PART 23—AIRWORTHINESS STAND-ARDS"},
                {"block_type": "paragraph", "text": "This amendment adds Part 23 [New]."},
            ],
            indexable=True,
        )
    ]
    blocks = parser._build_blocks("doc", pages)
    # 标题行成为 heading 块且作为后续正文的 section_path
    headings = [b for b in blocks if b.block_type == "heading"]
    assert len(headings) == 2
    body = [b for b in blocks if b.block_type == "paragraph"]
    assert body[0].section_path == [
        "Title 14—AERONAUTICS AND SPACE",
        "PART 23—AIRWORTHINESS STAND-ARDS",
    ]


def _parse_result(tmp_path: Path, result: RemoteParseResult, pages: int = 1):
    path = tmp_path / "sample.pdf"
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=600, height=800)
    writer.write(path)
    calls = []

    def parse_pages(path, numbers, **kwargs):
        calls.append((numbers, kwargs))
        if kwargs.get("progress_callback"):
            kwargs["progress_callback"](len(numbers), len(numbers))
        return result

    progress = []
    doc = ScannedRegulatoryPdfParser(SimpleNamespace(enabled=True, parse_pages=parse_pages)).parse(
        path, progress_callback=lambda done, total: progress.append((done, total))
    )
    assert calls[0][1]["parse_method"] == "ocr"
    assert progress == [(pages, pages)]
    return doc


def test_annotations_preserved_but_not_indexed(tmp_path: Path) -> None:
    result = RemoteParseResult(
        "mineru",
        page_blocks={
            1: [
                RemoteContentBlock(
                    1, "paragraph", "See anrrie t ind", bbox=[310, 37, 490, 85], text_level=2
                ),
                RemoteContentBlock(
                    1, "paragraph", "Title 14—AERONAUTICS AND SPACE", bbox=[78, 93, 269, 152]
                ),
                RemoteContentBlock(1, "paragraph", "PART 23—AIRWORTHINESS STANDARDS"),
                RemoteContentBlock(
                    1, "paragraph", "N. E. HALABY, Administrator.", bbox=[839, 477, 921, 506]
                ),
                RemoteContentBlock(1, "paragraph", "See figure A below", bbox=[300, 500, 450, 520]),
                RemoteContentBlock(
                    1,
                    "figure",
                    "",
                    bbox=[11, 147, 46, 359],
                    image_content=b"image",
                    image_filename="note.png",
                ),
            ]
        },
        excluded_blocks={
            1: [RemoteContentBlock(1, "annotation", "17965", source_type="page_number")]
        },
    )
    doc = _parse_result(tmp_path, result)
    text = "\n".join(c.text for c in doc.chunks)
    assert "See anrrie" not in text and "17965" not in text
    assert "N. E. HALABY" in text and "See figure A below" in text
    assert "See anrrie" in doc.pages[0].raw_text
    assert doc.pages[0].cleaned_text.startswith("# Title 14")
    assert len(doc.assets) == 1
    assert len(doc.qa["annotations"]) == 3
    assert any(g["gate"] == "annotation_review" for g in doc.qa["quality_gates"])


def test_formula_table_without_html_and_image_survive(tmp_path: Path) -> None:
    result = RemoteParseResult(
        "mineru",
        page_blocks={
            1: [
                RemoteContentBlock(1, "paragraph", "§ 23.629 Flutter prevention"),
                RemoteContentBlock(1, "formula", "F = m a", latex="F=ma"),
                RemoteContentBlock(1, "table", "Pitch 60 Roll 30"),
                RemoteContentBlock(
                    1, "figure", "Figure 1", image_content=b"png", image_filename="figure.png"
                ),
            ]
        },
    )
    doc = _parse_result(tmp_path, result)
    assert any("F=ma" in c.formula_latex for c in doc.chunks)
    assert "$$\nF=ma\n$$" in doc.pages[0].cleaned_text
    assert any("Pitch 60" in c.text for c in doc.chunks)
    assert doc.assets[0].asset_id in {a for c in doc.chunks for a in c.asset_ids}


def test_empty_result_cannot_pass_quality(tmp_path: Path) -> None:
    doc = _parse_result(tmp_path, RemoteParseResult("mineru", warnings=["OCR failed"]))
    assert doc.qa["page_count"] == 1
    assert not doc.pages[0].indexable
    assert any(g["status"] == "fail" for g in doc.qa["quality_gates"])
    assert doc.qa["remote_warnings"] == ["OCR failed"]


def test_image_only_page_is_preserved_and_requires_review(tmp_path: Path) -> None:
    doc = _parse_result(
        tmp_path,
        RemoteParseResult(
            "mineru",
            page_blocks={
                1: [
                    RemoteContentBlock(
                        1, "figure", "", image_content=b"png", image_filename="chart.png"
                    )
                ]
            },
        ),
    )
    assert doc.qa["content_coverage"] == 1
    assert doc.qa["text_coverage"] == 0
    assert doc.qa["image_only_pages"] == [1]
    assert doc.pages[0].indexable
    assert "assets/" in doc.pages[0].cleaned_text
    assert doc.assets[0].asset_id in doc.chunks[0].asset_ids
    assert any(g["gate"] == "image_only_pages" for g in doc.qa["quality_gates"])


def test_table_footnote_is_preserved(tmp_path: Path) -> None:
    html = "<table><tr><td>Pitch</td><td>Roll</td></tr><tr><td>60</td><td>30</td></tr></table>"
    result = RemoteParseResult(
        "mineru",
        page_blocks={
            1: [
                RemoteContentBlock(
                    1, "table", html + "\nOnly for temporary application.", table_html=html
                )
            ]
        },
    )
    doc = _parse_result(tmp_path, result)
    assert "Pitch=60" in doc.chunks[0].text
    assert "Only for temporary application." in doc.chunks[0].text


def test_image_area_uses_display_geometry_not_presence() -> None:
    from app.ingestion.parsers.scan_regulatory import _image_heavy

    page = SimpleNamespace(
        width=600, height=800, images=[{"x0": 0, "top": 0, "x1": 60, "bottom": 80}]
    )
    with patch("app.ingestion.parsers.scan_regulatory.pdfplumber.open") as opened:
        opened.return_value.__enter__.return_value.pages = [page]
        assert not _image_heavy(Path("x.pdf"))
        page.images[0].update(x1=600, bottom=800)
        assert _image_heavy(Path("x.pdf"))


def test_text_only_fallback_and_missing_page(tmp_path: Path) -> None:
    doc = _parse_result(tmp_path, RemoteParseResult("mineru", page_texts={1: "Regulatory body"}), 2)
    assert any("Regulatory body" in c.text for c in doc.chunks)
    assert doc.qa["text_only_fallback_pages"] == [1]
    assert doc.qa["missing_text_pages"] == [2]


def test_hierarchy_keeps_parents_and_replaces_siblings() -> None:
    titles = [
        "Title 14—Aeronautics",
        "Chapter I—Federal Aviation Agency",
        "PART 23—Standards",
        "Subpart B—Flight",
        "§ 23.21 Compliance",
        "Body one",
        "§ 23.23 Limits",
        "Body two",
    ]
    page = PageRecord(1, "", rich_blocks=[{"text": t, "block_type": "paragraph"} for t in titles])
    blocks = _parser()._build_blocks("doc", [page])
    assert blocks[-1].section_path == [*titles[:4], titles[-2]]
    chunks = _parser()._build_chunks("doc", blocks)
    assert (
        next(c for c in chunks if "Body one" in c.text).parent_chunk_id
        != next(c for c in chunks if "Body two" in c.text).parent_chunk_id
    )


def test_merged_table_rows_are_flagged_not_guessed(tmp_path: Path) -> None:
    html = (
        "<table><tr><td>Application</td><td>Pitch</td></tr>"
        "<tr><td>(a) temporary (b) prolonged</td><td>607510</td></tr></table>"
    )
    doc = _parse_result(
        tmp_path,
        RemoteParseResult(
            "mineru", page_blocks={1: [RemoteContentBlock(1, "table", html, table_html=html)]}
        ),
    )
    assert "607510" in doc.chunks[0].text
    assert doc.chunks[0].table_html == [html]
    assert any(g["gate"] == "table_row_alignment" for g in doc.qa["quality_gates"])


def test_long_tables_split_on_rows_without_losing_footnotes(tmp_path: Path) -> None:
    rows = "".join(f"<tr><td>Case {i}</td><td>{i}</td></tr>" for i in range(100))
    html = "<table><tr><td>Case</td><td>Value</td></tr>" + rows + "</table>"
    doc = _parse_result(
        tmp_path,
        RemoteParseResult(
            "mineru",
            page_blocks={
                1: [
                    RemoteContentBlock(
                        1, "table", html + "\nTemporary application only.", table_html=html
                    )
                ]
            },
        ),
    )
    assert len(doc.chunks) > 1
    assert sum(len(c.table_rows) - 1 for c in doc.chunks) == 100
    assert "Temporary application only." in doc.chunks[-1].text
    assert doc.blocks[0].table_html == html


def test_build_chunks_groups_under_english_heading_and_keeps_tables() -> None:
    parser = _parser()
    pages = [
        PageRecord(
            page_number=1,
            raw_text="",
            cleaned_text="",
            route="scan_regulatory_ocr",
            rich_blocks=[
                {"block_type": "paragraph", "text": "PART 23—AIRWORTHINESS STAND-ARDS"},
                {"block_type": "paragraph", "text": "This amendment adds Part 23 [New]."},
                {
                    "block_type": "table",
                    "text": "Altitude(pressure) Tolerance",
                    "table_html": "<table><tr><th>Alt</th><th>Press</th></tr></table>",
                },
            ],
            indexable=True,
        )
    ]
    blocks = parser._build_blocks("doc", pages)
    chunks = parser._build_chunks("doc", blocks)
    text_chunks = [c for c in chunks if c.content_type == "text"]
    table_chunks = [c for c in chunks if c.content_type == "table"]
    assert text_chunks
    assert text_chunks[0].title == "PART 23—AIRWORTHINESS STAND-ARDS"
    assert len(table_chunks) == 1
    assert table_chunks[0].table_html
