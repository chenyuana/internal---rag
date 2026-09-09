from types import SimpleNamespace

import pytest

from app.ingestion.parsers.hybrid_pdf import HybridPdfParser
from app.ingestion.parsers.remote import RemoteParseResult
from app.ingestion.pipeline import PageRecord, is_glyph_name_garbage, parse_pdf

BROKEN = "/i255".join("".join(f"/{n}" for n in range(40)) for _ in range(3))


def test_dense_numeric_font_codes_are_detected_despite_high_alnum_ratio():
    assert sum(c.isalnum() for c in BROKEN) / len(BROKEN) > .3
    assert is_glyph_name_garbage(BROKEN)
    assert is_glyph_name_garbage("".join(f"/{n}" for n in range(120)))


@pytest.mark.parametrize("text", [
    "Date 2026/08/31. Ratios 1/2 and 3/4. Control surface loads. " * 30,
    "2026/08/31\n" * 100,
    "Use /assets/23/48/source.pdf and /api/v1/documents/12/34 to read the source. " * 20,
    "压力/高度/温度：100/200/300；1/2；GB/T 1234 标准正文。" * 30,
    "§ 23.415 Ground gust conditions. H = K c S q; q = 14.6 sqrt(W/S). " * 20,
    " /0/1/2/i255 ",
])
def test_legitimate_slashes_numbers_and_formulas_not_flagged(text):
    assert not is_glyph_name_garbage(text)


def test_mixed_pdf_routes_only_corrupt_page_without_requiring_images(monkeypatch, tmp_path):
    class Page:
        def __init__(self, text):
            self.text = text

        def extract_text(self):
            return self.text

        def get(self, key, default=None):
            return default

    source = tmp_path / "mixed.pdf"
    source.write_bytes(b"fixture")
    monkeypatch.setattr("app.ingestion.pipeline.PdfReader", lambda _: SimpleNamespace(
        pages=[Page(BROKEN), Page("Normal searchable document. Control system requirements. " * 10)]
    ))
    doc = parse_pdf(source)
    assert doc.pages[0].route == "text_layer_review"
    assert doc.pages[0].text_layer_corruption["glyph_name_garbage"]
    assert doc.pages[0].image_count == 0
    assert doc.pages[1].route == "native_text"


def test_bad_ocr_response_cannot_be_marked_successful():
    page = PageRecord(1, BROKEN, route="text_layer_review")
    doc = SimpleNamespace(pages=[page], assets=[])
    result = RemoteParseResult("mineru", page_texts={1: BROKEN})
    assert HybridPdfParser._merge_pages(doc, result, {1}, route="remote_ocr") == set()
    assert page.route == "text_layer_review"
    assert page.raw_text == BROKEN
