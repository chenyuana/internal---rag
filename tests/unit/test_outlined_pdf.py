from types import SimpleNamespace

import pytest
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject

from app.ingestion.parsers.hybrid_pdf import HybridPdfParser
from app.ingestion.parsers.native_pdf import NativePdfParser
from app.ingestion.parsers.remote import RemoteParseResult
from app.ingestion.pipeline import has_outlined_text, is_watermark_only_page


def outline_pdf(path, count=150):
    writer = PdfWriter()
    page = writer.add_blank_page(width=595, height=842)
    stream = DecodedStreamObject()
    stream.set_data(b"10 10 m 11 12 13 14 15 16 c 11 12 13 14 15 16 c "
                    b"11 12 13 14 15 16 c 11 12 13 14 15 16 c h f\n" * count)
    page[NameObject('/Contents')] = writer._add_object(stream)
    writer.write(path)


def test_outline_detection_preserves_small_diagrams_and_searchable_text(tmp_path):
    path = tmp_path / 'paths.pdf'
    outline_pdf(path, 5)
    assert not has_outlined_text(PdfReader(path).pages[0], '')
    outline_pdf(path)
    page = PdfReader(path).pages[0]
    assert has_outlined_text(page, '')
    assert not has_outlined_text(page, 'Readable text remains available')


def test_small_image_does_not_hide_painted_body_as_watermark():
    class Page(dict):
        def get_contents(self):
            return SimpleNamespace(operations=[([], b'f')])

    page = Page({'/Resources': {'/XObject': {
        '/Table': SimpleNamespace(get_object=lambda: {
            '/Subtype': '/Image', '/Width': 391, '/Height': 173,
        }),
    }}})
    assert not is_watermark_only_page(page, '')


@pytest.mark.parametrize('recognized', [True, False])
def test_outline_pages_reach_ocr_and_empty_response_blocks_success(tmp_path, recognized):
    path = tmp_path / 'paths.pdf'
    outline_pdf(path)
    calls = []

    def parse_pages(source, pages, **kwargs):
        calls.append(pages)
        text = 'Control system requirements. The airplane must be safe. ' * 10
        return RemoteParseResult('mineru', page_texts={1: text} if recognized else {})

    parser = HybridPdfParser(
        NativePdfParser(),
        SimpleNamespace(enabled=True, name='mineru', parse_pages=parse_pages),
        SimpleNamespace(enabled=False),
        SimpleNamespace(enabled=False),
    )
    vector_calls = []
    def unexpected(*args, **kwargs):
        vector_calls.append(kwargs)
        raise AssertionError('Outlined text must not enter vector figure extraction')
    parser._vector.parse_pages = unexpected
    doc = parser.parse(path)
    assert not vector_calls
    assert calls and all(pages == [1] for pages in calls)
    if recognized:
        assert doc.pages[0].route == 'remote_ocr'
        assert doc.pages[0].cleaned_text
    else:
        assert any(g['gate'] == 'outlined_text_coverage' and g['status'] == 'fail'
                   for g in doc.qa['quality_gates'])
