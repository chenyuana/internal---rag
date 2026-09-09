"""Isolated OCR validation for a PDF whose text was converted to paths."""
import argparse
from pathlib import Path

from app.core.config import load_settings
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.pipeline import write_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    settings = load_settings().ingestion
    assert settings.mineru.enabled
    assert not settings.figure_vlm.enabled
    doc = ParserRegistry.with_builtins(settings).parse(
        args.source,
        progress_callback=lambda done, total: print(f'OCR {done}/{total}', flush=True),
    )
    result = write_document(args.output, doc)
    print(result['output_folder'], flush=True)
    print('pages', len(doc.pages), 'chunks', len(doc.chunks), flush=True)
    for page in doc.pages:
        print(page.page_number, page.route, len(page.cleaned_text), flush=True)
    print('gates', doc.qa['quality_gates'], flush=True)
    assert len(doc.pages) == 69
    assert all(page.route == 'remote_ocr' and page.cleaned_text.strip() for page in doc.pages)
    assert not doc.qa.get('watermark_pages')
    assert not any('页流程图' in chunk.text for chunk in doc.chunks)
    print('PASS: 69 pages recognized; no watermark exclusions or diagram placeholders.', flush=True)


if __name__ == '__main__':
    main()
