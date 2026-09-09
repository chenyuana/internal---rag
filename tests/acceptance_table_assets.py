"""Offline real-document evidence recovery; preserves the active job output."""
import json
from pathlib import Path
from unittest.mock import Mock

from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
from app.ingestion.pipeline import write_document


def main():
    root = Path('D:/internal-rag')
    previous_root = root / 'data/ingestion/outputs/54fea5847cb24b1f9f06c907e017d844/Docket1186__v0001__846D96E4__052b3a89'
    previous = json.loads((previous_root / 'document_ir.json').read_text(encoding='utf-8'))
    source = Path(previous['source']['path'])
    client = Mock()
    client.parse_pages.side_effect = AssertionError('Unexpected OCR')
    client.ocr_table_lines.side_effect = AssertionError('Unexpected table OCR')
    # Cache cleaning uses the normal text conversion, never a remote request.
    from app.ingestion.parsers.remote import MinerUClient
    client._item_text = MinerUClient._item_text
    parser = ScannedRegulatoryPdfParser(client)
    document = parser.reprocess_page(source, previous, previous_root, 4, 'clean')
    tables = [b for b in document.blocks if b.page_number == 4 and b.block_type == 'table']
    assert len(tables) == 3 and all(len(b.asset_ids) == 1 for b in tables)
    assert [len(b.table_rows) for b in tables] == [5, 13, 9]
    assert len(document.pages[3].table_hints) == 3
    assert all(p.cleaned_text == old['cleaned_text'] for p, old in zip(document.pages[:3], previous['pages'][:3]))
    output = Path(write_document(root / 'tmp/pdfs/docket1186-table-assets-verified', document,
                                display_name='Docket1186__v0001__846D96E4.pdf')['output_folder'])
    updated = json.loads((output / 'document_ir.json').read_text(encoding='utf-8'))
    again = parser.reprocess_page(source, updated, output, 4, 'clean')
    assert [a.asset_id for a in again.assets] == [a.asset_id for a in document.assets]
    for table in tables:
        assert any(table.block_id in c.block_ids and table.asset_ids[0] in c.asset_ids for c in document.chunks)
    print(json.dumps({'passed': True, 'output': str(output), 'table_images': [str(output / 'assets' / a.filename) for a in document.assets if a.page_number == 4]}, indent=2))


if __name__ == '__main__':
    main()
