from app.ingestion.parsers.table_review import table_review_gate
from app.ingestion.pipeline import PageRecord


def test_repaired_ocr_still_requires_review_from_original_evidence():
    page = PageRecord(12, '', layout_features={'ocr_blocks': [
        {'block_type': 'table', 'raw_ocr_item': {'table_body': '<table></table>'}}
    ]})
    gate = table_review_gate([page], {'merged_cell_review_pages': []})
    assert gate['status'] == 'fail'
    assert gate['pages'] == [12]


def test_cross_page_table_includes_all_source_pages():
    page = PageRecord(23, '', rich_blocks=[{'block_type': 'table',
        'source_page_start': 23, 'source_page_end': 24}])
    assert table_review_gate([page], {})['pages'] == [23, 24]


def test_simple_native_table_does_not_require_extra_gate():
    page = PageRecord(1, '', route='native_text', rich_blocks=[{'block_type': 'table'}])
    assert table_review_gate([page], {}) is None


def test_detected_risk_requires_review_without_ocr():
    assert table_review_gate([], {'merged_cell_review_pages': [4]})['pages'] == [4]


def test_scanned_parser_gate_survives_successful_structure_recovery():
    from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
    from types import SimpleNamespace
    page = PageRecord(1, '', route='scan_regulatory_ocr', rich_blocks=[
        {'block_type': 'table', 'table_review_status': 'recovered',
         'table_structure_issue': 'multiple_labeled_rows'}])
    qa = ScannedRegulatoryPdfParser(SimpleNamespace())._quality([page], [], [], [], [], [])
    assert qa['table_review_status'] == 'review_required'
    assert any(g['gate'] == 'table_manual_review' and g['status'] == 'fail'
               for g in qa['quality_gates'])
