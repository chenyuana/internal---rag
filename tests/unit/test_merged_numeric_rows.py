from types import SimpleNamespace

import pytest

from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
from app.ingestion.parsers.table_recovery import structure_issue
from app.ingestion.parsers.verified_tables import (
    AISLE_ORIGINAL_CELLS,
    AISLE_VERIFIED_CELLS,
    FINAL_RULE_26324_SHA256,
    verified_table,
)
from app.ingestion.pipeline import PageRecord, table_rows_from_html, table_to_html


@pytest.mark.parametrize("label,values", [
    ("10 or fewer ....11 through 19 ...", ["11212", "1520"]),
    ("Small .... Medium .... Large ....", ["51015", "2436"]),
])
def test_flags_merged_numeric_rows_without_guessing(label, values):
    markup = table_to_html([["Category", "A", "B"], [label, *values]])
    assert structure_issue(markup) == "merged_numeric_rows"
    assert verified_table("unknown", 10, markup) is None


@pytest.mark.parametrize("label,value", [
    ("Single label ....", "11212"),
    ("See A ... and B ...", "See note"),
    ("Total", "1520"),
])
def test_keeps_unambiguous_or_nonnumeric_rows(label, value):
    assert structure_issue(table_to_html([["Label", "Value"], [label, value]])) is None


def test_verified_repair_requires_source_page_and_original_cells():
    markup = table_to_html(AISLE_ORIGINAL_CELLS)
    repaired = verified_table(FINAL_RULE_26324_SHA256, 10, markup)
    assert table_rows_from_html(repaired) == AISLE_VERIFIED_CELLS
    assert structure_issue(repaired) is None
    assert verified_table("other", 10, markup) is None
    assert verified_table(FINAL_RULE_26324_SHA256, 9, markup) is None
    assert verified_table(FINAL_RULE_26324_SHA256, 10, markup.replace("1520", "1521")) is None


@pytest.mark.parametrize("known_source", [True, False])
def test_recovery_preserves_evidence_and_withholds_unknown_values(known_source):
    markup = table_to_html(AISLE_ORIGINAL_CELLS)
    item = {"block_type": "table", "text": markup + "\n1 A narrower width ...",
            "table_html": markup}
    page = PageRecord(10, markup, rich_blocks=[
        {"text": "§ 23.815Width of aisle."}, item,
    ])
    parser = ScannedRegulatoryPdfParser(SimpleNamespace())
    parser._recover_tables(None, [page], allow_ocr=False,
                           source_hash=FINAL_RULE_26324_SHA256 if known_source else "other")
    assert item["raw_table_html"] == markup
    blocks = parser._build_blocks("test", [page])
    table = next(b for b in blocks if b.block_type == "table")
    assert table.section_path == ["§ 23.815 Width of aisle."]
    if known_source:
        assert table.table_rows == AISLE_VERIFIED_CELLS
        assert "¹12" in table.text and "1 A narrower width" in table.text
        assert "Column 1=" not in table.text
        assert "11212" not in table.text and "1520" not in table.text
    else:
        assert item["table_review_status"] == "needs_review"
        assert table.table_html is None
        assert "11212" not in table.text
