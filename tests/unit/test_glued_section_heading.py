import pytest

from app.ingestion.parsers.scan_regulatory import (
    ScannedRegulatoryPdfParser,
    english_heading,
)
from app.ingestion.pipeline import PageRecord


@pytest.mark.parametrize("number,title", [
    ("23.815", "Width of aisle."),
    ("25.771", "Pilot compartment."),
    ("27.807", "Emergency exits."),
])
def test_glued_section_title_updates_table_context(number, title):
    heading = f"§ {number} {title}"
    assert english_heading(f"§ {number}{title}") == ("section", heading)
    parser = ScannedRegulatoryPdfParser(None)
    pages = [PageRecord(page_number=10, raw_text="", rich_blocks=[
        {"text": "§23.813 Emergency exit access."},
        {"text": f"§ {number}{title}"},
        {"text": "(b) The following values apply:"},
        {"block_type": "table", "table_html":
         "<table><tr><td>Seats</td><td>Width</td></tr>"
         "<tr><td>10</td><td>15</td></tr></table>"},
    ])]
    blocks = parser._build_blocks("glued-heading", pages)
    table = next(b for b in blocks if b.block_type == "table")
    assert table.section_path == [heading]
    chunks = parser._build_chunks("glued-heading", blocks)
    assert next(c for c in chunks if c.table_ids).section_path == [heading]


@pytest.mark.parametrize("text", [
    "See § 23.815Width of aisle.",
    "§ 23.815requires a minimum width.",
    "§ 23.815(a) applies to this airplane.",
    "23.815Width of aisle.",
    "§ 23.815A applies to this airplane.",
])
def test_does_not_promote_glued_references(text):
    assert english_heading(text) is None
