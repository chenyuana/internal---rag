import pytest

from app.ingestion import pipeline


@pytest.mark.parametrize("caption", ["Table RF3", "Table AB12", "Table A.1"])
def test_numbered_grouped_header_keeps_each_applicability_column(caption):
    rows = [
        [caption, "", "", ""],
        ["", "Certification", "Airplane groups", "Airplane groups"],
        ["Section", "Design costs", "Turbojet", "Turboprop"],
        ["23.777", "-25", "X", "X"],
        ["23.831", "300000", "X", ""],
    ]
    page = pipeline.PageRecord(30, "", rich_blocks=[{
        "block_type": "table", "table_rows": rows,
        "table_html": pipeline.table_to_html(rows),
        "table_title": caption + " shows costs for the following groups:",
    }])
    pipeline.merge_cross_page_tables([page], "grouped")
    block = next(b for b in pipeline.split_blocks("grouped", [page]) if b.table_id)
    assert block.table_title == caption
    assert block.table_header_rows == 2
    assert len(block.table_rows) == 4
    assert "Airplane groups / Turbojet=X" in block.text
    assert "Airplane groups / Turboprop=X" in block.text
    assert "Columns " not in block.text
    assert caption not in pipeline.table_rows_from_html(block.table_html)[0]
    assert sum(len(c.table_rows) - 2 for c in pipeline.build_chunks("grouped", [block])) == 2


def test_repeated_data_flags_do_not_become_grouped_header():
    assert pipeline._grouped_table_header_rows([
        ["Section", "Cost", "A", "B"],
        ["23.777", "-25", "X", "X"],
        ["23.831", "300000", "X", ""],
    ]) == 1


def test_title_reference_still_rejects_unlabelled_prose():
    assert pipeline._table_reference("The table shows 3 categories") == ""
