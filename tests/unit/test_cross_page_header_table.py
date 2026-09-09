from app.ingestion.pipeline import PageRecord, _strip_page_header_cell, merge_cross_page_tables


def _table(rows, page):
    return {
        "block_type": "table",
        "table_rows": rows,
        "table_html": "<table></table>",
        "table_header_rows": 1,
        "asset_ids": [],
        "bbox": [30, 30, 560, 800],
        "source_page_start": page,
        "source_page_end": page,
        "text": "table",
    }


def test_numeric_values_are_not_page_document_headers():
    assert [_strip_page_header_cell(str(value)) for value in range(9, 100)] == [
        str(value) for value in range(9, 100)
    ]
    assert _strip_page_header_cell("DRS_88-1 values") == "values"
    assert _strip_page_header_cell("23-43 values") == "23-43 values"


def test_header_only_first_page_merges_two_numeric_continuations():
    pages = [
        PageRecord(
            2,
            "",
            rich_blocks=[
                _table([["Proposed No.", "Amendment No.", "Proposed No.", "Amendment No."]], 2)
            ],
        ),
        PageRecord(
            3,
            "",
            rich_blocks=[
                _table([[str(n), str(n), str(n + 44), str(n + 35)] for n in range(1, 36)], 3)
            ],
        ),
        PageRecord(
            4,
            "",
            rich_blocks=[
                _table([[str(n), str(n - 7), str(n + 44), str(n + 34)] for n in range(36, 45)], 4)
            ],
        ),
    ]
    summary = merge_cross_page_tables(pages, "test-document")
    table = next(block for block in pages[0].rich_blocks if block["block_type"] == "table")
    assert summary["cross_page_table_count"] == 1
    assert table["source_page_start"] == 2 and table["source_page_end"] == 4
    assert len(table["table_rows"]) == 45
    assert table["table_rows"][1] == ["1", "1", "45", "36"]
    assert table["table_rows"][-1] == ["44", "37", "88", "78"]


def test_wide_single_column_regulatory_table_is_canonicalized():
    pages = [
        PageRecord(
            17,
            "",
            rich_blocks=[
                _table(
                    [
                        ["Motion and effect"],
                        ["(1) Powerplant controls: Fuel...............Forward for open."],
                    ],
                    17,
                )
            ],
        )
    ]

    summary = merge_cross_page_tables(pages, "test-document")
    table = next(block for block in pages[0].rich_blocks if block["block_type"] == "table")

    assert summary["invalid_table_pages"] == []
    assert summary["structured_table_pages"] == [17]
    assert table["table_id"]
    assert table["table_rows"] == [
        ["Motion and effect"],
        ["(1) Powerplant controls: Fuel...............Forward for open."],
    ]
