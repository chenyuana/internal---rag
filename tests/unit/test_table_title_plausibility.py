"""A table's "caption" must not be a truncated value from the table itself.

Docket 75-31 p. 65: the appendix table is grouped by proponent. The group label
printed above that page's rows was extracted as the table's caption
("General Aviation Manufacturers", a truncated form of the value
"General Aviation Manufacturers Association." in the table), and publishing it
as ``section_path`` displaced the real section -- the whole table appeared under
a proponent's name instead of "APPENDIX I - MISCELLANEOUS PROPOSALS DEFERRED.".
"""

from app.ingestion.pipeline import (
    PageRecord,
    clean_page,
    split_blocks,
)

APPENDIX = "APPENDIX I - MISCELLANEOUS PROPOSALS DEFERRED."
GROUP_ROWS = [
    ["14 CFR (FAR Sec.)", "Proposal No.", "Agenda Item", "Proponent"],
    ["23.347", "81", "C-15", "General Aviation Manufacturers Association."],
    ["25.631", "218", "H-33", "General Aviation Manufacturers Association."],
]


def _table_part(title: str, rows: list[list[str]]) -> dict:
    records = [
        "; ".join(f"{head}={cell}" for head, cell in zip(rows[0], row, strict=False))
        for row in rows[1:]
    ]
    return {
        "block_type": "table",
        "table_id": "t1",
        "table_title": title,
        "text": "\n".join([title, *records]),
        "table_rows": rows,
        "table_header_rows": 1,
        "table_html": "<table><tr><th>14 CFR</th></tr><tr><td>23.347</td></tr></table>",
    }


def _blocks(*, title: str, rows: list[list[str]], with_sibling: bool) -> list:
    pages = [
        PageRecord(
            page_number=64,
            raw_text="APPENDIX I - MISCELLANEOUS PROPOSALS DEFERRED.\n",
            cleaned_text="",
            route="native_text",
            indexable=True,
        )
    ]
    sibling = [_table_part("General Aviation Manufacturers", GROUP_ROWS)] if with_sibling else []
    pages.append(
        PageRecord(
            page_number=65,
            raw_text="",
            cleaned_text="",
            route="vector_layout",
            rich_blocks=[
                {
                    "block_type": "paragraph",
                    "text": "Group 1. Based upon the discussions at the Airworthiness Review",
                }
            ]
            + sibling,
            indexable=True,
        )
    )
    pages.append(
        PageRecord(
            page_number=66,
            raw_text="",
            cleaned_text="",
            route="vector_layout",
            rich_blocks=[_table_part(title, rows)],
            indexable=True,
        )
    )
    for page in pages:
        page.cleaned_text, _ = clean_page(page, set())
    return split_blocks("doc", pages)


def test_group_label_caption_does_not_displace_the_appendix_section() -> None:
    blocks = _blocks(title="General Aviation Manufacturers", rows=GROUP_ROWS, with_sibling=False)

    table = next(block for block in blocks if block.table_id)
    assert table.section_path == [APPENDIX]
    # The label survives in the table's own text (it is evidence).
    assert "General Aviation Manufacturers" in table.text


def test_truncated_value_from_a_sibling_table_is_also_rejected() -> None:
    # "Joint Airworthiness Requirements" is a truncated value of a proponent in
    # the previous page's table, not a caption.
    blocks = _blocks(
        title="Joint Airworthiness Requirements",
        rows=[
            ["14 CFR (FAR Sec.)", "Proponent"],
            ["21.17", "Joint Airworthiness Requirements Committee."],
        ],
        with_sibling=True,
    )

    table = next(block for block in blocks if block.table_id)
    assert table.section_path == [APPENDIX]


def test_real_captions_are_still_promoted() -> None:
    for caption in (
        "Table 1--Calibration Table",
        "Sec. 25.101 General",
        "Table B1 Certification Standard Atmospheric Rain Concentration",
    ):
        blocks = _blocks(title=caption, rows=GROUP_ROWS, with_sibling=True)
        table = next(block for block in blocks if block.table_id and block.page_number == 66)
        assert table.section_path == [caption], caption


def test_wrapped_tail_caption_is_rejected() -> None:
    rows = [["Control", "Maximum forces or torques"], ["Aileron", "100 lbs"]]
    page = PageRecord(
        page_number=1,
        raw_text="SUPPLEMENTARY INFORMATION:\n",
        cleaned_text="",
        route="native_text",
        indexable=True,
    )
    page.cleaned_text, _ = clean_page(page, set())
    page2 = PageRecord(
        page_number=2,
        raw_text="",
        cleaned_text="",
        route="vector_layout",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": (
                    "In addition to Notice No. 74-33, the following Airworthiness "
                    "Review Program Notices of"
                ),
            },
            _table_part("Proposed Rule Making have been issued:", rows),
        ],
        indexable=True,
    )
    blocks = split_blocks("doc", [page, page2])

    table = next(block for block in blocks if block.table_id)
    assert table.section_path == ["SUPPLEMENTARY INFORMATION:"]
