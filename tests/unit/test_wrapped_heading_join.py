"""A heading wrapped onto the next printed line must be joined.

Docket 75-31 p. 68 prints

    APPENDIX III - MISCELLANEOUS PROPOSALS REMOVED FROM CONSIDERATION FROM THE
    FIRST BIENNIAL AIRWORTHINESS REVIEW

as one heading over two lines.  Unjoined, the section label stops mid-phrase and
the tail line is handed to the table below as its caption, so the appendix table
was published under "FIRST BIENNIAL AIRWORTHINESS REVIEW" instead of the heading.
"""

from app.ingestion.pipeline import (
    PageRecord,
    _continues_heading,
    _join_wrapped_heading_lines,
    _title_repeats_section,
    clean_page,
    split_blocks,
)

HEAD = "APPENDIX III - MISCELLANEOUS PROPOSALS REMOVED FROM CONSIDERATION FROM THE"
TAIL = "FIRST BIENNIAL AIRWORTHINESS REVIEW"
FULL = f"{HEAD} {TAIL}"
TABLE_ROWS = [
    ["14 CFR (FAR Sec.)", "Proposal No.", "Agenda Item", "Proponent"],
    ["Pt. 21", "18", "L-21(Committee I).", "General Aviation Manufacturers Association."],
]


def test_all_caps_line_ending_on_a_connector_is_continued() -> None:
    assert _continues_heading(HEAD, TAIL) is True
    assert _join_wrapped_heading_lines([HEAD, TAIL, "Based on the FAA's review"]) == [
        FULL,
        "Based on the FAA's review",
    ]


def test_other_line_pairs_are_not_joined() -> None:
    # Ends on a content word, not a connector.
    assert (
        _continues_heading(
            "APPENDIX II MISCELLANEOUS PROPOSALS WITHDRAWN BY PROPONENT", TAIL
        )
        is False
    )
    # Fully formed heading already.
    assert (
        _continues_heading(
            "PART 23 - AIRWORTHINESS STANDARDS: NORMAL, UTILITY, AND ACROBATIC CATEGORY AIRPLANES",
            TAIL,
        )
        is False
    )
    # The following line is ordinary prose, not a heading.
    assert _continues_heading(HEAD, "Based on the FAA's review of the discussions") is False
    # Label-block lines are not headings.
    assert _continues_heading("CFR NPRM: Part 21|Part 23", "CITATION: [Federal Register]") is False


def test_clean_page_joins_the_wrapped_heading() -> None:
    page = PageRecord(
        page_number=68,
        raw_text=(
            f"{HEAD}\n\n{TAIL}\n\n"
            "Based on the FAA's review of the discussions at the Airworthiness Review Conference\n"
        ),
        cleaned_text="",
        route="native_text",
        indexable=True,
    )

    page.cleaned_text, _ = clean_page(page, set())

    assert FULL in page.cleaned_text.splitlines()


def _vector_page_with_wrapped_heading() -> PageRecord:
    return PageRecord(
        page_number=68,
        raw_text="",
        cleaned_text="",
        route="vector_layout",
        rich_blocks=[
            {"block_type": "paragraph", "text": HEAD},
            {"block_type": "paragraph", "text": TAIL},
            {
                "block_type": "paragraph",
                "text": "Based on the FAA's review of the discussions at the Airworthiness Review",
            },
            {
                "block_type": "table",
                "table_id": "t1",
                "table_title": TAIL,
                "text": "\n".join([TAIL, *("; ".join(row) for row in TABLE_ROWS)]),
                "table_rows": TABLE_ROWS,
                "table_header_rows": 1,
                "table_html": "<table><tr><th>14 CFR</th></tr></table>",
            },
        ],
        indexable=True,
    )


def test_vector_page_heading_and_table_share_the_full_section() -> None:
    blocks = split_blocks("doc", [_vector_page_with_wrapped_heading()])

    annex = next(block for block in blocks if block.block_type == "annex")
    assert annex.text == FULL
    table = next(block for block in blocks if block.table_id)
    # The table keeps the appendix section instead of the heading's tail line.
    assert table.section_path == [FULL]
    assert table.table_title == TAIL  # the block still carries the extracted caption


def test_a_title_already_inside_the_section_label_is_not_promoted() -> None:
    assert _title_repeats_section(TAIL, [FULL]) is True
    assert _title_repeats_section(TAIL, ["PART 135 - AIR TAXI OPERATORS"]) is False
    assert _title_repeats_section("Table 1--Calibration Table", [FULL]) is False
