"""Appendix banner attribution (Docket 75-19, pp. 28-31).

"APPENDIX I COMMITTEE IV (POWERPLANT) PROPOSALS DEFERRED GROUP 1" is printed as
a full-caps banner. Extraction glued it to the end of the preceding sentence and
to the sentence after it, so the heading event was lost and the whole appendix
-- its title page (p. 29) and its deferred-proposal table -- was published under
the section of the part that merely preceded it ("PART 35 ... Sec. 35.35 Blade
retention test.").
"""

from app.ingestion.pipeline import (
    PageRecord,
    build_chunks,
    clean_page,
    heading_kind,
    split_appendix_banner,
    split_blocks,
)

BANNER_I = "APPENDIX I COMMITTEE IV (POWERPLANT) PROPOSALS DEFERRED GROUP 1"
BANNER_II = "APPENDIX II COMMITTEE IV (POWERPLANT) PROPOSALS WITHDRAWN BY PROPONENT"
PART_35 = "PART 35 - AIRWORTHINESS STANDARDS; PROPELLERS"
SEC_35_35 = "Sec. 35.35 Blade retention test."
GLUED = (
    "Ref. Proposal No. 433; Sec. 35.35; Agenda Item P-84."
    f"{BANNER_I} Based upon the discussions at the Airworthiness Review Conference,"
    " the FAA has determined about:blank 28/35"
)


def _clean(pages: list[PageRecord]) -> list[PageRecord]:
    """Run the production cleaning pass so cleaned_text is built from raw_text."""
    for page in pages:
        page.cleaned_text, _ = clean_page(page, set())
    return pages


def _document_blocks():
    page_28 = PageRecord(
        page_number=28,
        raw_text=(
            f"{PART_35}\n{SEC_35_35}\n"
            "The hub and blade retention arrangement of propellers with detachable "
            "blades must be subjected to a centrifugal load.\n"
            f"{GLUED}\n"
        ),
        cleaned_text="",
        route="native_text",
        indexable=True,
    )
    page_29 = PageRecord(
        page_number=29,
        raw_text="",
        cleaned_text="",
        route="vector_layout",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "that the proposals listed below appear to have sufficient merit",
            },
            {
                "block_type": "table",
                "table_id": "table-29",
                "text": "14 CFR (FAR Sec.)=Pt. 23; Proposal No.=58; Agenda Item=L-56",
                "table_html": "<table><tr><th>14 CFR</th></tr><tr><td>Pt. 23</td></tr></table>",
                "table_rows": [["14 CFR"], ["Pt. 23"]],
                "table_header_rows": 1,
            },
        ],
        indexable=True,
    )
    return split_blocks("doc", _clean([page_28, page_29]))


def test_banner_is_split_out_of_the_glued_sentence() -> None:
    head, banner, tail = split_appendix_banner(GLUED)

    assert head == "Ref. Proposal No. 433; Sec. 35.35; Agenda Item P-84."
    assert banner == BANNER_I
    assert tail.startswith("Based upon the discussions")
    assert heading_kind(banner) == "annex"


def test_mid_sentence_appendix_reference_is_not_a_banner() -> None:
    prose = "See APPENDIX I for details of the deferred proposals."

    assert split_appendix_banner(prose) is None
    assert heading_kind(prose) is None


def test_glued_banner_attributes_the_following_pages_to_the_appendix() -> None:
    blocks = _document_blocks()

    banner = next(block for block in blocks if block.text == BANNER_I)
    assert banner.block_type == "annex"
    # The sentence the banner was glued to keeps its own section.
    head = next(
        block for block in blocks if block.text.rstrip().endswith("Agenda Item P-84.")
    )
    assert head.section_path == [PART_35, SEC_35_35]
    # Page 29 -- title page and deferred-proposal table -- follows the appendix.
    page_29 = [block for block in blocks if block.page_number == 29]
    assert page_29
    assert {tuple(block.section_path) for block in page_29} == {(BANNER_I,)}
    table = next(block for block in page_29 if block.table_id)
    assert table.section_path == [BANNER_I]
    # The banner is published as the chunks' section header (their context),
    # never duplicated into a chunk's body text.
    chunks = build_chunks("doc", blocks)
    table_chunk = next(chunk for chunk in chunks if chunk.table_ids)
    assert table_chunk.text.splitlines()[0] == BANNER_I
    assert all(chunk.text.count(BANNER_I) <= 1 for chunk in chunks)
    assert not any(chunk.text.strip() == BANNER_I for chunk in chunks)


def test_table_caption_banner_attributes_the_prose_above_the_table() -> None:
    page = PageRecord(
        page_number=30,
        raw_text="",
        cleaned_text="",
        route="vector_layout",
        rich_blocks=[
            {
                "block_type": "paragraph",
                "text": "The proposals listed below were withdrawn by their proponents.",
            },
            {
                "block_type": "table",
                "table_id": "table-30",
                "table_title": BANNER_II,
                "text": f"{BANNER_II}\n14 CFR (FAR Sec.)=23.993; Proposal No.=649",
                "table_html": "<table><tr><th>14 CFR</th></tr><tr><td>23.993</td></tr></table>",
                "table_rows": [["14 CFR"], ["23.993"]],
                "table_header_rows": 1,
            },
        ],
        indexable=True,
    )

    blocks = split_blocks("doc", _clean([page]))

    assert {tuple(block.section_path) for block in blocks if block.text} == {(BANNER_II,)}
    assert any(block.block_type == "annex" and block.text == BANNER_II for block in blocks)


def test_browser_page_footer_never_reaches_a_block() -> None:
    page = PageRecord(
        page_number=28,
        raw_text=(
            f"{PART_35}\n{SEC_35_35}\n"
            "The hub and blade retention arrangement of propellers must be tested."
            "about:blank 28/35\n"
        ),
        cleaned_text="",
        route="native_text",
        rich_blocks=[
            {"block_type": "paragraph", "text": "about:blank 29/35"},
            {"block_type": "paragraph", "text": "continued evidence on this page"},
        ],
        indexable=True,
    )

    blocks = split_blocks("doc", _clean([page]))

    assert all("about:blank" not in block.text for block in blocks)
    assert any(block.text == "continued evidence on this page" for block in blocks)
