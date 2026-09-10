"""Label lines: shape decides *what* is a label, position decides *whether* it divides.

Federal Register notices print their masthead labels (CFR NPRM / CITATION /
DOCKET NUMBER / SUBJECT / ACTION) in exactly the same shape as their division
headers (SUMMARY / DATES / ADDRESSES / SUPPLEMENTARY INFORMATION). Listing the
division names was the earlier approach; the general rule is that the masthead
comes first and stops at the first bare label or the first narrative paragraph.
"""

from app.ingestion.pipeline import (
    PageRecord,
    build_chunks,
    heading_kind,
    is_bare_label,
    split_blocks,
    split_label_line,
)

MASTHEAD = (
    "CFR NPRM: Part 21|Part 23|Part 25\n"
    "31|Part 33|Part 35\n"
    "CITATION: [Federal Register: June 10, 1975]\n"
    "DOCKET NUMBER: 14685\n"
    "SUBJECT: Airworthiness Review Program, Notice No. 7\n"
    "ACTION: Proposed Rules\n"
)
NARRATIVE = "The Federal Aviation administration is considering amending Parts 23, 25, 27 and 29."


def _blocks(*texts: str):
    pages = [
        PageRecord(
            page_number=index + 1,
            raw_text=text,
            cleaned_text="",
            route="native_text",
            indexable=True,
        )
        for index, text in enumerate(texts)
    ]
    from app.ingestion.pipeline import clean_page

    for page in pages:
        page.cleaned_text, _ = clean_page(page, set())
    return split_blocks("doc", pages)


def test_label_shape_is_recognized_without_a_name_list() -> None:
    assert split_label_line("SUPPLEMENTARY INFORMATION:") == ("SUPPLEMENTARY INFORMATION:", "")
    assert split_label_line("DATES: Comments must be received.") == (
        "DATES:",
        "Comments must be received.",
    )
    # Any all-caps label works, including ones never seen before.
    assert split_label_line("FOR FURTHER INFORMATION:") is not None
    assert is_bare_label("SUMMARY:") is True
    assert is_bare_label("SUMMARY: text") is False
    # Prose with a colon, and a lowercase label, are not labels.
    assert split_label_line("This is a paragraph: with a colon.") is None
    assert split_label_line("Summary: the notice amends Part 23.") is None
    # heading_kind no longer knows any label names: position decides.
    assert heading_kind("SUMMARY:") is None


def test_masthead_labels_do_not_open_sections() -> None:
    blocks = _blocks(MASTHEAD + "SUMMARY:\n" + NARRATIVE)

    by_text = {block.text.strip(): block for block in blocks}
    for label in ("CFR NPRM:", "CITATION:", "DOCKET NUMBER:", "SUBJECT:", "ACTION:"):
        assert by_text[label].section_path == [], label
    # The first bare label ends the masthead and opens the first section.
    assert by_text["SUMMARY:"].section_path == ["SUMMARY:"]
    assert by_text[NARRATIVE].section_path == ["SUMMARY:"]


def test_labels_after_the_masthead_open_sections() -> None:
    blocks = _blocks(
        MASTHEAD
        + "SUMMARY:\n"
        + NARRATIVE
        + "\nDATES: Comments must be received on or before September 8, 1975.\n"
        + "ADDRESSES:\n"
        + "Docket Management System, 800 Independence Avenue.\n"
        + "SUPPLEMENTARY INFORMATION:\n"
        + "This is the seventh in a series of notices.\n"
    )

    by_text = {block.text.strip(): block for block in blocks}
    assert by_text["DATES:"].section_path == ["DATES:"]
    assert (
        by_text["Comments must be received on or before September 8, 1975."].section_path
        == ["DATES:"]
    )
    assert by_text["ADDRESSES:"].section_path == ["ADDRESSES:"]
    assert by_text["SUPPLEMENTARY INFORMATION:"].section_path == ["SUPPLEMENTARY INFORMATION:"]
    assert by_text["This is the seventh in a series of notices."].section_path == [
        "SUPPLEMENTARY INFORMATION:"
    ]


def test_narrative_without_a_bare_label_still_closes_the_masthead() -> None:
    blocks = _blocks(MASTHEAD + NARRATIVE + "\nADDRESSES:\nDocket Management System.\n")

    by_text = {block.text.strip(): block for block in blocks}
    assert by_text["ADDRESSES:"].section_path == ["ADDRESSES:"]


def test_a_part_heading_with_an_inline_value_stays_one_heading() -> None:
    blocks = _blocks(
        "PART 23 -- AIRWORTHINESS STANDARDS: NORMAL, UTILITY, ACROBATIC CATEGORY AIRPLANES\n"
        "Sec. 23.1 Applicability.\n"
    )

    headings = [block for block in blocks if block.block_type == "chapter"]
    assert headings
    assert headings[0].text == (
        "PART 23 -- AIRWORTHINESS STANDARDS: NORMAL, UTILITY, ACROBATIC CATEGORY AIRPLANES"
    )
    # The Part heading is one section label, not a label plus a stray paragraph.
    assert headings[0].section_path == [headings[0].text]


def test_published_chunk_carries_the_label_section() -> None:
    blocks = _blocks(
        MASTHEAD
        + "SUPPLEMENTARY INFORMATION:\n"
        + "In addition to Notice No. 74-33, the following notices have been issued:\n"
        + "Notice No.=75-10; Federal Register Citation=(40 FR 10802; Mar. 7, 1975).\n"
    )
    chunks = build_chunks("doc", blocks)

    assert chunks
    assert all(
        chunk.section_path[:1] == ["SUPPLEMENTARY INFORMATION:"]
        for chunk in chunks
        if "Notice No.=75-10" in chunk.text or "In addition" in chunk.text
    )
