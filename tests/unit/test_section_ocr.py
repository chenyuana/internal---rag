from copy import deepcopy
from types import SimpleNamespace

import pytest

from app.ingestion.parsers.hybrid_pdf import HybridPdfParser
from app.ingestion.parsers.section_ocr import (
    _VERIFIED_SOURCE,
    finish_section_ocr,
    prepare_section_ocr,
)
from app.ingestion.pipeline import (
    PageRecord,
    _split_embedded_tables,
    build_chunks,
    merge_cross_page_tables,
    split_blocks,
    table_rows_from_html,
    table_to_html,
    table_to_semantic_text,
)
from app.ingestion.regulations import match_article_heading

METADATA = [
    ["SUBJECT GROUP:", "Control Surface and System Loads"],
    ["SECTION:", "23.415"],
    ["AMENRMBENT", "23-48"],
    ["EFFECTIVE DATE:", "07/25/2017"],
]
ROWS = [
    ["Surface", "K", "Position of controls"],
    ["(a) Aileron", "0.75", "Control column locked lashed in mid-position."],
    [
        "(b) Aileron",
        "±0.50",
        "Ailerons at full throw; + moment on one aileron, – momenton the other.",
    ],
    ["(c)Elevator", "±0.75", "(c) Elevator full up (−)."],
    ["(d)Elevator", "", "(d) Elevator full down (+)."],
    ["(e) Rudder", "±0.75", "(e) Rudder in neutral."],
    ["(f) Rudder", "", "(f) Rudder at full throw."],
]
FORMULA = [
    "H = K c S q",
    "where -",
    "H = limit hinge moment (ft.-Ibs.);",
    "c = mean chord of the control surface aft of the hinge line (ft.);",
    "S = area of control surface aft of the hinge line (sq. ft.);",
    "q = dynamic pressure (p.s.f.) based on a design speed not less than "
    "14.6 √(W/S) + 14.6 (f.p.s.) where W/S = wing loading at design maximum weight, "
    "except that the design speed need not exceed 88 (f.p.s.);",
    "K = limit hinge moment factor for ground gusts derived in paragraph (b). "
    "of this section. (For ailerons and elevators, a positive value of K indicates "
    "a moment tending to depress the surface and a negative value of K indicates "
    "a moment tending to raise the surface).",
]
TITLE = "§ 23.415 Ground gust conditions."


def paragraph(text, top):
    return {"block_type": "paragraph", "text": text, "bbox": [40, top, 500, top + 10]}


def sample_document(source_hash=_VERIFIED_SOURCE):
    first = [
        {
            "block_type": "table",
            "text": "",
            "table_html": table_to_html(METADATA),
            "bbox": [30, 30, 510, 100],
        }
    ]
    first.extend(
        paragraph(text, 150 + index * 30)
        for index, text in enumerate(
            [
                TITLE,
                "(a)The control system must be investigated as follows:",
                "(1) Control surface loads due to ground gusts.",
                "(2) The minimums specified in § 23.397(b). are used according to the formula:",
                *FORMULA,
                "(b) The limit hinge moment factor K for ground gusts must be derived as follows:",
            ]
        )
    )
    second = [
        {
            "block_type": "table",
            "text": "",
            "table_html": table_to_html(ROWS),
            "table_title": "Sec. 23.415",
            "bbox": [80, 30, 500, 280],
        },
        paragraph("(c) At all weights the airplane is tied down for winds up to 65 knots.", 300),
        paragraph("[Doc. No. 4080, 29 FR 17955, Dec. 18, 1964]", 400),
    ]
    return SimpleNamespace(
        pages=[
            PageRecord(1, "original OCR p1", route="remote_ocr", rich_blocks=first),
            PageRecord(2, "original OCR p2", route="remote_ocr", rich_blocks=second),
        ],
        assets=[],
        source_hash=source_hash,
        document_id="test-doc",
    )


def process(document):
    prepare_section_ocr(document)
    merge_cross_page_tables(document.pages, document.document_id)
    blocks = split_blocks(document.document_id, document.pages)
    chunks = build_chunks(document.document_id, blocks)
    gates = finish_section_ocr(document, blocks, chunks)
    return blocks, chunks, gates


def test_english_table_no_synthetic_chinese_record_prefix():
    text = table_to_semantic_text(ROWS)
    assert "记录" not in text
    assert "Surface=(f) Rudder" in text
    assert "K=±0.50" in text
    assert "K=" not in text.splitlines()[-1]


def test_label_value_form_keeps_all_rows_without_reusing_first_row_as_header():
    text = table_to_semantic_text(METADATA, title="Document information")
    assert text.splitlines() == [
        "Document information",
        *[f"{row[0].rstrip(':')}: {row[1]}" for row in METADATA],
    ]
    assert "Control Surface and System Loads=" not in text


def test_chinese_table_retains_existing_labels():
    text = table_to_semantic_text([["项目", "限值"], ["速度", "100"]])
    assert text == "记录1：项目=速度；限值=100"


def test_duplicate_english_headers_have_english_neutral_labels():
    assert "Columns 1-2=A" in table_to_semantic_text([["Item", "Item"], ["A", "A"]])


def test_labeled_data_rows_are_not_footnotes_even_with_empty_k_cells():
    segments, notes = _split_embedded_tables(ROWS, title="", page_number=2)
    assert len(segments) == 1 and not notes
    assert segments[0][1] == ROWS


@pytest.mark.parametrize(
    "note",
    [
        ["(a) This note applies to all rows.", "", ""],
        ["(a) This note applies to all rows."] * 3,
    ],
)
def test_real_fullwidth_note_can_still_be_separated(note):
    segments, notes = _split_embedded_tables([*ROWS, note], title="", page_number=2)
    assert segments[0][1] == ROWS
    assert len(notes) == 1


def test_section_heading_and_inline_reference_are_distinguished():
    assert match_article_heading(TITLE).normalized_id == "23.415"
    assert (
        match_article_heading("Sec. 33.77 Foreign object ingestion.").normalized_id
        == "33.77"
    )
    assert (
        match_article_heading("Section 33.77 Foreign object ingestion.").normalized_id
        == "33.77"
    )
    assert match_article_heading("Use § 23.415 for loads.") is None
    assert match_article_heading("§ 23.415") is None


def test_end_to_end_sections_tables_formula_and_source_evidence():
    document = sample_document()
    original = deepcopy(document.pages[0].rich_blocks)
    blocks, chunks, gates = process(document)
    assert not gates
    heading = next(block for block in blocks if block.text == TITLE)
    assert heading.block_type == "clause"
    assert heading.article_id_normalized == "23.415"
    assert all(chunk.title == TITLE for chunk in chunks if chunk.article_id_normalized)
    body = [block for block in blocks if block.page_number == 2]
    assert len(body) == 3  # intact table, (c), independent amendment history
    assert all(block.article_id_normalized == "23.415" for block in body)
    table = body[0]
    assert table.block_type == "table"
    assert len(table.table_rows) == 7
    assert table.table_rows[4][1] == table.table_rows[6][1] == ""
    assert table_rows_from_html(table.table_html) == table.table_rows
    assert not table.table_title
    assert "moment on the other" in table.text
    formula = next(block for block in blocks if block.block_type == "formula")
    assert formula.latex == FORMULA[0]
    assert all(f"{variable} =" in formula.text for variable in "HcSqK")
    assert "14.6 √(W/S) + 14.6" in formula.text
    assert "ft.-lbs." in formula.text and "paragraph (b) of" in formula.text
    assert any(formula.text in chunk.text for chunk in chunks)
    assert document.pages[0].raw_text == "original OCR p1"
    assert document.pages[0].layout_features["ocr_structure_source"] == original
    assert "## " + TITLE in document.pages[0].cleaned_text
    assert "\n\n[Doc. No." in document.pages[1].cleaned_text
    assert "AMENDMENT NUMBER: 23-48" in document.pages[0].cleaned_text


def test_verified_spelling_not_applied_to_other_source_hash():
    document = sample_document("another-source")
    blocks, _, gates = process(document)
    assert not gates
    text = "\n".join(block.text for block in blocks)
    assert "AMENRMBENT" in text
    assert "ft.-Ibs." in text
    assert "momenton" in text


@pytest.mark.parametrize("modification", ["no_section", "native_route", "missing_variable"])
def test_conservative_guards(modification):
    document = sample_document()
    if modification == "no_section":
        document.pages[0].rich_blocks[1]["text"] = "General document"
    elif modification == "native_route":
        for page in document.pages:
            page.route = "native_text"
    else:
        document.pages[0].rich_blocks = [
            item
            for item in document.pages[0].rich_blocks
            if not item.get("text", "").startswith("q =")
        ]
    prepare_section_ocr(document)
    assert not any(item["block_type"] == "formula" for item in document.pages[0].rich_blocks)


def test_gate_detects_html_retrieval_disagreement_and_split_formula():
    document = sample_document()
    blocks, chunks, _ = process(document)
    table = next(block for block in blocks if block.page_number == 2)
    table.table_rows = table.table_rows[:2]
    gates = finish_section_ocr(document, blocks, [])
    assert {page for gate in gates for page in gate["pages"]} == {1, 2}
    assert all(gate["status"] == "fail" for gate in gates)
    assert chunks


def test_hybrid_parser_version_is_updated():
    assert HybridPdfParser.version == "4"
