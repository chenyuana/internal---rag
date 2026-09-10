"""A table must travel with the sentence that names its subject.

Real case (Docket 75-10, page 24): "Sec. 25.397 Control system loads." carries
the table of *Limit pilot forces and torques*, but the caption is extracted as
an ordinary paragraph. The chunker published the caption as its own sibling
chunk, so the table reached retrieval under a broader title than its content
and the keyword anchors never mentioned pilot forces at all.
"""

from app.ingestion.pipeline import BlockRecord, build_chunks
from app.ingestion.publishers import RagflowPublisher
from app.ingestion.regulations import extract_keywords, table_lead_in_line

PART_25 = "PART 25 - AIRWORTHINESS STANDARDS: TRANSPORT CATEGORY AIRPLANES"
SEC_25_397 = "Sec. 25.397 Control system loads."
CAPTION = (
    "(c) Limit pilot forces and torques. The limit pilot forces and torques "
    "are as follows:"
)
TABLE_ROWS = [
    ["Control", "Maximum forces or torques", "Minimum forces or torques"],
    ["Aileron: Stick Wheel*", "100 lbs 80 D in.-lbs.**", "40 lbs. 40 D in.-lbs."],
    [
        "Elevator: Stick Wheel (symmetrical). Wheel (unsymmetrical).",
        "250 lbs. 300 lbs.",
        "100 lbs. 100 lbs. 100 lbs.",
    ],
    ["Rudder", "300 lbs.", "130 lbs."],
]
ARTICLE_ALIASES = [
    "25.397(c)",
    "25.397",
    "第25.397条",
    "25.397(C)",
    "第25.397条(C)",
    "第25.397条C",
    "第25.397条第C款",
    "25.397条C",
]


def _caption_block() -> BlockRecord:
    return BlockRecord(
        block_id="caption",
        block_type="paragraph",
        text=CAPTION,
        page_number=24,
        section_path=[PART_25, SEC_25_397],
        article_id_raw="25.397(c)",
        article_id_normalized="25.397(C)",
        article_parent_id="25.397",
        article_aliases=list(ARTICLE_ALIASES),
    )


def _table_block() -> BlockRecord:
    return BlockRecord(
        block_id="table",
        block_type="table",
        text="\n".join(
            "; ".join(
                f"{header}={value}"
                for header, value in zip(TABLE_ROWS[0], row, strict=False)
            )
            for row in TABLE_ROWS[1:]
        ),
        page_number=24,
        section_path=[PART_25, SEC_25_397],
        article_id_raw="25.397(c)",
        article_id_normalized="25.397(C)",
        article_parent_id="25.397",
        article_aliases=list(ARTICLE_ALIASES),
        asset_id="p0024-table-01",
        asset_ids=["p0024-table-01"],
        source_page_start=24,
        source_page_end=24,
        table_id="dc4504d16a22bb5598d3",
        table_rows=TABLE_ROWS,
        table_header_rows=1,
        table_row_pages=[24] * len(TABLE_ROWS),
    )


def _chunks(blocks: list[BlockRecord]):
    return build_chunks("doc", blocks)


def test_table_chunk_carries_the_caption_that_introduces_it() -> None:
    chunks = _chunks([_caption_block(), _table_block()])

    tables = [chunk for chunk in chunks if chunk.content_type == "table"]
    assert len(tables) == 1
    table = tables[0]
    assert CAPTION in table.text
    # The caption is evidence in its own right, so it stays attributable to
    # its own block instead of being copied into the table chunk anonymously.
    assert table.block_ids == ["caption", "table"]
    assert table.asset_ids == ["p0024-table-01"]
    assert table.page_start == 24


def test_caption_is_not_published_twice() -> None:
    chunks = _chunks([_caption_block(), _table_block()])

    joined = "\n".join(chunk.text for chunk in chunks)
    assert joined.count(CAPTION) == 1
    # No leftover sibling chunk that only repeats the caption.
    assert not [chunk for chunk in chunks if chunk.text.strip().endswith("as follows:")]


def test_substantive_paragraph_stays_in_its_own_chunk() -> None:
    body = BlockRecord(
        block_id="body",
        block_type="paragraph",
        text="The suitability and durability of materials must be established.",
        page_number=24,
        section_path=[PART_25, SEC_25_397],
    )

    chunks = _chunks([body, _table_block()])

    assert [chunk.content_type for chunk in chunks] == ["text", "table"]
    assert body.text not in chunks[1].text
    assert body.text in chunks[0].text


def test_detects_only_real_captions() -> None:
    assert table_lead_in_line(CAPTION) == CAPTION
    assert table_lead_in_line(f"{PART_25}\n{SEC_25_397}\n{CAPTION}\nx=1") == CAPTION
    assert table_lead_in_line("(a) Each control system must have stops.") is None
    assert table_lead_in_line("") is None


def test_keywords_anchor_the_caption_instead_of_the_part_banner() -> None:
    table = next(
        chunk for chunk in _chunks([_caption_block(), _table_block()])
        if chunk.content_type == "table"
    )

    keywords = extract_keywords(
        table.title,
        table.section_path,
        table.article_aliases,
        text=table.text,
    )

    assert {"Limit", "pilot", "forces", "torques"} <= set(keywords)
    # Bilingual counterparts, so a Chinese question can reach an English table.
    assert "副翼" in keywords
    assert "25.397(c)" in keywords
    # "PART 25 - AIRWORTHINESS STANDARDS: TRANSPORT CATEGORY AIRPLANES" repeats
    # on every Part-25 chunk and must not spend the keyword budget.
    assert not {"AIRWORTHINESS", "STANDARDS", "TRANSPORT", "CATEGORY", "AIRPLANES"} & set(
        keywords
    )
    # One Chinese spelling per distinct citation (the clause and its base
    # section); "第25.397条C" / "第25.397条第C款" were near-duplicates.
    chinese_citations = [item for item in keywords if "第25.397条" in item]
    assert chinese_citations == ["第25.397条(C)", "第25.397条"]


def test_caption_terms_exclude_boilerplate_and_function_words() -> None:
    # Real page 21 of Docket 75-10: "power corrections for vapor pressure must
    # be made in accordance with the following table:" -- "following" is a
    # caption cue, not a subject term.
    keywords = extract_keywords(
        "Sec. 25.101 General",
        ["Sec. 25.101 General"],
        [],
        text=(
            "Sec. 25.101 General\n"
            "power corrections for vapor pressure must be made in accordance with "
            "the following table:\nAltitude H (ft.)=0; Vapor pressure e=.403"
        ),
    )

    assert {"power", "corrections", "vapor", "pressure"} <= set(keywords)
    assert "following" not in keywords
    assert "made" not in keywords


def test_caption_cross_reference_is_not_published_as_a_keyword() -> None:
    # "2-54. By revising the lead-in of Sec. 25.603 to read as follows:" opens a
    # neighbouring clause; publishing 25.603 here would bind this section to it.
    text = (
        f"{PART_25}\n{SEC_25_397}\n"
        "2-54. By revising the lead-in of Sec. 25.603 to read as follows:"
    )

    keywords = extract_keywords(SEC_25_397, [PART_25, SEC_25_397], [], text=text)

    assert "25.603" not in keywords
    assert "revising" not in keywords


def test_published_content_carries_caption_and_anchors() -> None:
    table = next(
        chunk for chunk in _chunks([_caption_block(), _table_block()])
        if chunk.content_type == "table"
    )

    plan = RagflowPublisher.build_plan(
        dataset_id="kb-1",
        source_name="23-26-DRS_75-10.pdf",
        source_sha256="abc",
        document_ir={
            "chunks": [
                {
                    "chunk_id": table.chunk_id,
                    "title": table.title,
                    "text": table.text,
                    "page_start": table.page_start,
                    "page_end": table.page_end,
                    "section_path": table.section_path,
                    "article_id_normalized": table.article_id_normalized,
                    "article_aliases": table.article_aliases,
                    "table_ids": table.table_ids,
                    "table_html": table.table_html,
                    "content_type": table.content_type,
                }
            ]
        },
    )
    payload = plan.chunks[0].payload

    assert CAPTION in payload["content"]
    # The anchors travel in the content the operator previews, so what the
    # workbench shows is exactly what is uploaded.
    assert "检索锚点（非规范译文）：" in payload["content"]
    assert "pilot" in payload["content"]
    assert "Limit" in payload["important_keywords"]
    assert "AIRWORTHINESS" not in payload["important_keywords"]
    assert "content_type:table" in payload["tag_kwd"]
