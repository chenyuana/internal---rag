"""Anchors taken from a table's own structure.

A table caption can be thin ("SUPPLEMENTARY INFORMATION:", or a section title
that names the enclosing clause) while the table itself is highly specific. Its
field names say what the rows mean, its first column identifies the rows, and
identifier-shaped cells carry the codes a question names. Those are exactly the
strings a regulation question contains, so they belong in the retrieval anchors.
"""

from app.ingestion.publishers import RagflowPublisher, chunk_publish_facets
from app.ingestion.regulations import extract_keywords, table_anchor_terms

FORCES_ROWS = [
    ["Control", "Maximum forces or torques", "Minimum forces or torques"],
    ["Aileron: Stick Wheel*", "100 lbs 80 D in.-lbs.**", "40 lbs. 40 D in.-lbs."],
    [
        "Elevator: Stick Wheel (symmetrical). Wheel (unsymmetrical).",
        "250 lbs. 300 lbs.",
        "100 lbs. 100 lbs. 100 lbs.",
    ],
    ["Rudder", "300 lbs.", "130 lbs."],
]
NOTICE_ROWS = [
    ["Airworthiness Review Program Notice No.", "Notice No.", "Federal Register Citation"],
    ["2", "75-10", "(40 FR 10802; Mar. 7, 1975)."],
    ["3", "75-19", "(40 FR 21866; May 19, 1975)."],
    ["4", "75-20", "(40 FR 22110; May 20, 1975)."],
]
ATMOSPHERE_ROWS = [
    ["Altitude H (ft.)", "Vapor pressure e (In. Hg.)", "Density ratio", "Remarks"],
    ["0", "0.403", "0.99508", "Do."],
    ["1,000", "0.354", "0.96672", "Do."],
]


def test_field_names_and_row_identifiers_become_anchors() -> None:
    anchors = table_anchor_terms(FORCES_ROWS)

    assert anchors == [
        "Control",
        "Maximum forces or torques",
        "Minimum forces or torques",
        "Aileron: Stick Wheel",
        "Rudder",
    ]


def test_identifier_shaped_values_are_anchored() -> None:
    anchors = table_anchor_terms(NOTICE_ROWS)

    assert "Notice No." in anchors
    assert "Federal Register Citation" in anchors
    assert {"75-10", "75-19", "75-20"} <= set(anchors)
    # The row ordinal ("2") and a full citation cell are not anchors.
    assert "2" not in anchors


def test_measurements_and_generic_labels_are_not_anchors() -> None:
    anchors = table_anchor_terms(ATMOSPHERE_ROWS)

    assert {"Altitude H (ft.)", "Vapor pressure e (In. Hg.)", "Density ratio"} <= set(anchors)
    assert not {"0.403", "0.354", "0.99508", "0.96672"} & set(anchors)
    assert "Remarks" not in anchors


def test_column_labels_are_normalized_not_dumped_verbatim() -> None:
    rows = [
        [
            "Minimum forces or torques2",
            "Maximum forces or torques for design weight, weight equal to or "
            "less than 5,000 pounds1",
            "Density ratio p ? =0.0023769=0.99508",
            "Speciﬁc humidity w (lb. moisture per lb. dry air)",
        ],
        ["Aileron: Stick Wheel3", "67 lbs.", "0.99508", "0.00849"],
    ]

    anchors = table_anchor_terms(rows)

    assert anchors[:4] == [
        "Minimum forces or torques",  # footnote marker dropped
        "Maximum forces or torques for design",  # long qualifier trimmed
        "Density ratio p ?",  # OCR merged the label with its value
        "Speciﬁc humidity w",  # parenthetical qualifier dropped
    ]
    assert "Aileron: Stick Wheel" in anchors


def test_continuation_and_generic_rows_are_not_column_labels() -> None:
    # A continued table repeats a banner row ("DISTRIBUTION TABLE—Continued",
    # which OCR splits into "...—COn" + "tinued") where a header row would be.
    continued_rows = [
        ["DISTRIBUTION TABLE—COn", "tinued", ""],
        ["23.1", "0.5", "Do."],
    ]
    generic_rows = [["Section", "Remarks"], ["23.1", "Do."]]

    assert table_anchor_terms(continued_rows) == ["23.1"]
    assert table_anchor_terms(generic_rows) == ["23.1"]

    # There is deliberately no vocabulary of "banner words": across the stored
    # corpus the table path never produced one (46 documents, 420 anchors, and
    # the only all-caps labels are genuine headers such as "LIMIT FLIGHT LOAD
    # FACTORS"). A word list there only misfired -- it dropped real headers
    # like "Normal and utility categories".
    assert table_anchor_terms([["Normal and utility categories"], ["x"]]) == [
        "Normal and utility categories"
    ]


def test_keywords_merge_table_anchors_under_the_budget() -> None:
    keywords = extract_keywords(
        "SUPPLEMENTARY INFORMATION:",
        ["SUPPLEMENTARY INFORMATION:"],
        [],
        text=(
            "SUPPLEMENTARY INFORMATION:\n"
            "Airworthiness Review Program Notice No.=2; Notice No.=75-10; "
            "Federal Register Citation=(40 FR 10802; Mar. 7, 1975)."
        ),
        table_rows=NOTICE_ROWS,
    )

    assert {"Notice No.", "75-10", "75-19"} <= set(keywords)
    assert len(keywords) <= 20


def test_chunks_without_a_table_get_no_table_anchors() -> None:
    without = extract_keywords("Sec. 25.397", ["Sec. 25.397"], [], text="Control system loads.")
    with_empty = extract_keywords(
        "Sec. 25.397", ["Sec. 25.397"], [], text="Control system loads.", table_rows=[]
    )

    assert without == with_empty


def test_published_facets_carry_table_anchors() -> None:
    chunk = {
        "chunk_id": "table-1",
        "title": "Sec. 25.397 Control system loads.",
        "text": (
            "Control=Aileron: Stick Wheel; Maximum forces or torques=100 lbs; "
            "Minimum forces or torques=40 lbs"
        ),
        "page_start": 24,
        "page_end": 24,
        "section_path": ["Sec. 25.397 Control system loads."],
        "article_id_normalized": "25.397(C)",
        "article_aliases": ["25.397(c)"],
        "table_ids": ["t1"],
        "table_rows": FORCES_ROWS,
    }

    facets = chunk_publish_facets("23-26-DRS_75-10.pdf", chunk)

    assert "Maximum forces or torques" in facets.important_keywords
    assert "Aileron: Stick Wheel" in facets.important_keywords
    assert "Maximum forces or torques" in facets.content  # the 检索锚点 line

    plan = RagflowPublisher.build_plan(
        dataset_id="kb-1",
        source_name="23-26-DRS_75-10.pdf",
        source_sha256="abc",
        document_ir={"chunks": [chunk]},
    )
    assert "Maximum forces or torques" in plan.chunks[0].payload["important_keywords"]
