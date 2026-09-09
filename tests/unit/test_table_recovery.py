from __future__ import annotations

import copy
import json
import zipfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from app.core.config import RemoteParserSettings
from app.ingestion.parsers.remote import MinerUClient
from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
from app.ingestion.parsers.table_recovery import recover_two_columns, structure_issue
from app.ingestion.pipeline import PageRecord, split_merged_or_rows, table_rows_from_html

BROKEN = (
    '<table><tr><td colspan="2">Item</td><td rowspan="2">Tolerance +5%，-10%.</td></tr>'
    "<tr><td>Weight</td><td></td></tr><tr><td></td><td>Critical items affected</td>"
    "<td></td></tr><tr><td>by weight__ C.G.</td><td></td>"
    "<td>+5%， -1%. ±7% total travel.</td></tr></table>"
)
# Independent local OCR observations, not a production replacement table.
LINES = [
    {"text": text, "bbox": bbox, "score": score}
    for text, bbox, score in [
        ("Item", [422, 103, 580, 166], 1.0),
        ("Tolerance", [1018, 99, 1321, 165], 1.0),
        ("Weight..", [109, 165, 395, 227], 0.901),
        ("+5%，-10%.", [916, 166, 1336, 228], 0.948),
        ("Critical items affected", [115, 223, 871, 290], 0.998),
        ("by weight__.", [168, 285, 557, 357], 0.979),
        ("+5%，-1%.", [910, 283, 1310, 354], 0.93),
        ("C.G..", [114, 354, 298, 412], 0.961),
        ("±7% total travel.", [910, 352, 1462, 414], 0.993),
    ]
]


def test_page4_exact_cells_and_preserved_signs():
    assert structure_issue(BROKEN) == "fragmented_spanning_header"
    repaired = recover_two_columns(BROKEN, LINES)
    assert repaired
    assert table_rows_from_html(repaired) == [
        ["Item", "Tolerance"],
        ["Weight", "+5%,-10%."],
        ["Critical items affected by weight", "+5%,-1%."],
        ["C.G.", "±7% total travel."],
    ]
    assert structure_issue(repaired) is None


def test_recovery_not_tied_to_sample_labels_or_values():
    replacements = {"Weight": "Mass", "7%": "8.5%", "10%": "12%"}
    markup, lines = BROKEN, copy.deepcopy(LINES)
    for before, after in replacements.items():
        markup = markup.replace(before, after)
        for line in lines:
            line["text"] = line["text"].replace(before, after)
    repaired = recover_two_columns(markup, lines)
    assert repaired and "Mass" in repaired and "±8.5%" in repaired and "-12%" in repaired


@pytest.mark.parametrize("key", ["score", "bbox"])
def test_nonfinite_geometry_or_confidence_rejected(key):
    lines = copy.deepcopy(LINES)
    if key == "score":
        lines[0][key] = float("nan")
    else:
        lines[0][key][0] = float("nan")
    assert recover_two_columns(BROKEN, lines) is None


@pytest.mark.parametrize("damage", ["missing", "number", "sign", "confidence", "crossing", "bbox"])
def test_incomplete_or_conflicting_ocr_is_not_accepted(damage):
    lines = copy.deepcopy(LINES)
    if damage == "missing":
        lines.pop(7)
    elif damage == "number":
        lines[-1]["text"] = "±8% total travel."
    elif damage == "sign":
        lines[-1]["text"] = "+7% total travel."
    elif damage == "confidence":
        lines[-1]["score"] = 0.5
    elif damage == "crossing":
        lines[4]["bbox"][2] = 1100
    else:
        lines[0]["bbox"] = [1, 2]
    assert recover_two_columns(BROKEN, lines) is None


@pytest.mark.parametrize(
    "markup",
    [
        '<table><tr><th colspan="2">Loads</th></tr><tr><td>Left</td><td>Right</td></tr>'
        "<tr><td>5</td><td>10</td></tr></table>",
        '<table><tr><th>Item</th><th>Value</th></tr><tr><td rowspan="2">A</td><td>1</td>'
        "</tr><tr><td>2</td></tr></table>",
        "<table><tr><th>Item</th><th>Value</th></tr><tr><td>A</td><td></td></tr></table>",
    ],
)
def test_legitimate_spans_and_blank_cells_not_flagged(markup):
    assert structure_issue(markup) is None


@pytest.mark.parametrize("success", [True, False])
def test_parser_uses_one_structure_and_retains_evidence(success):
    item = {
        "block_type": "table",
        "text": BROKEN + "\nFootnote retained.",
        "table_html": BROKEN,
        "bbox": [514, 414, 723, 480],
    }
    page = PageRecord(4, BROKEN, rich_blocks=[item], indexable=True)
    client = SimpleNamespace(ocr_table_lines=Mock(return_value=LINES if success else []))
    parser = ScannedRegulatoryPdfParser(client)
    with patch("app.ingestion.parsers.scan_regulatory.crop_table_pdf", return_value=b"pdf"):
        parser._recover_tables(Path("source.pdf"), [page])
    assert item["raw_table_html"] == BROKEN
    assert item["raw_table_text"].endswith("Footnote retained.")
    blocks = parser._build_blocks("doc", [page])
    parser._render_pages([page], blocks, [])
    chunks = parser._build_chunks("doc", blocks)
    if success:
        assert blocks[0].table_header_rows == 1
        assert blocks[0].table_html in page.cleaned_text
        assert chunks[0].table_html == [blocks[0].table_html]
        assert "Item=C.G." in chunks[0].text
        assert "Footnote retained." in chunks[0].text
    else:
        assert "+5%" not in chunks[0].text
        assert not blocks[0].table_rows and not blocks[0].table_html
        assert "requires structural review" in page.cleaned_text


def test_crop_failure_keeps_document_reviewable():
    item = {
        "block_type": "table",
        "text": BROKEN,
        "table_html": BROKEN,
        "bbox": [514, 414, 723, 480],
    }
    page = PageRecord(4, BROKEN, rich_blocks=[item], indexable=True)
    parser = ScannedRegulatoryPdfParser(SimpleNamespace())
    with patch("app.ingestion.parsers.scan_regulatory.crop_table_pdf", side_effect=RuntimeError):
        parser._recover_tables(Path("source.pdf"), [page])
    assert item["table_review_status"] == "needs_review"
    assert "RuntimeError" in item["table_recovery_error"]


def test_remote_crop_uses_text_ocr_and_returns_coordinates():
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "crop_model.json",
            json.dumps([{"layout_dets": [dict(line, label="ocr_text") for line in LINES]}]),
        )

    def handler(request):
        assert b'name="table_enable"\r\n\r\nfalse' in request.content
        assert b'name="formula_enable"\r\n\r\nfalse' in request.content
        assert b'filename="table-crop.pdf"' in request.content
        return httpx.Response(200, content=buffer.getvalue())

    client = MinerUClient(
        RemoteParserSettings(enabled=True, base_url="http://test.local"),
        transport=httpx.MockTransport(handler),
    )
    assert [line["text"] for line in client.ocr_table_lines(b"pdf")] == [
        line["text"] for line in LINES
    ]


def test_split_merged_or_rows_splits_unambiguous_or_alternatives():
    # A row that is really two "OR"-alternative sub-rows, minified by the source
    # model into one row with doubled cells.  Must split cleanly into two rows.
    rows = [
        [
            "Stabilizer Trim PositionORPitch Control Position",
            "Full RangeFull Range",
            "±3% unless higheruniquely required±3% unless higheruniquely required",
            "11",
            "1%.1%.",
        ],
        # Legit value-level "or" must NOT be treated as an alternative separator.
        ["Roll altitude", "±60° or 100% of usable", "+2°", "1", "0.8°."],
    ]
    new_rows, ambiguous = split_merged_or_rows(rows)
    assert ambiguous is False
    assert new_rows[0] == [
        "Stabilizer Trim Position",
        "Full Range",
        "±3% unless higheruniquely required",
        "1",
        "1%.",
    ]
    assert new_rows[1] == [
        "Pitch Control Position",
        "Full Range",
        "±3% unless higheruniquely required",
        "1",
        "1%.",
    ]
    # The ordinary row is untouched.
    assert new_rows[2] == ["Roll altitude", "±60° or 100% of usable", "+2°", "1", "0.8°."]


def test_split_merged_or_rows_flags_ambiguous_fusion():
    # Columns here are single values (not doubled), so the fusion cannot be split
    # unambiguously.  The row is left unchanged and reported for review.
    rows = [
        [
            "Fan or N1 Speed or EPRor Cockpit indicationsUsed for AircraftCertification.ORProp. speed and Torque",
            "MaximumRange",
            "+5%",
            "11 (prop Speed)1 (torque)",
            "1%",
        ]
    ]
    new_rows, ambiguous = split_merged_or_rows(rows)
    assert ambiguous is True
    assert new_rows == rows


def test_split_merged_or_rows_is_idempotent():
    rows = [[
        "Stabilizer Trim PositionORPitch Control Position",
        "Full RangeFull Range",
        "±3%±3%",
        "11",
        "1%1%",
    ]]
    once, ambiguous_once = split_merged_or_rows(rows)
    twice, ambiguous_twice = split_merged_or_rows(once)
    assert ambiguous_once is False
    assert ambiguous_twice is False
    assert twice == once


@pytest.mark.parametrize(
    "label",
    [
        "Trailing Edge Flap Or Cockpit Control Selection",
        "Discrete or Analog",
        "Fan or N1 Speed or EPR",
    ],
)
def test_split_merged_or_rows_ignores_ordinary_label_or(label):
    rows = [[label, "Full Range", "±3%", "1", "1%"]]
    new_rows, ambiguous = split_merged_or_rows(rows)
    assert new_rows == rows
    assert ambiguous is False


def test_split_merged_or_rows_requires_two_supporting_columns():
    rows = [[
        "Primary PositionORSecondary Position",
        "Full RangeFull Range",
        "",
        "",
        "",
    ]]
    new_rows, ambiguous = split_merged_or_rows(rows)
    assert new_rows == rows
    assert ambiguous is True


def test_split_merged_or_rows_does_not_accept_three_repetitions():
    rows = [[
        "Primary PositionORSecondary Position",
        "ABCABCABC",
        "±3%±3%",
        "111",
        "1%1%",
    ]]
    new_rows, ambiguous = split_merged_or_rows(rows)
    assert new_rows == rows
    assert ambiguous is True
