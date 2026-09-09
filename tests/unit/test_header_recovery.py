import copy

from app.ingestion.parsers.header_recovery import SOURCE, recover_header_pages
from app.ingestion.pipeline import PageRecord, table_rows_from_html

FRICTION = (
    "TABLE III FRICTION Tolerance Altitude (feet) (feet) 1,000 ±70 2,000\\_ 70 3,000. 70 "
    "5,000 70 10,000. 80 15,000. 90 20,000\\_ 100 25,000. 120 30,000\\_ 140 "
    "35,000. 160 40,000. 180 50,000\\_ 250"
)


def page():
    texts = [FRICTION, "§ 91.170 Altimeter system tests and inspections.",
             "(a) No person may operate an airplane in controlled airspace under IFR unless, "
             "within the preceding 24 calendar months, each static pressure system and each "
             "altimeter instrument has been tested and inspected and found to comply with "
             "Appendix E of Part 43. The altimeter must be tested by an appropriately rated "
             "repair station.",
             "(b) Compliance with this section is not required until August 1, 1966."]
    items = [{"block_type": "annotation", "source_type": "header", "text": text,
              "bbox": [150, 80 + i * 90, 420, 150 + i * 90],
              "exclusion_reason": "mineru_excluded", "raw_ocr_item": {"text": text}}
             for i, text in enumerate(texts)]
    items.append({"block_type": "annotation", "source_type": "page_number", "text": "4",
                  "bbox": [430, 422, 455, 434], "exclusion_reason": "mineru_excluded"})
    return PageRecord(4, "original OCR", indexable=False, rich_blocks=items)


def test_empty_page_restored_with_original_labels_and_page_number_excluded():
    p = page()
    raw = copy.deepcopy([i["raw_ocr_item"] for i in p.rich_blocks[:-1]])
    recover_header_pages([p], "other-source")
    assert p.indexable and p.cleaned_char_count > 0
    assert all(i["block_type"] == "paragraph" for i in p.rich_blocks[:-1])
    assert [i["raw_ocr_item"] for i in p.rich_blocks[:-1]] == raw
    assert p.rich_blocks[-1]["exclusion_reason"] == "mineru_excluded"
    assert p.raw_text == "original OCR"


def test_verified_friction_table_only_for_exact_source_and_ocr():
    p = page()
    recover_header_pages([p], SOURCE)
    table = p.rich_blocks[0]
    assert table["block_type"] == "table"
    rows = table_rows_from_html(table["table_html"])
    assert len(rows) == 13 and rows[1] == ["1,000", "±70"] and rows[-1] == ["50,000", "250"]
    assert table["raw_table_text"] == FRICTION
    changed = page()
    changed.rich_blocks[0]["text"] = FRICTION.replace("250", "251")
    recover_header_pages([changed], SOURCE)
    assert changed.rich_blocks[0]["block_type"] == "paragraph"


def test_normal_page_with_body_does_not_restore_running_headers():
    p = page()
    p.rich_blocks.append({"block_type": "paragraph", "text": "Existing body."})
    before = copy.deepcopy(p.rich_blocks)
    recover_header_pages([p], SOURCE)
    assert p.rich_blocks == before


def test_missing_section_anchor_and_invalid_geometry_do_not_trigger():
    for damage in ("anchor", "geometry", "spread"):
        p = page()
        if damage == "anchor":
            p.rich_blocks[1]["text"] = "Ordinary running header"
        elif damage == "geometry":
            p.rich_blocks[1]["bbox"][0] = "invalid"
        else:
            for i in p.rich_blocks:
                i["bbox"] = [100, 50, 400, 60]
        before = copy.deepcopy(p.rich_blocks)
        recover_header_pages([p], SOURCE)
        assert p.rich_blocks == before
