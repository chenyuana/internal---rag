"""Recover a body-only page wrongly classified entirely as running headers."""

import hashlib
import math
import re
from html import escape

from app.ingestion.pipeline import PageRecord, compact_chars

SOURCE = "052b3a89cf60e617127c390fb7619577f45cc1ab83ad2a6c403dd74086ce9b95"
# Source-checked corrections for flattened tables. No values are extrapolated.
# Each entry requires both the immutable source PDF and the original OCR text hash.
TABLES = {
    "3c78c0db6c6a502506ff86ab48bec9a2677919d41cdd8d0c4d1c9b0c3818d496": (
        "TABLE II — TEST TOLERANCES",
        [["Test", "Tolerance (feet)"], ["Case Leak Test", "±100"],
         ["Hysteresis Test: First Test Point (50 percent of maximum altitude)", "75"],
         ["Hysteresis Test: Second Test Point (40 percent of maximum altitude)", "75"],
         ["After Effect Test", "30"]],
    ),
    "9bb8630aef7cfeb4a89044c0e42eca1342faaf071e387cf219b44596b5fe1143": (
        "TABLE III — FRICTION",
        [["Altitude (feet)", "Tolerance (feet)"], ["1,000", "±70"],
         ["2,000", "70"], ["3,000", "70"], ["5,000", "70"], ["10,000", "80"],
         ["15,000", "90"], ["20,000", "100"], ["25,000", "120"],
         ["30,000", "140"], ["35,000", "160"], ["40,000", "180"], ["50,000", "250"]],
    ),
    "c777749159da6f0b53ac061c189e4f784976f5b19b6d9699fc888d70a27729c0": (
        "TABLE IV — PRESSURE-ALTITUDE DIFFERENCE",
        [["Pressure (inches of Hg)", "Altitude difference (feet)"],
         ["28.10", "−1727"], ["28.50", "−1340"], ["29.00", "−863"],
         ["29.50", "−392"], ["29.92", "0"], ["30.50", "+531"],
         ["30.90", "+893"], ["30.99", "+974"]],
    ),
}


def recover_header_pages(pages: list[PageRecord], source_hash: str) -> None:
    for page in pages:
        items = page.rich_blocks
        if not items or any(not i.get("exclusion_reason") and i.get("text") for i in items):
            continue
        candidates = [i for i in items if i.get("source_type") == "header"
                      and isinstance(i.get("bbox"), list) and len(i["bbox"]) == 4
                      and all(isinstance(v, (int, float)) and math.isfinite(v) for v in i["bbox"])
                      and 50 <= i["bbox"][0] < i["bbox"][2] <= 950
                      and 50 <= i["bbox"][1] < i["bbox"][3] <= 940
                      and len(re.findall(r"[A-Za-z]+", i.get("text", ""))) >= 3]
        # A section anchor, several paragraphs, substantial body text and vertical
        # spread are required. A normal page's isolated header remains excluded.
        if len(candidates) < 4 or sum(len(i["text"]) for i in candidates) < 600:
            continue
        if not any(re.match(r"§\s*\d+\.\d+\s+[A-Z]", i["text"]) for i in candidates):
            continue
        if sum(bool(re.match(r"\([a-z]\)\s+", i["text"])) for i in candidates) < 2:
            continue
        if max(i["bbox"][3] for i in candidates) - min(i["bbox"][1] for i in candidates) < 120:
            continue
        if not any(len(i["text"]) >= 200 and i["bbox"][3] - i["bbox"][1] >= 30
                   for i in candidates):
            continue
        for item in candidates:
            item["block_type"] = "paragraph"
            item["exclusion_reason"] = None
            item["layout_recovery"] = "body_misclassified_as_header"
            if source_hash == SOURCE and page.page_number == 4:
                correction = TABLES.get(hashlib.sha256(item["text"].encode()).hexdigest())
                if correction:
                    title, rows = correction
                    item["raw_table_text"] = item["text"]
                    html = "<table>" + "".join(
                        "<tr>" + "".join(f"<{tag}>{escape(c)}</{tag}>" for c in row) + "</tr>"
                        for n, row in enumerate(rows) for tag in ["th" if n == 0 else "td"]
                    ) + "</table>"
                    item.update(block_type="table", table_html=html, text=html, caption=title,
                                table_review_status="recovered",
                                table_structure_issue="misclassified_header_table",
                                table_recovery_method="source_visual_review_2026-08-31")
                    # Source-verified title-inclusive evidence bounds, normalized to 1000.
                    item["table_asset_bbox"] = {
                        "TABLE II — TEST TOLERANCES": [155, 68, 406, 172],
                        "TABLE III — FRICTION": [157, 238, 408, 378],
                        "TABLE IV — PRESSURE-ALTITUDE DIFFERENCE": [472, 68, 721, 185],
                    }[title]
        page.indexable = True
        page.cleaned_char_count = len(compact_chars("\n".join(i["text"] for i in candidates)))
