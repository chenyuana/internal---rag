"""Real MinerU acceptance for the user-supplied 23-48.pdf; never publishes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import load_settings
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.pipeline import table_rows_from_html, write_document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    settings = load_settings().ingestion
    assert not settings.figure_vlm.enabled, "This acceptance must not enable VLM"
    doc = ParserRegistry.with_builtins(settings).parse(
        args.source,
        progress_callback=lambda done, total: print(f"OCR {done}/{total}", flush=True),
    )
    result = write_document(args.output, doc)
    print(result["output_folder"], flush=True)
    assert doc.qa["parser_version"] == "4"
    assert len(doc.pages) == 2
    assert all(page.route == "remote_ocr" for page in doc.pages)
    assert not any(gate["status"] == "fail" for gate in doc.qa["quality_gates"])
    heading = next(block for block in doc.blocks if block.text.startswith("§ 23.415"))
    assert heading.block_type == "clause" and heading.article_id_normalized == "23.415"
    metadata, table = [block for block in doc.blocks if block.block_type == "table"]
    assert metadata.table_rows == [
        ["SUBJECT GROUP:", "Control Surface and System Loads"],
        ["SECTION:", "23.415"],
        ["AMENDMENT NUMBER:", "23-48"],
        ["EFFECTIVE DATE:", "07/25/2017"],
    ]
    assert table.table_rows == [
        ["Surface", "K", "Position of controls"],
        ["(a) Aileron", "0.75", "Control column locked lashed in mid-position."],
        [
            "(b) Aileron",
            "±0.50",
            "Ailerons at full throw; + moment on one aileron, − moment on the other.",
        ],
        ["(c) Elevator", "±0.75", "(c) Elevator full up (−)."],
        ["(d) Elevator", "", "(d) Elevator full down (+)."],
        ["(e) Rudder", "±0.75", "(e) Rudder in neutral."],
        ["(f) Rudder", "", "(f) Rudder at full throw."],
    ]
    assert table_rows_from_html(table.table_html) == table.table_rows
    assert table.table_title == ""
    formula = next(block for block in doc.blocks if block.block_type == "formula")
    assert formula.latex == "H = K c S q"
    assert all(f"{variable} =" in formula.text for variable in "HcSqK")
    assert "14.6 √(W/S) + 14.6" in formula.text
    assert any(formula.text in chunk.text for chunk in doc.chunks)
    cleaned = "\n".join(page.cleaned_text for page in doc.pages)
    assert "## § 23.415 Ground gust conditions." in cleaned
    assert "记录" not in cleaned and "momenton" not in cleaned and "ft.-Ibs." not in cleaned
    assert "\n\n[Doc. No." in cleaned
    assert not any(
        "(b) Aileron" in block.text for block in doc.blocks if block.block_type != "table"
    )
    assert all(chunk.article_id_normalized == "23.415" for chunk in doc.chunks[1:])
    print(
        json.dumps(
            {
                "result": "PASS",
                "chunk_count": len(doc.chunks),
                "table_rows_including_headers": [len(metadata.table_rows), len(table.table_rows)],
                "formula_definition_groups": 1,
                "quality_gates": doc.qa["quality_gates"],
                "vlm_enabled": settings.figure_vlm.enabled,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
