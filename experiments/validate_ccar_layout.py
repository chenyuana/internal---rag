from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import IngestionSettings, RemoteParserSettings
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.pipeline import is_flattened_table_noise, write_document

TABLE_REGRESSION_PAGES = (258, 259, 260)
FIGURE_REGRESSION_PAGES = (187, 236)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate CCAR rich table and figure preprocessing regression pages."
    )
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    settings = IngestionSettings(
        root_dir=args.output,
        mineru=RemoteParserSettings(enabled=False),
        docling=RemoteParserSettings(enabled=False),
    )
    document = ParserRegistry.with_builtins(settings).parse(args.pdf)
    manifest = write_document(args.output, document, display_name=args.pdf.name)

    blocks_by_page = {
        page_number: [
            block
            for block in document.blocks
            if block.page_number == page_number
        ]
        for page_number in (*TABLE_REGRESSION_PAGES, *FIGURE_REGRESSION_PAGES)
    }
    assets_by_page = {
        page_number: [
            asset
            for asset in document.assets
            if asset.page_number == page_number
        ]
        for page_number in (*TABLE_REGRESSION_PAGES, *FIGURE_REGRESSION_PAGES)
    }
    summary = {
        "output_folder": manifest["output_folder"],
        "pages": {
            page_number: {
                "route": document.pages[page_number - 1].route,
                "table_blocks": sum(
                    block.block_type == "table"
                    for block in blocks_by_page[page_number]
                ),
                "asset_types": [
                    asset.asset_type for asset in assets_by_page[page_number]
                ],
                "flattened_numeric_blocks": sum(
                    is_flattened_table_noise(block.text)
                    for block in blocks_by_page[page_number]
                    if block.block_type != "table"
                ),
            }
            for page_number in (*TABLE_REGRESSION_PAGES, *FIGURE_REGRESSION_PAGES)
        },
        "chunk_count": len(document.chunks),
        "max_chunk_chars": max((len(chunk.text) for chunk in document.chunks), default=0),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    table_pages_pass = all(
        summary["pages"][page_number]["table_blocks"] > 0
        and summary["pages"][page_number]["flattened_numeric_blocks"] == 0
        for page_number in TABLE_REGRESSION_PAGES
    )
    expected_figure_counts = {187: 3, 236: 2}
    figure_pages_pass = all(
        "figure_page" in summary["pages"][page_number]["asset_types"]
        and summary["pages"][page_number]["asset_types"].count("figure")
        == expected_figure_counts[page_number]
        for page_number in FIGURE_REGRESSION_PAGES
    )
    return 0 if table_pages_pass and figure_pages_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
