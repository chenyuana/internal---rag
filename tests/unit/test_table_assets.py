from copy import deepcopy

from pypdf import PdfWriter

from app.ingestion.parsers.table_assets import restore_table_assets
from app.ingestion.pipeline import PageRecord


def test_table_asset_links_geometry_and_idempotence(tmp_path):
    path = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=600, height=800)
    writer.write(path)
    item = {"block_type": "table", "bbox": [100, 100, 400, 400],
            "caption": "TABLE I", "table_html": "<table></table>"}
    page = PageRecord(1, "", rich_blocks=[item])
    assets = []
    restore_table_assets(path, "hash", [page], assets)
    assert len(assets) == 1 and assets[0].content.startswith(b"\x89PNG")
    assert item["bbox"] == [100, 100, 400, 400]
    assert item["asset_ids"] == [assets[0].asset_id]
    assert assets[0].bbox == [97, 97, 403, 403]
    assert page.table_hints == ["TABLE I"] and page.image_count == 1
    before = deepcopy(item)
    restore_table_assets(path, "hash", [page], assets)
    assert len(assets) == 1 and item == before


def test_invalid_geometry_and_excluded_blocks_do_not_create_assets():
    item = {"block_type": "table", "bbox": [0, 0, float("nan"), 20]}
    excluded = {"block_type": "table", "exclusion_reason": "excluded"}
    page = PageRecord(1, "", rich_blocks=[item, excluded])
    assets = []
    restore_table_assets(None, "hash", [page], assets)
    assert not assets and len(page.table_hints) == 1
    assert item["table_asset_error"] == "missing_or_invalid_source_geometry"
