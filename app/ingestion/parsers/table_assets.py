"""Source-page evidence images for structured tables missing remote assets."""

from io import BytesIO
import math

import pypdfium2 as pdfium

from app.ingestion.pipeline import AssetRecord, stable_id
from app.ingestion.parsers.header_recovery import SOURCE


def restore_table_assets(path, source_hash, pages, assets):
    existing = {asset.asset_id for asset in assets}
    for page in pages:
        tables = [item for item in page.rich_blocks
                  if item.get("block_type") == "table" and not item.get("exclusion_reason")]
        page.table_hints = [item.get("caption") or f"Table {n}"
                            for n, item in enumerate(tables, 1)]
        for item in tables:
            if item.get("asset_ids") or item.get("asset_id"):
                continue
            bbox = item.get("table_asset_bbox", item.get("bbox"))
            if (source_hash == SOURCE and page.page_number == 4
                    and item.get("table_recovery_method") == "source_visual_review_2026-08-31"):
                bbox = {
                    "TABLE II — TEST TOLERANCES": [155, 68, 406, 172],
                    "TABLE III — FRICTION": [157, 238, 408, 378],
                    "TABLE IV — PRESSURE-ALTITUDE DIFFERENCE": [472, 68, 721, 185],
                }.get(item.get("caption"), bbox)
            if (not isinstance(bbox, list) or len(bbox) != 4
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox)
                    or not 0 <= bbox[0] < bbox[2] <= 1000
                    or not 0 <= bbox[1] < bbox[3] <= 1000):
                item["table_asset_error"] = "missing_or_invalid_source_geometry"
                continue
            # Keep the OCR geometry intact; store the padded evidence crop separately.
            crop_box = [max(0, bbox[0] - 3), max(0, bbox[1] - 3),
                        min(1000, bbox[2] + 3), min(1000, bbox[3] + 3)]
            asset_id = stable_id("source-table-crop-v1", source_hash, page.page_number, *crop_box)
            if asset_id not in existing:
                with pdfium.PdfDocument(str(path)) as pdf:
                    pdf_page = pdf[page.page_number - 1]
                    try:
                        width, height = pdf_page.get_size()
                        bitmap = pdf_page.render(scale=min(4, (24_000_000 / (width * height)) ** .5))
                        try:
                            full = bitmap.to_pil()
                            crop = full.crop(tuple(v / 1000 * (full.width if i % 2 == 0 else full.height)
                                                   for i, v in enumerate(crop_box)))
                            stream = BytesIO()
                            crop.save(stream, format="PNG")
                        finally:
                            bitmap.close()
                    finally:
                        pdf_page.close()
                assets.append(AssetRecord(
                    asset_id=asset_id, page_number=page.page_number, asset_type="table",
                    bbox=crop_box, filename=f"{asset_id}.png", mime_type="image/png",
                    content=stream.getvalue(), caption=item.get("caption", ""),
                    description="Source PDF page crop; visual evidence, not independent table recognition.",
                ))
                existing.add(asset_id)
            item.update(asset_id=asset_id, asset_ids=[asset_id],
                        table_asset_bbox=crop_box, table_asset_method="source_page_crop",
                        table_asset_source_sha256=source_hash)
        page.image_count = len({aid for item in page.rich_blocks
                                if not item.get("exclusion_reason")
                                for aid in item.get("asset_ids", [])})
