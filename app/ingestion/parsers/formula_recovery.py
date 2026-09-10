"""High-resolution, source-backed evidence crops for suspected formulas."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
from PIL import ImageOps


@dataclass(frozen=True, slots=True)
class FormulaCrop:
    """A locally rendered crop for OCR plus its reviewable PNG evidence."""

    ocr_pdf: bytes
    png: bytes
    bbox: list[float]


def _validated_crop_box(bbox: list[float]) -> list[float]:
    if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= 1000):
        raise ValueError("Invalid formula horizontal bounds")
    if not (0 <= bbox[1] < bbox[3] <= 1000):
        raise ValueError("Invalid formula vertical bounds")
    # A formula's fraction bar and superscripts often touch its detector box.
    # Keep a small source-page margin, but do not include adjacent prose.
    return [
        max(0.0, bbox[0] - 4),
        max(0.0, bbox[1] - 4),
        min(1000.0, bbox[2] + 4),
        min(1000.0, bbox[3] + 4),
    ]


def crop_formula_source(path: Path, page_number: int, bbox: list[float]) -> FormulaCrop:
    """Render a formula region at high resolution from source PDF pixels.

    This deliberately re-renders the source page instead of enlarging the
    remote service's existing thumbnail.  The latter cannot restore a lost
    subscript or fraction rule; rendering the vector/raster source at a larger
    scale can.  Both the OCR PDF and review PNG receive the same white border.
    """

    crop_bbox = _validated_crop_box([float(value) for value in bbox[:4]])
    with pdfium.PdfDocument(path) as document:
        page = document[page_number - 1]
        try:
            width, height = page.get_size()
            # Formula crops are small; use a sharper source raster than the
            # normal page route while retaining a fixed memory ceiling.
            scale = min(8, (32_000_000 / (width * height)) ** 0.5)
            bitmap = page.render(scale=scale)
            try:
                source = bitmap.to_pil()
                box = (
                    source.width * crop_bbox[0] / 1000,
                    source.height * crop_bbox[1] / 1000,
                    source.width * crop_bbox[2] / 1000,
                    source.height * crop_bbox[3] / 1000,
                )
                crop = source.crop(box).convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()

    padded = ImageOps.expand(crop, border=80, fill="white")
    png = BytesIO()
    padded.save(png, format="PNG")
    pdf = BytesIO()
    padded.save(pdf, format="PDF", resolution=300)
    return FormulaCrop(ocr_pdf=pdf.getvalue(), png=png.getvalue(), bbox=crop_bbox)
