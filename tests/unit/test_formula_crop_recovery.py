from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pypdf import PdfWriter

from app.ingestion.parsers.hybrid_pdf import (
    HybridPdfParser,
    _formula_context_score,
    _formula_context_symbols,
)
from app.ingestion.parsers.remote import RemoteContentBlock, RemoteParseResult


def test_formula_context_score_prefers_symbols_declared_after_formula() -> None:
    items = [
        {"block_type": "formula", "text": "bad"},
        {"block_type": "paragraph", "text": "where--"},
        {"block_type": "paragraph", "text": "C_TO = takeoff factor;"},
        {"block_type": "paragraph", "text": "V_S1 = stalling speed;"},
        {"block_type": "paragraph", "text": "W = takeoff weight."},
    ]
    symbols = _formula_context_symbols(items, 0)
    assert {"cto", "vs1"}.issubset(symbols)

    broken = r"\gamma_{2}=\frac{C\gamma_{0}}{\tilde{\nu}\tilde{s}_{1}}"
    corrected = r"n=\frac{C_{TO}V_{S1}^{2}}{(\tan^{2/3}\beta)W^{1/3}}"
    assert _formula_context_score(corrected, symbols)[0] > _formula_context_score(
        broken, symbols
    )[0]


def test_formula_crop_recovery_adopts_only_a_contextually_stronger_result(tmp_path: Path) -> None:
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as stream:
        writer.write(stream)

    original = r"\gamma_{2}=\frac{C\gamma_{0}}{\tilde{\nu}\tilde{s}_{1}}"
    candidate = r"n=\frac{C_{TO}V_{S1}^{2}}{(\tan^{2/3}\beta)W^{1/3}}"
    item = {
        "block_type": "formula",
        "text": "garbled formula",
        "latex": original,
        "bbox": [100, 100, 500, 250],
        "asset_ids": [],
    }
    page = SimpleNamespace(
        page_number=1,
        rich_blocks=[
            item,
            {"block_type": "paragraph", "text": "where--"},
            {"block_type": "paragraph", "text": "C_TO = takeoff factor;"},
            {"block_type": "paragraph", "text": "V_S1 = stalling speed;"},
            {"block_type": "paragraph", "text": "W = takeoff weight."},
        ],
    )
    document = SimpleNamespace(source_hash="source-hash", pages=[page], assets=[])
    mineru = SimpleNamespace(
        enabled=True,
        parse_pages=lambda *_args, **_kwargs: RemoteParseResult(
            parser_name="fake",
            page_blocks={
                1: [
                    RemoteContentBlock(
                        page_number=1,
                        block_type="formula",
                        text="n = corrected formula",
                        latex=candidate,
                    )
                ]
            },
        ),
    )
    parser = HybridPdfParser.__new__(HybridPdfParser)
    parser.mineru = mineru

    recovered, crops, warnings = parser._recover_suspect_formula_crops(
        document, source, {1}
    )

    assert (recovered, crops, warnings) == (1, 1, [])
    assert item["latex"] == candidate
    assert item["text"] == "n = corrected formula"
    assert item["asset_id"] in item["asset_ids"]
    assert len(document.assets) == 1
    assert document.assets[0].mime_type == "image/png"


def test_formula_crop_recovery_keeps_candidate_without_more_context_evidence(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    with source.open("wb") as stream:
        writer.write(stream)

    original = r"n=\frac{C_{TO}V_{S1}^{2}}{W}"
    item = {
        "block_type": "formula",
        "text": "original",
        "latex": original,
        "bbox": [100, 100, 500, 250],
        "asset_ids": [],
    }
    page = SimpleNamespace(
        page_number=1,
        rich_blocks=[
            item,
            {"block_type": "paragraph", "text": "C_TO = takeoff factor;"},
            {"block_type": "paragraph", "text": "V_S1 = stalling speed;"},
        ],
    )
    document = SimpleNamespace(source_hash="source-hash", pages=[page], assets=[])
    mineru = SimpleNamespace(
        enabled=True,
        parse_pages=lambda *_args, **_kwargs: RemoteParseResult(
            parser_name="fake",
            page_blocks={
                1: [
                    RemoteContentBlock(
                        page_number=1,
                        block_type="formula",
                        text="candidate",
                        latex=r"n=\frac{C_{TO}V_{S1}^{2}}{W^{1/3}}",
                    )
                ]
            },
        ),
    )
    parser = HybridPdfParser.__new__(HybridPdfParser)
    parser.mineru = mineru

    recovered, crops, _warnings = parser._recover_suspect_formula_crops(
        document, source, {1}
    )

    # Both results cover the same declared variables, so no costly crop OCR is
    # needed and the original equation remains authoritative.
    assert (recovered, crops) == (0, 0)
    assert item["latex"] == original
    assert document.assets == []
