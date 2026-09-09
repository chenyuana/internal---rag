"""Repeatable local PDF acceptance: live MinerU or provenance-checked OCR replay.

Run from repository root with ``python -m tests.acceptance_scan_regulatory``.
Never modifies the source PDF or publishes to RAGFlow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import httpx
import pypdfium2 as pdfium

from app.core.config import IngestionSettings, RemoteParserSettings, load_settings
from app.ingestion.parsers.registry import ParserRegistry
from app.ingestion.parsers.remote import MinerUClient
from app.ingestion.pipeline import write_document


def replay_transport(folder: Path, source: Path) -> httpx.MockTransport:
    origin = next(folder.glob("*_origin.pdf"))
    if hashlib.sha256(origin.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
        # MinerU reserializes its origin PDF. Verify every rendered page instead
        # of trusting a matching filename or bypassing the provenance check.
        with pdfium.PdfDocument(origin) as cached, pdfium.PdfDocument(source) as original:
            if len(cached) != len(original):
                raise ValueError("Cached origin page count differs from input.")
            for index in range(len(original)):
                left, right = original[index], cached[index]
                try:
                    a, b = left.render(scale=1), right.render(scale=1)
                    try:
                        if a.to_pil().tobytes() != b.to_pil().tobytes():
                            raise ValueError(f"Cached origin pixels differ on page {index + 1}")
                    finally:
                        a.close()
                        b.close()
                finally:
                    left.close()
                    right.close()
        print("Provenance: all cached origin pages are pixel-identical at 72 dpi.", flush=True)
    items = json.loads(next(folder.glob("*_content_list.json")).read_text(encoding="utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content

        def field(name: str) -> str:
            match = re.search(rb'name="' + name.encode() + rb'"\r\n\r\n([^\r]+)', body)
            assert match, name
            return match[1].decode()

        assert field("parse_method") == "ocr"
        start, end = int(field("start_page_id")), int(field("end_page_id"))
        selected = [
            dict(i, page_idx=i["page_idx"] - start)
            for i in items
            if start <= i.get("page_idx", -1) <= end
        ]
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("doc/doc_content_list.json", json.dumps(selected, ensure_ascii=False))
            images = {i.get("img_path") or i.get("image_path") for i in selected}
            for name in images - {None, ""}:
                image = (folder / name).resolve()
                if not image.is_relative_to(folder.resolve()):
                    raise ValueError("Image path outside OCR folder")
                archive.write(image, "doc/" + name)
        return httpx.Response(
            200, content=buffer.getvalue(), headers={"content-type": "application/zip"}
        )

    return httpx.MockTransport(handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--replay-batches", type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--live-table-recovery",
        action="store_true",
        help="Replay page OCR but run real local crop OCR for table recovery",
    )
    args = parser.parse_args()
    replay = bool(args.replay or args.replay_batches)
    if args.replay and args.replay_batches:
        parser.error("Choose one replay mode.")
    settings = (
        load_settings().ingestion
        if not replay
        else IngestionSettings(
            mineru=RemoteParserSettings(
                enabled=True, base_url="http://replay.local", page_batch_size=10
            )
        )
    )
    transport = replay_transport(args.replay, args.source) if args.replay else None
    if args.replay_batches:
        transport = api_batch_transport(args.replay_batches, args.source)
    registry = ParserRegistry.with_builtins(settings, mineru_transport=transport)
    assert registry.select(args.source).name == "scan-regulatory-pdf"
    if replay and args.live_table_recovery:
        registry.select(args.source).mineru.ocr_table_lines = MinerUClient(
            load_settings().ingestion.mineru
        ).ocr_table_lines
    document = registry.parse(
        args.source,
        progress_callback=lambda done, total: print(f"OCR pages {done}/{total}", flush=True),
    )
    output = write_document(args.output, document)
    content = "\n".join(c.text for c in document.chunks)
    known_assets = {a.asset_id for a in document.assets}
    excluded_ids = {b.block_id for b in document.blocks if b.block_type == "annotation"}
    indexed_ids = {b for c in document.chunks for b in c.block_ids}
    formula_blocks = [b for b in document.blocks if b.latex]
    page5_text = "\n".join(b.text for b in document.blocks if b.page_number == 5)
    page4_tables = [b for b in document.blocks if b.page_number == 4 and b.block_type == "table"]
    expected_page4 = [
        ["Item", "Tolerance"],
        ["Weight", "+5%,-10%"],
        ["Critical items affected by weight", "+5%,-1%"],
        ["C.G.", "±7% total travel"],
    ]
    page4_rows = (
        [[row[0], row[1].replace(", ", ",").rstrip(".")] for row in page4_tables[0].table_rows]
        if len(page4_tables) == 1 and all(len(r) == 2 for r in page4_tables[0].table_rows)
        else []
    )
    checks = {
        "page_count_46": len(document.pages) == 46,
        "page5_inline_climb_formula": (
            "with a V_s0 of more than 70" in page5_text
            and "at least 0.02 V_s0^2 (that is" in page5_text
        ),
        "page5_no_bold_command_corruption": "±b" not in page5_text,
        "page4_exact_cells": page4_rows == expected_page4,
        "page4_raw_evidence_retained": any(
            i.get("raw_table_html") and i.get("table_review_status") == "recovered"
            for i in document.pages[3].rich_blocks
        ),
        "page4_semantic_pairing": any(
            "Item=C.G." in c.text and "Tolerance=±7% total travel." in c.text
            for c in document.chunks
            if c.page_start == 4
        ),
        "page_content_coverage": document.qa["content_coverage"] == 1,
        "nonempty_chunks": bool(document.chunks),
        "handwritten_title_not_indexed": "See anrrie t ind" not in content,
        "handwritten_text_retained": any("See anrrie t ind" in p.raw_text for p in document.pages),
        "official_signature_preserved": "HALABY" in content,
        "annotations_not_indexed": not (excluded_ids & indexed_ids),
        "formulas_preserved": bool(formula_blocks)
        and all(
            b.block_id in indexed_ids and any(b.latex in c.formula_latex for c in document.chunks)
            for b in formula_blocks
        ),
        "asset_links_resolve": bool(known_assets)
        and all(a in known_assets for b in document.blocks for a in b.asset_ids),
        "tables_indexed": all(
            b.block_id in indexed_ids for b in document.blocks if b.block_type == "table"
        ),
        "hierarchical_sections": any(len(c.section_path) >= 4 for c in document.chunks),
        "page6_table_review_flagged": any(
            g["gate"] == "table_row_alignment" and "Page 6:" in g["message"]
            for g in document.qa["quality_gates"]
        ),
    }
    summary = {
        "mode": (
            "live_ocr_batch_replay"
            if args.replay_batches
            else "cached_ocr_replay"
            if args.replay
            else "live_mineru"
        ),
        "replay_inputs": [str(p) for p in (args.replay_batches or [args.replay]) if p],
        "source_sha256": document.source_hash,
        "live_table_recovery": args.live_table_recovery or not replay,
        "output_folder": output["output_folder"],
        "text_coverage": document.qa["text_coverage"],
        "content_coverage": document.qa["content_coverage"],
        "image_only_pages": document.qa["image_only_pages"],
        "checks": checks,
        "block_types": dict(Counter(b.block_type for b in document.blocks)),
        "pages": len(document.pages),
        "chunks": len(document.chunks),
        "assets": len(document.assets),
        "quality_gates": document.qa["quality_gates"],
        "structural_acceptance": all(checks.values()),
        "semantic_accuracy": "Page 4 cells checked exactly; other tables still require review.",
    }
    (args.output / "acceptance.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not all(checks.values()):
        raise SystemExit(1)


def api_batch_transport(roots: list[Path], source: Path) -> httpx.MockTransport:
    """Reprocess recorded local API batches after parser changes, without new OCR.

    Each API upload must match the source bytes. Roots must be supplied in the
    request order; page windows are checked against the middle.json page count.
    """
    source_hash = hashlib.sha256(source.read_bytes()).digest()
    responses = {}
    start = 0
    for root in roots:
        upload = root / "uploads" / source.name
        if hashlib.sha256(upload.read_bytes()).digest() != source_hash:
            raise ValueError(f"API upload hash mismatch: {root}")
        folder = root / source.stem / "ocr"
        count = len(
            json.loads(next(folder.glob("*_middle.json")).read_text(encoding="utf-8"))["pdf_info"]
        )
        items = json.loads(next(folder.glob("*_content_list.json")).read_text(encoding="utf-8"))
        buffer = BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("doc/doc_content_list.json", json.dumps(items, ensure_ascii=False))
            for name in {i.get("img_path") or i.get("image_path") for i in items} - {None, ""}:
                image = (folder / name).resolve()
                if not image.is_relative_to(folder.resolve()):
                    raise ValueError("Image path outside batch folder")
                archive.write(image, "doc/" + name)
        responses[(start, start + count - 1)] = buffer.getvalue()
        start += count

    def handler(request: httpx.Request) -> httpx.Response:
        bounds = []
        for field in (b"start_page_id", b"end_page_id"):
            match = re.search(rb'name="' + field + rb'"\r\n\r\n(\d+)', request.content)
            assert match
            bounds.append(int(match[1]))
        return httpx.Response(
            200, content=responses[tuple(bounds)], headers={"content-type": "application/zip"}
        )

    return httpx.MockTransport(handler)


if __name__ == "__main__":
    main()
