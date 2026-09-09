"""Live single-page OCR followed by cache-only cleaning; no publication or job mutation."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.core.config import load_settings
from app.ingestion.parsers.remote import MinerUClient
from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
from app.ingestion.pipeline import write_document


class CountingMinerU(MinerUClient):
    def __init__(self, settings):
        super().__init__(settings)
        self.requests = []

    def parse_pages(self, path, page_numbers, **kwargs):
        self.requests.append(list(page_numbers))
        return super().parse_pages(path, page_numbers, **kwargs)

    def ocr_table_lines(self, pdf):
        self.requests.append(["table_crop"])
        return super().ocr_table_lines(pdf)


def main():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("previous", type=Path)
    args_parser.add_argument("--page", type=int, default=5)
    args_parser.add_argument("--output", type=Path, required=True)
    args = args_parser.parse_args()
    previous = json.loads((args.previous / "document_ir.json").read_text(encoding="utf-8"))
    source = Path(previous["source"]["path"])
    mineru = CountingMinerU(load_settings().ingestion.mineru)
    parser = ScannedRegulatoryPdfParser(mineru)
    start = time.perf_counter()
    document = parser.reprocess_page(source, previous, args.previous, args.page, "clean")
    ocr_seconds = time.perf_counter() - start
    ocr_requests = list(mineru.requests)
    output = Path(write_document(args.output / "single-ocr", document)["output_folder"])
    updated = json.loads((output / "document_ir.json").read_text(encoding="utf-8"))
    mineru.requests.clear()
    start = time.perf_counter()
    cleaned = parser.reprocess_page(source, updated, output, args.page, "clean")
    clean_seconds = time.perf_counter() - start
    clean_output = write_document(args.output / "cached-clean", cleaned)
    checks = {
        "legacy_fallback_only_target_page": ocr_requests == [[args.page]],
        "cached_clean_zero_remote_requests": not mineru.requests,
        "other_pages_unchanged": all(
            p.cleaned_text == old["cleaned_text"]
            for p, old in zip(document.pages, previous["pages"], strict=True)
            if p.page_number != args.page
        ),
        "all_pages_retained": len(cleaned.pages) == len(previous["pages"]),
        "chunks_rebuilt": bool(cleaned.chunks),
        "assets_preserved": all(
            a.asset_id in {b.asset_id for b in cleaned.assets} for a in document.assets
        ),
        "formula_fixed": any("0.02 V_s0^2" in c.text for c in cleaned.chunks),
    }
    report = {
        "checks": checks,
        "passed": all(checks.values()),
        "single_page_ocr_seconds": ocr_seconds,
        "cached_clean_seconds": clean_seconds,
        "ocr_requests": ocr_requests,
        "clean_requests": mineru.requests,
        "output_folder": clean_output["output_folder"],
    }
    (args.output / "acceptance.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
