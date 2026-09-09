"""Stage a local, non-publishing reconstruction and section inventory audit.

Run from the project root with python -m scripts.audit_regulatory_ir. This is
an evidence inventory check, not an answer-accuracy evaluation. Input source
blocks and human-reviewed tables are preserved; no OCR or model is invoked.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from app.ingestion.pipeline import PageRecord, build_chunks, split_blocks
from app.services.query_analyzer import QueryAnalyzer


def audit(source: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, Any]:
    pages = []
    for page in source["pages"]:
        rich = [dict(b) for b in source["blocks"] if b["page_number"] == page["page_number"]]
        for item in rich:
            if item["block_type"] in {"clause", "article", "chapter", "annex", "heading"}:
                item["block_type"] = "paragraph"
        pages.append(PageRecord(
            page_number=page["page_number"], raw_text="", rich_blocks=rich,
            indexable=page.get("indexable", True),
        ))
    blocks = split_blocks(source["document_id"], pages)
    chunks = build_chunks(source["document_id"], blocks)
    before = source["chunks"]
    inventory: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        section = chunk.article_id_normalized
        if section:
            inventory.setdefault(section, []).append({
                "chunk_id": chunk.chunk_id, "page_start": chunk.page_start,
                "page_end": chunk.page_end, "title": chunk.title,
            })
    cases = []
    for question in questions:
        query = QueryAnalyzer().analyze(question["question"])
        found = {requested: [entry for section, entries in inventory.items()
                             if section == requested or section.startswith(requested + "(")
                             for entry in entries] for requested in query.article_ids}
        cases.append({**question, "requested_sections": query.article_ids,
                      "section_evidence": found,
                      "needs_review": [key for key, value in found.items() if not value],
                      "answer_score": None})
    return {
        "kind": "staged_evidence_inventory_not_answer_evaluation",
        "document_id": source["document_id"],
        "before": {"chunks": len(before), "with_section": sum(
            bool(c.get("article_id_normalized")) for c in before),
            "titles": len({c["title"] for c in before})},
        "after": {"chunks": len(chunks), "with_section": sum(
            bool(c.article_id_normalized) for c in chunks),
            "titles": len({c.title for c in chunks})},
        "table_ids_preserved": set(b.get("table_id") for b in source["blocks"]
                                   if b.get("table_id")) == set(b.table_id for b in blocks
                                                               if b.table_id),
        "table_rows_preserved": sorted(
            json.dumps(b.get("table_rows", []), ensure_ascii=False)
            for b in source["blocks"] if b.get("table_rows")
        ) == sorted(json.dumps(b.table_rows, ensure_ascii=False) for b in blocks if b.table_rows),
        "block_types": dict(Counter(b.block_type for b in blocks)),
        "cases": cases,
        "candidate_chunks": [asdict(c) for c in chunks],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-ir", type=Path, required=True)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.resolve() == args.input_ir.resolve().parent:
        parser.error("output-dir must be separate from the source artifact directory")
    source = json.loads(args.input_ir.read_text(encoding="utf-8"))
    questions = (
        [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines()
         if line.strip()] if args.questions else []
    )
    result = audit(source, questions)
    if not result["table_ids_preserved"] or not result["table_rows_preserved"]:
        parser.error("table evidence changed during reconstruction; candidate not written")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    chunks = result.pop("candidate_chunks")
    (args.output_dir / "candidate_chunks.jsonl").write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in chunks) + "\n", encoding="utf-8",
    )
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps({"before": result["before"], "after": result["after"],
                      "table_ids_preserved": result["table_ids_preserved"],
                      "questions": len(questions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
