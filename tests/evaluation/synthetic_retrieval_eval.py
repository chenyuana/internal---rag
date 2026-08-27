from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from app.schemas.retrieval import ChunkMetadata, RetrievedChunk
from app.services.cell_evidence import GENERIC_BIGRAMS, substantive_lines
from app.services.evidence_text import lexical_units
from app.services.requirement_taxonomy import ASPECT_RETRIEVAL_TERMS
from app.services.retrieval_fusion import RetrievalFusion
from app.services.subject_document_resolver import SubjectDocumentResolver

HERE = Path(__file__).resolve().parent


def _chunk(value: dict[str, Any]) -> RetrievedChunk:
    document_name = value["document_name"]
    return RetrievedChunk(
        chunk_id=value["chunk_id"],
        document_id=document_name,
        dataset_id="synthetic",
        text=value["text"],
        metadata=ChunkMetadata(
            document_name=document_name,
            section_id=value.get("section"),
        ),
        hybrid_score=float(value["score"]),
    )


def evaluate_case(case: dict[str, Any], *, use_taxonomy: bool) -> dict[str, Any]:
    cell = case["cell"]
    initial = [_chunk(value) for value in case["initial_candidates"]]
    translated = [_chunk(value) for value in case.get("translated_candidates", [])]
    all_candidates = [*initial, *translated]
    resolution = SubjectDocumentResolver().resolve(cell["subject"], all_candidates)
    scoped_initial = [item for item in initial if item.document_id in resolution.document_ids]
    scoped_translated = [
        item for item in translated if item.document_id in resolution.document_ids
    ]
    fused = RetrievalFusion().fuse(
        scoped_initial,
        scoped_translated if case["expected"]["translation_required"] else [],
    )
    keywords = lexical_units(cell["aspect"])
    if use_taxonomy:
        keywords.update(lexical_units(ASPECT_RETRIEVAL_TERMS.get(cell["aspect"], "")))
    keywords -= GENERIC_BIGRAMS
    exact_query = any(char.isdigit() for char in cell["aspect"])
    selected = [
        item.chunk_id
        for item in fused
        if substantive_lines(
            item,
            keywords,
            cell["aspect"] if use_taxonomy else None,
        )
        or (exact_query and (item.metadata.section_id or "") in cell["original_query"])
    ]
    expected = case["expected"]
    required = set(expected["selected_chunk_ids"])
    forbidden = set(expected["forbidden_chunk_ids"])
    selected_set = set(selected)
    passed = required.issubset(selected_set) and forbidden.isdisjoint(selected_set)
    return {
        "id": case["id"],
        "passed": passed,
        "selected_chunk_ids": selected,
        "missing_required": sorted(required - selected_set),
        "selected_forbidden": sorted(forbidden & selected_set),
    }


def load_cases() -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (HERE / "retrieval_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def run() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic synthetic retrieval A/B")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = load_cases()
    with_taxonomy = [evaluate_case(case, use_taxonomy=True) for case in cases]
    without_taxonomy = [evaluate_case(case, use_taxonomy=False) for case in cases]
    report = {
        "case_count": len(cases),
        "with_taxonomy_passed": sum(item["passed"] for item in with_taxonomy),
        "without_taxonomy_passed": sum(item["passed"] for item in without_taxonomy),
        "with_taxonomy": with_taxonomy,
        "without_taxonomy": without_taxonomy,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if all(item["passed"] for item in with_taxonomy) else 1


if __name__ == "__main__":
    raise SystemExit(run())
