from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

EVALUATION_DIR = Path(__file__).parents[1] / "evaluation"
ANSWERABILITY = {
    "ANSWERABLE",
    "PARTIALLY_ANSWERABLE",
    "UNANSWERABLE",
    "CONFLICTED",
}


def _load_jsonl(name: str) -> list[dict[str, Any]]:
    path = EVALUATION_DIR / name
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            pytest.fail(f"{path}:{line_number} is invalid JSON: {exc}")
        assert isinstance(value, dict), f"{path}:{line_number} must contain an object"
        rows.append(value)
    assert rows, f"{path} must not be empty"
    ids = [row.get("id") for row in rows]
    assert all(isinstance(case_id, str) and case_id for case_id in ids)
    assert len(ids) == len(set(ids)), f"{path} contains duplicate ids"
    return rows


def test_planning_cases_contract() -> None:
    rows = _load_jsonl("planning_cases.jsonl")

    for row in rows:
        assert isinstance(row.get("group"), str)
        assert isinstance(row.get("query"), str) and row["query"].strip()
        expected = row.get("expected")
        translation = row.get("translation")
        assert isinstance(expected, dict)
        assert expected.get("outcome") in {"plan", "fallback_required", "bounded_plan"}
        assert isinstance(translation, dict)
        assert isinstance(translation.get("allowed"), bool)
        assert translation.get("trigger") in {
            "never",
            "low_confidence_only",
            "no_document_or_low_confidence",
        }
        assert translation.get("max_variants_per_cell") in {0, 1}
        assert all(
            isinstance(value, str) and value.strip()
            for value in translation.get("must_preserve", [])
        )

        cell_count = expected.get("cell_count")
        if cell_count is not None:
            assert isinstance(cell_count, int) and 1 <= cell_count <= 8
        cell_count_max = expected.get("cell_count_max")
        if cell_count_max is not None:
            assert isinstance(cell_count_max, int) and 1 <= cell_count_max <= 8
        if row["group"] == "exact":
            assert translation["allowed"] is False
            assert translation["max_variants_per_cell"] == 0


def test_retrieval_cases_contract() -> None:
    rows = _load_jsonl("retrieval_cases.jsonl")

    for row in rows:
        cell = row.get("cell")
        expected = row.get("expected")
        initial = row.get("initial_candidates")
        translated = row.get("translated_candidates", [])
        assert isinstance(cell, dict)
        assert all(isinstance(cell.get(key), str) and cell[key] for key in ("subject", "aspect"))
        assert isinstance(cell.get("original_query"), str) and cell["original_query"]
        assert isinstance(initial, list) and initial
        assert isinstance(translated, list)
        assert isinstance(expected, dict)
        assert expected.get("status") in ANSWERABILITY | {"MISSING"}

        candidates = initial + translated
        chunk_ids = [candidate.get("chunk_id") for candidate in candidates]
        assert all(isinstance(chunk_id, str) and chunk_id for chunk_id in chunk_ids)
        assert len(chunk_ids) == len(set(chunk_ids))
        assert all(
            isinstance(candidate.get("document_name"), str)
            and isinstance(candidate.get("text"), str)
            and isinstance(candidate.get("score"), (int, float))
            for candidate in candidates
        )
        selected = expected.get("selected_chunk_ids", [])
        forbidden = expected.get("forbidden_chunk_ids", [])
        assert set(selected).issubset(chunk_ids)
        assert set(forbidden).issubset(chunk_ids)
        assert set(selected).isdisjoint(forbidden)
        assert expected.get("max_translation_calls") in {0, 1}
        assert expected.get("translation_required") is bool(translated)


def test_live_rag_cases_contract() -> None:
    rows = _load_jsonl("live_rag_cases.jsonl")
    assert any(row.get("must_pass") is True for row in rows)

    for row in rows:
        assert isinstance(row.get("must_pass"), bool)
        assert isinstance(row.get("knowledge_base_name"), str) and row["knowledge_base_name"]
        assert isinstance(row.get("query"), str) and row["query"].strip()
        expected = row.get("expected")
        budgets = row.get("budgets")
        assert isinstance(expected, dict)
        assert set(expected.get("status_in", [])).issubset(ANSWERABILITY)
        assert expected.get("status_in")
        assert isinstance(expected.get("must_cite"), bool)
        assert isinstance(expected.get("must_report_missing"), bool)
        assert isinstance(expected.get("required_cell_count"), int)
        for key in (
            "required_answer_terms",
            "allowed_document_keywords",
            "forbidden_document_keywords",
        ):
            assert isinstance(expected.get(key), list)
        assert isinstance(budgets, dict)
        for key in (
            "max_planner_calls",
            "max_translation_calls",
            "max_summary_calls",
            "max_total_model_calls",
        ):
            assert isinstance(budgets.get(key), int) and budgets[key] >= 0
        assert isinstance(budgets.get("latency_seconds"), (int, float))
        assert budgets["latency_seconds"] > 0
        component_calls = (
            budgets["max_planner_calls"]
            + budgets["max_translation_calls"]
            + budgets["max_summary_calls"]
        )
        assert budgets["max_total_model_calls"] >= component_calls
