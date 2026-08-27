from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import httpx

HERE = Path(__file__).resolve().parent
ANSWERABLE = {"ANSWERABLE", "PARTIALLY_ANSWERABLE"}


def load_cases(include_exploratory: bool) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in (HERE / "live_rag_cases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if include_exploratory:
        return rows
    return [row for row in rows if row["must_pass"]]


def dataset_map(client: httpx.Client) -> dict[str, str]:
    response = client.get("/api/v1/admin/datasets")
    response.raise_for_status()
    result: dict[str, str] = {}
    for item in response.json():
        name = item.get("name") or item.get("dataset_name")
        dataset_id = item.get("id") or item.get("dataset_id")
        if name and dataset_id:
            result[str(name)] = str(dataset_id)
    return result


def _contains_any(value: str, keywords: list[str]) -> bool:
    return any(keyword in value for keyword in keywords)


def validate(
    case: dict[str, Any], payload: dict[str, Any], elapsed: float
) -> tuple[list[str], list[str]]:
    expected = case["expected"]
    budgets = case["budgets"]
    failures: list[str] = []
    warnings: list[str] = []
    status = payload.get("status")
    answer = str(payload.get("answer") or "")
    citations = payload.get("citations") or []
    missing = payload.get("missing_information") or []

    if status not in expected["status_in"]:
        failures.append(f"status={status!r}, expected one of {expected['status_in']}")
    for term in expected.get("required_answer_terms", []):
        if term not in answer:
            failures.append(f"answer is missing required term {term!r}")
    for term in expected.get("forbidden_answer_terms", []):
        if term in answer:
            failures.append(f"answer contains forbidden term {term!r}")
    if expected["must_cite"] and status in ANSWERABLE and not citations:
        failures.append("answerable response has no citations")
    if expected["must_report_missing"] and status in {"PARTIALLY_ANSWERABLE", "UNANSWERABLE"}:
        if not missing:
            failures.append("partial/unanswerable response did not report missing information")

    allowed = expected["allowed_document_keywords"]
    forbidden = expected["forbidden_document_keywords"]
    for citation in citations:
        document_name = str(citation.get("document_name") or "")
        if allowed and not _contains_any(document_name, allowed):
            failures.append(f"citation uses out-of-scope document {document_name!r}")
        if forbidden and _contains_any(document_name, forbidden):
            failures.append(f"citation uses forbidden document {document_name!r}")

    if elapsed > budgets["latency_seconds"]:
        failures.append(
            f"latency {elapsed:.2f}s exceeds budget {budgets['latency_seconds']}s"
        )

    diagnostics = payload.get("diagnostics") or {}
    cells = diagnostics.get("cells")
    required_cell_count = expected["required_cell_count"]
    if cells is not None and len(cells) != required_cell_count:
        failures.append(f"cell_count={len(cells)}, expected {required_cell_count}")
    elif cells is None and required_cell_count > 0:
        warnings.append("cell count is not observable; planning_cases.jsonl remains authoritative")
    model_calls = diagnostics.get("model_calls")
    if model_calls is not None and model_calls > budgets["max_total_model_calls"]:
        failures.append(
            f"model_calls={model_calls}, budget={budgets['max_total_model_calls']}"
        )
    elif model_calls is None:
        warnings.append("model call count is not observable")
    breakdown = diagnostics.get("model_call_breakdown") or {}
    for component, budget_key in (
        ("planner", "max_planner_calls"),
        ("translation", "max_translation_calls"),
        ("summary", "max_summary_calls"),
    ):
        calls = int(breakdown.get(component, 0))
        if calls > budgets[budget_key]:
            failures.append(f"{component}_calls={calls}, budget={budgets[budget_key]}")
    return failures, warnings


def run(args: argparse.Namespace) -> int:
    cases = load_cases(args.include_exploratory)
    headers = {"X-User-ID": args.user_id}
    results: list[dict[str, Any]] = []
    # Live evaluation targets the local gateway.  Do not inherit a workstation's
    # HTTP(S)_PROXY settings: some corporate proxy configurations do not exempt
    # 127.0.0.1 and would make this reproducibility check hit the public network.
    with httpx.Client(
        base_url=args.base_url,
        headers=headers,
        timeout=args.timeout,
        trust_env=False,
    ) as client:
        datasets = dataset_map(client)
        for case in cases:
            started = time.perf_counter()
            failures: list[str] = []
            warnings: list[str] = []
            payload: dict[str, Any] = {}
            dataset_id = datasets.get(case["knowledge_base_name"])
            if dataset_id is None:
                failures.append(f"knowledge base {case['knowledge_base_name']!r} was not found")
                elapsed = time.perf_counter() - started
            else:
                try:
                    response = client.post(
                        "/api/v1/chat/completions",
                        json={"query": case["query"], "knowledge_base_ids": [dataset_id]},
                    )
                    elapsed = time.perf_counter() - started
                    response.raise_for_status()
                    payload = response.json()
                    validation_failures, warnings = validate(case, payload, elapsed)
                    failures.extend(validation_failures)
                except (httpx.HTTPError, ValueError) as exc:
                    elapsed = time.perf_counter() - started
                    failures.append(f"request failed: {exc}")

            results.append(
                {
                    "id": case["id"],
                    "must_pass": case["must_pass"],
                    "passed": not failures,
                    "elapsed_seconds": round(elapsed, 3),
                    "status": payload.get("status"),
                    "answer": payload.get("answer"),
                    "citation_count": len(payload.get("citations") or []),
                    "diagnostics": payload.get("diagnostics"),
                    "failures": failures,
                    "warnings": warnings,
                }
            )

    report = {
        "base_url": args.base_url,
        "case_count": len(results),
        "passed": sum(result["passed"] for result in results),
        "failed": sum(not result["passed"] for result in results),
        "results": results,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    required_failures = [
        result for result in results if result["must_pass"] and not result["passed"]
    ]
    return 1 if required_failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run live RAG evaluation cases")
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--user-id", default="evaluation-runner")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--include-exploratory", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (httpx.HTTPError, ValueError) as exc:
        print(f"evaluation setup failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
