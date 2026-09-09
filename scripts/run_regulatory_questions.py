"""Run an external question dataset against a configured gateway; record, never score blindly."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--knowledge-base-id", required=True)
    parser.add_argument("--document-id", required=True, action="append")
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only the named case ID; repeat for a representative subset.",
    )
    parser.add_argument("--user-id", default="development-user")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    cases = [json.loads(line) for line in args.questions.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    if args.case_id:
        requested = set(args.case_id)
        cases = [case for case in cases if case.get("id") in requested]
        missing = sorted(requested - {str(case.get("id")) for case in cases})
        if missing:
            parser.error("unknown --case-id value(s): " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream, httpx.Client(
        # Evaluation targets the local gateway.  Do not inherit a desktop or
        # corporate HTTP proxy: those proxies treat localhost as a webpage and
        # return a synthetic 400 before the request reaches intelligent QA.
        timeout=args.timeout,
        headers={"X-User-ID": args.user_id},
        trust_env=False,
    ) as client:
        for case in cases:
            started = time.perf_counter()
            record = {"id": case["id"], "question": case["question"], "answer_score": None}
            try:
                endpoint = args.base_url.rstrip("/") + "/api/v1/chat/completions"
                response = client.post(endpoint, json={
                    "query": case["question"],
                    "knowledge_base_ids": [args.knowledge_base_id],
                    "document_ids": args.document_id,
                })
                record["http_status"] = response.status_code
                try:
                    record["response"] = response.json()
                except ValueError:
                    # Preserve the server response for diagnosing proxy errors,
                    # HTML error pages, and truncated upstream responses.
                    record["response_text"] = response.text[:20_000]
            except httpx.HTTPError as exc:
                record["error"] = type(exc).__name__
                record["error_message"] = str(exc)
            record["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"{case['id']}: recorded", flush=True)


if __name__ == "__main__":
    main()
