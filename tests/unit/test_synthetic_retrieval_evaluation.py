from __future__ import annotations

from tests.evaluation.synthetic_retrieval_eval import evaluate_case, load_cases


def test_retrieval_cases_pass_with_migration_taxonomy() -> None:
    results = [evaluate_case(case, use_taxonomy=True) for case in load_cases()]
    assert all(result["passed"] for result in results), results


def test_taxonomy_removal_is_measured_before_cleanup() -> None:
    cases = load_cases()
    with_taxonomy = sum(evaluate_case(case, use_taxonomy=True)["passed"] for case in cases)
    without_taxonomy = sum(
        evaluate_case(case, use_taxonomy=False)["passed"] for case in cases
    )
    assert with_taxonomy >= without_taxonomy
