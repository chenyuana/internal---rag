from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.services.query_analyzer import QueryAnalyzer
from app.services.query_planning_service import QueryPlanningService

CASES = [
    json.loads(line)
    for line in (
        Path(__file__).parents[1] / "evaluation" / "planning_cases.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    if line.strip()
]


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
async def test_planning_target_cases(case: dict[str, Any], settings: Settings) -> None:
    service = QueryPlanningService(settings.planning, structured=None)
    outcome = await service.plan(QueryAnalyzer().analyze(case["query"]))
    plan = outcome.plan
    expected = case["expected"]

    assert plan.query_type == expected["query_type"]
    assert plan.subjects == expected["subjects"]
    assert plan.aspects == expected["aspects"]
    assert plan.synthesis_mode == expected["synthesis_mode"]
    if "cell_count" in expected:
        assert len(plan.cells) == expected["cell_count"]
    if "cell_count_max" in expected:
        assert len(plan.cells) <= expected["cell_count_max"]
    if "planner_source" in expected:
        assert plan.planner_source == expected["planner_source"]
    if warning := expected.get("warning_contains"):
        assert any(warning in item for item in plan.warnings)
    if case["translation"]["allowed"] is False:
        assert outcome.model_calls == 0
