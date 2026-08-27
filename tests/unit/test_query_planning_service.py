from __future__ import annotations

from typing import Any, cast

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.planning import PlannedCell, QueryPlanV2, RetrievalQuery
from app.services.query_analyzer import QueryAnalyzer
from app.services.query_planning_service import QueryPlanningService
from app.services.structured_planner_model import StructuredPlannerModel


class StructuredStub:
    def __init__(self, plan: QueryPlanV2 | None = None, *, fail: bool = False) -> None:
        self.plan_value = plan
        self.fail = fail
        self.calls = 0

    async def plan(self, query: object) -> QueryPlanV2:
        del query
        self.calls += 1
        if self.fail:
            raise AppError(code="TEST", message="failed", status_code=502)
        assert self.plan_value is not None
        return self.plan_value


def _model_plan() -> QueryPlanV2:
    subjects = ["甲设备", "乙设备"]
    cells = [
        PlannedCell(
            id=f"q{index}",
            subject=subject,
            aspect="精度",
            original_query=f"{subject} 精度",
            retrieval_queries=[
                RetrievalQuery(
                    kind="original",
                    text=f"{subject} 精度",
                    generated_by="planner",
                )
            ],
        )
        for index, subject in enumerate(subjects, start=1)
    ]
    return QueryPlanV2(
        query_type="comparison",
        subjects=subjects,
        aspects=["精度"],
        cells=cells,
        synthesis_mode="matrix",
        planner_source="model",
        confidence=0.9,
    )


async def test_exact_query_uses_deterministic_fast_path(settings: Settings) -> None:
    stub = StructuredStub(_model_plan())
    service = QueryPlanningService(
        settings.planning.model_copy(update={"shadow_mode": False, "model_enabled": True}),
        structured=cast(StructuredPlannerModel, cast(Any, stub)),
    )
    result = await service.plan(QueryAnalyzer().analyze("CCAR-25-R4 第25.981条是什么？"))
    assert result.plan.planner_source == "deterministic"
    assert result.model_calls == 0
    assert stub.calls == 0


async def test_model_plan_drives_when_shadow_is_disabled(settings: Settings) -> None:
    stub = StructuredStub(_model_plan())
    service = QueryPlanningService(
        settings.planning.model_copy(update={"shadow_mode": False, "model_enabled": True}),
        structured=cast(StructuredPlannerModel, cast(Any, stub)),
    )
    result = await service.plan(QueryAnalyzer().analyze("比较甲设备和乙设备在精度上的差异"))
    assert result.plan.planner_source == "model"
    assert result.legacy_plan.subqueries[0].query == "甲设备 精度"
    assert result.model_calls == 1


async def test_model_failure_falls_back_to_legacy(settings: Settings) -> None:
    stub = StructuredStub(fail=True)
    service = QueryPlanningService(
        settings.planning.model_copy(update={"shadow_mode": False, "model_enabled": True}),
        structured=cast(StructuredPlannerModel, cast(Any, stub)),
    )
    result = await service.plan(QueryAnalyzer().analyze("比较甲设备和乙设备在精度上的差异"))
    assert result.plan.planner_source == "legacy_fallback"
    assert result.fallback is True
