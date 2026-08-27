from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from app.core.config import PlanningSettings
from app.core.exceptions import AppError
from app.schemas.planning import QueryPlanV2
from app.schemas.retrieval import NormalizedQuery, QueryPlan
from app.services.query_plan_adapter import QueryPlanAdapter
from app.services.query_planner import QueryPlanner
from app.services.structured_planner_model import StructuredPlannerModel

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlanningOutcome:
    plan: QueryPlanV2
    legacy_plan: QueryPlan
    shadow_plan: QueryPlanV2 | None
    latency_ms: float
    model_calls: int
    fallback: bool


class QueryPlanningService:
    def __init__(
        self,
        settings: PlanningSettings,
        *,
        structured: StructuredPlannerModel | None,
        legacy: QueryPlanner | None = None,
    ) -> None:
        self._settings = settings
        self._structured = structured
        self._legacy = legacy or QueryPlanner()

    async def plan(self, query: NormalizedQuery) -> PlanningOutcome:
        started = time.perf_counter()
        legacy_plan = self._legacy.plan(query)
        legacy_v2 = QueryPlanAdapter.from_legacy(query, legacy_plan)
        if not self._settings.enabled:
            return self._outcome(started, legacy_v2, legacy_plan, None, 0, True)
        if self._use_fast_path(query):
            fast = QueryPlanAdapter.deterministic_single(query)
            return self._outcome(started, fast, QueryPlanAdapter.to_legacy(fast), None, 0, False)
        if not self._settings.model_enabled:
            return self._outcome(started, legacy_v2, legacy_plan, None, 0, False)
        if self._structured is None:
            return self._outcome(started, legacy_v2, legacy_plan, None, 0, True)

        try:
            model_plan = await self._structured.plan(query)
        except AppError as exc:
            logger.warning(
                "query_planner_fallback",
                extra={"error_code": exc.code, "error_type": exc.details.get("error_type")},
            )
            if not self._settings.fallback_to_legacy:
                raise
            return self._outcome(started, legacy_v2, legacy_plan, None, 1, True)
        if self._settings.shadow_mode:
            return self._outcome(started, legacy_v2, legacy_plan, model_plan, 1, False)
        return self._outcome(
            started,
            model_plan,
            QueryPlanAdapter.to_legacy(model_plan),
            None,
            1,
            False,
        )

    def _use_fast_path(self, query: NormalizedQuery) -> bool:
        if not self._settings.use_deterministic_fast_path:
            return False
        if query.article_ids or query.exact_tokens:
            return True
        return query.query_type in {"fact", "procedure", "summary", "unknown"}

    @staticmethod
    def _outcome(
        started: float,
        plan: QueryPlanV2,
        legacy_plan: QueryPlan,
        shadow_plan: QueryPlanV2 | None,
        model_calls: int,
        fallback: bool,
    ) -> PlanningOutcome:
        return PlanningOutcome(
            plan=plan,
            legacy_plan=legacy_plan,
            shadow_plan=shadow_plan,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            model_calls=model_calls,
            fallback=fallback,
        )
