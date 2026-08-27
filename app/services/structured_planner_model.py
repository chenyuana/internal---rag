from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, PlanningSettings
from app.core.exceptions import AppError
from app.schemas.planning import QueryPlanV2
from app.schemas.retrieval import NormalizedQuery

PROTECTED_TOKEN_RE = re.compile(
    r"(?:[A-Z]{2,}(?:[-/][A-Z0-9]+)+|\d+(?:\.\d+){1,}(?:条)?)",
    re.IGNORECASE,
)


class PlannerModel(Protocol):
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str: ...


class StructuredPlannerModel:
    def __init__(self, settings: PlanningSettings, model: PlannerModel) -> None:
        self._settings = settings
        self._model = model
        self._prompt = self._read_prompt(settings.prompt_path)

    async def plan(self, query: NormalizedQuery) -> QueryPlanV2:
        payload = {
            "normalized_query": query.normalized_query,
            "deterministic_query_type_hint": query.query_type,
            "exact_tokens": query.exact_tokens,
            "article_ids": query.article_ids,
            "max_cells": self._settings.max_subqueries,
        }
        try:
            raw = await asyncio.wait_for(
                self._model.chat_completion(
                    messages=[
                        {"role": "system", "content": self._prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_schema=QueryPlanV2.model_json_schema(),
                ),
                timeout=self._settings.timeout_seconds,
            )
            plan = QueryPlanV2.model_validate(self._parse_json(raw))
            plan = plan.model_copy(update={"planner_source": "model"})
            self._validate_preservation(query, plan)
            return plan
        except (TimeoutError, json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise AppError(
                code="QUERY_PLANNER_INVALID_RESPONSE",
                message="The query planner did not return a valid bounded plan.",
                status_code=502,
                details={"error_type": type(exc).__name__},
            ) from exc

    @staticmethod
    def _parse_json(raw: str) -> Any:
        content = raw.strip()
        if content.startswith("```") and content.endswith("```"):
            content = "\n".join(content.splitlines()[1:-1]).strip()
        return json.loads(content)

    @staticmethod
    def _validate_preservation(query: NormalizedQuery, plan: QueryPlanV2) -> None:
        cell_text = " ".join(
            [
                *(cell.original_query for cell in plan.cells),
                *plan.subjects,
                *plan.aspects,
            ]
        )
        missing = [token for token in query.exact_tokens if token not in cell_text]
        if missing:
            raise ValueError(f"planner removed exact tokens: {missing}")
        source_tokens = set(PROTECTED_TOKEN_RE.findall(query.normalized_query))
        planned_tokens = set(PROTECTED_TOKEN_RE.findall(cell_text))
        invented = planned_tokens - source_tokens
        if invented:
            raise ValueError(f"planner invented protected tokens: {sorted(invented)}")

    @staticmethod
    def _read_prompt(relative_path: str) -> str:
        path = (PROJECT_ROOT / Path(relative_path)).resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise AppError(
                code="INVALID_PROMPT_PATH",
                message="Prompt paths must remain inside the project directory.",
                status_code=500,
            )
        return path.read_text(encoding="utf-8").strip()
