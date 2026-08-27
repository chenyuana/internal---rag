from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.config import Settings
from app.core.exceptions import AppError
from app.services.query_analyzer import QueryAnalyzer
from app.services.structured_planner_model import StructuredPlannerModel


class PlannerReply:
    def __init__(self, value: str) -> None:
        self.value = value

    async def chat_completion(
        self, *, messages: list[dict[str, str]], response_schema: dict[str, Any]
    ) -> str:
        assert messages and response_schema
        return self.value


def _plan(query_text: str) -> dict[str, Any]:
    return {
        "version": "query-plan-v2",
        "query_type": "comparison",
        "subjects": ["甲型无人机", "乙型无人机"],
        "aspects": ["续航时间"],
        "cells": [
            {
                "id": "q1",
                "subject": "甲型无人机",
                "aspect": "续航时间",
                "original_query": "甲型无人机 续航时间",
                "retrieval_queries": [
                    {
                        "kind": "original",
                        "text": "甲型无人机 续航时间",
                        "generated_by": "planner",
                    }
                ],
            },
            {
                "id": "q2",
                "subject": "乙型无人机",
                "aspect": "续航时间",
                "original_query": "乙型无人机 续航时间",
                "retrieval_queries": [
                    {
                        "kind": "original",
                        "text": "乙型无人机 续航时间",
                        "generated_by": "planner",
                    }
                ],
            },
        ],
        "synthesis_mode": "matrix",
        "planner_source": "model",
        "confidence": 0.95,
        "warnings": [],
    }


async def test_structured_planner_accepts_valid_bounded_plan(settings: Settings) -> None:
    query = QueryAnalyzer().analyze("比较甲型无人机和乙型无人机的续航时间。")
    model = StructuredPlannerModel(
        settings.planning,
        PlannerReply(json.dumps(_plan(query.normalized_query), ensure_ascii=False)),
    )
    result = await model.plan(query)
    assert result.planner_source == "model"
    assert len(result.cells) == 2


async def test_structured_planner_rejects_bad_json(settings: Settings) -> None:
    model = StructuredPlannerModel(settings.planning, PlannerReply("not-json"))
    with pytest.raises(AppError, match="valid bounded plan"):
        await model.plan(QueryAnalyzer().analyze("比较甲设备和乙设备。"))


async def test_structured_planner_rejects_invented_standard_number(
    settings: Settings,
) -> None:
    query = QueryAnalyzer().analyze("比较甲型无人机和乙型无人机的续航时间。")
    value = _plan(query.normalized_query)
    value["cells"][0]["original_query"] += " GB/T-9999"
    value["cells"][0]["retrieval_queries"][0]["text"] += " GB/T-9999"
    model = StructuredPlannerModel(
        settings.planning,
        PlannerReply(json.dumps(value, ensure_ascii=False)),
    )
    with pytest.raises(AppError):
        await model.plan(query)
