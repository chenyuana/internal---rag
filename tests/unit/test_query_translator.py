from __future__ import annotations

import json
from typing import Any

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.planning import PlannedCell, RetrievalQuery
from app.services.query_analyzer import QueryAnalyzer
from app.services.query_translator import QueryTranslator


class TranslationReply:
    def __init__(self, payload: dict[str, object] | str) -> None:
        self.payload = payload

    async def chat_completion(
        self, *, messages: list[dict[str, str]], response_schema: dict[str, Any]
    ) -> str:
        assert messages and response_schema
        return self.payload if isinstance(self.payload, str) else json.dumps(self.payload)


class TranslationFailure:
    async def chat_completion(
        self, *, messages: list[dict[str, str]], response_schema: dict[str, Any]
    ) -> str:
        del messages, response_schema
        raise AppError(code="MODEL_DOWN", message="down", status_code=503)


def _cell() -> PlannedCell:
    query = "长大桥梁无人机精细化巡检 成果验收"
    return PlannedCell(
        id="q1",
        subject="长大桥梁无人机精细化巡检",
        aspect="成果验收",
        original_query=query,
        retrieval_queries=[
            RetrievalQuery(kind="original", text=query, generated_by="planner")
        ],
    )


async def test_translation_accepts_one_subject_preserving_variant(settings: Settings) -> None:
    translator = QueryTranslator(
        settings.translation.model_copy(update={"model_enabled": True}),
        TranslationReply(
            {
                "should_translate": True,
                "translated_query": "长大桥梁无人机精细化巡检 成果验收 巡检报告",
                "reason": "首次结果只有章节标题",
            }
        ),
    )
    result = await translator.translate(
        query=QueryAnalyzer().analyze("长大桥梁无人机精细化巡检成果验收要求是什么？"),
        cell=_cell(),
        section_titles=["9.4 巡检成果"],
    )
    assert result.decision.should_translate is True
    assert result.model_calls == 1


async def test_translation_rejects_changed_subject_and_invented_number(
    settings: Settings,
) -> None:
    translator = QueryTranslator(
        settings.translation.model_copy(update={"model_enabled": True}),
        TranslationReply(
            {
                "should_translate": True,
                "translated_query": "地质灾害摄影测量 成果验收 GB/T-9999",
                "reason": "改写",
            }
        ),
    )
    result = await translator.translate(
        query=QueryAnalyzer().analyze("长大桥梁无人机精细化巡检成果验收要求是什么？"),
        cell=_cell(),
        section_titles=[],
    )
    assert result.decision.should_translate is False
    assert result.fallback_reason == "ValueError"


async def test_exact_article_never_translates(settings: Settings) -> None:
    translator = QueryTranslator(
        settings.translation.model_copy(update={"model_enabled": True}),
        TranslationReply("not-used"),
    )
    result = await translator.translate(
        query=QueryAnalyzer().analyze("CCAR-25-R4 第25.981条是什么？"),
        cell=_cell(),
        section_titles=[],
    )
    assert result.model_calls == 0
    assert result.decision.should_translate is False


async def test_model_network_failure_keeps_original_query(settings: Settings) -> None:
    translator = QueryTranslator(
        settings.translation.model_copy(update={"model_enabled": True}),
        TranslationFailure(),
    )
    result = await translator.translate(
        query=QueryAnalyzer().analyze("长大桥梁无人机精细化巡检成果验收要求是什么？"),
        cell=_cell(),
        section_titles=[],
    )
    assert result.decision.should_translate is False
    assert result.fallback_reason == "AppError"
