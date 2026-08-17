from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.config import GenerationSettings
from app.core.exceptions import AppError
from app.schemas.chat import EvidenceAssessment, EvidenceSentence, StructuredAnswer
from app.services.answer_generator import AnswerGenerator


class FakeAnswerModel:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str:
        assert messages
        assert response_schema["type"] == "object"
        response = self.responses[self.calls]
        self.calls += 1
        return response


def valid_answer() -> str:
    return json.dumps(
        {
            "answerability": "ANSWERABLE",
            "answer": "任务周期为1000毫秒。[C1]",
            "claims": [
                {
                    "claim_id": "claim-1",
                    "claim": "任务周期为1000毫秒",
                    "citation_ids": ["C1"],
                }
            ],
            "missing_information": [],
            "conflicts": [],
        },
        ensure_ascii=False,
    )


def generation_inputs() -> tuple[EvidenceAssessment, list[EvidenceSentence]]:
    return (
        EvidenceAssessment(
            status="ANSWERABLE",
            covered_requirements=["执行周期"],
        ),
        [
            EvidenceSentence(
                citation_id="C1",
                chunk_id="chunk-1",
                text="HEALTH_MONITOR_PERIOD_MS = 1000U",
            )
        ],
    )


async def test_answer_generator_repairs_invalid_json_once() -> None:
    assessment, evidence = generation_inputs()
    model = FakeAnswerModel(["not-json", valid_answer()])
    generator = AnswerGenerator(settings=GenerationSettings(), model=model)

    result = await generator.generate(
        question="任务周期是多少？",
        assessment=assessment,
        evidence=evidence,
        semantic_validator=lambda _: None,
    )

    assert result.answer.answerability == "ANSWERABLE"
    assert result.repaired is True
    assert model.calls == 2


async def test_answer_generator_fails_safely_after_one_repair() -> None:
    assessment, evidence = generation_inputs()
    generator = AnswerGenerator(
        settings=GenerationSettings(),
        model=FakeAnswerModel(["not-json", "still-not-json"]),
    )

    with pytest.raises(AppError) as captured:
        await generator.generate(
            question="任务周期是多少？",
            assessment=assessment,
            evidence=evidence,
            semantic_validator=lambda _: None,
        )

    assert captured.value.code == "ANSWER_MODEL_INVALID_JSON"


async def test_answer_generator_repairs_semantically_invalid_output() -> None:
    assessment, evidence = generation_inputs()
    model = FakeAnswerModel([valid_answer(), valid_answer()])
    generator = AnswerGenerator(settings=GenerationSettings(), model=model)
    validation_calls = 0

    def validate(_: StructuredAnswer) -> None:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            raise ValueError("citation C99 is not allowed")

    result = await generator.generate(
        question="任务周期是多少？",
        assessment=assessment,
        evidence=evidence,
        semantic_validator=validate,
    )

    assert result.repaired is True
    assert validation_calls == 2
