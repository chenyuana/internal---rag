from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.config import GenerationSettings
from app.core.exceptions import AppError
from app.schemas.chat import (
    EvidenceAssessment,
    EvidenceSentence,
    ReferenceDocument,
    StructuredAnswer,
)
from app.services.answer_generator import AnswerGenerator, SanitizedAnswer


def test_parse_deduplicates_duplicate_claim_ids() -> None:
    """模型输出重复 claim_id 时自动唯一化，不抛 schema 校验错误。"""
    raw = json.dumps(
        {
            "answerability": "ANSWERABLE",
            "answer": "内容一。[C1] 内容二。[C2]",
            "claims": [
                {"claim_id": "c1", "claim": "内容一。", "citation_ids": ["C1"]},
                {"claim_id": "c1", "claim": "内容二。", "citation_ids": ["C2"]},
            ],
            "missing_information": [],
            "conflicts": [],
        }
    )

    parsed = AnswerGenerator._parse(raw)

    ids = [claim.claim_id for claim in parsed.claims]
    assert len(ids) == len(set(ids))
    assert parsed.claims[1].claim_id == "c1_2"


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


async def test_answer_generator_passes_reference_as_presentation_only_payload() -> None:
    assessment, evidence = generation_inputs()
    captured: dict[str, Any] = {}

    class CapturingModel(FakeAnswerModel):
        async def chat_completion(self, *, messages, response_schema):
            captured.update(json.loads(messages[1]["content"]))
            return await super().chat_completion(
                messages=messages,
                response_schema=response_schema,
            )

    generator = AnswerGenerator(
        settings=GenerationSettings(),
        model=CapturingModel([valid_answer()]),
    )
    await generator.generate(
        question="任务周期是多少？",
        assessment=assessment,
        evidence=evidence,
        reference_document=ReferenceDocument(name="模板.md", text="# 摘要\n分节表达"),
        semantic_validator=lambda _: None,
    )

    assert captured["presentation_reference"] == {
        "name": "模板.md",
        "text": "# 摘要\n分节表达",
    }


async def test_answer_generator_repairs_answerable_output_without_claims() -> None:
    assessment, evidence = generation_inputs()
    missing_claims = json.dumps(
        {
            "answerability": "ANSWERABLE",
            "answer": "任务周期为1000毫秒。[C1]",
            "missing_information": [],
            "conflicts": [],
        },
        ensure_ascii=False,
    )
    captured_repair_payload: dict[str, Any] = {}

    class CapturingRepairModel(FakeAnswerModel):
        async def chat_completion(
            self,
            *,
            messages: list[dict[str, str]],
            response_schema: dict[str, Any],
        ) -> str:
            if self.calls == 1:
                captured_repair_payload.update(json.loads(messages[1]["content"]))
            return await super().chat_completion(
                messages=messages,
                response_schema=response_schema,
            )

    model = CapturingRepairModel([missing_claims, valid_answer()])
    generator = AnswerGenerator(settings=GenerationSettings(), model=model)

    result = await generator.generate(
        question="任务周期是多少？",
        assessment=assessment,
        evidence=evidence,
        semantic_validator=lambda _: None,
    )

    assert result.repaired is True
    assert len(result.answer.claims) == 1
    assert "at least one claim is required" in captured_repair_payload["validation_error"]
    assert captured_repair_payload["allowed_citation_ids"] == ["C1"]


async def test_answer_generator_multiple_repair_rounds_until_valid() -> None:
    """json_repair_attempts=2 时允许多轮修复：每轮修复都可能导致新问题，
    校验失败后可再修（修复也可能引入新问题，如丢掉对比格子）。"""
    assessment, evidence = generation_inputs()
    model = FakeAnswerModel(["bad-1", "bad-2", valid_answer()])
    generator = AnswerGenerator(
        settings=GenerationSettings(json_repair_attempts=2),
        model=model,
    )
    failures = {"bad-1", "bad-2"}

    def validate(answer: StructuredAnswer) -> None:
        for claim in answer.claims:
            if claim.claim in failures:
                raise ValueError("still invalid")

    result = await generator.generate(
        question="任务周期是多少？",
        assessment=assessment,
        evidence=evidence,
        semantic_validator=validate,
    )

    assert result.repaired is True
    assert model.calls == 3
    assert result.answer.answerability == "ANSWERABLE"


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

    assert captured.value.code == "ANSWER_JSON_PARSE_FAILED"
    assert captured.value.details["validation_error"]


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


async def test_answer_generator_sanitizes_after_repair_semantic_failure() -> None:
    assessment, evidence = generation_inputs()
    model = FakeAnswerModel([valid_answer(), valid_answer()])
    generator = AnswerGenerator(settings=GenerationSettings(), model=model)

    def reject(_: StructuredAnswer) -> None:
        raise ValueError("unsupported numeric claim")

    def sanitize(
        _: StructuredAnswer,
        error: Exception,
    ) -> SanitizedAnswer:
        assert str(error) == "unsupported numeric claim"
        return SanitizedAnswer(
            answer=StructuredAnswer(
                answerability="UNANSWERABLE",
                answer="生成内容未通过校验。",
                claims=[],
                missing_information=["证据不足"],
                conflicts=[],
            ),
            warnings=["已安全降级"],
        )

    validation_calls = 0

    def validate(answer: StructuredAnswer) -> None:
        nonlocal validation_calls
        validation_calls += 1
        if answer.answerability != "UNANSWERABLE":
            reject(answer)

    result = await generator.generate(
        question="任务周期是多少？",
        assessment=assessment,
        evidence=evidence,
        semantic_validator=validate,
        semantic_sanitizer=sanitize,
    )

    assert result.answer.answerability == "UNANSWERABLE"
    assert result.repaired is True
    assert result.validation_degraded is True
    assert result.validation_warnings == ["已安全降级"]
    assert validation_calls == 3


async def test_answer_generator_initial_payload_carries_citation_whitelist() -> None:
    """首版生成 payload 必须带可引用编号白名单与出处映射，
    模型不得编造证据外的引用编号（避免 repair 阶段仍复现错误编号）。"""
    captured: dict[str, Any] = {}

    class CapturingModel:
        async def chat_completion(
            self,
            *,
            messages: list[dict[str, str]],
            response_schema: dict[str, Any],
        ) -> str:
            captured["payload"] = json.loads(messages[1]["content"])
            return valid_answer()

    assessment, _ = generation_inputs()
    evidence = [
        EvidenceSentence(
            citation_id="C1",
            chunk_id="chunk-1",
            text="A",
            document_name="CCAR-25-R4 运输类飞机适航标准.pdf",
            version="R4",
        ),
        EvidenceSentence(
            citation_id="C2",
            chunk_id="chunk-2",
            text="B",
            document_name="CCAR-26 运输类飞机的持续适航和安全改进规定.pdf",
            version="R2",
        ),
    ]
    generator = AnswerGenerator(settings=GenerationSettings(), model=CapturingModel())

    result = await generator.generate(
        question="任务周期是多少？",
        assessment=assessment,
        evidence=evidence,
        semantic_validator=lambda _: None,
    )

    assert result.repaired is False
    payload = captured["payload"]
    assert payload["allowed_citation_ids"] == ["C1", "C2"]
    assert (
        payload["citation_documents"]["C1"]["document_name"] == "CCAR-25-R4 运输类飞机适航标准.pdf"
    )
    assert {item["citation_id"] for item in payload["evidence"]} == {"C1", "C2"}


async def test_answer_generator_repair_payload_carries_citation_documents() -> None:
    # Repair 通道必须带 citation→document_name 出处映射，模型才能修正跨规章归因。
    captured: dict[str, Any] = {}

    class CapturingModel:
        async def chat_completion(
            self,
            *,
            messages: list[dict[str, str]],
            response_schema: dict[str, Any],
        ) -> str:
            captured["payload"] = json.loads(messages[1]["content"])
            return valid_answer()

    assessment, _ = generation_inputs()
    evidence = [
        EvidenceSentence(
            citation_id="C1",
            chunk_id="chunk-1",
            text="A",
            document_name="CCAR-25-R4 运输类飞机适航标准.pdf",
            version="R4",
        ),
        EvidenceSentence(
            citation_id="C4",
            chunk_id="chunk-4",
            text="B",
            document_name="CCAR-26 运输类飞机的持续适航和安全改进规定.pdf",
            version="R2",
        ),
    ]
    generator = AnswerGenerator(settings=GenerationSettings(), model=CapturingModel())
    validation_calls = 0

    def validate(_: StructuredAnswer) -> None:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            raise ValueError("scope violation")

    result = await generator.generate(
        question="任务周期是多少？",
        assessment=assessment,
        evidence=evidence,
        semantic_validator=validate,
    )

    assert result.repaired is True
    documents = captured["payload"]["citation_documents"]
    assert documents["C1"]["document_name"] == "CCAR-25-R4 运输类飞机适航标准.pdf"
    assert documents["C4"]["document_name"] == "CCAR-26 运输类飞机的持续适航和安全改进规定.pdf"
    assert documents["C4"]["version"] == "R2"
    repair_evidence = captured["payload"]["evidence"]
    assert {item["citation_id"] for item in repair_evidence} == {"C1", "C4"}
    assert repair_evidence[0]["text"] == "A"
    assert repair_evidence[1]["document_name"] == "CCAR-26 运输类飞机的持续适航和安全改进规定.pdf"
