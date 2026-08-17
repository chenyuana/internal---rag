from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, GenerationSettings
from app.core.exceptions import AppError
from app.schemas.chat import EvidenceAssessment, EvidenceSentence, StructuredAnswer


class AnswerModel(Protocol):
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str: ...


@dataclass(slots=True)
class GenerationResult:
    answer: StructuredAnswer
    repaired: bool


class AnswerGenerator:
    """Generate and validate one strict structured answer, with at most one repair."""

    def __init__(
        self,
        *,
        settings: GenerationSettings,
        model: AnswerModel,
    ) -> None:
        self._settings = settings
        self._model = model
        self._answer_prompt = self._read_prompt(settings.answer_prompt_path)
        self._repair_prompt = self._read_prompt(settings.repair_prompt_path)

    async def generate(
        self,
        *,
        question: str,
        assessment: EvidenceAssessment,
        evidence: list[EvidenceSentence],
        semantic_validator: Callable[[StructuredAnswer], None],
    ) -> GenerationResult:
        user_payload = {
            "question": question,
            "evidence_assessment": assessment.model_dump(),
            "evidence": [item.model_dump() for item in evidence],
        }
        raw = await self._model.chat_completion(
            messages=[
                {"role": "system", "content": self._answer_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            response_schema=StructuredAnswer.model_json_schema(),
        )
        try:
            answer = self._parse_and_validate(raw, semantic_validator)
            return GenerationResult(answer=answer, repaired=False)
        except (json.JSONDecodeError, ValidationError, ValueError) as first_error:
            if self._settings.json_repair_attempts == 0:
                raise self._invalid_output_error(first_error) from first_error
            first_error_message = str(first_error)

        repaired_raw = await self._model.chat_completion(
            messages=[
                {"role": "system", "content": self._repair_prompt},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "invalid_output": raw,
                            "validation_error": first_error_message,
                            "question": question,
                            "allowed_citation_ids": sorted(
                                {item.citation_id for item in evidence}
                            ),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            response_schema=StructuredAnswer.model_json_schema(),
        )
        try:
            answer = self._parse_and_validate(repaired_raw, semantic_validator)
        except (json.JSONDecodeError, ValidationError, ValueError) as repair_error:
            raise self._invalid_output_error(repair_error) from repair_error
        return GenerationResult(answer=answer, repaired=True)

    @staticmethod
    def _parse_and_validate(
        raw: str,
        semantic_validator: Callable[[StructuredAnswer], None],
    ) -> StructuredAnswer:
        content = raw.strip()
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1]).strip()
        parsed = json.loads(content)
        answer = StructuredAnswer.model_validate(parsed)
        semantic_validator(answer)
        return answer

    @staticmethod
    def _invalid_output_error(error: Exception) -> AppError:
        return AppError(
            code="ANSWER_MODEL_INVALID_JSON",
            message="The local answer model did not return a valid constrained JSON response.",
            status_code=502,
            details={"error_type": type(error).__name__},
        )

    @staticmethod
    def _read_prompt(relative_path: str) -> str:
        prompt_path = (PROJECT_ROOT / Path(relative_path)).resolve()
        if not prompt_path.is_relative_to(PROJECT_ROOT):
            raise AppError(
                code="INVALID_PROMPT_PATH",
                message="Prompt paths must remain inside the project directory.",
                status_code=500,
            )
        try:
            content = prompt_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise AppError(
                code="PROMPT_NOT_FOUND",
                message=f"Required prompt file is unavailable: {relative_path}",
                status_code=500,
            ) from exc
        if not content:
            raise AppError(
                code="PROMPT_EMPTY",
                message=f"Required prompt file is empty: {relative_path}",
                status_code=500,
            )
        return content
