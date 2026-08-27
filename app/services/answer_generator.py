from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, GenerationSettings
from app.core.exceptions import AppError
from app.schemas.chat import (
    EvidenceAssessment,
    EvidenceSentence,
    ReferenceDocument,
    StructuredAnswer,
)
from app.services.comparison_completeness_validator import CompletenessValidationError

logger = logging.getLogger(__name__)


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
    validation_degraded: bool = False
    validation_warnings: list[str] | None = None
    model_calls: int = 0


@dataclass(slots=True)
class SanitizedAnswer:
    answer: StructuredAnswer
    warnings: list[str]


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
        reference_document: ReferenceDocument | None = None,
        semantic_validator: Callable[[StructuredAnswer], None],
        semantic_sanitizer: (
            Callable[[StructuredAnswer, Exception], SanitizedAnswer | None] | None
        ) = None,
    ) -> GenerationResult:
        user_payload = {
            "question": question,
            "evidence_assessment": assessment.model_dump(),
            "evidence": [item.model_dump() for item in evidence],
            # 本次证据涉及的全部文档（去重）：同一主题多份规程（同名不同时间/
            # 地区、省标/市标）时，模型必须据此逐份分别输出，不得合并。
            "evidence_documents": list(
                dict.fromkeys(
                    item.document_name for item in evidence if item.document_name
                )
            ),
            # 可引用的编号白名单：证据只有 C1~C8 时，模型不得编造 C9/C10 等
            # 证据外的编号（首版生成就告知，避免 repair 阶段仍复现错误编号）。
            "allowed_citation_ids": sorted(
                {item.citation_id for item in evidence}
            ),
            "citation_documents": {
                item.citation_id: {
                    "document_name": item.document_name,
                    "version": item.version,
                }
                for item in evidence
                if item.document_name
            },
            "presentation_reference": (
                {
                    "name": reference_document.name,
                    "text": reference_document.text[:4_000],
                }
                if reference_document is not None
                else None
            ),
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
        model_calls = 1
        # 多轮修复：json_repair_attempts 为"可执行的修复轮数"（默认 1）。
        # 修复可能引入新问题（如补好一条 claim 却丢掉整个对比格子），
        # 因此校验失败后允许按配置再修；全部轮次耗尽仍失败才走 sanitize 降级。
        current_raw = raw
        last_answer: StructuredAnswer | None = None
        for attempt in range(self._settings.json_repair_attempts + 1):
            parsed: StructuredAnswer | None = None
            try:
                parsed = self._parse(current_raw)
                semantic_validator(parsed)
                return GenerationResult(
                    answer=parsed,
                    repaired=attempt > 0,
                    model_calls=model_calls,
                )
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                logger.warning(
                    "answer_validation_failed",
                    extra={
                        "validation_stage": "initial" if attempt == 0 else "repair",
                        "error_type": type(error).__name__,
                        "validation_error": str(error),
                    },
                )
                if parsed is not None:
                    last_answer = parsed
                if attempt >= self._settings.json_repair_attempts:
                    repair_error = error
                    break
                current_raw = await self._model.chat_completion(
                    messages=[
                        {"role": "system", "content": self._repair_prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "invalid_output": current_raw,
                                    "validation_error": str(error),
                                    "question": question,
                                    "allowed_citation_ids": sorted(
                                        {item.citation_id for item in evidence}
                                    ),
                                    "citation_documents": {
                                        item.citation_id: {
                                            "document_name": item.document_name,
                                            "version": item.version,
                                        }
                                        for item in evidence
                                        if item.document_name
                                    },
                                    "evidence": [item.model_dump() for item in evidence],
                                    "presentation_reference": (
                                        {
                                            "name": reference_document.name,
                                            "text": reference_document.text[:4_000],
                                        }
                                        if reference_document is not None
                                        else None
                                    ),
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    response_schema=StructuredAnswer.model_json_schema(),
                )
                model_calls += 1
        salvage_candidate = last_answer
        if semantic_sanitizer is not None and salvage_candidate is not None:
            sanitized = semantic_sanitizer(salvage_candidate, repair_error)
            if sanitized is not None:
                try:
                    semantic_validator(sanitized.answer)
                except CompletenessValidationError as completeness_error:
                    # 完整性缺失不是单条 claim 的问题，sanitize 无法裁剪解决：
                    # 保留已通过校验的答案，把缺失格子写入 missing_information
                    # 并标记降级，避免整个请求 502。
                    degraded = sanitized.answer.model_copy(
                        update={
                            "missing_information": [
                                *sanitized.answer.missing_information,
                                str(completeness_error),
                            ]
                        }
                    )
                    return GenerationResult(
                        answer=degraded,
                        repaired=True,
                        validation_degraded=True,
                        validation_warnings=[
                            *sanitized.warnings,
                            str(completeness_error),
                        ],
                        model_calls=model_calls,
                    )
                except ValueError as sanitize_error:
                    raise self._invalid_output_error(sanitize_error) from sanitize_error
                return GenerationResult(
                    answer=sanitized.answer,
                    repaired=True,
                    validation_degraded=True,
                    validation_warnings=sanitized.warnings,
                    model_calls=model_calls,
                )
        raise self._invalid_output_error(repair_error) from repair_error

    @staticmethod
    def _parse(raw: str) -> StructuredAnswer:
        content = raw.strip()
        if content.startswith("```") and content.endswith("```"):
            lines = content.splitlines()
            content = "\n".join(lines[1:-1]).strip()
        parsed = json.loads(content)
        # 模型可能输出重复 claim_id（StructuredAnswer 要求唯一）。claim_id 只是
        # 内部标识，重复无意义——自动唯一化（后续重复项加 _2/_3 后缀），避免
        # schema 校验失败后 first_answer 为 None 而无法走语义降级兜底。
        seen: dict[str, int] = {}
        for claim in parsed.get("claims", []):
            claim_id = claim.get("claim_id")
            if claim_id in seen:
                seen[claim_id] += 1
                claim["claim_id"] = f"{claim_id}_{seen[claim_id]}"
            else:
                seen[claim_id] = 1
        return StructuredAnswer.model_validate(parsed)

    @staticmethod
    def _invalid_output_error(error: Exception) -> AppError:
        if isinstance(error, json.JSONDecodeError):
            code = "ANSWER_JSON_PARSE_FAILED"
            message = "The local answer model did not return valid JSON."
        elif isinstance(error, ValidationError):
            code = "ANSWER_SCHEMA_VALIDATION_FAILED"
            message = "The local answer model response did not match the required schema."
        else:
            code = "ANSWER_SEMANTIC_VALIDATION_FAILED"
            message = "The generated answer did not pass evidence consistency checks."
        return AppError(
            code=code,
            message=message,
            status_code=502,
            details={
                "error_type": type(error).__name__,
                "validation_error": str(error),
            },
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
