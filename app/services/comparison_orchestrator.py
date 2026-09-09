from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.config import PROJECT_ROOT, GenerationSettings
from app.core.exceptions import AppError
from app.schemas.chat import EvidenceAssessment
from app.schemas.evidence import EvidenceRecord
from app.schemas.retrieval import CoverageCell, QueryPlan, SelectedChunk
from app.services.answer_generator import GenerationResult
from app.services.comparison_matrix_builder import ComparisonMatrixBuilder
from app.services.structured_planner_model import PROTECTED_TOKEN_RE

CITATION_RE = re.compile(r"\[C\d+\]")
NUMBER_CORE_RE = re.compile(r"\d+(?:\.\d+)*")


class SummaryPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str = Field(min_length=2, max_length=1000)


class SummaryModel(Protocol):
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str: ...


class ComparisonOrchestrator:
    """Build the verified matrix first; a single optional summary can never block it."""

    def __init__(self, settings: GenerationSettings) -> None:
        self._settings = settings
        self._builder = ComparisonMatrixBuilder()
        self._prompt = self._read_prompt(settings.matrix_summary_prompt_path)

    async def run(
        self,
        *,
        plan: QueryPlan,
        chunks: list[SelectedChunk],
        records: list[EvidenceRecord],
        assessment: EvidenceAssessment,
        coverage_matrix: list[CoverageCell],
        model: SummaryModel | None,
    ) -> GenerationResult | None:
        matrix = self._builder.build(
            plan=plan,
            chunks=chunks,
            records=records,
            assessment=assessment,
            coverage_matrix=coverage_matrix,
        )
        if matrix is None:
            return None
        if not self._settings.comparison_summary_enabled or model is None:
            covered = sum(cell.status == "covered" for cell in coverage_matrix)
            total = len(coverage_matrix)
            if matrix.answerability == "ANSWERABLE":
                overview = (
                    f"以下按{'、'.join(plan.aspects)}逐项区分"
                    f"{'与'.join(plan.subjects)}。"
                )
            else:
                overview = (
                    f"当前证据仅覆盖 {covered}/{total} 个对比项，无法完整比较"
                    f"{'与'.join(plan.subjects)}；以下仅列出已有依据，"
                    "其余项目明确标注为资料不足。"
                )
            answer = matrix.model_copy(update={"answer": f"{overview}\n\n{matrix.answer}"})
            return GenerationResult(answer=answer, repaired=False, model_calls=0)
        try:
            raw = await asyncio.wait_for(
                model.chat_completion(
                    messages=[
                        {"role": "system", "content": self._prompt},
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "matrix": matrix.answer,
                                    "missing_information": matrix.missing_information,
                                },
                                ensure_ascii=False,
                            ),
                        },
                    ],
                    response_schema=SummaryPayload.model_json_schema(),
                ),
                timeout=self._settings.comparison_summary_timeout_seconds,
            )
            summary = SummaryPayload.model_validate(self._parse(raw)).summary.strip()
            self._validate_summary(summary, matrix.answer)
            answer = matrix.model_copy(update={"answer": f"{summary}\n\n{matrix.answer}"})
            return GenerationResult(answer=answer, repaired=False, model_calls=1)
        except (AppError, TimeoutError, json.JSONDecodeError, ValidationError, ValueError):
            return GenerationResult(
                answer=matrix,
                repaired=False,
                validation_degraded=True,
                validation_warnings=["模型总结不可用，已直接展示确定性矩阵。"],
                model_calls=1,
            )

    @staticmethod
    def _parse(raw: str) -> Any:
        content = raw.strip()
        if content.startswith("```") and content.endswith("```"):
            content = "\n".join(content.splitlines()[1:-1]).strip()
        return json.loads(content)

    @staticmethod
    def _validate_summary(summary: str, matrix: str) -> None:
        # Citations and protected tokens (标准号/版本化条款号) must match exactly:
        # the summary may not introduce new ones.
        for extractor in (CITATION_RE.findall, PROTECTED_TOKEN_RE.findall):
            if set(extractor(summary)) - set(extractor(matrix)):
                raise ValueError("summary introduced unsupported protected content")
        # Numbers: tolerate reformatting (spacing/unit casing), but forbid any
        # numeric value that is not already present in the verified matrix.
        matrix_numbers = set(NUMBER_CORE_RE.findall(matrix))
        if set(NUMBER_CORE_RE.findall(summary)) - matrix_numbers:
            raise ValueError("summary introduced numbers not present in the matrix")

    @staticmethod
    def _read_prompt(relative_path: str) -> str:
        path = (PROJECT_ROOT / Path(relative_path)).resolve()
        return path.read_text(encoding="utf-8").strip()
