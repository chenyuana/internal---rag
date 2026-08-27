from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from app.core.config import PROJECT_ROOT, TranslationSettings
from app.core.exceptions import AppError
from app.schemas.planning import PlannedCell, TranslationDecision
from app.schemas.retrieval import NormalizedQuery
from app.services.evidence_text import lexical_units
from app.services.requirement_taxonomy import ASPECT_RETRIEVAL_TERMS
from app.services.structured_planner_model import PROTECTED_TOKEN_RE

NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)*(?:\s*[A-Za-z%℃]+)?")


class TranslationModel(Protocol):
    async def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        response_schema: dict[str, Any],
    ) -> str: ...


@dataclass(slots=True)
class TranslationOutcome:
    decision: TranslationDecision
    model_calls: int
    latency_ms: float
    fallback_reason: str | None = None


class QueryTranslator:
    def __init__(
        self,
        settings: TranslationSettings,
        model: TranslationModel | None,
    ) -> None:
        self._settings = settings
        self._model = model
        self._prompt = self._read_prompt(settings.prompt_path)

    async def translate(
        self,
        *,
        query: NormalizedQuery,
        cell: PlannedCell,
        section_titles: list[str],
    ) -> TranslationOutcome:
        started = time.perf_counter()
        if not self._settings.enabled or self._settings.max_variants_per_cell == 0:
            return self._disabled(started, "translation_disabled")
        if query.article_ids or query.exact_tokens:
            return self._disabled(started, "exact_query_translation_forbidden")
        if not self._settings.model_enabled or self._model is None:
            terms = ASPECT_RETRIEVAL_TERMS.get(cell.aspect)
            if not terms:
                return self._disabled(started, "translation_model_unavailable")
            translated = f"{cell.subject or ''} {cell.aspect} {terms}".strip()
            return TranslationOutcome(
                decision=TranslationDecision(
                    should_translate=True,
                    translated_query=translated,
                    reason="迁移期受控领域词表补充查询",
                ),
                model_calls=0,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                fallback_reason="taxonomy_fallback",
            )
        payload = {
            "original_question": query.original_query,
            "subject": cell.subject,
            "aspect": cell.aspect,
            "original_query": cell.original_query,
            "initial_section_titles": section_titles[:5],
        }
        try:
            raw = await asyncio.wait_for(
                self._model.chat_completion(
                    messages=[
                        {"role": "system", "content": self._prompt},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                    ],
                    response_schema=TranslationDecision.model_json_schema(),
                ),
                timeout=self._settings.timeout_seconds,
            )
            decision = TranslationDecision.model_validate(self._parse_json(raw))
            self._validate(query, cell, decision)
            return TranslationOutcome(
                decision=decision,
                model_calls=1,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except (
            AppError,
            TimeoutError,
            json.JSONDecodeError,
            ValidationError,
            ValueError,
        ) as exc:
            return TranslationOutcome(
                decision=TranslationDecision(
                    should_translate=False,
                    translated_query=None,
                    reason="转换输出不可用，保留原查询结果",
                ),
                model_calls=1,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                fallback_reason=type(exc).__name__,
            )

    @staticmethod
    def _validate(
        query: NormalizedQuery,
        cell: PlannedCell,
        decision: TranslationDecision,
    ) -> None:
        if not decision.should_translate:
            return
        translated = decision.translated_query or ""
        if cell.subject and cell.subject not in translated:
            raise ValueError("translated query changed or removed the subject")
        if not (lexical_units(cell.aspect) & lexical_units(translated)):
            raise ValueError("translated query changed the aspect")
        allowed = " ".join(
            [query.original_query, cell.original_query, cell.subject or "", cell.aspect]
        )
        invented_tokens = set(PROTECTED_TOKEN_RE.findall(translated)) - set(
            PROTECTED_TOKEN_RE.findall(allowed)
        )
        invented_numbers = set(NUMBER_RE.findall(translated)) - set(NUMBER_RE.findall(allowed))
        if invented_tokens or invented_numbers:
            raise ValueError("translated query invented protected tokens or numbers")

    @staticmethod
    def _parse_json(raw: str) -> Any:
        content = raw.strip()
        if content.startswith("```") and content.endswith("```"):
            content = "\n".join(content.splitlines()[1:-1]).strip()
        return json.loads(content)

    @staticmethod
    def _read_prompt(relative_path: str) -> str:
        path = (PROJECT_ROOT / Path(relative_path)).resolve()
        return path.read_text(encoding="utf-8").strip()

    @staticmethod
    def _disabled(started: float, reason: str) -> TranslationOutcome:
        return TranslationOutcome(
            decision=TranslationDecision(
                should_translate=False,
                translated_query=None,
                reason=reason,
            ),
            model_calls=0,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            fallback_reason=reason,
        )
