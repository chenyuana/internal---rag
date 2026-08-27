from __future__ import annotations

from typing import Any

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.chat import EvidenceAssessment
from app.schemas.retrieval import ChunkMetadata, CoverageCell, QueryPlan, SelectedChunk, SubQuery
from app.services.comparison_orchestrator import ComparisonOrchestrator


class SummaryReply:
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    async def chat_completion(
        self, *, messages: list[dict[str, str]], response_schema: dict[str, Any]
    ) -> str:
        assert messages and response_schema
        self.calls += 1
        return self.reply


class SummaryFailure:
    async def chat_completion(
        self, *, messages: list[dict[str, str]], response_schema: dict[str, Any]
    ) -> str:
        del messages, response_schema
        raise AppError(code="MODEL_DOWN", message="down", status_code=503)


def _inputs() -> tuple[QueryPlan, list[SelectedChunk], list[CoverageCell]]:
    plan = QueryPlan(
        query_type="comparison",
        subjects=["甲设备", "乙设备"],
        aspects=["设备选型"],
        subqueries=[
            SubQuery(id="q1", query="甲设备 设备选型", subject="甲设备", aspect="设备选型"),
            SubQuery(id="q2", query="乙设备 设备选型", subject="乙设备", aspect="设备选型"),
        ],
        synthesis_mode="matrix",
    )
    chunks = [
        SelectedChunk(
            citation_id=f"C{index}",
            chunk_id=f"c{index}",
            document_id=f"d{index}",
            dataset_id="kb-1",
            text=f"设备应配置相机{index}。",
            metadata=ChunkMetadata(document_name=f"{subject}规程"),
            hybrid_score=0.8,
            vector_score=0.7,
            keyword_score=0.9,
        )
        for index, subject in enumerate(plan.subjects, start=1)
    ]
    coverage = [
        CoverageCell(
            subquery_id=f"q{index}",
            subject=subject,
            aspect="设备选型",
            status="covered",
            chunk_ids=[f"c{index}"],
            citation_ids=[f"C{index}"],
        )
        for index, subject in enumerate(plan.subjects, start=1)
    ]
    return plan, chunks, coverage


async def test_invalid_summary_falls_back_to_matrix_after_one_call(
    settings: Settings,
) -> None:
    plan, chunks, coverage = _inputs()
    model = SummaryReply("not-json")
    configured = settings.generation.model_copy(update={"comparison_summary_enabled": True})
    result = await ComparisonOrchestrator(configured).run(
        plan=plan,
        chunks=chunks,
        records=[],
        assessment=EvidenceAssessment(status="ANSWERABLE"),
        coverage_matrix=coverage,
        model=model,
    )
    assert result is not None
    assert "| 对比项目 | 甲设备 | 乙设备 |" in result.answer.answer
    assert result.validation_degraded is True
    assert result.model_calls == 1
    assert model.calls == 1


async def test_summary_network_failure_still_returns_matrix(settings: Settings) -> None:
    plan, chunks, coverage = _inputs()
    configured = settings.generation.model_copy(update={"comparison_summary_enabled": True})
    result = await ComparisonOrchestrator(configured).run(
        plan=plan,
        chunks=chunks,
        records=[],
        assessment=EvidenceAssessment(status="ANSWERABLE"),
        coverage_matrix=coverage,
        model=SummaryFailure(),
    )
    assert result is not None
    assert result.validation_degraded is True
    assert result.model_calls == 1
    assert "| 对比项目 |" in result.answer.answer


async def test_partial_matrix_intro_reports_coverage_instead_of_complete_comparison(
    settings: Settings,
) -> None:
    plan, chunks, coverage = _inputs()
    coverage[1] = coverage[1].model_copy(update={"status": "not_specified", "chunk_ids": []})
    configured = settings.generation.model_copy(update={"comparison_summary_enabled": False})

    result = await ComparisonOrchestrator(configured).run(
        plan=plan,
        chunks=chunks,
        records=[],
        assessment=EvidenceAssessment(status="PARTIALLY_ANSWERABLE"),
        coverage_matrix=coverage,
        model=None,
    )

    assert result is not None
    assert result.answer.answer.startswith("当前证据仅覆盖 1/2 个对比项，无法完整比较")
