from __future__ import annotations

import pytest

from app.schemas.chat import CandidateClaim, EvidenceAssessment, StructuredAnswer
from app.schemas.retrieval import CoverageCell, QueryPlan, SubQuery
from app.services.comparison_completeness_validator import (
    ComparisonCompletenessValidator,
    CompletenessValidationError,
)


def _plan() -> QueryPlan:
    return QueryPlan(
        query_type="comparison",
        subjects=["长大桥梁无人机精细化巡检", "地质灾害倾斜摄影测量"],
        aspects=["设备选型", "像控布设"],
        subqueries=[
            SubQuery(id="q1", query="q1", subject="长大桥梁无人机精细化巡检", aspect="设备选型"),
            SubQuery(id="q2", query="q2", subject="地质灾害倾斜摄影测量", aspect="设备选型"),
            SubQuery(id="q3", query="q3", subject="长大桥梁无人机精细化巡检", aspect="像控布设"),
            SubQuery(id="q4", query="q4", subject="地质灾害倾斜摄影测量", aspect="像控布设"),
        ],
        synthesis_mode="matrix",
    )


def _matrix(cells: list[tuple[str, str, str, list[str]]]) -> list[CoverageCell]:
    """cells: (subquery_id, subject, aspect, citation_ids)。"""
    return [
        CoverageCell(
            subquery_id=subquery_id,
            subject=subject,
            aspect=aspect,
            status="covered",
            chunk_ids=[f"chunk-{cid}" for cid in citations],
            citation_ids=citations,
        )
        for subquery_id, subject, aspect, citations in cells
    ]


def _answer(claims: list[tuple[str, list[str]]]) -> StructuredAnswer:
    return StructuredAnswer(
        answerability="ANSWERABLE",
        answer="答案",
        claims=[
            CandidateClaim(claim_id=claim_id, claim=text, citation_ids=citations)
            for claim_id, text, citations in claims
        ],
    )


def test_completeness_passes_when_all_covered_cells_present() -> None:
    plan = _plan()
    matrix = _matrix(
        [
            ("q1", "长大桥梁无人机精细化巡检", "设备选型", ["C1"]),
            ("q2", "地质灾害倾斜摄影测量", "设备选型", ["C2"]),
            ("q3", "长大桥梁无人机精细化巡检", "像控布设", []),
            ("q4", "地质灾害倾斜摄影测量", "像控布设", ["C4"]),
        ]
    )
    assessment = EvidenceAssessment(
        status="ANSWERABLE",
        covered_requirements=[
            "长大桥梁无人机精细化巡检：设备选型",
            "地质灾害倾斜摄影测量：设备选型",
            "长大桥梁无人机精细化巡检：像控布设",
            "地质灾害倾斜摄影测量：像控布设",
        ],
    )
    answer = _answer(
        [
            ("1", "长大桥梁无人机精细化巡检设备选型要求续航≥30min", ["C1"]),
            ("2", "地质灾害倾斜摄影测量设备选型未明确具体参数", []),
            ("3", "长大桥梁无人机精细化巡检像控布设未明确说明", []),
            ("4", "地质灾害倾斜摄影测量像控布设要求满足空三原则", ["C4"]),
        ]
    )

    ComparisonCompletenessValidator().validate(
        answer,
        plan=plan,
        coverage_matrix=matrix,
        assessment=assessment,
    )


def test_completeness_rejects_missing_covered_cell() -> None:
    plan = _plan()
    matrix = _matrix(
        [
            ("q1", "长大桥梁无人机精细化巡检", "设备选型", ["C1"]),
            ("q2", "地质灾害倾斜摄影测量", "设备选型", []),
            ("q3", "长大桥梁无人机精细化巡检", "像控布设", []),
            ("q4", "地质灾害倾斜摄影测量", "像控布设", ["C4"]),
        ]
    )
    assessment = EvidenceAssessment(
        status="ANSWERABLE",
        covered_requirements=[
            "长大桥梁无人机精细化巡检：设备选型",
            "地质灾害倾斜摄影测量：设备选型",
            "长大桥梁无人机精细化巡检：像控布设",
            "地质灾害倾斜摄影测量：像控布设",
        ],
    )
    # 缺少"地质灾害倾斜摄影测量：像控布设"格子（无 claim 点名，也未引用 C4）
    answer = _answer(
        [
            ("1", "长大桥梁无人机精细化巡检设备选型要求续航≥30min", ["C1"]),
            ("2", "地质灾害倾斜摄影测量设备选型未明确具体参数", []),
            ("3", "长大桥梁无人机精细化巡检像控布设未明确说明", []),
        ]
    )

    with pytest.raises(CompletenessValidationError) as exc_info:
        ComparisonCompletenessValidator().validate(
            answer,
            plan=plan,
            coverage_matrix=matrix,
            assessment=assessment,
        )
    assert "地质灾害倾斜摄影测量：像控布设" in str(exc_info.value)


def test_completeness_skips_missing_cells_not_covered() -> None:
    plan = _plan()
    matrix = _matrix(
        [
            ("q1", "长大桥梁无人机精细化巡检", "设备选型", ["C1"]),
            ("q2", "地质灾害倾斜摄影测量", "设备选型", []),
        ]
    )
    assessment = EvidenceAssessment(
        status="PARTIALLY_ANSWERABLE",
        # 只有 2 个格子被覆盖（q3/q4 无证据）
        covered_requirements=[
            "长大桥梁无人机精细化巡检：设备选型",
            "地质灾害倾斜摄影测量：设备选型",
        ],
    )
    answer = _answer(
        [
            ("1", "长大桥梁无人机精细化巡检设备选型要求续航≥30min", ["C1"]),
            ("2", "地质灾害倾斜摄影测量设备选型未明确具体参数", []),
        ]
    )

    ComparisonCompletenessValidator().validate(
        answer,
        plan=plan,
        coverage_matrix=matrix,
        assessment=assessment,
    )


def test_completeness_skips_non_comparison_or_no_coverage() -> None:
    plan = QueryPlan(
        query_type="procedure",
        aspects=["流程"],
        subqueries=[SubQuery(id="q1", query="q1", aspect="流程")],
        synthesis_mode="sequence",
    )
    assessment = EvidenceAssessment(
        status="ANSWERABLE",
        covered_requirements=["流程"],
    )
    answer = _answer([("1", "流程内容", ["C1"])])

    ComparisonCompletenessValidator().validate(
        answer,
        plan=plan,
        coverage_matrix=[],
        assessment=assessment,
    )

    empty = EvidenceAssessment(status="ANSWERABLE", covered_requirements=[])
    ComparisonCompletenessValidator().validate(
        answer,
        plan=_plan(),
        coverage_matrix=[],
        assessment=empty,
    )
