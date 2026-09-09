from __future__ import annotations

from collections import defaultdict

from app.core.config import GenerationSettings
from app.schemas.chat import EvidenceAssessment, EvidenceConflict
from app.schemas.retrieval import CoverageCell, NormalizedQuery, SelectedChunk
from app.services.evidence_text import (
    has_parameter_value,
    is_parametric_text,
    measured_values,
    overlap_score,
    split_requirements,
    split_sentences,
)
from app.services.regulation_context import regulation_context


class EvidenceJudge:
    """Apply a deterministic pre-generation evidence gate."""

    def __init__(self, settings: GenerationSettings) -> None:
        self._settings = settings

    def assess(
        self,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
        coverage_matrix: list[CoverageCell] | None = None,
    ) -> EvidenceAssessment:
        context = regulation_context(query, chunks)
        if (
            query.article_ids and context["evidence_sections"]
            and len(context["unverified_sections"]) == len(query.article_ids)
        ):
            return EvidenceAssessment(
                status="UNANSWERABLE",
                missing_requirements=[
                    "本次证据未核实所问条号，需核对条号与主题是否一致；"
                    "不能据此认定条款不存在。"
                ],
            )
        if coverage_matrix:
            return self._assess_coverage_matrix(query, chunks, coverage_matrix)
        requirements = split_requirements(query.normalized_query)
        if not chunks:
            return EvidenceAssessment(
                status="UNANSWERABLE",
                missing_requirements=requirements,
            )

        covered: list[str] = []
        missing: list[str] = []
        relevant_by_requirement: dict[str, list[tuple[str, str]]] = {}
        for requirement in requirements:
            relevant = self._relevant_sentences(requirement, chunks)
            relevant_by_requirement[requirement] = relevant
            if self._is_covered(query, requirement, relevant):
                covered.append(requirement)
            else:
                missing.append(requirement)

        conflicts = (
            []
            if query.query_type == "comparison"
            else self._detect_measurement_conflicts(relevant_by_requirement)
        )
        if conflicts:
            status = "CONFLICTED"
        elif not covered:
            status = "UNANSWERABLE"
        elif missing:
            status = (
                "PARTIALLY_ANSWERABLE"
                if self._settings.allow_partial_answer
                else "UNANSWERABLE"
            )
        else:
            status = "ANSWERABLE"
        return EvidenceAssessment(
            status=status,
            covered_requirements=covered,
            missing_requirements=missing,
            conflicting_citations=conflicts,
        )

    def _assess_coverage_matrix(
        self,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
        coverage_matrix: list[CoverageCell],
    ) -> EvidenceAssessment:
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        covered: list[str] = []
        missing: list[str] = []
        for cell in coverage_matrix:
            label = self._coverage_label(cell)
            if cell.status != "covered":
                # 四态中仅 covered 视为需求被覆盖；not_specified /
                # low_confidence / no_document 都进入 missing（各自的 reason
                # 说明缺失原因，矩阵构建器会据此写"未明确说明"或保留疑似内容）。
                missing.append(label)
                continue
            if self._cell_has_value_evidence(cell, chunk_by_id):
                covered.append(label)
            else:
                missing.append(label)
        if not covered:
            status = "UNANSWERABLE"
        elif missing:
            status = (
                "PARTIALLY_ANSWERABLE"
                if self._settings.allow_partial_answer
                else "UNANSWERABLE"
            )
        else:
            status = "ANSWERABLE"

        relevant_by_requirement = {
            label: self._relevant_sentences(label, chunks)
            for label in covered
        }
        conflicts = (
            []
            if query.query_type == "comparison"
            else self._detect_measurement_conflicts(relevant_by_requirement)
        )
        if conflicts:
            status = "CONFLICTED"
        return EvidenceAssessment(
            status=status,
            covered_requirements=covered,
            missing_requirements=missing,
            conflicting_citations=conflicts,
        )

    @classmethod
    def _cell_has_value_evidence(
        cls,
        cell: CoverageCell,
        chunk_by_id: dict[str, SelectedChunk],
    ) -> bool:
        """A parameter-aspect cell is covered only when its chunks carry a
        concrete numeric value; non-parameter aspects keep the legacy rule."""
        if not cls._is_parametric_requirement(cell.aspect):
            return True
        return any(
            has_parameter_value(chunk_by_id[cid].text)
            for cid in cell.chunk_ids
            if cid in chunk_by_id
        )

    @staticmethod
    def _is_parametric_requirement(requirement: str) -> bool:
        return is_parametric_text(requirement)

    @staticmethod
    def _coverage_label(cell: CoverageCell) -> str:
        if cell.subject:
            return f"{cell.subject}：{cell.aspect}"
        return cell.aspect

    def _relevant_sentences(
        self,
        requirement: str,
        chunks: list[SelectedChunk],
    ) -> list[tuple[str, str]]:
        relevant: list[tuple[str, str]] = []
        for chunk in chunks:
            for sentence in split_sentences(chunk.text):
                if overlap_score(requirement, sentence) >= self._settings.min_evidence_overlap:
                    relevant.append((chunk.citation_id, sentence))
        return relevant

    def _is_covered(
        self,
        query: NormalizedQuery,
        requirement: str,
        relevant: list[tuple[str, str]],
    ) -> bool:
        if not relevant:
            return query.query_type == "summary"
        required_exact = [token for token in query.exact_tokens if token in requirement]
        if required_exact:
            combined = "\n".join(sentence for _, sentence in relevant)
            return all(token in combined for token in required_exact)
        if self._is_parametric_requirement(requirement):
            return any(has_parameter_value(sentence) for _, sentence in relevant)
        return True

    @staticmethod
    def _detect_measurement_conflicts(
        relevant_by_requirement: dict[str, list[tuple[str, str]]],
    ) -> list[EvidenceConflict]:
        conflicts: list[EvidenceConflict] = []
        for requirement, evidence in relevant_by_requirement.items():
            by_unit: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
            for citation_id, sentence in evidence:
                for unit, values in measured_values(sentence).items():
                    for value in values:
                        by_unit[unit][str(value.normalize())].add(citation_id)
            for unit, value_citations in by_unit.items():
                if len(value_citations) < 2:
                    continue
                citation_ids = sorted(
                    {citation for citations in value_citations.values() for citation in citations}
                )
                if len(citation_ids) < 2:
                    continue
                conflicts.append(
                    EvidenceConflict(
                        requirement=requirement,
                        citation_ids=citation_ids,
                        details=f"conflicting {unit} values: {', '.join(value_citations)}",
                    )
                )
        return conflicts
