from __future__ import annotations

from collections import defaultdict

from app.core.config import GenerationSettings
from app.schemas.chat import EvidenceAssessment, EvidenceConflict
from app.schemas.retrieval import CoverageCell, NormalizedQuery, SelectedChunk
from app.services.evidence_text import (
    measured_values,
    overlap_score,
    split_requirements,
    split_sentences,
)


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
        covered = [
            self._coverage_label(cell)
            for cell in coverage_matrix
            if cell.status == "covered"
        ]
        missing = [
            self._coverage_label(cell)
            for cell in coverage_matrix
            if cell.status == "missing"
        ]
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
