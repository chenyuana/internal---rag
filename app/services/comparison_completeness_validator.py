"""对比矩阵完整性校验：已覆盖的每个（主体×维度）格子必须在答案中出现。

对比矩阵模式下，模型 repair 会整体重写答案，可能漏掉某个维度行（实测把
"成果验收"整行丢掉的场景）。对 comparison 矩阵，每个被判定为 covered 的
格子必须有 claim 覆盖它：要么该 claim 文本点名主体核心词与维度词，要么
引用了该格子的 chunk。缺失 → 抛 CompletenessValidationError（触发 repair，
repair 提示词会带上缺失格子与其证据编号）；repair 后仍缺失时由生成端优雅
降级（把缺失格子写入 missing_information），不让整个请求 502。

注意：这是**答案级**校验，挂在 validate()（整答）路径，不进 per-claim
校验（sanitize 逐条裁剪时单条 claim 不可能覆盖全部格子，会被误伤）。
"""

from __future__ import annotations

from app.schemas.chat import CandidateClaim, EvidenceAssessment, StructuredAnswer
from app.schemas.retrieval import CoverageCell, QueryPlan
from app.services.query_scope import subject_core


class CompletenessValidationError(ValueError):
    """对比答案遗漏了已覆盖的矩阵格子。"""


class ComparisonCompletenessValidator:
    """Reject comparison answers that omit a covered subject × aspect cell."""

    def validate(
        self,
        answer: StructuredAnswer,
        *,
        plan: QueryPlan | None,
        coverage_matrix: list[CoverageCell] | None,
        assessment: EvidenceAssessment,
    ) -> None:
        if (
            plan is None
            or plan.query_type not in {"comparison", "multi_hop"}
            or len(plan.subjects) < 2
        ):
            return
        covered = set(assessment.covered_requirements or [])
        if not covered:
            return
        subqueries_by_label: dict[str, list[str]] = {}
        for subquery in plan.subqueries:
            if subquery.subject:
                label = f"{subquery.subject}：{subquery.aspect}"
                subqueries_by_label.setdefault(label, []).append(subquery.id)
        # 多对多：citation → 支撑的格子（子查询 id 集合），来自覆盖矩阵。
        chunk_cells: dict[str, set[str]] = {}
        for cell in coverage_matrix or []:
            for citation_id in cell.citation_ids:
                chunk_cells.setdefault(citation_id, set()).add(cell.subquery_id)
        for label in covered:
            if self._cell_present(
                label,
                answer.claims,
                subqueries_by_label,
                chunk_cells,
            ):
                continue
            cell_subqueries = subqueries_by_label.get(label, [])
            cell_citations = sorted(
                citation
                for citation, cells in chunk_cells.items()
                if cells & set(cell_subqueries)
            )
            raise CompletenessValidationError(
                f"对比答案缺少格子：{label}"
                + (
                    f"（该格有证据 [{', '.join(cell_citations)}]，"
                    "必须输出该格内容，无对应内容则写'当前资料中未明确说明'）"
                    if cell_citations
                    else "（该格有证据，必须输出其内容或写'当前资料中未明确说明'）"
                )
            )

    @staticmethod
    def _cell_present(
        label: str,
        claims: list[CandidateClaim],
        subqueries_by_label: dict[str, list[str]],
        chunk_cells: dict[str, set[str]],
    ) -> bool:
        subject, _, aspect = label.partition("：")
        subject_keywords = {subject, subject_core(subject)}
        subject_keywords.discard("")
        aspect_fragments = {
            aspect[index : index + length]
            for length in range(2, min(4, len(aspect)) + 1)
            for index in range(0, len(aspect) - length + 1)
        }
        cell_subqueries = set(subqueries_by_label.get(label, []))
        for claim in claims:
            if any(keyword in claim.claim for keyword in subject_keywords) and any(
                fragment in claim.claim for fragment in aspect_fragments
            ):
                return True
            if any(
                (chunk_cells.get(citation_id, set()) & cell_subqueries)
                for citation_id in claim.citation_ids
            ):
                return True
        return False
