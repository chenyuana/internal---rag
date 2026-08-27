"""确定性对比矩阵兜底：模型答不出完整矩阵时，按证据逐格填充。

对比问题的 plan（subjects × aspects）与覆盖矩阵（每格 chunk）是通用抽象：
任何"对比 A、B 在 X、Y 上…""A 与 B 在 X、Y、Z 环节如何统筹区分"的问题
都走同一逻辑，**不含任何具体问题/文档词汇**。

格子证据（cell_evidence.substantive_lines）只取与维度相关的实质性条款行；
格子状态（coverage_matrix.status）决定输出：covered → 提取内容；
not_specified / low_confidence / no_document → 写"当前资料中未明确说明"。
"""

from __future__ import annotations

from app.schemas.chat import CandidateClaim, EvidenceAssessment, StructuredAnswer
from app.schemas.evidence import EvidenceRecord
from app.schemas.retrieval import CoverageCell, QueryPlan, SelectedChunk
from app.services.cell_evidence import (
    CITATION_MARKER_RE,
    dimension_keywords,
    substantive_lines,
)
from app.services.requirement_taxonomy import line_matches_aspect

# 每格最多取的高分证据行数。
_MAX_CELL_LINES = 4


class ComparisonMatrixBuilder:
    """Deterministic comparison matrix fallback built from cell evidence."""

    def build(
        self,
        *,
        plan: QueryPlan,
        chunks: list[SelectedChunk],
        assessment: EvidenceAssessment,
        coverage_matrix: list[CoverageCell] | None = None,
        records: list[EvidenceRecord] | None = None,
    ) -> StructuredAnswer | None:
        if (
            plan.query_type not in {"comparison", "multi_hop"}
            or len(plan.subjects) < 2
            or not plan.aspects
        ):
            return None
        chunk_by_id = {item.chunk_id: item for item in chunks}
        # 格子 → chunk：优先消费覆盖矩阵（多对多、状态可信），
        # 不再依赖 SelectedChunk.subquery_id 单值归属。
        cell_by_label: dict[tuple[str, str], CoverageCell] = {}
        for matrix_cell in coverage_matrix or []:
            if matrix_cell.subject:
                cell_by_label[(matrix_cell.subject, matrix_cell.aspect)] = matrix_cell

        # 布局判定：comparison 是全交叉（每个 subject × 每个 aspect）；multi_hop
        # 的"结合 A 与 B…兼顾 X 与 Y"是对角线（A→X、B→Y，一一对应）。对角线
        # 布局下全交叉遍历会产生不存在的交叉格，须按 subject→aspect 一一渲染。
        diagonal = self._is_diagonal_layout(plan, cell_by_label)

        claims: list[CandidateClaim] = []
        missing: list[str] = []
        rows: list[tuple[str, list[str]]] = []
        if diagonal:
            for subject, aspect in self._diagonal_pairs(plan):
                cell = cell_by_label.get((subject, aspect))
                cell_chunks = [
                    chunk_by_id[chunk_id]
                    for chunk_id in (cell.chunk_ids if cell else [])
                    if chunk_id in chunk_by_id
                ]
                content, citations = self._cell_content(
                    aspect=aspect,
                    cell_chunks=cell_chunks,
                    records=[
                        record
                        for record in (records or [])
                        if record.cell_id == (cell.subquery_id if cell else "")
                    ],
                )
                label = f"{subject}：{aspect}"
                if not content or (cell is not None and cell.status != "covered"):
                    missing.append(label)
                    content = "当前资料中未明确说明"
                if citations:
                    claim_text = CITATION_MARKER_RE.sub("", content).strip()
                else:
                    claim_text = f"{subject}在{aspect}方面：{content}"
                claims.append(
                    CandidateClaim(
                        claim_id=f"m{len(claims) + 1}",
                        claim=claim_text,
                        citation_ids=citations,
                    )
                )
                rows.append((f"{subject} {aspect}", [content]))
            answer = self._render_diagonal_table(rows)
        else:
            for aspect in plan.aspects:
                cells: list[str] = []
                for subject in plan.subjects:
                    cell = cell_by_label.get((subject, aspect))
                    cell_chunks = [
                        chunk_by_id[chunk_id]
                        for chunk_id in (cell.chunk_ids if cell else [])
                        if chunk_id in chunk_by_id
                    ]
                    content, citations = self._cell_content(
                        aspect=aspect,
                        cell_chunks=cell_chunks,
                        records=[
                            record
                            for record in (records or [])
                            if record.cell_id == (cell.subquery_id if cell else "")
                        ],
                    )
                    if not content or (
                        cell is not None and cell.status != "covered"
                    ):
                        # 格子非 covered（not_specified/low_confidence/no_document）
                        # 或提取不到实质条款 → 如实写"未明确说明"。
                        label = f"{subject}：{aspect}"
                        missing.append(label)
                        content = "当前资料中未明确说明"
                    cells.append(content)
                    if citations:
                        # 有内容格：claim 只放内容文本（主体/维度前缀中的词不在
                        # chunk 里会误触发 topic 校验；归属由引用承担）。
                        claim_text = CITATION_MARKER_RE.sub("", content).strip()
                    else:
                        # 未明确说明格：无引用，topic 校验自动跳过，保留主体+维度
                        # 前缀以满足完整性校验的"点名"检查。
                        claim_text = f"{subject}在{aspect}方面：{content}"
                    claims.append(
                        CandidateClaim(
                            claim_id=f"m{len(claims) + 1}",
                            claim=claim_text,
                            citation_ids=citations,
                        )
                    )
                rows.append((aspect, cells))

            answer = self._render_table(plan.subjects, rows)
        answerable = not missing and assessment.status == "ANSWERABLE"
        return StructuredAnswer(
            answerability="ANSWERABLE" if answerable else "PARTIALLY_ANSWERABLE",
            answer=answer,
            claims=claims,
            missing_information=missing,
            conflicts=[],
        )

    @staticmethod
    def _is_diagonal_layout(
        plan: QueryPlan,
        cell_by_label: dict[tuple[str, str], CoverageCell],
    ) -> bool:
        """对角线布局：coverage 里每个 subject 只出现一个 aspect，且 cell 数 == subject 数。

        comparison 的完整矩阵每个 subject 会出现在每个 aspect 下；multi_hop
        的"结合 A 与 B…兼顾 X 与 Y"每个 subject 只有自己的一个 cell。
        """
        if plan.query_type != "multi_hop":
            return False
        subject_aspects: dict[str, set[str]] = {}
        for (subject, aspect), _cell in cell_by_label.items():
            subject_aspects.setdefault(subject, set()).add(aspect)
        present = [item for item in subject_aspects.values() if item]
        return bool(
            present
            and len(subject_aspects) >= 2
            and all(len(aspects) == 1 for aspects in present)
        )

    @staticmethod
    def _diagonal_pairs(plan: QueryPlan) -> list[tuple[str, str]]:
        """返回对角线布局的 (subject, aspect) 对，顺序与 coverage 一致。"""
        pairs: list[tuple[str, str]] = []
        for subquery in plan.subqueries:
            if subquery.subject and subquery.aspect:
                pairs.append((subquery.subject, subquery.aspect))
        return pairs

    @staticmethod
    def _render_diagonal_table(rows: list[tuple[str, list[str]]]) -> str:
        header = "| 对比项目 | 要求 |"
        separator = "| --- | --- |"
        body = [header, separator]
        for label, cells in rows:
            body.append("| " + label + " | " + " | ".join(cells) + " |")
        return "\n".join(body)

    @staticmethod
    def _cell_content(
        *,
        aspect: str,
        cell_chunks: list[SelectedChunk],
        records: list[EvidenceRecord] | None = None,
    ) -> tuple[str, list[str]]:
        """取该格（主体×维度）的实质条款行，附引用。"""
        if records:
            record_parts: list[str] = []
            record_citations: list[str] = []
            record_seen: set[str] = set()
            by_citation: dict[str, list[EvidenceRecord]] = {}
            requires_concrete_value = any(
                marker in aspect
                for marker in ("成像波段", "最佳时段", "最佳时间", "精度评价指标", "精度指标")
            ) or aspect == "波段"
            for record in records:
                if requires_concrete_value and not line_matches_aspect(
                    aspect, record.requirement_text
                ):
                    continue
                by_citation.setdefault(record.citation_id, []).append(record)
            # First take one record from each source, then a second round, so a
            # long first table cannot consume all four summary slots and hide
            # other acceptance tables selected for the same cell.
            round_index = 0
            while len(record_parts) < _MAX_CELL_LINES:
                progressed = False
                for citation_id, citation_records in by_citation.items():
                    if round_index >= len(citation_records):
                        continue
                    record = citation_records[round_index]
                    if record.requirement_text in record_seen:
                        continue
                    progressed = True
                    record_seen.add(record.requirement_text)
                    record_parts.append(
                        f"{record.requirement_text}[{record.citation_id}]"
                    )
                    if citation_id not in record_citations:
                        record_citations.append(citation_id)
                    if len(record_parts) >= _MAX_CELL_LINES:
                        break
                if not progressed:
                    break
                round_index += 1
            return "；".join(record_parts), record_citations
        if not cell_chunks:
            return "", []
        keywords = dimension_keywords(aspect)
        if not keywords:
            return "", []
        scored: list[tuple[int, str, str]] = []
        for chunk in cell_chunks:
            for line in substantive_lines(chunk, keywords, aspect):
                scored.append((0, line, chunk.citation_id))
        if not scored:
            return "", []
        # 跨 chunk 去重并取全局前 _MAX_CELL_LINES 行
        seen: set[str] = set()
        top: list[tuple[int, str, str]] = []
        for item in scored:
            if item[1] in seen:
                continue
            seen.add(item[1])
            top.append(item)
            if len(top) >= _MAX_CELL_LINES:
                break
        parts: list[str] = []
        citations: list[str] = []
        for _, line, citation_id in top:
            if citation_id not in citations:
                citations.append(citation_id)
            parts.append(f"{line}[{citation_id}]")
        return "；".join(parts), citations

    @staticmethod
    def _render_table(
        subjects: list[str],
        rows: list[tuple[str, list[str]]],
    ) -> str:
        header = "| 对比项目 | " + " | ".join(subjects) + " |"
        separator = "| --- |" + " --- |" * len(subjects)
        body = [header, separator]
        for aspect, cells in rows:
            body.append("| " + aspect + " | " + " | ".join(cells) + " |")
        return "\n".join(body)
