from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.schemas.retrieval import CoverageCell, QueryPlan, RetrievedChunk, SubQuery
from app.services.cell_evidence import (
    cell_has_substantive_evidence,
    dimension_keywords,
    substantive_lines,
)
from app.services.evidence_text import has_parameter_value, is_parametric_text

# Markdown 表格行。表格通常是对"见表X"要求的具体数值化（如"地面分辨率不
# 低于表1"），其数据行与查询的词汇重叠率低，需在选择时显式识别，否则会被
# 无关正文块挤出证据名额。
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


@dataclass(slots=True)
class CoverageSelection:
    chunks: list[RetrievedChunk]
    matrix: list[CoverageCell]
    # chunk_id -> 来源子查询 id（第二轮按主体文档加分用）。
    chunk_subqueries: dict[str, str] = field(default_factory=dict)
    # chunk_id -> 支撑的格子（子查询 id 集合）：同一条款可能同时支撑多个维度
    #（如"设备选型"与"成果验收"），下游（矩阵构建器/完整性/主体归因）应消费
    # coverage_matrix 或此多对多映射，而非 SelectedChunk.subquery_id 单值。
    chunk_cells: dict[str, set[str]] = field(default_factory=dict)


class CoverageSelector:
    """Select evidence by required-cell coverage before global relevance."""

    def select(
        self,
        plan: QueryPlan,
        results: dict[str, list[RetrievedChunk]],
        *,
        limit: int,
        missing_subjects: set[str] | None = None,
    ) -> CoverageSelection:
        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()
        chunk_subqueries: dict[str, str] = {}
        chunk_cells: dict[str, set[str]] = {}

        # Round one reserves one slot for each required retrieval cell.  A
        # retrieval rank is only a candidate rank: for known aspects, prefer
        # the first chunk that actually contains substantive evidence for that
        # aspect.  This prevents a high-ranked TOC or “成果清单” paragraph from
        # displacing a lower-ranked acceptance-limit table.
        for subquery in plan.subqueries:
            if len(selected) >= limit:
                break
            candidates = [
                item
                for item in results.get(subquery.id, [])
                if item.chunk_id not in selected_ids
            ]
            if not candidates:
                continue
            candidate = self._prefer_substantive_evidence(candidates, subquery.aspect)
            selected.append(candidate)
            selected_ids.add(candidate.chunk_id)
            chunk_subqueries[candidate.chunk_id] = subquery.id
            chunk_cells[candidate.chunk_id] = {
                cell.id
                for cell in plan.subqueries
                if candidate.chunk_id
                in {item.chunk_id for item in results.get(cell.id, [])}
            } or {subquery.id}

        # Some dimensions are inherently multi-evidence.  “成果验收”, for
        # example, can contain separate accuracy, density and product-quality
        # tables.  Fill a bounded semantic target per cell before global score
        # fill so a single high-ranked row does not make the entire dimension
        # appear complete.  Targets describe information shape, not table IDs.
        for subquery in plan.subqueries:
            target = self._target_evidence_count(subquery.aspect)
            if target <= 1 or len(selected) >= limit:
                continue
            candidates = results.get(subquery.id, [])
            selected_signatures = {
                signature
                for item in candidates
                if item.chunk_id in selected_ids
                if (signature := self._evidence_signature(item, subquery.aspect))
            }
            current = len(selected_signatures)
            ranked_candidates = []
            for candidate_index, candidate in enumerate(candidates):
                signature = self._evidence_signature(candidate, subquery.aspect)
                ranked_candidates.append(
                    (
                        0 if signature and signature.startswith("表") else 1,
                        candidate_index,
                        candidate,
                        signature,
                    )
                )
            for _, _, candidate, signature in sorted(ranked_candidates):
                if current >= target or len(selected) >= limit:
                    break
                if candidate.chunk_id in selected_ids:
                    continue
                if not signature or signature in selected_signatures:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
                chunk_subqueries[candidate.chunk_id] = subquery.id
                chunk_cells[candidate.chunk_id] = {subquery.id}
                selected_signatures.add(signature)
                current += 1

        # Multi-hop matrix: before score-based fill, guarantee each document
        # appearing in any subquery's candidates gets at least one
        # representative. Otherwise one keyword-heavy document (e.g. 5G 基站)
        # monopolizes the budget and aspect documents (机巢/河湖) that are
        # clearly present in candidates never enter the final evidence.
        if plan.synthesis_mode == "matrix" and plan.query_type == "multi_hop":
            doc_best: dict[str, tuple[float, RetrievedChunk]] = {}
            for _, candidates in results.items():
                for candidate in candidates:
                    if candidate.chunk_id in selected_ids:
                        continue
                    score = (
                        candidate.rerank_score
                        if candidate.rerank_score is not None
                        else candidate.hybrid_score
                    )
                    previous = doc_best.get(candidate.document_id)
                    if previous is None or score > previous[0]:
                        doc_best[candidate.document_id] = (score, candidate)
            for _, (_, candidate) in sorted(
                doc_best.items(),
                key=lambda kv: kv[1][0],
                reverse=True,
            ):
                if len(selected) >= limit:
                    break
                if candidate.chunk_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
                chunk_subqueries[candidate.chunk_id] = self._source_subquery(
                    plan, results, candidate.chunk_id
                )

        # Remaining budget rewards relevance. For comparison matrices, first
        # fill from the documents already associated with a comparison subject
        # (round-one picks): this pulls the subject's own 作业要求/巡查内容
        # sections into evidence (so "未规定" can be grounded) and keeps
        # unrelated documents (航测、桩位…) from occupying the budget.
        remaining: list[tuple[float, RetrievedChunk]] = []
        subject_documents = {item.document_id for item in selected}
        for _, candidates in results.items():
            for candidate in candidates:
                if candidate.chunk_id in selected_ids:
                    continue
                score = self._score(candidate)
                if candidate.document_id not in subject_documents:
                    score += 0.05
                remaining.append((score, candidate))
        remaining.sort(key=lambda item: item[0], reverse=True)
        if self._is_comparison_matrix(plan):
            # 对比矩阵：主体文档的候选优先占满预算，无关文档只在预算有剩余时进入。
            subject_remaining = [
                item
                for item in remaining
                if item[1].document_id in subject_documents
            ]
            others = [
                item for item in remaining if item[1].document_id not in subject_documents
            ]
            ordered: list[tuple[float, RetrievedChunk]] = [*subject_remaining, *others]
        else:
            ordered = remaining
        for _, candidate in ordered:
            if len(selected) >= limit:
                break
            if candidate.chunk_id in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(candidate.chunk_id)
            source = self._source_subquery(plan, results, candidate.chunk_id)
            chunk_subqueries[candidate.chunk_id] = source
            # 多对多：该 chunk 出现在哪些子查询的候选池里，就支撑哪些格子。
            chunk_cells[candidate.chunk_id] = {
                subquery.id
                for subquery in plan.subqueries
                if candidate.chunk_id in {
                    item.chunk_id for item in results.get(subquery.id, [])
                }
            } or {source}

        matrix = []
        for subquery in plan.subqueries:
            cell_chunks = [
                item
                for item in results.get(subquery.id, [])
                if item.chunk_id in selected_ids
            ]
            status, reason = self._cell_status(
                subquery,
                cell_chunks=cell_chunks,
                missing_subjects=missing_subjects or set(),
                retrieval_view=plan.query_type == "enumeration",
            )
            matrix.append(
                CoverageCell(
                    subquery_id=subquery.id,
                    subject=subquery.subject,
                    aspect=subquery.aspect,
                    status=status,
                    chunk_ids=[item.chunk_id for item in cell_chunks],
                    reason=reason,
                )
            )
        return CoverageSelection(
            chunks=selected,
            matrix=matrix,
            chunk_subqueries=chunk_subqueries,
            chunk_cells=chunk_cells,
        )

    @classmethod
    def _cell_status(
        cls,
        subquery: SubQuery,
        *,
        cell_chunks: list[RetrievedChunk],
        missing_subjects: set[str],
        retrieval_view: bool = False,
    ) -> tuple[str, str | None]:
        """格子四态判定：covered / not_specified / low_confidence / no_document。

        covered 要求该格 chunk 中能提取出 ≥1 条与维度相关的实质性条款行；
        有 chunk 但无实质条款（如桥梁"像控布设"只命中 3.2 自动巡检）为
        low_confidence；无任何选中 chunk 为 not_specified；主体文档未被发现
        为 no_document。
        """
        if subquery.subject and subquery.subject in missing_subjects:
            return "no_document", "subject_document_not_found"
        if not cell_chunks:
            return "not_specified", "no_dimension_evidence_selected"
        # 枚举计划中的 cell 是互补的“检索视角”，不是需要在正文中逐字出现的
        # 比较维度。跨语言适航资料里中文 aspect 与英文评论段落不会有词面重合；
        # 已通过相似度和 reranker 的候选即可作为该视角被覆盖。
        if retrieval_view:
            return "covered", None
        if cell_has_substantive_evidence(cell_chunks, subquery.aspect):
            return "covered", None
        return (
            "low_confidence",
            "chunks_without_substantive_dimension_content",
        )

    @staticmethod
    def _is_comparison_matrix(plan: QueryPlan) -> bool:
        return plan.synthesis_mode == "matrix" and plan.query_type == "comparison"

    @staticmethod
    def _source_subquery(
        plan: QueryPlan,
        results: dict[str, list[RetrievedChunk]],
        chunk_id: str,
    ) -> str:
        for subquery_id, candidates in results.items():
            if any(item.chunk_id == chunk_id for item in candidates):
                return subquery_id
        return "unknown"

    @staticmethod
    def _prefer_parameter_value(
        candidates: list[RetrievedChunk],
        aspect: str,
    ) -> RetrievedChunk:
        if not is_parametric_text(aspect):
            return candidates[0]
        param = [item for item in candidates if has_parameter_value(item.text)]
        if param:
            return param[0]
        # 参数类 aspect 下，含 Markdown 表格的块同样可能承载数值
        # （如"地面分辨率不低于表1"），优先选中以便表格数据进入证据。
        table = [item for item in candidates if _TABLE_ROW_RE.search(item.text)]
        if table:
            return table[0]
        return candidates[0]

    @classmethod
    def _prefer_substantive_evidence(
        cls,
        candidates: list[RetrievedChunk],
        aspect: str,
    ) -> RetrievedChunk:
        """Prefer semantic cell evidence while preserving retrieval order.

        The fallback to the original candidates is deliberate: retaining one
        trace chunk lets coverage report ``low_confidence`` instead of hiding
        the fact that retrieval found only headings or cross-stage material.
        """
        substantive = [
            item
            for item in candidates
            if cell_has_substantive_evidence([item], aspect)
        ]
        if aspect == "成果验收":
            table_evidence = [
                item
                for item in substantive
                if (cls._evidence_signature(item, aspect) or "").startswith("表")
            ]
            if table_evidence:
                return table_evidence[0]
        return cls._prefer_parameter_value(substantive or candidates, aspect)

    @staticmethod
    def _target_evidence_count(aspect: str) -> int:
        if aspect == "成果验收":
            return 4
        if aspect in {"设备选型", "像控布设"}:
            return 2
        return 1

    @staticmethod
    def _evidence_signature(chunk: RetrievedChunk, aspect: str) -> str | None:
        lines = substantive_lines(chunk, dimension_keywords(aspect), aspect)
        if not lines:
            return None
        first = " ".join(lines[0].split())
        # Converted table records begin with “caption；header: value”.  The
        # caption identifies the evidence family while ignoring duplicate
        # copies/versions of the same table.
        if first.startswith("表") and "；" in first:
            return first.split("；", 1)[0]
        return first

    @staticmethod
    def _score(chunk: RetrievedChunk) -> float:
        score = chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score
        if has_parameter_value(chunk.text):
            score += 0.1
        if _TABLE_ROW_RE.search(chunk.text):
            score += 0.05
        return score
