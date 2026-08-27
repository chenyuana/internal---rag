from __future__ import annotations

import re

from app.schemas.planning import PlannedCell, QueryPlanV2, RetrievalQuery
from app.schemas.retrieval import NormalizedQuery, QueryPlan, SubQuery


class QueryPlanAdapter:
    """Bridge QueryPlanV2 to legacy consumers during the bounded migration."""

    @staticmethod
    def from_legacy(query: NormalizedQuery, plan: QueryPlan) -> QueryPlanV2:
        cells: list[PlannedCell] = []
        subjects = plan.subjects[:4]
        aspects = plan.aspects[:8]
        source_subqueries = plan.subqueries
        warnings = ["legacy_planner_fallback"]
        query_type = plan.query_type
        synthesis_mode = plan.synthesis_mode
        if plan.query_type == "comparison" and len(subjects) < 2:
            query_type = "comparison"
            subjects = []
            aspects = []
            source_subqueries = [
                SubQuery(
                    id="q1",
                    query=query.normalized_query,
                    aspect=query.normalized_query,
                )
            ]
            synthesis_mode = "direct"
            warnings.append("对比主体不明确，需要补充主体后再检索")
        if plan.query_type == "comparison" and len(subjects) >= 2:
            planned_queries = {
                (subquery.subject, subquery.aspect): subquery.query
                for subquery in plan.subqueries
                if subquery.subject
            }
            max_aspects = max(1, 8 // len(subjects))
            if len(aspects) > max_aspects:
                warnings.append("对比矩阵超过规划上限，已保留不超过8格的核心维度")
            aspects = aspects[:max_aspects]
            source_subqueries = [
                SubQuery(
                    id=f"q{index}",
                    query=planned_queries.get(
                        (subject, aspect), f"{subject} {aspect}"
                    ),
                    subject=subject,
                    aspect=aspect,
                )
                for index, (aspect, subject) in enumerate(
                    ((aspect, subject) for aspect in aspects for subject in subjects),
                    start=1,
                )
            ]
        for subquery in source_subqueries[:8]:
            if plan.query_type == "comparison":
                original_query = subquery.query
            elif subquery.subject:
                original_query = f"{subquery.subject} {subquery.aspect}".strip()
            else:
                original_query = subquery.query
            cells.append(
                PlannedCell(
                    id=subquery.id,
                    subject=subquery.subject,
                    aspect=subquery.aspect,
                    original_query=original_query,
                    retrieval_queries=[
                        RetrievalQuery(
                            kind="original",
                            text=original_query,
                            generated_by="user" if len(plan.subqueries) == 1 else "planner",
                        )
                    ],
                    required=subquery.required,
                )
            )
        return QueryPlanV2(
            query_type=query_type,
            subjects=subjects,
            aspects=aspects,
            cells=cells,
            synthesis_mode=synthesis_mode,
            planner_source="legacy_fallback",
            confidence=0.6,
            warnings=warnings,
        )

    @staticmethod
    def to_legacy(plan: QueryPlanV2) -> QueryPlan:
        expansion_mode = {
            "sequence": "adjacent",
            "map_reduce": "document",
            "matrix": "section",
        }.get(plan.synthesis_mode, "none")
        return QueryPlan(
            query_type=plan.query_type,
            subjects=plan.subjects,
            aspects=plan.aspects,
            subqueries=[
                SubQuery(
                    id=cell.id,
                    query=cell.original_query,
                    subject=cell.subject,
                    aspect=cell.aspect,
                    required=cell.required,
                )
                for cell in plan.cells
            ],
            expansion_mode=expansion_mode,
            synthesis_mode=plan.synthesis_mode,
        )

    @staticmethod
    def deterministic_single(query: NormalizedQuery) -> QueryPlanV2:
        synthesis_mode = {
            "procedure": "sequence",
            "summary": "map_reduce",
        }.get(query.query_type, "direct")
        subjects, aspects = QueryPlanAdapter._simple_fields(query)
        return QueryPlanV2(
            query_type=query.query_type,
            subjects=subjects,
            aspects=aspects,
            cells=[
                PlannedCell(
                    id="q1",
                    aspect=query.normalized_query,
                    original_query=query.normalized_query,
                    retrieval_queries=[
                        RetrievalQuery(
                            kind="original",
                            text=query.normalized_query,
                            generated_by="user",
                        )
                    ],
                )
            ],
            synthesis_mode=synthesis_mode,
            planner_source="deterministic",
            confidence=1.0,
        )

    @staticmethod
    def _simple_fields(query: NormalizedQuery) -> tuple[list[str], list[str]]:
        text = query.normalized_query.rstrip("。！？?!")
        if query.exact_tokens:
            standard = re.match(r"(?P<standard>[A-Za-z]+(?:-[A-Za-z0-9]+)+)(?=第)", text)
            subjects = [standard.group("standard")] if standard else []
            constant = next(
                (token for token in query.exact_tokens if "_" in token),
                None,
            )
            if constant:
                return subjects, [constant]
            hexadecimal = next(
                (token for token in query.exact_tokens if token.lower().startswith("0x")),
                None,
            )
            if hexadecimal:
                prefix = text[: text.index(hexadecimal) + len(hexadecimal)]
                return subjects, [prefix]
            aspect = text
            if standard:
                aspect = aspect[len(standard.group("standard")) :]
                aspect = re.sub(r"^第\d+(?:\.\d+)*条对", "", aspect)
                aspect = re.sub(r"(?:有什么|有何|有哪些)要求$", "", aspect)
            else:
                aspect = re.sub(r"(?:是什么|有哪些|的内容是什么)$", "", aspect)
                if query.article_ids and "内容" in text:
                    aspect = re.sub(r"是什么$", "", text)
            return subjects, [aspect]
        definition = re.fullmatch(r"(.+?)的定义是什么", text)
        if definition:
            return [definition.group(1)], ["定义"]
        standard_question = re.fullmatch(r"(.+?)(?:通常)?依据哪个标准", text)
        if standard_question:
            return [standard_question.group(1)], ["依据标准"]
        if query.query_type == "procedure":
            complete = re.match(r"(.+?)从.+?到.+?的完整流程", text)
            if complete:
                return [complete.group(1)], ["完整流程"]
            upload = re.fullmatch(r"如何完成(.+?)的(.+)", text)
            if upload:
                return [upload.group(1)], [upload.group(2)]
        if query.query_type == "summary":
            summary = re.fullmatch(r"概述(.+?)对(.+?)的总体要求", text)
            if summary:
                return [summary.group(1)], [f"{summary.group(2)}总体要求"]
        if query.query_type == "unknown":
            return [text], [text]
        return [], [text]
