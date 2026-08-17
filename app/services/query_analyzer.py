from __future__ import annotations

import re

from app.ingestion.regulations import extract_article_references
from app.schemas.retrieval import NormalizedQuery, QueryType

EXACT_TOKEN_PATTERN = re.compile(r"\b(?:0[xX][0-9A-Fa-f]+|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+)\b")

# 综合/跨文档题信号：一个问句里要求同时覆盖多个维度或跨多份资料，
# 例如"结合播种技术规范与无人机通用服务管理要求……兼顾技术参数与运营管理哪些双重要求？"
# 这类问题应走多阶段检索，而不是被误判为单次检索的 fact。
_MULTI_DIMENSION_MARKERS = (
    "双重要求",
    "双重",
    "两方面",
    "两个维度",
    "两个层面",
    "兼顾",
    "综合要求",
)
# "结合 X 与/和 Y" 的跨文档结构。
_CROSS_DOC_PATTERN = re.compile(r"结合[^，,。！？!?]{0,30}[与和]")


class QueryAnalyzer:
    """Perform deterministic normalization without damaging technical symbols."""

    def analyze(self, query: str) -> NormalizedQuery:
        normalized = " ".join(query.split())
        query_type = self._classify(normalized)
        article_references = extract_article_references(normalized)
        article_ids = list(
            dict.fromkeys(reference.normalized_id for reference in article_references)
        )
        article_aliases = list(
            dict.fromkeys(
                alias
                for reference in article_references
                for alias in reference.aliases
            )
        )
        exact_tokens = list(
            dict.fromkeys([*EXACT_TOKEN_PATTERN.findall(normalized), *article_ids])
        )
        return NormalizedQuery(
            original_query=query,
            normalized_query=normalized,
            query_type=query_type,
            exact_tokens=exact_tokens,
            article_ids=article_ids,
            article_aliases=article_aliases,
        )

    @staticmethod
    def _classify(query: str) -> QueryType:
        lowered = query.lower()
        if any(token in query for token in ("对比", "比较", "差异")):
            return "comparison"
        if any(token in query for token in ("步骤", "如何", "怎么", "流程")):
            return "procedure"
        if any(token in query for token in ("总结", "概述", "归纳")):
            return "summary"
        if any(token in query for token in ("分别", "以及", "并且")):
            return "multi_hop"
        if any(token in query for token in _MULTI_DIMENSION_MARKERS):
            return "multi_hop"
        if _CROSS_DOC_PATTERN.search(query):
            return "multi_hop"
        if lowered.endswith("?") or query.endswith("？"):
            return "fact"
        return "unknown"
