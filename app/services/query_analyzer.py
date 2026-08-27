from __future__ import annotations

import re

from app.ingestion.regulations import extract_article_references
from app.schemas.retrieval import NormalizedQuery, QueryType

EXACT_TOKEN_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])"
    r"(?:0[xX][0-9A-Fa-f]+|[A-Za-z_][A-Za-z0-9_]*_[A-Za-z0-9_]+)"
    r"(?![A-Za-z0-9_])"
)

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
# 跨文档结构的引导动词："结合/综合/按照/依据/遵照 X 与/和 Y"。
# 不仅"结合"一种说法；"按照 A 和 B""综合 A 与 B 的要求"是同一语义。
_CROSS_DOC_PATTERN = re.compile(r"(?:结合|综合|按照|依据|遵照)[^，,。！？!?]{0,30}[与和]")

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
        # "对比/比较"之外，"区别/异同/区分"同样是两两比较题（如"…三个环节的
        # 标准要求如何统筹区分"、"A 和 B 有什么区别"），必须走对比矩阵路径，
        # 否则会被"如何"误判为 procedure、只回碎片。
        if any(token in query for token in ("对比", "比较", "差异", "区别", "异同", "区分")):
            return "comparison"
        # "…、…、… 三类要求如何协同/匹配/联动" 是跨文档多维度综合题，
        # 必须在 procedure（"如何"标记）之前识别为 multi_hop，
        # 按方面拆子查询，避免单查询被某一主题文档垄断。
        if any(token in query for token in ("协同", "联动", "匹配", "配合")) and (
            any(token in query for token in ("类要求", "类方面", "类环节", "类工作", "类需求"))
            or "同时使用" in query or "结合" in query
        ):
            return "multi_hop"
        if "既要" in query and "也要" in query:
            return "multi_hop"
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
