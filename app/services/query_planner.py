from __future__ import annotations

import re

from app.schemas.retrieval import NormalizedQuery, QueryPlan, SubQuery

_COMPARISON_PREFIX = re.compile(r"^(?:请)?(?:对比|比较)\s*")
_COMPARISON_SPLIT = re.compile(r"[、,，和与及]\s*")
_ASPECT_SPLIT = re.compile(r"[、,，和与及]\s*")
_REQUIREMENT_SPLIT = re.compile(r"(?:以及|并且|同时|；|;)\s*")
_COMPARISON_MARKERS = ("在", "从", "就")
_ASPECT_SUFFIXES = (
    "上有什么不同",
    "上有何不同",
    "方面有什么不同",
    "方面有何不同",
    "有什么不同",
    "有何不同",
    "的差异",
    "差异是什么",
    "进行对比",
    "进行比较",
)
_QUESTION_SUFFIXES = ("分别是什么", "分别有哪些", "是什么", "有哪些", "如何", "怎么")

# "结合 A 与/和 B" 的跨文档结构：A、B 分别是两个来源短语（如技术规范、管理要求）。
_CROSS_DOC_PATTERN = re.compile(r"结合(?P<a>[^与和，,。！？!?]+?)[与和](?P<b>[^，,。！？!?]+)")
# "兼顾 X 与 Y 哪些双重要求" 的维度对：X、Y 是两个维度（如技术参数、运营管理）。
_DUAL_ASPECT_PATTERN = re.compile(r"兼顾(.*?)(?:哪些)?(?:双重|双|两方)要求?")

# 宽泛维度词 → 本语料库具体检索词扩展。
#
# 高层维度词（如"技术参数""运营管理"）与文档章节的具体词汇（如"播种量设计"
# "现场勘察"）存在词汇鸿沟：直接用维度词检索，会命中字面含维度词的无关文档
# （例如水稻水直播规程的"7 整地技术参数"节），而真正含技术参数的章节因为不
# 含"技术参数"四字而排不进前几。扩展词把宽维度拉回本语料库的具体术语，
# 再交给 reranker 做语义重排。
#
# 注意：这是针对本语料库（无人机作业/服务类技术规程）的领域词表，只对
# 跨文档综合题（_cross_source_plan）生效，且只做维度词精确匹配。新增维度
# 词或修改扩展词时，必须同步补回归测试，避免污染普通单文档查询。
_DIMENSION_EXPANSIONS: dict[str, str] = {
    "技术参数": "播种量 航高 速度 种子处理 整地 作业设计",
    "运营管理": "管理制度 服务要求 现场勘察 设备匹配 制定方案 服务实施 成果提交",
}


class QueryPlanner:
    """Build a bounded, deterministic retrieval plan for complex questions."""

    def plan(self, query: NormalizedQuery) -> QueryPlan:
        if query.query_type == "comparison":
            comparison = self._comparison_plan(query)
            if comparison is not None:
                return comparison
        if query.query_type == "multi_hop":
            multi_hop = self._multi_hop_plan(query)
            if multi_hop is not None:
                return multi_hop
        if query.query_type == "procedure":
            return self._single_plan(
                query,
                expansion_mode="adjacent",
                synthesis_mode="sequence",
            )
        if query.query_type == "summary":
            return self._single_plan(
                query,
                expansion_mode="document",
                synthesis_mode="map_reduce",
            )
        return self._single_plan(query)

    def _comparison_plan(self, query: NormalizedQuery) -> QueryPlan | None:
        text = _COMPARISON_PREFIX.sub("", query.normalized_query).strip()
        marker_index = self._first_marker_index(text)
        if marker_index is None:
            return None
        subject_text = text[:marker_index].strip(" ，,：:")
        aspect_text = text[marker_index + 1 :].strip(" ，,：:")
        aspects = self._clean_aspects(aspect_text)
        subjects = self._clean_subjects(subject_text)
        if len(subjects) < 2 or not aspects:
            return None
        subqueries = [
            SubQuery(
                id=f"q{index}",
                query=f"{subject} {aspect}",
                subject=subject,
                aspect=aspect,
            )
            for index, (subject, aspect) in enumerate(
                (
                    (subject, aspect)
                    for aspect in aspects
                    for subject in subjects
                ),
                start=1,
            )
        ][:12]
        return QueryPlan(
            query_type=query.query_type,
            subjects=subjects,
            aspects=aspects,
            subqueries=subqueries,
            expansion_mode="section",
            synthesis_mode="matrix",
        )

    def _multi_hop_plan(self, query: NormalizedQuery) -> QueryPlan | None:
        parts = [
            self._strip_question_suffix(part.strip(" ，,。！？!?"))
            for part in _REQUIREMENT_SPLIT.split(query.normalized_query)
        ]
        aspects = list(dict.fromkeys(part for part in parts if part))
        if len(aspects) >= 2:
            return QueryPlan(
                query_type=query.query_type,
                aspects=aspects,
                subqueries=[
                    SubQuery(id=f"q{index}", query=aspect, aspect=aspect)
                    for index, aspect in enumerate(aspects[:12], start=1)
                ],
                synthesis_mode="matrix",
            )
        # 兜底：识别"结合 A 与 B"跨文档结构，拆成两个来源/维度子查询。
        return self._cross_source_plan(query)

    def _cross_source_plan(self, query: NormalizedQuery) -> QueryPlan | None:
        """拆解"结合 A 与 B……兼顾 X 与 Y"这类跨文档综合题。

        以主体 + 第一个维度为一个子查询，第二个来源 + 第二个维度为另一个子查询，
        使技术参数与运营管理分别命中各自的来源文档。
        """
        text = query.normalized_query
        matched = _CROSS_DOC_PATTERN.search(text)
        if matched is None:
            return None
        subject = text[: matched.start()].strip(" ，,。！？!?：:")
        source_b = matched.group("b").strip(" ，,。！？!?：:")
        if not subject or not source_b:
            return None
        dims = self._dual_aspects(text[matched.end() :])
        if dims:
            subqueries = [
                SubQuery(id="q1", query=self._expand_dimension(subject, dims[0]), aspect=dims[0]),
                SubQuery(id="q2", query=self._expand_dimension(source_b, dims[1]), aspect=dims[1]),
            ]
        else:
            source_a = matched.group("a").strip(" ，,。！？!?：:")
            subqueries = [
                SubQuery(id="q1", query=f"{subject} {source_a}", aspect=source_a),
                SubQuery(id="q2", query=source_b, aspect=source_b),
            ]
        return QueryPlan(
            query_type=query.query_type,
            subjects=[subject],
            aspects=[subquery.aspect for subquery in subqueries],
            subqueries=subqueries,
            synthesis_mode="matrix",
        )

    @staticmethod
    def _expand_dimension(base: str, dimension: str) -> str:
        expansion = _DIMENSION_EXPANSIONS.get(dimension)
        if expansion:
            return f"{base} {dimension} {expansion}"
        return f"{base} {dimension}"

    @staticmethod
    def _dual_aspects(text: str) -> list[str] | None:
        matched = _DUAL_ASPECT_PATTERN.search(text)
        if matched is None:
            return None
        segment = matched.group(1).strip(" ，,。！？!?：:")
        segment = segment.replace("哪些", "")
        parts = [part.strip() for part in re.split(r"[与和]", segment)]
        parts = [part for part in parts if len(part) >= 2]
        if len(parts) < 2:
            return None
        return parts[:2]

    @staticmethod
    def _single_plan(
        query: NormalizedQuery,
        *,
        expansion_mode: str = "none",
        synthesis_mode: str = "direct",
    ) -> QueryPlan:
        return QueryPlan.model_validate(
            {
                "query_type": query.query_type,
                "aspects": [query.normalized_query],
                "subqueries": [
                    {
                        "id": "q1",
                        "query": query.normalized_query,
                        "aspect": query.normalized_query,
                    }
                ],
                "expansion_mode": expansion_mode,
                "synthesis_mode": synthesis_mode,
            }
        )

    @staticmethod
    def _first_marker_index(text: str) -> int | None:
        positions = [text.find(marker) for marker in _COMPARISON_MARKERS]
        valid = [position for position in positions if position > 0]
        return min(valid) if valid else None

    @staticmethod
    def _clean_subjects(text: str) -> list[str]:
        subjects = [item.strip(" ，,：:") for item in _COMPARISON_SPLIT.split(text)]
        return list(dict.fromkeys(item for item in subjects if len(item) >= 2))

    def _clean_aspects(self, text: str) -> list[str]:
        text = text.rstrip(" ，,。！？!?：:")
        for suffix in _ASPECT_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                break
        aspects = [
            self._strip_question_suffix(item.strip(" ，,。！？!?：:"))
            for item in _ASPECT_SPLIT.split(text)
        ]
        return list(dict.fromkeys(item for item in aspects if len(item) >= 2))

    @staticmethod
    def _strip_question_suffix(text: str) -> str:
        for suffix in _QUESTION_SUFFIXES:
            if text.endswith(suffix):
                return text[: -len(suffix)].strip()
        return text
