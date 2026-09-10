from __future__ import annotations

import re

from app.schemas.retrieval import NormalizedQuery, QueryPlan, QueryType, SubQuery
from app.services.requirement_taxonomy import (
    ASPECT_RETRIEVAL_TERMS,
    SUBJECT_RETRIEVAL_TERMS,
)

_COMPARISON_PREFIX = re.compile(r"^(?:请)?(?:对比|比较)\s*")
_COMPARISON_SPLIT = re.compile(r"[、,，和与及]\s*")
_ASPECT_SPLIT = re.compile(r"[、,，和与及]\s*")
_REQUIREMENT_SPLIT = re.compile(r"(?:以及|并且|同时|；|;)\s*")
_COMPARISON_MARKERS = ("在", "从", "就")
# "对比A、B，二者X、Y有哪些差异"：用"二者/两者"定位主体与维度的边界。
_TWO_SIDES_RE = re.compile(r"[,，]\s*(?:二者|两者)(?:\s*之间)?\s*[，,：:]?\s*")
# 句末"差异形成的原因是什么？"这类追问，作为独立子查询，不混入参数维度。
_REASON_TAIL_RE = re.compile(
    r"(?:差异|区别|不同)?(?:形成|产生|造成|导致)的?原因"
    r"(?:是|有)?(?:什么|哪些)?[?？]?$"
)
# 维度后缀按长度降序，保证"有哪些明显差异"先于"明显差异"被剥离。
_ASPECT_SUFFIXES = tuple(
    sorted(
        (
            "上有什么不同",
            "上有何不同",
            "上有什么差异",
            "上有什么区别",
            "方面有什么不同",
            "方面有何不同",
            "方面有什么差异",
            "方面有什么区别",
            "有什么不同",
            "有何不同",
            "有什么差异",
            "有何差异",
            "有什么区别",
            "有何区别",
            "的差异",
            "差异是什么",
            "进行对比",
            "进行比较",
            "存在哪些区别",
            "存在什么区别",
            "存在哪些差异",
            "存在什么差异",
            "有哪些明显差异",
            "有何明显差异",
            "有哪些差异",
            "有哪些区别",
            "明显差异",
            "上如何统筹区分",
            "方面如何统筹区分",
            "上如何统筹协调",
            "方面如何统筹协调",
            "如何统筹区分",
            "如何统筹协调",
            "上如何区分",
            "如何区分",
            "上如何协调",
            "如何协调",
        ),
        key=len,
        reverse=True,
    )
)
_QUESTION_SUFFIXES = ("分别是什么", "分别有哪些", "是什么", "有哪些", "如何", "怎么")

# "结合 A 与/和 B" 的跨文档结构：A、B 分别是两个来源短语（如技术规范、管理要求）。
# 引导动词覆盖"结合/综合/按照/依据/遵照"，与 query_analyzer 的 _CROSS_DOC_PATTERN
# 保持同一语义；"按照 A 和 B…有哪些要求"与"结合 A 与 B…双重要求"应同等拆解。
_CROSS_DOC_PATTERN = re.compile(
    r"(?:结合|综合|按照|依据|遵照)(?P<a>[^与和，,。！？!?]+?)[与和](?P<b>[^，,。！？!?]+)"
)
# "兼顾 X 与 Y 哪些双重要求" 的维度对：X、Y 是两个维度（如技术参数、运营管理）。
_DUAL_ASPECT_PATTERN = re.compile(r"兼顾(.*?)(?:哪些)?(?:双重|双|两方)要求?")
# "…空域通信、机巢运维、数据归档三类要求…"：枚举维度 + N 类要求/方面。
_CATEGORY_ENUM_RE = re.compile(
    r"(?P<items>[^，,。！？!?？]{2,80}?)(?:等)?"
    r"(?P<n>两|三|四|五|六|七|八|九|几|[2-9])类"
    r"(?:要求|方面|环节|工作|需求|任务)"
)
_PARALLEL_REQUIREMENTS_RE = re.compile(
    r"(?P<subject>.+?)既要(?:满足)?(?P<a>.+?要求)[，,]?(?:也要)(?:满足)?(?P<b>.+?要求)"
)

# "…X、Y、Z 三个环节的标准要求如何统筹区分"：枚举维度后的数量词尾巴
# （"N 个环节/方面/维度/要素/阶段/层面/工作/流程"+标准要求+提问尾巴）不属
# 维度本身，预剥离避免最后一个维度被污染（"成果验收三个环节…"）。
_ASPECT_COUNT_TAIL_RE = re.compile(
    r"(?:[两三四五六七八九2-9])\s*个\s*"
    r"(?:环节|方面|维度|要素|阶段|层面|工作|流程)"
    r"(?:上|中)?(?:的)?(?:标准)?(?:要求|规定)?"
    r"[^、，,。！？!?]*$"
)
# 宽泛维度词 → 本语料库具体检索词扩展。
#
# 高层维度词（如"技术参数""运营管理"）与文档章节的具体词汇（如"播种量设计"
# "现场勘察"）存在词汇鸿沟：直接用维度词检索，会命中字面含维度词的无关文档
# （例如水稻水直播规程的"7 整地技术参数"节），而真正含技术参数的章节因为不
# 含"技术参数"四字而排不进前几。扩展词把宽维度拉回本语料库的具体术语，
# 再交给 reranker 做语义重排。
#
# 注意：这是针对本语料库（无人机作业/服务类技术规程）的领域词表，对
# 跨文档综合题（_cross_source_plan）与对比/比较矩阵子查询（_build_comparison_matrix）
# 生效，且只做维度词精确匹配。新增维度词或修改扩展词时，必须同步补回归测试，
# 避免污染普通单文档查询。
_DIMENSION_EXPANSIONS = ASPECT_RETRIEVAL_TERMS

# 对比主体词 → 本语料库具体检索词扩展。
#
# 用户口语化主体（如"水稻植保无人机""林地病虫害喷洒无人机"）与规范标题中的
# 术语（如"无人机防治稻纵卷叶螟""无人机喷洒防治林业有害生物"）存在词汇鸿沟，
# 直接用主体词检索会漏掉真正含参数的专业规范。扩展词把口语主体拉回规范术语，
# 且仅在 subquery.query 中生效，不影响 coverage 矩阵的 subject/aspect 显示。
_SUBJECT_EXPANSIONS = SUBJECT_RETRIEVAL_TERMS


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
        if query.query_type == "enumeration":
            return self._enumeration_plan(query)
        return self._single_plan(query)

    @staticmethod
    def _enumeration_plan(query: NormalizedQuery) -> QueryPlan:
        """为跨段落实体清单提供互补的语义与字面检索视角。

        英文触发词有意独立成子查询：适航资料的中文问法与英文原文之间存在
        词汇鸿沟，而机构名称通常只出现在 ``received comments from``、
        ``stated`` 等处理意见段落中。
        """
        text = query.normalized_query
        subqueries = [
            SubQuery(id="q1", query=text, aspect="实体类型与总述"),
            SubQuery(
                id="q2",
                query=f"{text} 具体名称 完整名单 机构 单位 组织",
                aspect="具体实体名称",
            ),
            SubQuery(
                id="q3",
                query=(
                    f"{text} received comments from commenter commented stated "
                    "requested recommended suggested objected"
                ),
                aspect="分散在讨论段落中的实体",
            ),
        ]
        return QueryPlan(
            query_type="enumeration",
            aspects=[item.aspect for item in subqueries],
            subqueries=subqueries,
            expansion_mode="document",
            synthesis_mode="map_reduce",
        )

    def _comparison_plan(self, query: NormalizedQuery) -> QueryPlan | None:
        text = query.normalized_query
        prompt_marker = max(text.rfind("请比较"), text.rfind("请对比"))
        if prompt_marker >= 0:
            text = text[prompt_marker + 1 :]
        text = _COMPARISON_PREFIX.sub("", text).strip()
        # "对比A、B，二者X、Y…"句式优先用"二者/两者"定位主体与维度边界；
        # 否则"存在/不在/实在"里的"在"会被 _first_marker_index 误判为比较标记。
        match = _TWO_SIDES_RE.search(text)
        if match is not None:
            subject_text = text[:match.start()].strip(" ，,：:")
            aspect_text = text[match.end():].strip(" ，,：:")
        else:
            marker_index = self._first_marker_index(text)
            if marker_index is None:
                possessive = re.fullmatch(
                    r"(?P<subjects>.+?)的(?P<aspect>[^。！？!?]+)[。！？!?]?",
                    text,
                )
                if possessive is not None:
                    return self._build_comparison_matrix(
                        query.query_type,
                        possessive.group("subjects"),
                        possessive.group("aspect"),
                    )
                return None
            subject_text = text[:marker_index].strip(" ，,：:")
            aspect_text = text[marker_index + 1 :].strip(" ，,：:")
        return self._build_comparison_matrix(query.query_type, subject_text, aspect_text)

    def _build_comparison_matrix(
        self,
        query_type: QueryType,
        subject_text: str,
        aspect_text: str,
    ) -> QueryPlan | None:
        subjects = self._clean_subjects(subject_text)
        if len(subjects) < 2:
            return None
        aspect_text, reason = self._extract_reason(aspect_text)
        aspects = self._clean_aspects(aspect_text)
        if not aspects:
            return None
        subqueries = [
            SubQuery(
                id=f"q{index}",
                query=self._expand_dimension(self._expand_subject(subject), aspect),
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
        if reason:
            subqueries.append(
                SubQuery(
                    id=f"q{len(subqueries) + 1}",
                    query=f"{' '.join(subjects)} {reason}",
                    subject="、".join(subjects),
                    aspect=reason,
                )
            )
            aspects = [*aspects, reason]
        return QueryPlan(
            query_type=query_type,
            subjects=subjects,
            aspects=aspects,
            subqueries=subqueries,
            expansion_mode="section",
            synthesis_mode="matrix",
        )

    @staticmethod
    def _extract_reason(aspect_text: str) -> tuple[str, str | None]:
        match = _REASON_TAIL_RE.search(aspect_text)
        if match is None:
            return aspect_text, None
        remaining = aspect_text[:match.start()].rstrip(" ，,。！？!?：:；;")
        return remaining, "差异形成原因"

    def _multi_hop_plan(self, query: NormalizedQuery) -> QueryPlan | None:
        parallel = _PARALLEL_REQUIREMENTS_RE.search(query.normalized_query)
        if parallel is not None:
            subject = parallel.group("subject").strip(" ，,。！？!?：:")
            aspects = [parallel.group("a"), parallel.group("b")]
            return QueryPlan(
                query_type="multi_hop",
                subjects=[subject],
                aspects=aspects,
                subqueries=[
                    SubQuery(
                        id=f"q{index}",
                        query=f"{subject} {aspect}",
                        subject=subject,
                        aspect=aspect,
                    )
                    for index, aspect in enumerate(aspects, start=1)
                ],
                synthesis_mode="matrix",
            )
        parts = [
            self._strip_question_suffix(part.strip(" ，,。！？!?"))
            for part in _REQUIREMENT_SPLIT.split(query.normalized_query)
        ]
        aspects = list(dict.fromkeys(part for part in parts if part))
        if len(aspects) < 2:
            aspects = self._enumerated_dimensions(query.normalized_query)
        if len(aspects) >= 2:
            # 子查询携带完整上下文（主体/场景词），避免单文档垄断候选池：
            # "同时使用 5G 低空基站与自动机巢开展河湖常态化无人机巡查，
            # 空域通信、机巢运维、数据归档三类要求如何协同匹配？"
            # → q1 "同时使用 5G 低空基站与自动机巢开展河湖常态化无人机巡查，
            #    空域通信" 等，让机巢/河湖文档有机会被各方面子查询召回。
            context = self._multi_hop_context(query.normalized_query, aspects)
            subject = self._normalize_context_subject(context)
            return QueryPlan(
                query_type=query.query_type,
                subjects=[subject] if subject else [],
                aspects=aspects,
                subqueries=[
                    SubQuery(
                        id=f"q{index}",
                        query=f"{context} {aspect}".strip(),
                        subject=subject or None,
                        aspect=aspect,
                    )
                    for index, aspect in enumerate(aspects[:12], start=1)
                ],
                synthesis_mode="matrix",
            )
        # 兜底：识别"结合 A 与 B"跨文档结构，拆成两个来源/维度子查询。
        return self._cross_source_plan(query)

    @staticmethod
    def _enumerated_dimensions(text: str) -> list[str]:
        """识别"…X、Y、Z 三类要求…"的枚举维度。"""
        match = _CATEGORY_ENUM_RE.search(text)
        if match is None:
            return []
        items = [
            item.strip(" ，,：:。！？!?")
            for item in match.group("items").split("、")
        ]
        return list(dict.fromkeys(item for item in items if len(item) >= 2))

    @staticmethod
    def _multi_hop_context(text: str, aspects: list[str]) -> str:
        """提取枚举维度之前的上下文作为子查询主体（去句末问句尾巴）。"""
        match = _CATEGORY_ENUM_RE.search(text)
        if match is None:
            return ""
        context = text[: match.start()].strip(" ，,：:。！？!?；;")
        for suffix in ("如何协同匹配", "如何协同", "如何匹配", "如何配合", "如何联动", "如何结合"):
            if context.endswith(suffix):
                context = context[: -len(suffix)].strip(" ，,：:")
                break
        return context

    @staticmethod
    def _normalize_context_subject(context: str) -> str:
        for prefix in ("同时使用", "开展", "进行", "实施"):
            if context.startswith(prefix):
                return context[len(prefix) :].strip()
        return context

    def _cross_source_plan(self, query: NormalizedQuery) -> QueryPlan | None:
        """拆解"结合 A 与 B……兼顾 X 与 Y"这类跨文档综合题。

        以主体 + 第一个维度为一个子查询，第二个来源 + 第二个维度为另一个子查询，
        使技术参数与运营管理分别命中各自的来源文档。
        """
        text = query.normalized_query
        matched = _CROSS_DOC_PATTERN.search(text)
        if matched is None:
            return None
        subject = self._strip_source_tail(
            text[: matched.start()].strip(" ，,。！？!?：:")
        )
        source_b = self._strip_source_tail(
            matched.group("b").strip(" ，,。！？!?：:")
        )
        if not source_b:
            return None
        if not subject:
            # "结合 A 与 B…" 位于句首时没有前导主语；第一个来源 A 本身就是
            # 一个声明主体（如"结合无人机播种治沙技术规程与无人机应用服务
            # 通用规范，…"），用 A 承担 q1 的主体。
            subject = self._strip_source_tail(
                matched.group("a").strip(" ，,。！？!?：:")
            )
        dims = self._dual_aspects(text[matched.end() :])
        if dims:
            subqueries = [
                SubQuery(
                    id="q1",
                    query=self._expand_dimension(subject, dims[0]),
                    subject=subject,
                    aspect=dims[0],
                ),
                SubQuery(
                    id="q2",
                    query=self._expand_dimension(source_b, dims[1]),
                    subject=source_b,
                    aspect=dims[1],
                ),
            ]
        else:
            source_a = self._strip_source_tail(
                matched.group("a").strip(" ，,。！？!?：:")
            )
            q1_query = f"{subject} {source_a}".strip() if subject != source_a else source_a
            subqueries = [
                SubQuery(id="q1", query=q1_query, subject=subject, aspect=source_a),
                SubQuery(id="q2", query=source_b, subject=source_b, aspect=source_b),
            ]
        return QueryPlan(
            query_type=query.query_type,
            subjects=[subject, source_b],
            aspects=[subquery.aspect for subquery in subqueries],
            subqueries=subqueries,
            synthesis_mode="matrix",
        )

    @staticmethod
    def _expand_subject(subject: str) -> str:
        expansion = _SUBJECT_EXPANSIONS.get(subject)
        if expansion:
            return f"{subject} {expansion}"
        return subject

    @staticmethod
    def _expand_dimension(base: str, dimension: str) -> str:
        expansion = _DIMENSION_EXPANSIONS.get(dimension)
        if expansion:
            return f"{base} {dimension} {expansion}"
        return f"{base} {dimension}"

    @staticmethod
    def _strip_source_tail(value: str) -> str:
        """剥离来源短语尾部的修饰尾巴（"的要求/的相关要求/的需要"等）。

        "综合播种技术规范与无人机通用服务规范的要求" 中来源 B 捕获为
        "无人机通用服务规范的要求"，尾巴不是来源名的一部分。
        """
        for tail in (
            "的相关要求",
            "的相关规定",
            "的通用要求",
            "的要求",
            "的规定",
            "的需要",
            "执行",
        ):
            if value.endswith(tail):
                return value[: -len(tail)]
        return value

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
        subjects = [item for item in subjects if len(item) >= 2]
        if subjects:
            suffix_match = re.match(r"^.+?型(?P<suffix>.+)$", subjects[-1])
            suffix = suffix_match.group("suffix") if suffix_match else ""
            if suffix:
                subjects = [
                    f"{item}{suffix}" if item.endswith("型") else item for item in subjects
                ]
        return list(dict.fromkeys(subjects))

    def _clean_aspects(self, text: str) -> list[str]:
        text = text.rstrip(" ，,。！？!?：:")
        # 剥离"…X、Y、Z 三个环节的标准要求如何统筹区分"的数量词尾巴，
        # 使最后一个维度保持干净（"成果验收"而非"成果验收三个环节…"）。
        text = _ASPECT_COUNT_TAIL_RE.sub("", text)
        # 剥离"二者在X、Y…"残留的开头比较标记（在/从/就），只剥一次。
        text = re.sub(r"^(?:在|从|就)", "", text)
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
