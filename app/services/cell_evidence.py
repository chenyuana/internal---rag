"""格子证据判定：维度相关且实质性的条款行提取与覆盖判定。

对比矩阵的每个（主体×维度）格是否"covered"，取决于该格 chunk 中能否提取出
与维度相关的**实质性条款行**（而非目录、章节标题、概念定义、Markdown 表格
原始行）。此模块是 coverage_selector（covered 判定）与
comparison_matrix_builder（格内容）的共用基础。

行级过滤规则：
- 元数据行（文档：/章节：/条号：/页码：）剔除；
- Markdown 表格原始行不直接作内容（但见 Stage 3：A.x 表格应转键值文本）；
- 编号标题行（"7.1.1 像控点布设"）、"文字+编号"引用行（"无人机设备维护
  保养 4.4"）、无实质内容的引导行剔除；
- 行长超过上限的章节目录/聚合行剔除；
- 与"维度词+扩展词"共享 ≥1 个**非通用 bigram**（控制/系统/检查/记录/数据/
  成果/要求…）即视为维度相关。
"""

from __future__ import annotations

import re
from html import unescape
from typing import Any

from app.services.evidence_text import lexical_units
from app.services.requirement_taxonomy import ASPECT_RETRIEVAL_TERMS, line_matches_aspect
from app.services.table_evidence import table_records, without_table_markup

# 行尾标点：带这些结尾的行是完整句/完整项，不与下一行合并。
_SENTENCE_END_RE = re.compile(r"[。；;！？!?：:）)]$")
# 合并跨行断句：PDF 硬换行常把"（无人驾驶飞行\n器）作为载体"切断。
# 仅当行尾无句读（或为右括号）且下一行不以编号/标题样式开头时合并，
# 避免把相邻独立句子错误粘连。
_HEADING_OR_NUMBER_START_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*|[A-Za-z]\d*|[（(]?[a-z]|[一二三四五六七八九十]+[、.])"
)


def merge_wrapped_lines(lines: list[str]) -> list[str]:
    """把被 PDF 硬换行切断的句子接回同一行（保守策略）。

    规则：当前行不以句末标点（。；！？：或右括号）结尾时，尝试与后续
    正文行合并；后续行若是编号/字母分项头/标题样式则视为新一段起点，
    不并入（并放回待处理）。逐行推进，不跨大段粘连。
    """
    return _merge_pass(lines)


def _merge_pass(lines: list[str]) -> list[str]:
    result: list[str] = []
    index = 0
    while index < len(lines):
        current = lines[index].strip()
        index += 1
        if not current:
            continue
        # 元数据行（文档：/章节：/条号：/页码：）不参与合并，直接保留；
        # 否则"章节：…基本要求"与下一行条款会合并成一行被 META 整行过滤。
        if META_LINE_RE.match(current):
            result.append(current)
            continue
        # 当前行自身是编号/标题样式（如"5.5 航高和速度设计"）时是独立段落，
        # 不与后续正文合并；后续正文作为新一段处理。
        if _HEADING_OR_NUMBER_START_RE.match(current):
            result.append(current)
            continue
        if _SENTENCE_END_RE.search(current):
            result.append(current)
            continue
        combined = current
        while index < len(lines):
            nxt = lines[index].strip()
            if not nxt:
                index += 1
                continue
            if _HEADING_OR_NUMBER_START_RE.match(nxt) or META_LINE_RE.match(nxt):
                # 下一行是新编号/标题项或元数据行：不并入，留待下一轮。
                break
            index += 1
            combined = combined + nxt
            if _SENTENCE_END_RE.search(combined):
                break
        result.append(combined)
    return result

# 与"维度词+扩展词"共享的非通用 bigram 数量下限。
MIN_SHARED_BIGRAMS = 1
# 行长度上限：超过视为章节目录/聚合噪声行（如内嵌整章 TOC 的长行）。
MAX_LINE_LENGTH = 90

# 元数据前缀行。
META_LINE_RE = re.compile(r"^(?:文档|章节|条号|页码)\s*[:：]?")
# Markdown 表格原始行。
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# 无句读的编号标题行（"7.1.1 像控点布设"）。
HEADING_LINE_RE = re.compile(
    r"^(?:\d+(?:\.\d+)*|[A-Za-z]\d*)\s*[\u4e00-\u9fff]{2,20}$"
)
# "文字+编号"章节引用行（"无人机设备维护保养 4.4"）。
TEXT_NUMBER_HEADING_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z]{2,20}\s*\d+(?:\.\d+)*$")
PLAIN_HEADING_RE = re.compile(r"^[\u4e00-\u9fffA-Za-z\s]{2,30}$")
TOC_LINE_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[A-Za-z]\d*)[.\s]+[\u4e00-\u9fffA-Za-z]{2,24}[。.·]*$"
)
# 目录行只列标题，不含条款性动词；带"应/宜/须/不得/按照/执行/要求/采用"等
# 的编号行是实际条款（如"8.2.2 应依据作业要求匹配作业现场条件出具现场勘察
# 报告。"），不得被 TOC_LINE_RE 误伤。
_CLAUSE_VERB_MARKERS = (
    "应",
    "宜",
    "须",
    "不得",
    "按照",
    "执行",
    "要求",
    "采用",
    "包括",
    "为",
    "应依据",
    "应制定",
)
# 空白验收记录表的说明行（“样式见表…”/“精度检查记录表”）不能替代
# 限差或实测值。带有明确限差、数值或合格条件的已填写记录仍允许通过。
FORM_REFERENCE_RE = re.compile(r"(?:样式见表|记录表|记录格式)")
NORMATIVE_VALUE_RE = re.compile(
    r"(?:限差|中误差|不大于|不小于|应满足|合格|超限|\d+(?:\.\d+)?\s*(?:m|cm|mm|像素|%|级))"
)
# 引用标记 [Cx]。
CITATION_MARKER_RE = re.compile(r"\[C\d+\]")
# 规范性引用文件/参考文献中的标准目录项只说明“引用了哪份标准”，并不
# 给出本题所问的实体要求。实际条款中的“按照 CH/T 3004 执行”通常以中文
# 句子开头，不受此规则影响。
STANDARD_REFERENCE_LINE_RE = re.compile(
    r"^(?:\[\d+\]\s*)?(?:GB|CH|DB|TD|JGJ|JTG|MH|ISO|IEC)(?:/T|/Z)?\s*\d",
    re.IGNORECASE,
)
# 文档范围句："本文件/本标准/本规范 规定了…/适用于…/确立…" 是文档引言
# 或范围章节的内容（列目录性质），不是任何环节的实质条款。例如
# "本文件规定了无人机应用服务的基本要求、机构要求、人员要求…" 若进入
# 运营管理格，会挤掉现场勘察/风险告知等真实条款的名额。
SCOPE_SENTENCE_RE = re.compile(
    r"^(?:本文件|本标准|本规范|本规程|本规定|本细则)"
    r"(?:规定了|适用于|确立了|规定了以下|包含以下)"
)

# 打分时剔除的通用 bigram：如"控制系统"中的"控制"与"控制点"共享，会把无关
# 内容（桥梁 3.2 自动巡检）误当成"像控布设"。行与关键词两侧都排除。
GENERIC_BIGRAMS = frozenset(
    {
        "控制",
        "系统",
        "作业",
        "要求",
        "检查",
        "记录",
        "录表",
        "数据",
        "成果",
        "技术",
        "进行",
        "使用",
        "包括",
        "相关",
        "内容",
        "方式",
        "方法",
        "过程",
        "信息",
        "规定",
        "规范",
        "按照",
        "根据",
        "以及",
        "并且",
        "同时",
        "需要",
        "应该",
        "必须",
        "不得",
        "或者",
    }
)


def dimension_keywords(aspect: str) -> set[str]:
    """维度词 + 维度扩展词的非通用 bigram 集合。"""
    keywords = lexical_units(aspect)
    keywords.update(lexical_units(ASPECT_RETRIEVAL_TERMS.get(aspect, "")))
    keywords -= GENERIC_BIGRAMS
    return keywords


def substantive_lines(
    chunk: Any,
    keywords: set[str],
    aspect: str | None = None,
) -> list[str]:
    """返回 chunk 中与维度相关的实质性条款行（按共享 bigram 数降序）。

    chunk 为带 ``text`` 的 RetrievedChunk/SelectedChunk（检索阶段与生成阶段
    共用同一判定逻辑）。
    """
    if not keywords:
        return []
    scored: list[tuple[int, str]] = []
    seen: set[str] = set()
    converted_tables = table_records(chunk.text)
    converted_table_set = set(converted_tables)
    raw_lines = without_table_markup(chunk.text).splitlines()
    lines = [*converted_tables, *merge_wrapped_lines(raw_lines)]
    for line in lines:
        # RAGFlow occasionally preserves numeric character references in
        # surrounding prose (for example ``适&#x5F53;``).  Decode text only after
        # table markup has been removed, so the answer never leaks entities.
        line = unescape(line).strip()
        if not line or META_LINE_RE.match(line) or TABLE_ROW_RE.match(line):
            continue
        # 行长上限针对目录/聚合噪声行（通常无句末标点）；合并后的完整句
        # （以句号/问号结尾）即使较长也是合法条款，不被误杀。
        if (
            len(line) > MAX_LINE_LENGTH
            and line not in converted_table_set
            and not re.search(r"[。！？!?]$", line)
        ):
            continue
        if HEADING_LINE_RE.match(line) or TEXT_NUMBER_HEADING_RE.match(line):
            continue
        if TOC_LINE_RE.match(line) and not any(
            marker in line for marker in _CLAUSE_VERB_MARKERS
        ):
            continue
        if PLAIN_HEADING_RE.match(line) and not any(
            marker in line for marker in ("应", "宜", "须", "不得", "采用", "包括", "配置", "为")
        ):
            continue
        if line.endswith(("：", ":")):
            continue
        if line.endswith(("满足以", "包括以")):
            continue
        if CITATION_MARKER_RE.search(line):
            continue
        if STANDARD_REFERENCE_LINE_RE.match(line):
            continue
        if SCOPE_SENTENCE_RE.match(line):
            continue
        if FORM_REFERENCE_RE.search(line) and not NORMATIVE_VALUE_RE.search(line):
            continue
        if any(
            marker in line
            for marker in ("automatic patrol", "manual patrol", "precision flying method")
        ):
            continue
        if aspect is not None and not line_matches_aspect(aspect, line):
            continue
        if line in seen:
            continue
        shared = len((keywords & lexical_units(line)) - GENERIC_BIGRAMS)
        if shared >= MIN_SHARED_BIGRAMS:
            seen.add(line)
            # A full table record carries the concrete values behind nearby
            # prose such as “参照表4执行”; rank it before that reference so the
            # matrix names the table and the UI can render it in full.
            table_bonus = 100 if line in converted_table_set else 0
            scored.append((shared + table_bonus, line))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [line for _, line in scored]


def cell_has_substantive_evidence(
    chunks: list[Any],
    aspect: str,
) -> bool:
    """格子是否含 ≥1 条与维度相关的实质性条款行（covered 判定的核心）。"""
    keywords = dimension_keywords(aspect)
    return any(substantive_lines(chunk, keywords, aspect) for chunk in chunks)
