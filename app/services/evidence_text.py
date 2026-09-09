from __future__ import annotations

import re
from decimal import Decimal

from app.services.domain_terms import normalize_concepts

SENTENCE_BOUNDARY = re.compile(r"(?<=[。！？!?；;])\s*|\r?\n+")
ASCII_TOKEN = re.compile(r"0[xX][0-9A-Fa-f]+|[A-Za-z_][A-Za-z0-9_.:-]*|\d+(?:\.\d+)?")
HAN_RUN = re.compile(r"[\u4e00-\u9fff]+")
MEASUREMENT = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>毫秒|ms|秒|s|分钟|min|小时|h|hz|khz|mhz|字节|bytes?)",
    re.IGNORECASE,
)
GENERIC_UNITS = {
    "什么",
    "多少",
    "如何",
    "怎么",
    "是否",
    "分别",
    "以及",
    "并且",
    "请问",
    "的是",
    "是什么",
    "中的",
}

# 物理量单位白名单：用于判断"参数类需求"的句子是否含具体数值（区别于纯定义句
# 或"3.1 作业高度"这类条款号）。覆盖农用/工业常见物理量：长度、速度、液体量、
# 面积、浓度、时间、频率、温度等。长单位（667m²、km/h、m/s）必须排在短单位
# （m、s、h）之前，避免交替匹配截断。
_PARAM_UNIT_PATTERN = (
    r"667m[²2]|km/h|m/s|m[²2]|kHz|MHz|Hz|"
    r"feet|foot|ft|knots?|lbs?|pounds?|degrees?|minutes?|seconds?|inches|"
    r"percent|million|billion|英尺|英寸|海里|节|磅|美元|分钟|"
    r"km|cm|mm|kg|mL|ml|min|ms|"
    r"L|m|g|t|s|h|%|℃|°|"
    r"亩|米|千米|厘米|毫米|升|毫升|公斤|克|吨|秒|分|小时"
)

_PARAM_VALUE_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:[~～\-—至到]\s*\d+(?:\.\d+)?\s*)?(?:"
    + _PARAM_UNIT_PATTERN
    + r")(?![A-Za-z0-9])",
    re.IGNORECASE,
)

# 参数公式/量纲定义信号：无数值+单位组合，但明确承载参数单位或公式。
# 例："单位为克每公顷 (g/hm^2)"、"依据公式（1）确定"。对"播种量计算逻辑"
# 这类问计算方式的参数问题，同样构成参数证据。
_PARAM_DIMENSION_RE = re.compile(
    r"单位(?:为|是)[^（(。；;]{0,12}[（(][^）)]{0,12}[）)]"
    r"|依据公式[（(]?\d+[）)]?"
)

# Markdown 表格行。含数字的表格块是"见表X"要求的具体数值化，应视为参数证据。
_TABLE_VALUE_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)


def split_sentences(text: str) -> list[str]:
    sentences = [" ".join(item.split()) for item in SENTENCE_BOUNDARY.split(text)]
    return [item for item in sentences if item]


def lexical_units(text: str) -> set[str]:
    text = normalize_concepts(text)
    units = {item.lower() for item in ASCII_TOKEN.findall(text)}
    for run in HAN_RUN.findall(text):
        units.update(run[index : index + 2] for index in range(max(0, len(run) - 1)))
    return {item for item in units if item and item not in GENERIC_UNITS}


def overlap_score(query: str, text: str) -> float:
    query_units = lexical_units(query)
    if not query_units:
        return 0
    return len(query_units & lexical_units(text)) / len(query_units)


def split_requirements(query: str) -> list[str]:
    parts = re.split(r"(?:以及|并且|同时|和|、|；|;)", query)
    cleaned = []
    for part in parts:
        item = part.strip(" ，,。！？!?")
        item = re.sub(
            r"(?:分别)?(?:是什么|是多少|为多少|有哪些|如何|怎么)$",
            "",
            item,
        ).strip()
        if item and lexical_units(item):
            cleaned.append(item)
    return list(dict.fromkeys(cleaned)) or [query]


def measured_values(text: str) -> dict[str, set[Decimal]]:
    values: dict[str, set[Decimal]] = {}
    for match in MEASUREMENT.finditer(text):
        value = Decimal(match.group("value"))
        unit = match.group("unit").lower()
        if unit in {"毫秒", "ms"}:
            key, normalized = "time_seconds", value / Decimal(1000)
        elif unit in {"秒", "s"}:
            key, normalized = "time_seconds", value
        elif unit in {"分钟", "min"}:
            key, normalized = "time_seconds", value * Decimal(60)
        elif unit in {"小时", "h"}:
            key, normalized = "time_seconds", value * Decimal(3600)
        elif unit == "khz":
            key, normalized = "frequency_hz", value * Decimal(1000)
        elif unit == "mhz":
            key, normalized = "frequency_hz", value * Decimal(1_000_000)
        elif unit == "hz":
            key, normalized = "frequency_hz", value
        else:
            key, normalized = "bytes", value
        values.setdefault(key, set()).add(normalized)
    return values


def has_parameter_value(text: str) -> bool:
    """Return True when text carries a concrete numeric parameter value.

    A parameter value is a number followed by a physical unit (e.g. ``2m~4m``,
    ``1.5 L/667m²``, ``3%``), as opposed to a bare clause number (``3.1 作业
    高度``) or a definition without any number. Used by the evidence gate to
    avoid treating a definition sentence as sufficient coverage for a
    parameter question.

    Also recognizes parameter formula / unit-definition signals (``单位为克每
    公顷 (g/hm^2)``, ``依据公式（1）确定``) and Markdown table blocks (the
    usual carrier of "见表X" concrete values) as parameter evidence.
    """
    if _PARAM_VALUE_RE.search(text):
        return True
    if _PARAM_DIMENSION_RE.search(text):
        return True
    # 含数字的 Markdown 表格块：数据行常为"数值+短词"，表头通常含单位，
    # 是"见表X"要求的具体数值化，应视为参数证据。
    if _TABLE_VALUE_RE.search(text) and re.search(r"\d", text):
        return True
    return False


# 参数类需求词：这类需求期待"数值+单位"的答案（施药量、作业高度等）。
_PARAMETER_TERMS = (
    "量",
    "高度",
    "速度",
    "速率",
    "距离",
    "温度",
    "压力",
    "浓度",
    "面积",
    "体积",
    "剂量",
    "幅宽",
    "喷幅",
    "航高",
    "时长",
    "频率",
    "分辨率",
)


def is_parametric_text(text: str) -> bool:
    """Return True when text asks for a numeric parameter (e.g. 施药量、作业高度)."""
    return any(term in text for term in _PARAMETER_TERMS)
