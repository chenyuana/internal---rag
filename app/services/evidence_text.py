from __future__ import annotations

import re
from decimal import Decimal

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


def split_sentences(text: str) -> list[str]:
    sentences = [" ".join(item.split()) for item in SENTENCE_BOUNDARY.split(text)]
    return [item for item in sentences if item]


def lexical_units(text: str) -> set[str]:
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
