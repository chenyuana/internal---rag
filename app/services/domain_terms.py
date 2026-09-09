"""Bilingual concepts, never document IDs, clause mappings or expected answers."""
from __future__ import annotations

import re

CONCEPTS: tuple[tuple[str, ...], ...] = (
    ("stall speed", "失速速度"),
    ("takeoff warning system", "起飞警告系统"),
    ("static directional stability", "静态方向稳定性", "静方向稳定性"),
    ("dynamic stability", "动态稳定性"),
    ("climb gradient", "爬升梯度"),
    ("emergency exit", "应急出口", "紧急出口"),
    ("thermal acoustic insulation", "热声绝缘", "隔热隔音"),
    ("oxygen equipment", "供氧设备"),
    ("power supply", "电源"),
    ("battery", "batteries", "蓄电池", "电池"),
    ("electronic equipment", "电子设备"),
    ("ignition system", "点火系统"),
    ("brakes", "brake", "刹车", "制动器"),
    ("pressurized cabin", "pressurised cabin", "增压舱"),
    ("inertia load", "惯性载荷"),
    ("high altitude", "高海拔", "高空"),
    ("failure condition", "故障状态"),
    ("special conditions", "特殊条件", "专用条件"),
    ("regulatory analysis", "regulatory analyses", "监管分析"),
    ("benefits", "收益", "效益"),
    ("costs", "成本"),
    ("effective date", "生效日期"),
    ("publication date", "发布日期"),
    ("substantive comments", "实质性评论", "实质性意见"),
    ("acceptance criteria", "通过标准", "验收标准", "合格判据"),
    ("test method", "试验方法", "测试方法"),
    ("at least", "not less than", "no less than", "至少", "不少于", "不低于"),
    ("at most", "not more than", "no more than", "至多", "不超过", "不大于"),
    ("greater than", "more than", "大于", "超过"),
    ("less than", "小于", "低于"),
)


def _pattern(term: str) -> str:
    escaped = re.escape(term).replace(r"\ ", r"[\s-]+")
    return rf"(?<![A-Za-z]){escaped}(?![A-Za-z])" if term.isascii() else escaped


_LOOKUP = {term.casefold(): i for i, terms in enumerate(CONCEPTS) for term in terms}
_PATTERN = re.compile("|".join(_pattern(t) for t in sorted(_LOOKUP, key=len, reverse=True)), re.I)


def normalize_concepts(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        term = re.sub(r"[\s-]+", " ", match[0]).casefold()
        return f" concept_{_LOOKUP[term]} "
    return _PATTERN.sub(replace, text)


def english_expansion(text: str) -> str:
    ids = {int(value) for value in re.findall(r"concept_(\d+)", normalize_concepts(text))}
    return " ".join(CONCEPTS[index][0] for index in sorted(ids))
