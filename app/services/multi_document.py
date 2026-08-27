"""同名/同主题多规程文档的识别与证据分组。

场景：知识库中可能同时存在"名称相同但发布时间/发布单位/适用范围不同"的多份
规程（如茂名市 DB4409/T 60—2024 与广东省 DB44/T 2708—2025，或同名规程的
不同年份版本）。此时系统不自行判断新旧，而是把证据按文档分组，每份规程单独
生成回答后由后端拼接分列，使来源天然清晰。
"""

from __future__ import annotations

import re

from app.schemas.chat import EvidenceSentence

# 书名号标题：《无人机监测松材线虫病致死松树技术规程》
_TITLE_RE = re.compile(r"《([^》]+)》")
# 文件名开头的发布日期前缀，如 "20241217：" / "20250721："
_DATE_PREFIX_RE = re.compile(r"^\d{8}\s*[:：]?\s*")
_PDF_SUFFIX_RE = re.compile(r"\.pdf$", re.IGNORECASE)

# 分列模式要求每组至少的证据句数（避免只有文档头/零散句的文档参与）。
_MIN_EVIDENCE_PER_GROUP = 2


def document_title(document_name: str) -> str:
    """提取文档标题：优先取书名号内内容；否则去掉日期前缀与 .pdf 后缀。"""
    match = _TITLE_RE.search(document_name)
    if match:
        return match.group(1).strip()
    text = _PDF_SUFFIX_RE.sub("", document_name.strip())
    text = _DATE_PREFIX_RE.sub("", text)
    return text.strip()


def titles_similar(left: str, right: str) -> bool:
    """两个标题是否视为"同名/同主题"：相等，或字符集合 Jaccard ≥ 0.7 且长度差在界内。"""
    if not left or not right:
        return False
    if left == right:
        return True
    if len(left) < 3 or len(right) < 3:
        return False
    chars_left = set(left)
    chars_right = set(right)
    union = chars_left | chars_right
    if not union:
        return False
    jaccard = len(chars_left & chars_right) / len(union)
    return jaccard >= 0.7 and abs(len(left) - len(right)) <= max(4, len(left) // 2)


def multi_document_groups(
    evidence: list[EvidenceSentence],
) -> list[tuple[str, list[EvidenceSentence]]]:
    """按 document_name 分组证据；仅当存在 ≥2 份"同名/同主题"文档且每组证据
    充足时返回**参与相似对**的文档分组（无关文档不参与分列），否则返回空列表
    （走默认合并生成路径）。
    """
    by_document: dict[str, list[EvidenceSentence]] = {}
    for item in evidence:
        name = item.document_name or ""
        if not name:
            continue
        by_document.setdefault(name, []).append(item)
    names = list(by_document)
    if len(names) < 2:
        return []
    titles = {name: document_title(name) for name in names}
    similar_names: set[str] = set()
    for index in range(len(names)):
        for other in range(index + 1, len(names)):
            if titles_similar(titles[names[index]], titles[names[other]]):
                similar_names.add(names[index])
                similar_names.add(names[other])
    if not similar_names:
        return []
    groups = [
        (name, by_document[name])
        for name in names
        if name in similar_names and len(by_document[name]) >= _MIN_EVIDENCE_PER_GROUP
    ]
    return groups if len(groups) >= 2 else []
