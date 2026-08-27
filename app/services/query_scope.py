"""查询显式声明的规程类别（问题域）识别与文档范围过滤。

场景：用户问题显式限定"结合 A、B 两类…规程/规范"时（如"结合林业有害生物
飞防、松材线虫病遥感监测两类无人机作业规范"），答案只能基于属于这些类别的
文档生成，不得用其它作物/领域规程（如马铃薯病虫害规范）的内容充当通用作业
要求。此模块只负责"是否触发 + 类别核心词提取 + 文档是否在范围内"的确定性
判定；触发与过滤时机在检索服务（procedure/summary 流程类问题）。
"""

from __future__ import annotations

import re

from app.services.multi_document import document_title

# "结合 X、Y 两类无人机作业规范/规程/标准" 句式。
# items 为两个（或多个）类别短语，如"林业有害生物飞防、松材线虫病遥感监测"。
_SCOPE_QUERY_RE = re.compile(
    r"结合\s*(?P<items>.+?)\s*(?:等\s*)?"
    r"(?:[两二三四五六七八九十各多])\s*类\s*(?:无人机\s*)?"
    r"(?:作业|监测|巡查|巡检|航测|航摄|管理)?\s*(?:规范|规程|标准)"
)

# 类别/主体短语的通用后缀（长后缀在前，避免"智能化巡检"被"巡检"先剥离残留
# "智能化"）。剥离后剩下的才是领域核心词，如"林业有害生物飞防"→"林业有害
# 生物"、"长大桥梁无人机精细化巡检"→"长大桥梁"、"地质灾害倾斜摄影测量"→
# "地质灾害"。
_GENERIC_SUFFIXES = (
    "精细化巡检",
    "智能化巡检",
    "倾斜摄影测量",
    "遥感监测",
    "常态化巡查",
    "倾斜航测",
    "摄影测量",
    "智能巡检",
    "药剂防治",
    "病虫害防治",
    "精细化",
    "智能化",
    "飞防",
    "巡查",
    "巡检",
    "监测",
    "航测",
    "航摄",
    "调查",
    "检测",
    "识别",
    "防治",
    "作业",
    "应用",
    "管理",
    "无人机",
    "低空",
    "智能",
    "自动化",
    "系统",
    "技术",
    "领域",
)

# 核心词最小长度：过短（如"河湖" 2 字也可）时仍允许，但"的/在"等无意义词不触发。
_MIN_CORE_LENGTH = 2


def declared_scope_cores(query: str) -> list[str]:
    """从问题中提取显式声明的规程类别核心词。

    仅在命中"结合…N 类…规范/规程/标准"句式时返回非空列表；其它问题返回
    空列表（不触发域过滤，如普通问题、对比题"对比…"、多跳题"同时使用…"）。
    """
    if not query:
        return []
    match = _SCOPE_QUERY_RE.search(query)
    if match is None:
        return []
    raw_items = re.split(r"[、，,与和及\s]+", match.group("items"))
    cores: list[str] = []
    for item in raw_items:
        item = item.strip()
        if not item:
            continue
        core = _strip_generic_suffixes(item)
        if len(core) >= _MIN_CORE_LENGTH:
            cores.append(core)
    return cores


def _strip_generic_suffixes(phrase: str) -> str:
    """迭代剥离通用后缀，得到领域核心短语。

    例："林业有害生物飞防"→"林业有害生物"；"松材线虫病遥感监测"→"松材线虫病"；
    "河湖智能巡检"→"河湖"。
    """
    current = phrase
    while True:
        stripped = False
        for suffix in _GENERIC_SUFFIXES:
            if current.endswith(suffix) and len(current) - len(suffix) >= _MIN_CORE_LENGTH:
                current = current[: -len(suffix)]
                stripped = True
                break
        if not stripped or len(current) < _MIN_CORE_LENGTH:
            break
    return current


def document_in_scope(document_name: str, cores: list[str]) -> bool:
    """文档标题是否包含任一声明类别核心词（子串匹配）。

    例：核心词"林业有害生物"/"松材线虫病"下，
    《无人机喷洒防治林业有害生物技术规程》与《无人机监测松材线虫病致死松树
    技术规程》在范围内；《基于植保无人机的马铃薯病虫害药剂防治作业规范》不在。
    """
    if not cores or not document_name:
        return False
    title = document_title(document_name)
    if not title:
        return False
    return any(core in title for core in cores)


def subject_core(subject: str) -> str:
    """对比主体短语剥离通用后缀后的核心词（如"长大桥梁无人机精细化巡检"
    →"长大桥梁"，"地质灾害倾斜摄影测量"→"地质灾害"）。"""
    return _strip_generic_suffixes(subject)


def document_belongs_to_subject(document_name: str, subject: str) -> bool:
    """对比主体短语是否属于该文档（用于对比矩阵子查询的候选域过滤）。

    例：主体"长大桥梁无人机精细化巡检"→核心"长大桥梁"，只有
    《长大桥梁无人机巡检作业技术规程》匹配；《水运工程桩位无人机测量导则》
    （仅共享"无人机"）与《地质灾害调查无人机低空摄影测量技术规程》不匹配。
    核心词过短/为空时退化为"标题包含主体核心的任一 2~6 字子串"。
    """
    if not document_name or not subject:
        return False
    title = document_title(document_name)
    if not title:
        return False
    core = _strip_generic_suffixes(subject)
    candidates: set[str] = set()
    if len(core) >= _MIN_CORE_LENGTH:
        candidates.add(core)
        for length in range(2, min(6, len(core)) + 1):
            for index in range(0, len(core) - length + 1):
                candidates.add(core[index : index + length])
    else:
        # 主体整体被剥空（如"无人机精细化巡检"）：用未剥前的 3~6 字子串兜底。
        for length in range(3, min(6, len(subject)) + 1):
            for index in range(0, len(subject) - length + 1):
                candidates.add(subject[index : index + length])
    return any(candidate in title for candidate in candidates)
