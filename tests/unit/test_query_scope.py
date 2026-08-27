from __future__ import annotations

from app.services.query_scope import (
    declared_scope_cores,
    document_belongs_to_subject,
    document_in_scope,
)


def test_declared_scope_cores_extracts_two_categories() -> None:
    query = (
        "结合林业有害生物飞防、松材线虫病遥感监测两类无人机作业规范，"
        "如何搭配完成林地病害识别-药剂防治全链条作业？"
    )
    assert declared_scope_cores(query) == ["林业有害生物", "松材线虫病"]


def test_declared_scope_cores_strips_various_generic_suffixes() -> None:
    assert declared_scope_cores(
        "结合地质灾害航测、河湖智能巡检两类无人机作业规范，如何组织？"
    ) == ["地质灾害", "河湖"]
    assert declared_scope_cores(
        "结合松材线虫病遥感监测与林业有害生物飞防两类规程，如何配合？"
    ) == ["松材线虫病", "林业有害生物"]


def test_declared_scope_cores_no_trigger_without_pattern() -> None:
    # 对比题："对比…" 开头，不触发
    assert (
        declared_scope_cores(
            "对比地质灾害倾斜航测、河湖智能巡检无人机，在航向/旁向重叠度、"
            "地面分辨率要求上有什么差异？"
        )
        == []
    )
    # 多跳题："同时使用…" 开头，不触发
    assert (
        declared_scope_cores(
            "同时使用 5G 低空基站与自动机巢开展河湖常态化无人机巡查，"
            "空域通信、机巢运维、数据归档三类要求如何协同匹配？"
        )
        == []
    )
    # 普通流程题：无"结合…N 类…规范"句式，不触发
    assert (
        declared_scope_cores(
            "采用无人机开展松材线虫病监测，从航摄规划到成果归档"
            "完整操作流程包含哪些关键环节？"
        )
        == []
    )
    assert declared_scope_cores("任务周期是多少？") == []
    assert declared_scope_cores("") == []


def test_document_in_scope_matches_category_cores() -> None:
    cores = ["林业有害生物", "松材线虫病"]
    assert (
        document_in_scope(
            "20240705：《无人机喷洒防治林业有害生物技术规程》.pdf",
            cores,
        )
        is True
    )
    assert (
        document_in_scope(
            "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf",
            cores,
        )
        is True
    )
    # 马铃薯是作物病虫害规范，不在"林业有害生物/松材线虫病"范围内
    assert (
        document_in_scope(
            "20250122：《基于植保无人机的马铃薯病虫害药剂防治作业规范》.pdf",
            cores,
        )
        is False
    )
    assert document_in_scope("", cores) is False
    assert document_in_scope("20250122：《…》.pdf", []) is False


def test_document_in_scope_without_title_brackets() -> None:
    # 无书名号时按去除日期前缀与 .pdf 后缀后的标题匹配
    assert (
        document_in_scope("20240705：无人机喷洒防治林业有害生物技术规程.pdf", ["林业有害生物"])
        is True
    )


def test_document_belongs_to_subject_matches_only_its_own_document() -> None:
    bridge = "20240124：《长大桥梁无人机巡检作业技术规程》.pdf"
    geo = "20230606：《地质灾害调查无人机低空摄影测量技术规程》.pdf"
    water = "20240719：《水运工程桩位无人机测量导则》.pdf"

    # 桥梁主体只匹配桥梁文档；水运导则仅共享"无人机"不匹配；地灾不匹配
    assert document_belongs_to_subject(bridge, "长大桥梁无人机精细化巡检") is True
    assert document_belongs_to_subject(geo, "长大桥梁无人机精细化巡检") is False
    assert document_belongs_to_subject(water, "长大桥梁无人机精细化巡检") is False

    # 地灾主体只匹配地灾文档（倾斜摄影测量被剥离为"地质灾害"）
    assert document_belongs_to_subject(geo, "地质灾害倾斜摄影测量") is True
    assert document_belongs_to_subject(bridge, "地质灾害倾斜摄影测量") is False
    assert document_belongs_to_subject(water, "地质灾害倾斜摄影测量") is False


def test_document_belongs_to_subject_fallback_substrings() -> None:
    # 核心过短的兜底：主体"水稻植保无人机"核心"水稻植保"，子串"水稻"命中
    assert (
        document_belongs_to_subject(
            "20250228：《水稻病虫害无人机施药技术规程》.pdf",
            "水稻植保无人机",
        )
        is True
    )
    assert document_belongs_to_subject("", "长大桥梁") is False
    assert (
        document_belongs_to_subject(
            "20240124：《长大桥梁无人机巡检作业技术规程》.pdf",
            "",
        )
        is False
    )
