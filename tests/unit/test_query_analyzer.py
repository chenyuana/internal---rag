from __future__ import annotations

from app.services.query_analyzer import QueryAnalyzer


def test_query_normalization_preserves_technical_symbols() -> None:
    result = QueryAnalyzer().analyze("  HEALTH_MONITOR_PERIOD_MS   和  0xAF  分别是什么？ ")

    assert result.normalized_query == "HEALTH_MONITOR_PERIOD_MS 和 0xAF 分别是什么？"
    assert result.exact_tokens == ["HEALTH_MONITOR_PERIOD_MS", "0xAF"]
    assert result.query_type == "multi_hop"


def test_query_analyzer_extracts_exact_regulation_article_aliases() -> None:
    result = QueryAnalyzer().analyze("第21.1条1的内容是什么？")

    assert result.article_ids == ["21.1(1)"]
    assert "21.1(1)" in result.exact_tokens
    assert "第21.1条1" in result.article_aliases
    assert "第21.1条第1款" in result.article_aliases


def test_cross_document_dual_requirement_is_multi_hop() -> None:
    """跨文档综合题（结合 A 与 B + 兼顾 X 与 Y 双重要求）应判为 multi_hop，
    而不是被误判为单次检索的 fact。"""
    result = QueryAnalyzer().analyze(
        "沙地无人机播种治沙作业，结合播种技术规范与无人机通用服务管理要求，"
        "完整项目实施需要兼顾技术参数与运营管理哪些双重要求？"
    )
    assert result.query_type == "multi_hop"


def test_dual_requirement_marker_is_multi_hop() -> None:
    result = QueryAnalyzer().analyze("项目实施需要兼顾安全与效率两方面要求？")
    assert result.query_type == "multi_hop"


def test_tongchou_qufen_three_stage_question_is_comparison() -> None:
    """"…三个环节的标准要求如何统筹区分"是两两比较题，不得被"如何"误判为 procedure。"""
    result = QueryAnalyzer().analyze(
        "长大桥梁无人机精细化巡检与地质灾害倾斜摄影测量，"
        "在设备选型、像控布设、成果验收三个环节的标准要求如何统筹区分"
    )
    assert result.query_type == "comparison"


def test_qubie_and_tongyi_are_comparison() -> None:
    assert QueryAnalyzer().analyze("A型无人机和B型无人机有什么区别？").query_type == "comparison"
    assert QueryAnalyzer().analyze("两种巡检方式的异同有哪些？").query_type == "comparison"


def test_inventory_questions_are_detected_without_changing_query_type() -> None:
    """枚举/计数型问题标记 requires_inventory，但 query_type 保持语义分类。"""
    version = QueryAnalyzer().analyze("CCAR-25 有哪些版本？")
    assert version.requires_inventory is True
    assert version.inventory_reason is not None
    assert version.query_type == "fact"

    rounds = QueryAnalyzer().analyze("测试流程一共几轮？")
    assert rounds.requires_inventory is True

    files = QueryAnalyzer().analyze("全部文件清单有哪些？")
    assert files.requires_inventory is True


def test_semantic_questions_are_not_marked_as_inventory() -> None:
    assert QueryAnalyzer().analyze("系统有哪些功能？").requires_inventory is False
    assert QueryAnalyzer().analyze("这个功能怎么用？").requires_inventory is False
    assert QueryAnalyzer().analyze("如何部署这个系统？").requires_inventory is False
