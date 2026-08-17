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
