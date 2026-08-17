from app.services.query_analyzer import QueryAnalyzer
from app.services.query_planner import QueryPlanner


def test_comparison_plan_builds_subject_aspect_matrix() -> None:
    query = QueryAnalyzer().analyze(
        "对比植被多光谱遥感无人机、普通可见光巡检无人机，"
        "在成像波段、作业最佳时段、精度评价指标上有什么不同？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.subjects == ["植被多光谱遥感无人机", "普通可见光巡检无人机"]
    assert plan.aspects == ["成像波段", "作业最佳时段", "精度评价指标"]
    assert len(plan.subqueries) == 6
    assert plan.synthesis_mode == "matrix"
    assert [(item.subject, item.aspect) for item in plan.subqueries] == [
        ("植被多光谱遥感无人机", "成像波段"),
        ("普通可见光巡检无人机", "成像波段"),
        ("植被多光谱遥感无人机", "作业最佳时段"),
        ("普通可见光巡检无人机", "作业最佳时段"),
        ("植被多光谱遥感无人机", "精度评价指标"),
        ("普通可见光巡检无人机", "精度评价指标"),
    ]


def test_fact_plan_stays_on_single_query_path() -> None:
    query = QueryAnalyzer().analyze("第21.1条的内容是什么？")

    plan = QueryPlanner().plan(query)

    assert plan.requires_multi_query is False
    assert plan.subqueries[0].query == query.normalized_query


def test_cross_document_plan_splits_into_two_source_dimensions() -> None:
    """跨文档综合题应拆成"主体+维度1"与"来源2+维度2"两个子查询。"""
    query = QueryAnalyzer().analyze(
        "沙地无人机播种治沙作业，结合播种技术规范与无人机通用服务管理要求，"
        "完整项目实施需要兼顾技术参数与运营管理哪些双重要求？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.requires_multi_query is True
    assert plan.synthesis_mode == "matrix"
    assert plan.aspects == ["技术参数", "运营管理"]
    assert [item.query for item in plan.subqueries] == [
        "沙地无人机播种治沙作业 技术参数 播种量 航高 速度 种子处理 整地 作业设计",
        "无人机通用服务管理要求 运营管理 管理制度 服务要求 现场勘察 设备匹配 制定方案 服务实施 成果提交",
    ]


def test_cross_document_plan_expands_broad_dimensions_only() -> None:
    """只有词表里的宽维度才被扩展；未收录维度保持原样，避免污染。"""
    query = QueryAnalyzer().analyze(
        "无人机作业，结合技术规范与管理要求，完整实施需要兼顾安全性与经济性哪些双重要求？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.requires_multi_query is True
    assert [item.query for item in plan.subqueries] == [
        "无人机作业 安全性",
        "管理要求 经济性",
    ]
