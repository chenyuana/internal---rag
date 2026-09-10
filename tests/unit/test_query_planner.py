from app.services.query_analyzer import QueryAnalyzer
from app.services.query_plan_adapter import QueryPlanAdapter
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


def test_comparison_two_sides_plan_splits_subjects_aspects_and_reason() -> None:
    """对比A、B，二者X、Y有哪些差异？原因？ 应拆成主体×维度矩阵 + 原因子查询。"""
    query = QueryAnalyzer().analyze(
        "对比林地病虫害喷洒无人机、水稻植保无人机，"
        "二者亩均施药量、作业高度有哪些明显差异？差异形成的原因是什么？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.subjects == ["林地病虫害喷洒无人机", "水稻植保无人机"]
    assert plan.aspects == ["亩均施药量", "作业高度", "差异形成原因"]
    assert plan.synthesis_mode == "matrix"
    assert plan.requires_multi_query is True
    assert [(item.subject, item.aspect) for item in plan.subqueries] == [
        ("林地病虫害喷洒无人机", "亩均施药量"),
        ("水稻植保无人机", "亩均施药量"),
        ("林地病虫害喷洒无人机", "作业高度"),
        ("水稻植保无人机", "作业高度"),
        ("林地病虫害喷洒无人机、水稻植保无人机", "差异形成原因"),
    ]
    # 主体词应扩展为规范术语，帮助检索命中含参数的专业文档。
    queries = {item.id: item.query for item in plan.subqueries}
    assert "林业 有害生物 喷洒 防治" in queries["q1"]
    assert "水稻 病虫害 施药 稻纵卷叶螟" in queries["q2"]
    # 合并后的原因子查询不做主体扩展。
    assert queries["q5"] == "林地病虫害喷洒无人机 水稻植保无人机 差异形成原因"

    adapted = QueryPlanAdapter.from_legacy(query, plan)
    adapted_queries = {cell.id: cell.original_query for cell in adapted.cells}
    assert "林业 有害生物 喷洒 防治" in adapted_queries["q1"]
    assert "水稻 病虫害 施药 稻纵卷叶螟" in adapted_queries["q2"]


def test_comparison_two_sides_without_reason_keeps_pure_matrix() -> None:
    query = QueryAnalyzer().analyze(
        "比较A型无人机和B型无人机，两者作业速度、续航时间有何差异？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.subjects == ["A型无人机", "B型无人机"]
    assert plan.aspects == ["作业速度", "续航时间"]
    assert len(plan.subqueries) == 4


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
        "无人机通用服务管理要求 运营管理 管理制度 服务要求 现场勘察 设备匹配 "
        "制定方案 服务实施 成果提交",
    ]
    # 每个子查询必须带声明主体，检索层才能走"主体发现 → 文档解析 → 按文档
    # 限定检索"，把无关文档（水稻水直播/低空旅游/CCAR-92 等）挡在候选之外。
    assert plan.subjects == ["沙地无人机播种治沙作业", "无人机通用服务管理要求"]
    assert [(item.subject, item.aspect) for item in plan.subqueries] == [
        ("沙地无人机播种治沙作业", "技术参数"),
        ("无人机通用服务管理要求", "运营管理"),
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


def test_cross_document_plan_leading_ju_he_uses_first_source_as_subject() -> None:
    """"结合"位于句首时，第一个来源本身充当 q1 主体，不得退化成单查询。"""
    query = QueryAnalyzer().analyze(
        "结合无人机播种治沙技术规程与无人机应用服务通用规范，"
        "项目实施在技术参数与运营管理上有什么不同要求？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.requires_multi_query is True
    assert plan.subjects == ["无人机播种治沙技术规程", "无人机应用服务通用规范"]
    assert [(item.subject, item.aspect) for item in plan.subqueries] == [
        ("无人机播种治沙技术规程", "无人机播种治沙技术规程"),
        ("无人机应用服务通用规范", "无人机应用服务通用规范"),
    ]


def test_comparison_two_sides_with_exist_marker_keeps_dimensions() -> None:
    """"存在哪些区别"里的"在"不得被 _first_marker_index 误判为比较标记。

    "对比A、B，二者X、Y存在哪些区别" 应拆成 2 主体×2 维度，维度词进入子查询，
    而不是把"二者作业航高""播种量计算逻辑存"当主体、aspect 只剩"哪些区别"。
    """
    query = QueryAnalyzer().analyze(
        "对比沙地无人机播种、早熟中稻无人机直播，"
        "二者作业航高、播种量计算逻辑存在哪些区别？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.subjects == ["沙地无人机播种", "早熟中稻无人机直播"]
    assert plan.aspects == ["作业航高", "播种量计算逻辑"]
    assert plan.synthesis_mode == "matrix"
    assert [(item.subject, item.aspect) for item in plan.subqueries] == [
        ("沙地无人机播种", "作业航高"),
        ("早熟中稻无人机直播", "作业航高"),
        ("沙地无人机播种", "播种量计算逻辑"),
        ("早熟中稻无人机直播", "播种量计算逻辑"),
    ]
    assert [item.query for item in plan.subqueries] == [
        "沙地无人机播种 作业航高 航高 速度",
        "早熟中稻无人机直播 作业航高 航高 速度",
        "沙地无人机播种 播种量计算逻辑 播种量",
        "早熟中稻无人机直播 播种量计算逻辑 播种量",
    ]


def test_comparison_two_sides_strips_leading_marker_from_aspect() -> None:
    """对比A、B，二者在X、Y上… 的 aspect 不应残留开头"在"。"""
    query = QueryAnalyzer().analyze(
        "对比甲型无人机、乙型无人机，二者在作业速度、续航时间上有什么不同？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.subjects == ["甲型无人机", "乙型无人机"]
    assert plan.aspects == ["作业速度", "续航时间"]
    assert plan.subqueries[0].query == "甲型无人机 作业速度"


def test_comparison_ground_resolution_aspect_cleaned_and_expanded() -> None:
    """"上有什么差异"后缀必须剥离，地面分辨率维度带表1/比例尺扩展词。"""
    query = QueryAnalyzer().analyze(
        "对比地质灾害倾斜航测、河湖智能巡检无人机，"
        "在航向/旁向重叠度、地面分辨率要求上有什么差异？"
    )

    plan = QueryPlanner().plan(query)

    assert plan.subjects == ["地质灾害倾斜航测", "河湖智能巡检无人机"]
    assert plan.aspects == ["航向/旁向重叠度", "地面分辨率要求"]
    q3 = plan.subqueries[2]
    assert q3.subject == "地质灾害倾斜航测"
    assert q3.aspect == "地面分辨率要求"
    # 维度扩展把表格块检索词（表1/比例尺）带进子查询。
    assert "表1" in q3.query and "比例尺" in q3.query
    q4 = plan.subqueries[3]
    assert q4.subject == "河湖智能巡检无人机"
    assert "表1" in q4.query


def test_comparison_aspect_fang_mian_suffix_stripped() -> None:
    """"方面有什么区别"后缀必须剥离。"""
    query = QueryAnalyzer().analyze("对比A型、B型无人机，在作业速度方面有什么区别？")

    plan = QueryPlanner().plan(query)

    assert "作业速度" in plan.aspects
    assert plan.subqueries[0].query.startswith("A型")


def test_multi_dimension_synergy_question_splits_aspects() -> None:
    """"…X、Y、Z 三类要求如何协同匹配"应识别为多维度综合题，按方面拆子查询。"""
    query = QueryAnalyzer().analyze(
        "同时使用 5G 低空基站与自动机巢开展河湖常态化无人机巡查，"
        "空域通信、机巢运维、数据归档三类要求如何协同匹配？"
    )

    assert query.query_type == "multi_hop"
    plan = QueryPlanner().plan(query)

    assert plan is not None
    assert plan.aspects == ["空域通信", "机巢运维", "数据归档"]
    assert len(plan.subqueries) == 3
    # 子查询必须携带完整上下文（主体/场景词），让机巢/河湖文档有机会被召回。
    assert "5G 低空基站" in plan.subqueries[0].query
    assert "自动机巢" in plan.subqueries[0].query
    assert "河湖" in plan.subqueries[0].query
    assert plan.subqueries[0].query.endswith("空域通信")
    assert plan.subqueries[1].query.endswith("机巢运维")
    assert plan.subqueries[2].query.endswith("数据归档")


def test_two_category_question_splits_aspects() -> None:
    query = QueryAnalyzer().analyze(
        "开展无人机植保作业，飞行安全、药液使用两类要求如何协同？"
    )

    assert query.query_type == "multi_hop"
    plan = QueryPlanner().plan(query)

    assert plan is not None
    assert plan.aspects == ["飞行安全", "药液使用"]


def test_comparison_three_stage_question_cleans_count_tail() -> None:
    """"…在X、Y、Z三个环节的标准要求如何统筹区分"：数量词尾巴不污染维度，
    应按 2 主体 × 3 维度拆成 6 个矩阵子查询。"""
    query = QueryAnalyzer().analyze(
        "长大桥梁无人机精细化巡检与地质灾害倾斜摄影测量，"
        "在设备选型、像控布设、成果验收三个环节的标准要求如何统筹区分"
    )

    assert query.query_type == "comparison"
    plan = QueryPlanner().plan(query)

    assert plan is not None
    assert plan.subjects == ["长大桥梁无人机精细化巡检", "地质灾害倾斜摄影测量"]
    assert plan.aspects == ["设备选型", "像控布设", "成果验收"]
    assert plan.synthesis_mode == "matrix"
    assert len(plan.subqueries) == 6
    assert [(item.subject, item.aspect) for item in plan.subqueries] == [
        ("长大桥梁无人机精细化巡检", "设备选型"),
        ("地质灾害倾斜摄影测量", "设备选型"),
        ("长大桥梁无人机精细化巡检", "像控布设"),
        ("地质灾害倾斜摄影测量", "像控布设"),
        ("长大桥梁无人机精细化巡检", "成果验收"),
        ("地质灾害倾斜摄影测量", "成果验收"),
    ]


def test_comparison_single_dimension_tongchou_qufen_tail_stripped() -> None:
    """无数量词的"…上如何统筹区分"尾巴也要剥离（走 _ASPECT_SUFFIXES）。"""
    query = QueryAnalyzer().analyze(
        "对比A型、B型巡检无人机，在设备选型上如何统筹区分？"
    )

    plan = QueryPlanner().plan(query)

    assert plan is not None
    assert plan.aspects == ["设备选型"]


def test_commenter_enumeration_uses_document_scan_queries() -> None:
    query = QueryAnalyzer().analyze(
        "FAA 23-62规则的主要评论方有哪些类型？请列举具体机构。"
    )

    plan = QueryPlanner().plan(query)

    assert plan.query_type == "enumeration"
    assert plan.expansion_mode == "document"
    assert plan.synthesis_mode == "map_reduce"
    assert len(plan.subqueries) == 3
    assert [item.id for item in plan.subqueries] == ["q1", "q2", "q3"]
    assert "received comments from" in plan.subqueries[2].query
    assert "objected" in plan.subqueries[2].query
