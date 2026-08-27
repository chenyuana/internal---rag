from __future__ import annotations

from app.schemas.chat import EvidenceAssessment, EvidenceSentence
from app.schemas.evidence import EvidenceRecord
from app.schemas.retrieval import ChunkMetadata, CoverageCell, QueryPlan, SelectedChunk, SubQuery
from app.services.citation_validator import CitationValidator
from app.services.claim_topic_validator import ClaimTopicValidator
from app.services.comparison_matrix_builder import ComparisonMatrixBuilder
from app.services.subject_grounding_validator import SubjectGroundingValidator


def _plan() -> QueryPlan:
    return QueryPlan(
        query_type="comparison",
        subjects=["长大桥梁无人机精细化巡检", "地质灾害倾斜摄影测量"],
        aspects=["设备选型", "像控布设"],
        subqueries=[
            SubQuery(id="q1", query="q1", subject="长大桥梁无人机精细化巡检", aspect="设备选型"),
            SubQuery(id="q2", query="q2", subject="地质灾害倾斜摄影测量", aspect="设备选型"),
            SubQuery(id="q3", query="q3", subject="长大桥梁无人机精细化巡检", aspect="像控布设"),
            SubQuery(id="q4", query="q4", subject="地质灾害倾斜摄影测量", aspect="像控布设"),
        ],
        synthesis_mode="matrix",
    )


def _chunk(chunk_id: str, subquery_id: str, text: str) -> SelectedChunk:
    return SelectedChunk(
        citation_id=f"C{chunk_id[-1]}",
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        dataset_id="kb-1",
        text=text,
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=0.7,
        subquery_id=subquery_id,
    )


def _matrix(cells: list[tuple[str, str, str, str, list[str]]]) -> list[CoverageCell]:
    """cells: (subquery_id, subject, aspect, status, chunk_ids)。"""
    return [
        CoverageCell(
            subquery_id=subquery_id,
            subject=subject,
            aspect=aspect,
            status=status,
            chunk_ids=chunk_ids,
        )
        for subquery_id, subject, aspect, status, chunk_ids in cells
    ]


def test_matrix_builder_fills_cells_from_evidence_and_passes_validators() -> None:
    plan = _plan()
    chunks = [
        _chunk(
            "c1",
            "q1",
            "章节：4 基本要求 / 4.3.2 无人机应满足以下要求\n"
            "a) 无人机续航时间 ≥30 min,标配载荷 ≥2.5 kg;\n"
            "b) 无人机具备六向全方位避障功能,避障距离 ≥2 m;",
        ),
        _chunk(
            "c2",
            "q2",
            "章节：5 作业准备 / 5.5.1 基本要求\n宜选用具备POS系统的测量相机。",
        ),
        _chunk(
            "c4",
            "q4",
            "章节：7 飞行实施 / 7.1.1 像控点布设\n像控点布设以满足空中三角测量为原则。",
        ),
    ]
    matrix = _matrix(
        [
            ("q1", "长大桥梁无人机精细化巡检", "设备选型", "covered", ["c1"]),
            ("q2", "地质灾害倾斜摄影测量", "设备选型", "covered", ["c2"]),
            ("q3", "长大桥梁无人机精细化巡检", "像控布设", "not_specified", []),
            ("q4", "地质灾害倾斜摄影测量", "像控布设", "covered", ["c4"]),
        ]
    )
    assessment = EvidenceAssessment(
        status="PARTIALLY_ANSWERABLE",
        covered_requirements=[
            "长大桥梁无人机精细化巡检：设备选型",
            "地质灾害倾斜摄影测量：设备选型",
            "地质灾害倾斜摄影测量：像控布设",
        ],
        missing_requirements=["长大桥梁无人机精细化巡检：像控布设"],
    )

    answer = ComparisonMatrixBuilder().build(
        plan=plan,
        chunks=chunks,
        assessment=assessment,
        coverage_matrix=matrix,
    )

    assert answer is not None
    assert answer.answerability == "PARTIALLY_ANSWERABLE"
    assert "长大桥梁无人机精细化巡检" in answer.answer
    assert "地质灾害倾斜摄影测量" in answer.answer
    assert "当前资料中未明确说明" in answer.answer  # 桥梁像控布设无证据
    assert len(answer.claims) == 4
    # 桥梁像控布设格无证据 → claim 无引用且含"未明确说明"（保留主体+维度前缀）
    bridge_control = next(
        c for c in answer.claims if "长大桥梁无人机精细化巡检" in c.claim and "像控布设" in c.claim
    )
    assert bridge_control.citation_ids == []
    assert "未明确说明" in bridge_control.claim
    # 有内容格：claim 为内容文本，引用承担主体归属（避免前缀词触发 topic 误拒）
    geo_equip = next(c for c in answer.claims if c.citation_ids == ["C2"])
    assert "具备POS系统的测量相机" in geo_equip.claim
    # 表格行/元数据行不得进入格内容
    assert "| --- |" not in answer.answer.split("设备选型")[1].split("| 像控布设")[0]

    # 关键：确定性产物必须通过既有校验链（citation/scope/grounding/subject/topic）
    evidence = [
        EvidenceSentence(citation_id="C1", chunk_id="c1", text="a) 无人机续航时间 ≥30 min"),
        EvidenceSentence(citation_id="C2", chunk_id="c2", text="宜选用具备POS系统的测量相机"),
        EvidenceSentence(
            citation_id="C4",
            chunk_id="c4",
            text="像控点布设以满足空中三角测量为原则",
        ),
    ]
    CitationValidator(require_citations=True).validate(
        answer,
        evidence=evidence,
        chunks=chunks,
        include_historical=False,
    )
    SubjectGroundingValidator().validate(answer, plan=plan, chunks=chunks)
    ClaimTopicValidator().validate(answer, chunks=chunks)


def test_matrix_builder_returns_none_for_non_comparison() -> None:
    plan = QueryPlan(
        query_type="procedure",
        aspects=["流程"],
        subqueries=[SubQuery(id="q1", query="q1", aspect="流程")],
        synthesis_mode="sequence",
    )
    assessment = EvidenceAssessment(status="ANSWERABLE", covered_requirements=["流程"])

    result = ComparisonMatrixBuilder().build(
        plan=plan,
        chunks=[],
        assessment=assessment,
        coverage_matrix=[],
    )

    assert result is None


def test_matrix_builder_diagonal_layout_for_multi_hop() -> None:
    """multi_hop"结合 A 与 B…兼顾 X 与 Y"是对角线布局：A→X、B→Y 一一对应。

    全交叉遍历会生成 (A, Y)/(B, X) 两个不存在的格子并误报缺失；对角线模式
    必须只渲染存在的 (subject, aspect) 对。
    """
    plan = QueryPlan(
        query_type="multi_hop",
        subjects=["沙地无人机播种治沙作业", "无人机通用服务管理要求"],
        aspects=["技术参数", "运营管理"],
        subqueries=[
            SubQuery(
                id="q1",
                query="q1",
                subject="沙地无人机播种治沙作业",
                aspect="技术参数",
            ),
            SubQuery(
                id="q2",
                query="q2",
                subject="无人机通用服务管理要求",
                aspect="运营管理",
            ),
        ],
        synthesis_mode="matrix",
    )
    chunks = [
        _chunk(
            "c1",
            "q1",
            "章节：5 无人机播种设计 / 5.5 航高和速度设计\n"
            "航高10m～50m，速度3m/s～8m/s。",
        ),
        _chunk(
            "c2",
            "q2",
            "章节：5 机构要求 / 5.4 运营管理制度\n应建立完善的运营管理制度。",
        ),
    ]
    matrix = _matrix(
        [
            ("q1", "沙地无人机播种治沙作业", "技术参数", "covered", ["c1"]),
            ("q2", "无人机通用服务管理要求", "运营管理", "covered", ["c2"]),
        ]
    )
    assessment = EvidenceAssessment(
        status="ANSWERABLE",
        covered_requirements=[
            "沙地无人机播种治沙作业：技术参数",
            "无人机通用服务管理要求：运营管理",
        ],
    )

    answer = ComparisonMatrixBuilder().build(
        plan=plan,
        chunks=chunks,
        assessment=assessment,
        coverage_matrix=matrix,
    )

    assert answer is not None
    assert answer.answerability == "ANSWERABLE"
    assert len(answer.claims) == 2
    # 两个对角线格都有内容且引用正确；不存在 (沙地, 运营管理) 交叉格。
    assert "航高10m～50m" in answer.answer
    assert "运营管理制度" in answer.answer
    assert answer.claims[0].citation_ids == ["C1"]
    assert answer.claims[1].citation_ids == ["C2"]
    assert "当前资料中未明确说明" not in answer.answer
    assert "沙地无人机播种治沙作业 技术参数" in answer.answer
    assert "无人机通用服务管理要求 运营管理" in answer.answer


def test_cell_content_round_robins_across_acceptance_sources() -> None:
    def record(index: int, citation_id: str, text: str) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=f"q1:c{index}:1",
            cell_id="q1",
            subject="地质灾害倾斜摄影测量",
            aspect="成果验收",
            document_id="doc-1",
            document_name="规程",
            requirement_type="quality_check",
            requirement_text=text,
            citation_id=citation_id,
            retrieval_score=0.8,
            confidence=0.75,
        )

    content, citations = ComparisonMatrixBuilder._cell_content(
        aspect="成果验收",
        cell_chunks=[],
        records=[
            record(1, "C1", "表3 第一行"),
            record(2, "C1", "表3 第二行"),
            record(3, "C1", "表3 第三行"),
            record(4, "C2", "表2 第一行"),
            record(5, "C3", "表4 第一行"),
        ],
    )

    assert citations == ["C1", "C2", "C3"]
    assert content.split("；")[:3] == [
        "表3 第一行[C1]",
        "表2 第一行[C2]",
        "表4 第一行[C3]",
    ]


def test_precision_metric_rejects_report_inventory_and_keeps_real_metric() -> None:
    def record(index: int, text: str) -> EvidenceRecord:
        return EvidenceRecord(
            evidence_id=f"q1:c{index}:1",
            cell_id="q1",
            subject="植被多光谱遥感无人机",
            aspect="精度评价指标",
            document_id="doc-1",
            document_name="规程",
            requirement_type="quality_check",
            requirement_text=text,
            citation_id=f"C{index}",
            retrieval_score=0.8,
            confidence=0.75,
        )

    content, citations = ComparisonMatrixBuilder._cell_content(
        aspect="精度评价指标",
        cell_chunks=[],
        records=[
            record(1, "报告内容包括模型选择、精度评价和面积统计等信息。"),
            record(2, "模型精度采用决定系数R²和均方根误差RMSE评价。"),
        ],
    )

    assert "报告内容包括" not in content
    assert "R²" in content
    assert citations == ["C2"]


def test_three_column_comparison_table_has_separate_headers() -> None:
    rendered = ComparisonMatrixBuilder._render_table(
        ["植被多光谱遥感无人机", "普通可见光巡检无人机"],
        [("成像波段", ["多光谱", "可见光"])],
    )

    assert rendered.splitlines() == [
        "| 对比项目 | 植被多光谱遥感无人机 | 普通可见光巡检无人机 |",
        "| --- | --- | --- |",
        "| 成像波段 | 多光谱 | 可见光 |",
    ]
