from app.schemas.retrieval import ChunkMetadata, QueryPlan, RetrievedChunk, SubQuery
from app.services.coverage_selector import CoverageSelector


def _chunk(
    chunk_id: str,
    document_id: str,
    score: float,
    text: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        dataset_id="kb-1",
        text=text or chunk_id,
        metadata=ChunkMetadata(document_name=f"{document_id}.pdf"),
        hybrid_score=score,
        rerank_score=score,
    )


def test_selector_reserves_slots_for_required_cells_and_deduplicates() -> None:
    plan = QueryPlan(
        query_type="comparison",
        subjects=["A", "B"],
        aspects=["波段", "时段"],
        subqueries=[
            SubQuery(id="q1", query="A 波段", subject="A", aspect="波段"),
            SubQuery(id="q2", query="B 波段", subject="B", aspect="波段"),
            SubQuery(id="q3", query="A 时段", subject="A", aspect="时段"),
            SubQuery(id="q4", query="B 时段", subject="B", aspect="时段"),
        ],
        synthesis_mode="matrix",
    )
    shared = _chunk("shared", "doc-a", 0.99, text="该设备支持多波段成像。")
    results = {
        "q1": [shared, _chunk("a-band", "doc-a", 0.8, text="a波段为近红外波段。")],
        "q2": [shared, _chunk("b-band", "doc-b", 0.7, text="b波段为可见光波段。")],
        "q3": [_chunk("a-time", "doc-a", 0.6, text="最佳作业时段为指定物候期。")],
        "q4": [],
    }

    selection = CoverageSelector().select(plan, results, limit=4)

    assert len({item.chunk_id for item in selection.chunks}) == len(selection.chunks)
    assert {item.chunk_id for item in selection.chunks} == {
        "shared",
        "b-band",
        "a-time",
        "a-band",
    }
    status = {cell.subquery_id: cell.status for cell in selection.matrix}
    assert status == {
        "q1": "covered",
        "q2": "covered",
        "q3": "covered",
        "q4": "not_specified",
    }


def test_selector_attributes_subqueries_and_prefers_subject_documents() -> None:
    """对比矩阵下：记录 chunk 归属子查询；第二轮先占主体文档，无关高分文档不得抢占名额。"""
    plan = QueryPlan(
        query_type="comparison",
        subjects=["A", "B"],
        aspects=["波段"],
        subqueries=[
            SubQuery(id="q1", query="A 波段", subject="A", aspect="波段"),
            SubQuery(id="q2", query="B 波段", subject="B", aspect="波段"),
        ],
        synthesis_mode="matrix",
    )
    results = {
        "q1": [
            _chunk("a1", "doc-a", 0.9, text="a1 波段覆盖近红外。"),
            _chunk("a2", "doc-a", 0.4, text="a2 波段覆盖可见光。"),
        ],
        "q2": [
            _chunk("b1", "doc-b", 0.8, text="b1 波段覆盖多光谱。"),
            _chunk("b2", "doc-b", 0.3, text="b2 波段覆盖热红外。"),
        ],
        # 无关文档的高分候选：对比矩阵下不应抢占主体文档名额。
        "qX": [_chunk("x1", "doc-x", 0.95, text="x1 波段无关内容。")],
    }

    selection = CoverageSelector().select(plan, results, limit=4)

    assert {item.chunk_id for item in selection.chunks} == {"a1", "a2", "b1", "b2"}
    assert selection.chunk_subqueries == {"a1": "q1", "b1": "q2", "a2": "q1", "b2": "q2"}


def test_selector_keeps_new_document_bonus_outside_comparison() -> None:
    """非对比矩阵保持原逻辑：新文档候选可获得小加分进入证据。"""
    plan = QueryPlan(
        query_type="multi_hop",
        subjects=[],
        aspects=["参数"],
        subqueries=[
            SubQuery(id="q1", query="参数", subject="", aspect="参数"),
        ],
        synthesis_mode="matrix",
    )
    results = {
        "q1": [
            _chunk("a1", "doc-a", 0.9, text="a1 参数要求。"),
            _chunk("b1", "doc-b", 0.88, text="b1 参数要求。"),
        ],
    }

    selection = CoverageSelector().select(plan, results, limit=2)

    assert {item.chunk_id for item in selection.chunks} == {"a1", "b1"}


def test_multi_hop_selector_guarantees_document_diversity() -> None:
    """multi_hop 矩阵：候选池中出现过的文档都至少进 1 个代表，避免单一
    关键词文档垄断名额（如 5G 基站文档挤掉机巢/河湖文档）。"""
    plan = QueryPlan(
        query_type="multi_hop",
        subjects=[],
        aspects=["空域通信", "机巢运维", "数据归档"],
        subqueries=[
            SubQuery(id="q1", query="5G基站 空域通信", subject="", aspect="空域通信"),
            SubQuery(id="q2", query="自动机巢 机巢运维", subject="", aspect="机巢运维"),
            SubQuery(id="q3", query="河湖巡查 数据归档", subject="", aspect="数据归档"),
        ],
        synthesis_mode="matrix",
    )
    results = {
        "q1": [
            _chunk("g1", "doc-5g", 0.90),
            _chunk("g2", "doc-5g", 0.89),
            _chunk("g3", "doc-5g", 0.88),
        ],
        "q2": [_chunk("d1", "doc-5g", 0.87), _chunk("n1", "doc-dock", 0.80)],
        "q3": [_chunk("g4", "doc-5g", 0.86), _chunk("l1", "doc-lake", 0.60)],
    }

    selection = CoverageSelector().select(plan, results, limit=8)

    ids = {item.chunk_id for item in selection.chunks}
    assert "n1" in ids, "机巢文档必须有代表进入证据"
    assert "l1" in ids, "河湖文档必须有代表进入证据"
    assert "g1" in ids


def test_selector_prefers_acceptance_table_over_toc_and_deliverables() -> None:
    plan = QueryPlan(
        query_type="comparison",
        subjects=["地质灾害倾斜摄影测量", "长大桥梁无人机精细化巡检"],
        aspects=["成果验收"],
        subqueries=[
            SubQuery(
                id="q1",
                query="地质灾害倾斜摄影测量成果验收标准要求",
                subject="地质灾害倾斜摄影测量",
                aspect="成果验收",
            ),
        ],
        synthesis_mode="matrix",
    )
    results = {
        "q1": [
            _chunk("toc", "doc-disaster", 0.95, text="8 成果检查 9 成果整理 10 成果提交"),
            _chunk(
                "deliverables",
                "doc-disaster",
                0.90,
                text="成果清单（见附录A.5）；测量技术成果报告按CH/T 1001执行。",
            ),
            _chunk(
                "table-3",
                "doc-disaster",
                0.72,
                text=(
                    "<table><caption>表3 无人机航测成果检查点平面位置及高程精度要求</caption>"
                    "<tr><th>项目</th><th>限差</th></tr>"
                    "<tr><td>平面位置精度</td><td>0.20 m</td></tr></table>"
                ),
            ),
        ]
    }

    selection = CoverageSelector().select(plan, results, limit=1)

    assert [item.chunk_id for item in selection.chunks] == ["table-3"]
    assert selection.matrix[0].status == "covered"


def test_acceptance_target_uses_distinct_tables_not_duplicate_versions() -> None:
    plan = QueryPlan(
        query_type="comparison",
        subjects=["地质灾害倾斜摄影测量", "另一主体"],
        aspects=["成果验收"],
        subqueries=[
            SubQuery(
                id="q1",
                query="地质灾害倾斜摄影测量成果验收",
                subject="地质灾害倾斜摄影测量",
                aspect="成果验收",
            )
        ],
        synthesis_mode="matrix",
    )

    def table(chunk_id: str, document_id: str, number: str, title: str) -> RetrievedChunk:
        return _chunk(
            chunk_id,
            document_id,
            0.8,
            text=(
                f"<table><caption>{number} {title}</caption>"
                "<tr><th>检查项目</th><th>限差</th></tr>"
                "<tr><td>平面位置精度</td><td>0.20 m</td></tr></table>"
            ),
        )

    results = {
        "q1": [
            _chunk(
                "quality-prose",
                "doc-v1",
                0.99,
                text="数字正射影像成果质量检查应核对实地检查点坐标精度。",
            ),
            table("table3-v1", "doc-v1", "表3", "平面高程精度要求"),
            table("table3-v2", "doc-v2", "表3", "平面高程精度要求"),
            table("table4", "doc-v1", "表4", "点云密度要求"),
            table("table2", "doc-v1", "表2", "加密点中误差要求"),
            table("table5", "doc-v1", "表5", "成果质量要求"),
        ]
    }

    selection = CoverageSelector().select(plan, results, limit=4)
    ids = [item.chunk_id for item in selection.chunks]

    assert "table3-v1" in ids
    assert "table3-v2" not in ids
    assert "quality-prose" not in ids
    assert set(ids) == {"table3-v1", "table4", "table2", "table5"}
