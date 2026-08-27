from __future__ import annotations

from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.query_analyzer import QueryAnalyzer
from app.services.regulation_answer_builder import RegulationAnswerBuilder


def _chunk(citation_id: str, body: str) -> SelectedChunk:
    return SelectedChunk(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id="doc-25",
        dataset_id="kb-1",
        text=(
            "文档：CCAR-25-R4 运输类飞机适航标准.pdf\n"
            "章节：第 25.981 条 燃油箱点燃防护\n"
            "条号：25.981\n"
            "页码：123\n\n"
            f"{body}"
        ),
        metadata=ChunkMetadata(
            document_name="CCAR-25-R4 运输类飞机适航标准.pdf",
            status="effective",
        ),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )


def test_builds_verbatim_top_level_clauses_across_chunks() -> None:
    chunks = [
        _chunk(
            "C2",
            "第 25.981 条 燃油箱点燃防护\n"
            "(a) 任一点不得有点火源存在。\n"
            "(1) 温度必须留有安全裕度。\n"
            "(b) 机队平均可燃性暴露时间不得超过3%。\n"
            "(1) 按附录N确定。",
        ),
        _chunk(
            "C1",
            "(c) 本条(b)不适用于采用减轻燃油蒸气点燃影响措施的燃油箱。\n"
            "(d) 必须建立关键设计构型控制限制。\n"
            "〔第四次修订〕\n燃油系统部件",
        ),
    ]

    answer = RegulationAnswerBuilder().build(
        QueryAnalyzer().analyze("运输类飞机燃油箱点燃防护的要求是什么？"),
        chunks,
    )

    assert answer is not None
    assert [claim.claim_id for claim in answer.claims] == [
        "article-a",
        "article-b",
        "article-c",
        "article-d",
    ]
    assert "不得超过3%" in answer.answer
    assert "本条(b)不适用" in answer.answer
    assert "燃油系统部件" not in answer.answer
    assert "[C2]" in answer.answer
    assert "[C1]" in answer.answer


def test_does_not_build_for_non_requirement_query() -> None:
    answer = RegulationAnswerBuilder().build(
        QueryAnalyzer().analyze("这份文件的发布日期是什么？"),
        [_chunk("C1", "(a) 任一点不得有点火源存在。")],
    )

    assert answer is None


def test_builds_complete_procedure_in_document_section_order() -> None:
    document = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
    chunks = [
        SelectedChunk(
            citation_id="C0",
            chunk_id="scope",
            document_id="doc-pine",
            dataset_id="kb-1",
            text=(
                f"文档：{document}\n章节：1 范围\n\n"
                "1 范围\n本文件规定了以无人机为平台监测松材线虫病致死松树的术语和定义、"
                "基本要求、航摄规划、作业准备、影像获取、数据处理、地面验证、"
                "档案管理和安全注意事项。"
            ),
            metadata=ChunkMetadata(document_name=document, status="effective"),
            hybrid_score=0.95,
            vector_score=0.85,
            keyword_score=1.0,
        ),
        SelectedChunk(
            citation_id="C1",
            chunk_id="planning",
            document_id="doc-pine",
            dataset_id="kb-1",
            text=(
                f"文档：{document}\n章节：5 航摄规划\n\n"
                "5 航摄规划\n5.1 航摄范围\n根据疫情信息确定航摄范围。\n"
                "5.2 航摄设计\n航摄像片应完全覆盖测区。"
            ),
            metadata=ChunkMetadata(document_name=document, status="effective"),
            hybrid_score=0.9,
            vector_score=0.8,
            keyword_score=1.0,
        ),
        SelectedChunk(
            citation_id="C2",
            chunk_id="remaining",
            document_id="doc-pine",
            dataset_id="kb-1",
            text=(
                f"文档：{document}\n章节：6 作业准备 / 10 档案管理\n\n"
                "6.1 飞行平台组装和调试\n检查连接、电池和通信。\n"
                "7 影像获取\n7.1 起飞\n选择适宜天气。\n7.5 影像质量\n不合格时补飞。\n"
                "8 数据处理\n当天飞行数据当天处理。\n"
                "9 地面验证\n针对影像处理结果开展现场验证。\n"
                "10 档案管理\n原始影像和处理数据应归档保存3年。\n"
                "11 安全注意事项\n遵守安全规定。"
            ),
            metadata=ChunkMetadata(document_name=document, status="effective"),
            hybrid_score=0.8,
            vector_score=0.7,
            keyword_score=0.9,
        ),
    ]

    answer = RegulationAnswerBuilder().build(
        QueryAnalyzer().analyze(
            "采用无人机开展松材线虫病监测，从航摄规划到成果归档的完整流程是什么？"
        ),
        chunks,
    )

    assert answer is not None
    assert answer.answerability == "ANSWERABLE"
    assert [
        value for value in ("航摄规划", "作业准备", "影像获取", "数据处理", "地面验证", "档案管理")
        if value in answer.answer
    ] == ["航摄规划", "作业准备", "影像获取", "数据处理", "地面验证", "档案管理"]
    assert "安全注意事项" not in answer.answer
    assert len(answer.claims) == 6
