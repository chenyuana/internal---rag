from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import ChunkMetadata, QueryPlan, SelectedChunk, SubQuery
from app.services.subject_grounding_validator import SubjectGroundingValidator


def _plan() -> QueryPlan:
    return QueryPlan(
        query_type="comparison",
        subjects=["地质灾害倾斜航测", "河湖智能巡检无人机"],
        aspects=["重叠度"],
        subqueries=[
            SubQuery(
                id="q1",
                query="地质灾害倾斜航测 重叠度",
                subject="地质灾害倾斜航测",
                aspect="重叠度",
            ),
            SubQuery(
                id="q2",
                query="河湖智能巡检无人机 重叠度",
                subject="河湖智能巡检无人机",
                aspect="重叠度",
            ),
        ],
        synthesis_mode="matrix",
    )


def _chunk(citation_id: str, chunk_id: str, subquery_id: str | None) -> SelectedChunk:
    return SelectedChunk(
        citation_id=citation_id,
        chunk_id=chunk_id,
        document_id=f"doc-{subquery_id}",
        dataset_id="kb-1",
        text=f"文档：{subquery_id}.pdf",
        metadata=ChunkMetadata(document_name=f"{subquery_id}.pdf"),
        hybrid_score=1.0,
        vector_score=0.0,
        keyword_score=0.0,
        subquery_id=subquery_id,
    )


def _answer(claim_text: str, citation_ids: list[str]) -> StructuredAnswer:
    return StructuredAnswer(
        answerability="ANSWERABLE",
        answer=claim_text,
        claims=[
            CandidateClaim(claim_id="c1", claim=claim_text, citation_ids=citation_ids)
        ],
    )


def test_rejects_cross_subject_attribution() -> None:
    """河湖侧 claim 引用地质灾害侧 chunk：必须拦截（数值虽在 chunk 中，主体归因错误）。"""
    chunks = [
        _chunk("C1", "c1", "q1"),  # 地质灾害
        _chunk("C2", "c2", "q2"),  # 河湖
    ]
    answer = _answer(
        "河湖智能巡检无人机在丘陵山地地区的航向重叠度一般为 50%~80%。",
        ["C1"],
    )
    try:
        SubjectGroundingValidator().validate(answer, plan=_plan(), chunks=chunks)
    except ValueError as exc:
        assert "C1" in str(exc)
    else:
        raise AssertionError("expected ValueError for cross-subject citation")


def test_accepts_same_subject_attribution() -> None:
    chunks = [
        _chunk("C1", "c1", "q1"),
        _chunk("C2", "c2", "q2"),
    ]
    answer = _answer(
        "地质灾害倾斜航测在丘陵山地地区的航向重叠度一般为 50%~80%。",
        ["C1"],
    )
    SubjectGroundingValidator().validate(answer, plan=_plan(), chunks=chunks)


def test_accepts_explicit_cross_subject_reference() -> None:
    """claim 同时点名两个主体（对比句）：两侧 chunk 都允许。"""
    chunks = [
        _chunk("C1", "c1", "q1"),
        _chunk("C2", "c2", "q2"),
    ]
    answer = _answer(
        "对比地质灾害倾斜航测与河湖智能巡检无人机，二者重叠度要求不同。",
        ["C1", "C2"],
    )
    SubjectGroundingValidator().validate(answer, plan=_plan(), chunks=chunks)


def test_skips_claim_without_subject_name() -> None:
    """未点名任何主体的共享/总结句：保守放行。"""
    chunks = [
        _chunk("C1", "c1", "q1"),
        _chunk("C2", "c2", "q2"),
    ]
    answer = _answer("二者在重叠度规定上存在差异。", ["C1"])
    SubjectGroundingValidator().validate(answer, plan=_plan(), chunks=chunks)


def test_skips_when_chunk_attribution_missing() -> None:
    """chunk 无 subquery_id（单查询回退等场景）：跳过校验。"""
    chunks = [
        SelectedChunk(
            citation_id="C1",
            chunk_id="c1",
            document_id="doc-x",
            dataset_id="kb-1",
            text="文档：x.pdf",
            metadata=ChunkMetadata(document_name="x.pdf"),
            hybrid_score=1.0,
            vector_score=0.0,
            keyword_score=0.0,
            subquery_id=None,
        )
    ]
    answer = _answer("河湖智能巡检无人机…50%~80%。", ["C1"])
    SubjectGroundingValidator().validate(answer, plan=_plan(), chunks=chunks)


def test_skips_non_comparison_plan() -> None:
    plan = QueryPlan(
        query_type="fact",
        subjects=[],
        aspects=["问题"],
        subqueries=[SubQuery(id="q1", query="问题", aspect="问题")],
        synthesis_mode="direct",
    )
    chunks = [_chunk("C1", "c1", "q1")]
    answer = _answer("任意内容。", ["C1"])
    SubjectGroundingValidator().validate(answer, plan=plan, chunks=chunks)
