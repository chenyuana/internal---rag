import pytest

from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.claim_topic_validator import ClaimTopicValidator, _info_bigrams


def chunk(citation_id: str, text: str) -> SelectedChunk:
    return SelectedChunk(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id=f"doc-{citation_id}",
        dataset_id="kb-1",
        text=text,
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )


def answer(*, claim_text: str, citation_ids: list[str]) -> StructuredAnswer:
    return StructuredAnswer(
        answerability="ANSWERABLE",
        answer=f"{claim_text}[{citation_ids[0]}]",
        claims=[
            CandidateClaim(
                claim_id="claim-1",
                claim=claim_text,
                citation_ids=citation_ids,
            )
        ],
    )


def test_info_bigrams_filters_stop_words() -> None:
    bigrams = _info_bigrams("在管制空域作业应获得空中交通管理机构批准")
    assert "空域" in bigrams
    assert "批准" in bigrams
    assert "作业" not in bigrams  # 停用双字词


def test_rejects_claim_topic_absent_from_cited_chunk() -> None:
    """复现林业飞防引用脱钩：'空域批准'挂到喷洒无人机设备 chunk 必须拦截。"""
    chunks = [
        chunk(
            "C7",
            "4 总体要求 4.2.1 喷洒作业无人机 应具备在民用航空主管部门的注册登记证明,"
            "具有飞行控制系统及喷洒系统,喷洒系统应配备双泵。",
        ),
    ]
    structured = answer(
        claim_text="在管制空域作业或飞行作业真高大于30 m时应获得空中交通管理机构批准",
        citation_ids=["C7"],
    )

    with pytest.raises(ValueError, match="空域"):
        ClaimTopicValidator().validate(structured, chunks=chunks)


def test_accepts_claim_topic_present_in_cited_chunk() -> None:
    """'空域批准'挂到正确的 5.1.1 空域批准 chunk 必须放行。"""
    chunks = [
        chunk(
            "C5",
            "5 作业前准备 5.1.1 在管制空域作业或飞行作业真高大于30 m,应获得空中交通管理机构批准。",
        ),
    ]
    structured = answer(
        claim_text="在管制空域作业或飞行作业真高大于30 m时应获得空中交通管理机构批准",
        citation_ids=["C5"],
    )

    ClaimTopicValidator().validate(structured, chunks=chunks)


def test_accepts_claim_supported_by_any_of_multiple_citations() -> None:
    chunks = [
        chunk("C7", "4 总体要求 4.2.1 喷洒作业无人机 应具备注册登记证明。"),
        chunk("C9", "5 作业前准备 5.1.1 在管制空域作业应获得空中交通管理机构批准。"),
    ]
    structured = answer(
        claim_text="在管制空域作业应获得空中交通管理机构批准",
        citation_ids=["C7", "C9"],
    )

    ClaimTopicValidator().validate(structured, chunks=chunks)


def test_skips_claim_without_enough_information_bigrams() -> None:
    """短 claim 信息 bigram <2 时跳过（保守，交由其它校验器）。"""
    chunks = [chunk("C1", "作业区域应进行标注。")]
    structured = answer(claim_text="标注", citation_ids=["C1"])

    ClaimTopicValidator().validate(structured, chunks=chunks)


def test_skips_unknown_citation() -> None:
    chunks = [chunk("C2", "配药设备应清洗。")]
    structured = answer(claim_text="电子围栏允许范围内作业", citation_ids=["C99"])

    ClaimTopicValidator().validate(structured, chunks=chunks)
