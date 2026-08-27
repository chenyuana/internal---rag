from __future__ import annotations

import pytest

from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.grounding_validator import ClaimGroundingValidator


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


def test_rejects_numeric_value_not_present_in_cited_chunk() -> None:
    # 复现 7% 误归因：claim 说 7%，但被引 chunk（25.981 (iii)(c)(d)）里没有 7%。
    chunks = [
        chunk("C1", "第 25.981 条 (c) 本条(b)不适用于采用减轻燃油蒸气点燃影响措施的燃油箱。"),
    ]
    structured = answer(
        claim_text="机队平均可燃性暴露水平不得超过7%",
        citation_ids=["C1"],
    )

    with pytest.raises(ValueError, match="7%"):
        ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_allows_numeric_value_present_in_cited_chunk() -> None:
    chunks = [
        chunk(
            "C2",
            "第 25.981 条 (b) 机队平均可燃性暴露时间不得超过3%，或机翼燃油箱暴露时间取较大者。",
        ),
    ]
    structured = answer(
        claim_text="机队平均可燃性暴露时间不得超过3%",
        citation_ids=["C2"],
    )

    ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_allows_value_present_in_any_of_multiple_cited_chunks() -> None:
    chunks = [
        chunk("C1", "第 25.981 条 (c) 本条(b)不适用于IMM燃油箱。"),
        chunk("C4", "第 26.33 条 机队平均可燃性暴露水平超过7%的燃油箱。"),
    ]
    structured = answer(
        claim_text="机队平均可燃性暴露水平不得超过7%",
        citation_ids=["C1", "C4"],
    )

    ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_claim_without_numeric_marks_is_skipped() -> None:
    chunks = [chunk("C1", "第 25.981 条 燃油箱点燃防护。")]
    structured = answer(
        claim_text="燃油箱内任一点不得存在点火源",
        citation_ids=["C1"],
    )

    ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_fullwidth_percent_is_normalized() -> None:
    chunks = [chunk("C2", "机队平均可燃性暴露时间不得超过3％（全角）。")]
    structured = answer(
        claim_text="机队平均可燃性暴露时间不得超过3%",
        citation_ids=["C2"],
    )

    ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_unit_value_case_is_normalized() -> None:
    chunks = [chunk("C1", "HEALTH_MONITOR_PERIOD_MS = 1000MS。")]
    structured = answer(
        claim_text="健康监控周期为1000ms",
        citation_ids=["C1"],
    )

    ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_unknown_citation_is_ignored() -> None:
    # 未知 citation 由 CitationValidator 负责；本校验器跳过不存在的 chunk。
    chunks = [chunk("C2", "机队平均可燃性暴露时间不得超过3%。")]
    structured = answer(
        claim_text="机队平均可燃性暴露水平不得超过7%",
        citation_ids=["C99"],
    )

    ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_allows_table_encoded_value_with_unit_in_header() -> None:
    """表1 数据行是裸数字、单位在表头：claim 的 5cm 应被认定为有据可查。"""
    chunks = [
        chunk(
            "C3",
            "表1 无人机航摄地面分辨率\n"
            "| 调查比例尺 | 地质灾害类型 | 地面分辨率值（cm） |\n"
            "| --- | --- | --- |\n"
            "| 1:500 | 地裂缝 | 5 |\n"
            "| 1:2 000 |  | 20 |\n",
        ),
    ]
    structured = answer(
        claim_text="1:500比例尺下地裂缝的地面分辨率为5cm",
        citation_ids=["C3"],
    )

    ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_rejects_table_value_absent_from_rows() -> None:
    """表格块内不存在该数字：即使表头有单位也不放行。"""
    chunks = [
        chunk(
            "C3",
            "表1 无人机航摄地面分辨率\n"
            "| 调查比例尺 | 地质灾害类型 | 地面分辨率值（cm） |\n"
            "| --- | --- | --- |\n"
            "| 1:500 | 地裂缝 | 5 |\n",
        ),
    ]
    structured = answer(
        claim_text="1:500比例尺下地裂缝的地面分辨率为30cm",
        citation_ids=["C3"],
    )

    with pytest.raises(ValueError, match="30cm"):
        ClaimGroundingValidator().validate(structured, chunks=chunks)


def test_table_relaxation_does_not_apply_to_prose_chunk() -> None:
    """非表格普通正文块仍按字面匹配：5cm 不能匹配只有数字的正文。"""
    chunks = [
        chunk("C1", "各航摄分区基准面的地面分辨率应根据调查比例尺确定，不低于表1中的要求。"),
    ]
    structured = answer(
        claim_text="地面分辨率为5cm",
        citation_ids=["C1"],
    )

    with pytest.raises(ValueError, match="5cm"):
        ClaimGroundingValidator().validate(structured, chunks=chunks)
