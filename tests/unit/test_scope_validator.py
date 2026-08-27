from __future__ import annotations

import pytest

from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.query_analyzer import QueryAnalyzer
from app.services.scope_validator import ScopeConsistencyValidator

CCAR25_DOC = "CCAR-25-R4 运输类飞机适航标准.pdf"
CCAR26_DOC = "CCAR-26 运输类飞机的持续适航和安全改进规定.pdf"


def chunk(
    citation_id: str,
    *,
    document: str,
    article: str | None = None,
    body: str,
    chapter: str | None = None,
    version: str | None = "R4",
) -> SelectedChunk:
    header = [f"文档：{document}"]
    if chapter:
        header.append(f"章节：{chapter}")
    if article:
        header.append(f"条号：{article}")
    header.append("页码：123-124")
    text = "\n".join(header) + "\n\n" + body
    return SelectedChunk(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id=f"doc-{citation_id}",
        dataset_id="kb-1",
        text=text,
        metadata=ChunkMetadata(
            document_name=document,
            version=version,
            status="effective",
        ),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )


def answer(
    *,
    answer_text: str,
    claim_text: str,
    citation_ids: list[str],
) -> StructuredAnswer:
    return StructuredAnswer(
        answerability="ANSWERABLE",
        answer=answer_text,
        claims=[
            CandidateClaim(
                claim_id="claim-1",
                claim=claim_text,
                citation_ids=citation_ids,
            )
        ],
    )


def ccar25_981() -> SelectedChunk:
    return chunk(
        "C1",
        document=CCAR25_DOC,
        article="25.981",
        chapter="第 25.981 条 燃油箱点燃防护",
        body="第 25.981 条 燃油箱点火源防护。机队平均可燃性暴露时间不得超过3%。",
    )


def ccar26_633() -> SelectedChunk:
    return chunk(
        "C4",
        document=CCAR26_DOC,
        article="26.33",
        chapter="第26.33条 型号合格证/型号认可证持有人一燃油箱可燃性",
        body=(
            "第26.33条 对于机队平均可燃性暴露水平超过7%的燃油箱，"
            "FRM必须满足附录M除M25.1外的所有要求。"
        ),
        version="R2",
    )


def test_rejects_cross_regulation_content_attributed_to_named_article() -> None:
    # 复现 08-18 跨规章语义混淆：回答声明"第 25.981 条"，claim 却引用 CCAR-26 26.33 的 chunk。
    # 关键：claim 文本顺带提到"M25.1"、CCAR-26 chunk 文本也含"M25.1"，都不能放行错误归因。
    query = QueryAnalyzer().analyze("对于运输类飞机燃油箱点燃防护的要求是什么？")
    chunks = [ccar25_981(), ccar26_633()]
    structured = answer(
        answer_text="主要依据《运输类飞机适航标准》第 25.981 条执行。[C1]",
        claim_text=(
            "对于设计成通常为空的且位于机身轮廓线以内的所有其他燃油箱，"
            "FRM 需满足《运输类飞机适航标准》附录 M（除第 M25.1 条）的要求，"
            "机队平均可燃性暴露水平不得超过 7%。"
        ),
        citation_ids=["C4"],
    )

    with pytest.raises(ValueError, match="C4"):
        ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)


def test_rejects_when_query_names_article_but_claim_cites_other_document() -> None:
    query = QueryAnalyzer().analyze("25.981 燃油箱点燃防护的要求是什么？")
    chunks = [ccar25_981(), ccar26_633()]
    structured = answer(
        answer_text="机队平均可燃性暴露水平不得超过7%。[C4]",
        claim_text="机队平均可燃性暴露水平不得超过7%",
        citation_ids=["C4"],
    )

    with pytest.raises(ValueError, match="C4"):
        ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)


def test_allows_same_document_sub_provision() -> None:
    # 同文档内的附录 chunk（无条号头、文档头属于主条号文档）应放行。
    query = QueryAnalyzer().analyze("对于运输类飞机燃油箱点燃防护的要求是什么？")
    chunks = [
        ccar25_981(),
        chunk(
            "C6",
            document=CCAR25_DOC,
            chapter="附录 N",
            body="附录 N N25.3 可燃性暴露分析应按要求执行。",
        ),
    ]
    structured = answer(
        answer_text="根据第 25.981 条，可燃性暴露分析需依据附录 N 的要求执行。[C6]",
        claim_text="根据第 25.981 条，可燃性暴露分析需依据附录 N 执行",
        citation_ids=["C6"],
    )

    ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)


def test_allows_honest_cross_reference_that_names_other_article() -> None:
    # 诚实交叉引用：claim 显式点名 26.33，引用 CCAR-26 chunk 应放行。
    query = QueryAnalyzer().analyze("对于运输类飞机燃油箱点燃防护的要求是什么？")
    chunks = [ccar25_981(), ccar26_633()]
    structured = answer(
        answer_text="根据第 25.981 条执行；补充要求见第 26.33 条。[C4]",
        claim_text="根据 CCAR-26 第 26.33 条，机队平均可燃性暴露水平不得超过7%",
        citation_ids=["C4"],
    )

    ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)


def test_no_article_declared_is_noop() -> None:
    query = QueryAnalyzer().analyze("HEALTH_MONITOR_PERIOD_MS 的值是多少？")
    chunks = [chunk("C1", document="手册", body="HEALTH_MONITOR_PERIOD_MS = 1000U。")]
    structured = answer(
        answer_text="任务周期为1000毫秒。[C1]",
        claim_text="任务周期为1000毫秒",
        citation_ids=["C1"],
    )

    ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)


def test_legacy_chunk_without_header_falls_back_to_alias() -> None:
    # 无"条号："头的遗留格式 chunk，若文本含主条号 alias 应放行。
    query = QueryAnalyzer().analyze("25.981 燃油箱点燃防护的要求是什么？")
    legacy = SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text="第 25.981 条 燃油箱点火源防护。",
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )
    structured = answer(
        answer_text="根据第 25.981 条 执行。[C1]",
        claim_text="根据第 25.981 条 执行",
        citation_ids=["C1"],
    )

    ScopeConsistencyValidator().validate(structured, query=query, chunks=[legacy])


def test_legacy_chunk_without_any_signal_is_skipped() -> None:
    # 无任何头部信号且文本不含主条号的 chunk 无法判定归属，保守放行避免误报。
    query = QueryAnalyzer().analyze("25.981 燃油箱点燃防护的要求是什么？")
    chunks = [ccar25_981()]
    legacy = SelectedChunk(
        citation_id="C9",
        chunk_id="chunk-9",
        document_id="doc-9",
        dataset_id="kb-1",
        text="对于机队平均可燃性暴露水平超过7%的燃油箱。",
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.8,
        vector_score=0.7,
        keyword_score=1,
    )
    structured = answer(
        answer_text="机队平均可燃性暴露水平不得超过7%。[C9]",
        claim_text="机队平均可燃性暴露水平不得超过7%",
        citation_ids=["C9"],
    )

    ScopeConsistencyValidator().validate(structured, query=query, chunks=[*chunks, legacy])


def test_unknown_citation_id_is_skipped() -> None:
    query = QueryAnalyzer().analyze("25.981 燃油箱点燃防护的要求是什么？")
    chunks = [ccar25_981()]
    structured = answer(
        answer_text="根据第 25.981 条 执行。[C99]",
        claim_text="根据第 25.981 条 执行",
        citation_ids=["C99"],
    )

    # 未知 citation 由 CitationValidator 负责拒绝，本校验器不重复处理。
    ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)


def test_rejects_file_scope_violation_when_answer_declares_document() -> None:
    # 模型绕过条号点名、改为文件级归因：回答声明"（CCAR-25-R4）"却引用 CCAR-26 chunk，
    # claim 未点名 CCAR-26。这正是 08-18 用户复现时回答仍把 7% 内容放进 CCAR-25-R4 语境的场景。
    query = QueryAnalyzer().analyze("对于运输类飞机燃油箱点燃防护的要求是什么？")
    chunks = [ccar25_981(), ccar26_633()]
    structured = answer(
        answer_text=(
            "根据《运输类飞机适航标准》（CCAR-25-R4），燃油箱点燃防护要求包括："
            "所有其他燃油箱的FRM需满足附录M除M25.1之外的要求。[C4]"
        ),
        claim_text=(
            "所有其他燃油箱的FRM需满足附录M除M25.1之外的要求，"
            "且机队平均可燃性暴露水平不得超过7%"
        ),
        citation_ids=["C4"],
    )

    with pytest.raises(ValueError, match="C4"):
        ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)


def test_allows_comparison_query_covering_both_documents() -> None:
    # 对比题显式声明两个文件，CCAR-26 的 chunk 应放行。
    query = QueryAnalyzer().analyze("CCAR-25 与 CCAR-26 对燃油箱可燃性要求的差异是什么？")
    chunks = [ccar25_981(), ccar26_633()]
    structured = answer(
        answer_text="CCAR-25-R4 与 CCAR-26 的要求存在差异。[C4]",
        claim_text="CCAR-26 要求机队平均可燃性暴露水平不得超过7%",
        citation_ids=["C4"],
    )

    ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)


def test_allows_claim_that_names_chunk_document() -> None:
    # 诚实交叉引用：claim 显式点名 CCAR-26，引用 CCAR-26 chunk 应放行。
    query = QueryAnalyzer().analyze("对于运输类飞机燃油箱点燃防护的要求是什么？")
    chunks = [ccar25_981(), ccar26_633()]
    structured = answer(
        answer_text="根据《运输类飞机适航标准》（CCAR-25-R4）执行，补充见 CCAR-26。[C4]",
        claim_text="根据 CCAR-26 第 26.33 条，机队平均可燃性暴露水平不得超过7%",
        citation_ids=["C4"],
    )

    ScopeConsistencyValidator().validate(structured, query=query, chunks=chunks)
