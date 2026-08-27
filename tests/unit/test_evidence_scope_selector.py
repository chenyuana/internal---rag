from __future__ import annotations

from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.evidence_scope_selector import EvidenceScopeSelector
from app.services.query_analyzer import QueryAnalyzer


def _chunk(
    citation_id: str,
    *,
    document: str,
    article: str,
    title: str,
) -> SelectedChunk:
    return SelectedChunk(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        document_id=f"doc-{document}",
        dataset_id="kb-1",
        text=(
            f"文档：{document}\n"
            f"章节：第 {article} 条 {title}\n"
            f"条号：{article}\n"
            f"{title}正文"
        ),
        metadata=ChunkMetadata(document_name=document, status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1.0,
    )


def test_selects_all_chunks_of_exact_title_article() -> None:
    chunks = [
        _chunk(
            "C1",
            document="CCAR-25-R4 运输类飞机适航标准.pdf",
            article="25.981",
            title="燃油箱点燃防护",
        ),
        _chunk(
            "C2",
            document="CCAR-25-R4 运输类飞机适航标准.pdf",
            article="25.981",
            title="燃油箱点燃防护",
        ),
        _chunk(
            "C4",
            document="CCAR-26 运输类飞机持续适航规定.pdf",
            article="26.33",
            title="燃油箱可燃性",
        ),
    ]

    selected = EvidenceScopeSelector().select(
        QueryAnalyzer().analyze("运输类飞机燃油箱点燃防护的要求是什么？"),
        chunks,
    )

    assert [chunk.citation_id for chunk in selected] == ["C1", "C2"]


def test_explicit_article_number_selects_article_without_title_match() -> None:
    chunks = [
        _chunk(
            "C1",
            document="CCAR-25-R4.pdf",
            article="25.981",
            title="燃油箱点燃防护",
        ),
        _chunk(
            "C4",
            document="CCAR-26.pdf",
            article="26.33",
            title="燃油箱可燃性",
        ),
    ]

    selected = EvidenceScopeSelector().select(
        QueryAnalyzer().analyze("第25.981条有哪些要求？"),
        chunks,
    )

    assert [chunk.citation_id for chunk in selected] == ["C1"]


def test_comparison_query_keeps_cross_document_evidence() -> None:
    chunks = [
        _chunk(
            "C1",
            document="CCAR-25-R4.pdf",
            article="25.981",
            title="燃油箱点燃防护",
        ),
        _chunk(
            "C4",
            document="CCAR-26.pdf",
            article="26.33",
            title="燃油箱可燃性",
        ),
    ]

    selected = EvidenceScopeSelector().select(
        QueryAnalyzer().analyze("比较第25.981条与第26.33条的差异"),
        chunks,
    )

    assert selected == chunks
