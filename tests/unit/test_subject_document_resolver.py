from __future__ import annotations

from app.schemas.retrieval import ChunkMetadata, RetrievedChunk
from app.services.subject_document_resolver import (
    SubjectDocumentResolver,
    subject_title_similarity,
    title_similarity,
)


def _candidate(
    document_id: str,
    name: str,
    subject: str | None = None,
    *,
    score: float = 0.8,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"chunk-{document_id}",
        document_id=document_id,
        dataset_id="kb-1",
        text="内容",
        metadata=ChunkMetadata(document_name=name, document_subject=subject),
        hybrid_score=score,
    )


def test_resolver_prefers_exact_document_subject_metadata() -> None:
    resolver = SubjectDocumentResolver()
    result = resolver.resolve(
        "地质灾害倾斜摄影测量",
        [
            _candidate("right", "无关标题", "地质灾害倾斜摄影测量"),
            _candidate("wrong", "地质灾害调查规程", "其他主体"),
        ],
    )
    assert result.document_ids == ["right"]
    assert result.source == "metadata"


def test_resolver_uses_whole_title_similarity_without_two_character_fallback() -> None:
    resolver = SubjectDocumentResolver()
    right = "长大桥梁无人机巡检作业技术规程"
    result = resolver.resolve(
        "长大桥梁无人机精细化巡检",
        [_candidate("right", right), _candidate("wrong", "水运工程桩位测量规程")],
    )
    assert title_similarity("长大桥梁无人机精细化巡检", right) > 0.38
    assert result.document_ids == ["right"]
    assert result.source == "title"


def test_resolver_returns_none_instead_of_cross_subject_filling() -> None:
    result = SubjectDocumentResolver().resolve(
        "火星隧道量子巡检",
        [_candidate("wrong", "水运工程桩位测量规程")],
    )
    assert result.document_ids == []
    assert result.source == "none"


def test_resolver_prefers_retrieval_leader_over_literal_title_match() -> None:
    """检索分数领先时优先于字面标题相似度。

    真实案例（2026-08-21 沙地题）：主体词"无人机通用服务管理要求"检索时，
    目标文档《无人机应用服务通用规范》hybrid 0.4273 领先，但字面标题更接近
    《微轻小型无人机机巢通用管理要求》（title 相似度更高）。检索分数通道应
    选前者，而不是被 bigram 撞车带偏。
    """
    resolver = SubjectDocumentResolver()
    result = resolver.resolve(
        "无人机通用服务管理要求",
        [
            _candidate(
                "right",
                "无人机应用服务通用规范",
                score=0.4273,
            ),
            _candidate(
                "literal-hit",
                "微轻小型无人机机巢通用管理要求",
                score=0.4069,
            ),
            _candidate("unrelated", "航空应急救援无人机操作规范", score=0.39),
        ],
    )
    assert result.document_ids == ["right"]
    assert result.source == "retrieval"


def test_resolver_retrieval_channel_requires_clear_margin() -> None:
    """检索分不分胜负时回退 title 通道，不掷硬币。"""
    resolver = SubjectDocumentResolver()
    result = resolver.resolve(
        "长大桥梁无人机精细化巡检",
        [
            _candidate("right", "长大桥梁无人机巡检作业技术规程", score=0.42),
            _candidate("wrong", "水运工程桩位测量规程", score=0.415),
        ],
    )
    # margin 0.005 < 0.015：检索通道不触发；title 通道选长大桥梁。
    assert result.document_ids == ["right"]
    assert result.source == "title"


def test_resolver_retrieval_channel_ignores_low_scores() -> None:
    """检索分过低（无证据题）时检索通道不触发，title 通道返回 none。"""
    result = SubjectDocumentResolver().resolve(
        "火星隧道量子巡检",
        [_candidate("wrong", "低空5G通信基站建设要求", score=0.2619)],
    )
    assert result.document_ids == []
    assert result.source == "none"


def test_resolver_uses_specialist_vocabulary_for_forestry_and_rice_titles() -> None:
    forestry = SubjectDocumentResolver().resolve(
        "林地病虫害喷洒无人机",
        [
            _candidate(
                "forestry",
                "20240705：《无人机喷洒防治林业有害生物技术规程》.pdf",
                score=0.4616,
            ),
            _candidate(
                "generic",
                "20240723：《植保无人机飞防农作物病虫害技术规范》.pdf",
                score=0.39,
            ),
        ],
    )
    rice = SubjectDocumentResolver().resolve(
        "水稻植保无人机",
        [
            _candidate(
                "rice",
                "20240821：《无人机防治稻纵卷叶螟施药技术规范》.pdf",
                score=0.4256,
            ),
            _candidate(
                "generic",
                "20240723：《植保无人机飞防农作物病虫害技术规范》.pdf",
                score=0.39,
            ),
        ],
    )

    assert subject_title_similarity(
        "林地病虫害喷洒无人机",
        "20240705：《无人机喷洒防治林业有害生物技术规程》.pdf",
    ) >= 0.30
    assert forestry.document_ids == ["forestry"]
    assert rice.document_ids == ["rice"]
