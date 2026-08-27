from __future__ import annotations

from app.core.config import GenerationSettings
from app.schemas.retrieval import ChunkMetadata, SelectedChunk
from app.services.evidence_extractor import EvidenceExtractor
from app.services.query_analyzer import QueryAnalyzer


def test_evidence_extractor_preserves_exact_technical_token() -> None:
    query = QueryAnalyzer().analyze("HEALTH_MONITOR_PERIOD_MS 的值是多少？")
    chunk = SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text=(
            "该模块负责系统初始化。\n"
            "HEALTH_MONITOR_PERIOD_MS = 1000U。\n"
            "其他任务使用独立调度器。"
        ),
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    assert evidence[0].citation_id == "C1"
    assert evidence[0].text == "HEALTH_MONITOR_PERIOD_MS = 1000U。"


def test_evidence_extractor_carries_source_document_from_chunk_metadata() -> None:
    query = QueryAnalyzer().analyze("25.981 燃油箱点燃防护的要求是什么？")
    chunk = SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text="第 25.981 条 燃油箱点火源防护。",
        metadata=ChunkMetadata(
            document_name="CCAR-25-R4《运输类飞机适航标准》",
            version="R4",
            status="effective",
        ),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    assert evidence[0].document_name == "CCAR-25-R4《运输类飞机适航标准》"
    assert evidence[0].version == "R4"


def test_evidence_extractor_leaves_source_document_null_when_absent() -> None:
    query = QueryAnalyzer().analyze("HEALTH_MONITOR_PERIOD_MS 的值是多少？")
    chunk = SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text="HEALTH_MONITOR_PERIOD_MS = 1000U。",
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    assert evidence[0].document_name is None
    assert evidence[0].version is None


def test_evidence_extractor_skips_meta_lines_and_prefers_parameter_value() -> None:
    """切块头部元数据行（文档/章节/条号）与章节标题不得抢占证据名额。

    参数类问题下，携带具体数值/单位定义的正文句必须优先被选为证据，
    否则模型只见"5.3 播种量"标题而看不到"2.0 kg~3.5 kg/667m²"数值。
    """
    query = QueryAnalyzer().analyze("早熟中稻无人机直播 播种量计算逻辑")
    chunk = SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text=(
            "文档：20241224：《早熟中稻无人机直播栽培技术规程》.pdf\n"
            "章节：5 播种 / 5.3 播种量\n"
            "条号：5.3 页码：6\n\n"
            "5 播种\n"
            "5.3 播种量\n"
            "常规籼稻播种量为每2.0 kg~3.5 kg/ 667 m2,"
            "常规粳稻播种量为每4.0 kg~6 kg/ 667 m2,"
            "杂交稻播种量为1.8 kg~2.0 kg/ 667 m2。"
        ),
        metadata=ChunkMetadata(
            document_name="20241224：《早熟中稻无人机直播栽培技术规程》.pdf",
            status="effective",
        ),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    assert evidence, "expected at least one evidence sentence"
    assert "文档：" not in evidence[0].text
    assert "章节：" not in evidence[0].text
    assert "条号：" not in evidence[0].text
    assert "常规籼稻" in evidence[0].text
    assert "2.0 kg" in evidence[0].text


def test_evidence_extractor_keeps_markdown_table_as_one_evidence() -> None:
    """Markdown 表格块应作为一条完整证据保留，而不是被逐行切分丢弃。

    表格数据行（"| 1:500 | 地裂缝 | 5 |"）与查询词汇重叠率极低，若按普通
    句子处理会被 overlap 排序挤出 evidence，导致"见表1"只有引用没有内容。
    """
    query = QueryAnalyzer().analyze("地质灾害倾斜航测 地面分辨率要求")
    chunk = SelectedChunk(
        citation_id="C3",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text=(
            "6 航线规划\n"
            "6.1 航摄地面分辨率的选择\n"
            "表1 无人机航摄地面分辨率\n"
            "| 调查比例尺 | 地质灾害类型 | 地面分辨率值（cm） |\n"
            "| --- | --- | --- |\n"
            "| 1:500 | 地裂缝 | 5 |\n"
            "| 1:500 | 崩塌、滑坡、泥石流 | 5 |\n"
            "| 1:1 000 |  | 10 |\n"
            "| 1:2 000 |  | 20 |"
        ),
        metadata=ChunkMetadata(
            document_name="20230606：《地质灾害调查无人机低空摄影测量技术规程》.pdf",
            status="effective",
        ),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    assert evidence, "expected table block to be kept"
    table_evidence = next(
        (item for item in evidence if "1:500" in item.text and "地裂缝" in item.text),
        None,
    )
    assert table_evidence is not None
    assert "| 调查比例尺 |" in table_evidence.text
    assert "| 1:2 000 |" in table_evidence.text


def test_evidence_extractor_merges_follow_on_into_subitem() -> None:
    """分项承接句必须并回所属分项，保留适用对象。

    句子切分会把"在丘陵山地地区，航向旁向重叠度50%~80%。"从
    "c) 机载激光雷达:旁向重叠度不小于30%;"切出，使模型丢失"机载激光雷达"
    归属而把 c 款数值当成整个作业类别的通用值。
    """
    query = QueryAnalyzer().analyze("地质灾害倾斜航测 航向/旁向重叠度")
    chunk = SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text=(
            "6.3.2 像片重叠度\n"
            "a) 垂直影像：旁向重叠度一般为40%~80%，最小不小于30%；"
            "航向重叠度一般为65%~85%，最小不小于60%；"
            "b) 倾斜摄影：当满足垂直影像重叠度后，倾斜影像的航向、旁向重叠度可不再重新设计；"
            "c) 机载激光雷达：旁向重叠度不小于30%；"
            "在丘陵山地地区，航向旁向重叠度50%~80%。"
        ),
        metadata=ChunkMetadata(
            document_name="20230606：《地质灾害调查无人机低空摄影测量技术规程》.pdf",
            status="effective",
        ),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    merged = next(
        (
            item.text
            for item in evidence
            if "50%~80%" in item.text and "丘陵山地" in item.text
        ),
        None,
    )
    assert merged is not None
    assert "机载激光雷达" in merged
    assert merged.index("机载激光雷达") < merged.index("丘陵山地")


def test_evidence_extractor_prefixes_source_document() -> None:
    """证据句必须内联来源标识（[来源：文档名]），使同一主题多份规程可区分。"""
    query = QueryAnalyzer().analyze("无人机监测松材线虫病 作业流程")
    chunk = SelectedChunk(
        citation_id="C5",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text=(
            "文档：20250721：《无人机监测松材线虫病致死松树技术规程》.pdf\n"
            "章节：4 航拍准备 / 4.1 无人机飞行器\n"
            "条号：4.1 页码：5-6\n\n"
            "4 航拍准备\n"
            "4.1 无人机飞行器\n"
            "无人机飞行器的关键性能指标主要有:"
            "a) 续航时间应大于 60 min;"
        ),
        metadata=ChunkMetadata(
            document_name="20250721：《无人机监测松材线虫病致死松树技术规程》.pdf",
            status="effective",
        ),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    assert evidence
    assert evidence[0].text.startswith(
        "[来源：20250721：《无人机监测松材线虫病致死松树技术规程》.pdf]"
    )
    assert "[4 航拍准备/4.1 无人机飞行器]" in evidence[0].text
    body = next(
        (item.text for item in evidence if "续航时间" in item.text),
        None,
    )
    assert body is not None
    assert body.startswith("[来源：20250721：《无人机监测松材线虫病致死松树技术规程》.pdf]")
    assert "续航时间应大于 60 min" in body


def test_evidence_extractor_no_prefix_without_document_name() -> None:
    query = QueryAnalyzer().analyze("HEALTH_MONITOR_PERIOD_MS 的值是多少？")
    chunk = SelectedChunk(
        citation_id="C1",
        chunk_id="chunk-1",
        document_id="doc-1",
        dataset_id="kb-1",
        text="HEALTH_MONITOR_PERIOD_MS = 1000U。",
        metadata=ChunkMetadata(status="effective"),
        hybrid_score=0.9,
        vector_score=0.8,
        keyword_score=1,
    )

    evidence = EvidenceExtractor(GenerationSettings()).extract(query, [chunk])

    assert evidence[0].text == "HEALTH_MONITOR_PERIOD_MS = 1000U。"
