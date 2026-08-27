from app.schemas.chat import EvidenceSentence
from app.services.multi_document import (
    document_title,
    multi_document_groups,
    titles_similar,
)

DOC_A = "20241217：《无人机监测松材线虫病致死松树技术规程》.pdf"
DOC_B = "20250721：《无人机监测松材线虫病致死松树技术规程》.pdf"
TITLE = "无人机监测松材线虫病致死松树技术规程"


def _evidence(citation_id: str, document_name: str, text: str = "内容。") -> EvidenceSentence:
    return EvidenceSentence(
        citation_id=citation_id,
        chunk_id=f"chunk-{citation_id}",
        text=text,
        document_name=document_name,
    )


def test_document_title_extracts_book_title() -> None:
    assert document_title(DOC_B) == TITLE


def test_document_title_falls_back_to_stripped_filename() -> None:
    assert document_title("20241217: 无书名号文档.pdf") == "无书名号文档"


def test_titles_similar_same_title() -> None:
    assert titles_similar(TITLE, TITLE)


def test_titles_similar_related_titles() -> None:
    assert titles_similar(TITLE, "无人机监测松材线虫病致 死松树技术规程")


def test_titles_not_similar_unrelated() -> None:
    assert not titles_similar(TITLE, "长大桥梁无人机巡检作业技术规程")


def test_multi_document_groups_splits_same_named_regulations() -> None:
    """同名不同日期的两份规程：按文档分组，各自证据独立。"""
    evidence = [
        _evidence("C1", DOC_A, "航摄规划内容一。"),
        _evidence("C2", DOC_A, "航摄规划内容二。"),
        _evidence("C5", DOC_B, "航拍作业内容一。"),
        _evidence("C6", DOC_B, "航拍作业内容二。"),
    ]

    groups = multi_document_groups(evidence)

    assert len(groups) == 2
    by_name = dict(groups)
    assert {item.citation_id for item in by_name[DOC_A]} == {"C1", "C2"}
    assert {item.citation_id for item in by_name[DOC_B]} == {"C5", "C6"}


def test_multi_document_groups_empty_for_unrelated_documents() -> None:
    evidence = [
        _evidence("C1", DOC_A, "内容一。"),
        _evidence("C2", DOC_A, "内容二。"),
        _evidence("C3", "20241018：《无人机河湖智能巡查要求》.pdf", "内容三。"),
        _evidence("C4", "20241018：《无人机河湖智能巡查要求》.pdf", "内容四。"),
    ]

    assert multi_document_groups(evidence) == []


def test_multi_document_groups_empty_when_group_evidence_insufficient() -> None:
    """某份规程只有 1 句证据时不触发分列（信息不足以单独组织回答）。"""
    evidence = [
        _evidence("C1", DOC_A, "内容一。"),
        _evidence("C2", DOC_A, "内容二。"),
        _evidence("C5", DOC_B, "仅一句。"),
    ]

    assert multi_document_groups(evidence) == []


def test_multi_document_groups_excludes_unrelated_documents() -> None:
    """分列只包含参与同名对的文档；无关文档（常绿果树）不进入分列。"""
    green = "20241018：《常绿果树养分诊断无人机多光谱遥感监测技术应用规范》.pdf"
    evidence = [
        _evidence("C1", DOC_A, "内容一。"),
        _evidence("C2", DOC_A, "内容二。"),
        _evidence("C5", DOC_B, "内容五。"),
        _evidence("C6", DOC_B, "内容六。"),
        _evidence("C10", green, "内容十。"),
        _evidence("C11", green, "内容十一。"),
    ]

    groups = multi_document_groups(evidence)

    assert [name for name, _ in groups] == [DOC_A, DOC_B]
