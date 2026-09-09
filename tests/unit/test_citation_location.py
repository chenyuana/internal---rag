import pytest

from app.services.citation_location import published_page_number
from app.services.citation_service import CitationService
from app.services.ragflow_client import RagflowClient

QUOTE = (
    "文档：CCAR-25-R4 运输类飞机适航标准.pdf\n"
    "章节：附录 B 图 3 成线性变化。 / 第 25.981 条 燃油箱点燃防护\n"
    "条号：25.981\n页码：123-124\n\n"
    "(c) 本条(b)不适用于采用减轻燃油蒸气点燃影响措施的燃油箱。"
)


@pytest.mark.parametrize(
    "quote, expected",
    [
        (QUOTE, 123),
        (QUOTE.replace("\n", " "), 123),
        (QUOTE.replace("123-124", "123–124"), 123),
        (QUOTE.replace("123-124", "124-123"), None),
        ("页码：123-124", None),
        ("文档：manual.pdf\n\n正文 页码：123-124", None),
        ("文档：manual.pdf\n页码：0", None),
    ],
)
def test_publisher_header_only(quote, expected):
    assert published_page_number(quote) == expected


def test_real_c1_shape_normalizes_and_reaches_citation():
    chunk = RagflowClient._normalize_chunk(
        {
            "id": "chunk",
            "doc_id": "ccar25",
            "kb_id": "kb",
            "content": QUOTE,
            "metadata": {"page_number": None},
        }
    )
    assert chunk.metadata.page_number == 123
    selected, citations = CitationService().build([chunk])
    assert selected[0].metadata.page_number == citations[0].page_number == 123
    # Also covers chunks supplied by retrieval paths other than RAGFlow normalization.
    chunk.metadata.page_number = None
    assert CitationService().build([chunk])[1][0].page_number == 123
    assert chunk.metadata.page_number is None


def test_explicit_page_wins_over_text_header():
    chunk = RagflowClient._normalize_chunk(
        {
            "id": "chunk",
            "doc_id": "doc",
            "kb_id": "kb",
            "content": QUOTE,
            "metadata": {"page_number": 125},
        }
    )
    assert chunk.metadata.page_number == 125
