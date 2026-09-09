from types import SimpleNamespace

from app.ingestion.parsers.llm_structure import StructureEvent, apply_events


def _chunk(page_start: int, **overrides):
    base = {
        "chunk_id": f"c{page_start}",
        "title": "old-title",
        "text": "body",
        "page_start": page_start,
        "page_end": page_start,
        "section_path": ["Sec. 23.1587 Performance information."],
        "article_id_raw": "23.1587(D)",
        "article_id_normalized": "23.1587(D)",
        "article_aliases": ["23.1587(D)"],
        "keywords": ["23.1587"],
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_apply_events_reassigns_appendix_to_appendix_section() -> None:
    # 附录F 边界事件（annex）在 page59；page66 的准则块应归到附录F，而不是 23.1587。
    chunks = [_chunk(11), _chunk(66)]
    events = [
        StructureEvent(page=59, seq=0, kind="annex",
                       section_path=["Appendix F", "Part II"], article_id=None,
                       title="Appendix F -- Part II"),
    ]
    changed = apply_events(chunks, events)
    assert changed == 2
    # 附錄F准则块不再带 23.1587，归属到附录F。
    appendix = chunks[1]
    assert appendix.article_id_normalized is None
    assert appendix.article_id_raw is None
    assert appendix.section_path == ["Appendix F", "Part II"]
    assert "Appendix F" in appendix.keywords
    # 位于边界之前的正文块保持无节（不再被 23.1587 吞并）。
    assert chunks[0].section_path == []
    assert chunks[0].article_id_normalized is None


def test_body_event_does_not_open_a_section() -> None:
    # 前言里 "Section 23.571(d) still requires ..." 是 body（正文引用），不是节标题；
    # 它不得改变当前节、也不得给后续 chunk 附加伪条号。
    chunks = [_chunk(11)]
    events = [
        StructureEvent(page=11, seq=0, kind="body",
                       section_path=[], article_id=None,
                       title="Section 23.571(d) still requires the damage tolerance option"),
    ]
    apply_events(chunks, events)
    assert chunks[0].section_path == []
    assert chunks[0].article_id_normalized is None
    assert "23.571" not in " ".join(chunks[0].keywords)
