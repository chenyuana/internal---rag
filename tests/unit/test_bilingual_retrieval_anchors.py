from app.ingestion.publishers import RagflowPublisher
from app.ingestion.regulations import build_question_aliases, extract_keywords
from app.services.domain_terms import matched_concept_aliases

_REQUIREMENTS = """Appendix F to Part 23--Test Procedure
(h) Requirements.
(1) There must be no flame propagation beyond 2 inches (51 mm) to the left
of the centerline of the pilot flame application.
(2) The flame time after removal of the pilot burner may not exceed 3 seconds
on any thermal/acoustic insulation specimen.
(4) Report the after-flame time."""


def test_glossary_matches_slash_separated_english_compounds() -> None:
    aliases = matched_concept_aliases(_REQUIREMENTS)

    assert "thermal acoustic insulation" in aliases
    assert "热声绝缘材料" in aliases
    assert "flame propagation" in aliases
    assert "火焰传播" in aliases
    assert "余焰时间" in aliases


def test_english_acceptance_chunk_gets_controlled_chinese_anchors() -> None:
    keywords = extract_keywords(
        "Appendix F to Part 23--Test Procedure",
        ["Appendix F to Part 23--Test Procedure"],
        [],
        text=_REQUIREMENTS,
    )
    questions = build_question_aliases(
        None,
        "Appendix F to Part 23--Test Procedure",
        text=_REQUIREMENTS,
    )

    assert {"通过标准", "合格判据", "热声绝缘材料", "火焰传播", "余焰时间"} <= set(keywords)
    assert "Test" not in keywords
    assert "Procedure" not in keywords
    assert any("通过标准" in question for question in questions)


def test_ragflow_plan_keeps_english_evidence_and_indexes_anchors() -> None:
    keywords = extract_keywords(
        "Appendix F to Part 23--Test Procedure",
        ["Appendix F to Part 23--Test Procedure"],
        [],
        text=_REQUIREMENTS,
    )
    plan = RagflowPublisher.build_plan(
        dataset_id="kb-1",
        source_name="23-62-finalRule-Docket.pdf",
        source_sha256="abc",
        document_ir={
            "chunks": [
                {
                    "chunk_id": "requirements",
                    "title": "Appendix F to Part 23--Test Procedure",
                    "text": _REQUIREMENTS,
                    "page_start": 69,
                    "page_end": 69,
                    "section_path": ["Appendix F to Part 23--Test Procedure"],
                    "keywords": keywords,
                }
            ]
        },
    )
    payload = plan.chunks[0].payload

    assert _REQUIREMENTS in payload["content"]
    assert "检索锚点（非规范译文）" in payload["content"]
    assert "通过标准" in payload["important_keywords"]
    assert "Test" not in payload["important_keywords"]
    assert "Procedure" not in payload["important_keywords"]
