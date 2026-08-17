from __future__ import annotations

import re

from app.schemas.chat import CandidateClaim, EvidenceSentence, StructuredAnswer
from app.schemas.retrieval import SelectedChunk

CITATION_MARKER = re.compile(r"\[(C[1-9]\d*)\]")

# Claim 文本中用于声明"资料缺失/未说明"的语义标记。允许这类 claim 不带引用，
# 因为"当前资料中未明确说明 X"是有信息量的诚实回答，不应因缺引用导致整个回答失败。
MISSING_SEMANTIC_MARKERS = (
    "未明确",
    "未说明",
    "未找到",
    "未提供",
    "未提及",
    "无资料",
    "没有找到",
    "没有资料",
    "不清楚",
    "无法对比",
    "缺乏",
)


class CitationValidator:
    """Reject model-created or out-of-context citation identifiers."""

    def __init__(self, *, require_citations: bool) -> None:
        self._require_citations = require_citations

    def validate(
        self,
        answer: StructuredAnswer,
        *,
        evidence: list[EvidenceSentence],
        chunks: list[SelectedChunk],
        include_historical: bool,
    ) -> None:
        evidence_ids = {item.citation_id for item in evidence}
        chunk_by_citation = {item.citation_id: item for item in chunks}
        answer_markers = set(CITATION_MARKER.findall(answer.answer))

        unknown_markers = answer_markers - evidence_ids
        if unknown_markers:
            raise ValueError(
                f"answer references citations outside the current evidence: "
                f"{sorted(unknown_markers)}"
            )

        cited_by_claim: set[str] = set()
        for claim in answer.claims:
            claim_citations = set(claim.citation_ids)
            if not claim_citations:
                if not self._require_citations:
                    continue
                if not self._is_missing_semantic_claim(claim):
                    raise ValueError(f"claim {claim.claim_id} has no citation")
                continue
            unknown = claim_citations - evidence_ids
            if unknown:
                raise ValueError(
                    f"claim {claim.claim_id} references unknown citations: {sorted(unknown)}"
                )
            cited_by_claim.update(claim_citations)

        if self._require_citations and answer.claims:
            missing_markers = cited_by_claim - answer_markers
            if missing_markers:
                raise ValueError(
                    f"claim citations are not present in answer text: {sorted(missing_markers)}"
                )

        for citation_id in cited_by_claim | answer_markers:
            chunk = chunk_by_citation.get(citation_id)
            if chunk is None:
                raise ValueError(f"citation {citation_id} has no backend chunk mapping")
            if not include_historical and chunk.metadata.status is not None and chunk.metadata.status != "effective":
                raise ValueError(f"citation {citation_id} is not from an effective document")

    @staticmethod
    def _is_missing_semantic_claim(claim: CandidateClaim) -> bool:
        """Return True when a citation-less claim honestly reports missing data.

        A claim that asserts "the material does not state X" is useful output and
        should not fail the whole answer. We accept it only when its wording is
        clearly about absence (missing markers present). Concrete factual claims
        without citations remain rejected.
        """
        return any(marker in claim.claim for marker in MISSING_SEMANTIC_MARKERS)
