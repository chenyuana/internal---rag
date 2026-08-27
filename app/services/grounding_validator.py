from __future__ import annotations

import re
import unicodedata

from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import SelectedChunk

# Numeric assertions that must be traceable to the cited chunks. We only check
# numbers with an explicit unit or percent sign: bare integers ("3 个油箱")
# are far too common to enforce. This catches the cross-document value swap
# (claim says "7%" while its cited chunk only contains "3%") without writing
# any regulation-specific dictionary.
_PERCENT_RE = re.compile(r"\d+(?:\.\d+)?\s*%")
_UNIT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*(?:"
    r"毫秒|ms|秒|s|分钟|min|小时|h|hz|khz|mhz|字节|bytes?"
    r"|米|m|千米|km|厘米|cm|毫米|mm"
    r"|伏特|v|千伏|kv|安培|a|瓦特|w|公斤|kg|克|g|吨|t"
    r")",
    re.IGNORECASE,
)
# Markdown 表格行：表格块的数据行是"裸数字"，单位在表头（如"地面分辨率值
# （cm）"），claim 中的"5cm"无法与"| 1:500 | 地裂缝 | 5 |"字面匹配。
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
# 拆分"数字+单位"标记，用于表格块的宽松 grounding。
_VALUE_UNIT_SPLIT_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*([a-zµμ°℃]+|[\u4e00-\u9fff]+)$",
    re.IGNORECASE,
)
# 中文单位 → 表头常用英文缩写。
_UNIT_ALIASES = {
    "厘米": "cm",
    "毫米": "mm",
    "千米": "km",
    "米": "m",
    "公里": "km",
    "公斤": "kg",
    "千克": "kg",
    "克": "g",
    "吨": "t",
    "毫升": "ml",
    "升": "l",
    "秒": "s",
    "分钟": "min",
    "小时": "h",
}


def _extract_numerical_marks(text: str) -> set[str]:
    """Extract normalized numeric assertions (percentages / unit values)."""
    normalized = unicodedata.normalize("NFKC", text)
    marks: set[str] = set()
    for match in _PERCENT_RE.finditer(normalized):
        marks.add(re.sub(r"\s+", "", match.group(0)))
    for match in _UNIT_RE.finditer(normalized):
        marks.add(re.sub(r"\s+", "", match.group(0)).lower())
    return marks


class ClaimGroundingValidator:
    """Reject claims whose numeric assertions are not supported by their
    citations.

    A claim that states "机队平均可燃性暴露水平不超过 7%" but cites chunks
    that never contain "7%" has an unfaithful citation: the number must come
    from the cited source. This is a general, regulation-agnostic grounding
    check (citation faithfulness) that complements the scope validator: scope
    decides *which document* a citation may point to, grounding decides whether
    the citation actually *supports the number* being claimed.

    The check is conservative: only percentages and numbers with an explicit
    unit are enforced, and a value is considered supported when it appears in
    *any* chunk cited by the claim.
    """

    def validate(
        self,
        answer: StructuredAnswer,
        *,
        chunks: list[SelectedChunk],
    ) -> None:
        chunk_by_citation = {item.citation_id: item for item in chunks}
        for claim in answer.claims:
            self._validate_claim(claim, chunk_by_citation)

    def _validate_claim(
        self,
        claim: CandidateClaim,
        chunk_by_citation: dict[str, SelectedChunk],
    ) -> None:
        claim_marks = _extract_numerical_marks(claim.claim)
        if not claim_marks:
            return
        supported: set[str] = set()
        found_chunk = False
        for citation_id in claim.citation_ids:
            chunk = chunk_by_citation.get(citation_id)
            if chunk is not None:
                found_chunk = True
                supported |= _extract_numerical_marks(chunk.text)
                supported |= self._table_encoded_marks(claim_marks, chunk.text)
        if not found_chunk:
            # Nothing to ground against; unknown citation ids are the
            # CitationValidator's responsibility.
            return
        unsupported = claim_marks - supported
        if not unsupported:
            return
        raise ValueError(
            f"claim {claim.claim_id} contains numeric values "
            f"{sorted(unsupported)} not found in its cited chunks"
        )

    @staticmethod
    def _table_encoded_marks(claim_marks: set[str], chunk_text: str) -> set[str]:
        """Accept numeric claims against table chunks whose rows store bare
        numbers and whose header carries the unit (e.g. ``| 1:500 | 地裂缝 |
        5 |`` under a ``地面分辨率值（cm）`` header).

        The value ``5cm`` cannot literally match ``5``, but the number is
        present in the table body and the unit is present in the table header,
        so the citation is faithful. Only applied to markdown-table chunks to
        avoid weakening prose grounding.
        """
        if not _TABLE_ROW_RE.search(chunk_text):
            return set()
        lowered = chunk_text.lower()
        supported: set[str] = set()
        for mark in claim_marks:
            match = _VALUE_UNIT_SPLIT_RE.match(mark)
            if match is None:
                continue
            number, unit = match.group(1), match.group(2).lower()
            if not re.search(rf"(?<!\d){re.escape(number)}(?!\d)", chunk_text):
                continue
            aliases = {unit, _UNIT_ALIASES.get(unit, "")} - {""}
            if any(alias in lowered for alias in aliases):
                supported.add(mark)
        return supported
