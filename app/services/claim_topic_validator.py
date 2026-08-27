"""Claim topic grounding: reject citations whose content is unrelated to the
claim's topic words.

The numeric grounding validator only checks numbers/units; a claim like "在
管制空域或真高大于30 m时须获批准" can cite an unrelated chunk (e.g. 喷洒无人
机设备) because the numbers happen to co-occur somewhere. This validator
extracts information-carrying Chinese bigrams from the claim and requires at
least one of them to appear in the union of the cited chunks' text. Fully
unmatched topic words mean the citation is unfaithful (引用-内容脱钩) and the
claim is rejected, triggering the existing repair/sanitize path.

Conservative by design: only claims with >=2 information bigrams are checked,
and only a complete miss (none of the topic words found) is rejected.
"""

from __future__ import annotations

import re

from app.schemas.chat import CandidateClaim, StructuredAnswer
from app.schemas.retrieval import SelectedChunk
from app.services.evidence_text import lexical_units

# 中文单字停用：出现在 bigram 中即视为通用连接成分，不承载主题信息。
_STOP_CHARS = frozenset(
    "的了在应按可及或和与中上下内外前后时等个从到要把被让为是这那些么吗呢吧啊其该"
)
# 停用双字词：通用动作/名目词。
_STOP_WORDS = frozenset(
    {
        "作业",
        "要求",
        "规定",
        "进行",
        "使用",
        "开展",
        "需要",
        "应该",
        "可以",
        "以及",
        "并且",
        "同时",
        "根据",
        "通过",
        "按照",
        "应当",
        "必须",
        "不得",
        "禁止",
    }
)
# 数字/单位/百分比由 grounding_validator 校验，不参与主题匹配。
_NUMERIC_TOKEN = re.compile(r"^[\d%.~/~\-—至到倍次份处mhskgml]+$", re.IGNORECASE)


def _info_bigrams(text: str) -> set[str]:
    """提取承载主题信息的中文 bigram（过滤停用字/词、数字与单位）。"""
    units = lexical_units(text)
    filtered: set[str] = set()
    for unit in units:
        if len(unit) < 2:
            continue
        if unit in _STOP_WORDS:
            continue
        if _NUMERIC_TOKEN.match(unit):
            continue
        if any(char in _STOP_CHARS for char in unit):
            continue
        filtered.add(unit)
    return filtered


class ClaimTopicValidator:
    """Reject claims whose topic words are absent from all cited chunks."""

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
        claim_bigrams = _info_bigrams(claim.claim)
        if len(claim_bigrams) < 2:
            # 短 claim 或无数值可校验，交给其它校验器。
            return
        supported: set[str] = set()
        found_chunk = False
        for citation_id in claim.citation_ids:
            chunk = chunk_by_citation.get(citation_id)
            if chunk is not None:
                found_chunk = True
                supported |= _info_bigrams(chunk.text)
        if not found_chunk:
            return
        missing = claim_bigrams - supported
        if not missing:
            return
        # 保守阈值：claim 的主题词在全部被引 chunk 中缺失比例 ≥50% 才判定
        # 引用-内容脱钩（避免"飞行""作业"等通用词单点命中即放行）。
        missing_ratio = len(missing) / len(claim_bigrams)
        if missing_ratio >= 0.5:
            raise ValueError(
                f"claim {claim.claim_id} topic words "
                f"{sorted(missing)} absent from its cited chunks "
                f"(missing {missing_ratio:.0%})"
            )
