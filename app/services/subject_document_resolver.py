from __future__ import annotations

import re
from collections import defaultdict

from app.schemas.planning import SubjectResolution
from app.schemas.retrieval import RetrievedChunk
from app.services.requirement_taxonomy import SUBJECT_RETRIEVAL_TERMS

NON_TITLE_TEXT_RE = re.compile(r"[^0-9A-Za-z\u4e00-\u9fff]+")

# 检索分数通道：discovery 检索里，目标文档的 chunk 最高 hybrid 分明显领先
# 其它文档时，优先用检索分数决定归属。title 相似度只用于两件事：
# 1) 作为门控，防止"检索分高但与主体字面无关"的文档被选中（如"管理要求"
#    检索时《机巢通用管理要求》检索分也高，但其标题与主体词的语义重合度
#    不足以通过门控时仍由 title 通道裁决）；
# 2) 检索分数不分胜负时回退 title 通道。
# 阈值来自 2026-08-21 真实实验：
#   目标文档   max hybrid  领先第二名   title 相似度
#   播种治沙   0.4584       0.1077       0.492
#   应用服务   0.4273       0.0204       0.45（对"通用服务管理要求"）
#   长大桥梁   0.4361       唯一          高
#   无证据题   0.2619       无           ——（远低于下限，不触发）
_RETRIEVAL_MIN_TOP_SCORE = 0.35
_RETRIEVAL_MIN_MARGIN = 0.015
_RETRIEVAL_TITLE_GATE = 0.30


def _normalized(value: str) -> str:
    return NON_TITLE_TEXT_RE.sub("", value).lower()


def _ngrams(value: str, size: int = 2) -> set[str]:
    normalized = _normalized(value)
    if len(normalized) < size:
        return {normalized} if normalized else set()
    return {normalized[index : index + size] for index in range(len(normalized) - size + 1)}


def title_similarity(subject: str, title: str) -> float:
    subject_grams = _ngrams(subject)
    title_grams = _ngrams(title)
    if not subject_grams or not title_grams:
        return 0.0
    shared = subject_grams & title_grams
    coverage = len(shared) / len(subject_grams)
    jaccard = len(shared) / len(subject_grams | title_grams)
    return 0.7 * coverage + 0.3 * jaccard


def subject_title_similarity(subject: str, title: str) -> float:
    """Score titles with the same bounded vocabulary used for discovery.

    User-facing subjects can be intentionally colloquial (``水稻植保无人机``)
    while the specialist title uses a pest name (``稻纵卷叶螟``). The expanded
    score is only used as an identity gate; retrieval still has to provide a
    clear score leader before a document is selected.
    """
    expansion = SUBJECT_RETRIEVAL_TERMS.get(subject)
    expanded = f"{subject} {expansion}" if expansion else subject
    return title_similarity(expanded, title)


class SubjectDocumentResolver:
    """Resolve a subject to documents using metadata and whole-title similarity."""

    def resolve(
        self,
        subject: str,
        candidates: list[RetrievedChunk],
        *,
        requested_document_ids: list[str] | None = None,
    ) -> SubjectResolution:
        if requested_document_ids:
            return SubjectResolution(
                subject=subject,
                document_ids=list(dict.fromkeys(requested_document_ids)),
                confidence=1.0,
                source="request",
            )
        by_document: dict[str, list[RetrievedChunk]] = defaultdict(list)
        for candidate in candidates:
            by_document[candidate.document_id].append(candidate)

        metadata_matches: list[tuple[str, str]] = []
        for document_id, chunks in by_document.items():
            document_subjects = {
                _normalized(chunk.metadata.document_subject or "") for chunk in chunks
            }
            if _normalized(subject) in document_subjects:
                metadata_matches.append(
                    (document_id, chunks[0].metadata.document_name or document_id)
                )
        if metadata_matches:
            return SubjectResolution(
                subject=subject,
                document_ids=[item[0] for item in metadata_matches],
                document_names=[item[1] for item in metadata_matches],
                confidence=1.0,
                source="metadata",
            )

        retrieval_leader = self._retrieval_leader(subject, by_document)
        if retrieval_leader is not None:
            return retrieval_leader

        scored: list[tuple[float, str, str]] = []
        for document_id, chunks in by_document.items():
            title = chunks[0].metadata.document_name or ""
            scored.append((subject_title_similarity(subject, title), document_id, title))
        scored.sort(reverse=True)
        if not scored or scored[0][0] < 0.38:
            return SubjectResolution(
                subject=subject,
                document_ids=[],
                document_names=[],
                confidence=scored[0][0] if scored else 0.0,
                source="none",
            )
        best = scored[0][0]
        selected = [item for item in scored if item[0] >= max(0.38, best - 0.08)]
        return SubjectResolution(
            subject=subject,
            document_ids=[item[1] for item in selected],
            document_names=[item[2] for item in selected],
            confidence=best,
            source="title",
        )

    @staticmethod
    def _retrieval_leader(
        subject: str,
        by_document: dict[str, list[RetrievedChunk]],
    ) -> SubjectResolution | None:
        """Return a resolution when retrieval scores unambiguously lead.

        The discovery retrieval was issued with the subject itself, so the
        document whose chunks score highest is what RAGFlow semantically
        associates with the subject.  This channel fixes declarative subjects
        like "无人机通用服务管理要求" where the intended document
        《无人机应用服务通用规范》 loses to a literal-title match
        《微轻小型无人机机巢通用管理要求》 on bigram similarity alone.
        """
        scored: list[tuple[float, float, str, str]] = []
        for document_id, chunks in by_document.items():
            scores = [chunk.hybrid_score for chunk in chunks if chunk.hybrid_score > 0]
            if not scores:
                continue
            title = chunks[0].metadata.document_name or ""
            scored.append(
                (max(scores), sum(scores) / len(scores), document_id, title)
            )
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        top_score = scored[0][0]
        if top_score < _RETRIEVAL_MIN_TOP_SCORE:
            return None
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if top_score - second_score < _RETRIEVAL_MIN_MARGIN:
            # Retrieval cannot separate the leader from the pack; let the
            # title channel decide instead of picking a coin-flip winner.
            return None
        document_id = scored[0][2]
        title = scored[0][3]
        title_score = subject_title_similarity(subject, title)
        if title_score < _RETRIEVAL_TITLE_GATE:
            # The top retrieval hit is not even loosely the same topic as the
            # subject.  Do not trust retrieval alone; fall back to the title
            # channel (which may still return none).
            return None
        return SubjectResolution(
            subject=subject,
            document_ids=[document_id],
            document_names=[title],
            confidence=round(top_score, 3),
            source="retrieval",
        )
