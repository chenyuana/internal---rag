from __future__ import annotations

import re

from app.core.config import GenerationSettings
from app.schemas.chat import EvidenceSentence
from app.schemas.retrieval import NormalizedQuery, SelectedChunk
from app.services.article_identity import owns_requested_article
from app.services.evidence_text import (
    has_parameter_value,
    is_parametric_text,
    overlap_score,
    split_sentences,
)

# 切块时写入的头部元数据行（文档名/章节/条号/页码），不是正文。这类行与查询
# 高度词汇重叠（如"文档：《早熟中稻无人机直播栽培技术规程》"含全部主体词），
# 在证据句子选择时会把真正的参数正文句挤出 top 名额，导致模型只见标题不见数值。
_META_PREFIX = re.compile(r"^(?:文档|章节|条号|页码|图表)[:：]")
# Markdown 表格行（含前导空格）。表格数据行与查询的词汇重叠率极低（多为数字与
# 短词），按普通句子切分会被 overlap 排序挤出 evidence，导致"表1 中的要求"有引用
# 无内容。合并为一条整体证据后按块内最高行打分保留。
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
# 条款内分项开头，如"a) 垂直影像："、"c) 机载激光雷达:"、"（一）通用要求："。
# 分项标记句独立成条。
_SUBITEM_HEAD_RE = re.compile(
    r"^[（(]?[a-zA-Z一二三四五六七八九十百\d][）)．.、]?\s*[^：:，,。！？!?]{0,12}[：:]"
)
# 分项承接句：切分会把"在丘陵山地地区，航向旁向重叠度50%~80%。"从
# "c) 机载激光雷达:…"中切出，使模型丢失适用对象归属（把 c 款数值当通用值）。
# 以承接词开头的句子并入前一分项。
_FOLLOW_ON_RE = re.compile(r"^(?:在|当|其|此|如|即|指|若|则)")


class EvidenceExtractor:
    """Select request-relevant source sentences without asking a model to invent evidence."""

    def __init__(self, settings: GenerationSettings) -> None:
        self._settings = settings

    def extract(
        self,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
    ) -> list[EvidenceSentence]:
        ranked: list[tuple[float, int, int, EvidenceSentence]] = []
        for chunk_rank, chunk in enumerate(chunks):
            if query.article_ids and owns_requested_article(chunk, query.article_ids):
                # A normative clause's intro, subordinate items and exceptions
                # form one evidence unit. Sentence ranking must not separate
                # a condition from the numeric requirement that it governs.
                body = "\n".join(
                    line for line in chunk.text.splitlines() if not _META_PREFIX.match(line)
                ).strip()
                if body:
                    ranked.append((1.0, chunk_rank, 0, EvidenceSentence(
                        citation_id=chunk.citation_id, chunk_id=chunk.chunk_id,
                        text=self._with_source(chunk.metadata.document_name, body,
                                               self._chunk_section_leaf(chunk)),
                        document_name=chunk.metadata.document_name, version=chunk.metadata.version,
                    )))
                    continue
            sentences = self._merge_subitem_follow_ons(
                self._merge_table_rows(
                    [
                        sentence
                        for sentence in split_sentences(chunk.text)
                        if not _META_PREFIX.match(sentence)
                    ]
                )
            ) or split_sentences(chunk.text)
            candidates = [
                (
                    self._sentence_score(query, sentence),
                    sentence_index,
                    sentence,
                )
                for sentence_index, sentence in enumerate(sentences)
            ]
            candidates.sort(key=lambda item: (-item[0], item[1]))
            selected = [
                item
                for item in candidates
                if item[0] >= self._settings.min_evidence_overlap
            ][: self._settings.max_sentences_per_citation]
            if not selected and candidates:
                selected = candidates[:1]
            for score, sentence_index, sentence in selected:
                ranked.append(
                    (
                        score,
                        chunk_rank,
                        sentence_index,
                        EvidenceSentence(
                            citation_id=chunk.citation_id,
                            chunk_id=chunk.chunk_id,
                            # 证据文本内联来源与环节标识：同一主题多份规程
                            # （同名不同时间/地区）需区分文档；"成果/产出"类
                            # 问题需区分环节归属（作业准备收集的辅助资料不是
                            # 内业产出）。
                            text=self._with_source(
                                chunk.metadata.document_name,
                                sentence,
                                self._chunk_section_leaf(chunk),
                            ),
                            document_name=chunk.metadata.document_name,
                            version=chunk.metadata.version,
                        ),
                    )
                )
        ranked.sort(key=lambda item: (-item[0], item[1], item[2]))
        limit = (
            self._settings.enumeration_max_evidence_sentences
            if query.query_type == "enumeration"
            else self._settings.max_evidence_sentences
        )
        return [
            item[3]
            for item in ranked[:limit]
        ]

    @staticmethod
    def _with_source(document_name: str | None, sentence: str, section: str | None = None) -> str:
        """给证据句加来源与环节前缀。

        格式如"[来源：20241217：…] [8 数据处理/8.1 空中三角测量] 空中三角
        测量作业流程…"。环节标注帮助模型区分内容归属（如"作业准备"收集的
        辅助资料不得当成"内业产出"）；无文档名/无章节时不加对应段。
        """
        prefix = ""
        if document_name:
            prefix += f"[来源：{document_name}]"
        if section:
            prefix += f"[{section}]"
        if prefix:
            return f"{prefix} {sentence}"
        return sentence

    @staticmethod
    def _chunk_section_leaf(chunk: SelectedChunk) -> str | None:
        """从 chunk 文本头部的"章节："行取章节路径（如"8 数据处理/8.1 空中
        三角测量"），压缩斜杠空格，无则 None。"""
        match = re.search(r"^章节：\s*(.+)$", chunk.text, re.MULTILINE)
        if match is None:
            return None
        leaf = re.sub(r"\s*/\s*", "/", match.group(1).strip())
        return leaf if leaf else None

    @staticmethod
    def _merge_table_rows(sentences: list[str]) -> list[str]:
        """Merge consecutive Markdown table rows into a single evidence sentence."""
        merged: list[str] = []
        buffer: list[str] = []
        for sentence in sentences:
            if _TABLE_ROW_RE.match(sentence):
                buffer.append(sentence)
            else:
                if buffer:
                    merged.append("\n".join(buffer))
                    buffer = []
                merged.append(sentence)
        if buffer:
            merged.append("\n".join(buffer))
        return merged

    @staticmethod
    def _merge_subitem_follow_ons(sentences: list[str]) -> list[str]:
        """Merge follow-on sentences back into their numbered sub-item.

        Sentence splitting on ``；``/``。`` can tear "在丘陵山地地区，航向旁向
        重叠度50%~80%。" away from "c) 机载激光雷达:旁向重叠度不小于30%;",
        so the model sees the value without its applicability subject and may
        generalize it. Follow-on openers (在/当/其/此/如…) are joined to the
        preceding sub-item head so the applicability label is preserved.
        """
        merged: list[str] = []
        for sentence in sentences:
            if _SUBITEM_HEAD_RE.match(sentence) or not merged:
                merged.append(sentence)
                continue
            # 仅当紧邻的前一句本身是分项头（如"c) 机载激光雷达:…"）时，
            # 承接句才并入；普通句子后的"其他/其中/其余…"不合并。
            if _FOLLOW_ON_RE.match(sentence) and _SUBITEM_HEAD_RE.match(merged[-1]):
                merged[-1] = f"{merged[-1]}{sentence}"
                continue
            merged.append(sentence)
        return merged

    @staticmethod
    def _sentence_score(query: NormalizedQuery, sentence: str) -> float:
        first_line = sentence.split("\n", 1)[0]
        if _TABLE_ROW_RE.match(first_line):
            # 表格块：用块内与查询最相关的一行打分，避免整块含大量数字/短词
            # 稀释 overlap 比例；同时表格通常是对正文"见表X"的具体化，需保留。
            lines = sentence.split("\n")
            score = max(overlap_score(query.normalized_query, line) for line in lines)
            if query.exact_tokens and any(token in sentence for token in query.exact_tokens):
                score += 1
            if is_parametric_text(query.normalized_query) and has_parameter_value(sentence):
                score += 1.0
            return score
        score = overlap_score(query.normalized_query, sentence)
        if query.exact_tokens and any(token in sentence for token in query.exact_tokens):
            score += 1
        # 参数类问题优先携带具体数值/单位/公式定义的句子，避免短标题行
        # （如"5.3 播种量"）靠纯词汇重叠占满名额而挤掉真正的参数正文句。
        if is_parametric_text(query.normalized_query) and has_parameter_value(sentence):
            score += 1.0
        return score
