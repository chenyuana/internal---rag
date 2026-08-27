from __future__ import annotations

import json
import re
from collections.abc import Callable

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.chat import (
    CandidateClaim,
    EvidenceAssessment,
    EvidenceSentence,
    ReferenceDocument,
    StructuredAnswer,
)
from app.schemas.documents import (
    DocumentSectionGenerateRequest,
    DocumentSectionGenerateResponse,
    GuidanceSuggestRequest,
    GuidanceSuggestResponse,
    SectionGuidanceItem,
)
from app.schemas.retrieval import RetrievalSearchRequest, SelectedChunk
from app.services.answer_generator import (
    AnswerGenerator,
    AnswerModel,
    SanitizedAnswer,
)
from app.services.answer_model_manager import AnswerModelManager
from app.services.citation_validator import CITATION_MARKER, CitationValidator
from app.services.evidence_extractor import EvidenceExtractor
from app.services.evidence_judge import EvidenceJudge
from app.services.grounding_validator import ClaimGroundingValidator
from app.services.reference_document import build_auxiliary_evidence
from app.services.retrieval_service import RetrievalExecution, RetrievalService
from app.services.scope_validator import ScopeConsistencyValidator


class DocumentSectionGenerator:
    """Generate one evidence-grounded long-form document section."""

    #: 表格/矩阵类章节：篇幅以要素完整覆盖为准，不设字数建议线。
    TABLE_LIKE_SECTIONS = frozenset({"符合性验证矩阵"})
    #: 各篇幅等级的建议正文基准字数（仅作软性建议，不构成硬门禁）。
    LENGTH_BASE_CHARS = {"brief": 350, "standard": 550, "detailed": 800}

    def __init__(
        self,
        *,
        settings: Settings,
        retrieval: RetrievalService,
        answer_model: AnswerModel | None,
        answer_models: AnswerModelManager | None = None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._answer_model = answer_model
        self._answer_models = answer_models
        document_settings = settings.generation.model_copy(
            update={
                "answer_prompt_path": settings.generation.document_section_prompt_path,
                "prompt_version": settings.generation.document_section_prompt_version,
                # Long-form sections are salvaged claim-by-claim after the first
                # response. Re-asking the small local model to rewrite the whole
                # section is slow and commonly introduces a new schema error.
                "json_repair_attempts": 0,
                "max_evidence_sentences": max(
                    settings.generation.max_evidence_sentences,
                    24,
                ),
            }
        )
        self._document_settings = document_settings
        self._judge = EvidenceJudge(document_settings)
        self._extractor = EvidenceExtractor(document_settings)
        self._citation_validator = CitationValidator(require_citations=True)
        self._grounding_validator = ClaimGroundingValidator()
        self._scope_validator = ScopeConsistencyValidator()

    async def generate(
        self,
        request: DocumentSectionGenerateRequest,
        *,
        user_id: str,
    ) -> DocumentSectionGenerateResponse:
        retrieval_query = self._retrieval_query(request)
        execution = await self._retrieval.execute(
            RetrievalSearchRequest(
                query=retrieval_query,
                knowledge_base_ids=request.knowledge_base_ids,
                filters=request.filters,
                candidate_top_k=40,
            ),
            user_id=user_id,
        )
        for document in request.supplemental_documents:
            self._attach_supplemental_document(execution, document)
        self._focus_section_evidence(request, execution)

        assessment = self._judge.assess(execution.query, execution.selected_chunks)
        evidence = self._extractor.extract(execution.query, execution.selected_chunks)
        if assessment.status in {"UNANSWERABLE", "CONFLICTED"} or not evidence:
            return DocumentSectionGenerateResponse(
                request_id=execution.request_id,
                status=assessment.status,
                content="",
                missing_information=self._unique(
                    assessment.missing_requirements
                    or [f"当前资料不足以编制“{request.section_title}”章节。"]
                ),
                prompt_version=self._document_settings.prompt_version,
                evidence_document_count=self._document_count(execution.selected_chunks),
                evidence_sentence_count=len(evidence),
                target_min_chars=self._length_target(request, len(evidence)),
                advisory_warnings=["当前证据不足，未进入章节正文生成。"],
            )

        model, source_id, model_name = await self._resolve_model(request, user_id=user_id)
        generator = AnswerGenerator(settings=self._document_settings, model=model)
        validate = self._validator(
            assessment=assessment,
            execution=execution,
            evidence=evidence,
            include_historical=request.filters.include_historical,
        )
        generated = await generator.generate(
            question=self._authoring_instruction(
                request,
                evidence_sentence_count=len(evidence),
            ),
            assessment=assessment,
            evidence=evidence,
            semantic_validator=validate,
            semantic_sanitizer=self._sanitizer(
                assessment=assessment,
                execution=execution,
                evidence=evidence,
                include_historical=request.filters.include_historical,
            ),
        )
        answer = generated.answer
        quality_passed, char_count, target_min, hard_warnings, advisory = (
            self._quality_gate(
                request,
                answer.answer,
                evidence_sentence_count=len(evidence),
            )
        )
        used = set(CITATION_MARKER.findall(answer.answer))
        used.update(
            citation_id
            for claim in answer.claims
            for citation_id in claim.citation_ids
        )
        return DocumentSectionGenerateResponse(
            request_id=execution.request_id,
            status=(
                answer.answerability
                if quality_passed or answer.answerability == "UNANSWERABLE"
                else "PARTIALLY_ANSWERABLE"
            ),
            content=answer.answer,
            claims=answer.claims,
            citations=[item for item in execution.citations if item.citation_id in used],
            missing_information=self._unique(
                [
                    *assessment.missing_requirements,
                    *answer.missing_information,
                    *(generated.validation_warnings or []),
                    *hard_warnings,
                ]
            ),
            validation_degraded=generated.validation_degraded,
            validation_warnings=generated.validation_warnings or [],
            quality_passed=quality_passed,
            generated_char_count=char_count,
            target_min_chars=target_min,
            coverage_warnings=hard_warnings,
            advisory_warnings=advisory,
            prompt_version=self._document_settings.prompt_version,
            model_source_id=source_id,
            model_name=model_name,
            evidence_document_count=self._document_count(execution.selected_chunks),
            evidence_sentence_count=len(evidence),
        )

    async def _resolve_model(
        self,
        request: DocumentSectionGenerateRequest | GuidanceSuggestRequest,
        *,
        user_id: str,
    ) -> tuple[AnswerModel, str | None, str | None]:
        if request.model is not None:
            if self._answer_models is None:
                raise AppError(
                    code="MODEL_SWITCHING_UNAVAILABLE",
                    message="当前服务未启用运行时模型切换。",
                    status_code=503,
                )
            model = await self._answer_models.resolve(request.model, user_id=user_id)
            return model, request.model.source_id, request.model.model_name
        if self._answer_model is None:
            raise AppError(
                code="ANSWER_MODEL_DISABLED",
                message="文档章节生成需要可用的答案模型。",
                status_code=503,
            )
        return (
            self._answer_model,
            "configured-answer",
            self._settings.models.answer.model_name,
        )

    async def suggest_guidances(
        self,
        request: GuidanceSuggestRequest,
        *,
        user_id: str,
    ) -> GuidanceSuggestResponse:
        """用章节生成模型为各章节撰写“本节编制要求”。

        模型只负责起草，结果按输入章节标题映射返回；未生成的章节保留原
        guidance，由前端/作者最终确认后再写入草稿。
        """
        model, _, _ = await self._resolve_model(request, user_id=user_id)
        messages = [
            {
                "role": "system",
                "content": "你是专业文档编制助手，为技术文档的每个章节撰写“本节编制要求”。",
            },
            {"role": "user", "content": self._guidance_instruction(request)},
        ]
        response_schema = {
            "type": "object",
            "properties": {
                "sections": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "guidance": {"type": "string"},
                        },
                        "required": ["title", "guidance"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["sections"],
            "additionalProperties": False,
        }
        raw = await model.chat_completion(
            messages=messages,
            response_schema=response_schema,
        )
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise AppError(
                code="GUIDANCE_SUGGEST_INVALID_JSON",
                message="编制要求生成结果解析失败，请重试。",
                status_code=502,
            ) from exc
        by_title: dict[str, str] = {}
        for item in payload.get("sections") or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            guidance = str(item.get("guidance") or "").strip()
            if title and guidance and title not in by_title:
                by_title[title] = guidance[:1000]
        sections: list[SectionGuidanceItem] = []
        for section in request.sections:
            generated = by_title.get(section.title.strip())
            sections.append(
                SectionGuidanceItem(
                    title=section.title,
                    guidance=generated or section.guidance,
                )
            )
        if not any(item.guidance for item in sections):
            raise AppError(
                code="GUIDANCE_SUGGEST_EMPTY",
                message="模型未生成有效的编制要求，请重试或手动编写。",
                status_code=502,
            )
        return GuidanceSuggestResponse(sections=sections)

    @staticmethod
    def _guidance_instruction(request: GuidanceSuggestRequest) -> str:
        numbered = "\n".join(
            f"{index}. {item.title}"
            + (f"（现有要求：{item.guidance}）" if item.guidance else "")
            for index, item in enumerate(request.sections, start=1)
        )
        return (
            f"为《{request.document_title}》的各章节撰写“本节编制要求”。\n"
            f"文档目标：{request.objective}\n"
            f"使用对象：{request.audience or '未指定'}\n"
            f"结构文件/格式模板摘录（仅作章节上下文参考，勿照抄）："
            f"{request.template_preview or '无'}\n"
            "章节列表：\n"
            f"{numbered}\n"
            "要求：\n"
            "- 每章 1-3 句，具体、可执行，用于约束章节正文的专业内容、边界、证据与输出要求；\n"
            "- 紧扣章节标题与文档目标；涉及数值、条款、判据的章节必须要求保留知识库引用；\n"
            "- 不修改章节标题、不新增或删除章节；\n"
            "- 只输出 JSON：{\"sections\": [{\"title\": \"章节标题\", \"guidance\": \"编制要求\"}]}"
        )

    def _validator(
        self,
        *,
        assessment: EvidenceAssessment,
        execution: RetrievalExecution,
        evidence: list[EvidenceSentence],
        include_historical: bool,
    ) -> Callable[[StructuredAnswer], None]:
        def validate(answer: StructuredAnswer) -> None:
            if assessment.status == "PARTIALLY_ANSWERABLE" and answer.answerability == "ANSWERABLE":
                raise ValueError("partial evidence cannot produce an ANSWERABLE section")
            self._validate_evidence_consistency(
                answer,
                execution=execution,
                evidence=evidence,
                include_historical=include_historical,
            )

        return validate

    def _sanitizer(
        self,
        *,
        assessment: EvidenceAssessment,
        execution: RetrievalExecution,
        evidence: list[EvidenceSentence],
        include_historical: bool,
    ) -> Callable[[StructuredAnswer, Exception], SanitizedAnswer]:
        def sanitize(
            answer: StructuredAnswer,
            validation_error: Exception,
        ) -> SanitizedAnswer:
            valid_claims: list[CandidateClaim] = []
            rejected_claim_ids: list[str] = []
            scope_context = CITATION_MARKER.sub("", answer.answer).strip()
            for claim in answer.claims:
                claim_text = CITATION_MARKER.sub("", claim.claim).strip()
                markers = "".join(
                    f"[{citation_id}]"
                    for citation_id in dict.fromkeys(claim.citation_ids)
                )
                candidate = StructuredAnswer(
                    answerability="PARTIALLY_ANSWERABLE",
                    answer=f"{scope_context}\n{claim_text}{markers}",
                    claims=[claim.model_copy(update={"claim": claim_text})],
                    missing_information=[],
                    conflicts=[],
                )
                try:
                    self._validate_evidence_consistency(
                        candidate,
                        execution=execution,
                        evidence=evidence,
                        include_historical=include_historical,
                    )
                except ValueError:
                    rejected_claim_ids.append(claim.claim_id)
                    continue
                valid_claims.append(candidate.claims[0])

            warnings = [
                f"生成结果未通过整体证据校验：{validation_error}",
            ]
            if rejected_claim_ids:
                warnings.append(
                    "已省略未通过证据校验的结论："
                    + "、".join(rejected_claim_ids)
                )
            if not valid_claims:
                return SanitizedAnswer(
                    answer=StructuredAnswer(
                        answerability="UNANSWERABLE",
                        answer="",
                        claims=[],
                        missing_information=self._unique(
                            [
                                *assessment.missing_requirements,
                                "本节生成内容均未通过逐条证据校验，未写入正文。",
                            ]
                        ),
                        conflicts=answer.conflicts,
                    ),
                    warnings=warnings,
                )

            paragraphs = []
            for claim in valid_claims:
                claim_text = claim.claim.rstrip("。；; ")
                markers = "".join(
                    f"[{citation_id}]"
                    for citation_id in dict.fromkeys(claim.citation_ids)
                )
                paragraphs.append(f"{claim_text}。{markers}")
            return SanitizedAnswer(
                answer=StructuredAnswer(
                    answerability="PARTIALLY_ANSWERABLE",
                    answer="\n\n".join(paragraphs),
                    claims=valid_claims,
                    missing_information=self._unique(
                        [
                            *assessment.missing_requirements,
                            *answer.missing_information,
                            "部分生成内容未通过逐条证据校验，已省略。",
                        ]
                    ),
                    conflicts=answer.conflicts,
                ),
                warnings=warnings,
            )

        return sanitize

    def _validate_evidence_consistency(
        self,
        answer: StructuredAnswer,
        *,
        execution: RetrievalExecution,
        evidence: list[EvidenceSentence],
        include_historical: bool,
    ) -> None:
        if answer.answerability not in {"ANSWERABLE", "PARTIALLY_ANSWERABLE"}:
            return
        self._citation_validator.validate(
            answer.model_copy(
                update={
                    "claims": [
                        claim
                        for claim in answer.claims
                        if not self._is_document_control_claim(claim)
                    ]
                }
            ),
            evidence=evidence,
            chunks=execution.selected_chunks,
            include_historical=include_historical,
        )
        self._grounding_validator.validate(
            answer,
            chunks=execution.selected_chunks,
        )
        self._scope_validator.validate(
            self._scope_validation_answer(answer),
            query=execution.query,
            chunks=execution.selected_chunks,
        )

    @staticmethod
    def _scope_validation_answer(answer: StructuredAnswer) -> StructuredAnswer:
        """Exclude document heading numbers from regulation article detection.

        Long-form sections commonly contain headings such as ``1.1`` and
        ``2.3``. The Q&A scope validator intentionally treats such tokens as
        possible article references, so feeding it the complete rendered body
        can create false cross-document failures. Candidate claims retain real
        standard/article references and are the appropriate validation surface.
        """
        claim_text = "\n".join(claim.claim for claim in answer.claims)
        return answer.model_copy(update={"answer": claim_text})

    @staticmethod
    def _is_document_control_claim(claim: CandidateClaim) -> bool:
        """Allow uncited authoring controls while keeping factual claims strict."""
        if claim.citation_ids:
            return False
        control_markers = (
            "本文档",
            "本文件",
            "本章节",
            "本章",
            "编制过程",
            "证据使用",
            "引用",
            "待补",
            "资料缺失",
        )
        return any(marker in claim.claim for marker in control_markers)

    @staticmethod
    def _focus_section_evidence(
        request: DocumentSectionGenerateRequest,
        execution: RetrievalExecution,
    ) -> None:
        """Keep overview sections anchored to direct unmanned-aircraft sources."""
        if request.section_title.strip() != "编制说明":
            return
        subject_markers = ("无人机", "无人直升机", "无人航空器", "UHS", "UAS")
        focused = [
            chunk
            for chunk in execution.selected_chunks
            if any(
                marker.lower()
                in f"{chunk.metadata.document_name or ''}\n{chunk.text}".lower()
                for marker in subject_markers
            )
        ]
        if not focused:
            return
        citation_ids = {chunk.citation_id for chunk in focused}
        execution.selected_chunks = focused
        execution.citations = [
            citation
            for citation in execution.citations
            if citation.citation_id in citation_ids
        ]

    @staticmethod
    def _retrieval_query(request: DocumentSectionGenerateRequest) -> str:
        section_hints = {
            "适航审定依据与审定基础": (
                "CCAR-21 型号合格证 申请 审定基础 适航标准 专用条件 "
                "等效安全水平 豁免 适航指令 中高风险无人直升机"
            ),
            "系统级安全性要求": (
                "功能危险 失效状态 灾难性 危险 严重 重大 概率 安全目标 "
                "共因失效 独立性"
            ),
            "结构、材料与载荷": (
                "限制载荷 极限载荷 安全系数 疲劳 损伤容限 振动 材料 "
                "结构试验"
            ),
            "动力、能源与推进系统": (
                "发动机 旋翼 螺旋桨 动力装置 燃油 电池 能源 热安全 "
                "推进失效"
            ),
        }
        return " ".join(
            part.strip()
            for part in (
                request.document_title,
                request.section_title,
                request.section_guidance,
                section_hints.get(request.section_title.strip(), ""),
            )
            if part.strip()
        )[:2000]

    @classmethod
    def _authoring_instruction(
        cls,
        request: DocumentSectionGenerateRequest,
        *,
        evidence_sentence_count: int = 0,
    ) -> str:
        audience = request.audience or "未指定"
        length_labels = {
            "brief": "精简",
            "standard": "标准",
            "detailed": "详细",
        }
        if request.section_title.strip() in cls.TABLE_LIKE_SECTIONS:
            length_clause = (
                f"篇幅等级：{length_labels[request.target_length]}；"
                "本节以要素完整覆盖为准（要求、来源、符合性方法、条件输入、判据、"
                "记录状态等），不以字数衡量篇幅。"
            )
        else:
            target_chars = cls._length_target(request, evidence_sentence_count)
            length_clause = (
                f"篇幅等级：{length_labels[request.target_length]}；"
                f"证据充分时建议本节正文篇幅约{target_chars}字；"
                "证据不足时如实列出缺口，不凑字、不虚构。"
            )
        return (
            f"为《{request.document_title}》撰写“{request.section_title}”章节正文。\n"
            f"文档目标：{request.objective}\n"
            f"使用对象：{audience}\n"
            f"{length_clause}\n"
            f"本节要求：{request.section_guidance}\n"
            "只输出可直接进入正式文档的本节正文；不要解释编制方法，不要复述用户指令。"
        )

    @classmethod
    def _quality_gate(
        cls,
        request: DocumentSectionGenerateRequest,
        content: str,
        *,
        evidence_sentence_count: int = 0,
    ) -> tuple[bool, int, int, list[str], list[str]]:
        """分两级检查章节质量。

        返回 ``(hard_passed, char_count, target_chars, hard_warnings,
        advisory_warnings)``：

        - ``hard_warnings``：内容实质问题（必需要素缺失、范围不当收缩），
          通过 ``hard_passed=False`` 阻断正文写入草稿；
        - ``advisory_warnings``：仅作建议（如篇幅偏短），不阻断——证据不足时
          如实写短并列出缺口是合法输出，不应被字数硬门槛惩罚。
        """
        char_count = len(re.sub(r"\s+", "", content))
        target_chars = cls._length_target(request, evidence_sentence_count)
        advisory: list[str] = []
        hard_warnings: list[str] = []
        if (
            request.section_title.strip() not in cls.TABLE_LIKE_SECTIONS
            and char_count < target_chars
        ):
            advisory.append(
                f"正文仅{char_count}字，低于{request.target_length}篇幅等级的"
                f"建议篇幅约{target_chars}字；证据不足时可如实列出缺口，不凑字。"
            )

        required_aspects = {
            "编制说明": {
                "编制目的": ("目的", "用于"),
                "适用对象": ("适用对象", "面向"),
                "文档边界": ("文档边界", "范围"),
                "术语": ("术语", "定义"),
                "证据原则": ("证据", "引用"),
            },
            "适用范围与无人机系统概述": {
                "构型": ("构型", "系统边界", "总体布置"),
                "任务场景": ("任务", "场景"),
                "运行环境": ("环境", "温度", "气象"),
                "重量边界": ("重量", "起飞重量"),
                "性能边界": ("性能", "航程", "续航"),
            },
            "适航审定依据与审定基础": {
                "法规或规章": ("法规", "规章", "CCAR"),
                "适航标准": ("适航标准", "标准"),
                "审定基础": ("审定基础", "型号合格"),
                "适用关系": ("适用", "直接依据", "参考"),
            },
            "系统级安全性要求": {
                "危险分析": ("危险分析", "功能危险", "FHA"),
                "失效状态": ("失效状态", "失效条件"),
                "风险等级": ("风险等级", "灾难性", "危险", "严重"),
                "安全目标": ("安全目标", "概率目标", "安全性目标"),
            },
            "结构、材料与载荷": {
                "结构强度": ("结构强度", "强度"),
                "载荷": ("限制载荷", "极限载荷", "载荷"),
                "疲劳": ("疲劳", "损伤容限", "寿命"),
                "材料与连接": ("材料", "连接", "紧固"),
                "振动": ("振动", "共振"),
            },
            "动力、能源与推进系统": {
                "动力装置": ("动力装置", "发动机", "推进"),
                "能源储备": ("能源", "燃油", "电池", "储备"),
                "热安全": ("热安全", "温度", "热失控", "冷却"),
                "异常处置": ("异常", "失效", "处置"),
            },
            "飞行控制、导航与自主功能": {
                "飞行控制": ("飞行控制", "控制律"),
                "导航": ("导航", "定位", "精度"),
                "模式转换": ("模式", "转换"),
                "边界保护": ("边界保护", "包线保护", "限制"),
                "自主功能": ("自主", "自动"),
            },
            "指挥控制链路与地面站": {
                "链路性能": ("链路", "通信"),
                "丢链处置": ("丢链", "链路中断", "失联"),
                "频谱或电磁兼容": ("频谱", "频率", "电磁兼容"),
                "地面控制站": ("地面站", "地面控制站"),
            },
            "软件与复杂电子硬件": {
                "软件": ("软件",),
                "复杂电子硬件": ("复杂电子硬件", "硬件"),
                "验证确认": ("验证", "确认"),
                "配置管理": ("配置管理", "构型管理"),
                "变更控制": ("变更", "更改"),
            },
            "环境适应性与电磁兼容": {
                "温度": ("温度", "高温", "低温"),
                "湿度或高度": ("湿度", "高度", "气压"),
                "振动冲击": ("振动", "冲击"),
                "降水": ("降水", "降雨", "降雪"),
                "电磁兼容": ("电磁兼容", "电磁环境", "EMC"),
            },
            "运行限制与应急处置": {
                "运行包线": ("运行包线", "使用包线", "飞行包线"),
                "禁限条件": ("限制", "禁止"),
                "故障告警": ("故障", "告警"),
                "返航迫降": ("返航", "迫降"),
                "终止飞行": ("终止飞行", "飞行终止"),
            },
            "持续适航与维修保障": {
                "检查周期": ("检查周期", "间隔", "周期"),
                "维修说明": ("维修", "维护"),
                "寿命限制": ("寿命限制", "时限", "寿命"),
                "故障报告": ("故障报告", "报告"),
                "持续适航文件": ("持续适航文件", "持续适航资料"),
            },
            "符合性验证方法": {
                "检查": ("检查",),
                "分析": ("分析",),
                "试验": ("试验",),
                "演示或相似性": ("演示", "相似性"),
                "判定准则": ("判定准则", "合格判据", "验收判据"),
            },
            "符合性验证矩阵": {
                "要求": ("要求",),
                "来源": ("来源", "条款"),
                "符合性方法": ("符合性方法", "验证方法"),
                "条件输入": ("条件", "输入"),
                "判据": ("判据", "准则"),
                "记录状态": ("记录", "状态"),
            },
            "结论与待补信息": {
                "覆盖情况": ("覆盖", "已识别"),
                "证据缺口": ("缺口", "待补"),
                "待试验项目": ("待试验", "试验项目"),
                "关闭计划": ("关闭计划", "后续计划", "行动计划"),
            },
        }
        for aspect, keywords in required_aspects.get(request.section_title.strip(), {}).items():
            if not any(keyword in content for keyword in keywords):
                hard_warnings.append(f"本节未覆盖必需要素：{aspect}。")

        narrowing = re.compile(
            r"(?:本文档|本文件|本章节).{0,100}"
            r"(?:特指|仅适用于|范围.{0,12}(?:限定|限于)|适用于《)",
            re.DOTALL,
        )
        if narrowing.search(content):
            hard_warnings.append(
                "生成内容疑似将整份文档范围收缩为某一证据文件的特定构型或场景。"
            )
        return not hard_warnings, char_count, target_chars, hard_warnings, advisory

    @classmethod
    def _length_target(
        cls,
        request: DocumentSectionGenerateRequest,
        evidence_sentence_count: int,
    ) -> int:
        """按篇幅等级计算建议正文篇幅（软性目标）。

        证据越少目标越低：证据不足时如实写短并列出缺口是正确行为，不应被
        固定字数线逼着凑字。表格/矩阵类章节返回基准值仅作展示，不参与检查。
        """
        base = cls.LENGTH_BASE_CHARS[request.target_length]
        if request.section_title.strip() in cls.TABLE_LIKE_SECTIONS:
            return base
        evidence_factor = min(1.0, evidence_sentence_count / 12.0)
        return max(200, round(base * evidence_factor))

    @staticmethod
    def _attach_supplemental_document(
        execution: RetrievalExecution,
        document: ReferenceDocument,
    ) -> None:
        citation_numbers = [
            int(item.citation_id[1:])
            for item in execution.selected_chunks
            if item.citation_id.startswith("C") and item.citation_id[1:].isdigit()
        ]
        chunks, citations = build_auxiliary_evidence(
            document,
            execution.query,
            citation_start=max(citation_numbers, default=0) + 1,
        )
        execution.selected_chunks.extend(chunks)
        execution.citations.extend(citations)

    @staticmethod
    def _document_count(chunks: list[SelectedChunk]) -> int:
        return len({item.document_id for item in chunks})

    @staticmethod
    def _unique(values: list[str]) -> list[str]:
        return list(dict.fromkeys(item for item in values if item))
