from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable
from typing import cast
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import AppError
from app.schemas.chat import (
    CandidateClaim,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatDiagnostics,
    DiagnosticCell,
    EvidenceAssessment,
    EvidenceSentence,
    ReferenceDocument,
    StructuredAnswer,
)
from app.schemas.retrieval import (
    Citation,
    CoverageCell,
    NormalizedQuery,
    QueryPlan,
    RetrievalSearchRequest,
    SelectedChunk,
)
from app.services.answer_generator import (
    AnswerGenerator,
    AnswerModel,
    GenerationResult,
    SanitizedAnswer,
)
from app.services.answer_model_manager import AnswerModelManager
from app.services.citation_validator import CITATION_MARKER, CitationValidator
from app.services.claim_topic_validator import ClaimTopicValidator
from app.services.comparison_completeness_validator import (
    ComparisonCompletenessValidator,
    CompletenessValidationError,
)
from app.services.comparison_matrix_builder import ComparisonMatrixBuilder
from app.services.comparison_orchestrator import ComparisonOrchestrator
from app.services.evidence_extractor import EvidenceExtractor
from app.services.evidence_judge import EvidenceJudge
from app.services.evidence_record_extractor import EvidenceRecordExtractor
from app.services.evidence_scope_selector import EvidenceScopeSelector
from app.services.evidence_text import has_parameter_value, is_parametric_text, overlap_score
from app.services.grounding_validator import ClaimGroundingValidator
from app.services.model_client import OpenAICompatibleModelClient
from app.services.multi_document import multi_document_groups
from app.services.reference_document import build_auxiliary_evidence
from app.services.regulation_answer_builder import RegulationAnswerBuilder
from app.services.regulation_context import amendment_scope_note
from app.services.retrieval_service import RetrievalExecution, RetrievalService
from app.services.scope_validator import ScopeConsistencyValidator
from app.services.subject_grounding_validator import SubjectGroundingValidator

COMPARISON_MODEL_FALLBACK_ERRORS = {
    "ANSWER_JSON_PARSE_FAILED",
    "ANSWER_SCHEMA_VALIDATION_FAILED",
    "ANSWER_SEMANTIC_VALIDATION_FAILED",
}


class RagPipeline:
    """Run retrieval and the stage-four constrained generation gates."""

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
        self._answer_models = answer_models
        self._judge = EvidenceJudge(settings.generation)
        self._extractor = EvidenceExtractor(settings.generation)
        self._scope_selector = EvidenceScopeSelector()
        self._regulation_answer_builder = RegulationAnswerBuilder()
        self._citation_validator = CitationValidator(
            require_citations=settings.generation.require_citations
        )
        self._scope_validator = ScopeConsistencyValidator()
        self._grounding_validator = ClaimGroundingValidator()
        self._subject_grounding_validator = SubjectGroundingValidator()
        self._claim_topic_validator = ClaimTopicValidator()
        self._comparison_completeness_validator = ComparisonCompletenessValidator()
        self._comparison_matrix_builder = ComparisonMatrixBuilder()
        self._comparison_orchestrator = ComparisonOrchestrator(settings.generation)
        self._evidence_record_extractor = EvidenceRecordExtractor()
        self._enforce_scope = settings.generation.enforce_scope_consistency
        self._enforce_grounding = settings.generation.enforce_claim_grounding
        self._enforce_subject_grounding = settings.generation.enforce_subject_grounding
        self._enforce_claim_topic = settings.generation.enforce_claim_topic
        self._enforce_comparison_completeness = (
            settings.generation.enforce_comparison_completeness
        )
        self._generator = (
            AnswerGenerator(settings=settings.generation, model=answer_model)
            if answer_model is not None
            else None
        )
        self._answer_model = answer_model

    @classmethod
    def from_services(
        cls,
        *,
        settings: Settings,
        retrieval: RetrievalService,
        answer_model: OpenAICompatibleModelClient | None,
        answer_models: AnswerModelManager | None = None,
    ) -> RagPipeline:
        return cls(
            settings=settings,
            retrieval=retrieval,
            answer_model=cast(AnswerModel | None, answer_model),
            answer_models=answer_models,
        )

    async def run(
        self,
        request: ChatCompletionRequest,
        *,
        user_id: str,
    ) -> ChatCompletionResponse:
        pipeline_started = time.perf_counter()
        execution = await self._retrieval.execute(
            RetrievalSearchRequest(
                query=request.query,
                knowledge_base_ids=request.knowledge_base_ids,
                document_ids=request.document_ids,
                filters=request.filters,
                candidate_top_k=request.candidate_top_k,
            ),
            user_id=user_id,
        )
        if request.reference_document is not None:
            self._attach_auxiliary_document(execution, request.reference_document)
        execution.selected_chunks = self._scope_selector.select(
            execution.query, execution.selected_chunks,
        )
        assessment = self._judge.assess(
            execution.query,
            execution.selected_chunks,
            execution.coverage_matrix,
        )
        if execution.query_plan_v2 is not None:
            execution.evidence_records = self._evidence_record_extractor.extract(
                plan=execution.query_plan_v2,
                chunks=execution.selected_chunks,
                coverage_matrix=execution.coverage_matrix,
            )
            execution.stage_counts["evidence_records"] = len(execution.evidence_records)
        conversation_id = request.conversation_id or str(uuid4())
        if assessment.status in {"UNANSWERABLE", "CONFLICTED"}:
            execution.stage_latencies_ms["total"] = round(
                (time.perf_counter() - pipeline_started) * 1000,
                2,
            )
            return self._safe_response(
                execution=execution,
                assessment=assessment,
                conversation_id=conversation_id,
            )
        generator = self._generator
        selected_answer_model = self._answer_model
        model_source_id: str | None = None
        model_name: str | None = None
        if request.model is not None:
            if self._answer_models is None:
                raise AppError(
                    code="MODEL_SWITCHING_UNAVAILABLE",
                    message="当前服务未启用运行时模型切换。",
                    status_code=503,
                )
            selected_model = await self._answer_models.resolve(
                request.model,
                user_id=user_id,
            )
            generator = AnswerGenerator(
                settings=self._settings.generation,
                model=selected_model,
            )
            selected_answer_model = selected_model
            model_source_id = request.model.source_id
            model_name = request.model.model_name
        elif generator is not None:
            model_source_id = "configured-answer"
            model_name = self._settings.models.answer.model_name
        is_matrix_first = (
            self._settings.generation.comparison_matrix_first
            and execution.query_plan is not None
            and execution.query_plan.query_type in {"comparison", "multi_hop"}
            # 矩阵构建要求 plan 至少两个主体且维度非空；query_type 判定与
            # subjects/aspects 不一致（如仅识别出 1 个主体）时不要进入矩阵
            # 路径，否则 build 返回 None 会使整次问答 500。
            and len(execution.query_plan.subjects) >= 2
            and bool(execution.query_plan.aspects)
        )
        if generator is None and not is_matrix_first:
            raise AppError(
                code="ANSWER_MODEL_DISABLED",
                message="An answer model is required for constrained generation.",
                status_code=503,
            )

        generation_chunks = self._scope_selector.select(
            execution.query,
            execution.selected_chunks,
        )
        execution.stage_counts["generation_evidence_chunks"] = len(generation_chunks)
        evidence = self._extractor.extract(execution.query, generation_chunks)
        validate, validate_claim, sanitize = self._build_validators(
            assessment=assessment,
            evidence=evidence,
            execution=execution,
            include_historical=request.filters.include_historical,
        )

        extractive_answer = self._regulation_answer_builder.build(
            execution.query,
            generation_chunks,
        )
        mode = self._settings.generation.answer_mode
        # Source-faithful Chinese rendering for foreign-language evidence. The
        # existing generation validators still run; deterministic extraction
        # remains available if generation cannot be validated.
        if (
            mode == "deterministic" and not is_matrix_first and generator is not None
            and self._settings.generation.cross_language_generation
            and re.search(r"[\u4e00-\u9fff]", request.query)
            and generation_chunks
        ):
            body = "\n".join(c.text for c in generation_chunks)
            if len(re.findall(r"[A-Za-z]", body)) > 4 * len(re.findall(
                r"[\u4e00-\u9fff]", body,
            )):
                mode = "llm"
                execution.stage_counts["cross_language_generation"] = 1
        matrix_fallback_applied = False
        matrix_first_applied = False
        generation: GenerationResult | None = None
        deterministic_answer: StructuredAnswer | None = None
        if mode != "llm" and is_matrix_first and execution.query_plan is not None:
            matrix_generation = await self._comparison_orchestrator.run(
                plan=execution.query_plan,
                chunks=execution.selected_chunks,
                records=execution.evidence_records,
                assessment=assessment,
                coverage_matrix=execution.coverage_matrix,
                model=selected_answer_model,
            )
            if matrix_generation is not None:
                self._validate_answerability(assessment, matrix_generation.answer)
                self._validate_evidence_consistency(
                    matrix_generation.answer,
                    # Deterministic matrix cells are built from EvidenceRecord
                    # objects across all selected chunks.  The generative
                    # EvidenceExtractor has a global sentence cap and may omit
                    # a later cell's citation even though that record is valid.
                    # Validate citation membership against the complete
                    # backend-selected set on this path.
                    evidence=self._citation_evidence_for_chunks(
                        execution.selected_chunks
                    ),
                    execution=execution,
                    include_historical=request.filters.include_historical,
                    skip_scope=True,
                )
                self._comparison_completeness_validator.validate(
                    matrix_generation.answer,
                    plan=execution.query_plan,
                    coverage_matrix=execution.coverage_matrix,
                    assessment=assessment,
                )
                if mode == "deterministic":
                    generation = matrix_generation
                    matrix_first_applied = True
                else:
                    deterministic_answer = matrix_generation.answer
            else:
                # 矩阵构建失败（plan 条件不满足等）：降级到 extractive /
                # generator 兜底，**不抛 500**，保证问答始终可用。
                execution.stage_counts["matrix_build_failed"] = 1
                matrix_fallback_applied = True
        if mode != "llm" and generation is None and extractive_answer is not None:
            # Deterministic regulation answers may intentionally cover every
            # selected process chapter.  The model-oriented EvidenceExtractor
            # applies a global sentence budget, so a valid later chapter (for
            # example C17/档案管理) may not be present in ``evidence``.  Validate
            # this path against all chunks that the builder actually received,
            # just as the deterministic comparison path does.
            self._validate_answerability(assessment, extractive_answer)
            self._validate_evidence_consistency(
                extractive_answer,
                evidence=self._citation_evidence_for_chunks(generation_chunks),
                execution=execution,
                include_historical=request.filters.include_historical,
            )
            if mode == "deterministic":
                generation = GenerationResult(
                    answer=extractive_answer,
                    repaired=False,
                )
            else:
                deterministic_answer = deterministic_answer or extractive_answer
        if generation is None:
            assert generator is not None
            try:
                groups = multi_document_groups(evidence)
                if groups:
                    # 分列模式：按文档对 generation_chunks 重新分组，每组**独立**
                    # 提取证据（每组享有完整证据句预算），避免全局提取后分组导致
                    # 每份规程证据不足、流程后半段（数据处理/验证/归档）缺失。
                    group_evidence = self._evidence_by_document(
                        execution.query,
                        generation_chunks,
                    )
                    generation = await self._generate_by_document(
                        generator=generator,
                        groups=group_evidence,
                        question=request.query,
                        assessment=assessment,
                        execution=execution,
                        include_historical=request.filters.include_historical,
                        reference_document=request.reference_document,
                    )
                else:
                    generation = await generator.generate(
                        question=request.query,
                        assessment=assessment,
                        evidence=evidence,
                        reference_document=request.reference_document,
                        semantic_validator=validate,
                        semantic_sanitizer=sanitize,
                    )
            except AppError as exc:
                if exc.code not in COMPARISON_MODEL_FALLBACK_ERRORS:
                    raise
                fallback = self._comparison_matrix_fallback(
                    execution=execution,
                    assessment=assessment,
                    evidence=evidence,
                    include_historical=request.filters.include_historical,
                    original=None,
                    warning="模型输出无法解析，已改用确定性对比矩阵输出。",
                )
                if fallback is None:
                    # 矩阵兜底也不可用：以提取式答案兜底（若存在），否则保留异常
                    # 由上层统一转 UNANSWERABLE，不再直接 500。
                    if extractive_answer is not None:
                        generation = GenerationResult(
                            answer=extractive_answer,
                            repaired=False,
                            validation_degraded=True,
                            validation_warnings=["模型输出无法解析，且确定性矩阵不可用，已退回提取式答案。"],
                        )
                    else:
                        raise
                else:
                    generation = fallback
                matrix_fallback_applied = True
        if mode == "hybrid" and deterministic_answer is not None and not matrix_fallback_applied:
            generation = self._hybrid_merge(generation, deterministic_answer)
        # 对比矩阵兜底：生成模型答不出完整矩阵（缺已覆盖格子或整体降级）时，
        # 用确定性逐格填充的矩阵替换——按证据逐格给内容（无内容写"未明确
        # 说明"），稳定输出完整矩阵，不依赖 9B 模型能力。
        if (
            self._enforce_comparison_completeness
            and mode != "hybrid"
            and execution.query_plan is not None
            and execution.query_plan.query_type in {"comparison", "multi_hop"}
            and not matrix_fallback_applied
            and not matrix_first_applied
            and (
                generation.validation_degraded
                or not self._comparison_answer_complete(
                    generation.answer,
                    plan=execution.query_plan,
                    coverage_matrix=execution.coverage_matrix,
                    assessment=assessment,
                )
            )
        ):
            fallback = self._comparison_matrix_fallback(
                execution=execution,
                assessment=assessment,
                evidence=evidence,
                include_historical=request.filters.include_historical,
                original=generation,
                warning="生成结果不完整，已改用确定性对比矩阵输出。",
            )
            if fallback is not None:
                generation = fallback
        if (
            mode == "deterministic"
            and matrix_first_applied
            and self._settings.generation.explain_enabled
            and generator is not None
            and generation is not None
        ):
            explanation = await generator.explain(generation.answer.answer)
            if explanation:
                merged = generation.answer.model_copy(
                    update={
                        "answer": (
                            generation.answer.answer
                            + "\n\n---\n通俗讲解（模型转述，依据见上文原文）：\n"
                            + explanation
                        )
                    }
                )
                generation = GenerationResult(
                    answer=merged,
                    repaired=generation.repaired,
                    validation_degraded=generation.validation_degraded,
                    validation_warnings=generation.validation_warnings,
                    model_calls=generation.model_calls + 1,
                )
        execution.model_call_counts[
            "summary" if matrix_first_applied else "answer"
        ] = generation.model_calls
        answer = generation.answer
        scope_note = amendment_scope_note(generation_chunks)
        if scope_note:
            answer = answer.model_copy(update={"answer": answer.answer + "\n\n" + scope_note})
        used_citations = set(CITATION_MARKER.findall(answer.answer))
        used_citations.update(
            citation_id for claim in answer.claims for citation_id in claim.citation_ids
        )
        execution.stage_latencies_ms["total"] = round(
            (time.perf_counter() - pipeline_started) * 1000,
            2,
        )
        return ChatCompletionResponse(
            request_id=execution.request_id,
            conversation_id=conversation_id,
            status=answer.answerability,
            answer=answer.answer,
            claims=answer.claims,
            citations=[
                citation
                for citation in execution.citations
                if citation.citation_id in used_citations
            ],
            missing_information=self._unique(
                [*assessment.missing_requirements, *answer.missing_information]
            ),
            conflicts=self._unique(
                [
                    *(item.details for item in assessment.conflicting_citations),
                    *answer.conflicts,
                ]
            ),
            evidence_assessment=assessment,
            prompt_version=self._settings.generation.prompt_version,
            json_repaired=generation.repaired,
            validation_degraded=generation.validation_degraded,
            validation_warnings=generation.validation_warnings or [],
            model_source_id=model_source_id,
            model_name=model_name,
            diagnostics=self._diagnostics(execution, assessment),
        )

    def _attach_auxiliary_document(
        self,
        execution: RetrievalExecution,
        document: ReferenceDocument,
    ) -> None:
        """Merge a temporary upload into the same evidence and citation path as RAGFlow."""
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
        execution.candidate_count += len(chunks)
        execution.stage_counts["auxiliary_document_chunks"] = len(chunks)
        self._augment_auxiliary_coverage(execution, chunks)

    def _augment_auxiliary_coverage(
        self,
        execution: RetrievalExecution,
        chunks: list[SelectedChunk],
    ) -> None:
        """Allow temporary evidence to fill planned cells when it directly matches them."""
        if not execution.coverage_matrix:
            return
        augmented: list[CoverageCell] = []
        for cell in execution.coverage_matrix:
            requirement = "：".join(part for part in (cell.subject, cell.aspect) if part)
            ranked = sorted(
                chunks,
                key=lambda chunk: overlap_score(requirement, chunk.text),
                reverse=True,
            )
            matching = [
                chunk
                for chunk in ranked
                if overlap_score(requirement, chunk.text)
                >= self._settings.generation.min_evidence_overlap
                and (
                    not is_parametric_text(cell.aspect)
                    or has_parameter_value(chunk.text)
                )
            ][:2]
            if not matching:
                augmented.append(cell)
                continue
            augmented.append(
                cell.model_copy(
                    update={
                        "status": "covered",
                        "chunk_ids": list(
                            dict.fromkeys([*cell.chunk_ids, *(item.chunk_id for item in matching)])
                        ),
                        "citation_ids": list(
                            dict.fromkeys(
                                [*cell.citation_ids, *(item.citation_id for item in matching)]
                            )
                        ),
                        "reason": "临时辅助资料提供了直接证据",
                    }
                )
            )
        execution.coverage_matrix = augmented

    def _build_validators(
        self,
        *,
        assessment: EvidenceAssessment,
        evidence: list[EvidenceSentence],
        execution: RetrievalExecution,
        include_historical: bool,
    ) -> tuple[
        Callable[[StructuredAnswer], None],
        Callable[[StructuredAnswer], None],
        Callable[[StructuredAnswer, Exception], SanitizedAnswer],
    ]:
        def validate(answer: StructuredAnswer) -> None:
            self._validate_answerability(assessment, answer)
            self._validate_evidence_consistency(
                answer,
                evidence=evidence,
                execution=execution,
                include_historical=include_historical,
            )
            if self._enforce_comparison_completeness:
                # 答案级校验：对比矩阵的已覆盖格子必须全部出现在答案中
                # （不进 per-claim 校验，避免 sanitize 逐条裁剪时误伤）。
                self._comparison_completeness_validator.validate(
                    answer,
                    plan=execution.query_plan,
                    coverage_matrix=execution.coverage_matrix,
                    assessment=assessment,
                )

        def validate_claim(answer: StructuredAnswer) -> None:
            self._validate_evidence_consistency(
                answer,
                evidence=evidence,
                execution=execution,
                include_historical=include_historical,
            )

        def sanitize(
            answer: StructuredAnswer,
            validation_error: Exception,
        ) -> SanitizedAnswer:
            return self._sanitize_answer(
                answer,
                assessment=assessment,
                claim_validator=validate_claim,
                validation_error=validation_error,
            )

        return validate, validate_claim, sanitize

    def _evidence_by_document(
        self,
        query: NormalizedQuery,
        chunks: list[SelectedChunk],
    ) -> list[tuple[str, list[EvidenceSentence]]]:
        """分列模式下按文档分组提取证据：每组独立享有证据句预算。"""
        by_document: dict[str, list[SelectedChunk]] = {}
        for chunk in chunks:
            name = chunk.metadata.document_name or ""
            if not name:
                continue
            by_document.setdefault(name, []).append(chunk)
        groups: list[tuple[str, list[EvidenceSentence]]] = []
        for name, doc_chunks in by_document.items():
            if len(doc_chunks) < 2:
                continue
            groups.append((name, self._extractor.extract(query, doc_chunks)))
        return groups

    async def _generate_by_document(
        self,
        *,
        generator: AnswerGenerator,
        groups: list[tuple[str, list[EvidenceSentence]]],
        question: str,
        assessment: EvidenceAssessment,
        execution: RetrievalExecution,
        include_historical: bool,
        reference_document: ReferenceDocument | None,
    ) -> GenerationResult:
        """同名/同主题多份规程时，逐份规程单独生成回答，后端按文档拼接分列。

        每份规程只看到自己的证据（引用编号全局唯一，无需重映射），模型不必
        同时区分多份文档；回答按"【文档名】"分节输出，来源天然清晰，且不
        涉及新旧/地域取舍判断。
        """
        answers: list[StructuredAnswer] = []
        claims: list[CandidateClaim] = []
        missing: list[str] = []
        conflicts: list[str] = []
        degraded = False
        repaired = False
        warnings: list[str] = []
        model_calls = 0
        for group_index, (document_name, doc_evidence) in enumerate(groups, start=1):
            validate, _, sanitize = self._build_validators(
                assessment=assessment,
                evidence=doc_evidence,
                execution=execution,
                include_historical=include_historical,
            )
            generation = await generator.generate(
                question=question,
                assessment=assessment,
                evidence=doc_evidence,
                reference_document=reference_document,
                semantic_validator=validate,
                semantic_sanitizer=sanitize,
            )
            model_calls += generation.model_calls
            body = generation.answer.answer
            answers.append(
                generation.answer.model_copy(update={"answer": f"【{document_name}】\n{body}"})
            )
            # 每组 claim_id 都从 c1/c2… 开始，合并前加组后缀保证全局唯一
            # （StructuredAnswer 要求 claim_id 唯一）。
            if group_index > 1:
                claims.extend(
                    claim.model_copy(update={"claim_id": f"{claim.claim_id}_g{group_index}"})
                    for claim in generation.answer.claims
                )
            else:
                claims.extend(generation.answer.claims)
            missing.extend(generation.answer.missing_information)
            conflicts.extend(generation.answer.conflicts)
            degraded = degraded or generation.validation_degraded
            repaired = repaired or generation.repaired
            if generation.validation_warnings:
                warnings.extend(generation.validation_warnings)
        merged = StructuredAnswer(
            answerability=(
                "PARTIALLY_ANSWERABLE"
                if any(
                    answer.answerability == "PARTIALLY_ANSWERABLE" for answer in answers
                )
                else "ANSWERABLE"
            ),
            answer="\n\n".join(answer.answer for answer in answers),
            claims=claims,
            missing_information=self._unique(missing),
            conflicts=self._unique(conflicts),
        )
        return GenerationResult(
            answer=merged,
            repaired=repaired,
            validation_degraded=degraded,
            validation_warnings=warnings or None,
            model_calls=model_calls,
        )

    def _validate_evidence_consistency(
        self,
        answer: StructuredAnswer,
        *,
        evidence: list[EvidenceSentence],
        execution: RetrievalExecution,
        include_historical: bool,
        skip_scope: bool = False,
    ) -> None:
        self._citation_validator.validate(
            answer,
            evidence=evidence,
            chunks=execution.selected_chunks,
            include_historical=include_historical,
        )
        if self._enforce_scope and not skip_scope:
            self._scope_validator.validate(
                answer,
                query=execution.query,
                chunks=execution.selected_chunks,
            )
        if self._enforce_grounding:
            self._grounding_validator.validate(
                answer,
                chunks=execution.selected_chunks,
            )
        if self._enforce_subject_grounding:
            self._subject_grounding_validator.validate(
                answer,
                plan=execution.query_plan,
                chunks=execution.selected_chunks,
            )
        if self._enforce_claim_topic:
            self._claim_topic_validator.validate(
                answer,
                chunks=execution.selected_chunks,
            )

    @staticmethod
    def _citation_evidence_for_chunks(
        chunks: list[SelectedChunk],
    ) -> list[EvidenceSentence]:
        """Expose every selected citation to deterministic answer validation.

        This is intentionally not used as model prompt evidence.  It only
        prevents the model-oriented sentence budget from invalidating a
        deterministic matrix whose EvidenceRecords came from later cells.
        """
        return [
            EvidenceSentence(
                citation_id=chunk.citation_id,
                chunk_id=chunk.chunk_id,
                text=chunk.text or chunk.chunk_id,
                document_name=chunk.metadata.document_name,
                version=chunk.metadata.version,
            )
            for chunk in chunks
        ]

    def _comparison_answer_complete(
        self,
        answer: StructuredAnswer,
        *,
        plan: QueryPlan,
        coverage_matrix: list[CoverageCell],
        assessment: EvidenceAssessment,
    ) -> bool:
        """对比矩阵答案是否覆盖全部已覆盖格子（供兜底触发判断）。"""
        try:
            self._comparison_completeness_validator.validate(
                answer,
                plan=plan,
                coverage_matrix=coverage_matrix,
                assessment=assessment,
            )
            return True
        except CompletenessValidationError:
            return False

    def _comparison_matrix_fallback(
        self,
        *,
        execution: RetrievalExecution,
        assessment: EvidenceAssessment,
        evidence: list[EvidenceSentence],
        include_historical: bool,
        original: GenerationResult | None,
        warning: str,
    ) -> GenerationResult | None:
        """Build and validate the deterministic matrix for comparison failures."""
        plan = execution.query_plan
        if (
            not self._enforce_comparison_completeness
            or plan is None
            or plan.query_type not in {"comparison", "multi_hop"}
        ):
            return None
        matrix_answer = self._comparison_matrix_builder.build(
            plan=plan,
            chunks=execution.selected_chunks,
            assessment=assessment,
            coverage_matrix=execution.coverage_matrix,
        )
        if matrix_answer is None:
            return None
        try:
            self._validate_answerability(assessment, matrix_answer)
            # 矩阵内容逐字取自所属主体 chunk；scope 的条号抽取会把内容里的
            # 4.1.1/A.3/2.5 等误判成声明条款，因此仅在该确定性路径跳过 scope。
            self._validate_evidence_consistency(
                matrix_answer,
                evidence=self._citation_evidence_for_chunks(
                    execution.selected_chunks
                ),
                execution=execution,
                include_historical=include_historical,
                skip_scope=True,
            )
            self._comparison_completeness_validator.validate(
                matrix_answer,
                plan=plan,
                coverage_matrix=execution.coverage_matrix,
                assessment=assessment,
            )
        except ValueError:
            return None
        return GenerationResult(
            answer=matrix_answer,
            repaired=original.repaired if original is not None else False,
            validation_degraded=True,
            validation_warnings=[
                *((original.validation_warnings or []) if original is not None else []),
                warning,
            ],
        )

    @classmethod
    def _sanitize_answer(
        cls,
        answer: StructuredAnswer,
        *,
        assessment: EvidenceAssessment,
        claim_validator: Callable[[StructuredAnswer], None],
        validation_error: Exception,
    ) -> SanitizedAnswer:
        valid_claims: list[CandidateClaim] = []
        rejected_claim_ids: list[str] = []
        # Scope declarations are often stated once in the answer preamble
        # (for example "第 25.981 条") instead of repeated in every claim.
        # Preserve that declaration context while removing all citation
        # markers, so per-claim validation cannot silently lose the scope that
        # caused the complete answer to fail.
        scope_context = CITATION_MARKER.sub("", answer.answer).strip()
        for claim in answer.claims:
            try:
                marker_text = "".join(
                    f"[{citation_id}]"
                    for citation_id in dict.fromkeys(claim.citation_ids)
                )
                claim_text = CITATION_MARKER.sub("", claim.claim).strip()
                candidate = StructuredAnswer(
                    answerability="ANSWERABLE",
                    answer=f"{scope_context}\n{claim_text}{marker_text}",
                    claims=[claim.model_copy(update={"claim": claim_text})],
                    missing_information=[],
                    conflicts=[],
                )
                claim_validator(candidate)
            except ValueError:
                rejected_claim_ids.append(claim.claim_id)
                continue
            valid_claims.append(candidate.claims[0])

        warnings = [
            f"生成结果未通过整体证据校验：{type(validation_error).__name__}",
        ]
        if rejected_claim_ids:
            warnings.append(
                "已省略未通过证据校验的结论：" + ", ".join(rejected_claim_ids)
            )
        if not valid_claims:
            safe_answer = StructuredAnswer(
                answerability="UNANSWERABLE",
                answer="检索到了相关资料，但生成内容未通过证据一致性校验，暂不能给出可靠结论。",
                claims=[],
                missing_information=cls._unique(
                    [
                        *assessment.missing_requirements,
                        "生成内容未通过证据一致性校验。",
                    ]
                ),
                conflicts=answer.conflicts,
            )
            return SanitizedAnswer(answer=safe_answer, warnings=warnings)

        answer_parts = []
        for claim in valid_claims:
            claim_text = claim.claim.rstrip("。；; ")
            markers = "".join(
                f"[{citation_id}]" for citation_id in dict.fromkeys(claim.citation_ids)
            )
            answer_parts.append(f"{claim_text}。{markers}")
        safe_answer = StructuredAnswer(
            answerability="PARTIALLY_ANSWERABLE",
            answer="".join(answer_parts),
            claims=valid_claims,
            missing_information=cls._unique(
                [
                    *assessment.missing_requirements,
                    *answer.missing_information,
                    "部分生成内容未通过证据一致性校验，已省略。",
                ]
            ),
            conflicts=answer.conflicts,
        )
        return SanitizedAnswer(answer=safe_answer, warnings=warnings)

    def _safe_response(
        self,
        *,
        execution: RetrievalExecution,
        assessment: EvidenceAssessment,
        conversation_id: str,
    ) -> ChatCompletionResponse:
        if assessment.status == "CONFLICTED":
            answer = "检索到的有效资料存在冲突，无法在不裁决冲突的情况下给出确定答案。"
            citation_ids = {
                citation_id
                for conflict in assessment.conflicting_citations
                for citation_id in conflict.citation_ids
            }
            citations = [
                item for item in execution.citations if item.citation_id in citation_ids
            ]
        else:
            answer = "现有资料中未找到足够证据，无法可靠回答该问题。"
            citations = []
        return ChatCompletionResponse(
            request_id=execution.request_id,
            conversation_id=conversation_id,
            status=assessment.status,
            answer=answer,
            claims=[],
            citations=citations,
            missing_information=assessment.missing_requirements,
            conflicts=[item.details for item in assessment.conflicting_citations],
            evidence_assessment=assessment,
            prompt_version=self._settings.generation.prompt_version,
            diagnostics=self._diagnostics(execution, assessment),
        )

    def _diagnostics(
        self,
        execution: RetrievalExecution,
        assessment: EvidenceAssessment,
    ) -> ChatDiagnostics | None:
        if not self._settings.planning.diagnostics_enabled:
            return None
        chunk_by_id = {chunk.chunk_id: chunk for chunk in execution.selected_chunks}
        record_counts: dict[str, int] = {}
        for record in execution.evidence_records:
            record_counts[record.cell_id] = record_counts.get(record.cell_id, 0) + 1
        cells = [
            DiagnosticCell(
                cell_id=cell.subquery_id,
                subject=cell.subject,
                aspect=cell.aspect,
                status=cell.status,
                document_ids=list(
                    dict.fromkeys(
                        chunk_by_id[chunk_id].document_id
                        for chunk_id in cell.chunk_ids
                        if chunk_id in chunk_by_id
                    )
                ),
                citation_ids=cell.citation_ids,
                evidence_record_count=record_counts.get(cell.subquery_id, 0),
            )
            for cell in execution.coverage_matrix
        ]
        plan = execution.query_plan_v2
        if not cells and plan is not None:
            synthesized_status = {
                "ANSWERABLE": "covered",
                "PARTIALLY_ANSWERABLE": "low_confidence",
                "UNANSWERABLE": "not_specified",
                "CONFLICTED": "low_confidence",
            }[assessment.status]
            document_ids = list(
                dict.fromkeys(chunk.document_id for chunk in execution.selected_chunks)
            )
            citation_ids = list(
                dict.fromkeys(chunk.citation_id for chunk in execution.selected_chunks)
            )
            cells = [
                DiagnosticCell(
                    cell_id=cell.id,
                    subject=cell.subject,
                    aspect=cell.aspect,
                    status=synthesized_status,
                    document_ids=document_ids,
                    citation_ids=citation_ids,
                    evidence_record_count=record_counts.get(cell.id, 0),
                )
                for cell in plan.cells
            ]
        model_breakdown = {
            key: value for key, value in execution.model_call_counts.items() if value
        }
        return ChatDiagnostics(
            plan_version=plan.version if plan is not None else "query-plan-v1",
            planner_source=plan.planner_source if plan is not None else "legacy",
            planned_cell_count=len(plan.cells) if plan is not None else len(cells),
            cells=cells,
            subject_document_ids={
                resolution.subject: resolution.document_ids
                for resolution in execution.subject_resolutions
            },
            retrieval_calls=execution.stage_counts.get("retrieval_calls", 1),
            translated_cell_ids=execution.translated_cell_ids,
            model_calls=sum(model_breakdown.values()),
            model_call_breakdown=model_breakdown,
            latencies_ms=execution.stage_latencies_ms,
            stage_counts=execution.stage_counts,
        )

    @staticmethod
    def _hybrid_merge(
        primary: GenerationResult, backup: StructuredAnswer | None,
    ) -> GenerationResult:
        """LLM 为主、确定性答案为补：把 LLM 未引用的证据条款补回答案。

        保证企业知识库"不漏条款"：确定性矩阵/提取式答案覆盖全部相关引用，
        LLM 生成时可能精简掉部分（如对比矩阵中的林木高度/郁闭度等上下文），
        这里按引用编号补回，避免证据丢失。
        """
        if backup is None or not backup.claims:
            return primary
        covered = {
            citation_id
            for claim in primary.answer.claims
            for citation_id in claim.citation_ids
        }
        missing = [
            claim
            for claim in backup.claims
            if not (set(claim.citation_ids) & covered)
        ]
        if not missing:
            return primary
        lines = [primary.answer.answer, "", "---", "补充（证据范围内、上文未列出的条款）："]
        appended: list[CandidateClaim] = []
        for index, claim in enumerate(missing, start=1):
            lines.append(f"{claim.claim} [{', '.join(claim.citation_ids)}]")
            appended.append(
                CandidateClaim(
                    claim_id=f"supplement-{index}",
                    claim=claim.claim,
                    citation_ids=claim.citation_ids,
                )
            )
        merged = primary.answer.model_copy(
            update={
                "answer": "\n\n".join(lines),
                "claims": [*primary.answer.claims, *appended],
            }
        )
        return GenerationResult(
            answer=merged,
            repaired=primary.repaired,
            validation_degraded=primary.validation_degraded,
            validation_warnings=primary.validation_warnings,
            model_calls=primary.model_calls,
        )

    @staticmethod
    def _validate_answerability(
        assessment: EvidenceAssessment,
        answer: StructuredAnswer,
    ) -> None:
        if (
            assessment.status == "PARTIALLY_ANSWERABLE"
            and answer.answerability == "ANSWERABLE"
        ):
            # Enterprise knowledge base: answers must be grounded in the retrieved
            # evidence and must never over-claim completeness on partial evidence.
            # Instead of failing the whole request (500), downgrade answerability
            # honestly. Claim-level grounding validators still strip any content not
            # backed by a citation.
            answer.answerability = "PARTIALLY_ANSWERABLE"
        # 不再要求模型把 judge 判定缺失的要求逐条写进 missing_information：
        # 响应层会把 assessment.missing_requirements 确定性合并进响应的
        # missing_information，模型漏填只会徒增修复轮次、引发答案震荡。

    @staticmethod
    def _unique(values: Iterable[str]) -> list[str]:
        return list(dict.fromkeys(value for value in values if value))


def citations_by_id(citations: list[Citation]) -> dict[str, Citation]:
    """Small typed helper retained for the stage-five verifier."""
    return {citation.citation_id: citation for citation in citations}
