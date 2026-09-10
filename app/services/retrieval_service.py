from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.middleware import current_request_id
from app.schemas.evidence import EvidenceRecord
from app.schemas.planning import PlannedCell, QueryPlanV2, RetrievalQuery, SubjectResolution
from app.schemas.retrieval import (
    Citation,
    CoverageCell,
    MetadataCondition,
    MetadataConditions,
    NormalizedQuery,
    QueryPlan,
    RagflowRetrievalRequest,
    RetrievalCandidateDebug,
    RetrievalDebugResponse,
    RetrievalFailure,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievedChunk,
    SelectedChunk,
)
from app.services.access_control import AccessControlService
from app.services.article_identity import owns_requested_article
from app.services.cell_evidence import cell_has_substantive_evidence
from app.services.citation_service import CitationService
from app.services.coverage_selector import CoverageSelector
from app.services.query_analyzer import QueryAnalyzer
from app.services.query_planning_service import QueryPlanningService
from app.services.query_scope import (
    declared_scope_cores,
    document_in_scope,
)
from app.services.query_translator import QueryTranslator, has_translatable_topic
from app.services.ragflow_client import RagflowClient
from app.services.registry import ServiceRegistry
from app.services.requirement_taxonomy import (
    ASPECT_SECONDARY_RETRIEVAL_TERMS,
    SUBJECT_RETRIEVAL_TERMS,
)
from app.services.reranker_client import RerankerClient
from app.services.retrieval_fusion import RetrievalFusion
from app.services.structured_planner_model import StructuredPlannerModel
from app.services.subject_document_resolver import SubjectDocumentResolver, title_similarity

# 非流程环节的顶层章节标题：流程题名额分配时不为它们保底名额，
# 避免"范围/术语/基本要求"等高分章节把数据处理/验证/归档等流程后半段挤出。
_NON_PROCESS_SECTION_TITLES = {
    "范围",
    "规范性引用文件",
    "术语和定义",
    "基本要求",
    "安全注意事项",
}


@dataclass(slots=True)
class RetrievalExecution:
    request_id: str
    query: NormalizedQuery
    selected_chunks: list[SelectedChunk]
    citations: list[Citation]
    candidate_count: int
    reranker_used: bool
    reranker_fallback: bool
    ragflow_request: RagflowRetrievalRequest
    stage_counts: dict[str, int]
    candidates: list[RetrievalCandidateDebug]
    query_plan: QueryPlan | None = None
    coverage_matrix: list[CoverageCell] = field(default_factory=list)
    query_plan_v2: QueryPlanV2 | None = None
    shadow_query_plan: QueryPlanV2 | None = None
    subject_resolutions: list[SubjectResolution] = field(default_factory=list)
    translated_cell_ids: list[str] = field(default_factory=list)
    model_call_counts: dict[str, int] = field(default_factory=dict)
    stage_latencies_ms: dict[str, float] = field(default_factory=dict)
    evidence_records: list[EvidenceRecord] = field(default_factory=list)
    #: 检索环节失败记录（结构化区分“零命中”与“请求失败”；失败不等于未收录）。
    failures: list[RetrievalFailure] = field(default_factory=list)
    #: 是否经过“精确锚点确定性兜底”（零命中后关键词模式再检索一轮）。
    deterministic_fallback: bool = False
    #: 枚举/计数型问题的文档清单边界（requires_inventory 时填充）。
    doc_inventory: dict[str, Any] | None = None

    def search_response(self) -> RetrievalSearchResponse:
        return RetrievalSearchResponse(
            request_id=self.request_id,
            query=self.query,
            chunks=self.selected_chunks,
            citations=self.citations,
            candidate_count=self.candidate_count,
            selected_count=len(self.selected_chunks),
            reranker_used=self.reranker_used,
            reranker_fallback=self.reranker_fallback,
            query_plan=self.query_plan,
            coverage_matrix=self.coverage_matrix,
            failures=self.failures,
            deterministic_fallback=self.deterministic_fallback,
            doc_inventory=self.doc_inventory,
        )

    def debug_response(self) -> RetrievalDebugResponse:
        return RetrievalDebugResponse(
            **self.search_response().model_dump(),
            ragflow_request=self.ragflow_request.to_payload(),
            stage_counts=self.stage_counts,
            candidates=self.candidates,
        )


class RetrievalService:
    def __init__(
        self,
        *,
        settings: Settings,
        ragflow: RagflowClient | None,
        reranker: RerankerClient | None,
        access_control: AccessControlService,
        planner_model: StructuredPlannerModel | None = None,
        translator: QueryTranslator | None = None,
    ) -> None:
        self._settings = settings
        self._ragflow = ragflow
        self._reranker = reranker
        self._access_control = access_control
        self._query_analyzer = QueryAnalyzer()
        self._query_planning = QueryPlanningService(
            settings.planning,
            structured=planner_model,
        )
        self._query_translator = translator or QueryTranslator(settings.translation, None)
        self._subject_resolver = SubjectDocumentResolver()
        self._fusion = RetrievalFusion(
            rrf_k=settings.retrieval.fusion_rrf_k,
            translated_weight=settings.retrieval.translated_query_weight,
        )
        self._coverage_selector = CoverageSelector()
        self._citation_service = CitationService()

    @classmethod
    def from_registry(
        cls,
        settings: Settings,
        registry: ServiceRegistry,
    ) -> RetrievalService:
        verifier = registry.verifier_model_client
        planner_model = (
            StructuredPlannerModel(settings.planning, verifier)
            if verifier is not None
            and settings.planning.enabled
            and settings.planning.model_enabled
            else None
        )
        return cls(
            settings=settings,
            ragflow=registry.ragflow_client,
            reranker=registry.reranker_client,
            access_control=AccessControlService(settings.access_control),
            planner_model=planner_model,
            translator=QueryTranslator(
                settings.translation,
                verifier if settings.translation.model_enabled else None,
            ),
        )

    async def execute(
        self,
        request: RetrievalSearchRequest,
        *,
        user_id: str,
    ) -> RetrievalExecution:
        started = time.perf_counter()
        query = self._query_analyzer.analyze(request.query)
        planning = await self._query_planning.plan(query)
        plan = planning.legacy_plan
        if not plan.requires_multi_query:
            execution = await self._execute_single(request, plan=plan, user_id=user_id)
            execution.query_plan = plan
        else:
            try:
                execution = await self._execute_plan(
                    request,
                    query=query,
                    plan=plan,
                    plan_v2=planning.plan,
                    user_id=user_id,
                )
            except AppError:
                execution = await self._execute_single(request, user_id=user_id)
                execution.query_plan = plan
                execution.stage_counts["multi_query_fallback"] = 1
        execution.query_plan_v2 = planning.plan
        execution.shadow_query_plan = planning.shadow_plan
        execution.model_call_counts["planner"] = planning.model_calls
        execution.stage_latencies_ms["planner"] = planning.latency_ms
        execution.stage_counts["planner_fallback"] = int(planning.fallback)
        if query.requires_inventory:
            # 枚举/计数型问题：列出授权数据集内的完整文档清单作为答案边界
            # （文档名/版本/轮次类答案不能只信检索 Top-N）。
            execution.doc_inventory = await self._build_doc_inventory(
                request,
                user_id=user_id,
                failures=execution.failures,
            )
        execution.stage_latencies_ms["retrieval_total"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )
        return execution

    async def _build_doc_inventory(
        self,
        request: RetrievalSearchRequest,
        *,
        user_id: str,
        failures: list[RetrievalFailure],
    ) -> dict[str, Any] | None:
        """聚合授权数据集内的文档清单（分页拉取，避免超过单页上限漏数）。"""
        if self._ragflow is None:
            return None
        try:
            allowed_datasets = await self._access_control.authorize_knowledge_bases(
                user_id=user_id,
                requested_ids=request.knowledge_base_ids,
            )
        except AppError as exc:
            failures.append(
                RetrievalFailure(
                    stage="inventory",
                    error_code=exc.code,
                    message=exc.message,
                    dataset_ids=list(request.knowledge_base_ids),
                    retryable=False,
                    recovered=False,
                )
            )
            return None
        by_dataset: dict[str, list[str]] = {}
        total = 0
        for dataset_id in allowed_datasets:
            names: list[str] = []
            started = time.perf_counter()
            try:
                page = 1
                while True:
                    page_docs = await self._ragflow.list_documents(
                        dataset_id,
                        page=page,
                        page_size=1024,
                    )
                    names.extend(item["name"] for item in page_docs)
                    if len(page_docs) < 1024 or page >= 10:
                        break
                    page += 1
            except AppError as exc:
                failures.append(
                    RetrievalFailure(
                        stage="inventory",
                        error_code=exc.code,
                        message=exc.message,
                        dataset_ids=[dataset_id],
                        retryable=exc.code == "RAGFLOW_UNAVAILABLE",
                        recovered=False,
                        latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    )
                )
                continue
            by_dataset[dataset_id] = names
            total += len(names)
        return {"total": total, "by_dataset": by_dataset}

    async def _execute_plan(
        self,
        request: RetrievalSearchRequest,
        *,
        query: NormalizedQuery,
        plan: QueryPlan,
        plan_v2: QueryPlanV2,
        user_id: str,
    ) -> RetrievalExecution:
        # comparison 以及"结合 A 与 B"这类每个 cell 带不同声明主体的 multi_hop 题，
        # 走 cell 驱动执行器（主体发现 → 文档解析 → 按文档限定检索）。所有 cell
        # 共享同一主体（如"同时使用 5G 基站与自动机巢…三类要求"的合并主体、
        # "既要…也要…"）或无 subject 的面式拆分题保持原路径——若误路由，discovery
        # 会把单一主体解析到单一文档，其余来源文档（机巢/河湖等）被全部过滤。
        multi_hop_subjects = {
            cell.subject for cell in plan_v2.cells if cell.subject
        }
        if plan.query_type == "comparison" or (
            plan.query_type == "multi_hop"
            and plan_v2.cells
            and all(cell.subject for cell in plan_v2.cells)
            and len(multi_hop_subjects) >= 2
        ):
            return await self._execute_comparison_plan(
                request,
                query=query,
                plan=plan,
                plan_v2=plan_v2,
                user_id=user_id,
            )
        executions: list[tuple[str, RetrievalExecution]] = []
        missing_subjects: set[str] = set()
        failures: list[RetrievalFailure] = []
        subqueries = plan.subqueries[: self._settings.retrieval.max_complex_subqueries]
        cell_by_id = {cell.id: cell for cell in plan_v2.cells}
        # 单数规则/文档的实体枚举采用两阶段检索：首个“总述”视角在请求范围
        # 内定位目标文档，后续名称/提及视角只扫描该 document_id，避免其它 FAA
        # 规则中高频的 ``received comments from`` 段落混入答案。
        enumeration_document_ids = list(request.document_ids)
        for subquery in subqueries:
            started = time.perf_counter()
            try:
                scoped_document_ids = (
                    enumeration_document_ids
                    if plan.query_type == "enumeration" and enumeration_document_ids
                    else request.document_ids
                )
                execution = await self._execute_single(
                    request.model_copy(
                        update={
                            "query": subquery.query,
                            "document_ids": scoped_document_ids,
                        }
                    ),
                    user_id=user_id,
                    apply_reranker=False,
                    final_limit=self._settings.retrieval.complex_candidates_per_subquery,
                    similarity_threshold=(
                        self._settings.retrieval.complex_similarity_threshold
                    ),
                )
            except AppError as exc:
                # 部分子查询失败不整题失败：记录结构化 failures 后跳过该子查询，
                # 其余子查询继续；最终由调用方决定是部分回答还是服务错误。
                failures.append(
                    RetrievalFailure(
                        stage="subquery",
                        error_code=exc.code,
                        message=exc.message,
                        dataset_ids=list(request.knowledge_base_ids),
                        retryable=exc.code == "RAGFLOW_UNAVAILABLE",
                        recovered=False,
                        latency_ms=round((time.perf_counter() - started) * 1000, 2),
                    )
                )
                continue
            if (
                plan.query_type == "enumeration"
                and not enumeration_document_ids
                and execution.selected_chunks
            ):
                # selected_chunks 已按本视角的检索得分排序；问题指向单一规则时，
                # 第一名所属文档就是随后文档内扫描的边界。
                enumeration_document_ids = [execution.selected_chunks[0].document_id]
                scoped_document_ids = enumeration_document_ids
            # 单主体 multi_hop（如"同时使用5G基站与自动机巢…空域通信/机巢
            # 运维/数据归档三类要求"）的每个子查询也是全库检索：aspect 词被
            # 完整长主语稀释后，该方面的独立语义检索视角可能漏掉字面相关但
            # 排名靠后的章节（如河湖规程的"9 资料归档"）。与 comparison 路径
            # 一致，用 aspect 的补充查询再检索一次并融合，而非拼进主查询。
            planned_cell = cell_by_id.get(subquery.id)
            if planned_cell is not None:
                semantic_query = self._secondary_cell_retrieval_query(planned_cell)
                if semantic_query is not None:
                    try:
                        semantic_execution = await self._execute_single(
                            request.model_copy(
                                update={
                                    "query": semantic_query,
                                    "document_ids": scoped_document_ids,
                                }
                            ),
                            user_id=user_id,
                            apply_reranker=False,
                            final_limit=(
                                self._settings.retrieval.complex_candidates_per_subquery
                            ),
                            similarity_threshold=(
                                self._settings.retrieval.complex_similarity_threshold
                            ),
                        )
                    except AppError:
                        semantic_execution = None
                    if semantic_execution is not None:
                        fused = self._fusion.fuse(
                            [self._as_retrieved(item) for item in execution.selected_chunks],
                            [
                                self._as_retrieved(item)
                                for item in semantic_execution.selected_chunks
                            ],
                        )
                        execution.selected_chunks = [
                            SelectedChunk(citation_id="", **item.model_dump())
                            for item in fused
                        ]
                        execution.candidate_count += semantic_execution.candidate_count
                        execution.candidates.extend(semantic_execution.candidates)
            executions.append((subquery.id, execution))

        if not executions:
            raise AppError(
                code="RAGFLOW_ALL_SUBQUERIES_FAILED",
                message="所有检索子查询均失败，无法完成检索。",
                status_code=503,
                details={"failures": [failure.model_dump() for failure in failures]},
            )

        results = self._complex_results(executions)
        complex_reranker_used = await self._rerank_complex_candidates(
            plan,
            results,
        )
        selection = self._coverage_selector.select(
            plan,
            results,
            limit=(
                self._settings.retrieval.enumeration_final_limit
                if plan.query_type == "enumeration"
                else self._settings.retrieval.complex_final_limit
            ),
            missing_subjects=missing_subjects,
        )
        selected, citations = self._citation_service.build(selection.chunks)
        # 把"chunk 属于哪个子查询/对比主体"带到下游（subject grounding 校验）。
        selected = [
            item.model_copy(
                update={"subquery_id": selection.chunk_subqueries.get(item.chunk_id)}
            )
            for item in selected
        ]
        citation_by_chunk = {item.chunk_id: item.citation_id for item in selected}
        for cell in selection.matrix:
            cell.citation_ids = [
                citation_by_chunk[chunk_id]
                for chunk_id in cell.chunk_ids
                if chunk_id in citation_by_chunk
            ]

        candidates: list[RetrievalCandidateDebug] = []
        for subquery_id, execution in executions:
            for candidate in execution.candidates:
                copied = candidate.model_copy(update={"subquery_id": subquery_id})
                copied.selected = copied.chunk_id in citation_by_chunk
                copied.citation_id = citation_by_chunk.get(copied.chunk_id)
                candidates.append(copied)
        first_execution = executions[0][1]
        return RetrievalExecution(
            request_id=current_request_id.get() or str(uuid4()),
            query=query,
            selected_chunks=selected,
            citations=citations,
            candidate_count=sum(item.candidate_count for _, item in executions),
            reranker_used=(
                complex_reranker_used
                or any(item.reranker_used for _, item in executions)
            ),
            reranker_fallback=any(item.reranker_fallback for _, item in executions),
            ragflow_request=first_execution.ragflow_request,
            stage_counts={
                "subqueries": len(executions),
                "retrieval_calls": len(executions),
                "ragflow_returned": sum(
                    item.stage_counts.get("ragflow_returned", 0)
                    for _, item in executions
                ),
                "post_filter_accepted": sum(
                    item.stage_counts.get("post_filter_accepted", 0)
                    for _, item in executions
                ),
                "final_selected": len(selected),
                "coverage_cells": len(selection.matrix),
                "covered_cells": sum(
                    cell.status == "covered" for cell in selection.matrix
                ),
                **(
                    {"enumeration_document_locked": 1}
                    if plan.query_type == "enumeration" and enumeration_document_ids
                    else {}
                ),
                **({"subquery_failures": len(failures)} if failures else {}),
            },
            candidates=candidates,
            query_plan=plan,
            coverage_matrix=selection.matrix,
            failures=failures,
            deterministic_fallback=any(
                item.deterministic_fallback for _, item in executions
            ),
        )

    async def _execute_comparison_plan(
        self,
        request: RetrievalSearchRequest,
        *,
        query: NormalizedQuery,
        plan: QueryPlan,
        plan_v2: QueryPlanV2,
        user_id: str,
    ) -> RetrievalExecution:
        """Resolve subject documents once, then retrieve and fuse each matrix cell."""
        semaphore = asyncio.Semaphore(
            self._settings.retrieval.max_parallel_ragflow_requests
        )

        async def retrieve_once(
            cell_request: RetrievalSearchRequest,
        ) -> RetrievalExecution:
            async with semaphore:
                return await self._execute_single(
                    cell_request,
                    user_id=user_id,
                    apply_reranker=False,
                    final_limit=self._settings.retrieval.per_cell_candidate_top_k,
                    similarity_threshold=(
                        self._settings.retrieval.complex_similarity_threshold
                    ),
                )

        retry_count = 0

        async def retrieve_with_retry(
            cell_request: RetrievalSearchRequest,
        ) -> tuple[RetrievalExecution | None, AppError | None]:
            nonlocal retry_count
            try:
                return await retrieve_once(cell_request), None
            except AppError:
                retry_count += 1
                try:
                    return await retrieve_once(cell_request), None
                except AppError as retry_error:
                    return None, retry_error

        subjects = list(
            dict.fromkeys(cell.subject for cell in plan_v2.cells if cell.subject)
        )
        discovery: dict[str, RetrievalExecution] = {}
        if not request.document_ids:
            discovered = await asyncio.gather(
                *(
                    retrieve_with_retry(
                        request.model_copy(
                            update={
                                "query": self._subject_discovery_query(subject),
                                "document_ids": [],
                            }
                        )
                    )
                    for subject in subjects
                )
            )
            discovery_errors: list[AppError] = []
            for subject, (execution, error) in zip(subjects, discovered, strict=True):
                if execution is not None:
                    discovery[subject] = execution
                if error is not None:
                    discovery_errors.append(error)
            if discovery_errors and not discovery:
                raise discovery_errors[0]

        resolutions: dict[str, SubjectResolution] = {}
        for subject in subjects:
            subject_discovery = discovery.get(subject)
            candidates = [
                self._as_retrieved(item)
                for item in (
                    subject_discovery.selected_chunks if subject_discovery is not None else []
                )
            ]
            resolutions[subject] = self._subject_resolver.resolve(
                subject,
                candidates,
                requested_document_ids=request.document_ids,
            )

        missing_subjects = {
            subject for subject, resolution in resolutions.items() if not resolution.document_ids
        }
        cell_by_id = {cell.id: cell for cell in plan_v2.cells}
        executions: list[tuple[str, RetrievalExecution]] = []
        translation_attempts = 0
        translation_calls = 0
        translation_latency_ms = 0.0
        translated_cell_ids: list[str] = []
        section_expansion_calls = 0
        semantic_expansion_calls = 0
        for subquery in plan.subqueries:
            cell = cell_by_id.get(subquery.id)
            if cell is None or not cell.subject:
                continue
            document_ids = resolutions[cell.subject].document_ids
            if not document_ids:
                continue
            scoped_request = request.model_copy(
                update={
                    "query": self._cell_retrieval_query(cell),
                    "document_ids": document_ids,
                }
            )
            execution, _ = await retrieve_with_retry(scoped_request)
            if execution is None:
                continue
            section_expansion_calls += await self._expand_section_family(
                execution,
                subject_document_ids=document_ids,
                subject=cell.subject,
            )
            execution.selected_chunks = [
                item for item in execution.selected_chunks if item.document_id in document_ids
            ]

            # A broad aspect gets one independently ranked semantic view.  It
            # is fused with the original result instead of diluting the dense
            # query by appending a long synonym list.
            semantic_query = self._secondary_cell_retrieval_query(cell)
            if semantic_query is not None:
                semantic_execution, _ = await retrieve_with_retry(
                    request.model_copy(
                        update={
                            "query": semantic_query,
                            "document_ids": document_ids,
                        }
                    )
                )
                if semantic_execution is not None:
                    semantic_expansion_calls += 1
                    fused = self._fusion.fuse(
                        [self._as_retrieved(item) for item in execution.selected_chunks],
                        [
                            self._as_retrieved(item)
                            for item in semantic_execution.selected_chunks
                        ],
                    )
                    execution.selected_chunks = [
                        SelectedChunk(citation_id="", **item.model_dump())
                        for item in fused
                        if item.document_id in document_ids
                    ]
                    execution.candidate_count += semantic_execution.candidate_count
                    execution.candidates.extend(semantic_execution.candidates)

            has_evidence = cell_has_substantive_evidence(
                execution.selected_chunks,
                cell.aspect,
            )
            if (
                not has_evidence
                and translation_attempts
                < self._settings.translation.max_total_translations
            ):
                translation_attempts += 1
                outcome = await self._query_translator.translate(
                    query=query,
                    cell=cell,
                    section_titles=[
                        item.metadata.section_title
                        or self._chunk_section(item)
                        or ""
                        for item in execution.selected_chunks
                    ],
                )
                translation_calls += outcome.model_calls
                translation_latency_ms += outcome.latency_ms
                if outcome.decision.should_translate:
                    translated_query = outcome.decision.translated_query or ""
                    translated_execution, _ = await retrieve_with_retry(
                        request.model_copy(
                            update={
                                "query": translated_query,
                                "document_ids": document_ids,
                            }
                        )
                    )
                    if translated_execution is None:
                        executions.append((cell.id, execution))
                        continue
                    fused = self._fusion.fuse(
                        [self._as_retrieved(item) for item in execution.selected_chunks],
                        [
                            self._as_retrieved(item)
                            for item in translated_execution.selected_chunks
                        ],
                    )
                    execution.selected_chunks = [
                        SelectedChunk(citation_id="", **item.model_dump())
                        for item in fused
                        if item.document_id in document_ids
                    ]
                    execution.candidate_count += translated_execution.candidate_count
                    execution.candidates.extend(translated_execution.candidates)
                    translated_cell_ids.append(cell.id)
            executions.append((cell.id, execution))

        results = self._complex_results(executions)
        complex_reranker_used = await self._rerank_complex_candidates(plan, results)
        selection = self._coverage_selector.select(
            plan,
            results,
            limit=self._settings.retrieval.complex_final_limit,
            missing_subjects=missing_subjects,
        )
        selected, citations = self._citation_service.build(selection.chunks)
        selected = [
            item.model_copy(
                update={"subquery_id": selection.chunk_subqueries.get(item.chunk_id)}
            )
            for item in selected
        ]
        citation_by_chunk = {item.chunk_id: item.citation_id for item in selected}
        for coverage in selection.matrix:
            coverage.citation_ids = [
                citation_by_chunk[chunk_id]
                for chunk_id in coverage.chunk_ids
                if chunk_id in citation_by_chunk
            ]

        debug_candidates: list[RetrievalCandidateDebug] = []
        for cell_id, execution in executions:
            for candidate in execution.candidates:
                copied = candidate.model_copy(update={"subquery_id": cell_id})
                copied.selected = copied.chunk_id in citation_by_chunk
                copied.citation_id = citation_by_chunk.get(copied.chunk_id)
                debug_candidates.append(copied)
        all_executions = [*discovery.values(), *(item for _, item in executions)]
        first_execution = all_executions[0]
        return RetrievalExecution(
            request_id=current_request_id.get() or str(uuid4()),
            query=query,
            selected_chunks=selected,
            citations=citations,
            candidate_count=sum(item.candidate_count for item in all_executions),
            reranker_used=complex_reranker_used,
            reranker_fallback=any(item.reranker_fallback for item in all_executions),
            ragflow_request=first_execution.ragflow_request,
            stage_counts={
                "subqueries": len(plan.subqueries),
                "retrieval_calls": (
                    len(all_executions)
                    + len(translated_cell_ids)
                    + section_expansion_calls
                    + semantic_expansion_calls
                    + retry_count
                ),
                "subject_discovery_calls": len(discovery),
                "translated_retrieval_calls": len(translated_cell_ids),
                "section_expansion_calls": section_expansion_calls,
                "semantic_expansion_calls": semantic_expansion_calls,
                "ragflow_retries": retry_count,
                "ragflow_returned": sum(
                    item.stage_counts.get("ragflow_returned", 0)
                    for item in all_executions
                ),
                "final_selected": len(selected),
                "coverage_cells": len(selection.matrix),
                "covered_cells": sum(
                    cell.status == "covered" for cell in selection.matrix
                ),
            },
            candidates=debug_candidates,
            query_plan=plan,
            coverage_matrix=selection.matrix,
            query_plan_v2=plan_v2,
            subject_resolutions=list(resolutions.values()),
            translated_cell_ids=translated_cell_ids,
            model_call_counts={"translation": translation_calls},
            stage_latencies_ms={"translation": round(translation_latency_ms, 2)},
        )

    @staticmethod
    def _cell_retrieval_query(cell: PlannedCell) -> str:
        """Use the planned cell query without concatenating a keyword bag.

        The aspect taxonomy is useful for classifying returned evidence, but
        appending every synonym to a dense-retrieval query dilutes its intent.
        In live evaluation the concise acceptance query ranked the normative
        precision table first; the keyword bag pushed it outside the cell's
        candidate window.
        """
        return cell.original_query

    @staticmethod
    def _subject_discovery_query(subject: str) -> str:
        """Bridge user-facing subjects to terminology used by document titles.

        Subject resolution happens before per-cell retrieval and therefore must
        receive the same bounded domain expansion as the legacy query planner.
        Without it, broad literal matches such as a generic plant-protection
        standard can become the document scope for two distinct specialist
        subjects, after which every matrix cell is forced into that one file.
        """
        expansion = SUBJECT_RETRIEVAL_TERMS.get(subject)
        return f"{subject} {expansion}" if expansion else subject

    @staticmethod
    def _secondary_cell_retrieval_query(cell: PlannedCell) -> str | None:
        """按方面生成独立的语义检索查询（与原始查询融合，不拼接稀释）。

        已知方面的扩展词表（ASPECT_SECONDARY_RETRIEVAL_TERMS）描述可复用的
        概念（如"成果验收"的精度/限差/点云密度）；词表未覆盖的方面（如
        "数据归档""机巢运维"）以"主体 + 方面词"作为补充查询——方面词是
        用户问题的维度，主体提供文档归属上下文（如"河湖"限定才能命中河湖
        规程的"9 资料归档"章节；纯方面词全库竞争会被其它归档内容挤出）。
        词表与回退都是"独立检索视角"而非拼进主查询，不污染其它 cell。
        """
        terms = ASPECT_SECONDARY_RETRIEVAL_TERMS.get(cell.aspect)
        if terms:
            return f"{cell.subject or ''} {terms}".strip()
        if cell.aspect:
            return f"{cell.subject or ''} {cell.aspect}".strip()
        return None

    async def _execute_single(
        self,
        request: RetrievalSearchRequest,
        *,
        plan: QueryPlan | None = None,
        user_id: str,
        apply_reranker: bool = True,
        final_limit: int | None = None,
        similarity_threshold: float | None = None,
    ) -> RetrievalExecution:
        query = self._query_analyzer.analyze(request.query)
        # 流程/汇总类问题（完整操作流程、关键环节、从…到…）需要覆盖各环节
        # 章节：放宽重排输入与最终名额，并按"文档+章节"多样性选择，确保
        # 规划/作业/处理/验证/归档等每个环节都有代表进证据（同一主题存在
        # 多份规程时各自章节也能进入）。
        is_process_question = plan is not None and plan.query_type in {"procedure", "summary"}
        # 问题显式限定规程类别（"结合 A、B 两类…规程/规范"）时，procedure/summary
        # 类问题只允许声明类别内的文档进入生成证据（域过滤）。
        scope_cores: list[str] = []
        if is_process_question and self._settings.retrieval.enforce_declared_scope:
            scope_cores = declared_scope_cores(query.normalized_query)
        rerank_output_k = (
            self._settings.retrieval.procedure_rerank_top_k
            if is_process_question
            else self._settings.retrieval.rerank_top_k
        )
        resolved_final_limit = final_limit or (
            self._settings.retrieval.procedure_final_limit
            if is_process_question
            else min(
                self._settings.retrieval.rerank_top_k,
                self._settings.retrieval.max_final_chunks,
            )
        )
        allowed_datasets = await self._access_control.authorize_knowledge_bases(
            user_id=user_id,
            requested_ids=request.knowledge_base_ids,
        )
        if self._ragflow is None:
            raise AppError(
                code="RAGFLOW_DISABLED",
                message="RAGFlow retrieval is not enabled.",
                status_code=503,
            )

        metadata_values = request.filters.metadata_values()
        # Legacy externally published chunks carry lifecycle state in tags,
        # not RAGFlow document_metadata. Pushing the implicit
        # status=effective condition down would make RAGFlow return zero
        # candidates before this service can apply its backward-compatible
        # local filter.
        ragflow_metadata_values = {
            name: value
            for name, value in metadata_values.items()
            if name != "status"
        }
        metadata_condition = (
            MetadataConditions(
                conditions=[
                    MetadataCondition(name=name, value=value)
                    for name, value in ragflow_metadata_values.items()
                ]
            )
            if ragflow_metadata_values
            else None
        )
        candidate_top_k = min(
            request.candidate_top_k or self._settings.retrieval.candidate_top_k,
            self._settings.retrieval.candidate_top_k,
        )
        retrieval_question = query.normalized_query
        if query.article_aliases:
            retrieval_question += "\n精确条号：" + " ".join(query.article_aliases)
        ragflow_request = RagflowRetrievalRequest(
            question=retrieval_question,
            dataset_ids=allowed_datasets,
            document_ids=request.document_ids,
            page_size=candidate_top_k,
            similarity_threshold=(
                similarity_threshold
                if similarity_threshold is not None
                else self._settings.retrieval.similarity_threshold
            ),
            vector_similarity_weight=self._settings.retrieval.vector_similarity_weight,
            # RAGFlow's ``keyword`` option calls the tenant chat model before
            # retrieval. Article identifiers are already preserved verbatim in
            # ``question`` and handled by exact ownership ranking below, so they
            # must not implicitly trigger that slow, generative expansion.
            keyword=self._settings.retrieval.enable_ragflow_keyword_extraction,
            metadata_condition=metadata_condition,
        )
        raw_chunks = await self._ragflow.retrieve(ragflow_request)
        failures: list[RetrievalFailure] = []
        translation_calls = 0
        translated_retrieval_calls = 0
        # Single-query article lookups never visit the multi-cell translation
        # path. Add at most one bounded variant here, retaining original hits,
        # access filters and document IDs. Never translate recursive cell calls.
        if (
            plan is not None and not plan.requires_multi_query
            and query.article_ids and has_translatable_topic(query)
            and self._settings.translation.max_total_translations > 0
        ):
            cell = PlannedCell(
                id="q1", subject=None, aspect=query.original_query[:200],
                original_query=query.original_query[:500],
                retrieval_queries=[RetrievalQuery(
                    kind="original", text=query.original_query[:500], generated_by="user",
                )],
            )
            outcome = await self._query_translator.translate(
                query=query, cell=cell,
                section_titles=[c.metadata.section_title or "" for c in raw_chunks[:5]],
            )
            translation_calls = outcome.model_calls
            if outcome.decision.should_translate:
                try:
                    supplemental = await self._ragflow.retrieve(ragflow_request.model_copy(
                        update={"question": outcome.decision.translated_query},
                    ))
                    translated_retrieval_calls = 1
                    raw_chunks.extend(supplemental)
                except AppError as exc:
                    failures.append(RetrievalFailure(
                        stage="article_translation", error_code=exc.code,
                        message=exc.message, dataset_ids=allowed_datasets,
                        document_ids=request.document_ids,
                    ))
        deterministic_fallback = False
        if (
            not raw_chunks
            and self._settings.retrieval.enable_deterministic_fallback
        ):
            # 零命中 ≠ 未收录：用精确锚点（条款号/标准号/版本/引号术语/查询专有词）
            # 走 RAGFlow 关键词模式再检索一轮。结果与首次合并后统一过过滤、
            # rerank 与证据门禁；来源标记 deterministic_fallback，不伪装成向量命中。
            fallback_request = self._anchor_fallback_request(ragflow_request, query)
            if fallback_request is not None:
                fallback_started = time.perf_counter()
                try:
                    fallback_chunks = await self._ragflow.retrieve(fallback_request)
                except AppError as exc:
                    fallback_chunks = []
                    failures.append(
                        RetrievalFailure(
                            stage="deterministic_fallback",
                            error_code=exc.code,
                            message=exc.message,
                            dataset_ids=list(request.knowledge_base_ids),
                            document_ids=list(request.document_ids),
                            retryable=exc.code == "RAGFLOW_UNAVAILABLE",
                            recovered=False,
                            latency_ms=round(
                                (time.perf_counter() - fallback_started) * 1000,
                                2,
                            ),
                        )
                    )
                if fallback_chunks:
                    raw_chunks = [
                        chunk.model_copy(
                            update={"retrieval_stage": "deterministic_fallback"}
                        )
                        for chunk in fallback_chunks
                    ]
                    deterministic_fallback = True
        debug_items: list[RetrievalCandidateDebug] = []
        accepted: list[RetrievedChunk] = []
        seen: set[tuple[str, str]] = set()
        for raw_rank, chunk in enumerate(raw_chunks, start=1):
            reason = self._filter_reason(
                chunk,
                allowed_datasets=allowed_datasets,
                document_ids=request.document_ids,
                metadata_values=metadata_values,
                article_aliases=query.article_aliases,
                article_ids=query.article_ids,
                similarity_threshold=similarity_threshold,
            )
            dedup_key = (chunk.document_id, " ".join(chunk.text.split()))
            if reason is None and dedup_key in seen:
                reason = "duplicate_chunk"
            if reason is None:
                seen.add(dedup_key)
                accepted.append(chunk)
            debug_items.append(self._debug_item(chunk, raw_rank=raw_rank, reason=reason))

        accepted.sort(
            key=lambda item: (
                owns_requested_article(item, query.article_ids),
                item.hybrid_score,
            ),
            reverse=True,
        )
        if is_process_question:
            await self._supplement_process_documents(
                accepted,
                seen=seen,
                query=query,
                retrieval_question=retrieval_question,
                allowed_datasets=allowed_datasets,
                candidate_top_k=candidate_top_k,
                similarity_threshold=similarity_threshold,
                metadata_values=metadata_values,
                scope_cores=scope_cores,
            )
            accepted = self._filter_process_topic_documents(
                accepted,
                query.normalized_query,
            )
            accepted.sort(
                key=lambda item: (
                    owns_requested_article(item, query.article_ids),
                    item.hybrid_score,
                ),
                reverse=True,
            )
        scope_excluded_ids: set[str] = set()
        if scope_cores:
            # 域过滤：范围外文档（如马铃薯规范）不进 rerank/最终证据，其内容
            # 不得充当问题限定类别的通用作业要求。
            accepted, scope_excluded_ids = self._filter_declared_scope(
                accepted,
                scope_cores,
            )
        if is_process_question:
            # 同名/同主题多规程场景（如茂名/省两份松材线虫规程）：按 hybrid
            # 分数取前 N 会让其中一份的流程章节全被挤出。改为同名对文档均衡
            # 轮转（每份文档按章节多样性取代表），其余文档按分数补足，保证
            # 各规程的流程章节都有机会进入重排与最终证据。
            rerank_input = self._balanced_rerank_input(
                accepted,
                limit=self._settings.retrieval.procedure_rerank_input_k,
            )
        else:
            rerank_input = accepted[: self._settings.retrieval.rerank_input_k]
        exact_matches = [
            item
            for item in rerank_input
            if owns_requested_article(item, query.article_ids)
        ]
        reranker_used = False
        reranker_fallback = False
        ranked: list[RetrievedChunk]
        if exact_matches:
            exact_ids = {item.chunk_id for item in exact_matches}
            remaining = [item for item in rerank_input if item.chunk_id not in exact_ids]
            ranked = [
                item.model_copy(update={"rerank_score": 1.0})
                for item in exact_matches
            ]
            ranked.extend(self._fallback_rank(remaining))
        elif apply_reranker and rerank_input and self._reranker is not None:
            try:
                results = await self._reranker.rerank(
                    query=query.normalized_query,
                    documents=[item.text for item in rerank_input],
                    top_n=min(
                        rerank_output_k,
                        len(rerank_input),
                    ),
                )
                ranked = [
                    rerank_input[result.index].model_copy(update={"rerank_score": result.score})
                    for result in results
                    if result.index < len(rerank_input)
                ]
                ranked.sort(
                    key=lambda item: (
                        item.rerank_score if item.rerank_score is not None else float("-inf")
                    ),
                    reverse=True,
                )
                ranked = [
                    item
                    for item in ranked
                    if item.rerank_score is not None
                    and item.rerank_score >= self._settings.retrieval.min_rerank_score
                ]
                reranker_used = True
            except AppError:
                if not self._settings.retrieval.allow_rerank_fallback:
                    raise
                ranked = self._fallback_rank(rerank_input)
                reranker_fallback = True
        elif apply_reranker and rerank_input:
            if (
                self._settings.models.reranker.required
                or not self._settings.retrieval.allow_rerank_fallback
            ):
                raise AppError(
                    code="RERANKER_DISABLED",
                    message="The reranker is required but not enabled.",
                    status_code=503,
                )
            ranked = self._fallback_rank(rerank_input)
            reranker_fallback = True
        else:
            ranked = []

        if not apply_reranker:
            ranked = self._fallback_rank(rerank_input)
        if is_process_question:
            # 章节多样性 + 同名多规程文档配额：流程各环节（航摄规划/作业准备/
            # 影像获取/数据处理/地面验证/档案管理等）都有代表进入证据；同名
            # 多规程时每份规程按文档配额分配名额（基于 rerank 输入而非被裁剪
            # 的 rerank 输出，避免省标准流程章节被 top-N 裁掉）。
            final_chunks = self._balanced_final_chunks(
                rerank_input,
                limit=resolved_final_limit,
            )
        else:
            final_chunks = ranked[:resolved_final_limit]
        selected, citations = self._citation_service.build(final_chunks)
        selected_by_id = {
            item.chunk_id: (index, item.citation_id) for index, item in enumerate(selected, start=1)
        }
        rerank_scores = {item.chunk_id: item.rerank_score for item in ranked}
        accepted_ids = {item.chunk_id for item in accepted}
        rerank_input_ids = {item.chunk_id for item in rerank_input}
        for item in debug_items:
            item.rerank_score = rerank_scores.get(item.chunk_id)
            if item.chunk_id in selected_by_id:
                item.final_rank, item.citation_id = selected_by_id[item.chunk_id]
                item.selected = True
            elif item.filter_reason is None and item.chunk_id in scope_excluded_ids:
                item.filter_reason = "outside_declared_scope"
            elif item.filter_reason is None and item.chunk_id not in accepted_ids:
                item.filter_reason = "filtered"
            elif item.filter_reason is None and item.chunk_id not in rerank_input_ids:
                item.filter_reason = "outside_rerank_input"
            elif (
                item.filter_reason is None and reranker_used and item.chunk_id not in rerank_scores
            ):
                item.filter_reason = "reranker_not_returned"
            elif item.filter_reason is None:
                item.filter_reason = "below_final_cutoff"

        return RetrievalExecution(
            request_id=current_request_id.get() or str(uuid4()),
            query=query,
            selected_chunks=selected,
            citations=citations,
            candidate_count=len(raw_chunks),
            reranker_used=reranker_used,
            reranker_fallback=reranker_fallback,
            ragflow_request=ragflow_request,
            stage_counts={
                "ragflow_returned": len(raw_chunks),
                "post_filter_accepted": len(accepted),
                "rerank_input": len(rerank_input),
                "article_exact_matches": len(exact_matches),
                "article_translated_retrieval_calls": translated_retrieval_calls,
                "final_selected": len(selected),
                **({"deterministic_fallback": 1} if deterministic_fallback else {}),
            },
            candidates=debug_items,
            model_call_counts={"translator": translation_calls},
            failures=failures,
            deterministic_fallback=deterministic_fallback,
        )

    @staticmethod
    def _anchor_terms(
        query: NormalizedQuery,
        ragflow_request: RagflowRetrievalRequest,
    ) -> list[str]:
        """提取确定性精确锚点：条款号/标准号、版本号、引号术语、查询专有词。

        用于零命中兜底的关键词模式检索；只保留 2 字符以上且去重保序的锚点。
        """
        anchors: list[str] = []
        anchors.extend(query.article_aliases)
        anchors.extend(query.article_ids)
        anchors.extend(
            re.findall(
                r"[\u201c\"]([^\u201d\"]{2,40})[\u201d\"]",
                ragflow_request.question,
            )
        )
        anchors.extend(
            re.findall(
                r"\b[A-Z]{2,6}-\d+(?:[A-Z]-\d+)?(?:[-.]\d+)*\b",
                ragflow_request.question,
            )
        )
        anchors.extend(
            re.findall(r"\bv\d+(?:\.\d+){1,3}\b", ragflow_request.question, re.IGNORECASE)
        )
        return list(dict.fromkeys(anchor for anchor in anchors if len(anchor) >= 2))[:12]

    @classmethod
    def _anchor_fallback_request(
        cls,
        ragflow_request: RagflowRetrievalRequest,
        query: NormalizedQuery,
    ) -> RagflowRetrievalRequest | None:
        """构造关键词模式兜底请求；锚点与原始查询无差异时返回 None（避免无意义重试）。"""
        anchors = cls._anchor_terms(query, ragflow_request)
        if not anchors:
            return None
        anchor_query = " ".join(anchors)
        if " ".join(ragflow_request.question.split()) == anchor_query:
            return None
        return ragflow_request.model_copy(
            update={
                "question": anchor_query,
                "keyword": True,
                "similarity_threshold": min(ragflow_request.similarity_threshold, 0.2),
                "page_size": min(ragflow_request.page_size, 20),
            }
        )

    @staticmethod
    def _as_retrieved(chunk: SelectedChunk) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            dataset_id=chunk.dataset_id,
            text=chunk.text,
            metadata=chunk.metadata,
            hybrid_score=chunk.hybrid_score,
            vector_score=chunk.vector_score,
            keyword_score=chunk.keyword_score,
            rerank_score=chunk.rerank_score,
        )

    @classmethod
    def _filter_process_topic_documents(
        cls,
        chunks: list[RetrievedChunk],
        query: str,
    ) -> list[RetrievedChunk]:
        """Drop generic process documents when one title clearly matches the topic.

        Procedure questions often have no explicit planner subject. A generic
        ``无人机/航摄`` overlap can therefore pull unrelated manuals into the
        final document rotation. When the question covers at least 30% title
        affinity for a clear leading topic, keep that topic family (including
        same-title regional/year variants) and exclude distant titles. If no
        title is a clear match, preserve the original candidates.
        """
        from app.services.multi_document import document_title

        by_document: dict[str, list[RetrievedChunk]] = {}
        for item in chunks:
            name = item.metadata.document_name or ""
            by_document.setdefault(name, []).append(item)
        if len(by_document) < 2:
            return chunks
        scores = {
            name: title_similarity(document_title(name), query)
            for name in by_document
            if name
        }
        if not scores:
            return chunks
        best = max(scores.values())
        if best < 0.30:
            return chunks
        cutoff = max(0.30, best - 0.08)
        allowed_names = {name for name, score in scores.items() if score >= cutoff}
        if not allowed_names:
            return chunks
        return [
            item for item in chunks if (item.metadata.document_name or "") in allowed_names
        ]

    @classmethod
    def _balanced_rerank_input(
        cls,
        accepted: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """流程类问题的 rerank 输入：流程环节保底（文档相关性优先）+ 补足。

        流程题的证据名额按 rerank/hybrid 分数截断时，高分章节（术语/基本
        要求）和字面贴近问题的前段章节会占满名额，把数据处理、地面验证、
        档案管理等后半段环节挤出。流程环节保底（_process_section_floor）
        为每个规程文档的每个顶层流程章节保留代表，且**文档按相关性排序**：
        最相关文档（如最贴题的马铃薯规程）的完整流程环节优先进入，其余
        文档按相关性依次分配，避免多份同主题规程均摊名额导致每份都不完整。
        剩余名额按章节多样性（_diverse_by_section）补足。
        """
        def _doc_score(name: str) -> float:
            scores = [
                item.rerank_score if item.rerank_score is not None else item.hybrid_score
                for item in accepted
                if (item.metadata.document_name or "") == name
            ]
            return max(scores, default=0.0)

        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()
        # 流程环节保底：对**所有**能解析出流程章节的文档生效（不限于同名对）。
        # 文档按相关性降序，最相关文档的完整流程环节优先填满名额。
        floor = cls._process_section_floor(accepted)
        floor_by_document: dict[str, list[RetrievedChunk]] = {}
        for item in floor:
            name = item.metadata.document_name or ""
            floor_by_document.setdefault(name, []).append(item)
        names = sorted(
            floor_by_document,
            key=lambda name: _doc_score(name),
            reverse=True,
        )
        round_index = 0
        while len(selected) < limit:
            progressed = False
            for name in names:
                if round_index >= len(floor_by_document[name]):
                    continue
                candidate = floor_by_document[name][round_index]
                if candidate.chunk_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
                progressed = True
                if len(selected) >= limit:
                    break
            if not progressed:
                break
            round_index += 1
        if len(selected) < limit:
            remaining = cls._diverse_by_section(
                [item for item in accepted if item.chunk_id not in selected_ids],
                limit=limit,
            )
            for candidate in remaining:
                if len(selected) >= limit:
                    break
                if candidate.chunk_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
        if len(selected) < limit:
            for item in accepted:
                if len(selected) >= limit:
                    break
                if item.chunk_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.chunk_id)
        return selected

    @classmethod
    def _balanced_final_chunks(
        cls,
        rerank_input: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """最终证据选择：流程环节保底（文档相关性优先）+ 章节多样性补足。

        基于 rerank 输入（而非被 top-N 裁剪的 rerank 输出）选择。每个规程
        文档的每个顶层流程章节至少保留一个代表，且**文档按相关性排序**：
        最相关文档（如最贴题的马铃薯规程）的完整流程环节优先进入最终证据，
        其余文档按相关性依次分配，避免多份同主题规程均摊名额导致每份都不
        完整。
        """
        def _doc_score(name: str) -> float:
            scores = [
                item.rerank_score if item.rerank_score is not None else item.hybrid_score
                for item in rerank_input
                if (item.metadata.document_name or "") == name
            ]
            return max(scores, default=0.0)

        selected: list[RetrievedChunk] = []
        selected_ids: set[str] = set()
        # 流程环节保底：对**所有**能解析出流程章节的文档生效（不限于同名对）。
        # 文档按相关性降序，最相关文档的完整流程环节优先填满名额。
        floor = cls._process_section_floor(rerank_input)
        floor_by_document: dict[str, list[RetrievedChunk]] = {}
        for item in floor:
            name = item.metadata.document_name or ""
            floor_by_document.setdefault(name, []).append(item)
        names = sorted(
            floor_by_document,
            key=lambda name: _doc_score(name),
            reverse=True,
        )
        round_index = 0
        while len(selected) < limit:
            progressed = False
            for name in names:
                if round_index >= len(floor_by_document[name]):
                    continue
                candidate = floor_by_document[name][round_index]
                if candidate.chunk_id in selected_ids:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
                progressed = True
                if len(selected) >= limit:
                    break
            if not progressed:
                break
            round_index += 1
        if len(selected) < limit:
            remaining = cls._diverse_by_section(
                [item for item in rerank_input if item.chunk_id not in selected_ids],
                limit=limit,
            )
            for item in remaining:
                if len(selected) >= limit:
                    break
                selected.append(item)
                selected_ids.add(item.chunk_id)
        if len(selected) < limit:
            for item in rerank_input:
                if len(selected) >= limit:
                    break
                if item.chunk_id in selected_ids:
                    continue
                selected.append(item)
                selected_ids.add(item.chunk_id)
        return selected

    async def _supplement_process_documents(
        self,
        accepted: list[RetrievedChunk],
        *,
        seen: set[tuple[str, str]],
        query: NormalizedQuery,
        retrieval_question: str,
        allowed_datasets: list[str],
        candidate_top_k: int,
        similarity_threshold: float | None,
        metadata_values: dict[str, str],
        scope_cores: list[str] | None = None,
    ) -> None:
        """流程类问题的文档级补充检索。

        混合检索按相关度取前 N 候选时，流程后半段章节（如地质灾害规程的
        8.1 空三/8.5 点云解算）会被前半段章节挤掉，根本进不了候选池。对以下
        文档用 document_ids 过滤单独检索一次，其各章节必然进入候选：
        - 问题显式限定规程类别时（scope_cores 非空）：声明类别内的每份文档
          （而非分数最高的主文档——它可能被范围外文档如马铃薯规范抢占）；
        - 同名/同主题多规程的每份参与文档（省标/市标分列场景）；
        - 单主题流程问题中分数最高的主文档（保障其流程全程覆盖）。
        """
        target_names = self._supplement_target_names(accepted, scope_cores)
        if not target_names:
            return
        if self._ragflow is None:
            return
        document_ids: dict[str, str] = {}
        for item in accepted:
            name = item.metadata.document_name or ""
            if name in target_names and name not in document_ids:
                document_ids[name] = item.document_id
        for document_id in document_ids.values():
            try:
                extra_request = RagflowRetrievalRequest(
                    question=retrieval_question,
                    dataset_ids=allowed_datasets,
                    document_ids=[document_id],
                    # 文档身份已锁定，目的是把该文档的**全部流程章节**（含数据
                    # 处理/验证/归档等与问题字面距离远的后半段）拉进候选：page_size
                    # 必须大于文档 chunk 总数，否则"10 档案管理"等低相关度章节会
                    # 被 top-N 截断，保底逻辑也无从发挥作用。RAGFlow page_size
                    # 上限为 100，直接取 100 覆盖整份文档。
                    page_size=100,
                    # The document identity is already fixed. A relevance
                    # threshold here would remove later process sections whose
                    # wording differs from the user's start/end terms (for
                    # example data processing, field validation and archive
                    # management). Retrieve the document outline broadly, then
                    # let section diversity and reranking choose representatives.
                    similarity_threshold=0.0,
                    vector_similarity_weight=self._settings.retrieval.vector_similarity_weight,
                    keyword=False,
                    metadata_condition=None,
                )
                extra_raw = await self._ragflow.retrieve(extra_request)
            except AppError:
                continue
            for chunk in extra_raw:
                reason = self._filter_reason(
                    chunk,
                    allowed_datasets=allowed_datasets,
                    document_ids=None,
                    metadata_values=metadata_values,
                    article_aliases=query.article_aliases,
                    article_ids=query.article_ids,
                    similarity_threshold=0.0,
                )
                dedup_key = (chunk.document_id, " ".join(chunk.text.split()))
                if reason is None and dedup_key in seen:
                    reason = "duplicate_chunk"
                if reason is None:
                    seen.add(dedup_key)
                    accepted.append(chunk)

    async def _expand_section_family(
        self,
        execution: RetrievalExecution,
        *,
        subject_document_ids: list[str],
        subject: str,
    ) -> int:
        """章节族扩展：对执行结果中每个带"章节：…/N.x 叶章节"头的 chunk，
        用叶章节标题做高 BM25 权重的混合检索补充同文档子条款（如 9.4 →
        9.4.1/9.4.2），追加到执行结果。RAGFlow 的 keyword=True 会额外调用
        LLM 做关键词提取；这类标题已经足够精确，必须关闭该选项，避免每个格子
        触发一次串行模型调用并拖垮单 worker API。"""
        if self._ragflow is None or not execution.selected_chunks:
            return 0
        datasets = sorted(
            {item.dataset_id for item in execution.selected_chunks if item.dataset_id}
        )
        if not datasets:
            return 0
        expanded: list[SelectedChunk] = list(execution.selected_chunks)
        expanded_ids = {item.chunk_id for item in expanded}
        seen_families: set[str] = set()
        retrieval_calls = 0
        for chunk in execution.selected_chunks:
            if (
                retrieval_calls
                >= self._settings.retrieval.max_section_expansion_queries_per_cell
            ):
                break
            leaf = self._chunk_section(chunk)
            if not leaf or leaf in seen_families:
                continue
            seen_families.add(leaf)
            try:
                retrieval_calls += 1
                family_request = RagflowRetrievalRequest(
                    question=leaf,
                    dataset_ids=datasets,
                    document_ids=subject_document_ids,
                    page_size=self._settings.retrieval.candidate_top_k,
                    similarity_threshold=0.0,
                    vector_similarity_weight=self._settings.retrieval.vector_similarity_weight,
                    keyword=False,
                    metadata_condition=None,
                )
                family_raw = await self._ragflow.retrieve(family_request)
            except AppError:
                continue
            for item in family_raw:
                if item.chunk_id in expanded_ids:
                    continue
                if item.document_id not in subject_document_ids:
                    continue
                expanded.append(
                    SelectedChunk(
                        # 章节族候选仍处于内部选择阶段；正式引用编号会在
                        # CitationService.build() 中统一重新分配。
                        citation_id="",
                        chunk_id=item.chunk_id,
                        document_id=item.document_id,
                        dataset_id=item.dataset_id,
                        text=item.text,
                        metadata=item.metadata,
                        hybrid_score=item.hybrid_score,
                        vector_score=item.vector_score,
                        keyword_score=item.keyword_score,
                        rerank_score=item.rerank_score,
                    )
                )
                expanded_ids.add(item.chunk_id)
        execution.selected_chunks = expanded
        return retrieval_calls

    @classmethod
    def _supplement_target_names(
        cls,
        accepted: list[RetrievedChunk],
        scope_cores: list[str] | None,
    ) -> set[str]:
        """补充检索的目标文档名集合（纯函数，便于单测）。

        优先级：显式声明类别内的文档 > 同名/同主题对文档 > 分数最高主文档。
        """
        similar_names = cls._similar_document_names(accepted)
        target_names = set(similar_names)
        if scope_cores:
            in_scope_names = {
                item.metadata.document_name
                for item in accepted
                if item.metadata.document_name
                and document_in_scope(item.metadata.document_name, scope_cores)
            }
            if in_scope_names:
                # 声明类别优先：范围外文档（即使分数最高）不作为补充目标。
                return target_names | in_scope_names
            return set()
        if not similar_names and accepted:
            best = max(accepted, key=lambda item: item.hybrid_score)
            if best.metadata.document_name:
                target_names.add(best.metadata.document_name)
        return target_names

    @classmethod
    def _filter_declared_scope(
        cls,
        chunks: list[RetrievedChunk],
        scope_cores: list[str],
    ) -> tuple[list[RetrievedChunk], set[str]]:
        """按声明类别过滤 chunk：保留范围内文档，返回保留列表与被排除 chunk id。"""
        kept: list[RetrievedChunk] = []
        excluded: set[str] = set()
        for item in chunks:
            if document_in_scope(item.metadata.document_name or "", scope_cores):
                kept.append(item)
            else:
                excluded.add(item.chunk_id)
        return kept, excluded

    @staticmethod
    def _similar_document_names(chunks: list[RetrievedChunk]) -> set[str]:
        """返回参与"同名/同主题"对的文档名集合。"""
        from app.services.multi_document import document_title, titles_similar

        names: list[str] = []
        for item in chunks:
            name = item.metadata.document_name or ""
            if name and name not in names:
                names.append(name)
        similar: set[str] = set()
        for index in range(len(names)):
            for other in range(index + 1, len(names)):
                if titles_similar(
                    document_title(names[index]),
                    document_title(names[other]),
                ):
                    similar.add(names[index])
                    similar.add(names[other])
        return similar

    @staticmethod
    def _chunk_section(chunk: Any) -> str:
        """Return the top-level process chapter for section diversity.

        New chunks expose a ``章节：`` path. Legacy RAGFlow chunks in this
        deployment have no section metadata/header, so fall back to the first
        numbered heading in their body (``6.1`` belongs to chapter ``6``).
        """
        match = re.search(r"^章节：\s*(.+)$", chunk.text, re.MULTILINE)
        if match is not None:
            root = match.group(1).strip().split("/", maxsplit=1)[0].strip()
            root_match = re.match(r"^(\d{1,2})", root)
            return f"chapter:{root_match.group(1)}" if root_match else root
        legacy = re.search(
            r"(?m)^\s*(?:(?P<number>\d{1,2})(?:\.\d+)+|(?P<top>\d{1,2}))\s+\S+",
            chunk.text,
        )
        if legacy is None:
            return ""
        number = legacy.group("number") or legacy.group("top")
        return f"chapter:{number}"

    @classmethod
    def _diverse_by_section(
        cls,
        ranked: list[RetrievedChunk],
        *,
        limit: int,
    ) -> list[RetrievedChunk]:
        """按"文档+章节"去重的多样性选择。

        流程类问题中同一环节（如"5 航摄规划"）可能对应同一文档的多个 chunk，
        只保留重排分数最高的一个；不同文档的同名章节（省标/市标）按文档区分，
        各自可进入证据，确保多份规程的流程章节都能覆盖。
        """
        selected: list[RetrievedChunk] = []
        section_counts: dict[tuple[str, str], int] = {}
        for item in ranked:
            if len(selected) >= limit:
                break
            section = cls._chunk_section(item)
            if not section:
                # 无章节头（如术语/附录）的 chunk 不去重，按分数进入。
                selected.append(item)
                continue
            key = (item.document_id, section)
            # A process chapter commonly spans multiple chunks (for example
            # 6.1 assembly/testing and 6.3 device-record checks). Keep two
            # representatives per top-level chapter so the core work is not
            # reduced to whichever subsection has the highest retrieval score.
            if section_counts.get(key, 0) >= 2:
                continue
            section_counts[key] = section_counts.get(key, 0) + 1
            selected.append(item)
        return selected

    @classmethod
    def _process_section_floor(
        cls,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """规程文档的流程环节保底：每文档每顶层流程章节保留一个最高分代表。

        流程类问题的名额选择按 rerank/hybrid 分数轮转时，术语/基本要求等
        高分章节可能占满 22/18 个名额，把数据处理、地面验证、档案管理等
        后半段流程章节挤出（它们字面远离"从航摄规划到成果归档"）。这里按
        "文档+顶层流程章节"取每个环节的最高分代表，保证每个环节至少一个
        代表有机会进入 rerank 输入与最终证据。

        仅对**能解析出 ≥2 个不同顶层流程章节**的文档做保底：这是流程规程
        的结构特征（如"5 飞防作业人员要求 / 6 农药选择及配制要求 / 7 作业
        要求 / 8 作业后处理"），不依赖"1 范围"声明 chunk 是否已在候选内
        （声明 chunk 可能因分数低未进候选，但其它流程章节已在）。无流程
        结构的文档（如巡检要求、作业手册）不参与保底，避免其任意章节被误
        当流程环节占用名额。非流程章节（范围/术语/基本要求/安全注意事项/
        附录）不参与保底。
        """
        def _score(item: RetrievedChunk) -> float:
            return item.rerank_score if item.rerank_score is not None else item.hybrid_score

        # 按文档收集可解析的顶层流程章节标题。
        by_document: dict[str, dict[str, RetrievedChunk]] = {}
        for item in chunks:
            name = item.metadata.document_name or ""
            if not name:
                continue
            top_title = cls._top_chapter_title(item)
            if top_title is None:
                continue
            # "3 术语和定义" → "术语和定义"：与 _NON_PROCESS_SECTION_TITLES 的
            # 纯标题比对，避免编号前缀导致非流程章节混入保底。
            bare_title = re.sub(r"^\d{1,2}\s+", "", top_title).strip()
            if bare_title in _NON_PROCESS_SECTION_TITLES or bare_title.startswith("附录"):
                continue
            existing = by_document.get(name, {}).get(top_title)
            if existing is None or _score(item) > _score(existing):
                by_document.setdefault(name, {})[top_title] = item

        best: dict[tuple[str, str], RetrievedChunk] = {}
        for _name, sections in by_document.items():
            if len(sections) < 2:
                # 非流程规程（如巡检要求）：仅 1 个可解析章节，不参与保底。
                continue
            for top_title, item in sections.items():
                key = (item.document_id, top_title)
                existing = best.get(key)
                if existing is None or _score(item) > _score(existing):
                    best[key] = item
        return sorted(
            best.values(),
            key=lambda item: (item.metadata.document_name or "", item.hybrid_score),
            reverse=True,
        )

    @classmethod
    def _top_chapter_title(cls, chunk: RetrievedChunk) -> str | None:
        """返回 chunk 所属顶层章节的标题（如"5 航摄规划"），无法识别返回 None。"""
        match = re.search(r"^章节：\s*(.+)$", chunk.text, re.MULTILINE)
        if match is not None:
            root = match.group(1).strip().split("/", 1)[0].strip()
            if root:
                return root
        legacy = re.search(
            r"(?m)^\s*(?:(?P<number>\d{1,2})(?:\.\d+)+|(?P<top>\d{1,2}))\s+(?P<title>\S.*)$",
            chunk.text,
        )
        if legacy is None:
            return None
        title = legacy.group("title").strip()
        if not title:
            return None
        number = legacy.group("number") or legacy.group("top")
        return f"{number} {title}"

    def _complex_results(
        self,
        executions: list[tuple[str, RetrievalExecution]],
    ) -> dict[str, list[RetrievedChunk]]:
        return {
            subquery_id: [self._as_retrieved(item) for item in execution.selected_chunks]
            for subquery_id, execution in executions
        }

    async def _rerank_complex_candidates(
        self,
        plan: QueryPlan,
        results: dict[str, list[RetrievedChunk]],
    ) -> bool:
        if self._reranker is None:
            return False
        subqueries = {item.id: item for item in plan.subqueries}
        reranked_any = False
        remaining_budget = (
            self._settings.retrieval.enumeration_rerank_input_k
            if plan.query_type == "enumeration"
            else self._settings.retrieval.complex_rerank_input_k
        )
        for subquery_id, candidates in results.items():
            subquery = subqueries.get(subquery_id)
            if subquery is None or not candidates or remaining_budget <= 0:
                continue
            rerank_input = candidates[:remaining_budget]
            reranked = await self._reranker.rerank(
                query=subquery.query,
                documents=[item.text for item in rerank_input],
                top_n=len(rerank_input),
            )
            scores = {
                rerank_input[item.index].chunk_id: item.score
                for item in reranked
                if item.index < len(rerank_input)
            }
            for candidate in candidates:
                candidate.rerank_score = scores.get(candidate.chunk_id)
            candidates[:] = [
                candidate
                for candidate in candidates
                if candidate.rerank_score is not None
                and candidate.rerank_score
                >= self._settings.retrieval.complex_min_rerank_score
            ]
            candidates.sort(key=self._complex_candidate_score, reverse=True)
            remaining_budget -= len(rerank_input)
            reranked_any = True
        return reranked_any

    @staticmethod
    def _complex_candidate_score(chunk: RetrievedChunk) -> float:
        return chunk.rerank_score if chunk.rerank_score is not None else chunk.hybrid_score

    def _filter_reason(
        self,
        chunk: RetrievedChunk,
        *,
        allowed_datasets: list[str],
        document_ids: list[str] | None = None,
        metadata_values: dict[str, str],
        article_aliases: list[str],
        article_ids: list[str] | None = None,
        similarity_threshold: float | None = None,
    ) -> str | None:
        if chunk.dataset_id not in allowed_datasets:
            return "unauthorized_dataset"
        if document_ids and chunk.document_id not in document_ids:
            return "document_scope_mismatch"
        exact_article_match = owns_requested_article(chunk, article_ids or [])
        if (
            chunk.hybrid_score
            < (
                similarity_threshold
                if similarity_threshold is not None
                else self._settings.retrieval.similarity_threshold
            )
            and not exact_article_match
        ):
            return "below_similarity_threshold"
        metadata = chunk.metadata.model_dump()
        for name, expected in metadata_values.items():
            actual = metadata.get(name)
            if name == "status" and expected == "effective" and actual is None:
                continue
            if actual != expected:
                return "inactive_document" if name == "status" else f"metadata_mismatch:{name}"
        return None

    @staticmethod
    def _fallback_rank(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        return [item.model_copy(update={"rerank_score": item.hybrid_score}) for item in chunks]

    @staticmethod
    def _debug_item(
        chunk: RetrievedChunk,
        *,
        raw_rank: int,
        reason: str | None,
    ) -> RetrievalCandidateDebug:
        return RetrievalCandidateDebug(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            dataset_id=chunk.dataset_id,
            document_name=chunk.metadata.document_name,
            version=chunk.metadata.version,
            chapter_path=chunk.metadata.chapter_path,
            page_number=chunk.metadata.page_number,
            hybrid_score=chunk.hybrid_score,
            vector_score=chunk.vector_score,
            keyword_score=chunk.keyword_score,
            raw_rank=raw_rank,
            filter_reason=reason,
            retrieval_stage=chunk.retrieval_stage,
        )
