from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from app.core.config import Settings
from app.core.exceptions import AppError
from app.core.middleware import current_request_id
from app.ingestion.regulations import text_contains_article_alias
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
    RetrievalSearchRequest,
    RetrievalSearchResponse,
    RetrievedChunk,
    SelectedChunk,
)
from app.services.access_control import AccessControlService
from app.services.citation_service import CitationService
from app.services.coverage_selector import CoverageSelector
from app.services.query_analyzer import QueryAnalyzer
from app.services.query_planner import QueryPlanner
from app.services.ragflow_client import RagflowClient
from app.services.registry import ServiceRegistry
from app.services.reranker_client import RerankerClient


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
    ) -> None:
        self._settings = settings
        self._ragflow = ragflow
        self._reranker = reranker
        self._access_control = access_control
        self._query_analyzer = QueryAnalyzer()
        self._query_planner = QueryPlanner()
        self._coverage_selector = CoverageSelector()
        self._citation_service = CitationService()

    @classmethod
    def from_registry(
        cls,
        settings: Settings,
        registry: ServiceRegistry,
    ) -> RetrievalService:
        return cls(
            settings=settings,
            ragflow=registry.ragflow_client,
            reranker=registry.reranker_client,
            access_control=AccessControlService(settings.access_control),
        )

    async def execute(
        self,
        request: RetrievalSearchRequest,
        *,
        user_id: str,
    ) -> RetrievalExecution:
        query = self._query_analyzer.analyze(request.query)
        plan = self._query_planner.plan(query)
        if not plan.requires_multi_query:
            execution = await self._execute_single(request, user_id=user_id)
            execution.query_plan = plan
            return execution
        try:
            return await self._execute_plan(
                request,
                query=query,
                plan=plan,
                user_id=user_id,
            )
        except AppError:
            execution = await self._execute_single(request, user_id=user_id)
            execution.query_plan = plan
            execution.stage_counts["multi_query_fallback"] = 1
            return execution

    async def _execute_plan(
        self,
        request: RetrievalSearchRequest,
        *,
        query: NormalizedQuery,
        plan: QueryPlan,
        user_id: str,
    ) -> RetrievalExecution:
        executions: list[tuple[str, RetrievalExecution]] = []
        subqueries = plan.subqueries[: self._settings.retrieval.max_complex_subqueries]
        for subquery in subqueries:
            execution = await self._execute_single(
                request.model_copy(update={"query": subquery.query}),
                user_id=user_id,
                apply_reranker=False,
                final_limit=self._settings.retrieval.complex_candidates_per_subquery,
                similarity_threshold=(
                    self._settings.retrieval.complex_similarity_threshold
                ),
            )
            executions.append((subquery.id, execution))

        results = self._complex_results(executions)
        complex_reranker_used = await self._rerank_complex_candidates(
            plan,
            results,
        )
        selection = self._coverage_selector.select(
            plan,
            results,
            limit=self._settings.retrieval.max_final_chunks,
        )
        selected, citations = self._citation_service.build(selection.chunks)
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
            },
            candidates=candidates,
            query_plan=plan,
            coverage_matrix=selection.matrix,
        )

    async def _execute_single(
        self,
        request: RetrievalSearchRequest,
        *,
        user_id: str,
        apply_reranker: bool = True,
        final_limit: int | None = None,
        similarity_threshold: float | None = None,
    ) -> RetrievalExecution:
        query = self._query_analyzer.analyze(request.query)
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
            keyword=(
                self._settings.retrieval.enable_ragflow_keyword_extraction
                or bool(query.article_ids)
            ),
            metadata_condition=metadata_condition,
        )
        raw_chunks = await self._ragflow.retrieve(ragflow_request)
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
                text_contains_article_alias(item.text, query.article_aliases),
                item.hybrid_score,
            ),
            reverse=True,
        )
        rerank_input = accepted[: self._settings.retrieval.rerank_input_k]
        exact_matches = [
            item
            for item in rerank_input
            if text_contains_article_alias(item.text, query.article_aliases)
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
                        self._settings.retrieval.rerank_top_k,
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
        resolved_final_limit = final_limit or min(
            self._settings.retrieval.rerank_top_k,
            self._settings.retrieval.max_final_chunks,
        )
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
                "final_selected": len(selected),
            },
            candidates=debug_items,
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
        remaining_budget = self._settings.retrieval.complex_rerank_input_k
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
        document_ids: list[str],
        metadata_values: dict[str, str],
        article_aliases: list[str],
        similarity_threshold: float | None = None,
    ) -> str | None:
        if chunk.dataset_id not in allowed_datasets:
            return "unauthorized_dataset"
        if document_ids and chunk.document_id not in document_ids:
            return "document_scope_mismatch"
        exact_article_match = text_contains_article_alias(chunk.text, article_aliases)
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
        )
