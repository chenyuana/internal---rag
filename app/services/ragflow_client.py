from __future__ import annotations

import time

import httpx

from app.core.config import RagflowSettings
from app.core.exceptions import AppError
from app.schemas.retrieval import ChunkMetadata, RagflowRetrievalRequest, RetrievedChunk
from app.services.citation_location import published_page_number
from app.services.probe import ProbeResult


class RagflowClient:
    """Small adapter boundary around RAGFlow's configurable HTTP health endpoint."""

    def __init__(
        self,
        settings: RagflowSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {}
        api_key = settings.api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._health_path = settings.health_path
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=settings.timeout_seconds,
            transport=transport,
            trust_env=False,
        )

    async def probe(self) -> ProbeResult:
        started = time.perf_counter()
        try:
            response = await self._client.get(self._health_path.lstrip("/"))
            response.raise_for_status()
            return ProbeResult(
                ready=True,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except httpx.HTTPError as exc:
            return ProbeResult(
                ready=False,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=type(exc).__name__,
            )

    async def retrieve(self, request: RagflowRetrievalRequest) -> list[RetrievedChunk]:
        """Call RAGFlow's official retrieval endpoint and normalize versioned fields."""
        try:
            response = await self._client.post(
                "api/v1/retrieval",
                json=request.to_payload(),
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError(
                code="RAGFLOW_UNAVAILABLE",
                message="RAGFlow is unavailable or returned invalid JSON.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc

        if not isinstance(payload, dict):
            raise AppError(
                code="RAGFLOW_INVALID_RESPONSE",
                message="RAGFlow returned an unexpected response.",
                status_code=502,
            )
        code = payload.get("code", 0)
        if code not in (0, None):
            raise AppError(
                code="RAGFLOW_RETRIEVAL_FAILED",
                message="RAGFlow rejected the retrieval request.",
                status_code=502,
                details={"ragflow_code": code, "ragflow_message": payload.get("message")},
            )
        data = payload.get("data", payload)
        raw_chunks = data.get("chunks", []) if isinstance(data, dict) else []
        if not isinstance(raw_chunks, list):
            raise AppError(
                code="RAGFLOW_INVALID_RESPONSE",
                message="RAGFlow response does not contain a chunk list.",
                status_code=502,
            )
        return [
            normalized
            for item in raw_chunks
            if isinstance(item, dict) and (normalized := self._normalize_chunk(item)) is not None
        ]

    async def list_datasets(self) -> list[dict[str, str]]:
        """Return the available RAGFlow datasets as ``[{"id", "name"}]``.

        Used by the chat interface to let users choose which knowledge base
        to query. Kept intentionally light (id + name only).
        """
        try:
            response = await self._client.get("api/v1/datasets")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError(
                code="RAGFLOW_UNAVAILABLE",
                message="RAGFlow is unavailable or returned invalid JSON while listing datasets.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            items = data.get("docs", [])
        else:
            return []
        if not isinstance(items, list):
            return []
        result: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            dataset_id = item.get("id")
            name = item.get("name")
            if isinstance(dataset_id, str) and dataset_id and isinstance(name, str) and name:
                result.append({"id": dataset_id, "name": name})
        return result

    async def list_documents(
        self,
        dataset_id: str,
        page: int = 1,
        page_size: int = 1024,
    ) -> list[dict[str, str]]:
        """Return one page of documents inside a dataset as ``[{"id", "name"}]``.

        Used by enumeration/counting questions to build the answer boundary
        (the full document inventory) instead of trusting retrieval Top-N.
        Callers page through with ``page`` until fewer than ``page_size`` items
        are returned.
        """
        try:
            response = await self._client.get(
                f"api/v1/datasets/{dataset_id}/documents",
                params={"page": page, "page_size": page_size},
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError(
                code="RAGFLOW_UNAVAILABLE",
                message="RAGFlow is unavailable while listing documents.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc
        if not isinstance(payload, dict):
            return []
        data = payload.get("data")
        if isinstance(data, dict):
            items = data.get("docs", [])
        elif isinstance(data, list):
            items = data
        else:
            return []
        if not isinstance(items, list):
            return []
        result: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            document_id = item.get("id")
            name = item.get("name")
            if isinstance(document_id, str) and document_id and isinstance(name, str) and name:
                result.append({"id": document_id, "name": name})
        return result

    async def download_document(self, dataset_id: str, document_id: str) -> tuple[bytes, str]:
        """下载文档原始文件（PDF），返回 (content, content_type)。"""
        try:
            response = await self._client.get(
                f"api/v1/datasets/{dataset_id}/documents/{document_id}"
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AppError(
                code="RAGFLOW_UNAVAILABLE",
                message="RAGFlow 无法下载该文档原始文件。",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc
        content_type = response.headers.get("content-type", "application/pdf")
        return response.content, content_type

    @staticmethod
    def _normalize_chunk(item: dict[str, object]) -> RetrievedChunk | None:
        chunk_id = item.get("id") or item.get("chunk_id")
        document_id = item.get("document_id") or item.get("doc_id")
        dataset_id = item.get("dataset_id") or item.get("kb_id")
        text = item.get("content") or item.get("content_with_weight") or item.get("text")
        if not all(
            isinstance(value, str) and value
            for value in (
                chunk_id,
                document_id,
                dataset_id,
                text,
            )
        ):
            return None
        raw_metadata = item.get("document_metadata") or item.get("metadata") or {}
        metadata_values = dict(raw_metadata) if isinstance(raw_metadata, dict) else {}
        document_name = (
            metadata_values.get("document_name")
            or item.get("document_name")
            or item.get("document_keyword")
            or item.get("docnm_kwd")
        )
        metadata_values["document_name"] = document_name
        # RAGFlow 把页码放在 positions 里（[[page, x1, x2, y1, y2], ...]），
        # 而非 document_metadata；取首条位置的分页号作为该 chunk 所在页码。
        positions = item.get("positions")
        if isinstance(positions, list) and positions:
            first_position = positions[0]
            if (
                isinstance(first_position, (list, tuple))
                and first_position
                and isinstance(first_position[0], (int, float))
            ):
                if not metadata_values.get("page_number"):
                    metadata_values["page_number"] = int(first_position[0])
        if not metadata_values.get("page_number"):
            metadata_values["page_number"] = published_page_number(str(text))
        allowed_metadata = {
            field: metadata_values.get(field)
            for field in ChunkMetadata.model_fields
            if metadata_values.get(field) is not None
        }
        return RetrievedChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            dataset_id=dataset_id,
            text=text,
            metadata=ChunkMetadata.model_validate(allowed_metadata),
            hybrid_score=RagflowClient._numeric_score(item.get("similarity")),
            vector_score=RagflowClient._numeric_score(item.get("vector_similarity")),
            keyword_score=RagflowClient._numeric_score(item.get("term_similarity")),
        )

    @staticmethod
    def _numeric_score(value: object) -> float:
        if isinstance(value, int | float | str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return 0.0

    async def close(self) -> None:
        await self._client.aclose()
