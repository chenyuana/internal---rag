from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from app.core.config import RagflowSettings
from app.core.exceptions import AppError
from app.ingestion.regulations import (
    build_question_aliases,
    extract_keywords,
    match_article_heading,
)

SOURCE_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@dataclass(frozen=True, slots=True)
class RagflowChunkPlan:
    internal_chunk_id: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RagflowPlan:
    dataset_id: str
    source_name: str
    source_sha256: str
    chunks: tuple[RagflowChunkPlan, ...]
    plan_sha256: str


@dataclass(frozen=True, slots=True)
class PublishFacets:
    """The searchable fragments of one chunk exactly as RAGFlow receives them."""

    content: str
    important_keywords: list[str]
    questions: list[str]


def chunk_publish_facets(
    source_name: str,
    raw_chunk: Mapping[str, Any],
) -> PublishFacets:
    """Render ``content`` / ``important_keywords`` / ``questions`` for one chunk.

    This is the single source of truth for what leaves the gateway: the
    publisher uploads these fragments and the workbench shows these same
    fragments, so an operator reviewing "查看发送到 RAGFlow 的实际内容" can never
    be shown something different from the upload.
    """
    text = str(raw_chunk.get("text", "")).strip()
    section_path = [
        str(value).strip()
        for value in raw_chunk.get("section_path", [])
        if str(value).strip()
    ]
    title = str(raw_chunk.get("title", "")).strip()
    article_id = str(raw_chunk.get("article_id_normalized") or "").strip()
    article_aliases = [
        str(value).strip()
        for value in raw_chunk.get("article_aliases", [])
        if str(value).strip()
    ]
    chunk_keywords = extract_keywords(
        title,
        section_path,
        article_aliases,
        text=text,
        table_rows=raw_chunk.get("table_rows"),
    )
    stored_questions = [
        str(value).strip()
        for value in raw_chunk.get("question_aliases", [])
        if str(value).strip()
    ]
    questions = stored_questions or build_question_aliases(
        match_article_heading(title),
        title,
        text=text,
    )
    table_ids = [
        str(value).strip()
        for value in raw_chunk.get("table_ids", [])
        if str(value).strip()
    ]
    # Raw LaTeX is NOT put into tag_kwd: every formula would be a unique
    # high-cardinality tag and the backslashes/braces risk breaking tag
    # parsing. Instead it is appended to the content under a marked section.
    formula_latex = [
        str(value).strip()
        for value in raw_chunk.get("formula_latex", [])
        if str(value).strip() and str(value).strip().casefold() != "none"
    ]

    prefix_parts = [f"文档：{source_name}"]
    if section_path:
        prefix_parts.append("章节：" + " / ".join(section_path))
    if article_id:
        prefix_parts.append(f"条号：{article_id}")
    page_start = int(raw_chunk.get("page_start", 0))
    page_end = int(raw_chunk.get("page_end", page_start))
    if page_start > 0:
        page_label = (
            str(page_start) if page_end in (0, page_start) else f"{page_start}-{page_end}"
        )
        prefix_parts.append(f"页码：{page_label}")
    # Controlled glossary/structure anchors let Chinese questions rank an
    # English source chunk. They are explicitly marked as metadata, never a
    # replacement for the source evidence below.
    if chunk_keywords:
        prefix_parts.append("检索锚点（非规范译文）：" + "；".join(chunk_keywords))
    content = "\n".join(prefix_parts) + "\n\n" + text
    if formula_latex:
        content += "\n\n[原始公式]\n" + "\n".join(formula_latex)
    return PublishFacets(
        content=content,
        important_keywords=[
            item
            for item in dict.fromkeys(
                [article_id, *article_aliases, *chunk_keywords, *table_ids]
            )
            if item
        ],
        questions=list(dict.fromkeys(questions)),
    )


class RagflowPublisher:
    def __init__(
        self,
        settings: RagflowSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        api_key = settings.api_key.get_secret_value()
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=settings.publish_timeout_seconds,
            transport=transport,
            trust_env=False,
        )

    @property
    def has_api_key(self) -> bool:
        return bool(self._settings.api_key.get_secret_value())

    @staticmethod
    def _unwrap(payload: Any, *, operation: str) -> Any:
        if not isinstance(payload, dict):
            raise AppError(
                code="RAGFLOW_INVALID_RESPONSE",
                message=f"RAGFlow returned invalid JSON while attempting to {operation}.",
                status_code=502,
            )
        code = payload.get("code", 0)
        if code not in (0, None):
            raise AppError(
                code="RAGFLOW_PUBLISH_REJECTED",
                message=f"RAGFlow rejected the request to {operation}.",
                status_code=502,
                details={
                    "ragflow_code": code,
                    "ragflow_message": payload.get("message"),
                },
            )
        return payload.get("data")

    @staticmethod
    def build_plan(
        *,
        dataset_id: str,
        source_name: str,
        source_sha256: str,
        document_ir: dict[str, Any],
        asset_root: Path | None = None,
    ) -> RagflowPlan:
        planned: list[RagflowChunkPlan] = []
        asset_lookup = {
            str(asset.get("asset_id")): asset
            for asset in document_ir.get("assets", [])
            if isinstance(asset, dict) and asset.get("asset_id")
        }
        for raw_chunk in document_ir.get("chunks", []):
            if not isinstance(raw_chunk, dict):
                continue
            text = str(raw_chunk.get("text", "")).strip()
            chunk_id = str(raw_chunk.get("chunk_id", "")).strip()
            if not text or not chunk_id:
                continue
            article_id = str(raw_chunk.get("article_id_normalized") or "").strip()
            # Rebuild retrieval metadata at publication time (``chunk_publish_facets``).
            # This upgrades previously parsed IR too: old documents may contain a
            # repeated section heading split into generic important keywords, and
            # operator-authored question aliases in an existing IR are preserved.
            facets = chunk_publish_facets(source_name, raw_chunk)
            page_start = int(raw_chunk.get("page_start", 0))
            page_end = int(raw_chunk.get("page_end", page_start))
            table_html = [
                str(value).strip()
                for value in raw_chunk.get("table_html", [])
                if str(value).strip()
            ]
            table_ids = [
                str(value).strip()
                for value in raw_chunk.get("table_ids", [])
                if str(value).strip()
            ]
            # Table chunks already carry retrieval-oriented field/value text.
            # Re-appending the full HTML duplicates every cell, inflates the
            # embedding input, and can make malformed OCR geometry look more
            # trustworthy than it is. The lossless HTML stays in document_ir;
            # RAGFlow receives the semantic text plus table/image metadata.
            # Title and section path already exist in ``content``. Promoting
            # them to important keywords on every child chunk makes a repeated
            # appendix heading dominate chunk-specific terms and causes a
            # stable-but-wrong document-order ranking.
            tags = [
                f"source_sha256:{source_sha256}",
                f"internal_chunk_id:{chunk_id}",
                f"chunk_order:{len(planned):08d}",
            ]
            if page_start > 0:
                tags.append(f"page:{page_start}-{max(page_start, page_end)}")
            if article_id:
                tags.append(f"article_id:{article_id}")
            if table_html:
                tags.append("content_type:table")
            elif str(raw_chunk.get("content_type", "")).strip().casefold() == "figure":
                tags.append("content_type:figure")
            tags.extend(f"table_id:{table_id}" for table_id in table_ids)
            for related in raw_chunk.get("related_chunks", []):
                if isinstance(related, dict) and related.get("chunk_id"):
                    tags.append(f"related_chunk_id:{related['chunk_id']}")
            # Raw LaTeX is NOT put into tag_kwd: every formula would be a unique
            # high-cardinality tag and the backslashes/braces risk breaking tag
            # parsing. ``chunk_publish_facets`` keeps it in the content instead.
            if "[原始公式]" in facets.content:
                tags.append("has_formula")
            payload = {
                "content": facets.content,
                "important_keywords": facets.important_keywords,
                "questions": facets.questions,
                "tag_kwd": tags,
                "chunk_order": len(planned),
                "page_numbers": (
                    list(range(page_start, max(page_start, page_end) + 1))
                    if page_start > 0
                    else []
                ),
            }
            chunk_asset_ids = [
                str(value)
                for value in raw_chunk.get("asset_ids", [])
                if str(value)
            ]
            positions: list[list[int]] = []
            image_attached = False
            for asset_id in chunk_asset_ids:
                asset = asset_lookup.get(asset_id)
                if not isinstance(asset, dict):
                    continue
                bbox = asset.get("bbox")
                asset_page = int(asset.get("page_number", page_start or 1))
                if (
                    isinstance(bbox, list)
                    and len(bbox) >= 4
                    and all(isinstance(value, int | float) for value in bbox[:4])
                ):
                    positions.append(
                        [
                            asset_page,
                            round(float(bbox[0])),
                            round(float(bbox[2])),
                            round(float(bbox[1])),
                            round(float(bbox[3])),
                        ]
                    )
                if image_attached or asset_root is None:
                    continue
                relative_path = str(asset.get("relative_path", "")).strip()
                if not relative_path:
                    continue
                root = asset_root.resolve()
                asset_path = (root / relative_path).resolve()
                if not asset_path.is_relative_to(root) or not asset_path.is_file():
                    continue
                payload["image_base64"] = base64.b64encode(
                    asset_path.read_bytes()
                ).decode("ascii")
                image_attached = True
            if positions:
                payload["positions"] = positions
            planned.append(RagflowChunkPlan(internal_chunk_id=chunk_id, payload=payload))

        canonical = {
            "dataset_id": dataset_id,
            "source_name": source_name,
            "source_sha256": source_sha256,
            "chunks": [
                {
                    "internal_chunk_id": chunk.internal_chunk_id,
                    "payload": chunk.payload,
                }
                for chunk in planned
            ],
        }
        plan_sha256 = hashlib.sha256(
            json.dumps(
                canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return RagflowPlan(
            dataset_id=dataset_id,
            source_name=source_name,
            source_sha256=source_sha256,
            chunks=tuple(planned),
            plan_sha256=plan_sha256,
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        operation: str,
        **kwargs: Any,
    ) -> Any:
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError(
                code="RAGFLOW_UNAVAILABLE",
                message=f"RAGFlow is unavailable while attempting to {operation}.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc
        return self._unwrap(payload, operation=operation)

    async def upload_document(self, plan: RagflowPlan, source_path: Path) -> str:
        media_type = SOURCE_MEDIA_TYPES.get(
            Path(plan.source_name).suffix.casefold(),
            mimetypes.guess_type(plan.source_name)[0] or "application/octet-stream",
        )
        try:
            with source_path.open("rb") as source:
                data = await self._request_json(
                    "POST",
                    f"api/v1/datasets/{plan.dataset_id}/documents",
                    operation="upload the source document",
                    files={"file": (plan.source_name, source, media_type)},
                )
        except AppError:
            recovered_document_id = await self._recover_existing_document(plan)
            if recovered_document_id:
                return recovered_document_id
            raise
        documents = data if isinstance(data, list) else [data]
        for document in documents:
            if isinstance(document, dict):
                document_id = document.get("id")
                if isinstance(document_id, str):
                    return document_id
        raise AppError(
            code="RAGFLOW_INVALID_RESPONSE",
            message="RAGFlow upload response does not contain a document ID.",
            status_code=502,
        )

    async def _list_exact_documents(self, plan: RagflowPlan) -> list[dict[str, Any]]:
        matches: list[dict[str, Any]] = []
        page = 1
        while True:
            data = await self._request_json(
                "GET",
                f"api/v1/datasets/{plan.dataset_id}/documents",
                operation="recover an existing document after upload failure",
                params={
                    "page": page,
                    "page_size": 100,
                    "keywords": plan.source_name,
                },
            )
            if not isinstance(data, dict):
                break
            docs = data.get("docs", [])
            if not isinstance(docs, list):
                break
            matches.extend(
                doc
                for doc in docs
                if isinstance(doc, dict) and str(doc.get("name", "")) == plan.source_name
            )
            total = int(data.get("total", len(matches)))
            if page * 100 >= total or not docs:
                break
            page += 1
        return matches

    async def _existing_chunks(
        self,
        dataset_id: str,
        document_id: str,
    ) -> tuple[set[str], set[str], dict[str, str]]:
        contents: set[str] = set()
        source_hashes: set[str] = set()
        managed_chunk_ids: dict[str, str] = {}
        page = 1
        fetched = 0
        while True:
            data = await self._request_json(
                "GET",
                f"api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
                operation="list existing chunks",
                params={"page": page, "page_size": 100},
            )
            if not isinstance(data, dict):
                break
            raw_chunks = data.get("chunks", [])
            if not isinstance(raw_chunks, list):
                break
            fetched += len(raw_chunks)
            for chunk in raw_chunks:
                if not isinstance(chunk, dict):
                    continue
                if isinstance(chunk.get("content"), str):
                    contents.add(chunk["content"])
                tags = chunk.get("tag_kwd", [])
                if isinstance(tags, list):
                    source_hashes.update(
                        tag.removeprefix("source_sha256:")
                        for tag in tags
                        if isinstance(tag, str) and tag.startswith("source_sha256:")
                    )
                    chunk_id = chunk.get("id")
                    content = chunk.get("content")
                    if (
                        isinstance(chunk_id, str)
                        and isinstance(content, str)
                        and any(
                            isinstance(tag, str)
                            and tag.startswith("internal_chunk_id:")
                            for tag in tags
                        )
                    ):
                        managed_chunk_ids[content] = chunk_id
            total = int(data.get("total", len(contents)))
            if fetched >= total or not raw_chunks:
                break
            page += 1
        return contents, source_hashes, managed_chunk_ids

    async def _recover_existing_document(self, plan: RagflowPlan) -> str | None:
        candidates = await self._list_exact_documents(plan)
        empty_candidate: str | None = None
        for candidate in candidates:
            document_id = candidate.get("id")
            if not isinstance(document_id, str) or not document_id:
                continue
            contents, source_hashes, _ = await self._existing_chunks(
                plan.dataset_id,
                document_id,
            )
            if plan.source_sha256 in source_hashes:
                return document_id
            if not contents and empty_candidate is None:
                empty_candidate = document_id
        if empty_candidate:
            return empty_candidate
        if candidates:
            raise AppError(
                code="RAGFLOW_DOCUMENT_NAME_CONFLICT",
                message=(
                    "RAGFlow already contains a different document with the same "
                    f"name: {plan.source_name}"
                ),
                status_code=409,
                details={"dataset_id": plan.dataset_id},
            )
        return None

    async def publish(
        self,
        plan: RagflowPlan,
        source_path: Path,
        *,
        document_id: str | None = None,
        progress: Callable[[str, int, int], None] | None = None,
    ) -> tuple[str, int, int]:
        if not self.has_api_key:
            raise AppError(
                code="RAGFLOW_API_KEY_REQUIRED",
                message="A RAGFlow API key is required for live publication.",
                status_code=409,
            )
        resolved_document_id = document_id or await self.upload_document(plan, source_path)
        published = 0
        skipped = 0
        if progress:
            progress(resolved_document_id, published, skipped)
        existing, _, managed_chunk_ids = await self._existing_chunks(
            plan.dataset_id,
            resolved_document_id,
        )
        desired_contents = {
            str(chunk.payload["content"])
            for chunk in plan.chunks
        }
        stale_chunk_ids = [
            chunk_id
            for content, chunk_id in managed_chunk_ids.items()
            if content not in desired_contents
        ]
        for offset in range(0, len(stale_chunk_ids), 100):
            await self._request_json(
                "DELETE",
                (
                    f"api/v1/datasets/{plan.dataset_id}/documents/"
                    f"{resolved_document_id}/chunks"
                ),
                operation="remove stale published chunks",
                json={"chunk_ids": stale_chunk_ids[offset : offset + 100]},
            )
        if stale_chunk_ids:
            stale_ids = set(stale_chunk_ids)
            existing.difference_update(
                content
                for content, chunk_id in managed_chunk_ids.items()
                if chunk_id in stale_ids
            )
        pending_chunks: list[RagflowChunkPlan] = []
        queued_contents: set[str] = set()
        for chunk in plan.chunks:
            content = str(chunk.payload["content"])
            if content in existing or content in queued_contents:
                skipped += 1
                if progress:
                    progress(resolved_document_id, published, skipped)
                continue
            pending_chunks.append(chunk)
            queued_contents.add(content)

        batch_size = self._settings.publish_batch_size
        for offset in range(0, len(pending_chunks), batch_size):
            batch = pending_chunks[offset : offset + batch_size]
            await self._request_json(
                "POST",
                (
                    f"api/v1/datasets/{plan.dataset_id}/documents/"
                    f"{resolved_document_id}/chunks/batch"
                ),
                operation=(
                    f"publish chunk batch {offset // batch_size + 1} "
                    f"({len(batch)} chunks)"
                ),
                json={"chunks": [chunk.payload for chunk in batch]},
            )
            for chunk in batch:
                existing.add(str(chunk.payload["content"]))
                published += 1
                if progress:
                    progress(resolved_document_id, published, skipped)
        return resolved_document_id, published, skipped

    async def close(self) -> None:
        await self._client.aclose()
