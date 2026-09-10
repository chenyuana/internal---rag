from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shutil
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import UploadFile
from pydantic import SecretStr

from app.core.config import IngestionSettings, LlmStructureSettings, RagflowSettings
from app.core.exceptions import AppError
from app.ingestion.exporters import (
    build_batch_export_archive,
    build_export_archive,
    render_markdown,
)
from app.ingestion.llm_structure_store import module_store
from app.ingestion.parsers import ParserRegistry
from app.ingestion.parsers.docx import DOCX_MIME_TYPE, validate_docx_package
from app.ingestion.parsers.figure_vision import FigureVisionClient
from app.ingestion.parsers.hybrid_pdf import HybridPdfParser
from app.ingestion.parsers.llm_structure import LlmStructureAnnotator
from app.ingestion.parsers.native_pdf import NativePdfParser
from app.ingestion.parsers.remote import DoclingClient, MinerUClient
from app.ingestion.parsers.scan_regulatory import ScannedRegulatoryPdfParser
from app.ingestion.pipeline import write_document
from app.ingestion.publishers import (
    RagflowPlan,
    RagflowPublisher,
    chunk_publish_facets,
)
from app.schemas.ingestion import (
    IngestionChunkUpdateResult,
    IngestionJob,
    IngestionJobStatus,
    IngestionPageDetail,
    IngestionPreview,
    IngestionProgress,
    IngestionQualitySummary,
    IngestionQueueStatus,
    IngestionReview,
    ParserServiceStatus,
    PublicationStatus,
    RagflowPublication,
    RagflowPublishPlan,
    RagflowPublishResult,
    ReviewStatus,
)

logger = logging.getLogger(__name__)
SAFE_FILENAME_RE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


class IngestionJobService:
    def __init__(
        self,
        settings: IngestionSettings,
        parser_registry: ParserRegistry | None = None,
        ragflow_settings: RagflowSettings | None = None,
        ragflow_publisher: RagflowPublisher | None = None,
    ) -> None:
        self._settings = settings
        self._root = settings.root_dir.resolve()
        self._originals = self._root / "originals"
        self._jobs = self._root / "jobs"
        self._outputs = self._root / "outputs"
        self._temporary = self._root / "temporary"
        for folder in (self._originals, self._jobs, self._outputs, self._temporary):
            folder.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._parsers = parser_registry or ParserRegistry.with_builtins(settings)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._scheduled: set[str] = set()
        self._worker_task: asyncio.Task[None] | None = None
        self._active_job_id: str | None = None
        self._publisher = ragflow_publisher
        if self._publisher is None and ragflow_settings is not None:
            self._publisher = RagflowPublisher(ragflow_settings)

    @property
    def root_dir(self) -> Path:
        return self._root

    @staticmethod
    def _now() -> datetime:
        return datetime.now(UTC)

    async def start(self) -> None:
        if self._worker_task is not None and not self._worker_task.done():
            return
        for job in self.list_jobs(limit=200):
            if job.status not in {
                IngestionJobStatus.QUEUED,
                IngestionJobStatus.RUNNING,
            }:
                continue
            if job.status == IngestionJobStatus.RUNNING:
                job.status = IngestionJobStatus.QUEUED
                job.updated_at = self._now()
                self._write_job(job)
            await self.enqueue(job.job_id)
        self._worker_task = asyncio.create_task(
            self._worker_loop(),
            name="ingestion-serial-worker",
        )

    async def enqueue(self, job_id: str) -> None:
        if job_id == self._active_job_id or job_id in self._scheduled:
            return
        self._scheduled.add(job_id)
        await self._queue.put(job_id)

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._queue.get()
            self._scheduled.discard(job_id)
            self._active_job_id = job_id
            try:
                await self.process_job(job_id)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception(
                    "ingestion_worker_unhandled_error",
                    extra={"job_id": job_id},
                )
            finally:
                self._active_job_id = None
                self._queue.task_done()

    def queue_status(self) -> IngestionQueueStatus:
        return IngestionQueueStatus(
            worker_concurrency=self._settings.worker_concurrency,
            active_job_id=self._active_job_id,
            queued_job_ids=sorted(self._scheduled),
        )

    @staticmethod
    def _safe_filename(filename: str | None) -> str:
        candidate = Path(filename or "upload.bin").name
        candidate = SAFE_FILENAME_RE.sub("_", candidate).strip(" .")
        return candidate[:180] or "upload.bin"

    @staticmethod
    def _safe_relative_path(relative_path: str | None) -> str | None:
        if not relative_path:
            return None
        parts: list[str] = []
        for raw_part in relative_path.replace("\\", "/").split("/"):
            part = raw_part.strip()
            if not part or part == ".":
                continue
            if part == "..":
                return None
            safe_part = SAFE_FILENAME_RE.sub("_", part).strip(" .")
            if safe_part:
                parts.append(safe_part[:180])
        normalized = "/".join(parts)
        return normalized[:1000] or None

    def _job_path(self, job_id: str) -> Path:
        return self._jobs / job_id / "job.json"

    def _write_job(self, job: IngestionJob) -> None:
        path = self._job_path(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            job.model_dump_json(indent=2),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _read_job(self, job_id: str) -> IngestionJob:
        path = self._job_path(job_id)
        if not path.is_file():
            raise AppError(
                code="INGESTION_JOB_NOT_FOUND",
                message="Ingestion job does not exist.",
                status_code=404,
                details={"job_id": job_id},
            )
        return IngestionJob.model_validate_json(path.read_text(encoding="utf-8"))

    async def create_job(
        self,
        upload: UploadFile,
        *,
        knowledge_base_id: str | None,
        source_relative_path: str | None = None,
        batch_id: str | None = None,
        sequence_in_batch: int | None = None,
        created_by: str,
        llm_structure_enabled: bool = False,
    ) -> IngestionJob:
        if not self._settings.enabled:
            raise AppError(
                code="INGESTION_DISABLED",
                message="Document ingestion is disabled.",
                status_code=503,
            )
        source_name = self._safe_filename(upload.filename)
        extension = Path(source_name).suffix.casefold()
        allowed = {item.casefold() for item in self._settings.allowed_extensions}
        if extension not in allowed:
            raise AppError(
                code="UNSUPPORTED_DOCUMENT_TYPE",
                message="This document type is not enabled for ingestion.",
                status_code=415,
                details={"extension": extension, "allowed_extensions": sorted(allowed)},
            )

        temporary_path = self._temporary / f"{uuid4().hex}.part"
        digest = hashlib.sha256()
        size_bytes = 0
        try:
            with temporary_path.open("wb") as stream:
                while content := await upload.read(1024 * 1024):
                    size_bytes += len(content)
                    if size_bytes > self._settings.max_upload_bytes:
                        raise AppError(
                            code="UPLOAD_TOO_LARGE",
                            message="Uploaded document exceeds the configured size limit.",
                            status_code=413,
                            details={
                                "max_upload_bytes": self._settings.max_upload_bytes,
                            },
                        )
                    digest.update(content)
                    stream.write(content)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        if size_bytes == 0:
            temporary_path.unlink(missing_ok=True)
            raise AppError(
                code="EMPTY_UPLOAD",
                message="Uploaded document is empty.",
                status_code=400,
            )
        with temporary_path.open("rb") as stream:
            signature = stream.read(5)
        if extension == ".pdf" and signature != b"%PDF-":
            temporary_path.unlink(missing_ok=True)
            raise AppError(
                code="DOCUMENT_SIGNATURE_MISMATCH",
                message="The uploaded file extension does not match its content.",
                status_code=415,
                details={"extension": extension},
            )
        if extension == ".docx":
            try:
                validate_docx_package(temporary_path)
            except AppError:
                temporary_path.unlink(missing_ok=True)
                raise

        source_sha256 = digest.hexdigest()
        original_folder = self._originals / source_sha256[:2] / source_sha256
        original_folder.mkdir(parents=True, exist_ok=True)
        original_path = original_folder / f"source{extension}"
        if original_path.exists():
            temporary_path.unlink(missing_ok=True)
        else:
            temporary_path.replace(original_path)

        now = self._now()
        job = IngestionJob(
            job_id=uuid4().hex,
            status=IngestionJobStatus.QUEUED,
            source_name=source_name,
            source_relative_path=self._safe_relative_path(source_relative_path),
            batch_id=(batch_id or "").strip()[:100] or None,
            sequence_in_batch=sequence_in_batch,
            source_sha256=source_sha256,
            media_type=(
                "application/pdf"
                if extension == ".pdf"
                else DOCX_MIME_TYPE
                if extension == ".docx"
                else upload.content_type or "application/octet-stream"
            ),
            size_bytes=size_bytes,
            knowledge_base_id=knowledge_base_id,
            created_by=created_by,
            original_path=str(original_path),
            llm_structure_enabled=llm_structure_enabled,
            progress=IngestionProgress(),
            review=IngestionReview(updated_at=now),
            created_at=now,
            updated_at=now,
        )
        self._write_job(job)
        return job

    def _parse_with_structure(
        self,
        source_path: Path,
        job: IngestionJob,
        progress_callback: Callable[[int, int], None] | None,
    ) -> object:
        """Run the selected parser, optionally enabling LLM structure annotation.

        The ingestion console toggle decides whether this job re-annotates
        chunk metadata (section_path/article_id/title/keywords) with the LLM.
        Model endpoint / API key / model name come from the console settings
        (process memory, per operator) and fall back to the ``llm_structure``
        env config; on any error the heuristic metadata is kept.
        """
        settings = self._resolve_structure_settings(job.created_by)
        annotator: LlmStructureAnnotator | None = None
        if job.llm_structure_enabled and settings is not None:
            annotator = LlmStructureAnnotator(settings)
            self._parsers.set_structure_annotator(annotator)
        try:
            document = self._parsers.parse(source_path, progress_callback=progress_callback)
        finally:
            if annotator is not None:
                self._parsers.clear_structure_annotator()
                annotator.close()
        return document

    def _resolve_structure_settings(self, user_id: str) -> LlmStructureSettings | None:
        """Return LLM structure settings for this operator, or ``None`` to skip."""
        store_cfg = module_store.get(user_id)
        if store_cfg is not None and store_cfg.api_key:
            return LlmStructureSettings(
                base_url=store_cfg.base_url,
                model=store_cfg.model,
                api_key=SecretStr(store_cfg.api_key),
                max_tokens=store_cfg.max_tokens,
                page_batch_size=store_cfg.page_batch_size,
                timeout_seconds=store_cfg.timeout_seconds,
            )
        if self._settings.llm_structure.api_key.get_secret_value():
            return self._settings.llm_structure
        return None

    async def process_job(self, job_id: str) -> IngestionJob:
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self._read_job(job_id)
            if job.status == IngestionJobStatus.RUNNING:
                return job
            job.status = IngestionJobStatus.RUNNING
            job.attempts += 1
            job.updated_at = self._now()
            job.error_code = None
            job.error_message = None
            job.progress = IngestionProgress(
                stage="preparing",
                percent=5,
                message="正在准备原文件与解析环境",
            )
            self._write_job(job)
            try:
                source_path = Path(job.original_path)

                def _report_ocr_progress(done: int, total: int) -> None:
                    # Called from the parser worker thread after each MinerU
                    # batch. Update the persisted job so the UI shows live OCR
                    # progress instead of sitting at 15% for the whole run.
                    nonlocal job
                    percent = 15 + int(67 * done / total) if total else 15
                    job.progress = IngestionProgress(
                        stage="parsing",
                        percent=min(percent, 81),
                        message=(
                            f"页面处理中：{done}/{total} 页"
                            if done < total
                            else f"页面处理完成：{total} 页，正在合并"
                        ),
                        current=done,
                        total=total,
                        indeterminate=False,
                    )
                    job.updated_at = self._now()
                    self._write_job(job)

                job.progress = IngestionProgress(
                    stage="parsing",
                    percent=15,
                    message="解析器正在识别页面、清洗内容并生成分块",
                    indeterminate=True,
                )
                job.updated_at = self._now()
                self._write_job(job)
                if job.page_reprocess:
                    previous_ir = self._load_document_ir(job)
                    previous_parser = previous_ir.get("parser", {}).get("name")
                    if previous_parser == "scan-regulatory-pdf":
                        page_parser = ScannedRegulatoryPdfParser(
                            MinerUClient(self._settings.mineru)
                        )
                    elif previous_parser == "hybrid-pdf":
                        page_parser = HybridPdfParser(
                            NativePdfParser(),
                            MinerUClient(self._settings.mineru),
                            DoclingClient(self._settings.docling),
                            FigureVisionClient(self._settings.figure_vlm),
                        )
                    else:
                        raise ValueError(
                            f"Parser {previous_parser!r} does not support page reprocessing"
                        )
                    document = await asyncio.to_thread(
                        page_parser.reprocess_page,
                        source_path,
                        previous_ir,
                        Path(job.output_path or ""),
                        job.page_reprocess["page_number"],
                        job.page_reprocess["mode"],
                        progress_callback=_report_ocr_progress,
                    )
                    document.qa.setdefault("page_reprocess", {})["previous_output_path"] = (
                        job.output_path
                    )
                else:
                    document = await asyncio.to_thread(
                        self._parse_with_structure,
                        source_path,
                        job,
                        _report_ocr_progress,
                    )
                page_count = int(document.qa.get("page_count", len(document.pages)))
                job.progress = IngestionProgress(
                    stage="writing",
                    percent=82,
                    message="解析完成，正在写入统一文档结构",
                    current=page_count,
                    total=page_count,
                )
                job.updated_at = self._now()
                self._write_job(job)
                output_root = self._outputs / job.job_id
                if job.page_reprocess:
                    # Write a new revision completely before switching the active pointer.
                    output_root = output_root / "revisions" / uuid4().hex
                output = await asyncio.to_thread(
                    write_document,
                    output_root,
                    document,
                    display_name=job.source_name,
                )
                job.progress = IngestionProgress(
                    stage="quality_check",
                    percent=94,
                    message="正在执行文本覆盖率与质量门禁检查",
                    current=page_count,
                    total=page_count,
                )
                job.updated_at = self._now()
                self._write_job(job)
                quality = document.qa
                gates = quality.get("quality_gates", [])
                failures = sum(item.get("status") == "fail" for item in gates)
                warnings = sum(item.get("status") == "warn" for item in gates)
                if failures:
                    status = IngestionJobStatus.NEEDS_REVIEW
                elif warnings:
                    status = IngestionJobStatus.WARNING
                else:
                    status = IngestionJobStatus.COMPLETED
                job.status = status
                if job.page_reprocess:
                    job.review = IngestionReview(updated_at=self._now())
                    job.publication = RagflowPublication(updated_at=self._now())
                job.output_path = str(output["output_folder"])
                job.quality = IngestionQualitySummary(
                    route=document.route,
                    page_count=int(quality.get("page_count", 0)),
                    chunk_count=int(quality.get("chunk_count", 0)),
                    text_coverage=float(quality.get("text_coverage", 0)),
                    toc_page_count=len(quality.get("toc_pages", [])),
                    warning_count=warnings,
                    failure_count=failures,
                )
                progress_message = {
                    IngestionJobStatus.COMPLETED: "解析与质量检查已完成",
                    IngestionJobStatus.WARNING: "解析完成，请抽检警告项",
                    IngestionJobStatus.NEEDS_REVIEW: "解析完成，质量门禁要求人工复核",
                }[status]
                job.progress = IngestionProgress(
                    stage=status.value,
                    percent=100,
                    message=progress_message,
                    current=job.quality.page_count,
                    total=job.quality.page_count,
                )
            except AppError as exc:
                job.status = IngestionJobStatus.FAILED
                job.error_code = exc.code
                job.error_message = exc.message
                job.progress = IngestionProgress(
                    stage="failed",
                    percent=job.progress.percent,
                    message=exc.message,
                )
            except Exception as exc:
                logger.exception(
                    "ingestion_job_failed",
                    extra={"job_id": job_id, "error_type": type(exc).__name__},
                )
                job.status = IngestionJobStatus.FAILED
                job.error_code = "INGESTION_PROCESSING_FAILED"
                job.error_message = str(exc)
                job.progress = IngestionProgress(
                    stage="failed",
                    percent=job.progress.percent,
                    message=str(exc),
                )
            if job.page_reprocess:
                if job.status == IngestionJobStatus.FAILED:
                    previous = IngestionJob.model_validate(job.page_reprocess["previous_job"])
                    previous.error_code = job.error_code
                    previous.error_message = "单页处理失败，旧结果已保留：" + (
                        job.error_message or ""
                    )
                    previous.progress.message = previous.error_message
                    job = previous
                job.page_reprocess = None
            job.updated_at = self._now()
            self._write_job(job)
            return job

    def get_job(self, job_id: str) -> IngestionJob:
        return self._read_job(job_id)

    def list_jobs(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        query: str | None = None,
        review_status: ReviewStatus | None = None,
    ) -> list[IngestionJob]:
        normalized_query = (query or "").strip().casefold()
        jobs: list[IngestionJob] = []
        for path in self._jobs.glob("*/job.json"):
            job = IngestionJob.model_validate_json(path.read_text(encoding="utf-8"))
            if review_status is not None and job.review.status != review_status:
                continue
            if normalized_query:
                searchable = "\n".join(
                    str(value or "")
                    for value in (
                        job.source_name,
                        job.source_relative_path,
                        job.job_id,
                        job.source_sha256,
                        job.knowledge_base_id,
                        job.status.value,
                        job.publication.status.value,
                        job.review.status.value,
                        job.review.reviewer,
                        job.review.note,
                        job.error_code,
                        job.error_message,
                    )
                ).casefold()
                if normalized_query not in searchable:
                    continue
            jobs.append(job)
        jobs.sort(
            key=lambda job: (
                job.created_at,
                job.sequence_in_batch if job.sequence_in_batch is not None else 2**31,
                (job.source_relative_path or job.source_name).casefold(),
                job.job_id,
            )
        )
        return jobs[offset : offset + limit]

    async def reprocess_page(self, job_id: str, page_number: int, mode: str) -> IngestionJob:
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self._read_job(job_id)

            def reject(message: str) -> None:
                raise AppError(
                    code="INGESTION_PAGE_REPROCESS_BLOCKED", message=message, status_code=409
                )

            if job.status not in {
                IngestionJobStatus.COMPLETED,
                IngestionJobStatus.WARNING,
                IngestionJobStatus.NEEDS_REVIEW,
            }:
                reject("请等待任务完成后再进行单页处理。")
            if job.publication.document_id or job.publication.status in {
                PublicationStatus.PUBLISHING,
                PublicationStatus.PUBLISHED,
                PublicationStatus.FAILED,
            }:
                reject("已发布或部分发布的任务暂不支持单页处理，避免影响知识库中的旧版本。")
            ir = self._load_document_ir(job)
            if ir.get("parser", {}).get("name") not in {
                "scan-regulatory-pdf",
                "hybrid-pdf",
            }:
                reject("此解析类型暂不支持单页处理。")
            if any(c.get("manual_revision") for c in ir.get("chunks", [])) or any(
                p.get("manual_edited") for p in ir.get("pages", [])
            ):
                reject("文档包含人工修订；重建 Chunk 可能产生冲突，已保留原结果并取消操作。")
            if mode not in {"clean", "ocr"} or not any(
                p.get("page_number") == page_number for p in ir.get("pages", [])
            ):
                raise AppError(
                    code="INGESTION_PAGE_INVALID", message="页码或处理方式无效。", status_code=422
                )
            job.page_reprocess = {
                "page_number": page_number,
                "mode": mode,
                "previous_job": job.model_dump(mode="json"),
            }
            job.status = IngestionJobStatus.QUEUED
            job.progress = IngestionProgress(message=f"等待处理第 {page_number} 页；其他页复用缓存")
            job.updated_at = self._now()
            self._write_job(job)
            return job

    async def retry_job(self, job_id: str) -> IngestionJob:
        job = self._read_job(job_id)
        if job.status == IngestionJobStatus.RUNNING or job.page_reprocess:
            raise AppError(
                code="INGESTION_JOB_RUNNING",
                message="A running ingestion job cannot be retried.",
                status_code=409,
            )
        now = self._now()
        previous_dataset_id = job.publication.dataset_id or job.knowledge_base_id
        job.status = IngestionJobStatus.QUEUED
        job.progress = IngestionProgress(
            stage="queued",
            percent=0,
            message="等待重新进入处理队列",
        )
        job.review = IngestionReview(
            status=ReviewStatus.UNREVIEWED,
            note="资料已重新处理，需要重新审核。",
            updated_at=now,
        )
        job.publication = RagflowPublication(
            status=PublicationStatus.NOT_REQUESTED,
            stage="not_requested",
            percent=0,
            message="资料已重新处理，当前结果尚未发送到 RAGFlow",
            dataset_id=previous_dataset_id,
            updated_at=now,
        )
        job.updated_at = now
        self._write_job(job)
        return job

    async def update_review(
        self,
        job_id: str,
        *,
        status: ReviewStatus,
        note: str | None,
        reviewer: str,
        quality_gate_override: bool = False,
    ) -> IngestionJob:
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self._read_job(job_id)
            if status == ReviewStatus.REWORK and not note:
                raise AppError(
                    code="INGESTION_REWORK_NOTE_REQUIRED",
                    message="标记为需返工时必须填写具体问题。",
                    status_code=422,
                )
            if job.status in {IngestionJobStatus.QUEUED, IngestionJobStatus.RUNNING}:
                raise AppError(
                    code="INGESTION_REVIEW_NOT_READY",
                    message="资料仍在处理队列中，暂不能提交人工审核结果。",
                    status_code=409,
                )
            if job.publication.status in {
                PublicationStatus.PUBLISHING,
                PublicationStatus.PUBLISHED,
            }:
                raise AppError(
                    code="INGESTION_REVIEW_LOCKED",
                    message="正在发送或已经发送的资料不能修改审核结论。",
                    status_code=409,
                )
            if quality_gate_override and status != ReviewStatus.APPROVED:
                raise AppError(
                    code="INGESTION_REVIEW_OVERRIDE_INVALID",
                    message="人工质量门禁放行只能与审核通过同时提交。",
                    status_code=422,
                )

            overridden_quality_gates: list[dict[str, Any]] = []
            if status == ReviewStatus.APPROVED:
                if job.status == IngestionJobStatus.NEEDS_REVIEW:
                    if not quality_gate_override:
                        raise AppError(
                            code="INGESTION_REVIEW_QUALITY_BLOCKED",
                            message=(
                                "质量门禁尚未通过；请返工后重新处理，或使用“人工确认无误”"
                                "显式放行并保留审核记录。"
                            ),
                            status_code=409,
                            details={"status": job.status.value},
                        )
                    document_ir = self._load_document_ir(job)
                    qa = document_ir.get("qa", {})
                    gates = qa.get("quality_gates", []) if isinstance(qa, dict) else []
                    overridden_quality_gates = [
                        dict(gate)
                        for gate in gates
                        if isinstance(gate, dict) and gate.get("status") == "fail"
                    ]
                    if not overridden_quality_gates:
                        raise AppError(
                            code="INGESTION_REVIEW_OVERRIDE_UNAVAILABLE",
                            message="当前任务没有可人工放行的失败质量门禁。",
                            status_code=409,
                        )
                elif job.status not in {
                    IngestionJobStatus.COMPLETED,
                    IngestionJobStatus.WARNING,
                }:
                    raise AppError(
                        code="INGESTION_REVIEW_QUALITY_BLOCKED",
                        message="当前任务状态不能标记为审核通过。",
                        status_code=409,
                        details={"status": job.status.value},
                    )

            now = self._now()
            previous = job.review
            started_at = previous.started_at
            reviewed_at = previous.reviewed_at
            if status == ReviewStatus.IN_REVIEW and started_at is None:
                started_at = now
            if status in {ReviewStatus.APPROVED, ReviewStatus.REWORK}:
                started_at = started_at or now
                reviewed_at = now
            elif status == ReviewStatus.UNREVIEWED:
                started_at = None
                reviewed_at = None
            job.review = IngestionReview(
                status=status,
                note=note,
                reviewer=reviewer,
                started_at=started_at,
                reviewed_at=reviewed_at,
                quality_gate_override=bool(overridden_quality_gates),
                overridden_quality_gates=overridden_quality_gates,
                quality_gate_overridden_at=(now if overridden_quality_gates else None),
                updated_at=now,
            )
            job.updated_at = now
            self._write_job(job)
            audit_path = self._job_path(job_id).parent / "review_history.jsonl"
            with audit_path.open("a", encoding="utf-8") as audit:
                audit.write(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "from_status": previous.status.value,
                            "to_status": status.value,
                            "note": note,
                            "reviewer": reviewer,
                            "quality_gate_override": bool(overridden_quality_gates),
                            "overridden_quality_gates": overridden_quality_gates,
                            "updated_at": now.isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            return job

    @staticmethod
    def _validated_child_path(path: Path, root: Path, *, label: str) -> Path:
        resolved = path.resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise AppError(
                code="INGESTION_DELETE_PATH_INVALID",
                message=f"The {label} path is outside the ingestion storage root.",
                status_code=500,
            )
        return resolved

    def _original_is_shared(self, job_id: str, original_path: Path) -> bool:
        for path in self._jobs.glob("*/job.json"):
            if path.parent.name == job_id:
                continue
            try:
                other = IngestionJob.model_validate_json(
                    path.read_text(encoding="utf-8"),
                )
            except (OSError, ValueError):
                logger.warning(
                    "ingestion_job_reference_check_skipped",
                    extra={"job_path": str(path)},
                )
                continue
            if Path(other.original_path).resolve() == original_path:
                return True
        return False

    async def delete_job(self, job_id: str) -> None:
        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self._read_job(job_id)
            if (
                job_id == self._active_job_id
                or job_id in self._scheduled
                or job.status
                in {
                    IngestionJobStatus.QUEUED,
                    IngestionJobStatus.RUNNING,
                }
            ):
                raise AppError(
                    code="INGESTION_JOB_BUSY",
                    message="正在处理或等待处理的资料不能删除，请稍后重试。",
                    status_code=409,
                )
            if job.publication.status == PublicationStatus.PUBLISHING:
                raise AppError(
                    code="INGESTION_JOB_BUSY",
                    message="正在发送到 RAGFlow 的资料不能删除，请等待发送完成。",
                    status_code=409,
                )

            job_folder = self._validated_child_path(
                self._job_path(job_id).parent,
                self._jobs,
                label="job",
            )
            output_folder = self._validated_child_path(
                self._outputs / job_id,
                self._outputs,
                label="output",
            )
            original_path = self._validated_child_path(
                Path(job.original_path),
                self._originals,
                label="original",
            )
            original_is_shared = self._original_is_shared(job_id, original_path)

            try:
                if output_folder.exists():
                    shutil.rmtree(output_folder)
                shutil.rmtree(job_folder)
                if not original_is_shared:
                    original_path.unlink(missing_ok=True)
                    for parent in (original_path.parent, original_path.parent.parent):
                        try:
                            parent.rmdir()
                        except OSError:
                            break
            except OSError as exc:
                raise AppError(
                    code="INGESTION_DELETE_FAILED",
                    message="删除接入任务的本地文件失败。",
                    status_code=500,
                    details={"job_id": job_id, "reason": str(exc)},
                ) from exc
        self._locks.pop(job_id, None)

    def _publish_views(
        self,
        source_name: str,
        chunks: Iterable[Any],
    ) -> list[dict[str, Any]]:
        """Attach the exact fragments RAGFlow will receive to each chunk.

        The workbench renders these instead of re-deriving the prefix, so the
        "查看发送到 RAGFlow 的实际内容" panel cannot drift from the upload.
        """
        views: list[dict[str, Any]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            facets = chunk_publish_facets(source_name, chunk)
            views.append(
                {
                    **chunk,
                    "ragflow_content": facets.content,
                    "ragflow_important_keywords": facets.important_keywords,
                }
            )
        return views

    def preview(self, job_id: str, *, chunk_limit: int | None = None) -> IngestionPreview:
        job = self._read_job(job_id)
        if not job.output_path:
            raise AppError(
                code="INGESTION_PREVIEW_NOT_READY",
                message="The ingestion preview is not available yet.",
                status_code=409,
                details={"status": job.status.value},
            )
        document_ir_path = Path(job.output_path) / "document_ir.json"
        if not document_ir_path.is_file():
            raise AppError(
                code="INGESTION_OUTPUT_MISSING",
                message="The ingestion output is incomplete.",
                status_code=500,
            )
        payload: dict[str, Any] = json.loads(document_ir_path.read_text(encoding="utf-8"))
        pages = payload.get("pages", [])
        page_routes = Counter(str(page.get("route", "unknown")) for page in pages)
        page_summaries = [
            {
                key: page.get(key)
                for key in (
                    "page_number",
                    "route",
                    "raw_char_count",
                    "cleaned_char_count",
                    "table_hints",
                    "is_toc",
                    "indexable",
                    "block_count",
                    "chunk_count",
                    "article_ids",
                    "rich_block_count",
                    "asset_ids",
                    "page_type",
                    "image_count",
                    "image_coverage",
                    "layout_features",
                    "manual_edit_count",
                    "manual_edited",
                )
            }
            for page in pages
            if isinstance(page, dict)
        ]
        limit = chunk_limit or self._settings.preview_chunk_limit
        return IngestionPreview(
            job_id=job.job_id,
            status=job.status,
            source_name=job.source_name,
            route=payload.get("route"),
            quality=payload.get("qa", {}),
            page_routes=dict(page_routes),
            pages=page_summaries,
            chunks=self._publish_views(job.source_name, payload.get("chunks", [])[:limit]),
        )

    def page_detail(self, job_id: str, page_number: int) -> IngestionPageDetail:
        job = self._read_job(job_id)
        payload = self._load_document_ir(job)
        pages = payload.get("pages", [])
        page = next(
            (
                item
                for item in pages
                if isinstance(item, dict) and item.get("page_number") == page_number
            ),
            None,
        )
        if page is None:
            raise AppError(
                code="INGESTION_PAGE_NOT_FOUND",
                message="The requested parsed page does not exist.",
                status_code=404,
                details={"job_id": job_id, "page_number": page_number},
            )
        raw_text = ""
        baseline_path = Path(job.output_path or "") / "baseline.json"
        if baseline_path.is_file():
            baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
            baseline_pages = baseline.get("pages", []) if isinstance(baseline, dict) else []
            raw_page = next(
                (
                    item
                    for item in baseline_pages
                    if isinstance(item, dict) and item.get("page_number") == page_number
                ),
                None,
            )
            if raw_page is not None:
                raw_text = str(raw_page.get("raw_text", ""))
        detail_page = {**page, "raw_text": raw_text}
        blocks = [
            item
            for item in payload.get("blocks", [])
            if isinstance(item, dict)
            and int(item.get("source_page_start") or item.get("page_number") or 0)
            <= page_number
            <= int(item.get("source_page_end") or item.get("page_number") or 0)
        ]
        chunks = [
            item
            for item in payload.get("chunks", [])
            if isinstance(item, dict)
            and int(item.get("page_start", 0)) <= page_number <= int(item.get("page_end", 0))
        ]
        asset_version = int(job.updated_at.timestamp() * 1_000_000)
        assets = [
            {
                **item,
                "url": (
                    f"/api/v1/ingestion/jobs/{job.job_id}/assets/"
                    f"{item.get('asset_id', '')}?v={asset_version}"
                ),
            }
            for item in payload.get("assets", [])
            if isinstance(item, dict) and item.get("page_number") == page_number
        ]
        parser = payload.get("parser", {})
        trace = parser.get("trace", []) if isinstance(parser, dict) else []
        quality = payload.get("qa", {})
        gates = quality.get("quality_gates", []) if isinstance(quality, dict) else []
        parser_name = parser.get("name") if isinstance(parser, dict) else None
        block_reason = ""
        if parser_name not in {"scan-regulatory-pdf", "hybrid-pdf"}:
            block_reason = "此解析类型暂不支持单页处理。"
        elif job.status not in {
            IngestionJobStatus.COMPLETED,
            IngestionJobStatus.WARNING,
            IngestionJobStatus.NEEDS_REVIEW,
        }:
            block_reason = "任务正在排队或处理中，请等待完成。"
        elif job.publication.document_id or job.publication.status in {
            PublicationStatus.PUBLISHING,
            PublicationStatus.PUBLISHED,
            PublicationStatus.FAILED,
        }:
            block_reason = "已发布或部分发布的任务暂不支持单页处理，以保护知识库旧版本。"
        elif any(chunk.get("manual_revision") for chunk in payload.get("chunks", [])) or any(
            item.get("manual_edited") for item in payload.get("pages", [])
        ):
            block_reason = "文档包含人工修订；重建 Chunk 可能产生冲突，已保留原结果并取消操作。"
        return IngestionPageDetail(
            job_id=job.job_id,
            source_name=job.source_name,
            page_number=page_number,
            page=detail_page,
            blocks=blocks,
            chunks=self._publish_views(job.source_name, chunks),
            assets=assets,
            parser_trace=trace,
            quality_gates=gates,
            page_reprocess={"supported": not block_reason, "reason": block_reason},
            source_url=f"/api/v1/ingestion/jobs/{job.job_id}/source#page={page_number}",
        )

    @staticmethod
    def _write_text_atomic(path: Path, content: str) -> None:
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def _write_json_atomic(cls, path: Path, payload: Any) -> None:
        cls._write_text_atomic(
            path,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

    async def update_chunk(
        self,
        job_id: str,
        chunk_id: str,
        *,
        text: str,
        reason: str | None,
        updated_by: str,
    ) -> IngestionChunkUpdateResult:
        """Persist a pre-publication manual Chunk revision with an audit trail.

        The parsed source layer and original PDF remain immutable.  RAGFlow publishes
        from document_ir.json, so changing that Chunk is the smallest safe edit unit
        and mirrors DeepDoc's Chunk editor behavior.
        """

        lock = self._locks.setdefault(job_id, asyncio.Lock())
        async with lock:
            job = self._read_job(job_id)
            if (
                job_id == self._active_job_id
                or job_id in self._scheduled
                or job.status
                in {
                    IngestionJobStatus.QUEUED,
                    IngestionJobStatus.RUNNING,
                }
                or job.publication.status == PublicationStatus.PUBLISHING
            ):
                raise AppError(
                    code="INGESTION_CHUNK_EDIT_BUSY",
                    message="正在处理或发送中的资料不能修改，请等待任务完成。",
                    status_code=409,
                )
            if job.publication.status == PublicationStatus.PUBLISHED:
                raise AppError(
                    code="INGESTION_CHUNK_ALREADY_PUBLISHED",
                    message=(
                        "该版本已经发送到 RAGFlow。为避免本地内容与知识库不一致，"
                        "请在 RAGFlow 中编辑 Chunk，或重新处理后生成新版本。"
                    ),
                    status_code=409,
                )
            if job.status not in {
                IngestionJobStatus.COMPLETED,
                IngestionJobStatus.WARNING,
                IngestionJobStatus.NEEDS_REVIEW,
            }:
                raise AppError(
                    code="INGESTION_CHUNK_EDIT_NOT_READY",
                    message="当前任务还没有可人工修订的解析结果。",
                    status_code=409,
                    details={"status": job.status.value},
                )

            document_ir = self._load_document_ir(job)
            chunks = document_ir.get("chunks", [])
            chunk = next(
                (
                    item
                    for item in chunks
                    if isinstance(item, dict) and item.get("chunk_id") == chunk_id
                ),
                None,
            )
            if chunk is None:
                raise AppError(
                    code="INGESTION_CHUNK_NOT_FOUND",
                    message="指定的 Chunk 不存在。",
                    status_code=404,
                    details={"job_id": job_id, "chunk_id": chunk_id},
                )

            now = self._now()
            previous_text = str(chunk.get("text", ""))
            previous_revision = chunk.get("manual_revision", {})
            if not isinstance(previous_revision, dict):
                previous_revision = {}
            revision = int(previous_revision.get("revision", 0)) + 1
            had_structured_table = bool(
                chunk.get("table_html")
                or chunk.get("table_rows")
                or chunk.get("content_type") == "table"
            )
            previous_hash = hashlib.sha256(previous_text.encode("utf-8")).hexdigest()
            current_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            chunk["text"] = text
            chunk["manual_revision"] = {
                "revision": revision,
                "updated_at": now.isoformat(),
                "updated_by": updated_by,
                "reason": reason,
                "original_text_sha256": previous_revision.get(
                    "original_text_sha256",
                    previous_hash,
                ),
                "previous_text_sha256": previous_hash,
                "current_text_sha256": current_hash,
                "replaced_structured_table": had_structured_table,
            }
            if had_structured_table:
                # Do not append stale extracted HTML after an operator has replaced
                # the table representation.  Blocks/assets retain the source evidence.
                chunk["table_html"] = []
                chunk["table_rows"] = []
                chunk["content_type"] = "manual_text"

            page_start = int(chunk.get("page_start", 0))
            page_end = int(chunk.get("page_end", page_start))
            for page in document_ir.get("pages", []):
                if not isinstance(page, dict):
                    continue
                page_number = int(page.get("page_number", 0))
                if page_start <= page_number <= page_end:
                    page["manual_edit_count"] = int(page.get("manual_edit_count", 0)) + 1
                    page["manual_edited"] = True

            qa = document_ir.get("qa", {})
            if not isinstance(qa, dict):
                qa = {}
                document_ir["qa"] = qa
            gates = qa.get("quality_gates", [])
            if not isinstance(gates, list):
                gates = []
            gates = [
                gate
                for gate in gates
                if not isinstance(gate, dict) or gate.get("gate") != "manual_revision"
            ]
            manual_revision_count = sum(
                bool(item.get("manual_revision")) for item in chunks if isinstance(item, dict)
            )
            gates.append(
                {
                    "gate": "manual_revision",
                    "status": "warn",
                    "message": (
                        f"已有 {manual_revision_count} 个 Chunk 经人工修订；发布前请复核修改记录。"
                    ),
                }
            )
            qa["quality_gates"] = gates
            qa["manual_revision_count"] = manual_revision_count

            output_folder = Path(job.output_path or "")
            self._write_json_atomic(output_folder / "document_ir.json", document_ir)
            self._write_json_atomic(output_folder / "qa.json", qa)
            self._write_text_atomic(
                output_folder / "chunks.jsonl",
                "".join(
                    json.dumps(item, ensure_ascii=False) + "\n"
                    for item in chunks
                    if isinstance(item, dict)
                ),
            )
            with (output_folder / "manual_edits.jsonl").open("a", encoding="utf-8") as audit:
                audit.write(
                    json.dumps(
                        {
                            "job_id": job_id,
                            "chunk_id": chunk_id,
                            "revision": revision,
                            "updated_at": now.isoformat(),
                            "updated_by": updated_by,
                            "reason": reason,
                            "previous_text_sha256": previous_hash,
                            "current_text_sha256": current_hash,
                            "replaced_structured_table": had_structured_table,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            warning_count = sum(
                isinstance(gate, dict) and gate.get("status") == "warn" for gate in gates
            )
            job.quality.warning_count = warning_count
            if job.status == IngestionJobStatus.COMPLETED:
                job.status = IngestionJobStatus.WARNING
            job.progress = IngestionProgress(
                stage=job.status.value,
                percent=100,
                message="人工修订已保存，请在发布前复核修改内容",
                current=job.quality.page_count,
                total=job.quality.page_count,
            )
            previous_dataset_id = job.publication.dataset_id or job.knowledge_base_id
            job.publication = RagflowPublication(
                status=PublicationStatus.NOT_REQUESTED,
                stage="not_requested",
                percent=0,
                message="内容已人工修订，需要重新生成发布计划",
                dataset_id=previous_dataset_id,
                updated_at=now,
            )
            job.review = IngestionReview(
                status=ReviewStatus.UNREVIEWED,
                note="Chunk 已人工修订，需要重新审核。",
                updated_at=now,
            )
            job.updated_at = now
            self._write_job(job)

            return IngestionChunkUpdateResult(
                job_id=job_id,
                chunk_id=chunk_id,
                text=text,
                content_type=str(chunk.get("content_type", "text")),
                page_start=page_start,
                page_end=page_end,
                revision=revision,
                updated_at=now,
            )

    def source_file(self, job_id: str) -> tuple[Path, str, str]:
        job = self._read_job(job_id)
        source_path = Path(job.original_path).resolve()
        if not source_path.is_relative_to(self._root) or not source_path.is_file():
            raise AppError(
                code="INGESTION_SOURCE_MISSING",
                message="The original ingestion source is unavailable.",
                status_code=404,
                details={"job_id": job_id},
            )
        return source_path, job.media_type, job.source_name

    def asset_file(self, job_id: str, asset_id: str) -> tuple[Path, str]:
        job = self._read_job(job_id)
        payload = self._load_document_ir(job)
        asset = next(
            (
                item
                for item in payload.get("assets", [])
                if isinstance(item, dict) and item.get("asset_id") == asset_id
            ),
            None,
        )
        if asset is None or not job.output_path:
            raise AppError(
                code="INGESTION_ASSET_NOT_FOUND",
                message="The requested parsed asset does not exist.",
                status_code=404,
                details={"job_id": job_id, "asset_id": asset_id},
            )
        output_folder = Path(job.output_path).resolve()
        relative_path = Path(str(asset.get("relative_path", "")))
        asset_path = (output_folder / relative_path).resolve()
        if not asset_path.is_relative_to(output_folder) or not asset_path.is_file():
            raise AppError(
                code="INGESTION_ASSET_MISSING",
                message="The requested parsed asset file is unavailable.",
                status_code=404,
                details={"job_id": job_id, "asset_id": asset_id},
            )
        return asset_path, str(asset.get("mime_type", "application/octet-stream"))

    def export_job(
        self,
        job_id: str,
        *,
        format: str = "zip",
    ) -> tuple[bytes, str, str]:
        job = self._read_job(job_id)
        if job.status not in {
            IngestionJobStatus.COMPLETED,
            IngestionJobStatus.WARNING,
            IngestionJobStatus.NEEDS_REVIEW,
        }:
            raise AppError(
                code="INGESTION_EXPORT_NOT_READY",
                message="清洗结果尚未就绪；只有质量通过、需要抽检或需要复核的资料才能导出。",
                status_code=409,
                details={"status": job.status.value},
            )
        document_ir = self._load_document_ir(job)
        asset_root = Path(job.output_path) if job.output_path else None
        stem = Path(job.source_name).stem or "document"
        safe_stem = SAFE_FILENAME_RE.sub("_", stem).strip(" .") or "document"
        normalized = (format or "zip").casefold()
        if normalized == "md":
            markdown = render_markdown(document_ir, include_images=False)
            return (
                markdown.encode("utf-8"),
                f"{safe_stem}.md",
                "text/markdown; charset=utf-8",
            )
        if normalized != "zip":
            raise AppError(
                code="INGESTION_EXPORT_FORMAT_UNSUPPORTED",
                message="不支持的导出格式；仅支持 zip 或 md。",
                status_code=422,
                details={"format": format},
            )
        archive = build_export_archive(document_ir, asset_root, job.source_name)
        return archive, f"{safe_stem}.zip", "application/zip"

    def export_jobs(self, job_ids: list[str]) -> tuple[bytes, str, str]:
        documents: list[tuple[dict[str, Any], Path | None, str]] = []
        for job_id in dict.fromkeys(job_ids):
            job = self._read_job(job_id)
            if job.status not in {
                IngestionJobStatus.COMPLETED,
                IngestionJobStatus.WARNING,
                IngestionJobStatus.NEEDS_REVIEW,
            }:
                raise AppError(
                    code="INGESTION_EXPORT_NOT_READY",
                    message=f"“{job.source_name}”的清洗结果尚未就绪，无法导出。",
                    status_code=409,
                    details={"job_id": job_id, "status": job.status.value},
                )
            document_ir = self._load_document_ir(job)
            documents.append(
                (
                    document_ir,
                    Path(job.output_path) if job.output_path else None,
                    job.source_name,
                )
            )
        if not documents:
            raise AppError(
                code="INGESTION_EXPORT_EMPTY",
                message="没有可导出的资料。",
                status_code=422,
            )
        archive = build_batch_export_archive(documents)
        return archive, "cleaned-export.zip", "application/zip"

    def published_source_file(self, dataset_id: str, document_id: str):
        """Find the immutable local source for an exact publication identity."""
        for job_path in self._jobs.glob("*/job.json"):
            job = IngestionJob.model_validate_json(job_path.read_text(encoding="utf-8"))
            if (job.publication.dataset_id == dataset_id
                    and job.publication.document_id == document_id):
                return self.source_file(job.job_id)
        return None

    def _load_document_ir(self, job: IngestionJob) -> dict[str, Any]:
        if not job.output_path:
            raise AppError(
                code="INGESTION_OUTPUT_NOT_READY",
                message="The ingestion output is not ready for publication.",
                status_code=409,
                details={"status": job.status.value},
            )
        path = Path(job.output_path) / "document_ir.json"
        if not path.is_file():
            raise AppError(
                code="INGESTION_OUTPUT_MISSING",
                message="The ingestion output is incomplete.",
                status_code=500,
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AppError(
                code="INGESTION_OUTPUT_INVALID",
                message="The ingestion output has an invalid document IR.",
                status_code=500,
            )
        return payload

    @staticmethod
    def _public_plan(job: IngestionJob, plan: RagflowPlan) -> RagflowPublishPlan:
        return RagflowPublishPlan(
            job_id=job.job_id,
            dataset_id=plan.dataset_id,
            source_name=plan.source_name,
            source_sha256=plan.source_sha256,
            chunk_count=len(plan.chunks),
            plan_sha256=plan.plan_sha256,
            sample_payloads=[chunk.payload for chunk in plan.chunks[:3]],
        )

    async def publish_job(
        self,
        job_id: str,
        *,
        dataset_id: str | None,
        dry_run: bool,
    ) -> RagflowPublishResult:
        job = self._read_job(job_id)
        manually_overridden = (
            job.status == IngestionJobStatus.NEEDS_REVIEW
            and job.review.status == ReviewStatus.APPROVED
            and job.review.quality_gate_override
            and bool(job.review.overridden_quality_gates)
        )
        if (
            job.status not in {IngestionJobStatus.COMPLETED, IngestionJobStatus.WARNING}
            and not manually_overridden
        ):
            raise AppError(
                code="INGESTION_QUALITY_GATE_BLOCKED",
                message=("只有质量通过、警告级，或已由人工显式确认失败门禁无误的任务才能发布。"),
                status_code=409,
                details={"status": job.status.value},
            )
        if not dry_run and job.review.status != ReviewStatus.APPROVED:
            raise AppError(
                code="INGESTION_REVIEW_REQUIRED",
                message="资料尚未完成人工审核；请先标记为审核通过再发送到 RAGFlow。",
                status_code=409,
                details={"review_status": job.review.status.value},
            )
        resolved_dataset_id = (dataset_id or job.knowledge_base_id or "").strip()
        if not resolved_dataset_id:
            raise AppError(
                code="RAGFLOW_DATASET_REQUIRED",
                message="A target RAGFlow dataset ID is required.",
                status_code=422,
            )
        if self._publisher is None:
            raise AppError(
                code="RAGFLOW_PUBLISHER_DISABLED",
                message="The RAGFlow publisher is not configured.",
                status_code=503,
            )

        document_ir = self._load_document_ir(job)
        plan = self._publisher.build_plan(
            dataset_id=resolved_dataset_id,
            source_name=job.source_name,
            source_sha256=job.source_sha256,
            document_ir=document_ir,
            asset_root=Path(job.output_path) if job.output_path else None,
        )
        if not plan.chunks:
            raise AppError(
                code="INGESTION_NO_PUBLISHABLE_CHUNKS",
                message="The ingestion output contains no publishable chunks.",
                status_code=409,
            )
        public_plan = self._public_plan(job, plan)
        now = self._now()
        if dry_run:
            job.publication = RagflowPublication(
                status=PublicationStatus.PLANNED,
                stage="planned",
                percent=0,
                message=f"发布计划已生成，共 {len(plan.chunks)} 个 Chunk",
                dataset_id=resolved_dataset_id,
                document_id=job.publication.document_id,
                plan_sha256=plan.plan_sha256,
                planned_chunk_count=len(plan.chunks),
                published_chunk_count=job.publication.published_chunk_count,
                skipped_chunk_count=job.publication.skipped_chunk_count,
                updated_at=now,
            )
            job.updated_at = now
            self._write_job(job)
            return RagflowPublishResult(
                dry_run=True,
                plan=public_plan,
                publication=job.publication,
            )

        if (
            job.publication.status == PublicationStatus.PUBLISHED
            and job.publication.dataset_id == resolved_dataset_id
            and job.publication.plan_sha256 == plan.plan_sha256
        ):
            return RagflowPublishResult(
                dry_run=False,
                plan=public_plan,
                publication=job.publication,
            )

        previous_document_id = (
            job.publication.document_id
            if job.publication.dataset_id == resolved_dataset_id
            else None
        )
        job.publication = RagflowPublication(
            status=PublicationStatus.PUBLISHING,
            stage="uploading_document",
            percent=2,
            message="正在向 RAGFlow 上传原始文件",
            dataset_id=resolved_dataset_id,
            document_id=previous_document_id,
            plan_sha256=plan.plan_sha256,
            planned_chunk_count=len(plan.chunks),
            updated_at=now,
        )
        job.updated_at = now
        self._write_job(job)

        def progress(document_id: str, published: int, skipped: int) -> None:
            completed_chunks = published + skipped
            total_chunks = max(1, len(plan.chunks))
            job.publication.document_id = document_id
            job.publication.stage = "publishing_chunks"
            job.publication.percent = min(
                99,
                5 + int(completed_chunks / total_chunks * 94),
            )
            job.publication.message = f"正在写入 Chunk：{completed_chunks}/{len(plan.chunks)}"
            job.publication.published_chunk_count = published
            job.publication.skipped_chunk_count = skipped
            job.publication.updated_at = self._now()
            job.updated_at = job.publication.updated_at
            self._write_job(job)

        try:
            document_id, published, skipped = await self._publisher.publish(
                plan,
                Path(job.original_path),
                document_id=previous_document_id,
                progress=progress,
            )
            job.publication.status = PublicationStatus.PUBLISHED
            job.publication.stage = "published"
            job.publication.percent = 100
            job.publication.message = f"发布完成：新增 {published}，跳过 {skipped}"
            job.publication.document_id = document_id
            job.publication.published_chunk_count = published
            job.publication.skipped_chunk_count = skipped
            job.publication.error_code = None
            job.publication.error_message = None
        except AppError as exc:
            job.publication.status = PublicationStatus.FAILED
            job.publication.stage = "failed"
            job.publication.message = exc.message
            job.publication.error_code = exc.code
            job.publication.error_message = exc.message
            job.publication.updated_at = self._now()
            job.updated_at = job.publication.updated_at
            self._write_job(job)
            raise
        job.publication.updated_at = self._now()
        job.updated_at = job.publication.updated_at
        self._write_job(job)
        return RagflowPublishResult(
            dry_run=False,
            plan=public_plan,
            publication=job.publication,
        )

    async def parser_status(self) -> list[ParserServiceStatus]:
        statuses = [
            ParserServiceStatus(
                name="native-pdf",
                enabled=True,
                ready=True,
                base_url="local",
            )
        ]
        for plugin in self._parsers.plugins:
            if not isinstance(plugin, HybridPdfParser):
                statuses.append(
                    ParserServiceStatus(
                        name=plugin.name,
                        enabled=True,
                        ready=True,
                        base_url="local",
                    )
                )
                continue
            clients: tuple[
                MinerUClient | DoclingClient | FigureVisionClient,
                ...,
            ] = (
                plugin.mineru,
                plugin.docling,
                plugin.figure_vlm,
            )
            for client in clients:
                ready, detail = await asyncio.to_thread(client.probe)
                statuses.append(
                    ParserServiceStatus(
                        name=client.name,
                        enabled=client.enabled,
                        ready=ready,
                        base_url=client.settings.base_url,
                        detail=detail,
                    )
                )
        return statuses

    async def close(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        if self._publisher is not None:
            await self._publisher.close()
        close_parsers = getattr(self._parsers, "close", None)
        if callable(close_parsers):
            await asyncio.to_thread(close_parsers)
