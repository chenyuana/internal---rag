from __future__ import annotations

import time
from typing import Annotated, cast

from fastapi import APIRouter, File, Header, Request, UploadFile

from app.api.dependencies import RagPipelineDependency
from app.core.exceptions import AppError
from app.core.middleware import current_request_id
from app.schemas.chat import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ReferenceDocumentParsed,
)
from app.services.performance_monitor import PerformanceMonitor
from app.services.reference_document import parse_reference_document

router = APIRouter()
UserIdHeader = Annotated[str, Header(alias="X-User-ID")]


@router.post("/reference-documents/parse", response_model=ReferenceDocumentParsed)
async def parse_reference(
    file: Annotated[UploadFile, File()],
) -> ReferenceDocumentParsed:
    return parse_reference_document(file.filename or "reference-document", await file.read())


@router.post("/completions", response_model=ChatCompletionResponse)
async def completions(
    payload: ChatCompletionRequest,
    pipeline: RagPipelineDependency,
    request: Request,
    user_id: UserIdHeader = "development-user",
) -> ChatCompletionResponse:
    """Answer pipeline wrapped with live performance telemetry recording."""
    monitor = cast(PerformanceMonitor, request.app.state.performance_monitor)
    request_id = current_request_id.get() or "unknown"
    started = time.perf_counter()
    try:
        result = await pipeline.run(payload, user_id=user_id)
        diagnostics = result.diagnostics
        monitor.record(
            request_id=request_id,
            question=payload.query,
            ok=True,
            error_type=None,
            status=result.status,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            latencies_ms=diagnostics.latencies_ms if diagnostics else None,
            model_calls=diagnostics.model_calls if diagnostics else 0,
            retrieval_calls=diagnostics.retrieval_calls if diagnostics else 0,
            stage_counts=diagnostics.stage_counts if diagnostics else None,
            answer=result.answer,
            contexts=[citation.quote for citation in result.citations if citation.quote],
        )
        return result
    except Exception as exc:  # noqa: BLE001 - record every failure, then re-raise
        error_type = exc.code if isinstance(exc, AppError) else type(exc).__name__
        monitor.record(
            request_id=request_id,
            question=payload.query,
            ok=False,
            error_type=error_type,
            status=None,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
        raise
