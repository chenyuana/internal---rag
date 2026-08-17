from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, Header, Query, Response, UploadFile, status
from fastapi.responses import FileResponse

from app.api.dependencies import IngestionServiceDependency
from app.schemas.ingestion import (
    IngestionChunkUpdateRequest,
    IngestionChunkUpdateResult,
    IngestionJob,
    IngestionPageDetail,
    IngestionPreview,
    IngestionPublishRequest,
    IngestionQueueStatus,
    IngestionReviewUpdateRequest,
    ParserServiceStatus,
    RagflowPublishResult,
    ReviewStatus,
)

router = APIRouter()
UserIdHeader = Annotated[str, Header(alias="X-User-ID")]


@router.get("/parsers/status", response_model=list[ParserServiceStatus])
async def get_parser_status(
    ingestion: IngestionServiceDependency,
) -> list[ParserServiceStatus]:
    return await ingestion.parser_status()


@router.get("/queue", response_model=IngestionQueueStatus)
async def get_ingestion_queue(
    ingestion: IngestionServiceDependency,
) -> IngestionQueueStatus:
    return ingestion.queue_status()


@router.post(
    "/jobs",
    response_model=IngestionJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_ingestion_job(
    ingestion: IngestionServiceDependency,
    file: Annotated[UploadFile, File()],
    knowledge_base_id: Annotated[str | None, Form()] = None,
    source_relative_path: Annotated[str | None, Form()] = None,
    batch_id: Annotated[str | None, Form(max_length=100)] = None,
    sequence_in_batch: Annotated[int | None, Form(ge=0)] = None,
    user_id: UserIdHeader = "development-user",
) -> IngestionJob:
    job = await ingestion.create_job(
        file,
        knowledge_base_id=knowledge_base_id,
        source_relative_path=source_relative_path,
        batch_id=batch_id,
        sequence_in_batch=sequence_in_batch,
        created_by=user_id,
    )
    await ingestion.enqueue(job.job_id)
    return job


@router.get("/jobs", response_model=list[IngestionJob])
async def list_ingestion_jobs(
    ingestion: IngestionServiceDependency,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    query: Annotated[str | None, Query(max_length=200)] = None,
    offset: Annotated[int, Query(ge=0)] = 0,
    review_status: Annotated[ReviewStatus | None, Query()] = None,
) -> list[IngestionJob]:
    return ingestion.list_jobs(
        limit=limit,
        offset=offset,
        query=query,
        review_status=review_status,
    )


@router.get("/jobs/{job_id}", response_model=IngestionJob)
async def get_ingestion_job(
    job_id: str,
    ingestion: IngestionServiceDependency,
) -> IngestionJob:
    return ingestion.get_job(job_id)


@router.get("/jobs/{job_id}/preview", response_model=IngestionPreview)
async def get_ingestion_preview(
    job_id: str,
    ingestion: IngestionServiceDependency,
    chunk_limit: Annotated[int | None, Query(ge=1, le=100)] = None,
) -> IngestionPreview:
    return ingestion.preview(job_id, chunk_limit=chunk_limit)


@router.get("/jobs/{job_id}/pages/{page_number}", response_model=IngestionPageDetail)
async def get_ingestion_page_detail(
    job_id: str,
    page_number: int,
    ingestion: IngestionServiceDependency,
) -> IngestionPageDetail:
    return ingestion.page_detail(job_id, page_number)


@router.patch(
    "/jobs/{job_id}/chunks/{chunk_id}",
    response_model=IngestionChunkUpdateResult,
)
async def update_ingestion_chunk(
    job_id: str,
    chunk_id: str,
    request: IngestionChunkUpdateRequest,
    ingestion: IngestionServiceDependency,
    user_id: UserIdHeader = "local-operator",
) -> IngestionChunkUpdateResult:
    return await ingestion.update_chunk(
        job_id,
        chunk_id,
        text=request.text,
        reason=request.reason,
        updated_by=user_id,
    )


@router.patch(
    "/jobs/{job_id}/review",
    response_model=IngestionJob,
)
async def update_ingestion_review(
    job_id: str,
    request: IngestionReviewUpdateRequest,
    ingestion: IngestionServiceDependency,
    user_id: UserIdHeader = "local-operator",
) -> IngestionJob:
    return await ingestion.update_review(
        job_id,
        status=request.status,
        note=request.note,
        reviewer=user_id,
        quality_gate_override=request.quality_gate_override,
    )


@router.get("/jobs/{job_id}/source", response_class=FileResponse)
async def get_ingestion_source(
    job_id: str,
    ingestion: IngestionServiceDependency,
) -> FileResponse:
    path, media_type, _ = ingestion.source_file(job_id)
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Content-Disposition": "inline",
            "Cache-Control": "private, max-age=3600",
        },
    )


@router.get("/jobs/{job_id}/assets/{asset_id}", response_class=FileResponse)
async def get_ingestion_asset(
    job_id: str,
    asset_id: str,
    ingestion: IngestionServiceDependency,
) -> FileResponse:
    path, media_type = ingestion.asset_file(job_id, asset_id)
    return FileResponse(
        path,
        media_type=media_type,
        headers={
            "Content-Disposition": "inline",
            # Parsed assets are rewritten in place when a job is retried.
            # Never let a browser reuse a stale crop under the same asset id.
            "Cache-Control": "private, no-store, max-age=0",
        },
    )


@router.post(
    "/jobs/{job_id}/retry",
    response_model=IngestionJob,
    status_code=status.HTTP_202_ACCEPTED,
)
async def retry_ingestion_job(
    job_id: str,
    ingestion: IngestionServiceDependency,
) -> IngestionJob:
    job = await ingestion.retry_job(job_id)
    await ingestion.enqueue(job.job_id)
    return job


@router.delete(
    "/jobs/{job_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
)
async def delete_ingestion_job(
    job_id: str,
    ingestion: IngestionServiceDependency,
) -> Response:
    await ingestion.delete_job(job_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/jobs/{job_id}/publish", response_model=RagflowPublishResult)
async def publish_ingestion_job(
    job_id: str,
    request: IngestionPublishRequest,
    ingestion: IngestionServiceDependency,
) -> RagflowPublishResult:
    return await ingestion.publish_job(
        job_id,
        dataset_id=request.dataset_id,
        dry_run=request.dry_run,
    )
