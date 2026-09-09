from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header
from fastapi.responses import FileResponse, Response

from app.api.dependencies import (
    IngestionServiceDependency,
    RetrievalServiceDependency,
    ServiceRegistryDependency,
    SettingsDependency,
)
from app.core.exceptions import AppError
from app.schemas.documents import (
    DocumentSectionGenerateRequest,
    DocumentSectionGenerateResponse,
    GuidanceSuggestRequest,
    GuidanceSuggestResponse,
)
from app.services.document_section_generator import DocumentSectionGenerator

router = APIRouter()
UserIdHeader = Annotated[str, Header(alias="X-User-ID")]


@router.get("/document/{dataset_id}/{document_id}/file")
async def document_file(
    dataset_id: str,
    document_id: str,
    services: ServiceRegistryDependency,
    ingestion: IngestionServiceDependency,
) -> Response:
    """代理 RAGFlow 的原始文档下载，供前端"定位到原文"按页码跳转。"""
    # Published RAGFlow documents may be cleaned Markdown, not the source PDF.
    # Resolve only by the exact dataset/document publication identity, never a
    # client-supplied filesystem path or an ambiguous display filename.
    local_source = ingestion.published_source_file(dataset_id, document_id)
    if local_source is not None:
        path, media_type, _ = local_source
        if media_type != "application/pdf":
            raise AppError(
                code="SOURCE_NOT_PDF",
                message="该引用的原始文件不是 PDF。",
                status_code=415,
            )
        return FileResponse(
            path,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "inline",
                "Cache-Control": "no-store",
            },
        )
    if services.ragflow_client is None:
        raise AppError(
            code="RAGFLOW_DISABLED",
            message="RAGFlow 未启用。",
            status_code=503,
        )
    content, content_type = await services.ragflow_client.download_document(
        dataset_id,
        document_id,
    )
    if b"%PDF-" not in content[:1024]:
        raise AppError(
            code="SOURCE_NOT_PDF",
            message="未找到原始 PDF；知识库中可能仅保存了清洗文本。",
            status_code=415,
        )
    return Response(
        content=content,
        media_type="application/pdf",
        headers={
            "Cache-Control": "no-store",
            "Content-Disposition": "inline",
        },
    )


@router.post("/sections/generate", response_model=DocumentSectionGenerateResponse)
async def generate_section(
    payload: DocumentSectionGenerateRequest,
    settings: SettingsDependency,
    retrieval: RetrievalServiceDependency,
    services: ServiceRegistryDependency,
    user_id: UserIdHeader = "development-user",
) -> DocumentSectionGenerateResponse:
    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=retrieval,
        answer_model=services.answer_model_client,
        answer_models=services.answer_model_manager,
    )
    return await generator.generate(payload, user_id=user_id)


@router.post("/sections/guidance-suggest", response_model=GuidanceSuggestResponse)
async def suggest_section_guidances(
    payload: GuidanceSuggestRequest,
    settings: SettingsDependency,
    retrieval: RetrievalServiceDependency,
    services: ServiceRegistryDependency,
    user_id: UserIdHeader = "development-user",
) -> GuidanceSuggestResponse:
    generator = DocumentSectionGenerator(
        settings=settings,
        retrieval=retrieval,
        answer_model=services.answer_model_client,
        answer_models=services.answer_model_manager,
    )
    return await generator.suggest_guidances(payload, user_id=user_id)
