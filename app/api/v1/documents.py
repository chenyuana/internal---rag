from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header

from app.api.dependencies import (
    RetrievalServiceDependency,
    ServiceRegistryDependency,
    SettingsDependency,
)
from app.schemas.documents import (
    DocumentSectionGenerateRequest,
    DocumentSectionGenerateResponse,
    GuidanceSuggestRequest,
    GuidanceSuggestResponse,
)
from app.services.document_section_generator import DocumentSectionGenerator

router = APIRouter()
UserIdHeader = Annotated[str, Header(alias="X-User-ID")]


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
