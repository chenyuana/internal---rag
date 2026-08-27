from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header, Response, status

from app.api.dependencies import ServiceRegistryDependency, SettingsDependency
from app.core.exceptions import AppError
from app.schemas.health import ServiceStatus
from app.schemas.models import (
    AnswerModelCatalog,
    ModelConnectionCreate,
    ModelConnectionCreated,
)

router = APIRouter()
UserIdHeader = Annotated[str, Header(alias="X-User-ID")]


@router.get("/models/status", response_model=list[ServiceStatus])
async def model_status(
    services: ServiceRegistryDependency,
) -> list[ServiceStatus]:
    return await services.probe_models()


@router.get("/answer-models", response_model=AnswerModelCatalog)
async def answer_models(
    services: ServiceRegistryDependency,
    user_id: UserIdHeader = "development-user",
) -> AnswerModelCatalog:
    """List configured/local models and this user's in-memory API connections."""
    return await services.answer_model_manager.catalog(user_id=user_id)


@router.post(
    "/model-connections",
    response_model=ModelConnectionCreated,
    status_code=status.HTTP_201_CREATED,
)
async def connect_answer_model(
    payload: ModelConnectionCreate,
    services: ServiceRegistryDependency,
    user_id: UserIdHeader = "development-user",
) -> ModelConnectionCreated:
    source = await services.answer_model_manager.connect(payload, user_id=user_id)
    return ModelConnectionCreated(source=source)


@router.delete("/model-connections/{source_id}", status_code=status.HTTP_204_NO_CONTENT)
async def disconnect_answer_model(
    source_id: str,
    services: ServiceRegistryDependency,
    user_id: UserIdHeader = "development-user",
) -> Response:
    await services.answer_model_manager.disconnect(source_id, user_id=user_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/ragflow/status", response_model=ServiceStatus)
async def ragflow_status(
    services: ServiceRegistryDependency,
) -> ServiceStatus:
    return await services.probe_ragflow()


@router.get("/datasets", response_model=list[dict[str, str]])
async def list_datasets(
    services: ServiceRegistryDependency,
) -> list[dict[str, str]]:
    """List RAGFlow datasets for the chat interface knowledge-base selector."""
    ragflow = services.ragflow_client
    if ragflow is None:
        raise AppError(
            code="RAGFLOW_DISABLED",
            message="RAGFlow is not enabled.",
            status_code=503,
        )
    return await ragflow.list_datasets()


@router.get("/config", response_model=dict[str, object])
async def effective_config(
    settings: SettingsDependency,
) -> dict[str, object]:
    """Return an intentionally redacted view of the effective configuration."""
    return settings.public_view()
