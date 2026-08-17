from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import ServiceRegistryDependency, SettingsDependency
from app.core.exceptions import AppError
from app.schemas.health import ServiceStatus
from app.services.registry import ServiceRegistry

router = APIRouter()


@router.get("/models/status", response_model=list[ServiceStatus])
async def model_status(
    services: ServiceRegistryDependency,
) -> list[ServiceStatus]:
    return await services.probe_models()


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
