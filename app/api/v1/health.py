from __future__ import annotations

from fastapi import APIRouter, Response, status

from app.api.dependencies import ServiceRegistryDependency, SettingsDependency
from app.schemas.health import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDependency) -> HealthResponse:
    """Liveness check; it intentionally does not call external services."""
    return HealthResponse(
        status="ok",
        service=settings.app.name,
        version=settings.app.version,
        environment=settings.app.environment,
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ReadinessResponse}},
)
async def ready(
    response: Response,
    services: ServiceRegistryDependency,
) -> ReadinessResponse:
    """Readiness check for every enabled external dependency."""
    dependencies = await services.probe_all()
    is_ready = all(item.status == "ready" for item in dependencies if item.required)
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="ready" if is_ready else "not_ready",
        dependencies=dependencies,
    )
