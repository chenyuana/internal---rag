from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from app.core.config import Settings
from app.ingestion.jobs import IngestionJobService
from app.pipelines.rag_pipeline import RagPipeline
from app.services.registry import ServiceRegistry
from app.services.retrieval_service import RetrievalService


def get_settings(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_service_registry(request: Request) -> ServiceRegistry:
    return cast(ServiceRegistry, request.app.state.services)


def get_retrieval_service(request: Request) -> RetrievalService:
    return cast(RetrievalService, request.app.state.retrieval_service)


def get_rag_pipeline(request: Request) -> RagPipeline:
    return cast(RagPipeline, request.app.state.rag_pipeline)


def get_ingestion_service(request: Request) -> IngestionJobService:
    return cast(IngestionJobService, request.app.state.ingestion_service)


SettingsDependency = Annotated[Settings, Depends(get_settings)]
ServiceRegistryDependency = Annotated[
    ServiceRegistry,
    Depends(get_service_registry),
]
RetrievalServiceDependency = Annotated[
    RetrievalService,
    Depends(get_retrieval_service),
]
RagPipelineDependency = Annotated[RagPipeline, Depends(get_rag_pipeline)]
IngestionServiceDependency = Annotated[
    IngestionJobService,
    Depends(get_ingestion_service),
]
