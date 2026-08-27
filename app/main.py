from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import Settings, load_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.middleware import RequestContextMiddleware
from app.ingestion.jobs import IngestionJobService
from app.pipelines.rag_pipeline import RagPipeline
from app.services.performance_monitor import PerformanceMonitor
from app.services.ragas_evaluator import RagasTaskManager
from app.services.registry import ServiceRegistry
from app.services.retrieval_service import RetrievalService


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application with explicit, testable dependencies."""
    resolved_settings = settings or load_settings()
    configure_logging(resolved_settings.logging)

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.settings = resolved_settings
        application.state.performance_monitor = PerformanceMonitor()
        application.state.ragas_manager = RagasTaskManager(resolved_settings.ragas)
        application.state.ingestion_service = IngestionJobService(
            resolved_settings.ingestion,
            ragflow_settings=resolved_settings.ragflow,
        )
        await application.state.ingestion_service.start()
        application.state.services = ServiceRegistry.from_settings(resolved_settings)
        application.state.retrieval_service = RetrievalService.from_registry(
            resolved_settings,
            application.state.services,
        )
        application.state.rag_pipeline = RagPipeline.from_services(
            settings=resolved_settings,
            retrieval=application.state.retrieval_service,
            answer_model=application.state.services.answer_model_client,
            answer_models=application.state.services.answer_model_manager,
        )
        try:
            yield
        finally:
            await application.state.ingestion_service.close()
            await application.state.services.close()

    application = FastAPI(
        title="Internal RAG Gateway",
        version=resolved_settings.app.version,
        description="Evidence-grounded gateway for internal technical documents.",
        lifespan=lifespan,
    )
    web_root = Path(__file__).resolve().parent / "web"
    application.mount(
        "/assets",
        StaticFiles(directory=web_root / "static"),
        name="web-assets",
    )

    @application.get("/ingestion", include_in_schema=False)
    async def ingestion_interface() -> FileResponse:
        return FileResponse(
            web_root / "ingestion.html", headers={"Cache-Control": "no-store"}
        )

    @application.get("/chat", include_in_schema=False)
    async def chat_interface() -> FileResponse:
        return FileResponse(web_root / "chat.html", headers={"Cache-Control": "no-store"})

    @application.get("/documents", include_in_schema=False)
    async def document_workbench_interface() -> FileResponse:
        return FileResponse(
            web_root / "documents.html", headers={"Cache-Control": "no-store"}
        )

    @application.get("/evaluation", include_in_schema=False)
    async def evaluation_interface() -> FileResponse:
        return FileResponse(
            web_root / "evaluation.html", headers={"Cache-Control": "no-store"}
        )

    @application.get("/monitor", include_in_schema=False)
    async def monitor_interface() -> FileResponse:
        return FileResponse(
            web_root / "monitor.html", headers={"Cache-Control": "no-store"}
        )

    application.add_middleware(RequestContextMiddleware)
    register_exception_handlers(application)
    application.include_router(api_router)
    return application


app = create_app()
