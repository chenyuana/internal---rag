from fastapi import APIRouter

from app.api.v1.admin import router as admin_router
from app.api.v1.chat import router as chat_router
from app.api.v1.documents import router as documents_router
from app.api.v1.evaluation import router as evaluation_router
from app.api.v1.health import router as health_router
from app.api.v1.ingestion import router as ingestion_router
from app.api.v1.monitor import router as monitor_router
from app.api.v1.retrieval import router as retrieval_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])
api_router.include_router(retrieval_router, prefix="/api/v1/retrieval", tags=["retrieval"])
api_router.include_router(chat_router, prefix="/api/v1/chat", tags=["chat"])
api_router.include_router(
    documents_router,
    prefix="/api/v1/documents",
    tags=["documents"],
)
api_router.include_router(
    evaluation_router,
    prefix="/api/v1/evaluation",
    tags=["evaluation"],
)
api_router.include_router(
    ingestion_router,
    prefix="/api/v1/ingestion",
    tags=["ingestion"],
)
api_router.include_router(
    monitor_router,
    prefix="/api/v1/monitor",
    tags=["monitor"],
)
