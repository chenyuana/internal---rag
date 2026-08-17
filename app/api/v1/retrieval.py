from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header

from app.api.dependencies import RetrievalServiceDependency
from app.schemas.retrieval import (
    RetrievalDebugResponse,
    RetrievalSearchRequest,
    RetrievalSearchResponse,
)

router = APIRouter()
UserIdHeader = Annotated[str, Header(alias="X-User-ID")]


@router.post("/search", response_model=RetrievalSearchResponse)
async def search(
    payload: RetrievalSearchRequest,
    retrieval: RetrievalServiceDependency,
    user_id: UserIdHeader = "development-user",
) -> RetrievalSearchResponse:
    execution = await retrieval.execute(payload, user_id=user_id)
    return execution.search_response()


@router.post("/debug", response_model=RetrievalDebugResponse)
async def debug(
    payload: RetrievalSearchRequest,
    retrieval: RetrievalServiceDependency,
    user_id: UserIdHeader = "development-user",
) -> RetrievalDebugResponse:
    execution = await retrieval.execute(payload, user_id=user_id)
    return execution.debug_response()
