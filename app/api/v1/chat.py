from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Header

from app.api.dependencies import RagPipelineDependency
from app.schemas.chat import ChatCompletionRequest, ChatCompletionResponse

router = APIRouter()
UserIdHeader = Annotated[str, Header(alias="X-User-ID")]


@router.post("/completions", response_model=ChatCompletionResponse)
async def completions(
    payload: ChatCompletionRequest,
    pipeline: RagPipelineDependency,
    user_id: UserIdHeader = "development-user",
) -> ChatCompletionResponse:
    return await pipeline.run(payload, user_id=user_id)
