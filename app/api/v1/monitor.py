from __future__ import annotations

import json
import logging
import time
from typing import Annotated, Any, Literal, cast

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from app.api.dependencies import SettingsDependency
from app.core.exceptions import AppError
from app.pipelines.rag_pipeline import RagPipeline
from app.schemas.chat import ChatCompletionRequest
from app.schemas.models import AnswerModelSelection, ModelConnectionCreate
from app.services.evaluation_results import EvaluationResultsReader
from app.services.performance_monitor import DEFAULT_MINUTES, PerformanceMonitor
from app.services.ragas_evaluator import (
    RagasError,
    RagasTaskManager,
    build_question_case,
)
from app.services.registry import ServiceRegistry

logger = logging.getLogger(__name__)

router = APIRouter(tags=["monitor"])

# The question-mode answering reuses the chat pipeline with the same user the
# chat page uses, so per-user runtime model connections resolve identically.
RAGAS_ANSWER_USER_ID = "dev"


class AnswerModelSelectionPayload(BaseModel):
    source_id: str
    model_name: str


class RagasRunRequest(BaseModel):
    mode: Literal["question", "benchmark"] = "question"
    # question mode
    question: str | None = None
    ground_truth: str | None = None
    knowledge_base_ids: list[str] = Field(default_factory=list)
    answer_model: AnswerModelSelectionPayload | None = None
    # benchmark mode: empty question_id = first `limit` questions with answers
    question_id: str | None = None
    question_ids: list[str] = Field(default_factory=list, max_length=100)
    limit: int = Field(default=5, ge=1, le=100)


class RagasProfileUpdate(BaseModel):
    source_id: str
    model: str
    persist: bool = False


class RagasConnectionCreate(BaseModel):
    name: str = ""
    base_url: str
    api_key: str = ""
    # Optional explicit model list (comma-friendly list); when empty the
    # endpoint tries to discover models via GET /models.
    models: list[str] = Field(default_factory=list)
    embedding_model: str = ""


def _monitor(request: Request) -> PerformanceMonitor:
    return cast(PerformanceMonitor, request.app.state.performance_monitor)


def _ragas(request: Request) -> RagasTaskManager:
    return cast(RagasTaskManager, request.app.state.ragas_manager)


async def _ollama_models(base_url: str) -> list[str]:
    """Best-effort listing of models installed in the local Ollama server."""
    try:
        import httpx

        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            payload = response.json()
            return sorted(model["name"] for model in payload.get("models", []))
    except Exception:  # noqa: BLE001 - listing is best-effort
        return []


async def _api_models(base_url: str, api_key: str) -> list[str]:
    """Discover model ids from an OpenAI-compatible ``GET /models`` endpoint."""
    import httpx

    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    root = base_url.rstrip("/")
    last_error: Exception | None = None
    async with httpx.AsyncClient(timeout=8.0) as client:
        for url in (f"{root}/models", f"{root}/v1/models"):
            try:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                payload = response.json()
                models = [
                    str(model.get("id") or model.get("name") or "").strip()
                    for model in payload.get("data", [])
                ]
                models = sorted(model for model in models if model)
                if models:
                    return models
            except Exception as exc:  # noqa: BLE001 - try the next path
                last_error = exc
    raise RagasError(f"无法从 {base_url} 获取模型列表（{last_error}）；请手动填写模型列表")


@router.get("/stats")
async def monitor_stats(
    request: Request,
    minutes: Annotated[
        float,
        Query(ge=0, le=24 * 60, description="聚合窗口（分钟），0 = 全部缓存"),
    ] = DEFAULT_MINUTES,
) -> dict[str, Any]:
    """Real-time aggregated performance metrics for recent chat requests."""
    return _monitor(request).stats(minutes=minutes)


@router.get("/services")
async def monitor_services(request: Request) -> dict[str, Any]:
    """Live health probe of every external dependency (RAGFlow / Ollama / ...)."""
    services = cast(ServiceRegistry, request.app.state.services)
    dependencies = await services.probe_all()
    is_ready = all(item.status == "ready" for item in dependencies if item.required)
    return {
        "status": "ready" if is_ready else "not_ready",
        "checked_at": time.time(),
        "dependencies": [item.model_dump() for item in dependencies],
    }


# --- RAGAS calibration evaluation -----------------------------------------


@router.get("/ragas/status")
async def ragas_status(request: Request) -> dict[str, Any]:
    """Whether RAGAS is available, which models it would use, and the last result."""
    return _ragas(request).status()


@router.get("/ragas/profile")
async def ragas_profile(request: Request) -> dict[str, Any]:
    """Model sources (local + API connections) and the active judge selection."""
    profile = _ragas(request).profile()
    ollama_models = await _ollama_models(profile["ollama_base_url"])
    profile["sources"][0]["models"] = ollama_models
    profile["ollama_models"] = ollama_models
    return profile


@router.put("/ragas/profile")
async def ragas_profile_update(
    request: Request, payload: RagasProfileUpdate
) -> dict[str, Any]:
    """Make the given model from the given source the active judge (no restart)."""
    try:
        profile = _ragas(request).select_model(
            source_id=payload.source_id,
            model=payload.model,
            persist=payload.persist,
        )
    except RagasError as exc:
        raise AppError(
            code="RAGAS_PROFILE_INVALID", message=str(exc), status_code=400
        ) from exc
    return profile


@router.post("/ragas/connections")
async def ragas_connections_create(
    request: Request, payload: RagasConnectionCreate
) -> dict[str, Any]:
    """Connect an OpenAI-compatible API and expose its models in the picker.

    The same connection is also registered as a chat answer model (user "dev"),
    so its models can answer questions through the chat pipeline and appear in
    the 回答模型 dropdown.
    """
    warning: str | None = None
    try:
        models = [model.strip() for model in payload.models if model.strip()]
        if not models:
            models = await _api_models(payload.base_url, payload.api_key)
        profile = _ragas(request).add_connection(
            name=payload.name,
            base_url=payload.base_url,
            api_key=payload.api_key,
            models=models,
            embedding_model=payload.embedding_model,
        )
        active_source_id = profile["active"]["source_id"]
    except RagasError as exc:
        raise AppError(
            code="RAGAS_CONNECTION_INVALID", message=str(exc), status_code=400
        ) from exc

    # Mirror the connection into the chat answer-model manager so its models
    # can also answer (retrieval + citations) for RAGAS question-mode runs.
    try:
        services = cast(Any, request.app.state.services)
        answer_source = await services.answer_model_manager.connect(
            ModelConnectionCreate(
                name=payload.name or "API 连接",
                base_url=payload.base_url,
                api_key=payload.api_key,
            ),
            user_id=RAGAS_ANSWER_USER_ID,
        )
        _ragas(request).set_answer_source(
            active_source_id, answer_source.source_id
        )
    except Exception as exc:  # noqa: BLE001 - judging still works without it
        logger.warning("ragas connection answer-model registration failed: %s", exc)
        warning = (
            "已接入判题模型；但注册回答模型失败（模型列表探测失败或服务未启用），"
            "回答模型下拉暂不含此 API，仍可用作判题。"
        )
    response = profile
    if warning:
        response["warning"] = warning
    return response


@router.delete("/ragas/connections/{source_id}")
async def ragas_connections_delete(
    source_id: str, request: Request
) -> dict[str, Any]:
    """Remove an API connection; the active choice falls back to local."""
    manager = _ragas(request)
    answer_source_id = manager.connection_answer_source(source_id)
    try:
        profile = manager.remove_connection(source_id)
    except RagasError as exc:
        raise AppError(
            code="RAGAS_CONNECTION_INVALID", message=str(exc), status_code=400
        ) from exc
    if answer_source_id:
        try:
            services = cast(Any, request.app.state.services)
            await services.answer_model_manager.disconnect(
                answer_source_id, user_id=RAGAS_ANSWER_USER_ID
            )
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            logger.warning("ragas connection answer-model cleanup failed: %s", exc)
    return profile


@router.get("/ragas/benchmark-questions")
async def ragas_benchmark_questions(
    settings: SettingsDependency,
) -> dict[str, Any]:
    """List the benchmark dataset's questions so the UI can pick one directly."""
    path = settings.ragas.benchmark_path
    questions: list[dict[str, str | None]] = []
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                questions.append(
                    {
                        "id": entry.get("id"),
                        "question": entry.get("question"),
                        "priority": entry.get("priority"),
                        "category": entry.get("category"),
                    }
                )
    return {"questions": questions}


def _chat_request(
    query: str,
    knowledge_base_ids: list[str],
    answer_model: AnswerModelSelectionPayload | None,
) -> ChatCompletionRequest:
    """Chat pipeline payload; ``answer_model`` selects the answering model."""
    model = None
    if answer_model is not None:
        model = AnswerModelSelection(
            source_id=answer_model.source_id,
            model_name=answer_model.model_name,
        )
    return ChatCompletionRequest(
        query=query,
        knowledge_base_ids=knowledge_base_ids,
        model=model,
    )


async def _build_question_case(
    request: Request,
    *,
    question: str,
    ground_truth: str | None,
    knowledge_base_ids: list[str],
    answer_model: AnswerModelSelectionPayload | None,
) -> list[dict[str, Any]]:
    """Answer a user-specified question through the chat pipeline and wrap the
    result (answer + citation quotes as contexts) into one RAGAS case.

    ``answer_model`` selects which chat answer model answers (same catalog as
    the chat page); when omitted the configured default answer model is used.
    """
    if not knowledge_base_ids:
        raise RagasError("请先选择知识库")
    pipeline = cast(RagPipeline, request.app.state.rag_pipeline)
    payload = _chat_request(question, knowledge_base_ids, answer_model)
    try:
        result = await pipeline.run(payload, user_id=RAGAS_ANSWER_USER_ID)
    except Exception as exc:  # noqa: BLE001 - surface the real failure reason
        logger.warning("ragas question-mode chat failed: %s", exc)
        raise RagasError(f"智能问答链路失败：{type(exc).__name__} {exc}") from exc
    return build_question_case(
        question,
        result.answer,
        [citation.quote for citation in result.citations],
        ground_truth,
    )


@router.post("/ragas/run")
async def ragas_run(
    request: Request,
    settings: SettingsDependency,
    payload: RagasRunRequest,
) -> dict[str, str]:
    """Start a RAGAS evaluation on a specified question or a benchmark question."""
    if not settings.ragas.enabled:
        raise AppError(code="RAGAS_DISABLED", message="RAGAS is disabled.", status_code=503)
    cases: list[dict[str, Any]] | None = None
    try:
        if payload.mode == "question":
            question = (payload.question or "").strip()
            if not question:
                raise RagasError("请填写要评测的问题")
            cases = await _build_question_case(
                request,
                question=question,
                ground_truth=payload.ground_truth,
                knowledge_base_ids=payload.knowledge_base_ids,
                answer_model=payload.answer_model,
            )
        task_id = await _ragas(request).start(
            mode=payload.mode,
            limit=payload.limit,
            monitor=_monitor(request),
            reader=EvaluationResultsReader(settings.evaluation.database_path),
            cases=cases,
            question_id=payload.question_id,
            question_ids=payload.question_ids,
        )
    except RagasError as exc:
        raise AppError(code="RAGAS_RUN_FAILED", message=str(exc), status_code=400) from exc
    return {"task_id": task_id}


@router.get("/ragas/tasks/{task_id}")
async def ragas_task(task_id: str, request: Request) -> dict[str, Any]:
    """Progress and result of a RAGAS task."""
    task = _ragas(request).get_task(task_id)
    if task is None:
        raise AppError(
            code="RAGAS_TASK_NOT_FOUND",
            message="RAGAS task not found.",
            status_code=404,
        )
    return task


@router.post("/ragas/cancel")
async def ragas_cancel(request: Request) -> dict[str, bool]:
    """Request cancellation of the running RAGAS task (takes effect between cases)."""
    return {"cancelled": await _ragas(request).cancel()}
