from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query

from app.api.dependencies import SettingsDependency
from app.core.exceptions import AppError
from app.services.evaluation_results import EvaluationResultsError, EvaluationResultsReader

router = APIRouter()


def _reader(settings: SettingsDependency) -> EvaluationResultsReader:
    if not settings.evaluation.enabled:
        raise AppError(
            code="EVALUATION_DISABLED",
            message="Evaluation results are disabled.",
            status_code=503,
        )
    return EvaluationResultsReader(settings.evaluation.database_path)


@router.get("/runs", response_model=list[dict[str, Any]])
def list_runs(
    settings: SettingsDependency,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict[str, Any]]:
    try:
        return _reader(settings).list_runs(limit)
    except EvaluationResultsError as exc:
        raise AppError(
            code="EVALUATION_STORE_UNAVAILABLE",
            message="Evaluation results could not be read.",
            status_code=503,
        ) from exc


@router.get("/runs/{run_id}", response_model=dict[str, Any])
def get_run(run_id: str, settings: SettingsDependency) -> dict[str, Any]:
    try:
        result = _reader(settings).load_run(run_id)
    except EvaluationResultsError as exc:
        raise AppError(
            code="EVALUATION_STORE_UNAVAILABLE",
            message="Evaluation results could not be read.",
            status_code=503,
        ) from exc
    if result is None:
        raise AppError(
            code="EVALUATION_RUN_NOT_FOUND",
            message="Evaluation run was not found.",
            status_code=404,
        )
    return result
