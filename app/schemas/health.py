from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str
    environment: str


class ServiceStatus(BaseModel):
    name: str
    enabled: bool
    required: bool
    status: Literal["ready", "unavailable", "disabled"]
    latency_ms: float | None = None
    detail: str | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    dependencies: list[ServiceStatus]
