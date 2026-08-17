from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProbeResult:
    ready: bool
    latency_ms: float | None = None
    detail: str | None = None
