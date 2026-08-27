"""In-process, real-time performance telemetry for the chat answer pipeline.

Records every `/api/v1/chat/completions` request (status, latency, per-stage
timings, model/retrieval call counts) into a bounded in-memory ring buffer and
aggregates them on demand.  Nothing is persisted: the buffer resets when the
gateway restarts, which is exactly what we want during iterative development.

All access is guarded by a plain lock so the service stays safe even if
FastAPI later runs handlers in a thread pool.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any

# Default ring-buffer capacity (requests kept in memory at most).
MAX_RECORDS = 1000
# Default aggregation window used by the monitor API when no `minutes` is given.
DEFAULT_MINUTES = 15.0
# How many of the most recent records to include in the `recent` payload.
RECENT_LIMIT = 20
# Max question length kept for display (avoid huge payloads).
QUESTION_MAX_CHARS = 120

LATENCY_PERCENTILES = (50.0, 90.0, 95.0, 99.0)


@dataclass(slots=True)
class RequestRecord:
    ts: float
    request_id: str
    question: str
    ok: bool
    error_type: str | None
    status: str | None
    latency_ms: float
    latencies_ms: dict[str, float] | None = None
    model_calls: int = 0
    retrieval_calls: int = 0
    stage_counts: dict[str, int] | None = None
    # Kept for RAGAS calibration evaluation; excluded from the /stats payload.
    answer: str | None = None
    contexts: list[str] = field(default_factory=list)


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    """Nearest-rank percentile; returns None for an empty sample."""
    if not sorted_values:
        return None
    n = len(sorted_values)
    index = max(0, min(n - 1, math.ceil(percentile / 100.0 * n) - 1))
    return sorted_values[index]


class PerformanceMonitor:
    """Bounded ring buffer + on-demand aggregation of request telemetry."""

    def __init__(self, max_records: int = MAX_RECORDS) -> None:
        self._max_records = max_records
        self._records: deque[RequestRecord] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self.started_at = time.time()

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    def record(
        self,
        *,
        request_id: str,
        question: str,
        ok: bool,
        error_type: str | None,
        latency_ms: float,
        status: str | None = None,
        latencies_ms: dict[str, float] | None = None,
        model_calls: int = 0,
        retrieval_calls: int = 0,
        stage_counts: dict[str, int] | None = None,
        answer: str | None = None,
        contexts: list[str] | None = None,
    ) -> None:
        record = RequestRecord(
            ts=time.time(),
            request_id=request_id,
            question=(question or "")[:QUESTION_MAX_CHARS],
            ok=ok,
            error_type=error_type,
            status=status,
            latency_ms=round(latency_ms, 3),
            latencies_ms=dict(latencies_ms) if latencies_ms else None,
            model_calls=int(model_calls),
            retrieval_calls=int(retrieval_calls),
            stage_counts=dict(stage_counts) if stage_counts else None,
            answer=answer,
            contexts=list(contexts) if contexts else [],
        )
        with self._lock:
            self._records.append(record)

    def recent_cases(self, limit: int = 10) -> list[dict[str, Any]]:
        """Most recent successful requests with answer + evidence contexts (for RAGAS)."""
        with self._lock:
            records = [
                r
                for r in self._records
                if r.ok and r.answer and r.contexts
            ][-limit:][::-1]
        return [
            {
                "id": r.request_id,
                "ts": r.ts,
                "question": r.question,
                "answer": r.answer,
                "contexts": r.contexts,
            }
            for r in records
        ]

    def stats(self, minutes: float = DEFAULT_MINUTES) -> dict[str, Any]:
        """Aggregate metrics over the last `minutes` (<=0 means the full buffer)."""
        now = time.time()
        cutoff = 0.0 if minutes <= 0 else now - minutes * 60.0
        with self._lock:
            records = [r for r in self._records if r.ts >= cutoff]
            recent = list(self._records)[-RECENT_LIMIT:][::-1]

        total = len(records)
        ok_count = sum(1 for r in records if r.ok)
        failed_count = total - ok_count

        latencies = sorted(r.latency_ms for r in records)
        percentiles = {
            f"p{int(p)}": _percentile(latencies, p) for p in LATENCY_PERCENTILES
        }

        ok_records = [r for r in records if r.ok]
        model_calls_total = sum(r.model_calls for r in records)
        retrieval_calls_total = sum(r.retrieval_calls for r in records)

        error_distribution: dict[str, int] = dict(
            Counter(r.error_type or "UNKNOWN" for r in records if not r.ok)
        )
        status_distribution: dict[str, int] = dict(
            Counter(r.status or "—" for r in records)
        )

        # Per-stage average latency over successful requests that reported it.
        stage_timings: dict[str, float] = {}
        stage_samples: Counter[str] = Counter()
        stage_sums: dict[str, float] = {}
        stage_counts_aggregate: Counter[str] = Counter()
        for r in ok_records:
            if r.latencies_ms:
                for stage, ms in r.latencies_ms.items():
                    stage_sums[stage] = stage_sums.get(stage, 0.0) + ms
                    stage_samples[stage] += 1
            if r.stage_counts:
                stage_counts_aggregate.update(r.stage_counts)
        for stage in stage_sums:
            stage_timings[stage] = round(stage_sums[stage] / stage_samples[stage], 1)

        return {
            "window_minutes": minutes,
            "effective_seconds": round(now - cutoff, 1) if cutoff else None,
            "first_recorded_at": records[0].ts if records else None,
            "last_recorded_at": records[-1].ts if records else None,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "buffer_capacity": self._max_records,
            "totals": {
                "requests": total,
                "success": ok_count,
                "failed": failed_count,
                "success_rate": round(ok_count / total, 4) if total else None,
            },
            "latency_ms": {
                **percentiles,
                "avg": round(sum(latencies) / len(latencies), 1) if latencies else None,
                "max": latencies[-1] if latencies else None,
            },
            "model_calls": {
                "total": model_calls_total,
                "avg": round(model_calls_total / total, 2) if total else None,
            },
            "retrieval_calls": {
                "total": retrieval_calls_total,
                "avg": round(retrieval_calls_total / total, 2) if total else None,
            },
            "error_distribution": error_distribution,
            "status_distribution": status_distribution,
            "stage_timings_ms": stage_timings,
            "stage_counts": dict(stage_counts_aggregate),
            "recent": [
                {
                    "ts": r.ts,
                    "request_id": r.request_id,
                    "question": r.question,
                    "ok": r.ok,
                    "error_type": r.error_type,
                    "status": r.status,
                    "latency_ms": r.latency_ms,
                    "model_calls": r.model_calls,
                    "retrieval_calls": r.retrieval_calls,
                }
                for r in recent
            ],
        }


# Module-level singleton; the app factory also attaches one to app.state.
_monitor: PerformanceMonitor | None = None


def get_monitor() -> PerformanceMonitor:
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor
