from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.services.performance_monitor import PerformanceMonitor


class TestPerformanceMonitor:
    def test_record_and_stats_aggregation(self) -> None:
        monitor = PerformanceMonitor(max_records=100)
        now = time.time()
        # 3 successful requests with known latencies, 1 failure.
        for i, latency in enumerate([100.0, 200.0, 300.0]):
            monitor.record(
                request_id=f"req-{i}",
                question=f"问题 {i}",
                ok=True,
                error_type=None,
                status="ANSWERABLE",
                latency_ms=latency,
                latencies_ms={"retrieval": 40.0, "generate": latency - 40.0},
                model_calls=2,
                retrieval_calls=1,
                stage_counts={"retrieval": 1, "generate": 1},
            )
        monitor.record(
            request_id="req-fail",
            question="会失败的问题",
            ok=False,
            error_type="UPSTREAM_TIMEOUT",
            status=None,
            latency_ms=5000.0,
        )

        stats = monitor.stats(minutes=0)

        totals = stats["totals"]
        assert totals["requests"] == 4
        assert totals["success"] == 3
        assert totals["failed"] == 1
        assert totals["success_rate"] == 0.75

        latency = stats["latency_ms"]
        # Nearest-rank percentiles over [100, 200, 300, 5000].
        assert latency["p50"] == 200.0
        assert latency["p90"] == 5000.0
        assert latency["p95"] == 5000.0
        assert latency["avg"] == 1400.0

        assert stats["model_calls"]["total"] == 6
        assert stats["retrieval_calls"]["total"] == 3
        assert stats["error_distribution"] == {"UPSTREAM_TIMEOUT": 1}
        assert stats["stage_timings_ms"]["retrieval"] == 40.0

        # Recent list is newest-first.
        assert stats["recent"][0]["request_id"] == "req-fail"
        assert stats["recent"][0]["ok"] is False
        assert stats["recent"][-1]["request_id"] == "req-0"
        assert now <= stats["recent"][0]["ts"] <= time.time()

    def test_empty_stats(self) -> None:
        stats = PerformanceMonitor().stats(minutes=15)
        assert stats["totals"]["requests"] == 0
        assert stats["totals"]["success_rate"] is None
        assert stats["latency_ms"]["p50"] is None
        assert stats["recent"] == []

    def test_window_filters_old_records(self) -> None:
        monitor = PerformanceMonitor()
        monitor.record(
            request_id="old",
            question="旧请求",
            ok=True,
            error_type=None,
            status="ANSWERABLE",
            latency_ms=10.0,
        )
        # Simulate the record being older than the requested window.
        with monitor._lock:  # noqa: SLF001 - test-only access
            monitor._records[0].ts = time.time() - 3600  # noqa: SLF001
        stats = monitor.stats(minutes=15)
        assert stats["totals"]["requests"] == 0


class TestMonitorApi:
    def test_stats_endpoint_returns_schema(self, client: TestClient) -> None:
        response = client.get("/api/v1/monitor/stats")
        assert response.status_code == 200
        payload = response.json()
        assert payload["totals"]["requests"] == 0
        assert set(payload["latency_ms"]) >= {"p50", "p90", "p95", "p99"}
        assert payload["recent"] == []

    def test_services_endpoint_probes_dependencies(
        self, client: TestClient, settings: Settings
    ) -> None:
        response = client.get("/api/v1/monitor/services")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] in {"ready", "not_ready"}
        assert isinstance(payload["dependencies"], list)
        # All external services are disabled in the test fixture.
        assert all(item["status"] == "disabled" for item in payload["dependencies"])

    def test_monitor_page_served(self, client: TestClient) -> None:
        response = client.get("/monitor")
        assert response.status_code == 200
        assert "运行监控" in response.text
        assert "monitor.js" in response.text
        assert 'data-ragas-mode="question"' in response.text
        assert 'id="ragasSettingsDialog"' in response.text
        assert 'id="ragasPersist"' not in response.text
        assert 'id="ragasBenchmarkLimit"' not in response.text
