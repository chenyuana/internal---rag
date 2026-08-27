from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def _create_store(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY, started_at TEXT, dataset_sha256 TEXT,
                config_fingerprint TEXT, knowledge_base TEXT, mode TEXT,
                git_commit TEXT, version_json TEXT, summary_json TEXT
            );
            CREATE TABLE question_results (
                run_id TEXT NOT NULL, case_id TEXT NOT NULL, category TEXT,
                priority TEXT, ok INTEGER, question TEXT, answer TEXT, status TEXT,
                answer_rules_json TEXT, judge_json TEXT, retrieval_json TEXT,
                record_summary_json TEXT, raw_path TEXT,
                PRIMARY KEY (run_id, case_id)
            );
            """
        )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "candidate",
                "2026-08-24T06:03:51Z",
                "dataset-hash",
                "config-hash",
                "manuals",
                "run",
                "abc123",
                json.dumps({"gateway": "0.1.0"}),
                json.dumps({"total": 1, "passed": 1, "pass_rate": 1.0}),
            ),
        )
        connection.execute(
            "INSERT INTO question_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "candidate",
                "case-1",
                "factual",
                "P0",
                1,
                "测试问题",
                "测试答案",
                "passed",
                json.dumps({"passed": True}),
                json.dumps({"scores": {"correctness": 5}}),
                json.dumps({"recall_at_k": 1.0}),
                json.dumps({"latency_ms": 42}),
                "runs/candidate/case-1.json",
            ),
        )


def _client(settings: Settings, database_path: Path) -> TestClient:
    configured = settings.model_copy(
        update={
            "evaluation": settings.evaluation.model_copy(
                update={"enabled": True, "database_path": database_path}
            )
        }
    )
    return TestClient(create_app(configured))


def test_evaluation_page_is_served(client: TestClient) -> None:
    response = client.get("/evaluation")

    assert response.status_code == 200
    assert 'id="dashboard"' in response.text


def test_list_and_load_evaluation_run(settings: Settings, tmp_path: Path) -> None:
    database_path = tmp_path / "runs.db"
    _create_store(database_path)
    regression_dir = tmp_path / "runs" / "candidate"
    regression_dir.mkdir(parents=True)
    (regression_dir / "regression_report.json").write_text(
        json.dumps({"regression_failed": False, "regressions": []}), encoding="utf-8"
    )

    with _client(settings, database_path) as client:
        listing = client.get("/api/v1/evaluation/runs?limit=10")
        detail = client.get("/api/v1/evaluation/runs/candidate")

    assert listing.status_code == 200
    assert listing.json()[0]["run_id"] == "candidate"
    assert listing.json()[0]["regression"]["regression_failed"] is False
    assert detail.status_code == 200
    assert detail.json()["cases"][0]["case_id"] == "case-1"
    assert detail.json()["cases"][0]["judge"]["scores"]["correctness"] == 5


def test_missing_evaluation_store_returns_empty_list(settings: Settings, tmp_path: Path) -> None:
    with _client(settings, tmp_path / "missing.db") as client:
        response = client.get("/api/v1/evaluation/runs")

    assert response.status_code == 200
    assert response.json() == []
