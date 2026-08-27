from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class EvaluationResultsError(RuntimeError):
    """The evaluation result store could not be read."""


class EvaluationResultsReader:
    """Small read-only adapter for rag-evaluation's SQLite result store."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @property
    def available(self) -> bool:
        return self.database_path.is_file()

    def _connect(self) -> sqlite3.Connection:
        if not self.available:
            raise EvaluationResultsError(f"evaluation database not found: {self.database_path}")
        try:
            connection = sqlite3.connect(
                f"file:{self.database_path.as_posix()}?mode=ro",
                uri=True,
                timeout=2,
            )
        except sqlite3.Error as exc:
            raise EvaluationResultsError(f"cannot open evaluation database: {exc}") from exc
        connection.row_factory = sqlite3.Row
        return connection

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        if not self.available:
            return []
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    "SELECT run_id, started_at, knowledge_base, mode, git_commit, "
                    "version_json, summary_json FROM runs ORDER BY started_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise EvaluationResultsError(f"cannot list evaluation runs: {exc}") from exc
        return [
            {
                "run_id": row["run_id"],
                "started_at": row["started_at"],
                "knowledge_base": row["knowledge_base"],
                "mode": row["mode"],
                "git_commit": row["git_commit"],
                "version": _load_json(row["version_json"]),
                "summary": _load_json(row["summary_json"]),
                "regression": self._load_regression(row["run_id"]),
            }
            for row in rows
        ]

    def load_run(self, run_id: str) -> dict[str, Any] | None:
        try:
            with self._connect() as connection:
                run = connection.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                if run is None:
                    return None
                cases = connection.execute(
                    "SELECT * FROM question_results WHERE run_id = ? ORDER BY rowid",
                    (run_id,),
                ).fetchall()
        except sqlite3.Error as exc:
            raise EvaluationResultsError(f"cannot load evaluation run: {exc}") from exc
        return {
            "run_id": run["run_id"],
            "started_at": run["started_at"],
            "dataset_sha256": run["dataset_sha256"],
            "config_fingerprint": run["config_fingerprint"],
            "knowledge_base": run["knowledge_base"],
            "mode": run["mode"],
            "git_commit": run["git_commit"],
            "version": _load_json(run["version_json"]),
            "summary": _load_json(run["summary_json"]),
            "regression": self._load_regression(run_id),
            "cases": [_case_payload(row) for row in cases],
        }

    def _load_regression(self, run_id: str) -> dict[str, Any] | None:
        path = self.database_path.parent / "runs" / run_id / "regression_report.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None


def _load_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _case_payload(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "case_id": row["case_id"],
        "category": row["category"],
        "priority": row["priority"],
        "ok": bool(row["ok"]),
        "question": row["question"],
        "answer": row["answer"],
        "status": row["status"],
        "answer_rules": _load_json(row["answer_rules_json"]),
        "judge": _load_json(row["judge_json"]),
        "retrieval": _load_json(row["retrieval_json"]),
        "record_summary": _load_json(row["record_summary_json"]),
        "raw_path": row["raw_path"],
    }
