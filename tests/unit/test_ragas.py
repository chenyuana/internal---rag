from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

from app.core.config import RagasSettings, Settings
from app.services.evaluation_results import EvaluationResultsReader
from app.services.performance_monitor import PerformanceMonitor
from app.services.ragas_evaluator import (
    RagasError,
    RagasTaskManager,
    _aggregate,
    _clean_scores,
    _make_embeddings,
    _make_llm,
    _metric_names,
    build_benchmark_cases,
    build_question_case,
    build_recent_cases,
    mask_api_key,
)


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
                "latest",
                "2026-08-26T00:00:00Z",
                "hash",
                "fingerprint",
                "manuals",
                "run",
                "abc123",
                "{}",
                "{}",
            ),
        )
        for index, question in enumerate(["问题一", "问题二"]):
            connection.execute(
                "INSERT INTO question_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "latest",
                    f"case-{index}",
                    "factual",
                    "P0",
                    1,
                    question,
                    f"答案{index + 1}",
                    "passed",
                    "{}",
                    "{}",
                    "{}",
                    "{}",
                    None,
                ),
            )


class TestBuildCases:
    def test_recent_cases_uses_successful_requests(self) -> None:
        monitor = PerformanceMonitor()
        monitor.record(
            request_id="ok-1",
            question="问题A",
            ok=True,
            error_type=None,
            status="ANSWERABLE",
            latency_ms=100,
            answer="回答A",
            contexts=["证据1", "证据2"],
        )
        monitor.record(
            request_id="fail-1",
            question="问题B",
            ok=False,
            error_type="TIMEOUT",
            status=None,
            latency_ms=100,
            answer="回答B",
            contexts=["证据3"],
        )
        cases = build_recent_cases(monitor, limit=10)
        assert len(cases) == 1
        assert cases[0]["id"] == "ok-1"
        assert cases[0]["answer"] == "回答A"
        assert cases[0]["contexts"] == ["证据1", "证据2"]
        assert cases[0]["ground_truth"] is None

    def test_benchmark_cases_pairs_dataset_with_latest_run(self, tmp_path: Path) -> None:
        benchmark = tmp_path / "benchmark.jsonl"
        benchmark.write_text(
            "\n".join(
                json.dumps(entry, ensure_ascii=False)
                for entry in [
                    {
                        "id": "b1",
                        "question": "问题一",
                        "reference_answer": "标准答案1",
                        "evidence_texts": ["证据A", "证据B"],
                    },
                    {
                        "id": "b2",
                        "question": "问题二",
                        "reference_answer": "标准答案2",
                        "evidence_texts": ["证据C"],
                    },
                    {"id": "b3", "question": "没有答案的题", "evidence_texts": []},
                ]
            ),
            encoding="utf-8",
        )
        db_path = tmp_path / "runs.db"
        _create_store(db_path)
        settings = RagasSettings(benchmark_path=benchmark)

        cases = build_benchmark_cases(settings, EvaluationResultsReader(db_path), limit=10)

        assert len(cases) == 2  # b3 has no stored answer -> skipped
        assert cases[0]["question"] == "问题一"
        assert cases[0]["answer"] == "答案1"
        assert cases[0]["ground_truth"] == "标准答案1"
        assert cases[0]["contexts"] == ["证据A", "证据B"]

    def test_benchmark_cases_respects_limit(self, tmp_path: Path) -> None:
        benchmark = tmp_path / "benchmark.jsonl"
        benchmark.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"b{i}",
                        "question": f"问题{'一二三'[i]}",
                        "reference_answer": "标准",
                        "evidence_texts": ["证据"],
                    },
                    ensure_ascii=False,
                )
                for i in range(3)
            ),
            encoding="utf-8",
        )
        db_path = tmp_path / "runs.db"
        _create_store(db_path)
        # 问题三 needs a stored answer too, so the limit actually cuts.
        with sqlite3.connect(db_path) as connection:
            connection.execute(
                "INSERT INTO question_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    "latest",
                    "case-x",
                    "factual",
                    "P0",
                    1,
                    "问题三",
                    "答案3",
                    "passed",
                    "{}",
                    "{}",
                    "{}",
                    "{}",
                    None,
                ),
            )
        cases = build_benchmark_cases(
            RagasSettings(benchmark_path=benchmark),
            EvaluationResultsReader(db_path),
            limit=2,
        )
        assert len(cases) == 2
        assert cases[0]["question"] == "问题一"

    def test_benchmark_cases_select_single_question(self, tmp_path: Path) -> None:
        benchmark = tmp_path / "benchmark.jsonl"
        benchmark.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"b{i}",
                        "question": f"问题{'一二三'[i]}",
                        "reference_answer": "标准",
                        "evidence_texts": ["证据"],
                    },
                    ensure_ascii=False,
                )
                for i in range(3)
            ),
            encoding="utf-8",
        )
        db_path = tmp_path / "runs.db"
        _create_store(db_path)
        cases = build_benchmark_cases(
            RagasSettings(benchmark_path=benchmark),
            EvaluationResultsReader(db_path),
            limit=10,
            question_id="b1",
        )
        assert len(cases) == 1
        assert cases[0]["id"] == "b1"
        assert cases[0]["question"] == "问题二"
        assert cases[0]["answer"] == "答案2"

    def test_benchmark_cases_select_multiple_questions(self, tmp_path: Path) -> None:
        benchmark = tmp_path / "benchmark.jsonl"
        benchmark.write_text(
            "\n".join(
                json.dumps(
                    {
                        "id": f"b{i}",
                        "question": f"问题{'一二三'[i]}",
                        "reference_answer": "标准",
                        "evidence_texts": ["证据"],
                    },
                    ensure_ascii=False,
                )
                for i in range(3)
            ),
            encoding="utf-8",
        )
        db_path = tmp_path / "runs.db"
        _create_store(db_path)
        cases = build_benchmark_cases(
            RagasSettings(benchmark_path=benchmark),
            EvaluationResultsReader(db_path),
            limit=10,
            question_ids=["b0", "b1"],
        )
        assert [case["id"] for case in cases] == ["b0", "b1"]

    def test_build_question_case(self) -> None:
        cases = build_question_case(
            "问题X", "回答X", ["证据1", "", "证据2"], "标准答案X"
        )
        assert len(cases) == 1
        assert cases[0]["question"] == "问题X"
        assert cases[0]["contexts"] == ["证据1", "证据2"]
        assert cases[0]["ground_truth"] == "标准答案X"

    def test_build_question_case_requires_answer_and_contexts(self) -> None:
        with pytest.raises(RagasError, match="无法评测"):
            build_question_case("问题", None, ["证据"], None)
        with pytest.raises(RagasError, match="无法评测"):
            build_question_case("问题", "回答", [], None)

    def test_build_question_case_blank_ground_truth_becomes_none(self) -> None:
        cases = build_question_case("问题", "回答", ["证据"], "   ")
        assert cases[0]["ground_truth"] is None


class TestScoreHelpers:
    def test_metric_selection_without_ground_truth(self) -> None:
        """ragas 0.2.15's context_precision also requires the reference column,
        so no-ground-truth runs must only evaluate faithfulness + answer_relevancy."""
        assert _metric_names(has_ground_truth=False) == ["faithfulness", "answer_relevancy"]
        assert _metric_names(has_ground_truth=True) == [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        assert "context_precision" not in _metric_names(has_ground_truth=False)

    def test_clean_scores_normalises_nan(self) -> None:
        scores = _clean_scores(
            {
                "question": "q",
                "answer": "a",
                "contexts": ["c"],
                "faithfulness": 0.8,
                "answer_relevancy": float("nan"),
                "ignored_column": "x",
            }
        )
        assert scores["faithfulness"] == 0.8
        assert scores["answer_relevancy"] is None
        assert "ignored_column" not in scores
        assert "question" not in scores

    def test_aggregate_skips_none(self) -> None:
        aggregates = _aggregate(
            [
                {"scores": {"faithfulness": 0.5, "answer_relevancy": 1.0}},
                {"scores": {"faithfulness": 0.7, "answer_relevancy": None}},
            ]
        )
        assert aggregates["faithfulness"] == {"mean": 0.6, "count": 2}
        assert aggregates["answer_relevancy"] == {"mean": 1.0, "count": 1}
        assert aggregates["context_recall"]["count"] == 0


class TestBackendSelection:
    """Judge LLM / embeddings come from the API when configured, else local Ollama."""

    def test_defaults_use_local_ollama(self) -> None:
        from langchain_ollama import ChatOllama, OllamaEmbeddings

        settings = RagasSettings()
        assert settings.llm_backend == "local"
        assert isinstance(_make_llm(settings).langchain_llm, ChatOllama)
        assert isinstance(_make_embeddings(settings).embeddings, OllamaEmbeddings)

    def test_api_model_switches_judge_to_openai_compatible(self) -> None:
        from langchain_openai import ChatOpenAI

        settings = RagasSettings(
            api_base_url="https://api.example.com/v1",
            api_key="secret",
            api_model="judge-model",
        )
        assert settings.llm_backend == "api"
        assert isinstance(_make_llm(settings).langchain_llm, ChatOpenAI)

    def test_api_embedding_model_switches_embeddings(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        settings = RagasSettings(
            api_base_url="https://api.example.com/v1",
            api_key="secret",
            api_model="judge",
            api_embedding_model="embed-model",
        )
        assert settings.llm_backend == "api"
        assert isinstance(_make_embeddings(settings).embeddings, OpenAIEmbeddings)

    def test_embeddings_follow_local_backend(self) -> None:
        from langchain_ollama import OllamaEmbeddings

        # api_embedding_model configured but no api_model -> active backend is
        # local, so embeddings stay on local Ollama.
        settings = RagasSettings(api_embedding_model="embed-model")
        assert settings.llm_backend == "local"
        assert isinstance(_make_embeddings(settings).embeddings, OllamaEmbeddings)

    def test_api_key_never_leaks_in_public_view(
        self, settings: Settings
    ) -> None:
        settings.ragas.api_key = "super-secret-key"
        public = settings.public_view()["ragas"]
        assert isinstance(public, dict)
        assert "api_key" not in public
        assert "super-secret-key" not in str(public)


class TestRagasTaskManager:
    async def test_start_rejects_when_ragas_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("app.services.ragas_evaluator.is_installed", lambda: False)
        manager = RagasTaskManager(RagasSettings())
        try:
            await manager.start(
                mode="recent",
                limit=5,
                monitor=PerformanceMonitor(),
                reader=EvaluationResultsReader(Path("missing.db")),
            )
        except Exception as exc:  # noqa: BLE001
            assert "RAGAS is not installed" in str(exc)
        else:
            raise AssertionError("expected RagasError")

    async def test_start_rejects_no_cases(self) -> None:
        manager = RagasTaskManager(RagasSettings())
        try:
            await manager.start(
                mode="benchmark",
                limit=5,
                monitor=PerformanceMonitor(),
                reader=EvaluationResultsReader(Path("missing.db")),
            )
        except Exception as exc:  # noqa: BLE001
            assert "no evaluation runs stored" in str(exc)
        else:
            raise AssertionError("expected RagasError")

    async def test_start_rejects_unknown_mode(self) -> None:
        manager = RagasTaskManager(RagasSettings())
        try:
            await manager.start(
                mode="recent",
                limit=5,
                monitor=PerformanceMonitor(),
                reader=EvaluationResultsReader(Path("missing.db")),
            )
        except Exception as exc:  # noqa: BLE001
            assert "unknown RAGAS mode" in str(exc)
        else:
            raise AssertionError("expected RagasError")

    async def test_start_question_mode_requires_explicit_cases(self) -> None:
        manager = RagasTaskManager(RagasSettings())
        try:
            await manager.start(
                mode="question",
                limit=5,
                monitor=PerformanceMonitor(),
                reader=EvaluationResultsReader(Path("missing.db")),
            )
        except Exception as exc:  # noqa: BLE001
            assert "requires explicit cases" in str(exc)
        else:
            raise AssertionError("expected RagasError")

    async def test_unknown_task_returns_none(self) -> None:
        manager = RagasTaskManager(RagasSettings())
        assert manager.get_task("nope") is None
        assert await manager.cancel("nope") is False


class TestRagasProfile:
    """Runtime model sources and selection without restarting the gateway."""

    def _manager(self, env_path: Path | None = None) -> RagasTaskManager:
        return RagasTaskManager(RagasSettings(), env_path=env_path or Path("unused"))

    def test_mask_api_key(self) -> None:
        assert mask_api_key(None) is None
        assert mask_api_key("short") == "****"
        assert mask_api_key("sk-abcdefgh12345678") == "sk-****5678"

    def test_seed_connection_from_settings(self) -> None:
        manager = RagasTaskManager(
            RagasSettings(
                api_base_url="https://api.deepseek.com",
                api_key="sk-secret-12345678",
                api_model="deepseek-chat",
            )
        )
        profile = manager.profile()
        assert profile["active"]["source_id"] == "default"
        assert profile["active"]["model"] == "deepseek-chat"
        sources = {source["source_id"]: source for source in profile["sources"]}
        assert "default" in sources
        assert sources["default"]["models"] == ["deepseek-chat"]
        assert sources["default"]["api_key_masked"] == "sk-****5678"
        assert "sk-secret-12345678" not in json.dumps(profile)

    def test_select_local_model(self) -> None:
        profile = self._manager().select_model(
            source_id="local", model="qwen3:4b-test"
        )
        assert profile["active"]["source_id"] == "local"
        assert profile["active"]["model"] == "qwen3:4b-test"
        assert profile["active"]["kind"] == "local"

    def test_select_model_from_api_source(self) -> None:
        manager = self._manager()
        profile = manager.add_connection(
            name="Test API", base_url="https://x/v1", api_key="sk-key-12345678",
            models=["model-a", "model-b"],
        )
        source_id = profile["active"]["source_id"]
        profile = manager.select_model(source_id=source_id, model="model-b")
        assert profile["active"]["source_id"] == source_id
        assert profile["active"]["model"] == "model-b"
        assert profile["active"]["kind"] == "api"

    def test_select_unknown_source_rejected(self) -> None:
        with pytest.raises(RagasError, match="unknown model source"):
            self._manager().select_model(source_id="nope", model="x")

    def test_add_connection_auto_selects_first_model(self) -> None:
        profile = self._manager().add_connection(
            name="DeepSeek", base_url="https://api.deepseek.com",
            api_key="sk-key-12345678", models=["deepseek-chat", "deepseek-reasoner"],
        )
        assert profile["active"]["source_id"].startswith("conn-")
        assert profile["active"]["model"] == "deepseek-chat"
        api_sources = [s for s in profile["sources"] if s["source_type"] == "api"]
        assert len(api_sources) == 1
        assert api_sources[0]["models"] == ["deepseek-chat", "deepseek-reasoner"]
        assert api_sources[0]["api_key_masked"] == "sk-****5678"
        assert "sk-key-12345678" not in json.dumps(profile)

    def test_add_connection_requires_models(self) -> None:
        with pytest.raises(RagasError, match="没有可用模型"):
            self._manager().add_connection(
                name="X", base_url="https://x", api_key="", models=[]
            )

    def test_remove_connection_falls_back_to_local(self) -> None:
        manager = self._manager()
        profile = manager.add_connection(
            name="X", base_url="https://x", api_key="sk-x", models=["m1"],
        )
        source_id = profile["active"]["source_id"]
        profile = manager.remove_connection(source_id)
        assert profile["active"]["source_id"] == "local"
        assert profile["active"]["model"] == "qwen3:4b-instruct-2507-q4_K_M"
        assert not [s for s in profile["sources"] if s["source_type"] == "api"]

    def test_remove_local_source_rejected(self) -> None:
        with pytest.raises(RagasError, match="local source"):
            self._manager().remove_connection("local")

    def test_persist_writes_and_replaces_env_lines(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text("APP_ENV=development\nRAGAS_API_MODEL=old-model\n", encoding="utf-8")
        manager = RagasTaskManager(RagasSettings(), env_path=env_path)
        profile = manager.add_connection(
            name="X", base_url="https://x", api_key="sk-secret-value",
            models=["deepseek-chat"],
        )
        manager.select_model(
            source_id=profile["active"]["source_id"],
            model="deepseek-chat",
            persist=True,
        )
        content = env_path.read_text(encoding="utf-8")
        assert "APP_ENV=development" in content  # unrelated lines untouched
        assert "RAGAS_API_MODEL=deepseek-chat" in content  # replaced, not duplicated
        assert content.count("RAGAS_API_MODEL=") == 1
        assert "RAGAS_API_KEY=sk-secret-value" in content
        assert "RAGAS_JUDGE_MODEL=" in content

    def test_persist_local_clears_api_lines(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        env_path.write_text(
            "RAGAS_API_MODEL=deepseek-chat\nRAGAS_API_KEY=sk-old\n",
            encoding="utf-8",
        )
        manager = RagasTaskManager(
            RagasSettings(
                api_base_url="https://x", api_key="sk-old", api_model="deepseek-chat",
            ),
            env_path=env_path,
        )
        manager.select_model(source_id="local", model="qwen3:4b-test", persist=True)
        content = env_path.read_text(encoding="utf-8")
        assert "RAGAS_API_MODEL=" in content  # present but empty
        assert "RAGAS_API_MODEL=deepseek-chat" not in content
        assert "RAGAS_API_KEY=" in content
        assert "RAGAS_API_KEY=sk-old" not in content

    def test_persist_creates_env_if_missing(self, tmp_path: Path) -> None:
        env_path = tmp_path / ".env"
        manager = RagasTaskManager(RagasSettings(), env_path=env_path)
        manager.persist_profile()
        content = env_path.read_text(encoding="utf-8")
        assert "RAGAS_JUDGE_MODEL=qwen3:4b-instruct-2507-q4_K_M" in content

    def test_all_nan_scores_surface_diagnostic(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """raise_exceptions=False turns per-metric API failures into NaN; the
        case must surface a diagnostic instead of a silent blank row."""
        from app.services import ragas_evaluator as module

        class FakeResult:
            class FakeRow:
                def to_dict(self) -> dict[str, float]:
                    return {
                        "faithfulness": float("nan"),
                        "answer_relevancy": float("nan"),
                        "context_precision": float("nan"),
                    }

            class FakePandas:
                @property
                def iloc(self) -> list[FakeResult.FakeRow]:
                    return [FakeResult.FakeRow()]

            def to_pandas(self) -> FakeResult.FakePandas:
                return self.FakePandas()

        monkeypatch.setattr(module, "_call_evaluate", lambda *args, **kwargs: FakeResult())
        cases = [
            {"id": "c1", "question": "q", "contexts": ["c"], "answer": "a", "ground_truth": "g"}
        ]
        result = module.evaluate_cases(
            cases, settings=RagasSettings(), on_progress=lambda current, total: None
        )
        case = result["cases"][0]
        assert case["scores"] == {
            "faithfulness": None,
            "answer_relevancy": None,
            "context_precision": None,
        }
        assert case["error"] and "未产出分数" in case["error"]


class TestRagasApi:
    def test_status_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/monitor/ragas/status")
        assert response.status_code == 200
        payload = response.json()
        assert "installed" in payload
        assert payload["running"] is False
        assert payload["judge_model"]

    def test_run_rejects_question_mode_without_question(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/monitor/ragas/run",
            json={"mode": "question", "question": "", "knowledge_base_ids": ["kb"]},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "RAGAS_RUN_FAILED"
        assert "问题" in response.json()["error"]["message"]

    def test_run_rejects_question_mode_without_kb(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/monitor/ragas/run",
            json={"mode": "question", "question": "测试问题", "knowledge_base_ids": []},
        )
        assert response.status_code == 400
        assert "知识库" in response.json()["error"]["message"]

    def test_benchmark_questions_endpoint(self, client: TestClient) -> None:
        response = client.get("/api/v1/monitor/ragas/benchmark-questions")
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload["questions"], list)
        if payload["questions"]:
            assert payload["questions"][0]["id"]
            assert payload["questions"][0]["question"]

    def test_run_accepts_multiple_benchmark_question_ids(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/monitor/ragas/run",
            json={"mode": "benchmark", "question_ids": ["missing-a", "missing-b"]},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "RAGAS_RUN_FAILED"

    def test_chat_request_carries_answer_model(self) -> None:
        from app.api.v1.monitor import AnswerModelSelectionPayload, _chat_request

        request = _chat_request(
            "问题",
            ["kb-1"],
            AnswerModelSelectionPayload(source_id="configured", model_name="qwen3.5:9b"),
        )
        assert request.model is not None
        assert request.model.source_id == "configured"
        assert request.model.model_name == "qwen3.5:9b"

        plain = _chat_request("问题", ["kb-1"], None)
        assert plain.model is None

    def test_run_accepts_answer_model_payload(self, client: TestClient) -> None:
        # answer_model parses fine; the run then fails at the chat pipeline
        # (services disabled in the test env), proving the payload was accepted.
        response = client.post(
            "/api/v1/monitor/ragas/run",
            json={
                "mode": "question",
                "question": "测试问题",
                "knowledge_base_ids": ["kb-1"],
                "answer_model": {"source_id": "configured", "model_name": "qwen3.5:9b"},
            },
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "RAGAS_RUN_FAILED"

    def test_task_not_found(self, client: TestClient) -> None:
        response = client.get("/api/v1/monitor/ragas/tasks/does-not-exist")
        assert response.status_code == 404

    def test_cancel_without_running_task(self, client: TestClient) -> None:
        response = client.post("/api/v1/monitor/ragas/cancel")
        assert response.status_code == 200
        assert response.json()["cancelled"] is False

    def test_profile_get_redacts_key(self, client: TestClient) -> None:
        response = client.get("/api/v1/monitor/ragas/profile")
        assert response.status_code == 200
        payload = response.json()
        assert isinstance(payload["ollama_models"], list)
        assert payload["sources"][0]["source_id"] == "local"
        for source in payload["sources"]:
            assert "api_key" not in source  # raw key field never exposed
            if source["source_type"] == "api":
                assert source["api_key_masked"]

    def test_profile_update_via_api(self, client: TestClient) -> None:
        # The test fixture seeds the "default" connection from the real .env.
        response = client.put(
            "/api/v1/monitor/ragas/profile",
            json={"source_id": "default", "model": "deepseek-chat"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["active"]["source_id"] == "default"
        assert payload["active"]["model"] == "deepseek-chat"

    def test_profile_update_rejects_unknown_source(
        self, client: TestClient
    ) -> None:
        response = client.put(
            "/api/v1/monitor/ragas/profile",
            json={"source_id": "nope", "model": "x"},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "RAGAS_PROFILE_INVALID"

    def test_profile_update_switches_back_to_local(self, client: TestClient) -> None:
        response = client.put(
            "/api/v1/monitor/ragas/profile",
            json={"source_id": "local", "model": "qwen3:4b-test"},
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["active"]["source_id"] == "local"
        assert payload["active"]["model"] == "qwen3:4b-test"

    def test_connections_create_with_manual_models(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = cast(Any, client).app.state.services.answer_model_manager

        async def fake_connect(request: Any, *, user_id: str) -> dict[str, Any]:
            return {"source_id": "api-fake123", "name": request.name}

        monkeypatch.setattr(manager, "connect", fake_connect)
        response = client.post(
            "/api/v1/monitor/ragas/connections",
            json={
                "name": "测试连接",
                "base_url": "https://api.example.com/v1",
                "api_key": "sk-abcdefgh12345678",
                "models": ["judge-a", "judge-b"],
            },
        )
        assert response.status_code == 200
        payload = response.json()
        new_sources = [
            s for s in payload["sources"]
            if s["source_type"] == "api" and s["models"] == ["judge-a", "judge-b"]
        ]
        assert len(new_sources) == 1
        assert new_sources[0]["name"] == "测试连接"
        assert payload["active"]["model"] == "judge-a"  # auto-selected
        assert "abcdefgh12345678" not in json.dumps(payload)

    def test_connections_create_fetch_failure_without_models(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from app.api.v1 import monitor as monitor_module

        async def fail_fetch(base_url: str, api_key: str) -> list[str]:
            raise RagasError("could not fetch models")

        monkeypatch.setattr(monitor_module, "_api_models", fail_fetch)
        response = client.post(
            "/api/v1/monitor/ragas/connections",
            json={"name": "X", "base_url": "https://x/v1", "api_key": ""},
        )
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "RAGAS_CONNECTION_INVALID"

    def test_connections_delete_and_fallback_to_local(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = cast(Any, client).app.state.services.answer_model_manager

        async def fake_connect(request: Any, *, user_id: str) -> dict[str, Any]:
            return {"source_id": "api-fake123", "name": request.name}

        async def fake_disconnect(source_id: str, *, user_id: str) -> None:
            return None

        monkeypatch.setattr(manager, "connect", fake_connect)
        monkeypatch.setattr(manager, "disconnect", fake_disconnect)
        created = client.post(
            "/api/v1/monitor/ragas/connections",
            json={
                "name": "X",
                "base_url": "https://x/v1",
                "api_key": "sk-x",
                "models": ["m1"],
            },
        ).json()
        source_id = created["active"]["source_id"]
        response = client.delete(f"/api/v1/monitor/ragas/connections/{source_id}")
        assert response.status_code == 200
        payload = response.json()
        assert payload["active"]["source_id"] == "local"
        assert not [s for s in payload["sources"] if s["source_id"] == source_id]

    def test_connections_delete_local_rejected(self, client: TestClient) -> None:
        response = client.delete("/api/v1/monitor/ragas/connections/local")
        assert response.status_code == 400
        assert response.json()["error"]["code"] == "RAGAS_CONNECTION_INVALID"
