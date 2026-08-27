"""RAGAS calibration evaluation, exposed from the live monitor page.

Runs the RAGAS metric suite (faithfulness / answer_relevancy / context_precision /
context_recall) over a set of Q&A cases using the local Ollama models, as an
external cross-check for the project's own deterministic and LLM-judge metrics.

Two case sources are supported:

- ``recent``: the last successful chat requests recorded by the live monitor
  (answer + citation quotes as contexts; no ground truth, so ``context_recall``
  is skipped);
- ``benchmark``: the sibling rag-evaluation benchmark dataset paired with the
  stored answers from the latest evaluation run (full 4-metric evaluation).

RAGAS itself is imported lazily so the gateway boots and serves normally even
when the package is not installed.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Intranet system: never send analytics/tracking to ragas' external service.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

from app.core.config import PROJECT_ROOT, RagasSettings  # noqa: E402
from app.services.evaluation_results import EvaluationResultsReader  # noqa: E402
from app.services.performance_monitor import PerformanceMonitor  # noqa: E402

logger = logging.getLogger(__name__)

# Metric name -> ragas.metrics attribute. context_recall additionally needs
# ground_truth on the case.
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
ALL_METRICS = list(METRIC_NAMES)
# ragas 0.2.15 requires the `reference` (ground truth) column for BOTH
# context_precision AND context_recall, so without a ground truth only the
# faithfulness / answer_relevancy pair can run.
NO_GROUND_TRUTH_METRICS = ["faithfulness", "answer_relevancy"]


def _metric_names(has_ground_truth: bool) -> list[str]:
    return ALL_METRICS if has_ground_truth else NO_GROUND_TRUTH_METRICS


def mask_api_key(api_key: str | None) -> str | None:
    """Mask a secret key for display: keep the first 3 and last 4 characters."""
    if not api_key:
        return None
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:3]}****{api_key[-4:]}"


class RagasError(RuntimeError):
    """RAGAS evaluation could not be started or completed."""


def is_installed() -> bool:
    try:
        import ragas  # noqa: F401

        return True
    except ImportError:
        return False


def ragas_version() -> str | None:
    try:
        import ragas

        return getattr(ragas, "__version__", None)
    except ImportError:
        return None


def _make_llm(settings: RagasSettings) -> Any:
    """Judge LLM: OpenAI-compatible API when the active backend is api, else local."""
    if settings.llm_backend == "api" and settings.api_model:
        from langchain_openai import ChatOpenAI

        chat: Any = ChatOpenAI(
            model=settings.api_model,
            api_key=settings.api_key or "not-set",
            base_url=settings.api_base_url or None,
            temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama

        chat = ChatOllama(
            model=settings.judge_model,
            base_url=settings.ollama_base_url,
            temperature=0,
        )
    try:
        from ragas.llms import LangchainLLMWrapper

        return LangchainLLMWrapper(chat)
    except ImportError:  # very old ragas
        return chat


def _make_embeddings(settings: RagasSettings) -> Any:
    """Embeddings: API when configured *and* the active backend is api, else local."""
    if settings.api_embedding_model and settings.llm_backend != "local":
        from langchain_openai import OpenAIEmbeddings

        embeddings: Any = OpenAIEmbeddings(
            model=settings.api_embedding_model,
            openai_api_key=settings.api_key or "not-set",
            openai_api_base=settings.api_base_url or None,
            # langchain-openai defaults to tokenizing inputs (check_embedding_ctx_length)
            # which sends token IDs instead of text — most OpenAI-compatible endpoints
            # (incl. Ollama /v1) reject that with "invalid input type".
            check_embedding_ctx_length=False,
        )
    else:
        from langchain_ollama import OllamaEmbeddings

        embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        return LangchainEmbeddingsWrapper(embeddings)
    except ImportError:
        return embeddings


def _call_evaluate(dataset: Any, metrics: list[Any], llm: Any, embeddings: Any) -> Any:
    """Call ``ragas.evaluate`` with the calling convention of the installed version."""
    import inspect

    from ragas import evaluate

    parameters = inspect.signature(evaluate).parameters
    kwargs: dict[str, Any] = {"dataset": dataset, "metrics": metrics}
    if "llm" in parameters:
        kwargs["llm"] = llm
    if "embeddings" in parameters:
        kwargs["embeddings"] = embeddings
    if "raise_exceptions" in parameters:
        kwargs["raise_exceptions"] = False
    # Retry transient API failures (rate limits, timeouts, 5xx) per metric.
    if "run_config" in parameters:
        try:
            from ragas.run_config import RunConfig

            kwargs["run_config"] = RunConfig(max_retries=3, max_wait=60)
        except ImportError:
            pass
    # ragas 0.3+ removed llm/embeddings kwargs; configure the global config instead.
    if "llm" not in parameters:
        try:
            from ragas.config import GlobalConfig

            GlobalConfig.llm = llm
            GlobalConfig.embeddings = embeddings
        except Exception:  # noqa: BLE001 - best-effort config
            logger.warning("ragas global config could not be set", exc_info=True)
    return evaluate(**kwargs)


def _clean_scores(payload: dict[str, Any]) -> dict[str, float | None]:
    """Extract metric scores from a ragas row, normalising NaN to None."""
    scores: dict[str, float | None] = {}
    for key, value in payload.items():
        if key in {"question", "answer", "contexts", "ground_truth"}:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        scores[key] = None if math.isnan(number) else round(number, 4)
    return scores


def build_recent_cases(
    monitor: PerformanceMonitor, limit: int
) -> list[dict[str, Any]]:
    return [
        {
            "id": case["id"],
            "question": case["question"],
            "contexts": case["contexts"],
            "answer": case["answer"],
            "ground_truth": None,
        }
        for case in monitor.recent_cases(limit=limit)
    ]


def build_question_case(
    question: str,
    answer: str | None,
    citation_quotes: list[str] | None,
    ground_truth: str | None,
) -> list[dict[str, Any]]:
    """Wrap one Q&A pair (answer + citation quotes as contexts) into a RAGAS case."""
    contexts = [quote for quote in (citation_quotes or []) if quote]
    if not answer or not answer.strip() or not contexts:
        raise RagasError("问答未产生回答或引用证据（可能无法回答），无法评测")
    return [
        {
            "id": "custom-question",
            "question": question,
            "contexts": contexts,
            "answer": answer,
            "ground_truth": (ground_truth or "").strip() or None,
        }
    ]


def build_benchmark_cases(
    settings: RagasSettings,
    reader: EvaluationResultsReader,
    limit: int,
    question_id: str | None = None,
    question_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """benchmark.jsonl questions/evidence/ground-truth + answers from latest run.

    ``question_id`` restricts the run to one specific benchmark question;
    when None the first ``limit`` questions with a stored answer are used.
    """
    path = settings.benchmark_path
    if not path.is_file():
        raise RagasError(f"benchmark dataset not found: {path}")
    entries: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RagasError(f"invalid benchmark line: {exc}") from exc
            entries.append(entry)

    runs = reader.list_runs(limit=1)
    if not runs:
        raise RagasError("no evaluation runs stored; run `rag-eval run` first")
    run = reader.load_run(runs[0]["run_id"])
    if not run:
        raise RagasError(f"latest evaluation run {runs[0]['run_id']} could not be loaded")
    answers_by_question = {
        (case.get("question") or "").strip(): case.get("answer") or ""
        for case in run["cases"]
    }

    selected_ids = set(question_ids or ([] if question_id is None else [question_id]))
    cases: list[dict[str, Any]] = []
    for entry in entries:
        if selected_ids and entry.get("id") not in selected_ids:
            continue
        question = (entry.get("question") or "").strip()
        answer = answers_by_question.get(question)
        if not question or not answer:
            continue
        cases.append(
            {
                "id": entry.get("id") or question[:40],
                "question": question,
                "contexts": list(entry.get("evidence_texts") or []),
                "answer": answer,
                "ground_truth": entry.get("reference_answer"),
            }
        )
        if not selected_ids and len(cases) >= limit:
            break
    return cases


def evaluate_cases(
    cases: list[dict[str, Any]],
    *,
    settings: RagasSettings,
    on_progress: Callable[[int, int], None],
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run RAGAS over cases (blocking); returns aggregates + per-case scores."""
    from datasets import Dataset

    metrics = [
        getattr(metrics_module(), name)
        for name in _metric_names(bool(cases[0].get("ground_truth")))
    ]
    llm = _make_llm(settings)
    embeddings = _make_embeddings(settings)

    started = time.time()
    per_case: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        if should_cancel and should_cancel():
            return {
                "cancelled": True,
                "evaluated": len(per_case),
                "total": len(cases),
                "cases": per_case,
                "aggregates": _aggregate(per_case),
                "duration_s": round(time.time() - started, 1),
            }
        row: dict[str, Any] = {
            "question": case["question"],
            "contexts": case["contexts"],
            "answer": case["answer"],
        }
        if case.get("ground_truth"):
            row["ground_truth"] = case["ground_truth"]
        try:
            result = _call_evaluate(Dataset.from_list([row]), metrics, llm, embeddings)
            scores = _clean_scores(result.to_pandas().iloc[0].to_dict())
            error = None
        except Exception as exc:  # noqa: BLE001 - per-case failures are isolated
            logger.warning("ragas case failed: %s", exc)
            scores = {}
            error = str(exc)[:300]
        # raise_exceptions=False turns per-metric API failures into NaN, so a
        # case can end up with no scores at all *and* no exception. Surface that
        # instead of showing a silent blank row.
        if not scores or all(value is None for value in scores.values()):
            error = error or (
                "全部指标未产出分数：底层 LLM/Embedding 调用失败"
                "（检查 API 地址/Key/模型名/额度，或服务限流）。"
            )
        per_case.append(
            {
                "id": case["id"],
                "question": case["question"][:120],
                "scores": scores,
                "error": error,
                "contexts_count": len(case["contexts"]),
                "has_ground_truth": bool(case.get("ground_truth")),
                "answer": case["answer"],
                "ground_truth": case.get("ground_truth"),
                "contexts": case["contexts"],
            }
        )
        on_progress(index, len(cases))

    return {
        "cancelled": False,
        "evaluated": len(per_case),
        "total": len(cases),
        "cases": per_case,
        "aggregates": _aggregate(per_case),
        "duration_s": round(time.time() - started, 1),
    }


def _aggregate(
    per_case: list[dict[str, Any]],
) -> dict[str, dict[str, float | int | None]]:
    aggregates: dict[str, dict[str, float | int | None]] = {}
    for name in METRIC_NAMES:
        values = [
            case["scores"][name]
            for case in per_case
            if name in case["scores"] and case["scores"][name] is not None
        ]
        if not values:
            aggregates[name] = {"mean": None, "count": 0}
            continue
        aggregates[name] = {
            "mean": round(sum(values) / len(values), 4),
            "count": len(values),
        }
    return aggregates


def metrics_module() -> Any:
    from ragas import metrics

    return metrics


@dataclass(slots=True)
class ApiConnection:
    """One OpenAI-compatible API connection with its discovered models."""

    source_id: str
    name: str
    base_url: str
    api_key: str
    models: list[str]
    embedding_model: str = ""
    # Source id in the chat answer-model manager when this connection was also
    # registered there (so its models can answer questions through the chat
    # pipeline). Empty when only used for judging.
    answer_source_id: str = ""


class RagasTaskManager:
    """Runs at most one RAGAS evaluation at a time, with progress/cancel.

    Model management mirrors the chat page: the judge model comes from either
    the local Ollama source or any connected OpenAI-compatible API source, and
    the monitor UI picks freely among all sources' models.  Connections live in
    process memory; ``persist=True`` writes the *active* choice to ``.env`` so
    it survives a restart.
    """

    LOCAL_SOURCE = "local"

    # Keys written to .env when the profile is persisted (kept in sync with the
    # env-injectable defaults in config/default.yaml).
    ENV_KEYS = (
        "RAGAS_JUDGE_MODEL",
        "RAGAS_EMBEDDING_MODEL",
        "RAGAS_OLLAMA_BASE_URL",
        "RAGAS_API_BASE_URL",
        "RAGAS_API_KEY",
        "RAGAS_API_MODEL",
        "RAGAS_API_EMBEDDING_MODEL",
    )

    def __init__(self, settings: RagasSettings, env_path: Path | None = None) -> None:
        self._settings = settings.model_copy(deep=True)
        self._env_path = env_path or (PROJECT_ROOT / ".env")
        self._connections: dict[str, ApiConnection] = {}
        self._active_source = self.LOCAL_SOURCE
        self._active_model = self._settings.judge_model
        # Seed a connection from the persisted config so an .env-configured API
        # is available immediately after a restart.
        if settings.api_model:
            self._connections["default"] = ApiConnection(
                source_id="default",
                name="默认 API 连接",
                base_url=settings.api_base_url,
                api_key=settings.api_key,
                models=[settings.api_model],
                embedding_model=settings.api_embedding_model,
            )
            self._active_source = "default"
            self._active_model = settings.api_model
        self._tasks: dict[str, dict[str, Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._last_result: dict[str, Any] | None = None
        self._last_error: str | None = None

    # --- runtime profile & model sources ------------------------------------

    def _effective_settings(self) -> RagasSettings:
        """Settings view matching the active selection for _make_llm/_make_embeddings."""
        eff = self._settings.model_copy(deep=True)
        if self._active_source == self.LOCAL_SOURCE:
            eff.active_backend = "local"
            eff.judge_model = self._active_model
            eff.api_model = ""
            eff.api_base_url = ""
            eff.api_key = ""
            eff.api_embedding_model = ""
        else:
            conn = self._connections[self._active_source]
            eff.active_backend = "api"
            eff.judge_model = self._active_model
            eff.api_model = self._active_model
            eff.api_base_url = conn.base_url
            eff.api_key = conn.api_key
            eff.api_embedding_model = conn.embedding_model
        return eff

    def profile(self) -> dict[str, Any]:
        """Model sources + active selection for the monitor UI (keys masked).

        The local source's model list is filled in by the API layer from a live
        Ollama ``/api/tags`` probe.
        """
        return {
            "ollama_base_url": self._settings.ollama_base_url,
            "active": {
                "source_id": self._active_source,
                "kind": "local" if self._active_source == self.LOCAL_SOURCE else "api",
                "model": self._active_model,
                "embedding_model": (
                    self._effective_settings().api_embedding_model
                    or self._settings.embedding_model
                ),
            },
            "sources": [
                {
                    "source_id": self.LOCAL_SOURCE,
                    "name": "本地 Ollama",
                    "source_type": "local",
                    "models": [],
                    "removable": False,
                },
                *[
                    {
                        "source_id": connection.source_id,
                        "name": connection.name,
                        "source_type": "api",
                        "models": connection.models,
                        "base_url": connection.base_url,
                        "api_key_masked": mask_api_key(connection.api_key),
                        "removable": True,
                    }
                    for connection in self._connections.values()
                ],
            ],
        }

    def select_model(
        self, source_id: str, model: str, persist: bool = False
    ) -> dict[str, Any]:
        """Make ``model`` from ``source_id`` the active judge model."""
        model = (model or "").strip()
        if not model:
            raise RagasError("model is required")
        if source_id != self.LOCAL_SOURCE and source_id not in self._connections:
            raise RagasError(f"unknown model source: {source_id}")
        self._active_source = source_id
        self._active_model = model
        if persist:
            self.persist_profile()
        return self.profile()

    def add_connection(
        self,
        *,
        name: str,
        base_url: str,
        api_key: str,
        models: list[str],
        embedding_model: str = "",
    ) -> dict[str, Any]:
        """Register an API connection; ``models`` must be provided by the caller
        (fetched from the endpoint's ``/models`` by the API layer, or manual)."""
        name = (name or "").strip() or "API 连接"
        base_url = (base_url or "").strip()
        models = [model for model in (models or []) if (model or "").strip()]
        if not base_url:
            raise RagasError("api_base_url is required")
        if not models:
            raise RagasError(
                "连接没有可用模型：请提供模型列表，或确认端点支持 GET /models"
            )
        source_id = f"conn-{uuid.uuid4().hex[:8]}"
        self._connections[source_id] = ApiConnection(
            source_id=source_id,
            name=name,
            base_url=base_url,
            api_key=api_key or "",
            models=models,
            embedding_model=(embedding_model or "").strip(),
        )
        # Auto-select the connection's first model (the chat page behaves the same).
        self._active_source = source_id
        self._active_model = models[0]
        return self.profile()

    def remove_connection(self, source_id: str) -> dict[str, Any]:
        """Drop an API connection; the active choice falls back to local."""
        if source_id == self.LOCAL_SOURCE:
            raise RagasError("cannot remove the local source")
        if source_id not in self._connections:
            raise RagasError(f"unknown model source: {source_id}")
        del self._connections[source_id]
        if self._active_source == source_id:
            self._active_source = self.LOCAL_SOURCE
            self._active_model = self._settings.judge_model
        return self.profile()

    def connection_answer_source(self, source_id: str) -> str:
        """The chat answer-model source id this connection was registered as."""
        connection = self._connections.get(source_id)
        return connection.answer_source_id if connection else ""

    def set_answer_source(self, source_id: str, answer_source_id: str) -> None:
        """Remember the chat answer-model source id for a RAGAS connection."""
        connection = self._connections.get(source_id)
        if connection is not None:
            connection.answer_source_id = answer_source_id

    def persist_profile(self) -> None:
        """Write the active profile to ``.env`` (replace-or-append the RAGAS_* lines).

        Persists the *effective* state: when the active backend is local, the
        API fields are written empty so a restart reproduces the local choice.
        """
        eff = self._effective_settings()
        api_active = eff.llm_backend == "api"
        values: dict[str, str] = {
            "RAGAS_JUDGE_MODEL": eff.judge_model,
            "RAGAS_EMBEDDING_MODEL": eff.embedding_model,
            "RAGAS_OLLAMA_BASE_URL": eff.ollama_base_url,
            "RAGAS_API_BASE_URL": eff.api_base_url if api_active else "",
            "RAGAS_API_KEY": eff.api_key if api_active else "",
            "RAGAS_API_MODEL": eff.api_model if api_active else "",
            "RAGAS_API_EMBEDDING_MODEL": eff.api_embedding_model if api_active else "",
        }
        try:
            lines = self._env_path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        updated: list[str] = []
        seen: set[str] = set()
        for line in lines:
            key = line.split("=", 1)[0].strip() if "=" in line else ""
            if key in self.ENV_KEYS:
                updated.append(f"{key}={values[key]}")
                seen.add(key)
            else:
                updated.append(line)
        for key in self.ENV_KEYS:
            if key not in seen:
                updated.append(f"{key}={values[key]}")
        self._env_path.write_text("\n".join(updated) + "\n", encoding="utf-8")

    def status(self) -> dict[str, Any]:
        running = next(
            (
                task
                for task in self._tasks.values()
                if task["state"] in {"queued", "running"}
            ),
            None,
        )
        eff = self._effective_settings()
        return {
            "installed": is_installed(),
            "version": ragas_version(),
            "backend": eff.llm_backend,
            "judge_model": eff.judge_model,
            "embedding_model": eff.embedding_model,
            "ollama_base_url": eff.ollama_base_url,
            "api_base_url": eff.api_base_url or None,
            "api_model": eff.api_model or None,
            "api_embedding_model": eff.api_embedding_model or None,
            "api_key_configured": bool(eff.api_key),
            "api_key_masked": mask_api_key(eff.api_key),
            "running": bool(running),
            "running_task_id": running["task_id"] if running else None,
            "last_result": self._last_result,
            "last_error": self._last_error,
        }

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        return self._tasks.get(task_id)

    async def start(
        self,
        *,
        mode: str,
        limit: int,
        monitor: PerformanceMonitor,
        reader: EvaluationResultsReader,
        cases: list[dict[str, Any]] | None = None,
        question_id: str | None = None,
        question_ids: list[str] | None = None,
    ) -> str:
        if not is_installed():
            raise RagasError(
                "RAGAS is not installed in the gateway environment; "
                "run: pip install \"ragas==0.2.15\" \"langchain-ollama==0.3.10\""
            )
        if mode not in {"question", "benchmark"}:
            raise RagasError(f"unknown RAGAS mode: {mode}")
        if any(task["state"] in {"queued", "running"} for task in self._tasks.values()):
            raise RagasError("another RAGAS run is already in progress")

        if cases is None:
            # Benchmark mode builds cases from the dataset + the latest run.
            if mode != "benchmark":
                raise RagasError(f"{mode} mode requires explicit cases")
            cases = build_benchmark_cases(
                self._settings, reader, limit, question_id=question_id, question_ids=question_ids
            )
        if not cases:
            raise RagasError(
                "no evaluable cases: 指定问题需先成功问答并产生证据，"
                "基准集需先运行 rag-eval run 存储答案"
            )

        task_id = uuid.uuid4().hex[:12]
        task: dict[str, Any] = {
            "task_id": task_id,
            "mode": mode,
            "limit": len(cases),
            "state": "queued",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "progress": {"current": 0, "total": len(cases)},
            "result": None,
            "error": None,
        }
        self._tasks[task_id] = task
        self._cancel_events[task_id] = asyncio.Event()
        asyncio.create_task(self._run(task_id, task, cases))
        return task_id

    async def cancel(self, task_id: str | None = None) -> bool:
        """Request cancellation of a running task (takes effect between cases)."""
        target = task_id or next(
            (
                tid
                for tid, task in self._tasks.items()
                if task["state"] in {"queued", "running"}
            ),
            None,
        )
        if target is None or target not in self._cancel_events:
            return False
        self._cancel_events[target].set()
        return True

    async def _run(
        self, task_id: str, task: dict[str, Any], cases: list[dict[str, Any]]
    ) -> None:
        task["state"] = "running"
        task["started_at"] = time.time()
        cancel_event = self._cancel_events[task_id]

        def on_progress(current: int, total: int) -> None:
            task["progress"] = {"current": current, "total": total}

        try:
            effective = self._effective_settings()
            result = await asyncio.to_thread(
                evaluate_cases,
                cases,
                settings=effective,
                on_progress=on_progress,
                should_cancel=cancel_event.is_set,
            )
            task["result"] = result
            task["state"] = "cancelled" if result["cancelled"] else "done"
            if not result["cancelled"]:
                result["model"] = effective.api_model or effective.judge_model
                result["backend"] = effective.llm_backend
                self._last_result = {
                    "task_id": task_id,
                    "mode": task["mode"],
                    "finished_at": time.time(),
                    "backend": effective.llm_backend,
                    "aggregates": result["aggregates"],
                    "metrics": [
                        name
                        for name in METRIC_NAMES
                        if name in result["aggregates"]
                    ],
                    "model": result["model"],
                    "duration_s": result["duration_s"],
                }
        except Exception as exc:  # noqa: BLE001 - surface as task error
            logger.exception("ragas run failed")
            task["state"] = "error"
            task["error"] = str(exc)[:500]
            self._last_error = str(exc)[:500]
        finally:
            task["finished_at"] = time.time()
            self._cancel_events.pop(task_id, None)
