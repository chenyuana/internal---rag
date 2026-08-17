from __future__ import annotations

import asyncio
import math
import time

import httpx

from app.core.config import ModelEndpointSettings
from app.core.exceptions import AppError
from app.schemas.retrieval import RerankResult
from app.services.probe import ProbeResult


class RerankerClient:
    """Adapter for generic rerank APIs and Qwen3 reranker completion scoring."""

    _QWEN3_SYSTEM_PROMPT = (
        "Judge whether the Document meets the requirements based on the Query "
        "and the Instruct provided. Note that the answer can only be "
        '"yes" or "no".'
    )
    _QWEN3_INSTRUCTION = (
        "Given a web search query, retrieve relevant passages that answer the query"
    )

    def __init__(
        self,
        settings: ModelEndpointSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        headers = {}
        api_key = settings.api_key.get_secret_value()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._model_name = settings.model_name
        self._operation_path = settings.operation_path or "/rerank"
        self._client = httpx.AsyncClient(
            base_url=settings.base_url.rstrip("/") + "/",
            headers=headers,
            timeout=settings.timeout_seconds,
            transport=transport,
            trust_env=False,
        )

    async def probe(self) -> ProbeResult:
        started = time.perf_counter()
        try:
            response = await self._client.get("models")
            response.raise_for_status()
            return ProbeResult(
                ready=True,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
            )
        except httpx.HTTPError as exc:
            return ProbeResult(
                ready=False,
                latency_ms=round((time.perf_counter() - started) * 1000, 2),
                detail=type(exc).__name__,
            )

    async def rerank(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        if self._operation_path.rstrip("/") == "/completion":
            return await self._rerank_with_qwen3_completion(
                query=query,
                documents=documents,
                top_n=top_n,
            )

        try:
            response = await self._client.post(
                self._operation_path.lstrip("/"),
                json={
                    "model": self._model_name,
                    "query": query,
                    "documents": documents,
                    "top_n": top_n,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError(
                code="RERANKER_UNAVAILABLE",
                message="The reranker service is unavailable or returned invalid JSON.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc

        raw_results = payload.get("results", payload.get("data", []))
        if not isinstance(raw_results, list):
            raise AppError(
                code="RERANKER_INVALID_RESPONSE",
                message="The reranker response does not contain a result list.",
                status_code=502,
            )
        results: list[RerankResult] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            index = item.get("index", item.get("document_index"))
            score = item.get("relevance_score", item.get("score"))
            if isinstance(index, int) and isinstance(score, int | float):
                results.append(RerankResult(index=index, score=float(score)))
        return results

    async def _rerank_with_qwen3_completion(
        self,
        *,
        query: str,
        documents: list[str],
        top_n: int,
    ) -> list[RerankResult]:
        semaphore = asyncio.Semaphore(4)

        async def score_document(document: str) -> float:
            async with semaphore:
                return await self._score_qwen3_document(query=query, document=document)

        scores = await asyncio.gather(
            *(score_document(document) for document in documents)
        )
        ranked = sorted(
            enumerate(scores),
            key=lambda item: item[1],
            reverse=True,
        )
        limit = max(0, min(top_n, len(ranked)))
        return [
            RerankResult(index=index, score=score)
            for index, score in ranked[:limit]
        ]

    async def _score_qwen3_document(self, *, query: str, document: str) -> float:
        prompt = (
            "<|im_start|>system\n"
            f"{self._QWEN3_SYSTEM_PROMPT}<|im_end|>\n"
            "<|im_start|>user\n"
            f"<Instruct>: {self._QWEN3_INSTRUCTION}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        )
        try:
            response = await self._client.post(
                self._operation_path.lstrip("/"),
                json={
                    "prompt": prompt,
                    "n_predict": 1,
                    "n_probs": 100,
                    "temperature": 1,
                    "top_k": 0,
                    "top_p": 1,
                    "min_p": 0,
                    "grammar": 'root ::= "yes" | "no"',
                    "stream": False,
                    "cache_prompt": True,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AppError(
                code="RERANKER_UNAVAILABLE",
                message="The reranker service is unavailable or returned invalid JSON.",
                status_code=503,
                details={"error_type": type(exc).__name__},
            ) from exc

        probabilities = payload.get("completion_probabilities")
        if not isinstance(probabilities, list) or not probabilities:
            raise AppError(
                code="RERANKER_INVALID_RESPONSE",
                message="The reranker response does not contain token probabilities.",
                status_code=502,
            )
        first_token = probabilities[0]
        if not isinstance(first_token, dict):
            raise AppError(
                code="RERANKER_INVALID_RESPONSE",
                message="The reranker response contains invalid token probabilities.",
                status_code=502,
            )

        top_logprobs = first_token.get("top_logprobs", [])
        yes_logprob: float | None = None
        no_logprob: float | None = None
        if isinstance(top_logprobs, list):
            for item in top_logprobs:
                if not isinstance(item, dict):
                    continue
                token = item.get("token")
                logprob = item.get("logprob")
                if not isinstance(logprob, int | float):
                    continue
                if token == "yes":
                    yes_logprob = float(logprob)
                elif token == "no":
                    no_logprob = float(logprob)

        if yes_logprob is not None and no_logprob is not None:
            difference = max(-60.0, min(60.0, no_logprob - yes_logprob))
            return 1.0 / (1.0 + math.exp(difference))
        if yes_logprob is not None:
            return 1.0
        if no_logprob is not None:
            return 0.0

        generated = payload.get("content")
        if isinstance(generated, str):
            normalized = generated.strip().lower()
            if normalized == "yes":
                return 1.0
            if normalized == "no":
                return 0.0
        raise AppError(
            code="RERANKER_INVALID_RESPONSE",
            message="The reranker response does not contain yes/no scores.",
            status_code=502,
        )

    async def close(self) -> None:
        await self._client.aclose()
