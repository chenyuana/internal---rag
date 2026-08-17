from __future__ import annotations

import httpx

from app.core.config import ModelEndpointSettings
from app.services.reranker_client import RerankerClient


async def test_reranker_adapter_normalizes_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/rerank"
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.95},
                    {"index": 0, "relevance_score": 0.75},
                ]
            },
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=True,
        base_url="http://reranker.local/v1",
        model_name="Qwen3-Reranker-0.6B",
        operation_path="/rerank",
    )
    client = RerankerClient(settings, transport=httpx.MockTransport(handler))
    try:
        results = await client.rerank(
            query="任务周期",
            documents=["第一段", "第二段"],
            top_n=2,
        )
    finally:
        await client.close()

    assert [(item.index, item.score) for item in results] == [(1, 0.95), (0, 0.75)]


async def test_qwen3_completion_scoring_ranks_chinese_documents() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/completion"
        payload = request.read().decode("utf-8")
        if "Qwen3-Embedding-0.6B" in payload:
            yes_logprob, no_logprob = -0.1, -5.0
        else:
            yes_logprob, no_logprob = -6.0, -0.2
        return httpx.Response(
            200,
            json={
                "content": "yes" if yes_logprob > no_logprob else "no",
                "completion_probabilities": [
                    {
                        "top_logprobs": [
                            {"token": "yes", "logprob": yes_logprob},
                            {"token": "no", "logprob": no_logprob},
                        ]
                    }
                ],
            },
        )

    settings = ModelEndpointSettings(
        enabled=True,
        required=True,
        base_url="http://reranker.local",
        model_name="qwen3-reranker-0.6b-q8_0",
        operation_path="/completion",
    )
    client = RerankerClient(settings, transport=httpx.MockTransport(handler))
    try:
        results = await client.rerank(
            query="RAG系统使用什么嵌入模型？",
            documents=[
                "明天北京天气晴朗。",
                "本系统使用Qwen3-Embedding-0.6B生成向量。",
                "Docker Desktop使用Hyper-V。",
            ],
            top_n=2,
        )
    finally:
        await client.close()

    assert [item.index for item in results] == [1, 0]
    assert results[0].score > 0.99
    assert results[1].score < 0.01
