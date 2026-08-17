from __future__ import annotations

import httpx

from app.core.config import RagflowSettings
from app.schemas.retrieval import RagflowRetrievalRequest
from app.services.ragflow_client import RagflowClient


async def test_ragflow_health_path_is_configurable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/custom-health"
        return httpx.Response(200, json={"status": "ok"})

    settings = RagflowSettings(
        enabled=True,
        required=True,
        base_url="http://ragflow.local",
        health_path="/custom-health",
    )
    client = RagflowClient(settings, transport=httpx.MockTransport(handler))
    try:
        result = await client.probe()
    finally:
        await client.close()

    assert result.ready is True


async def test_ragflow_client_does_not_inherit_host_proxy_settings() -> None:
    settings = RagflowSettings(
        enabled=True,
        required=True,
        base_url="http://ragflow.local",
    )
    client = RagflowClient(settings)
    try:
        assert client._client._trust_env is False
    finally:
        await client.close()


async def test_ragflow_retrieval_is_normalized() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/retrieval"
        payload = __import__("json").loads(request.content)
        assert payload["dataset_ids"] == ["kb-1"]
        assert payload["keyword"] is False
        assert payload["reference_metadata"]["include"] is True
        assert "status" in payload["reference_metadata"]["fields"]
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "chunks": [
                        {
                            "id": "chunk-1",
                            "document_id": "doc-1",
                            "kb_id": "kb-1",
                            "content": "HEALTH_MONITOR_PERIOD_MS = 1000U",
                            "document_keyword": "design.pdf",
                            "document_metadata": {
                                "version": "V2.3",
                                "status": "effective",
                                "page_number": 37,
                            },
                            "similarity": 0.91,
                            "vector_similarity": 0.82,
                            "term_similarity": 1.0,
                        }
                    ]
                },
            },
        )

    settings = RagflowSettings(
        enabled=True,
        required=True,
        base_url="http://ragflow.local",
    )
    client = RagflowClient(settings, transport=httpx.MockTransport(handler))
    try:
        chunks = await client.retrieve(
            RagflowRetrievalRequest(
                question="周期是多少？",
                dataset_ids=["kb-1"],
            )
        )
    finally:
        await client.close()

    assert len(chunks) == 1
    assert chunks[0].metadata.document_name == "design.pdf"
    assert chunks[0].metadata.page_number == 37
    assert chunks[0].hybrid_score == 0.91


async def test_ragflow_list_datasets_parses_list_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/datasets"
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": [
                    {"id": "kb-1", "name": "法规11"},
                    {"id": "kb-2", "name": "知识库问答"},
                ],
            },
        )

    settings = RagflowSettings(
        enabled=True,
        required=True,
        base_url="http://ragflow.local",
    )
    client = RagflowClient(settings, transport=httpx.MockTransport(handler))
    try:
        datasets = await client.list_datasets()
    finally:
        await client.close()

    assert datasets == [{"id": "kb-1", "name": "法规11"}, {"id": "kb-2", "name": "知识库问答"}]


async def test_ragflow_list_datasets_parses_docs_wrapper() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "code": 0,
                "data": {
                    "docs": [
                        {"id": "kb-1", "name": "法规"},
                        {"id": "kb-2", "name": "test"},
                    ]
                },
            },
        )

    settings = RagflowSettings(
        enabled=True,
        required=True,
        base_url="http://ragflow.local",
    )
    client = RagflowClient(settings, transport=httpx.MockTransport(handler))
    try:
        datasets = await client.list_datasets()
    finally:
        await client.close()

    assert datasets == [{"id": "kb-1", "name": "法规"}, {"id": "kb-2", "name": "test"}]
