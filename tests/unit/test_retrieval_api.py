from __future__ import annotations

from fastapi.testclient import TestClient


def test_retrieval_returns_safe_error_when_ragflow_is_disabled(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "query": "任务周期是多少？",
            "knowledge_base_ids": ["kb-1"],
        },
    )

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "RAGFLOW_DISABLED"
