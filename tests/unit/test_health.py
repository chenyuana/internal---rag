from __future__ import annotations

from fastapi.testclient import TestClient


def test_liveness_does_not_require_external_services(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "test-request"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request"
    assert response.json()["status"] == "ok"


def test_development_is_ready_with_optional_services_disabled(
    client: TestClient,
) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert all(item["status"] == "disabled" for item in payload["dependencies"])


def test_admin_config_is_redacted(client: TestClient) -> None:
    response = client.get("/api/v1/admin/config")

    assert response.status_code == 200
    text = response.text
    assert "change-me-now" not in text
    assert "api_key" not in text
