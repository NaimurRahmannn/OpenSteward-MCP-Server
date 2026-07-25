"""Tests for the OpenSteward FastAPI application."""

import json
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from opensteward.app import app
from opensteward.settings import get_settings

TEST_BEARER_TOKEN = "test-token-with-at-least-32-characters"


@pytest.fixture(autouse=True)
def reset_settings_cache() -> Iterator[None]:
    """Keep environment-backed authentication settings isolated."""

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Create a test client with application lifespan enabled."""

    with TestClient(
        app,
        base_url="http://localhost:8000",
    ) as test_client:
        yield test_client


def test_health_endpoint(
    client: TestClient,
) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "name": "OpenSteward",
        "version": "0.1.0",
    }


def test_readiness_endpoint(
    client: TestClient,
) -> None:
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "environment": "development",
        "checks": {
            "mcp": "ready",
        },
    }


def test_mcp_endpoint_requires_authentication(
    client: TestClient,
) -> None:
    response = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        },
        json={},
    )

    assert response.status_code == 401
    assert response.json() == {
        "error": "invalid_token",
        "error_description": "Authentication required",
    }
    assert response.headers["www-authenticate"].startswith("Bearer ")


def test_mcp_endpoint_rejects_invalid_bearer_token(
    client: TestClient,
) -> None:
    response = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Authorization": "Bearer invalid-token",
            "Content-Type": "application/json",
        },
        json={},
    )

    assert response.status_code == 401


def test_mcp_endpoint_accepts_configured_bearer_token(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callers = {
        "mcp-inspector": {
            "token": TEST_BEARER_TOKEN,
            "installation_ids": [148549890],
        },
    }
    monkeypatch.setenv(
        "OPENSTEWARD_MCP_AUTHORIZED_CALLERS",
        json.dumps(callers),
    )
    get_settings.cache_clear()

    response = client.post(
        "/mcp",
        headers={
            "Accept": "application/json, text/event-stream",
            "Authorization": f"Bearer {TEST_BEARER_TOKEN}",
            "Content-Type": "application/json",
        },
        json={},
    )

    # The payload is intentionally not a valid MCP message. A configured
    # credential must reach MCP protocol validation instead of auth rejection.
    assert response.status_code == 400
    assert response.json()["error"]["code"] == -32602
