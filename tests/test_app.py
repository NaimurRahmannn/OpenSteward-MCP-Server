"""Tests for the OpenSteward FastAPI application."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from opensteward.app import app


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Create a test client with application lifespan enabled."""

    with TestClient(
        app,
        base_url="http://localhost",
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


def test_mcp_endpoint_is_mounted(
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

    # The request is intentionally not a valid MCP message.
    # We only verify that the request reaches the MCP application
    # instead of returning a missing-route response.
    assert response.status_code != 404