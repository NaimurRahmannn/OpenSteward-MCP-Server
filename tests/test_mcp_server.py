"""Integration tests for the OpenSteward MCP server."""

from collections.abc import AsyncGenerator

import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

from opensteward.mcp.server import mcp


@pytest.fixture
def anyio_backend() -> str:
    """Run MCP integration tests with the asyncio backend."""

    return "asyncio"


@pytest.fixture
async def client_session() -> AsyncGenerator[ClientSession]:
    """Create an in-memory MCP client connected to OpenSteward."""

    async with create_connected_server_and_client_session(
        mcp,
        raise_exceptions=True,
    ) as session:
        yield session


@pytest.mark.anyio
async def test_mcp_server_exposes_expected_tools(
    client_session: ClientSession,
) -> None:
    result = await client_session.list_tools()

    tool_names = {
        tool.name
        for tool in result.tools
    }

    assert "system_status" in tool_names
    assert "estimate_review_cost" in tool_names


@pytest.mark.anyio
async def test_system_status_tool(
    client_session: ClientSession,
) -> None:
    result = await client_session.call_tool(
        "system_status",
        {},
    )

    assert result.isError is False
    assert result.structuredContent is not None

    data = result.structuredContent

    assert data["name"] == "OpenSteward"
    assert data["version"] == "0.1.0"
    assert data["stage"] == "version-1-foundation"
    assert data["mode"] == "read-only"


@pytest.mark.anyio
async def test_estimate_review_cost_tool(
    client_session: ClientSession,
) -> None:
    result = await client_session.call_tool(
        "estimate_review_cost",
        {
            "factors": {
                "change_size": 80,
                "component_risk": 90,
                "test_gap": 70,
                "ownership_dispersion": 40,
                "public_api_impact": 60,
                "ci_risk": 20,
                "reviewer_load": 50,
            }
        },
    )

    assert result.isError is False
    assert result.structuredContent is not None

    data = result.structuredContent

    assert data["score"] == 64
    assert data["level"] == "high"
    assert len(data["contributions"]) == 7
    assert "substantial maintainer attention" in data["summary"]