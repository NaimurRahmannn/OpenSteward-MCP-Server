"""MCP integration tests for repository policy capabilities."""

import json
from collections.abc import AsyncGenerator

import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import (
    create_connected_server_and_client_session,
)
from mcp.types import TextResourceContents

import opensteward.mcp.policy_capabilities as policy_capabilities
from opensteward.mcp.server import mcp
from opensteward.policy import (
    LoadedRepositoryPolicy,
    PolicyLoadError,
    PolicySource,
    RepositoryPolicy,
)


@pytest.fixture
def anyio_backend() -> str:
    """Run MCP tests using asyncio."""

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
async def test_policy_tool_and_resource_are_discoverable(
    client_session: ClientSession,
) -> None:
    tools_result = await client_session.list_tools()

    tool_names = {
        tool.name
        for tool in tools_result.tools
    }

    assert "evaluate_repository_policy" in tool_names

    resources_result = await client_session.list_resources()

    resource_uris = {
        str(resource.uri)
        for resource in resources_result.resources
    }

    assert "steward://repository/policy" in resource_uris


@pytest.mark.anyio
async def test_evaluate_repository_policy_tool(
    client_session: ClientSession,
) -> None:
    policy_yaml = """
    version: 1

    pull_requests:
      linked_issue_required_for:
        - security

      tests_required_for:
        - security

      preferred_maximum_diff_lines: 500

    protected_paths:
      - pattern: src/security/**
        risk: critical
        human_review_required: true

    review:
      required_approvals:
        default: 1
        public_api: 2
        security: 3
    """

    result = await client_session.call_tool(
        "evaluate_repository_policy",
        {
            "policy_yaml": policy_yaml,
            "contribution": {
                "changed_files": [
                    "src/security/auth.py",
                ],
                "additions": 600,
                "deletions": 100,
                "categories": [
                    "security",
                ],
                "linked_issue_numbers": [],
                "tests_changed": False,
                "current_approvals": 1,
            },
        },
    )

    assert result.isError is False
    assert result.structuredContent is not None

    data = result.structuredContent
    packet = data["packet"]
    assert packet["recommendation"] == "request_changes"
    assert packet["ready_for_detailed_review"] is False

    assert len(packet["blocking_requirements"]) == 3
    assert len(packet["warnings"]) == 2

    assert {
        item["rule"]
        for item in packet["blocking_requirements"]
    } == {
        "required_tests",
        "linked_issue",
        "required_approvals",
    }

    assert {
        item["rule"]
        for item in packet["warnings"]
    } == {
        "preferred_diff_size",
        "protected_path",
    }

    assert len(packet["suggested_next_actions"]) >= 3
    assert "2 remaining" in packet["approval_summary"]
    assert data["policy_source"] == "memory"
    assert data["source_reference"] == "mcp:policy_yaml"
    assert data["used_defaults"] is False
    assert data["policy_version"] == 1

    evaluation = data["evaluation"]

    assert evaluation["compliant"] is False
    assert evaluation["requires_human_review"] is True
    assert evaluation["required_approvals"] == 3
    assert evaluation["current_approvals"] == 1
    assert evaluation["remaining_approvals"] == 2
    assert (
        evaluation["highest_protected_path_risk"]
        == "critical"
    )


@pytest.mark.anyio
async def test_policy_tool_returns_explainable_findings(
    client_session: ClientSession,
) -> None:
    result = await client_session.call_tool(
        "evaluate_repository_policy",
        {
            "policy_yaml": """
            version: 1

            pull_requests:
              tests_required_for:
                - bug_fix

            review:
              required_approvals:
                default: 1
                public_api: 2
                security: 2
            """,
            "contribution": {
                "changed_files": [
                    "src/application.py",
                ],
                "categories": [
                    "bug_fix",
                ],
                "tests_changed": False,
                "current_approvals": 1,
            },
        },
    )

    assert result.isError is False
    assert result.structuredContent is not None

    findings = result.structuredContent["evaluation"]["findings"]

    test_finding = next(
        finding
        for finding in findings
        if finding["rule"] == "required_tests"
    )

    assert test_finding["status"] == "failed"
    assert test_finding["severity"] == "high"
    assert "tests_changed:false" in test_finding["evidence"]
    assert test_finding["remediation"] is not None


@pytest.mark.anyio
async def test_repository_policy_resource(
    client_session: ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_policy = LoadedRepositoryPolicy(
        policy=RepositoryPolicy.model_validate(
            {
                "pull_requests": {
                    "preferred_maximum_diff_lines": 750,
                }
            }
        ),
        source=PolicySource.REPOSITORY_FILE,
        source_reference=".opensteward.yml",
    )

    monkeypatch.setattr(
        policy_capabilities,
        "load_repository_policy_with_metadata",
        lambda: loaded_policy,
    )

    result = await client_session.read_resource(
        "steward://repository/policy"
    )

    assert len(result.contents) == 1

    content = result.contents[0]

    assert isinstance(content, TextResourceContents)

    data = json.loads(content.text)

    assert data["source"] == "repository_file"
    assert data["source_reference"] == ".opensteward.yml"
    assert data["used_defaults"] is False

    assert (
        data["policy"]["pull_requests"]
        ["preferred_maximum_diff_lines"]
        == 750
    )


def test_policy_handler_rejects_invalid_yaml() -> None:
    with pytest.raises(
        PolicyLoadError,
        match="invalid YAML syntax",
    ):
        policy_capabilities.evaluate_repository_policy(
            policy_yaml="""
            version: 1
            protected_paths:
              - pattern: [
            """,
            contribution={
                "changed_files": [
                    "src/application.py",
                ],
            },
        )