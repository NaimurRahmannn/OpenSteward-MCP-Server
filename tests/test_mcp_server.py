"""Integration tests for the OpenSteward MCP server."""

from collections.abc import AsyncGenerator
from datetime import UTC, datetime

import pytest
from mcp.client.session import ClientSession
from mcp.shared.memory import create_connected_server_and_client_session

import opensteward.mcp.github_capabilities as github_capabilities
from opensteward.github import (
    GitHubPullRequestAssessmentRequest,
    GitHubPullRequestAssessmentResult,
    GitHubRelatedWorkRequest,
    GitHubRelatedWorkResult,
    GitHubRelatedWorkSnapshotSummary,
    GitHubRepositoryRef,
    GitHubReviewCostRequest,
    GitHubReviewCostResult,
    knowledge_repository_from_github,
)
from opensteward.knowledge import (
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalQuery,
    KnowledgeRelatedWorkService,
    KnowledgeSourceKind,
)
from opensteward.mcp.server import mcp
from tests.github.test_review_cost import (
    assessment_result,
    completed_review_cost_result,
)


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


async def github_related_work_result() -> GitHubRelatedWorkResult:
    """Build one real serializable deterministic related-work result."""

    repository = GitHubRepositoryRef(owner="acme", name="framework")
    knowledge_repository = knowledge_repository_from_github(repository)
    collected_at = datetime(2026, 5, 1, tzinfo=UTC)
    query = KnowledgeLexicalQuery(
        repository=knowledge_repository,
        text="parser",
    )
    item = KnowledgeItem(
        repository=knowledge_repository,
        item_type=KnowledgeItemType.ISSUE,
        external_id="17",
        source_kind=KnowledgeSourceKind.GITHUB,
        state=KnowledgeItemState.CLOSED,
        title="Adopt parser registry",
        body="Parser registry architecture.",
        summary="Parser registry decision.",
        url="https://github.com/acme/framework/issues/17",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 2, 1, tzinfo=UTC),
        closed_at=datetime(2026, 2, 1, tzinfo=UTC),
        labels=["architecture"],
        affected_paths=["src/parser/registry.py"],
        components=["parser"],
        decision_significance=DecisionSignificance.HIGH,
    )
    related_work = await KnowledgeRelatedWorkService().find(
        query,
        [item],
        as_of=collected_at,
    )
    snapshot = GitHubRelatedWorkSnapshotSummary(
        repository=repository,
        knowledge_repository=knowledge_repository,
        requested_ref="main",
        resolved_commit_sha="a" * 40,
        adr_tree_sha="b" * 40,
        collected_at=collected_at,
        adr_snapshot_commit_date=datetime(2026, 4, 30, tzinfo=UTC),
        complete=True,
        total_count=1,
        issue_count=1,
        pull_request_count=0,
        adr_count=0,
        warnings=[],
    )
    return GitHubRelatedWorkResult(
        repository=repository,
        snapshot=snapshot,
        related_work=related_work,
        warnings=[],
    )


class StaticRelatedWorkRunner:
    """Return one prepared result through the registered MCP function."""

    def __init__(self, result: GitHubRelatedWorkResult) -> None:
        self.result = result
        self.calls: list[GitHubRelatedWorkRequest] = []

    async def find(
        self,
        request: GitHubRelatedWorkRequest,
    ) -> GitHubRelatedWorkResult:
        self.calls.append(request)
        return self.result


class StaticAssessmentRunner:
    """Return one prepared legacy assessment through the registered tool."""

    def __init__(self, result: GitHubPullRequestAssessmentResult) -> None:
        self.result = result
        self.calls: list[GitHubPullRequestAssessmentRequest] = []

    async def assess(
        self,
        request: GitHubPullRequestAssessmentRequest,
    ) -> GitHubPullRequestAssessmentResult:
        self.calls.append(request)
        return self.result


class StaticReviewCostRunner:
    """Return one prepared review-cost result through the registered tool."""

    def __init__(self, result: GitHubReviewCostResult) -> None:
        self.result = result
        self.calls: list[GitHubReviewCostRequest] = []

    async def assess(
        self,
        request: GitHubReviewCostRequest,
    ) -> GitHubReviewCostResult:
        self.calls.append(request)
        return self.result


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
    assert "evaluate_repository_policy" in tool_names
    assert "assess_pull_request" in tool_names
    assert "find_related_work" in tool_names
    assert "assess_review_cost" in tool_names

    resources = await client_session.list_resources()
    assert any(
        str(resource.uri) == "steward://repository/policy"
        for resource in resources.resources
    )


@pytest.mark.anyio
async def test_assess_pull_request_schema_preserves_legacy_public_fields(
    client_session: ClientSession,
) -> None:
    result = await client_session.list_tools()
    tool = next(tool for tool in result.tools if tool.name == "assess_pull_request")

    assert tool.outputSchema is not None
    assert set(tool.outputSchema["properties"]) == {
        "read_only",
        "installation_id",
        "summary",
        "policy",
        "conversion",
        "packet",
        "evaluation",
    }
    assert "installation_id" in tool.outputSchema["required"]


@pytest.mark.anyio
async def test_assess_pull_request_invocation_preserves_legacy_output(
    client_session: ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = assessment_result()
    runner = StaticAssessmentRunner(expected)
    monkeypatch.setattr(
        github_capabilities,
        "_assessment_runner",
        runner,
    )

    result = await client_session.call_tool(
        "assess_pull_request",
        {
            "installation_id": 73,
            "repository": {
                "owner": "acme",
                "name": "framework",
            },
            "pull_number": 17,
        },
    )

    assert result.isError is False
    assert result.structuredContent is not None
    data = result.structuredContent
    assert data["installation_id"] == 41
    assert "snapshot" not in data
    assert "repository_policy" not in data
    for field in ("summary", "policy", "conversion", "packet", "evaluation"):
        assert field in data
    assert len(runner.calls) == 1
    assert runner.calls[0].installation_id == 73


@pytest.mark.anyio
async def test_find_related_work_schema_is_structured_without_duplicate_repository(
    client_session: ClientSession,
) -> None:
    result = await client_session.list_tools()
    tool = next(tool for tool in result.tools if tool.name == "find_related_work")

    properties = tool.inputSchema["properties"]
    assert set(properties) == {
        "installation_id",
        "repository",
        "git_ref",
        "query",
        "snapshot_options",
        "related_work_options",
    }
    query_schema = tool.inputSchema["$defs"]["GitHubRelatedWorkQuery"]
    assert "repository" not in query_schema["properties"]
    assert tool.outputSchema is not None
    assert {
        "repository",
        "snapshot",
        "related_work",
        "warnings",
        "mode",
        "returned_count",
        "source_history_complete",
        "ranking_coverage_complete",
        "result_truncated",
        "complete",
    } <= set(tool.outputSchema["properties"])


@pytest.mark.anyio
async def test_find_related_work_invocation_returns_provenance_evidence_and_coverage(
    client_session: ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = await github_related_work_result()
    runner = StaticRelatedWorkRunner(expected)
    monkeypatch.setattr(
        github_capabilities,
        "_related_work_runner",
        runner,
    )

    result = await client_session.call_tool(
        "find_related_work",
        {
            "installation_id": 73,
            "repository": {
                "owner": "acme",
                "name": "framework",
            },
            "git_ref": "main",
            "query": {
                "text": "parser",
            },
        },
    )

    assert result.isError is False
    assert result.structuredContent is not None
    data = result.structuredContent
    assert data["snapshot"]["requested_ref"] == "main"
    assert data["snapshot"]["resolved_commit_sha"] == "a" * 40
    assert data["snapshot"]["adr_tree_sha"] == "b" * 40
    assert data["related_work"]["matches"][0]["lexical_match"]["evidence"]
    assert data["source_history_complete"] is True
    assert data["ranking_coverage_complete"] is True
    assert data["result_truncated"] is False
    assert data["complete"] is True
    assert len(runner.calls) == 1
    assert runner.calls[0].installation_id == 73
    serialized = str(data).lower()
    assert "installation_id" not in serialized
    assert "token" not in serialized
    assert "private_key" not in serialized


@pytest.mark.anyio
async def test_assess_review_cost_schema_is_structured(
    client_session: ClientSession,
) -> None:
    result = await client_session.list_tools()
    tool = next(tool for tool in result.tools if tool.name == "assess_review_cost")

    assert set(tool.inputSchema["properties"]) == {
        "installation_id",
        "repository",
        "pull_number",
        "policy_path",
        "explicit_categories",
        "conversion_options",
        "snapshot_options",
        "related_work_options",
        "review_cost_options",
    }
    assert tool.outputSchema is not None
    assert {
        "repository",
        "pull_request",
        "pull_request_assessment",
        "related_work",
        "review_cost",
        "warnings",
        "score",
        "level",
        "complete",
    } <= set(tool.outputSchema["properties"])
    assessment_property = tool.outputSchema["properties"][
        "pull_request_assessment"
    ]
    assessment_definition = tool.outputSchema["$defs"][
        assessment_property["$ref"].removeprefix("#/$defs/")
    ]
    assert set(assessment_definition["properties"]) == {
        "read_only",
        "summary",
        "policy",
        "conversion",
        "packet",
        "evaluation",
    }


@pytest.mark.anyio
async def test_assess_review_cost_invocation_returns_explainable_coverage(
    client_session: ClientSession,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = await completed_review_cost_result()
    runner = StaticReviewCostRunner(expected)
    monkeypatch.setattr(
        github_capabilities,
        "_review_cost_runner",
        runner,
    )

    result = await client_session.call_tool(
        "assess_review_cost",
        {
            "installation_id": 73,
            "repository": {
                "owner": "acme",
                "name": "framework",
            },
            "pull_number": 17,
        },
    )

    assert result.isError is False
    assert result.structuredContent is not None
    data = result.structuredContent
    assert data["pull_request"]["pull_number"] == 17
    assert len(data["review_cost"]["contributions"]) == 5
    assert data["review_cost"]["reducers"]
    assert data["review_cost"]["warnings"] == []
    assert data["score"] == expected.score
    assert data["level"] == expected.level.value
    assert data["complete"] is True
    assert data["related_work"]["source_history_complete"] is True
    assert data["related_work"]["ranking_coverage_complete"] is True
    assert data["related_work"]["result_truncated"] is False
    nested_assessment = data["pull_request_assessment"]
    assert {
        "read_only",
        "summary",
        "policy",
        "conversion",
        "packet",
        "evaluation",
    } == set(nested_assessment)
    assert "installation_id" not in nested_assessment
    assert "snapshot" not in nested_assessment
    assert "repository_policy" not in nested_assessment
    assert len(runner.calls) == 1
    assert runner.calls[0].installation_id == 73
    serialized = str(data).lower()
    assert "installation_id" not in serialized
    assert "private_key" not in serialized
    assert "credentials" not in serialized


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
