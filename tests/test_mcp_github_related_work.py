"""Tests for the thin GitHub related-work MCP capability."""

from copy import deepcopy
from typing import Any

import pytest

import opensteward.mcp.github_capabilities as capabilities
from opensteward.github import (
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubRelatedWorkQuery,
    GitHubRelatedWorkRequest,
    GitHubRepositoryRef,
)
from opensteward.knowledge import KnowledgeRelatedWorkOptions

REPOSITORY = GitHubRepositoryRef(owner="acme", name="framework")


class RecordingRunner:
    """Record capability delegation and return one opaque result."""

    def __init__(
        self,
        *,
        result: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result if result is not None else object()
        self.error = error
        self.calls: list[GitHubRelatedWorkRequest] = []

    async def find(self, request: GitHubRelatedWorkRequest) -> Any:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_capability_constructs_request_defaults_and_delegates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(capabilities, "_related_work_runner", runner)
    labels = ["architecture"]
    paths = ["src/parser/service.py"]
    query = GitHubRelatedWorkQuery(
        text="parser",
        labels=labels,
        affected_paths=paths,
    )
    labels_before = deepcopy(labels)
    paths_before = deepcopy(paths)

    result = await capabilities.find_related_work(
        installation_id=41,
        repository=REPOSITORY,
        git_ref="  refs/heads/main  ",
        query=query,
    )

    assert result is runner.result
    assert len(runner.calls) == 1
    request = runner.calls[0]
    assert request.installation_id == 41
    assert request.repository == REPOSITORY
    assert request.git_ref == "refs/heads/main"
    assert request.query == query
    assert request.query.text == "parser"
    assert request.query.labels == labels
    assert request.query.affected_paths == paths
    assert request.snapshot_options == GitHubHistoricalKnowledgeSnapshotOptions()
    assert request.related_work_options == KnowledgeRelatedWorkOptions()
    assert labels == labels_before
    assert paths == paths_before


@pytest.mark.asyncio
async def test_capability_preserves_explicit_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(capabilities, "_related_work_runner", runner)
    snapshot_options = GitHubHistoricalKnowledgeSnapshotOptions()
    related_options = KnowledgeRelatedWorkOptions(max_results=3)

    await capabilities.find_related_work(
        installation_id=41,
        repository=REPOSITORY,
        git_ref="main",
        query=GitHubRelatedWorkQuery(text="parser"),
        snapshot_options=snapshot_options,
        related_work_options=related_options,
    )

    request = runner.calls[0]
    assert request.snapshot_options == snapshot_options
    assert request.related_work_options == related_options


class SentinelRunnerError(RuntimeError):
    """Distinct runner failure."""


@pytest.mark.asyncio
async def test_capability_propagates_runner_errors_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = SentinelRunnerError("runner failed")
    runner = RecordingRunner(error=error)
    monkeypatch.setattr(capabilities, "_related_work_runner", runner)

    with pytest.raises(SentinelRunnerError) as exc_info:
        await capabilities.find_related_work(
            installation_id=41,
            repository=REPOSITORY,
            git_ref="main",
            query=GitHubRelatedWorkQuery(text="parser"),
        )

    assert exc_info.value is error
    assert len(runner.calls) == 1


def test_capability_docstring_is_explicitly_bounded_and_read_only() -> None:
    docstring = capabilities.find_related_work.__doc__ or ""
    source_names = {
        "issues",
        "pull requests",
        "paths",
        "ADRs",
    }

    assert "bounded historical GitHub" in docstring
    assert all(name in docstring for name in source_names)
    assert "explainable related-work matches" in docstring
    assert "source-history" in docstring
    assert "ranking completeness" in docstring
    assert "GitHub App installation" in docstring
    assert "read-only" in docstring
    assert "does not comment, label, edit, close, merge" in docstring
    assert "httpx" not in capabilities.find_related_work.__code__.co_names
