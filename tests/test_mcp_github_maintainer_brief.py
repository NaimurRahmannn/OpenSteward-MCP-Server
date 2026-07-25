"""Tests for the thin GitHub maintainer-brief MCP capability."""

from copy import deepcopy
from typing import Any

import pytest

import opensteward.mcp.github_capabilities as capabilities
from opensteward.github import (
    GitHubContributionInputOptions,
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubMaintainerBriefRequest,
    GitHubRepositoryRef,
)
from opensteward.knowledge import KnowledgeRelatedWorkOptions
from opensteward.policy import ContributionCategory
from opensteward.review_intelligence import ReviewCostAssessmentOptions

REPOSITORY = GitHubRepositoryRef(owner="acme", name="framework")


class RecordingRunner:
    def __init__(
        self,
        *,
        result: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result if result is not None else object()
        self.error = error
        self.calls: list[GitHubMaintainerBriefRequest] = []

    async def build(self, request: GitHubMaintainerBriefRequest) -> Any:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_capability_constructs_defaults_copies_lists_and_delegates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(capabilities, "_maintainer_brief_runner", runner)
    categories = [ContributionCategory.ARCHITECTURE]
    before = deepcopy(categories)

    result = await capabilities.get_maintainer_brief(
        installation_id=41,
        repository=REPOSITORY,
        pull_number=17,
        explicit_categories=categories,
    )

    assert result is runner.result
    assert len(runner.calls) == 1
    request = runner.calls[0]
    assert request.installation_id == 41
    assert request.repository == REPOSITORY
    assert request.pull_number == 17
    assert request.policy_path == ".opensteward.yml"
    assert request.explicit_categories == categories
    assert request.explicit_categories is not categories
    assert request.conversion_options == GitHubContributionInputOptions()
    assert request.snapshot_options == GitHubHistoricalKnowledgeSnapshotOptions()
    assert request.related_work_options == KnowledgeRelatedWorkOptions()
    assert request.review_cost_options == ReviewCostAssessmentOptions()
    assert categories == before


@pytest.mark.asyncio
async def test_capability_preserves_all_explicit_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(capabilities, "_maintainer_brief_runner", runner)
    conversion = GitHubContributionInputOptions(
        require_complete_file_list=False
    )
    snapshot = GitHubHistoricalKnowledgeSnapshotOptions()
    related = KnowledgeRelatedWorkOptions(max_results=3)
    review = ReviewCostAssessmentOptions(max_evidence_items_per_signal=4)

    await capabilities.get_maintainer_brief(
        installation_id=41,
        repository=REPOSITORY,
        pull_number=17,
        policy_path="config/steward.yml",
        explicit_categories=[ContributionCategory.SECURITY],
        conversion_options=conversion,
        snapshot_options=snapshot,
        related_work_options=related,
        review_cost_options=review,
    )

    request = runner.calls[0]
    assert request.policy_path == "config/steward.yml"
    assert request.explicit_categories == [ContributionCategory.SECURITY]
    assert request.conversion_options == conversion
    assert request.snapshot_options == snapshot
    assert request.related_work_options == related
    assert request.review_cost_options == review


class SentinelRunnerError(RuntimeError):
    """Distinct runner failure."""


@pytest.mark.asyncio
async def test_capability_propagates_runner_error_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = SentinelRunnerError("runner failed")
    runner = RecordingRunner(error=error)
    monkeypatch.setattr(capabilities, "_maintainer_brief_runner", runner)

    with pytest.raises(SentinelRunnerError) as exc_info:
        await capabilities.get_maintainer_brief(
            installation_id=41,
            repository=REPOSITORY,
            pull_number=17,
        )

    assert exc_info.value is error
    assert len(runner.calls) == 1


def test_docstring_states_boundaries_and_read_only_behavior() -> None:
    docstring = capabilities.get_maintainer_brief.__doc__ or ""

    for text in (
        "contribution readiness",
        "repository policy",
        "related historical work",
        "review cost",
        "maintainer routing",
        "author action or maintainer review",
        "security",
        "database",
        "deployment",
        "architecture",
        "general",
        "deterministic",
        "incomplete evidence and coverage",
        "GitHub App installation",
        "read-only",
        "does not comment, label, approve, reject, request changes, close, merge",
        "does not decide whether the pull request should",
        "contributor skill or trustworthiness",
        "use an LLM",
    ):
        assert text in docstring
    assert "httpx" not in capabilities.get_maintainer_brief.__code__.co_names
