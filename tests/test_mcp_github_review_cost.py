"""Tests for the thin GitHub review-cost MCP capability."""

from copy import deepcopy
from typing import Any

import pytest

import opensteward.mcp.github_capabilities as capabilities
from opensteward.github import (
    GitHubContributionInputOptions,
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubRepositoryRef,
    GitHubReviewCostRequest,
)
from opensteward.knowledge import KnowledgeRelatedWorkOptions
from opensteward.policy import ContributionCategory
from opensteward.review_intelligence import ReviewCostAssessmentOptions

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
        self.calls: list[GitHubReviewCostRequest] = []

    async def assess(self, request: GitHubReviewCostRequest) -> Any:
        self.calls.append(request)
        if self.error is not None:
            raise self.error
        return self.result


@pytest.mark.asyncio
async def test_capability_constructs_default_request_and_delegates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(capabilities, "_review_cost_runner", runner)
    categories = [ContributionCategory.ARCHITECTURE]
    categories_before = deepcopy(categories)

    result = await capabilities.assess_review_cost(
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
    assert request.conversion_options == GitHubContributionInputOptions()
    assert request.snapshot_options == GitHubHistoricalKnowledgeSnapshotOptions()
    assert request.related_work_options == KnowledgeRelatedWorkOptions()
    assert request.review_cost_options == ReviewCostAssessmentOptions()
    assert categories == categories_before


@pytest.mark.asyncio
async def test_capability_preserves_all_explicit_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = RecordingRunner()
    monkeypatch.setattr(capabilities, "_review_cost_runner", runner)
    conversion = GitHubContributionInputOptions(
        require_complete_file_list=False
    )
    snapshot = GitHubHistoricalKnowledgeSnapshotOptions()
    related = KnowledgeRelatedWorkOptions(max_results=3)
    review_cost = ReviewCostAssessmentOptions(
        max_evidence_items_per_signal=4
    )

    await capabilities.assess_review_cost(
        installation_id=41,
        repository=REPOSITORY,
        pull_number=17,
        policy_path="config/steward.yml",
        explicit_categories=[ContributionCategory.SECURITY],
        conversion_options=conversion,
        snapshot_options=snapshot,
        related_work_options=related,
        review_cost_options=review_cost,
    )

    request = runner.calls[0]
    assert request.policy_path == "config/steward.yml"
    assert request.explicit_categories == [ContributionCategory.SECURITY]
    assert request.conversion_options == conversion
    assert request.snapshot_options == snapshot
    assert request.related_work_options == related
    assert request.review_cost_options == review_cost


class SentinelRunnerError(RuntimeError):
    """Distinct runner failure."""


@pytest.mark.asyncio
async def test_capability_propagates_runner_errors_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = SentinelRunnerError("runner failed")
    runner = RecordingRunner(error=error)
    monkeypatch.setattr(capabilities, "_review_cost_runner", runner)

    with pytest.raises(SentinelRunnerError) as exc_info:
        await capabilities.assess_review_cost(
            installation_id=41,
            repository=REPOSITORY,
            pull_number=17,
        )

    assert exc_info.value is error
    assert runner.calls[0].pull_number == 17


def test_capability_docstring_is_explicitly_bounded_and_read_only() -> None:
    docstring = capabilities.assess_review_cost.__doc__ or ""

    assert "expected maintainer effort" in docstring
    for source in (
        "pull request",
        "repository policy",
        "checks",
        "reviews",
        "changed paths",
        "related historical work",
    ):
        assert source in docstring
    for driver in ("structural", "risk", "validation", "historical"):
        assert driver in docstring
    assert "source or ranking coverage" in docstring
    assert "GitHub App installation" in docstring
    assert "read-only" in docstring
    assert "does not comment, label, approve, reject, close" in docstring
    assert "contributor skill" in docstring
    assert "trustworthiness" in docstring
    assert "httpx" not in capabilities.assess_review_cost.__code__.co_names
