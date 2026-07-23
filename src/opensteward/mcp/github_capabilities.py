"""Live GitHub-backed MCP capabilities."""

from opensteward.github import (
    GitHubContributionInputOptions,
    GitHubPullRequestAssessmentRequest,
    GitHubPullRequestAssessmentResult,
    GitHubPullRequestAssessmentRunner,
    GitHubRepositoryRef,
    LiveGitHubPullRequestAssessmentRunner,
)
from opensteward.policy import (
    DEFAULT_POLICY_FILENAME,
    ContributionCategory,
)


_assessment_runner: (
    GitHubPullRequestAssessmentRunner
) = LiveGitHubPullRequestAssessmentRunner()


async def assess_pull_request(
    installation_id: int,
    repository: GitHubRepositoryRef,
    pull_number: int,
    policy_path: str = DEFAULT_POLICY_FILENAME,
    explicit_categories: list[
        ContributionCategory
    ] | None = None,
    conversion_options: (
        GitHubContributionInputOptions
        | None
    ) = None,
) -> GitHubPullRequestAssessmentResult:
    """Assess a GitHub pull request against trusted repository policy.

    The tool retrieves pull-request evidence using GitHub App
    installation authentication. Repository policy is loaded from the
    pull request's base commit, not from the contributor-controlled
    head commit.

    This tool is read-only. It does not comment, label, approve,
    request changes, close, merge, or modify the pull request.
    """

    request = GitHubPullRequestAssessmentRequest(
        installation_id=installation_id,
        repository=repository,
        pull_number=pull_number,
        policy_path=policy_path,
        explicit_categories=(
            explicit_categories
            or []
        ),
        conversion_options=(
            conversion_options
            or GitHubContributionInputOptions()
        ),
    )

    return await _assessment_runner.assess(
        request
    )