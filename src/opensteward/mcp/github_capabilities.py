"""Live GitHub-backed MCP capabilities."""

from opensteward.github import (
    GitHubContributionInputOptions,
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubPullRequestAssessmentRequest,
    GitHubPullRequestAssessmentResult,
    GitHubPullRequestAssessmentRunner,
    GitHubRelatedWorkQuery,
    GitHubRelatedWorkRequest,
    GitHubRelatedWorkResult,
    GitHubRelatedWorkRunner,
    GitHubRepositoryRef,
    LiveGitHubPullRequestAssessmentRunner,
    LiveGitHubRelatedWorkRunner,
)
from opensteward.knowledge import KnowledgeRelatedWorkOptions
from opensteward.policy import (
    DEFAULT_POLICY_FILENAME,
    ContributionCategory,
)

_assessment_runner: (
    GitHubPullRequestAssessmentRunner
) = LiveGitHubPullRequestAssessmentRunner()

_related_work_runner: GitHubRelatedWorkRunner = LiveGitHubRelatedWorkRunner()


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


async def find_related_work(
    installation_id: int,
    repository: GitHubRepositoryRef,
    git_ref: str,
    query: GitHubRelatedWorkQuery,
    snapshot_options: GitHubHistoricalKnowledgeSnapshotOptions | None = None,
    related_work_options: KnowledgeRelatedWorkOptions | None = None,
) -> GitHubRelatedWorkResult:
    """Search bounded historical GitHub issues, pull requests, paths, and ADRs.

    The tool returns explainable related-work matches and reports source-history
    and ranking completeness explicitly. It uses GitHub App installation
    authentication and is read-only.

    It does not comment, label, edit, close, merge, or otherwise modify
    repository content.
    """

    request = GitHubRelatedWorkRequest(
        installation_id=installation_id,
        repository=repository,
        git_ref=git_ref,
        query=query,
        snapshot_options=(
            snapshot_options
            if snapshot_options is not None
            else GitHubHistoricalKnowledgeSnapshotOptions()
        ),
        related_work_options=(
            related_work_options
            if related_work_options is not None
            else KnowledgeRelatedWorkOptions()
        ),
    )
    return await _related_work_runner.find(request)
