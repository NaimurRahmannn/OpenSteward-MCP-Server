"""End-to-end GitHub pull-request policy assessments."""

from typing import Literal, Protocol

from pydantic import (
    Field,
    field_validator,
)

from opensteward.github.contribution_inputs import (
    GitHubContributionInputOptions,
    GitHubContributionInputResult,
    build_contribution_policy_input_from_snapshot,
)
from opensteward.github.models import (
    GitHubRepositoryRef,
    StrictGitHubModel,
)
from opensteward.github.pull_requests import (
    GitHubChecksSummary,
    GitHubPullRequestSnapshot,
)
from opensteward.github.repositories import (
    GitHubRepositoryPolicyResult,
)
from opensteward.policy import (
    DEFAULT_POLICY_FILENAME,
    ContributionCategory,
    MaintainerPolicyPacket,
    PolicyEvaluationResult,
    PolicySource,
    build_maintainer_policy_packet,
    evaluate_contribution_policy,
    normalize_repository_path,
)


class GitHubPullRequestAssessmentError(RuntimeError):
    """Raised when a trustworthy PR assessment cannot be produced."""


class GitHubPullRequestAssessmentRequest(StrictGitHubModel):
    """Input required to assess one GitHub pull request."""

    installation_id: int = Field(gt=0)

    repository: GitHubRepositoryRef

    pull_number: int = Field(gt=0)

    policy_path: str = DEFAULT_POLICY_FILENAME

    explicit_categories: list[
        ContributionCategory
    ] = Field(
        default_factory=list,
    )

    conversion_options: (
        GitHubContributionInputOptions
    ) = Field(
        default_factory=(
            GitHubContributionInputOptions
        ),
    )

    @field_validator("policy_path")
    @classmethod
    def normalize_policy_path(
        cls,
        value: str,
    ) -> str:
        """Validate the repository-relative policy path."""

        return normalize_repository_path(
            value
        )

    @field_validator("explicit_categories")
    @classmethod
    def reject_duplicate_categories(
        cls,
        categories: list[
            ContributionCategory
        ],
    ) -> list[ContributionCategory]:
        """Reject duplicate explicit category hints."""

        if len(categories) != len(set(categories)):
            raise ValueError(
                "Explicit contribution categories must be unique."
            )

        return categories


class GitHubPullRequestAssessmentSummary(
    StrictGitHubModel
):
    """Concise GitHub snapshot information returned to MCP clients."""

    repository: GitHubRepositoryRef

    pull_number: int = Field(gt=0)
    title: str = Field(min_length=1)
    html_url: str = Field(min_length=1)

    author_login: str = Field(min_length=1)

    state: str = Field(min_length=1)
    draft: bool
    merged: bool

    mergeable: bool | None = None
    mergeable_state: str | None = None

    base_ref: str = Field(min_length=1)
    base_sha: str = Field(min_length=1)

    head_ref: str = Field(min_length=1)
    head_sha: str = Field(min_length=1)

    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    diff_lines: int = Field(ge=0)

    changed_files_reported: int = Field(ge=0)
    files_collected: int = Field(ge=0)
    files_complete: bool

    review_history_count: int = Field(ge=0)
    effective_review_count: int = Field(ge=0)

    human_approval_count: int = Field(ge=0)

    head_commit_human_approval_count: int = Field(
        ge=0,
    )

    human_changes_requested_count: int = Field(
        ge=0,
    )

    checks: GitHubChecksSummary


class GitHubPullRequestAssessmentPolicy(
    StrictGitHubModel
):
    """Policy provenance used for one PR assessment."""

    source: PolicySource

    source_reference: str = Field(
        min_length=1,
    )

    used_defaults: bool

    policy_version: int = Field(
        ge=1,
    )

    trusted_ref: str = Field(
        min_length=1,
    )

    requested_path: str = Field(
        min_length=1,
    )

    policy_file_present: bool

    policy_file_sha: str | None = None

    policy_file_html_url: str | None = None


class GitHubPullRequestAssessmentResult(
    StrictGitHubModel
):
    """Complete evidence-backed GitHub PR assessment."""

    read_only: Literal[True] = True

    installation_id: int = Field(
        gt=0,
    )

    summary: GitHubPullRequestAssessmentSummary

    policy: GitHubPullRequestAssessmentPolicy

    conversion: GitHubContributionInputResult

    packet: MaintainerPolicyPacket

    evaluation: PolicyEvaluationResult


class PullRequestSnapshotLoader(Protocol):
    """PR snapshot behavior needed by the assessment service."""

    async def get_pull_request_snapshot(
        self,
        repository: GitHubRepositoryRef,
        pull_number: int,
    ) -> GitHubPullRequestSnapshot:
        """Load one pull-request snapshot."""

        ...


class RepositoryPolicyLoader(Protocol):
    """Repository policy behavior needed by the assessment service."""

    async def load_repository_policy(
        self,
        repository: GitHubRepositoryRef,
        *,
        policy_path: str = DEFAULT_POLICY_FILENAME,
        git_ref: str | None = None,
    ) -> GitHubRepositoryPolicyResult:
        """Load one repository policy."""

        ...


class GitHubPullRequestAssessmentRunner(Protocol):
    """Behavior used by the MCP assessment capability."""

    async def assess(
        self,
        request: GitHubPullRequestAssessmentRequest,
    ) -> GitHubPullRequestAssessmentResult:
        """Assess one pull request."""

        ...


def _validate_snapshot_repository(
    request: GitHubPullRequestAssessmentRequest,
    snapshot: GitHubPullRequestSnapshot,
) -> None:
    """Ensure the snapshot belongs to the requested repository."""

    requested_repository = (
        request.repository.full_name.casefold()
    )

    if (
        snapshot.repository.full_name.casefold()
        != requested_repository
    ):
        raise GitHubPullRequestAssessmentError(
            "The pull-request snapshot repository does not "
            "match the requested repository."
        )

    base_repository = (
        snapshot
        .pull_request
        .base
        .repository_full_name
    )

    if base_repository is None:
        raise GitHubPullRequestAssessmentError(
            "GitHub did not identify the pull request's "
            "base repository."
        )

    if (
        base_repository.casefold()
        != requested_repository
    ):
        raise GitHubPullRequestAssessmentError(
            "The pull request's base repository does not "
            "match the requested repository."
        )


def _validate_policy_result(
    request: GitHubPullRequestAssessmentRequest,
    snapshot: GitHubPullRequestSnapshot,
    policy_result: GitHubRepositoryPolicyResult,
) -> None:
    """Ensure policy provenance matches the trusted PR base."""

    if (
        policy_result.repository.full_name.casefold()
        != request.repository.full_name.casefold()
    ):
        raise GitHubPullRequestAssessmentError(
            "The loaded policy repository does not match "
            "the assessed pull request."
        )

    if (
        policy_result.requested_ref
        != snapshot.pull_request.base.sha
    ):
        raise GitHubPullRequestAssessmentError(
            "The loaded policy does not use the pull request's "
            "trusted base commit."
        )


def _build_summary(
    snapshot: GitHubPullRequestSnapshot,
) -> GitHubPullRequestAssessmentSummary:
    """Create the concise MCP-facing snapshot summary."""

    pull_request = snapshot.pull_request

    return GitHubPullRequestAssessmentSummary(
        repository=snapshot.repository,
        pull_number=pull_request.number,
        title=pull_request.title,
        html_url=pull_request.html_url,
        author_login=pull_request.author.login,
        state=pull_request.state,
        draft=pull_request.draft,
        merged=pull_request.merged,
        mergeable=pull_request.mergeable,
        mergeable_state=(
            pull_request.mergeable_state
        ),
        base_ref=pull_request.base.ref,
        base_sha=pull_request.base.sha,
        head_ref=pull_request.head.ref,
        head_sha=pull_request.head.sha,
        additions=pull_request.additions,
        deletions=pull_request.deletions,
        diff_lines=pull_request.diff_lines,
        changed_files_reported=(
            pull_request.changed_files_count
        ),
        files_collected=len(snapshot.files),
        files_complete=(
            not snapshot.files_truncated
        ),
        review_history_count=len(
            snapshot.reviews
        ),
        effective_review_count=len(
            snapshot.effective_reviews
        ),
        human_approval_count=(
            snapshot.human_approval_count
        ),
        head_commit_human_approval_count=(
            snapshot
            .head_commit_human_approval_count
        ),
        human_changes_requested_count=(
            snapshot
            .human_changes_requested_count
        ),
        checks=snapshot.checks,
    )


def _build_policy_summary(
    snapshot: GitHubPullRequestSnapshot,
    result: GitHubRepositoryPolicyResult,
) -> GitHubPullRequestAssessmentPolicy:
    """Create policy provenance for the MCP response."""

    policy_file = result.policy_file

    return GitHubPullRequestAssessmentPolicy(
        source=result.loaded_policy.source,
        source_reference=(
            result.loaded_policy.source_reference
        ),
        used_defaults=(
            result.loaded_policy.used_defaults
        ),
        policy_version=(
            result.loaded_policy.policy.version
        ),
        trusted_ref=(
            snapshot.pull_request.base.sha
        ),
        requested_path=result.requested_path,
        policy_file_present=(
            result.policy_file_present
        ),
        policy_file_sha=(
            policy_file.sha
            if policy_file is not None
            else None
        ),
        policy_file_html_url=(
            policy_file.html_url
            if policy_file is not None
            else None
        ),
    )


class GitHubPullRequestAssessmentService:
    """Orchestrate one end-to-end GitHub PR assessment."""

    def __init__(
        self,
        *,
        pull_request_loader: PullRequestSnapshotLoader,
        policy_loader: RepositoryPolicyLoader,
    ) -> None:
        self._pull_request_loader = (
            pull_request_loader
        )

        self._policy_loader = policy_loader

    async def assess(
        self,
        request: GitHubPullRequestAssessmentRequest,
    ) -> GitHubPullRequestAssessmentResult:
        """Assess one pull request against trusted base policy."""

        snapshot = await (
            self
            ._pull_request_loader
            .get_pull_request_snapshot(
                request.repository,
                request.pull_number,
            )
        )

        _validate_snapshot_repository(
            request,
            snapshot,
        )

        trusted_base_sha = (
            snapshot.pull_request.base.sha
        )

        policy_result = await (
            self
            ._policy_loader
            .load_repository_policy(
                request.repository,
                policy_path=request.policy_path,
                git_ref=trusted_base_sha,
            )
        )

        _validate_policy_result(
            request,
            snapshot,
            policy_result,
        )

        conversion = (
            build_contribution_policy_input_from_snapshot(
                snapshot,
                explicit_categories=(
                    request.explicit_categories
                ),
                options=(
                    request.conversion_options
                ),
            )
        )

        evaluation = evaluate_contribution_policy(
            policy=(
                policy_result.loaded_policy.policy
            ),
            contribution=(
                conversion.contribution
            ),
        )

        packet = build_maintainer_policy_packet(
            evaluation
        )

        return GitHubPullRequestAssessmentResult(
            installation_id=(
                request.installation_id
            ),
            summary=_build_summary(
                snapshot
            ),
            policy=_build_policy_summary(
                snapshot,
                policy_result,
            ),
            conversion=conversion,
            packet=packet,
            evaluation=evaluation,
        )