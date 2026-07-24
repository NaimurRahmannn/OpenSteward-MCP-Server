"""GitHub pull-request snapshot collection."""

import asyncio
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from urllib.parse import quote

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from opensteward.github.models import (
    GitHubRepositoryRef,
    StrictGitHubModel,
)
from opensteward.github.rest_client import (
    DEFAULT_GITHUB_ACCEPT,
    GitHubRestResponse,
)
from opensteward.policy import normalize_repository_path

GITHUB_PAGE_SIZE = 100
MAX_PULL_REQUEST_FILES = 3_000
MAX_PAGINATION_PAGES = 100


class GitHubPullRequestSnapshotError(RuntimeError):
    """Raised when a complete pull-request snapshot cannot be collected."""


class GitHubPullRequestReviewState(StrEnum):
    """Review states returned by GitHub."""

    APPROVED = "APPROVED"
    CHANGES_REQUESTED = "CHANGES_REQUESTED"
    COMMENTED = "COMMENTED"
    DISMISSED = "DISMISSED"
    PENDING = "PENDING"


class GitHubCheckRunStatus(StrEnum):
    """Current execution state of a GitHub check run."""

    QUEUED = "queued"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    WAITING = "waiting"
    REQUESTED = "requested"
    PENDING = "pending"


class GitHubCheckRunConclusion(StrEnum):
    """Final conclusion of a completed GitHub check run."""

    ACTION_REQUIRED = "action_required"
    CANCELLED = "cancelled"
    FAILURE = "failure"
    NEUTRAL = "neutral"
    SUCCESS = "success"
    SKIPPED = "skipped"
    STALE = "stale"
    TIMED_OUT = "timed_out"


class GitHubChecksState(StrEnum):
    """Normalized combined state of collected check runs."""

    NONE = "none"
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class _GitHubApiModel(BaseModel):
    """Base model for forward-compatible GitHub API responses."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )


class GitHubPullRequestActor(StrictGitHubModel):
    """Minimal account information for a PR participant."""

    id: int = Field(gt=0)
    login: str = Field(min_length=1)
    account_type: str = Field(min_length=1)


class GitHubPullRequestBranch(StrictGitHubModel):
    """One side of a pull request."""

    ref: str = Field(min_length=1)
    sha: str = Field(min_length=1)
    repository_full_name: str | None = None


class GitHubPullRequestDetails(StrictGitHubModel):
    """Normalized pull-request metadata."""

    id: int = Field(gt=0)
    number: int = Field(gt=0)

    title: str = Field(min_length=1)
    body: str | None = None

    state: str = Field(min_length=1)
    draft: bool
    merged: bool

    mergeable: bool | None = None
    mergeable_state: str | None = None

    html_url: str = Field(min_length=1)
    labels: list[str] = Field(default_factory=list)

    author: GitHubPullRequestActor

    base: GitHubPullRequestBranch
    head: GitHubPullRequestBranch

    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changed_files_count: int = Field(ge=0)
    commits_count: int = Field(ge=0)

    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    merged_at: datetime | None = None

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[str]) -> list[str]:
        """Require non-empty labels that are unique case-insensitively."""

        if any(not label for label in labels):
            raise ValueError("Pull-request labels must not be empty.")
        keys = [label.casefold() for label in labels]
        if len(keys) != len(set(keys)):
            raise ValueError("Pull-request labels must be unique case-insensitively.")
        return labels

    @computed_field
    @property
    def diff_lines(self) -> int:
        """Return added plus deleted lines."""

        return self.additions + self.deletions


class GitHubPullRequestFile(StrictGitHubModel):
    """Normalized file information from a pull request."""

    filename: str = Field(min_length=1)
    status: str = Field(min_length=1)

    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changes: int = Field(ge=0)

    previous_filename: str | None = None

    @field_validator(
        "filename",
        "previous_filename",
    )
    @classmethod
    def normalize_file_path(
        cls,
        value: str | None,
    ) -> str | None:
        """Normalize repository-relative file paths."""

        if value is None:
            return None

        return normalize_repository_path(value)


class GitHubPullRequestReview(StrictGitHubModel):
    """One normalized GitHub pull-request review."""

    id: int = Field(gt=0)
    reviewer: GitHubPullRequestActor | None = None

    state: GitHubPullRequestReviewState

    body: str | None = None
    html_url: str | None = None

    submitted_at: datetime | None = None
    commit_id: str | None = None

    on_head_commit: bool = False


class GitHubCheckRun(StrictGitHubModel):
    """One check run associated with the PR head commit."""

    id: int = Field(gt=0)
    name: str = Field(min_length=1)
    head_sha: str = Field(min_length=1)

    status: GitHubCheckRunStatus
    conclusion: GitHubCheckRunConclusion | None = None

    html_url: str | None = None
    details_url: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None

    app_name: str | None = None
    app_slug: str | None = None


class GitHubChecksSummary(StrictGitHubModel):
    """Combined state derived from collected check runs."""

    state: GitHubChecksState

    total_count: int = Field(ge=0)
    pending_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    skipped_count: int = Field(ge=0)


class GitHubPullRequestSnapshot(StrictGitHubModel):
    """Complete normalized snapshot of one GitHub pull request."""

    repository: GitHubRepositoryRef
    pull_request: GitHubPullRequestDetails

    files: list[GitHubPullRequestFile] = Field(
        default_factory=list,
    )

    reviews: list[GitHubPullRequestReview] = Field(
        default_factory=list,
    )

    effective_reviews: list[GitHubPullRequestReview] = Field(
        default_factory=list,
    )

    check_runs: list[GitHubCheckRun] = Field(
        default_factory=list,
    )

    checks: GitHubChecksSummary

    @computed_field
    @property
    def changed_paths(self) -> list[str]:
        """Return changed file paths in GitHub order."""

        return [
            file.filename
            for file in self.files
        ]

    @computed_field
    @property
    def files_truncated(self) -> bool:
        """Return whether GitHub omitted files from the response."""

        return (
            len(self.files)
            < self.pull_request.changed_files_count
        )

    @computed_field
    @property
    def human_approval_count(self) -> int:
        """Count current human approvals, excluding the PR author."""

        return sum(
            review.state
            == GitHubPullRequestReviewState.APPROVED
            and _is_human_non_author_review(
                review,
                author_id=self.pull_request.author.id,
            )
            for review in self.effective_reviews
        )

    @computed_field
    @property
    def head_commit_human_approval_count(self) -> int:
        """Count human approvals submitted for the current head SHA."""

        return sum(
            review.state
            == GitHubPullRequestReviewState.APPROVED
            and review.on_head_commit
            and _is_human_non_author_review(
                review,
                author_id=self.pull_request.author.id,
            )
            for review in self.effective_reviews
        )

    @computed_field
    @property
    def human_changes_requested_count(self) -> int:
        """Count current human changes-requested decisions."""

        return sum(
            review.state
            == GitHubPullRequestReviewState.CHANGES_REQUESTED
            and _is_human_non_author_review(
                review,
                author_id=self.pull_request.author.id,
            )
            for review in self.effective_reviews
        )


class GitHubJsonClient(Protocol):
    """REST-client behavior required by the snapshot service."""

    async def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        response_type: Any = Any,
        accept: str = DEFAULT_GITHUB_ACCEPT,
    ) -> GitHubRestResponse[Any]:
        """Retrieve and validate one GitHub JSON response."""

        ...


class _ApiActor(_GitHubApiModel):
    id: int = Field(gt=0)
    login: str = Field(min_length=1)
    type: str = Field(min_length=1)


class _ApiLabel(_GitHubApiModel):
    name: str = Field(min_length=1)


class _ApiRepository(_GitHubApiModel):
    id: int = Field(gt=0)
    name: str = Field(min_length=1)
    full_name: str = Field(min_length=1)


class _ApiPullRequestBranch(_GitHubApiModel):
    ref: str = Field(min_length=1)
    sha: str = Field(min_length=1)
    repo: _ApiRepository | None = None


class _ApiPullRequest(_GitHubApiModel):
    id: int = Field(gt=0)
    number: int = Field(gt=0)

    title: str = Field(min_length=1)
    body: str | None = None

    state: str = Field(min_length=1)
    draft: bool = False
    merged: bool = False

    mergeable: bool | None = None
    mergeable_state: str | None = None

    html_url: str = Field(min_length=1)
    labels: list[_ApiLabel] = Field(default_factory=list)

    user: _ApiActor
    base: _ApiPullRequestBranch
    head: _ApiPullRequestBranch

    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changed_files: int = Field(ge=0)
    commits: int = Field(ge=0)

    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    merged_at: datetime | None = None


class _ApiPullRequestFile(_GitHubApiModel):
    filename: str = Field(min_length=1)
    status: str = Field(min_length=1)

    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changes: int = Field(ge=0)

    previous_filename: str | None = None


class _ApiPullRequestReview(_GitHubApiModel):
    id: int = Field(gt=0)

    user: _ApiActor | None = None
    state: GitHubPullRequestReviewState

    body: str | None = None
    html_url: str | None = None

    submitted_at: datetime | None = None
    commit_id: str | None = None


class _ApiCheckApp(_GitHubApiModel):
    name: str | None = None
    slug: str | None = None


class _ApiCheckRun(_GitHubApiModel):
    id: int = Field(gt=0)
    name: str = Field(min_length=1)
    head_sha: str = Field(min_length=1)

    status: GitHubCheckRunStatus
    conclusion: GitHubCheckRunConclusion | None = None

    html_url: str | None = None
    details_url: str | None = None

    started_at: datetime | None = None
    completed_at: datetime | None = None

    app: _ApiCheckApp | None = None


class _ApiCheckRunsPage(_GitHubApiModel):
    total_count: int = Field(ge=0)
    check_runs: list[_ApiCheckRun] = Field(
        default_factory=list,
    )


_FAILURE_CONCLUSIONS = {
    GitHubCheckRunConclusion.ACTION_REQUIRED,
    GitHubCheckRunConclusion.CANCELLED,
    GitHubCheckRunConclusion.FAILURE,
    GitHubCheckRunConclusion.STALE,
    GitHubCheckRunConclusion.TIMED_OUT,
}


def _build_repository_path(
    repository: GitHubRepositoryRef,
) -> str:
    """Build a safely encoded repository REST path."""

    owner = quote(
        repository.owner,
        safe="",
    )

    name = quote(
        repository.name,
        safe="",
    )

    return f"/repos/{owner}/{name}"


def _validate_pull_number(
    pull_number: int,
) -> int:
    """Require a positive pull-request number."""

    if pull_number <= 0:
        raise ValueError(
            "GitHub pull-request number must be positive."
        )

    return pull_number


def _convert_actor(
    actor: _ApiActor | None,
) -> GitHubPullRequestActor | None:
    """Convert a GitHub API actor into the public model."""

    if actor is None:
        return None

    return GitHubPullRequestActor(
        id=actor.id,
        login=actor.login,
        account_type=actor.type,
    )


def _convert_branch(
    branch: _ApiPullRequestBranch,
) -> GitHubPullRequestBranch:
    """Convert one pull-request branch."""

    return GitHubPullRequestBranch(
        ref=branch.ref,
        sha=branch.sha,
        repository_full_name=(
            branch.repo.full_name
            if branch.repo is not None
            else None
        ),
    )


def _convert_pull_request(
    pull_request: _ApiPullRequest,
) -> GitHubPullRequestDetails:
    """Convert the main pull-request response."""

    author = _convert_actor(
        pull_request.user
    )

    assert author is not None

    labels: list[str] = []
    seen_labels: set[str] = set()
    for label in pull_request.labels:
        key = label.name.casefold()
        if key not in seen_labels:
            seen_labels.add(key)
            labels.append(label.name)

    return GitHubPullRequestDetails(
        id=pull_request.id,
        number=pull_request.number,
        title=pull_request.title,
        body=pull_request.body,
        state=pull_request.state,
        draft=pull_request.draft,
        merged=pull_request.merged,
        mergeable=pull_request.mergeable,
        mergeable_state=pull_request.mergeable_state,
        html_url=pull_request.html_url,
        labels=labels,
        author=author,
        base=_convert_branch(
            pull_request.base
        ),
        head=_convert_branch(
            pull_request.head
        ),
        additions=pull_request.additions,
        deletions=pull_request.deletions,
        changed_files_count=(
            pull_request.changed_files
        ),
        commits_count=pull_request.commits,
        created_at=pull_request.created_at,
        updated_at=pull_request.updated_at,
        closed_at=pull_request.closed_at,
        merged_at=pull_request.merged_at,
    )


def _convert_file(
    file: _ApiPullRequestFile,
) -> GitHubPullRequestFile:
    """Convert one changed-file response."""

    return GitHubPullRequestFile(
        filename=file.filename,
        status=file.status,
        additions=file.additions,
        deletions=file.deletions,
        changes=file.changes,
        previous_filename=file.previous_filename,
    )


def _convert_review(
    review: _ApiPullRequestReview,
    *,
    head_sha: str,
) -> GitHubPullRequestReview:
    """Convert one review and mark whether it targets the head SHA."""

    return GitHubPullRequestReview(
        id=review.id,
        reviewer=_convert_actor(
            review.user
        ),
        state=review.state,
        body=review.body,
        html_url=review.html_url,
        submitted_at=review.submitted_at,
        commit_id=review.commit_id,
        on_head_commit=(
            review.commit_id == head_sha
        ),
    )


def _convert_check_run(
    check_run: _ApiCheckRun,
) -> GitHubCheckRun:
    """Convert one check-run response."""

    return GitHubCheckRun(
        id=check_run.id,
        name=check_run.name,
        head_sha=check_run.head_sha,
        status=check_run.status,
        conclusion=check_run.conclusion,
        html_url=check_run.html_url,
        details_url=check_run.details_url,
        started_at=check_run.started_at,
        completed_at=check_run.completed_at,
        app_name=(
            check_run.app.name
            if check_run.app is not None
            else None
        ),
        app_slug=(
            check_run.app.slug
            if check_run.app is not None
            else None
        ),
    )


def _build_effective_reviews(
    reviews: list[GitHubPullRequestReview],
) -> list[GitHubPullRequestReview]:
    """Calculate the latest effective decision for each reviewer.

    Comment-only and pending reviews do not replace an existing
    approval or changes-requested decision. A dismissed review removes
    the reviewer's active decision.
    """

    decisions: dict[
        int,
        GitHubPullRequestReview,
    ] = {}

    for review in reviews:
        reviewer = review.reviewer

        if reviewer is None:
            continue

        if review.state in {
            GitHubPullRequestReviewState.APPROVED,
            GitHubPullRequestReviewState.CHANGES_REQUESTED,
        }:
            decisions[reviewer.id] = review
            continue

        if (
            review.state
            == GitHubPullRequestReviewState.DISMISSED
        ):
            decisions.pop(
                reviewer.id,
                None,
            )

    return list(
        decisions.values()
    )


def _is_human_non_author_review(
    review: GitHubPullRequestReview,
    *,
    author_id: int,
) -> bool:
    """Return whether a review represents a human other than the author."""

    reviewer = review.reviewer

    if reviewer is None:
        return False

    return (
        reviewer.id != author_id
        and reviewer.account_type.casefold()
        == "user"
    )


def _summarize_checks(
    check_runs: list[GitHubCheckRun],
) -> GitHubChecksSummary:
    """Build a deterministic combined checks summary."""

    pending_count = 0
    success_count = 0
    failure_count = 0
    neutral_count = 0
    skipped_count = 0

    for check_run in check_runs:
        if (
            check_run.status
            != GitHubCheckRunStatus.COMPLETED
            or check_run.conclusion is None
        ):
            pending_count += 1
            continue

        if (
            check_run.conclusion
            == GitHubCheckRunConclusion.SUCCESS
        ):
            success_count += 1
        elif (
            check_run.conclusion
            == GitHubCheckRunConclusion.NEUTRAL
        ):
            neutral_count += 1
        elif (
            check_run.conclusion
            == GitHubCheckRunConclusion.SKIPPED
        ):
            skipped_count += 1
        elif check_run.conclusion in _FAILURE_CONCLUSIONS:
            failure_count += 1

    if failure_count:
        state = GitHubChecksState.FAILURE
    elif pending_count:
        state = GitHubChecksState.PENDING
    elif check_runs:
        state = GitHubChecksState.SUCCESS
    else:
        state = GitHubChecksState.NONE

    return GitHubChecksSummary(
        state=state,
        total_count=len(check_runs),
        pending_count=pending_count,
        success_count=success_count,
        failure_count=failure_count,
        neutral_count=neutral_count,
        skipped_count=skipped_count,
    )


class GitHubPullRequestService:
    """Collect normalized pull-request snapshots from GitHub."""

    def __init__(
        self,
        *,
        rest_client: GitHubJsonClient,
    ) -> None:
        self._rest_client = rest_client

    async def get_pull_request_snapshot(
        self,
        repository: GitHubRepositoryRef,
        pull_number: int,
    ) -> GitHubPullRequestSnapshot:
        """Collect metadata, files, reviews, and checks for one PR."""

        validated_number = _validate_pull_number(
            pull_number
        )

        repository_path = _build_repository_path(
            repository
        )

        pull_path = (
            f"{repository_path}"
            f"/pulls/{validated_number}"
        )

        pull_response = await self._rest_client.get_json(
            pull_path,
            response_type=_ApiPullRequest,
        )

        pull_request = _convert_pull_request(
            pull_response.data
        )

        files_task = self._collect_files(
            repository_path=repository_path,
            pull_number=validated_number,
        )

        reviews_task = self._collect_reviews(
            repository_path=repository_path,
            pull_number=validated_number,
            head_sha=pull_request.head.sha,
        )

        checks_task = self._collect_check_runs(
            repository_path=repository_path,
            head_sha=pull_request.head.sha,
        )

        (
            files,
            reviews,
            check_runs,
        ) = await asyncio.gather(
            files_task,
            reviews_task,
            checks_task,
        )

        return GitHubPullRequestSnapshot(
            repository=repository,
            pull_request=pull_request,
            files=files,
            reviews=reviews,
            effective_reviews=(
                _build_effective_reviews(
                    reviews
                )
            ),
            check_runs=check_runs,
            checks=_summarize_checks(
                check_runs
            ),
        )

    async def _collect_files(
        self,
        *,
        repository_path: str,
        pull_number: int,
    ) -> list[GitHubPullRequestFile]:
        """Collect changed files up to GitHub's documented limit."""

        path = (
            f"{repository_path}"
            f"/pulls/{pull_number}/files"
        )

        files: list[
            GitHubPullRequestFile
        ] = []

        for page in range(
            1,
            MAX_PAGINATION_PAGES + 1,
        ):
            response = await self._rest_client.get_json(
                path,
                params={
                    "per_page": GITHUB_PAGE_SIZE,
                    "page": page,
                },
                response_type=list[
                    _ApiPullRequestFile
                ],
            )

            page_items = response.data

            files.extend(
                _convert_file(item)
                for item in page_items
            )

            if (
                len(files)
                >= MAX_PULL_REQUEST_FILES
            ):
                return files[
                    :MAX_PULL_REQUEST_FILES
                ]

            if len(page_items) < GITHUB_PAGE_SIZE:
                return files

        raise GitHubPullRequestSnapshotError(
            "Pull-request file pagination exceeded the "
            "configured safety limit."
        )

    async def _collect_reviews(
        self,
        *,
        repository_path: str,
        pull_number: int,
        head_sha: str,
    ) -> list[GitHubPullRequestReview]:
        """Collect the chronological pull-request review history."""

        path = (
            f"{repository_path}"
            f"/pulls/{pull_number}/reviews"
        )

        reviews: list[
            GitHubPullRequestReview
        ] = []

        for page in range(
            1,
            MAX_PAGINATION_PAGES + 1,
        ):
            response = await self._rest_client.get_json(
                path,
                params={
                    "per_page": GITHUB_PAGE_SIZE,
                    "page": page,
                },
                response_type=list[
                    _ApiPullRequestReview
                ],
            )

            page_items = response.data

            reviews.extend(
                _convert_review(
                    item,
                    head_sha=head_sha,
                )
                for item in page_items
            )

            if len(page_items) < GITHUB_PAGE_SIZE:
                return reviews

        raise GitHubPullRequestSnapshotError(
            "Pull-request review pagination exceeded the "
            "configured safety limit."
        )

    async def _collect_check_runs(
        self,
        *,
        repository_path: str,
        head_sha: str,
    ) -> list[GitHubCheckRun]:
        """Collect latest check runs for the current head commit."""

        encoded_sha = quote(
            head_sha,
            safe="",
        )

        path = (
            f"{repository_path}"
            f"/commits/{encoded_sha}/check-runs"
        )

        check_runs: list[
            GitHubCheckRun
        ] = []

        expected_total: int | None = None

        for page in range(
            1,
            MAX_PAGINATION_PAGES + 1,
        ):
            response = await self._rest_client.get_json(
                path,
                params={
                    "filter": "latest",
                    "per_page": GITHUB_PAGE_SIZE,
                    "page": page,
                },
                response_type=_ApiCheckRunsPage,
            )

            page_data = response.data

            if expected_total is None:
                expected_total = (
                    page_data.total_count
                )

            check_runs.extend(
                _convert_check_run(item)
                for item in page_data.check_runs
            )

            if (
                len(check_runs)
                >= expected_total
                or len(page_data.check_runs)
                < GITHUB_PAGE_SIZE
            ):
                return check_runs

        raise GitHubPullRequestSnapshotError(
            "Check-run pagination exceeded the configured "
            "safety limit."
        )
