"""GitHub adaptation and orchestration for evidence-derived review cost."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol, Self

from pydantic import (
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)

from opensteward.github.assessments import (
    GitHubPullRequestAssessmentRequest,
    GitHubPullRequestAssessmentResult,
)
from opensteward.github.contribution_inputs import GitHubContributionInputOptions
from opensteward.github.historical_knowledge import (
    knowledge_repository_from_github,
)
from opensteward.github.historical_snapshot import (
    GitHubHistoricalKnowledgeSnapshotOptions,
)
from opensteward.github.models import GitHubRepositoryRef, StrictGitHubModel
from opensteward.github.related_work import (
    GitHubRelatedWorkQuery,
    GitHubRelatedWorkRequest,
    GitHubRelatedWorkResult,
)
from opensteward.knowledge import (
    DecisionSignificance,
    KnowledgeItemReference,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalQuery,
    KnowledgeRelatedWorkOptions,
    KnowledgeSourceKind,
)
from opensteward.policy import (
    DEFAULT_POLICY_FILENAME,
    ContributionCategory,
    PolicySource,
    match_protected_paths,
    normalize_repository_path,
)
from opensteward.review_intelligence import (
    ReviewCostAssessment,
    ReviewCostAssessmentInput,
    ReviewCostAssessmentOptions,
    ReviewCostChangedFile,
    ReviewCostChangeType,
    ReviewCostHistoricalContext,
    ReviewCostLevel,
)

DEFAULT_REVIEW_COST_PREFERRED_DIFF_SIZE = 700


class GitHubReviewCostError(ValueError):
    """Raised when GitHub review-cost orchestration is inconsistent."""


class GitHubReviewCostRequest(StrictGitHubModel):
    """Input for one live evidence-derived GitHub review-cost assessment."""

    installation_id: int = Field(gt=0)
    repository: GitHubRepositoryRef
    pull_number: int = Field(gt=0)
    policy_path: str = DEFAULT_POLICY_FILENAME
    explicit_categories: list[ContributionCategory] = Field(default_factory=list)
    conversion_options: GitHubContributionInputOptions = Field(
        default_factory=GitHubContributionInputOptions
    )
    snapshot_options: GitHubHistoricalKnowledgeSnapshotOptions = Field(
        default_factory=GitHubHistoricalKnowledgeSnapshotOptions
    )
    related_work_options: KnowledgeRelatedWorkOptions = Field(
        default_factory=KnowledgeRelatedWorkOptions
    )
    review_cost_options: ReviewCostAssessmentOptions = Field(
        default_factory=ReviewCostAssessmentOptions
    )

    @field_validator("policy_path")
    @classmethod
    def normalize_policy_path(cls, policy_path: str) -> str:
        """Reuse the existing assessment request's path validation."""

        return normalize_repository_path(policy_path)

    @field_validator("explicit_categories")
    @classmethod
    def validate_categories(
        cls,
        categories: list[ContributionCategory],
    ) -> list[ContributionCategory]:
        if len(categories) != len(set(categories)):
            raise ValueError("Explicit contribution categories must be unique.")
        return categories


class GitHubReviewCostPullRequestSummary(StrictGitHubModel):
    """Concise pull-request identity and structural totals."""

    repository: GitHubRepositoryRef
    pull_number: int = Field(gt=0)
    title: str = Field(min_length=1)
    url: str | None
    base_sha: str = Field(min_length=1)
    head_sha: str = Field(min_length=1)
    draft: bool
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)
    changed_files: int = Field(ge=0)
    commits: int = Field(ge=0)
    assessed_at: datetime

    @field_validator("assessed_at")
    @classmethod
    def normalize_assessed_at(cls, assessed_at: datetime) -> datetime:
        if assessed_at.tzinfo is None or assessed_at.utcoffset() is None:
            raise ValueError("assessed_at must be timezone-aware.")
        return assessed_at.astimezone(UTC)


def _stable_unique(*groups: list[str]) -> list[str]:
    values: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            if value not in seen:
                seen.add(value)
                values.append(value)
    return values


class GitHubReviewCostResult(StrictGitHubModel):
    """GitHub evidence and explainable review-cost output."""

    model_config = ConfigDict(json_schema_mode_override="serialization")

    repository: GitHubRepositoryRef
    pull_request: GitHubReviewCostPullRequestSummary
    pull_request_assessment: GitHubPullRequestAssessmentResult
    related_work: GitHubRelatedWorkResult
    review_cost: ReviewCostAssessment
    warnings: list[str]

    @field_validator("warnings")
    @classmethod
    def validate_warning_text(cls, warnings: list[str]) -> list[str]:
        if any(not warning for warning in warnings):
            raise ValueError("GitHub review-cost warnings must not be empty.")
        if len(warnings) != len(set(warnings)):
            raise ValueError("GitHub review-cost warnings must be unique.")
        return warnings

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.pull_request.repository != self.repository:
            raise ValueError("Pull-request summary must belong to repository.")
        assessment_summary = self.pull_request_assessment.summary
        if (
            assessment_summary.repository != self.repository
            or assessment_summary.pull_number != self.pull_request.pull_number
        ):
            raise ValueError(
                "Pull-request assessment must identify the summarized pull request."
            )
        if self.related_work.repository != self.repository:
            raise ValueError("Related work must belong to repository.")
        knowledge_repository = knowledge_repository_from_github(self.repository)
        if self.review_cost.repository != knowledge_repository:
            raise ValueError("Review cost must use the GitHub knowledge repository.")
        reference = self.review_cost.pull_request
        if (
            reference.repository != knowledge_repository
            or reference.item_type != KnowledgeItemType.PULL_REQUEST
            or reference.external_id != str(self.pull_request.pull_number)
        ):
            raise ValueError("Review cost must identify the summarized pull request.")
        if self.review_cost.assessed_at != self.pull_request.assessed_at:
            raise ValueError("Review-cost time must match pull-request summary time.")
        expected_warnings = _stable_unique(
            self.related_work.warnings,
            self.review_cost.warnings,
        )
        if self.warnings != expected_warnings:
            raise ValueError(
                "Warnings must equal stable related-work and review-cost warnings."
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_result(
        self,
        _handler: Any,
        info: Any,
    ) -> dict[str, Any]:
        """Serialize structured evidence without invocation credentials."""

        mode = "json" if info.mode == "json" else "python"
        serialized: dict[str, Any] = {
            "repository": self.repository.model_dump(
                mode=mode,
                exclude_computed_fields=True,
            ),
            "pull_request": self.pull_request.model_dump(
                mode=mode,
                exclude_computed_fields=True,
            ),
            "pull_request_assessment": (
                self.pull_request_assessment.model_dump(
                    mode=mode,
                    exclude_computed_fields=True,
                )
            ),
            "related_work": self.related_work.model_dump(mode=mode),
            "review_cost": self.review_cost.model_dump(
                mode=mode,
                exclude={
                    "repository": {"full_name", "key"},
                    "pull_request": {
                        "repository": {"full_name", "key"},
                        "key": True,
                    },
                },
            ),
            "warnings": list(self.warnings),
        }
        if not info.exclude_computed_fields:
            serialized.update(
                score=self.score,
                level=self.level,
                complete=self.complete,
            )
        return serialized

    @computed_field
    @property
    def score(self) -> int:
        return self.review_cost.score

    @computed_field
    @property
    def level(self) -> ReviewCostLevel:
        return self.review_cost.level

    @computed_field
    @property
    def complete(self) -> bool:
        return self.related_work.complete and self.review_cost.complete


class PullRequestAssessor(Protocol):
    async def assess(
        self,
        request: GitHubPullRequestAssessmentRequest,
    ) -> GitHubPullRequestAssessmentResult:
        ...


class GitHubRelatedWorkFinder(Protocol):
    async def find(
        self,
        request: GitHubRelatedWorkRequest,
    ) -> GitHubRelatedWorkResult:
        ...


class ReviewCostAssessor(Protocol):
    def assess(
        self,
        assessment_input: ReviewCostAssessmentInput,
        *,
        assessed_at: datetime,
        options: ReviewCostAssessmentOptions | None = None,
    ) -> ReviewCostAssessment:
        ...


class GitHubReviewCostRunner(Protocol):
    async def assess(
        self,
        request: GitHubReviewCostRequest,
    ) -> GitHubReviewCostResult:
        ...


def _assessment_request(
    request: GitHubReviewCostRequest,
) -> GitHubPullRequestAssessmentRequest:
    return GitHubPullRequestAssessmentRequest(
        installation_id=request.installation_id,
        repository=request.repository,
        pull_number=request.pull_number,
        policy_path=request.policy_path,
        explicit_categories=list(request.explicit_categories),
        conversion_options=request.conversion_options,
    )


def _validate_assessment(
    request: GitHubReviewCostRequest,
    assessment: GitHubPullRequestAssessmentResult,
) -> None:
    if (
        assessment.summary.repository != request.repository
        or assessment.summary.pull_number != request.pull_number
    ):
        raise GitHubReviewCostError(
            "Pull-request assessment identifies another repository or pull request."
        )
    if (
        assessment.snapshot.repository != request.repository
        or assessment.snapshot.pull_request.number != request.pull_number
    ):
        raise GitHubReviewCostError(
            "Pull-request assessment snapshot identifies another pull request."
        )


def _related_query(
    assessment: GitHubPullRequestAssessmentResult,
) -> GitHubRelatedWorkQuery:
    pull_request = assessment.snapshot.pull_request
    text = pull_request.title
    if pull_request.body:
        text = f"{text}\n\n{pull_request.body}"
    paths: list[str] = []
    seen: set[str] = set()
    for changed_file in assessment.snapshot.files:
        for path in (changed_file.filename, changed_file.previous_filename):
            if path is not None and path not in seen:
                seen.add(path)
                paths.append(path)
    return GitHubRelatedWorkQuery(
        text=text,
        labels=list(pull_request.labels),
        affected_paths=paths,
        components=[],
        item_types=[
            KnowledgeItemType.ISSUE,
            KnowledgeItemType.PULL_REQUEST,
            KnowledgeItemType.ADR,
        ],
        states=[],
    )


def _validate_related_work(
    result: GitHubRelatedWorkResult,
    *,
    repository: GitHubRepositoryRef,
    query: KnowledgeLexicalQuery,
    base_sha: str,
) -> None:
    if result.repository != repository:
        raise GitHubReviewCostError(
            "Related-work result belongs to another repository."
        )
    if result.related_work.query != query:
        raise GitHubReviewCostError(
            "Related-work result contains another derived query."
        )
    if result.snapshot.requested_ref != base_sha:
        raise GitHubReviewCostError(
            "Related-work result does not use the assessed pull request base SHA."
        )


def _change_type(status: str) -> ReviewCostChangeType:
    try:
        return ReviewCostChangeType(status.casefold())
    except ValueError:
        return ReviewCostChangeType.UNKNOWN


def _merge_conflict(
    assessment: GitHubPullRequestAssessmentResult,
) -> bool | None:
    pull_request = assessment.snapshot.pull_request
    if pull_request.mergeable is not None:
        return not pull_request.mergeable
    state = (
        pull_request.mergeable_state.casefold()
        if pull_request.mergeable_state
        else None
    )
    if state == "dirty":
        return True
    if state == "clean":
        return False
    return None


def _historical_context(
    related_work: GitHubRelatedWorkResult,
) -> ReviewCostHistoricalContext:
    matches = related_work.related_work.matches
    return ReviewCostHistoricalContext(
        related_match_count=related_work.related_work.returned_count,
        rejected_or_superseded_count=sum(
            match.item.state
            in {KnowledgeItemState.REJECTED, KnowledgeItemState.SUPERSEDED}
            for match in matches
        ),
        high_significance_count=sum(
            match.item.decision_significance
            in {DecisionSignificance.HIGH, DecisionSignificance.CRITICAL}
            for match in matches
        ),
        unresolved_count=sum(
            match.item.state
            in {
                KnowledgeItemState.OPEN,
                KnowledgeItemState.DRAFT,
                KnowledgeItemState.UNKNOWN,
            }
            for match in matches
        ),
        source_history_complete=related_work.source_history_complete,
        ranking_coverage_complete=related_work.ranking_coverage_complete,
        result_truncated=related_work.result_truncated,
    )


def _assessment_input(
    assessment: GitHubPullRequestAssessmentResult,
    related_work: GitHubRelatedWorkResult,
) -> ReviewCostAssessmentInput:
    snapshot = assessment.snapshot
    pull_request = snapshot.pull_request
    files = [
        ReviewCostChangedFile(
            path=file.filename,
            previous_path=file.previous_filename,
            change_type=_change_type(file.status),
            additions=file.additions,
            deletions=file.deletions,
        )
        for file in snapshot.files
    ]
    if len(files) != pull_request.changed_files_count:
        raise GitHubReviewCostError(
            "Review-cost input requires the complete pull-request file list."
        )
    policy_present = (
        assessment.policy.policy_file_present
        and assessment.policy.source != PolicySource.DEFAULT
    )
    all_paths = [
        path
        for file in files
        for path in (file.path, file.previous_path)
        if path is not None
    ]
    protected_paths: list[str] = []
    if policy_present:
        matched = {
            match.path
            for match in match_protected_paths(
                all_paths,
                assessment.repository_policy.protected_paths,
            )
        }
        protected_paths = list(
            dict.fromkeys(path for path in all_paths if path in matched)
        )
    knowledge_repository = knowledge_repository_from_github(snapshot.repository)
    reference = KnowledgeItemReference(
        repository=knowledge_repository,
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id=str(pull_request.number),
        source_kind=KnowledgeSourceKind.GITHUB,
        title=pull_request.title,
        url=pull_request.html_url,
    )

    # The current policy schema does not define required check names.
    return ReviewCostAssessmentInput(
        repository=knowledge_repository,
        pull_request=reference,
        files=files,
        commit_count=pull_request.commits_count,
        preferred_diff_size=(
            assessment.repository_policy.pull_requests.preferred_maximum_diff_lines
            if policy_present
            else DEFAULT_REVIEW_COST_PREFERRED_DIFF_SIZE
        ),
        policy_present=policy_present,
        protected_changed_paths=protected_paths,
        draft=pull_request.draft,
        merge_conflict=_merge_conflict(assessment),
        required_checks_total=0,
        required_checks_passed=0,
        required_checks_failed=0,
        required_checks_pending=0,
        approval_count=snapshot.human_approval_count,
        changes_requested_count=snapshot.human_changes_requested_count,
        historical_context=_historical_context(related_work),
    )


def _normalize_clock(clock_value: datetime) -> datetime:
    if clock_value.tzinfo is None or clock_value.utcoffset() is None:
        raise GitHubReviewCostError(
            "Review-cost clock must return a timezone-aware datetime."
        )
    return clock_value.astimezone(UTC)


def _default_clock() -> datetime:
    return datetime.now(UTC)


class GitHubReviewCostService:
    """Orchestrate PR assessment, related work, and review-cost scoring."""

    def __init__(
        self,
        *,
        pull_request_assessor: PullRequestAssessor,
        related_work_finder: GitHubRelatedWorkFinder,
        review_cost_assessor: ReviewCostAssessor,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._pull_request_assessor = pull_request_assessor
        self._related_work_finder = related_work_finder
        self._review_cost_assessor = review_cost_assessor
        self._clock = clock or _default_clock

    async def assess(
        self,
        request: GitHubReviewCostRequest,
    ) -> GitHubReviewCostResult:
        """Derive review cost from live read-only GitHub evidence."""

        assessment = await self._pull_request_assessor.assess(
            _assessment_request(request)
        )
        _validate_assessment(request, assessment)

        query_input = _related_query(assessment)
        knowledge_repository = knowledge_repository_from_github(request.repository)
        query = query_input.to_knowledge_query(knowledge_repository)
        base_sha = assessment.snapshot.pull_request.base.sha
        related_request = GitHubRelatedWorkRequest(
            installation_id=request.installation_id,
            repository=request.repository,
            git_ref=base_sha,
            query=query_input,
            snapshot_options=request.snapshot_options,
            related_work_options=request.related_work_options,
        )
        related_work = await self._related_work_finder.find(related_request)
        _validate_related_work(
            related_work,
            repository=request.repository,
            query=query,
            base_sha=base_sha,
        )

        assessment_input = _assessment_input(assessment, related_work)
        assessed_at = _normalize_clock(self._clock())
        review_cost = self._review_cost_assessor.assess(
            assessment_input,
            assessed_at=assessed_at,
            options=request.review_cost_options,
        )
        if (
            review_cost.repository != assessment_input.repository
            or review_cost.pull_request != assessment_input.pull_request
        ):
            raise GitHubReviewCostError(
                "Review-cost assessment identifies another repository or pull request."
            )
        pull_request = assessment.snapshot.pull_request
        summary = GitHubReviewCostPullRequestSummary(
            repository=request.repository,
            pull_number=pull_request.number,
            title=pull_request.title,
            url=pull_request.html_url,
            base_sha=pull_request.base.sha,
            head_sha=pull_request.head.sha,
            draft=pull_request.draft,
            additions=pull_request.additions,
            deletions=pull_request.deletions,
            changed_files=len(assessment_input.files),
            commits=pull_request.commits_count,
            assessed_at=assessed_at,
        )
        warnings = _stable_unique(
            related_work.warnings,
            review_cost.warnings,
        )
        return GitHubReviewCostResult(
            repository=request.repository,
            pull_request=summary,
            pull_request_assessment=assessment,
            related_work=related_work,
            review_cost=review_cost,
            warnings=warnings,
        )
