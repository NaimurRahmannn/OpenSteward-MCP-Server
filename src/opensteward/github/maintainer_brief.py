"""GitHub adaptation and orchestration for structured maintainer briefs."""

from typing import Annotated, Any, Literal, Protocol, Self

from pydantic import (
    ConfigDict,
    Field,
    PlainSerializer,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)

from opensteward.github.assessments import (
    GitHubPullRequestAssessmentPolicy,
    GitHubPullRequestAssessmentResult,
    GitHubPullRequestAssessmentSummary,
)
from opensteward.github.contribution_inputs import (
    GitHubContributionInputOptions,
    GitHubContributionInputResult,
)
from opensteward.github.historical_knowledge import (
    knowledge_repository_from_github,
)
from opensteward.github.historical_snapshot import (
    GitHubHistoricalKnowledgeSnapshotOptions,
)
from opensteward.github.models import GitHubRepositoryRef, StrictGitHubModel
from opensteward.github.related_work import GitHubRelatedWorkResult
from opensteward.github.review_cost import (
    GitHubReviewCostError,
    GitHubReviewCostPullRequestSummary,
    GitHubReviewCostRequest,
    GitHubReviewCostResult,
    _assessment_input,
)
from opensteward.knowledge import KnowledgeItemType, KnowledgeRelatedWorkOptions
from opensteward.maintainer_brief import (
    MaintainerAttentionRecommendation,
    MaintainerBrief,
    MaintainerBriefInput,
    MaintainerPathRiskSummary,
    MaintainerReadinessSummary,
    MaintainerReviewRoute,
)
from opensteward.policy import (
    DEFAULT_POLICY_FILENAME,
    ContributionCategory,
    MaintainerPolicyPacket,
    PolicyEvaluationResult,
    PolicyFindingStatus,
    normalize_repository_path,
)
from opensteward.review_intelligence import (
    ReviewCostAssessment,
    ReviewCostAssessmentInput,
    ReviewCostAssessmentOptions,
    ReviewCostLevel,
    ReviewCostPathCategory,
    classify_review_cost_paths,
)


class GitHubMaintainerBriefError(ValueError):
    """Raised when GitHub maintainer-brief orchestration is inconsistent."""


class GitHubMaintainerBriefRequest(StrictGitHubModel):
    """Input for one live read-only GitHub maintainer brief."""

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

    def to_review_cost_request(self) -> GitHubReviewCostRequest:
        """Convert to the existing Phase 5 request without sharing caller lists."""

        return GitHubReviewCostRequest(
            installation_id=self.installation_id,
            repository=self.repository,
            pull_number=self.pull_number,
            policy_path=self.policy_path,
            explicit_categories=list(self.explicit_categories),
            conversion_options=self.conversion_options,
            snapshot_options=self.snapshot_options,
            related_work_options=self.related_work_options,
            review_cost_options=self.review_cost_options,
        )


class _GitHubMaintainerAssessmentOutput(StrictGitHubModel):
    """Established public assessment fields safe for nested output."""

    read_only: Literal[True] = True
    summary: GitHubPullRequestAssessmentSummary
    policy: GitHubPullRequestAssessmentPolicy
    conversion: GitHubContributionInputResult
    packet: MaintainerPolicyPacket
    evaluation: PolicyEvaluationResult


def _redact_pull_request_assessment(
    assessment: GitHubPullRequestAssessmentResult,
) -> _GitHubMaintainerAssessmentOutput:
    return _GitHubMaintainerAssessmentOutput(
        read_only=assessment.read_only,
        summary=assessment.summary,
        policy=assessment.policy,
        conversion=assessment.conversion,
        packet=assessment.packet,
        evaluation=assessment.evaluation,
    )


_MaintainerAssessmentResult = Annotated[
    GitHubPullRequestAssessmentResult,
    PlainSerializer(
        _redact_pull_request_assessment,
        return_type=_GitHubMaintainerAssessmentOutput,
    ),
]


def _stable_unique(*groups: list[str]) -> list[str]:
    return list(dict.fromkeys(value for group in groups for value in group))


class GitHubMaintainerBriefResult(StrictGitHubModel):
    """Complete GitHub evidence and credential-redacted maintainer brief."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_mode_override="serialization",
    )

    repository: GitHubRepositoryRef
    pull_request: GitHubReviewCostPullRequestSummary
    pull_request_assessment: _MaintainerAssessmentResult
    related_work: GitHubRelatedWorkResult
    review_cost: ReviewCostAssessment
    brief: MaintainerBrief
    warnings: list[str]

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        if any(not warning for warning in warnings):
            raise ValueError("GitHub maintainer-brief warnings must not be empty.")
        if len(warnings) != len(set(warnings)):
            raise ValueError("GitHub maintainer-brief warnings must be unique.")
        return warnings

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        if self.pull_request.repository != self.repository:
            raise ValueError("Pull-request summary must belong to repository.")
        assessment_summary = self.pull_request_assessment.summary
        if (
            assessment_summary.repository != self.repository
            or assessment_summary.pull_number != self.pull_request.pull_number
            or self.related_work.repository != self.repository
        ):
            raise ValueError("GitHub evidence must identify the summarized pull request.")
        knowledge_repository = knowledge_repository_from_github(self.repository)
        reference = self.review_cost.pull_request
        if (
            self.review_cost.repository != knowledge_repository
            or reference.repository != knowledge_repository
            or reference.item_type != KnowledgeItemType.PULL_REQUEST
            or reference.external_id != str(self.pull_request.pull_number)
            or self.brief.repository != knowledge_repository
            or self.brief.pull_request != reference
        ):
            raise ValueError("Brief evidence must use the GitHub knowledge identities.")
        if (
            self.brief.generated_at != self.pull_request.assessed_at
            or self.brief.generated_at != self.review_cost.assessed_at
        ):
            raise ValueError("Brief time must match the review-cost assessment time.")
        if self.brief.related_work != self.related_work.related_work:
            raise ValueError("Brief related work must match the GitHub related work.")
        expected = _stable_unique(
            self.related_work.warnings,
            self.review_cost.warnings,
            self.brief.warnings,
        )
        if self.warnings != expected:
            raise ValueError(
                "Warnings must equal stable related-work, review-cost, and brief warnings."
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_result(
        self,
        _handler: Any,
        info: Any,
    ) -> dict[str, Any]:
        """Serialize all public evidence while redacting invocation credentials."""

        mode = "json" if info.mode == "json" else "python"
        brief = self.brief.model_dump(
            mode=mode,
            exclude={
                "repository": {"full_name", "key"},
                "pull_request": {
                    "repository": {"full_name", "key"},
                    "key": True,
                },
                "attention": {
                    "repository": {"full_name", "key"},
                    "pull_request": {
                        "repository": {"full_name", "key"},
                        "key": True,
                    },
                },
                "review_cost": {
                    "repository": {"full_name", "key"},
                    "pull_request": {
                        "repository": {"full_name", "key"},
                        "key": True,
                    },
                },
            },
        )
        brief["related_work"] = self.brief.related_work.model_dump(
            mode=mode,
            exclude_computed_fields=True,
        )
        serialized: dict[str, Any] = {
            "repository": self.repository.model_dump(
                mode=mode,
                exclude_computed_fields=True,
            ),
            "pull_request": self.pull_request.model_dump(
                mode=mode,
                exclude_computed_fields=True,
            ),
            "pull_request_assessment": _redact_pull_request_assessment(
                self.pull_request_assessment
            ).model_dump(
                mode=mode,
                exclude_computed_fields=True,
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
            "brief": brief,
            "warnings": list(self.warnings),
        }
        if not info.exclude_computed_fields:
            serialized.update(
                recommendation=self.recommendation,
                routes=self.routes,
                review_cost_score=self.review_cost_score,
                review_cost_level=self.review_cost_level,
                returned_related_work=self.returned_related_work,
                complete=self.complete,
            )
        return serialized

    @computed_field
    @property
    def recommendation(self) -> MaintainerAttentionRecommendation:
        return self.brief.recommendation

    @computed_field
    @property
    def routes(self) -> list[MaintainerReviewRoute]:
        return list(self.brief.routes)

    @computed_field
    @property
    def review_cost_score(self) -> int:
        return self.brief.review_cost_score

    @computed_field
    @property
    def review_cost_level(self) -> ReviewCostLevel:
        return self.brief.review_cost_level

    @computed_field
    @property
    def returned_related_work(self) -> int:
        return self.brief.related_match_count

    @computed_field
    @property
    def complete(self) -> bool:
        return (
            self.related_work.complete
            and self.review_cost.complete
            and self.brief.complete
        )


class GitHubReviewCostAssessor(Protocol):
    async def assess(
        self,
        request: GitHubReviewCostRequest,
    ) -> GitHubReviewCostResult:
        ...


class MaintainerBriefBuilder(Protocol):
    def build(self, brief_input: MaintainerBriefInput) -> MaintainerBrief:
        ...


class GitHubMaintainerBriefRunner(Protocol):
    """Public runner boundary used by the MCP capability."""

    async def build(
        self,
        request: GitHubMaintainerBriefRequest,
    ) -> GitHubMaintainerBriefResult:
        ...


def _path_risk(
    adapted: ReviewCostAssessmentInput,
) -> MaintainerPathRiskSummary:
    classifications = classify_review_cost_paths(
        adapted.files,
        adapted.protected_changed_paths,
    )

    def count(category: ReviewCostPathCategory) -> int:
        return sum(category in item.categories for item in classifications)

    return MaintainerPathRiskSummary(
        changed_path_count=len(classifications),
        protected_path_count=count(ReviewCostPathCategory.PROTECTED),
        security_sensitive_path_count=count(
            ReviewCostPathCategory.SECURITY_SENSITIVE
        ),
        database_migration_path_count=count(
            ReviewCostPathCategory.DATABASE_MIGRATION
        ),
        automation_or_deployment_path_count=count(
            ReviewCostPathCategory.AUTOMATION_OR_DEPLOYMENT
        ),
        dependency_manifest_path_count=count(
            ReviewCostPathCategory.DEPENDENCY_MANIFEST
        ),
    )


def _readiness(
    result: GitHubReviewCostResult,
    adapted: ReviewCostAssessmentInput,
) -> MaintainerReadinessSummary:
    assessment = result.pull_request_assessment
    evaluation = assessment.evaluation
    findings = evaluation.findings
    policy_blocked = any(
        finding.status == PolicyFindingStatus.FAILED for finding in findings
    )
    # findings is the authoritative evidence collection. The typed aggregate
    # flags capture repository-wide attention even when no individual finding exists.
    policy_attention_required = (
        policy_blocked
        or evaluation.requires_human_review
        or not evaluation.compliant
        or any(
            finding.status == PolicyFindingStatus.WARNING
            for finding in findings
        )
    )
    return MaintainerReadinessSummary(
        policy_present=adapted.policy_present,
        policy_blocked=policy_blocked,
        policy_attention_required=policy_attention_required,
        policy_finding_count=len(findings),
        draft=adapted.draft,
        merge_conflict=adapted.merge_conflict,
        required_checks_total=adapted.required_checks_total,
        required_checks_passed=adapted.required_checks_passed,
        required_checks_failed=adapted.required_checks_failed,
        required_checks_pending=adapted.required_checks_pending,
        approval_count=adapted.approval_count,
        changes_requested_count=adapted.changes_requested_count,
    )


class GitHubMaintainerBriefService:
    """Adapt one completed Phase 5 result without recollecting evidence."""

    def __init__(
        self,
        *,
        review_cost_assessor: GitHubReviewCostAssessor,
        brief_builder: MaintainerBriefBuilder,
    ) -> None:
        self._review_cost_assessor = review_cost_assessor
        self._brief_builder = brief_builder

    async def build(
        self,
        request: GitHubMaintainerBriefRequest,
    ) -> GitHubMaintainerBriefResult:
        review_cost = await self._review_cost_assessor.assess(
            request.to_review_cost_request()
        )
        if (
            review_cost.repository != request.repository
            or review_cost.pull_request.pull_number != request.pull_number
        ):
            raise GitHubMaintainerBriefError(
                "Review-cost result identifies another repository or pull request."
            )
        assessment = review_cost.pull_request_assessment
        if (
            assessment.installation_id != request.installation_id
            or assessment.summary.repository != request.repository
            or assessment.summary.pull_number != request.pull_number
            or assessment.snapshot.repository != request.repository
            or assessment.snapshot.pull_request.number != request.pull_number
        ):
            raise GitHubMaintainerBriefError(
                "Pull-request assessment identifies another repository or pull request."
            )

        if (
            review_cost.related_work.related_work.options
            != request.related_work_options
            or review_cost.related_work.snapshot.requested_ref
            != assessment.snapshot.pull_request.base.sha
        ):
            raise GitHubMaintainerBriefError(
                "Review-cost result is inconsistent with the source request."
            )
        try:
            adapted = _assessment_input(assessment, review_cost.related_work)
        except GitHubReviewCostError as exc:
            raise GitHubMaintainerBriefError(
                "Review-cost evidence cannot be adapted safely."
            ) from exc
        historical_context = adapted.historical_context
        if historical_context is None:
            raise GitHubMaintainerBriefError(
                "Review-cost result lacks historical context."
            )
        brief_input = MaintainerBriefInput(
            repository=review_cost.review_cost.repository,
            pull_request=review_cost.review_cost.pull_request,
            generated_at=review_cost.review_cost.assessed_at,
            readiness=_readiness(review_cost, adapted),
            path_risk=_path_risk(adapted),
            historical_context=historical_context,
            review_cost=review_cost.review_cost,
            related_work=review_cost.related_work.related_work,
        )
        brief = self._brief_builder.build(brief_input)
        if (
            brief.repository != brief_input.repository
            or brief.pull_request != brief_input.pull_request
            or brief.generated_at != brief_input.generated_at
        ):
            raise GitHubMaintainerBriefError(
                "Maintainer brief identifies another repository, pull request, or time."
            )
        warnings = _stable_unique(
            review_cost.related_work.warnings,
            review_cost.review_cost.warnings,
            brief.warnings,
        )
        return GitHubMaintainerBriefResult(
            repository=request.repository,
            pull_request=review_cost.pull_request,
            pull_request_assessment=assessment,
            related_work=review_cost.related_work,
            review_cost=review_cost.review_cost,
            brief=brief,
            warnings=warnings,
        )
