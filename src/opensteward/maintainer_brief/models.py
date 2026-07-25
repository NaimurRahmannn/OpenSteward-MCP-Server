"""Provider-independent models for maintainer attention and brief generation."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import (
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from opensteward.knowledge import (
    KnowledgeItemReference,
    KnowledgeItemType,
    KnowledgeRelatedWorkResult,
    KnowledgeRepositoryRef,
    StrictKnowledgeModel,
)
from opensteward.review_intelligence import (
    ReviewCostAssessment,
    ReviewCostHistoricalContext,
    ReviewCostLevel,
)

MAX_MAINTAINER_ATTENTION_REASONS = 50
MAX_MAINTAINER_REVIEW_ROUTES = 5
MAX_MAINTAINER_RECOMMENDED_ACTIONS = 20
MAX_MAINTAINER_BRIEF_WARNINGS = 100
MAX_MAINTAINER_EVALUATION_CASES = 500


class MaintainerAttentionRecommendation(StrEnum):
    """Recommended queue treatment for a pull request."""

    AUTHOR_ACTION_FIRST = "author_action_first"
    ROUTINE_REVIEW = "routine_review"
    HIGH_PRIORITY_REVIEW = "high_priority_review"
    IMMEDIATE_REVIEW = "immediate_review"


class MaintainerReviewRoute(StrEnum):
    """Maintainer expertise needed to review a pull request."""

    SECURITY = "security"
    DATABASE = "database"
    RELEASE_OR_DEPLOYMENT = "release_or_deployment"
    ARCHITECTURE = "architecture"
    GENERAL = "general"


class MaintainerAttentionReasonKind(StrEnum):
    """Ordered evidence kinds supporting an attention recommendation."""

    POLICY_BLOCKER = "policy_blocker"
    MERGE_CONFLICT = "merge_conflict"
    FAILED_REQUIRED_CHECKS = "failed_required_checks"
    CHANGES_REQUESTED = "changes_requested"
    DRAFT_PULL_REQUEST = "draft_pull_request"
    PENDING_REQUIRED_CHECKS = "pending_required_checks"
    CRITICAL_REVIEW_COST = "critical_review_cost"
    HIGH_REVIEW_COST = "high_review_cost"
    PROTECTED_PATHS = "protected_paths"
    SECURITY_SENSITIVE_PATHS = "security_sensitive_paths"
    DATABASE_MIGRATION = "database_migration"
    AUTOMATION_OR_DEPLOYMENT = "automation_or_deployment"
    REJECTED_OR_SUPERSEDED_HISTORY = "rejected_or_superseded_history"
    HIGH_SIGNIFICANCE_HISTORY = "high_significance_history"
    UNRESOLVED_HISTORY = "unresolved_history"
    INCOMPLETE_EVIDENCE = "incomplete_evidence"


_BLOCKING_REASON_KINDS = {
    MaintainerAttentionReasonKind.POLICY_BLOCKER,
    MaintainerAttentionReasonKind.MERGE_CONFLICT,
    MaintainerAttentionReasonKind.FAILED_REQUIRED_CHECKS,
    MaintainerAttentionReasonKind.CHANGES_REQUESTED,
    MaintainerAttentionReasonKind.DRAFT_PULL_REQUEST,
    MaintainerAttentionReasonKind.PENDING_REQUIRED_CHECKS,
}


def _normalize_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _validate_unique_text(values: list[str], field_name: str) -> list[str]:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique strings.")
    return values


class MaintainerReadinessSummary(StrictKnowledgeModel):
    """Contribution-readiness evidence relevant to maintainer routing."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_mode_override="serialization",
    )

    policy_present: bool
    policy_blocked: bool
    policy_attention_required: bool
    policy_finding_count: int = Field(ge=0)
    draft: bool
    merge_conflict: bool | None
    required_checks_total: int = Field(ge=0)
    required_checks_passed: int = Field(ge=0)
    required_checks_failed: int = Field(ge=0)
    required_checks_pending: int = Field(ge=0)
    approval_count: int = Field(ge=0)
    changes_requested_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_readiness(self) -> Self:
        if (
            self.required_checks_passed
            + self.required_checks_failed
            + self.required_checks_pending
            != self.required_checks_total
        ):
            raise ValueError(
                "Required check subcounts must sum to required_checks_total."
            )
        if self.policy_blocked and not self.policy_attention_required:
            raise ValueError("policy_blocked requires policy_attention_required.")
        if self.policy_blocked and self.policy_finding_count == 0:
            raise ValueError(
                "A blocking policy state requires at least one policy finding."
            )
        return self

    @computed_field
    @property
    def has_readiness_blocker(self) -> bool:
        return (
            self.policy_blocked
            or self.merge_conflict is True
            or self.required_checks_failed > 0
            or self.changes_requested_count > 0
            or self.draft
            or self.required_checks_pending > 0
        )

    @computed_field
    @property
    def all_required_checks_passed(self) -> bool:
        return (
            self.required_checks_total > 0
            and self.required_checks_passed == self.required_checks_total
        )


class MaintainerPathRiskSummary(StrictKnowledgeModel):
    """Counts of structured path categories relevant to specialist routing."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_mode_override="serialization",
    )

    changed_path_count: int = Field(ge=0)
    protected_path_count: int = Field(ge=0)
    security_sensitive_path_count: int = Field(ge=0)
    database_migration_path_count: int = Field(ge=0)
    automation_or_deployment_path_count: int = Field(ge=0)
    dependency_manifest_path_count: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        category_counts = (
            self.protected_path_count,
            self.security_sensitive_path_count,
            self.database_migration_path_count,
            self.automation_or_deployment_path_count,
            self.dependency_manifest_path_count,
        )
        if any(count > self.changed_path_count for count in category_counts):
            raise ValueError(
                "Path-risk category counts must not exceed changed_path_count."
            )
        return self

    @computed_field
    @property
    def specialist_risk_count(self) -> int:
        return sum(
            count > 0
            for count in (
                self.security_sensitive_path_count,
                self.database_migration_path_count,
                self.automation_or_deployment_path_count,
                self.protected_path_count,
            )
        )

    @computed_field
    @property
    def has_specialist_risk(self) -> bool:
        return self.specialist_risk_count > 0


class MaintainerAttentionReason(StrictKnowledgeModel):
    """One explicit reason supporting maintainer attention treatment."""

    kind: MaintainerAttentionReasonKind
    explanation: str = Field(min_length=1)
    blocking: bool

    @model_validator(mode="after")
    def validate_blocking(self) -> Self:
        if self.blocking != (self.kind in _BLOCKING_REASON_KINDS):
            raise ValueError(
                "Maintainer attention reason blocking flag must match its kind."
            )
        return self


class MaintainerAttentionAssessment(StrictKnowledgeModel):
    """Deterministic attention recommendation with evidence and routes."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_mode_override="serialization",
    )

    repository: KnowledgeRepositoryRef
    pull_request: KnowledgeItemReference
    assessed_at: datetime
    recommendation: MaintainerAttentionRecommendation
    routes: list[MaintainerReviewRoute] = Field(
        min_length=1,
        max_length=MAX_MAINTAINER_REVIEW_ROUTES,
    )
    reasons: list[MaintainerAttentionReason] = Field(
        max_length=MAX_MAINTAINER_ATTENTION_REASONS,
    )
    warnings: list[str] = Field(max_length=MAX_MAINTAINER_BRIEF_WARNINGS)

    @field_validator("assessed_at")
    @classmethod
    def normalize_assessed_at(cls, assessed_at: datetime) -> datetime:
        return _normalize_datetime(assessed_at, "assessed_at")

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        return _validate_unique_text(warnings, "Maintainer attention warnings")

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        if self.pull_request.repository != self.repository:
            raise ValueError("Pull-request reference must belong to repository.")
        if self.pull_request.item_type != KnowledgeItemType.PULL_REQUEST:
            raise ValueError("Maintainer attention requires a pull-request reference.")
        if len(self.routes) != len(set(self.routes)):
            raise ValueError("Maintainer review routes must be unique.")
        expected_routes = [route for route in MaintainerReviewRoute if route in self.routes]
        if self.routes != expected_routes:
            raise ValueError("Maintainer review routes must use the exact route order.")
        if MaintainerReviewRoute.GENERAL in self.routes and self.routes != [
            MaintainerReviewRoute.GENERAL
        ]:
            raise ValueError("GENERAL must be the only route when present.")

        reason_kinds = [reason.kind for reason in self.reasons]
        if len(reason_kinds) != len(set(reason_kinds)):
            raise ValueError("Maintainer attention reason kinds must be unique.")
        expected_reasons = [
            kind for kind in MaintainerAttentionReasonKind if kind in reason_kinds
        ]
        if reason_kinds != expected_reasons:
            raise ValueError(
                "Maintainer attention reasons must use the exact reason order."
            )

        blocking = any(reason.blocking for reason in self.reasons)
        if self.recommendation == MaintainerAttentionRecommendation.AUTHOR_ACTION_FIRST:
            if not blocking:
                raise ValueError("AUTHOR_ACTION_FIRST requires a blocking reason.")
        elif blocking:
            raise ValueError("Ready-for-review recommendations forbid blocking reasons.")

        if self.recommendation == MaintainerAttentionRecommendation.ROUTINE_REVIEW:
            elevated = {
                MaintainerAttentionReasonKind.CRITICAL_REVIEW_COST,
                MaintainerAttentionReasonKind.HIGH_REVIEW_COST,
            }
            if elevated & set(reason_kinds):
                raise ValueError("ROUTINE_REVIEW forbids elevated review-cost reasons.")
            if self.routes != [MaintainerReviewRoute.GENERAL]:
                raise ValueError("ROUTINE_REVIEW requires the GENERAL route.")
        elif (
            self.recommendation
            == MaintainerAttentionRecommendation.HIGH_PRIORITY_REVIEW
            and not self.reasons
        ):
            raise ValueError("HIGH_PRIORITY_REVIEW requires a nonblocking reason.")
        elif self.recommendation == MaintainerAttentionRecommendation.IMMEDIATE_REVIEW:
            immediate = (
                MaintainerAttentionReasonKind.CRITICAL_REVIEW_COST in reason_kinds
                or MaintainerAttentionReasonKind.SECURITY_SENSITIVE_PATHS
                in reason_kinds
                or (
                    MaintainerAttentionReasonKind.REJECTED_OR_SUPERSEDED_HISTORY
                    in reason_kinds
                    and MaintainerAttentionReasonKind.HIGH_SIGNIFICANCE_HISTORY
                    in reason_kinds
                )
            )
            if not immediate:
                raise ValueError(
                    "IMMEDIATE_REVIEW requires critical, security, or combined "
                    "historical evidence."
                )
        return self

    @computed_field
    @property
    def requires_author_action(self) -> bool:
        return (
            self.recommendation
            == MaintainerAttentionRecommendation.AUTHOR_ACTION_FIRST
        )

    @computed_field
    @property
    def ready_for_review(self) -> bool:
        return not self.requires_author_action

    @computed_field
    @property
    def has_specialist_route(self) -> bool:
        return self.routes != [MaintainerReviewRoute.GENERAL]

    @computed_field
    @property
    def complete(self) -> bool:
        return not self.warnings


class MaintainerBriefInput(StrictKnowledgeModel):
    """All provider-neutral evidence required to construct a brief."""

    repository: KnowledgeRepositoryRef
    pull_request: KnowledgeItemReference
    generated_at: datetime
    readiness: MaintainerReadinessSummary
    path_risk: MaintainerPathRiskSummary
    historical_context: ReviewCostHistoricalContext
    review_cost: ReviewCostAssessment
    related_work: KnowledgeRelatedWorkResult

    @field_validator("generated_at")
    @classmethod
    def normalize_generated_at(cls, generated_at: datetime) -> datetime:
        return _normalize_datetime(generated_at, "generated_at")

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        if self.pull_request.repository != self.repository:
            raise ValueError("Pull-request reference must belong to repository.")
        if self.pull_request.item_type != KnowledgeItemType.PULL_REQUEST:
            raise ValueError("Maintainer briefs require a pull-request reference.")
        if (
            self.review_cost.repository != self.repository
            or self.review_cost.pull_request != self.pull_request
        ):
            raise ValueError(
                "Review cost must identify the brief repository and pull request."
            )
        if (
            self.related_work.repository != self.repository
            or self.related_work.query.repository != self.repository
        ):
            raise ValueError("Related work must identify the brief repository.")
        if (
            self.historical_context.related_match_count
            != self.related_work.returned_count
        ):
            raise ValueError(
                "Historical related count must match returned related work."
            )
        if (
            self.historical_context.ranking_coverage_complete
            != self.related_work.complete_ranking_coverage
        ):
            raise ValueError(
                "Historical ranking coverage must match related work."
            )
        if (
            self.historical_context.result_truncated
            != self.related_work.truncated
        ):
            raise ValueError("Historical truncation must match related work.")
        return self


class MaintainerBrief(StrictKnowledgeModel):
    """Complete structured evidence and deterministic next actions."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_mode_override="serialization",
    )

    repository: KnowledgeRepositoryRef
    pull_request: KnowledgeItemReference
    generated_at: datetime
    readiness: MaintainerReadinessSummary
    path_risk: MaintainerPathRiskSummary
    historical_context: ReviewCostHistoricalContext
    attention: MaintainerAttentionAssessment
    review_cost: ReviewCostAssessment
    related_work: KnowledgeRelatedWorkResult
    recommended_actions: list[str] = Field(
        max_length=MAX_MAINTAINER_RECOMMENDED_ACTIONS
    )
    warnings: list[str] = Field(max_length=MAX_MAINTAINER_BRIEF_WARNINGS)

    @field_validator("generated_at")
    @classmethod
    def normalize_generated_at(cls, generated_at: datetime) -> datetime:
        return _normalize_datetime(generated_at, "generated_at")

    @field_validator("recommended_actions")
    @classmethod
    def validate_actions(cls, actions: list[str]) -> list[str]:
        return _validate_unique_text(actions, "Recommended actions")

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        return _validate_unique_text(warnings, "Maintainer brief warnings")

    @model_validator(mode="after")
    def validate_brief(self) -> Self:
        if self.pull_request.repository != self.repository:
            raise ValueError("Pull-request reference must belong to repository.")
        if self.pull_request.item_type != KnowledgeItemType.PULL_REQUEST:
            raise ValueError("Maintainer briefs require a pull-request reference.")
        if (
            self.attention.repository != self.repository
            or self.attention.pull_request != self.pull_request
            or self.review_cost.repository != self.repository
            or self.review_cost.pull_request != self.pull_request
            or self.related_work.repository != self.repository
            or self.related_work.query.repository != self.repository
        ):
            raise ValueError("All brief evidence must use the same identities.")
        if (
            self.attention.assessed_at != self.generated_at
            or self.review_cost.assessed_at != self.generated_at
        ):
            raise ValueError("Brief evidence times must match generated_at.")
        expected_warnings = list(
            dict.fromkeys(
                [
                    *self.review_cost.warnings,
                    *self.related_work.warnings,
                    *self.attention.warnings,
                ]
            )
        )
        if self.warnings != expected_warnings:
            raise ValueError(
                "Warnings must equal stable review-cost, related-work, and "
                "attention warnings."
            )
        return self

    @computed_field
    @property
    def recommendation(self) -> MaintainerAttentionRecommendation:
        return self.attention.recommendation

    @computed_field
    @property
    def routes(self) -> list[MaintainerReviewRoute]:
        return list(self.attention.routes)

    @computed_field
    @property
    def review_cost_score(self) -> int:
        return self.review_cost.score

    @computed_field
    @property
    def review_cost_level(self) -> ReviewCostLevel:
        return self.review_cost.level

    @computed_field
    @property
    def related_match_count(self) -> int:
        return self.related_work.returned_count

    @computed_field
    @property
    def complete(self) -> bool:
        return (
            self.attention.complete
            and self.review_cost.complete
            and self.historical_context.source_history_complete
            and self.historical_context.ranking_coverage_complete
            and not self.historical_context.result_truncated
        )
