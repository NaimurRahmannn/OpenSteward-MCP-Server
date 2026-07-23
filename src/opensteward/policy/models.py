"""Typed repository policy models for OpenSteward."""

from enum import StrEnum
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class StrictPolicyModel(BaseModel):
    """Base model for strict repository policy validation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class RiskLevel(StrEnum):
    """Risk classification for repository paths and changes."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class PolicySource(StrEnum):
    """Origin of the active repository policy."""

    DEFAULT = "default"
    REPOSITORY_FILE = "repository_file"
    MEMORY = "memory"

class ContributionCategory(StrEnum):
    """Contribution categories understood by the policy engine."""

    BUG_FIX = "bug_fix"
    OBSERVABLE_BEHAVIOR = "observable_behavior"
    PUBLIC_API = "public_api"
    ARCHITECTURE = "architecture"
    DATABASE_MIGRATION = "database_migration"
    SECURITY = "security"
    DEPENDENCY_ADDITION = "dependency_addition"
class PullRequestPolicy(StrictPolicyModel):
    """Rules applied to incoming pull requests."""

    linked_issue_required_for: list[ContributionCategory] = Field(
        default_factory=list,
    )

    tests_required_for: list[ContributionCategory] = Field(
        default_factory=lambda: [
            ContributionCategory.BUG_FIX,
            ContributionCategory.OBSERVABLE_BEHAVIOR,
            ContributionCategory.PUBLIC_API,
        ],
    )

    preferred_maximum_diff_lines: int = Field(
        default=500,
        ge=1,
        le=100_000,
    )

    @field_validator(
        "linked_issue_required_for",
        "tests_required_for",
    )
    @classmethod
    def reject_duplicate_categories(
        cls,
        categories: list[ContributionCategory],
    ) -> list[ContributionCategory]:
        """Reject duplicate contribution categories."""

        if len(categories) != len(set(categories)):
            raise ValueError(
                "Contribution categories must not contain duplicates."
            )

        return categories


class ProtectedPathRule(StrictPolicyModel):
    """Rule describing a sensitive repository path pattern."""

    pattern: str = Field(min_length=1)
    risk: RiskLevel = RiskLevel.HIGH
    human_review_required: bool = True

    @field_validator("pattern")
    @classmethod
    def validate_pattern(cls, pattern: str) -> str:
        """Normalize and validate a repository-relative glob pattern."""

        normalized = pattern.replace("\\", "/").strip()

        while normalized.startswith("./"):
            normalized = normalized[2:]

        if not normalized:
            raise ValueError(
                "Protected path patterns must not be empty."
            )

        if normalized.startswith("/"):
            raise ValueError(
                "Protected path patterns must be repository-relative."
            )

        parts = normalized.split("/")

        if any(part == "" for part in parts):
            raise ValueError(
                "Protected path patterns must not contain empty segments."
            )

        if any(part in {".", ".."} for part in parts):
            raise ValueError(
                "Protected path patterns must not contain '.' or '..' segments."
            )

        return normalized
    
class ProtectedPathMatch(StrictPolicyModel):
    """Evidence that a repository path matched a protected-path rule."""

    path: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    risk: RiskLevel
    human_review_required: bool
    explanation: str = Field(min_length=1)

class RequiredApprovalsPolicy(StrictPolicyModel):
    """Minimum human approvals required for sensitive changes."""

    default: int = Field(default=1, ge=0, le=20)
    public_api: int = Field(default=2, ge=0, le=20)
    security: int = Field(default=2, ge=0, le=20)

    @model_validator(mode="after")
    def validate_sensitive_approval_counts(self) -> Self:
        """Prevent sensitive changes from requiring fewer approvals."""

        if self.public_api < self.default:
            raise ValueError(
                "Public API approvals cannot be lower than the default."
            )

        if self.security < self.default:
            raise ValueError(
                "Security approvals cannot be lower than the default."
            )

        return self


class ReviewPolicy(StrictPolicyModel):
    """Maintainer review and workload rules."""

    maximum_pending_reviews_per_reviewer: int = Field(
        default=8,
        ge=1,
        le=100,
    )

    required_approvals: RequiredApprovalsPolicy = Field(
        default_factory=RequiredApprovalsPolicy,
    )


class AutomationPolicy(StrictPolicyModel):
    """Controls which proposed actions may be automated."""

    publish_check_runs: bool = False
    publish_comments: bool = False
    apply_labels: bool = False
    require_human_approval: bool = True


class RepositoryPolicy(StrictPolicyModel):
    """Complete validated OpenSteward repository policy."""

    version: Literal[1] = 1
    pull_requests: PullRequestPolicy = Field(
        default_factory=PullRequestPolicy,
    )

    protected_paths: list[ProtectedPathRule] = Field(
        default_factory=list,
    )

    review: ReviewPolicy = Field(
        default_factory=ReviewPolicy,
    )

    automation: AutomationPolicy = Field(
        default_factory=AutomationPolicy,
    )

    @field_validator("protected_paths")
    @classmethod
    def reject_duplicate_path_patterns(
        cls,
        rules: list[ProtectedPathRule],
    ) -> list[ProtectedPathRule]:
        """Reject multiple rules targeting the exact same pattern."""

        patterns = [rule.pattern for rule in rules]

        if len(patterns) != len(set(patterns)):
            raise ValueError(
                "Protected path patterns must be unique."
            )

        return rules
class LoadedRepositoryPolicy(StrictPolicyModel):
    """A validated policy together with its origin metadata."""

    policy: RepositoryPolicy
    source: PolicySource
    source_reference: str = Field(min_length=1)

    @computed_field
    @property
    def used_defaults(self) -> bool:
        """Return whether OpenSteward's built-in defaults were used."""

        return self.source == PolicySource.DEFAULT