"""Provider-independent models for evidence-derived review-cost assessment."""

import re
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
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
    KnowledgeRepositoryRef,
    StrictKnowledgeModel,
)

MAX_REVIEW_COST_CHANGED_FILES = 3_000
MAX_REVIEW_COST_PROTECTED_PATHS = 1_000
MAX_REVIEW_COST_EVIDENCE_ITEMS = 100
MAX_REVIEW_COST_WARNINGS = 100
MAX_REVIEW_COST_SCORE = 100
MAX_REVIEW_COST_BASIS_POINTS = 10_000

REVIEW_COST_CHANGE_SIZE_WEIGHT = 25
REVIEW_COST_CHANGE_DISPERSION_WEIGHT = 15
REVIEW_COST_RISK_SENSITIVE_PATHS_WEIGHT = 25
REVIEW_COST_VALIDATION_GAPS_WEIGHT = 20
REVIEW_COST_HISTORICAL_COMPLEXITY_WEIGHT = 15


class ReviewCostLevel(StrEnum):
    """Expected maintainer review-effort level."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewCostSignal(StrEnum):
    """Signals contributing to evidence-derived review cost."""

    CHANGE_SIZE = "change_size"
    CHANGE_DISPERSION = "change_dispersion"
    RISK_SENSITIVE_PATHS = "risk_sensitive_paths"
    VALIDATION_GAPS = "validation_gaps"
    HISTORICAL_COMPLEXITY = "historical_complexity"


class ReviewCostChangeType(StrEnum):
    """Normalized changed-file operation."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"
    RENAMED = "renamed"
    COPIED = "copied"
    UNKNOWN = "unknown"


class ReviewCostPathCategory(StrEnum):
    """Deterministic review-cost path categories."""

    PROTECTED = "protected"
    SECURITY_SENSITIVE = "security_sensitive"
    DATABASE_MIGRATION = "database_migration"
    AUTOMATION_OR_DEPLOYMENT = "automation_or_deployment"
    DEPENDENCY_MANIFEST = "dependency_manifest"
    TEST = "test"
    DOCUMENTATION = "documentation"
    PRODUCTION = "production"


def normalize_review_cost_path(path: str) -> str:
    """Normalize and validate one repository-relative path."""

    normalized = path.replace("\\", "/").strip()
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        raise ValueError("Review-cost paths must not be empty.")
    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError("Review-cost paths must be repository-relative.")
    parts = normalized.split("/")
    if any(not part for part in parts):
        raise ValueError("Review-cost paths must not contain empty segments.")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("Review-cost paths must not contain '.' or '..' segments.")
    return normalized


def _validate_unique_text(values: list[str], field_name: str) -> list[str]:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must contain unique strings.")
    return values


class ReviewCostChangedFile(StrictKnowledgeModel):
    """One normalized file contributing to review structure and path risk."""

    path: str
    previous_path: str | None = None
    change_type: ReviewCostChangeType
    additions: int = Field(ge=0)
    deletions: int = Field(ge=0)

    @field_validator("path", "previous_path")
    @classmethod
    def normalize_paths(cls, value: str | None) -> str | None:
        """Normalize current and previous repository paths."""

        return None if value is None else normalize_review_cost_path(value)

    @model_validator(mode="after")
    def validate_previous_path(self) -> Self:
        """Require a supplied previous path to identify another path."""

        if self.previous_path == self.path:
            raise ValueError("previous_path must differ from path.")
        return self

    @computed_field
    @property
    def changes(self) -> int:
        """Return additions plus deletions."""

        return self.additions + self.deletions

    @computed_field
    @property
    def directory(self) -> str:
        """Return the parent directory or root marker."""

        parent = str(PurePosixPath(self.path).parent)
        return parent if parent else "."

    @computed_field
    @property
    def top_level_component(self) -> str:
        """Return the first directory segment or root marker."""

        parts = self.path.split("/")
        return parts[0] if len(parts) > 1 else "(root)"


class ReviewCostHistoricalContext(StrictKnowledgeModel):
    """Classified related-work counts and coverage."""

    related_match_count: int = Field(ge=0)
    rejected_or_superseded_count: int = Field(ge=0)
    high_significance_count: int = Field(ge=0)
    unresolved_count: int = Field(ge=0)
    source_history_complete: bool
    ranking_coverage_complete: bool
    result_truncated: bool

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Ensure every classified count is bounded by returned matches."""

        classified = (
            self.rejected_or_superseded_count,
            self.high_significance_count,
            self.unresolved_count,
        )
        if any(count > self.related_match_count for count in classified):
            raise ValueError(
                "Historical classified counts must not exceed related_match_count."
            )
        return self


class ReviewCostAssessmentInput(StrictKnowledgeModel):
    """All provider-neutral evidence required for review-cost scoring."""

    repository: KnowledgeRepositoryRef
    pull_request: KnowledgeItemReference
    files: list[ReviewCostChangedFile] = Field(
        default_factory=list,
        max_length=MAX_REVIEW_COST_CHANGED_FILES,
    )
    commit_count: int = Field(ge=0)
    preferred_diff_size: int = Field(ge=1)
    policy_present: bool
    protected_changed_paths: list[str] = Field(
        default_factory=list,
        max_length=MAX_REVIEW_COST_PROTECTED_PATHS,
    )
    draft: bool
    merge_conflict: bool | None
    required_checks_total: int = Field(ge=0)
    required_checks_passed: int = Field(ge=0)
    required_checks_failed: int = Field(ge=0)
    required_checks_pending: int = Field(ge=0)
    approval_count: int = Field(ge=0)
    changes_requested_count: int = Field(ge=0)
    historical_context: ReviewCostHistoricalContext | None = None

    @field_validator("protected_changed_paths")
    @classmethod
    def normalize_protected_paths(cls, paths: list[str]) -> list[str]:
        """Normalize and require unique protected changed paths."""

        normalized = [normalize_review_cost_path(path) for path in paths]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Protected changed paths must be unique.")
        return normalized

    @model_validator(mode="after")
    def validate_input(self) -> Self:
        """Validate identity, path coverage, and check accounting."""

        if self.pull_request.repository != self.repository:
            raise ValueError("Pull-request reference must belong to repository.")
        if self.pull_request.item_type != KnowledgeItemType.PULL_REQUEST:
            raise ValueError("Review cost requires a pull-request reference.")
        paths = [file.path for file in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError("Review-cost changed file paths must be unique.")
        available_paths = {
            path
            for file in self.files
            for path in (file.path, file.previous_path)
            if path is not None
        }
        if any(path not in available_paths for path in self.protected_changed_paths):
            raise ValueError(
                "Every protected changed path must identify a changed file path."
            )
        if (
            self.required_checks_passed
            + self.required_checks_failed
            + self.required_checks_pending
            != self.required_checks_total
        ):
            raise ValueError(
                "Required check subcounts must sum to required_checks_total."
            )
        return self

    @computed_field
    @property
    def additions(self) -> int:
        return sum(file.additions for file in self.files)

    @computed_field
    @property
    def deletions(self) -> int:
        return sum(file.deletions for file in self.files)

    @computed_field
    @property
    def total_changes(self) -> int:
        return self.additions + self.deletions

    @computed_field
    @property
    def changed_file_count(self) -> int:
        return len(self.files)

    @computed_field
    @property
    def distinct_directory_count(self) -> int:
        return len({file.directory for file in self.files})

    @computed_field
    @property
    def distinct_component_count(self) -> int:
        return len({file.top_level_component for file in self.files})


class ReviewCostAssessmentOptions(StrictKnowledgeModel):
    """Bounds for explanatory review-cost evidence."""

    max_evidence_items_per_signal: int = Field(
        default=20,
        ge=1,
        le=MAX_REVIEW_COST_EVIDENCE_ITEMS,
    )


class ReviewCostSignalContribution(StrictKnowledgeModel):
    """One explainable weighted review-cost signal."""

    model_config = ConfigDict(json_schema_mode_override="serialization")

    signal: ReviewCostSignal
    signal_score: int = Field(ge=0, le=MAX_REVIEW_COST_SCORE)
    weight_percent: int = Field(ge=1, le=100)
    weighted_basis_points: int = Field(
        ge=0,
        le=MAX_REVIEW_COST_BASIS_POINTS,
    )
    explanation: str = Field(min_length=1)
    evidence: list[str] = Field(
        default_factory=list,
        max_length=MAX_REVIEW_COST_EVIDENCE_ITEMS,
    )

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, evidence: list[str]) -> list[str]:
        return _validate_unique_text(evidence, "Review-cost evidence")

    @model_validator(mode="after")
    def validate_weighted_score(self) -> Self:
        if self.weighted_basis_points != self.signal_score * self.weight_percent:
            raise ValueError(
                "weighted_basis_points must equal signal_score * weight_percent."
            )
        return self

    @computed_field
    @property
    def weighted_score(self) -> float:
        return self.weighted_basis_points / 100


_SIGNAL_WEIGHTS = {
    ReviewCostSignal.CHANGE_SIZE: REVIEW_COST_CHANGE_SIZE_WEIGHT,
    ReviewCostSignal.CHANGE_DISPERSION: REVIEW_COST_CHANGE_DISPERSION_WEIGHT,
    ReviewCostSignal.RISK_SENSITIVE_PATHS: (
        REVIEW_COST_RISK_SENSITIVE_PATHS_WEIGHT
    ),
    ReviewCostSignal.VALIDATION_GAPS: REVIEW_COST_VALIDATION_GAPS_WEIGHT,
    ReviewCostSignal.HISTORICAL_COMPLEXITY: (
        REVIEW_COST_HISTORICAL_COMPLEXITY_WEIGHT
    ),
}


class ReviewCostAssessment(StrictKnowledgeModel):
    """Complete explainable evidence-derived review-cost assessment."""

    model_config = ConfigDict(json_schema_mode_override="serialization")

    repository: KnowledgeRepositoryRef
    pull_request: KnowledgeItemReference
    assessed_at: datetime
    contributions: list[ReviewCostSignalContribution]
    reducers: list[str]
    warnings: list[str] = Field(max_length=MAX_REVIEW_COST_WARNINGS)

    @field_validator("assessed_at")
    @classmethod
    def normalize_assessed_at(cls, assessed_at: datetime) -> datetime:
        if assessed_at.tzinfo is None or assessed_at.utcoffset() is None:
            raise ValueError("assessed_at must be timezone-aware.")
        return assessed_at.astimezone(UTC)

    @field_validator("reducers")
    @classmethod
    def validate_reducers(cls, reducers: list[str]) -> list[str]:
        return _validate_unique_text(reducers, "Review-cost reducers")

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        return _validate_unique_text(warnings, "Review-cost warnings")

    @model_validator(mode="after")
    def validate_assessment(self) -> Self:
        if self.pull_request.repository != self.repository:
            raise ValueError("Pull-request reference must belong to repository.")
        if self.pull_request.item_type != KnowledgeItemType.PULL_REQUEST:
            raise ValueError("Review cost requires a pull-request reference.")
        expected_signals = list(ReviewCostSignal)
        signals = [contribution.signal for contribution in self.contributions]
        if signals != expected_signals:
            raise ValueError(
                "Review-cost contributions must use the exact signal order."
            )
        for contribution in self.contributions:
            if contribution.weight_percent != _SIGNAL_WEIGHTS[contribution.signal]:
                raise ValueError("Review-cost contribution weights must be exact.")
        if self.total_weighted_basis_points > MAX_REVIEW_COST_BASIS_POINTS:
            raise ValueError("Total weighted basis points must not exceed 10,000.")
        return self

    @computed_field
    @property
    def total_weighted_basis_points(self) -> int:
        return sum(
            contribution.weighted_basis_points
            for contribution in self.contributions
        )

    @computed_field
    @property
    def score(self) -> int:
        return (self.total_weighted_basis_points + 50) // 100

    @computed_field
    @property
    def level(self) -> ReviewCostLevel:
        if self.score < 25:
            return ReviewCostLevel.LOW
        if self.score < 50:
            return ReviewCostLevel.MODERATE
        if self.score < 75:
            return ReviewCostLevel.HIGH
        return ReviewCostLevel.CRITICAL

    @computed_field
    @property
    def primary_drivers(self) -> list[ReviewCostSignal]:
        indexed = [
            (index, contribution)
            for index, contribution in enumerate(self.contributions)
            if contribution.signal_score > 0
        ]
        indexed.sort(
            key=lambda item: (-item[1].weighted_basis_points, item[0])
        )
        return [contribution.signal for _, contribution in indexed]

    @computed_field
    @property
    def complete(self) -> bool:
        return not self.warnings
