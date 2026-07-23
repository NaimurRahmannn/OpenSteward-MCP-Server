"""Storage-independent project knowledge and decision models."""

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class StrictKnowledgeModel(BaseModel):
    """Base model for strict knowledge-domain validation."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


class KnowledgeItemType(StrEnum):
    """Semantic types of historical project knowledge."""

    ISSUE = "issue"
    PULL_REQUEST = "pull_request"
    DISCUSSION = "discussion"
    ADR = "adr"
    MAINTAINER_DECISION = "maintainer_decision"
    RELEASE_NOTE = "release_note"
    DOCUMENTATION = "documentation"


class KnowledgeSourceKind(StrEnum):
    """Sources from which project knowledge may be obtained."""

    GITHUB = "github"
    REPOSITORY_FILE = "repository_file"
    MANUAL = "manual"


class KnowledgeItemState(StrEnum):
    """Normalized lifecycle states for knowledge items."""

    OPEN = "open"
    CLOSED = "closed"
    MERGED = "merged"
    REJECTED = "rejected"
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"
    UNKNOWN = "unknown"


class DecisionSignificance(StrEnum):
    """Project impact assigned to an item or decision."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DecisionStatus(StrEnum):
    """Confidence and lifecycle states for project decisions."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    EXCEPTION = "exception"
    UNKNOWN = "unknown"


class DecisionEvidenceStrength(StrEnum):
    """Confidence levels assigned to decision evidence."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    AUTHORITATIVE = "authoritative"


class DecisionEvidenceRelationship(StrEnum):
    """Ways evidence relates to a project decision."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    DEFINES_EXCEPTION = "defines_exception"


class DecisionEvidenceKind(StrEnum):
    """Kinds of source events that provide decision evidence."""

    MAINTAINER_STATEMENT = "maintainer_statement"
    ACCEPTED_DOCUMENT = "accepted_document"
    MERGE_OUTCOME = "merge_outcome"
    REJECTION_OUTCOME = "rejection_outcome"
    RELEASE_COMMITMENT = "release_commitment"
    REPEATED_REFERENCE = "repeated_reference"
    OTHER = "other"


class KnowledgeActorType(StrEnum):
    """Types of actors associated with project knowledge."""

    USER = "user"
    BOT = "bot"
    ORGANIZATION = "organization"
    SYSTEM = "system"
    UNKNOWN = "unknown"


def _normalize_aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")

    return value.astimezone(UTC)


def _normalize_repository_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()

    while normalized.startswith("./"):
        normalized = normalized[2:]

    if not normalized:
        raise ValueError("Repository paths must not be empty.")

    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError("Repository paths must be repository-relative.")

    parts = normalized.split("/")

    if any(part == "" for part in parts):
        raise ValueError("Repository paths must not contain empty segments.")

    if any(part in {".", ".."} for part in parts):
        raise ValueError("Repository paths must not contain '.' or '..' segments.")

    return normalized


def _normalize_unique_paths(paths: list[str]) -> list[str]:
    normalized = [_normalize_repository_path(path) for path in paths]

    if len(normalized) != len(set(normalized)):
        raise ValueError("Repository paths must be unique after normalization.")

    return normalized


def _validate_unique_non_empty_strings(values: list[str], field_name: str) -> list[str]:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")

    casefolded = [value.casefold() for value in values]
    if len(casefolded) != len(set(casefolded)):
        raise ValueError(f"{field_name} must be unique case-insensitively.")

    return values


class KnowledgeRepositoryRef(StrictKnowledgeModel):
    """Provider-independent repository identity."""

    provider: str = Field(min_length=1)
    namespace: str = Field(min_length=1)
    name: str = Field(min_length=1)

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, provider: str) -> str:
        """Normalize and validate a repository provider."""

        normalized = provider.lower()
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", normalized):
            raise ValueError(
                "Repository providers must begin with a lowercase letter and contain only "
                "lowercase letters, digits, underscores, or hyphens."
            )

        return normalized

    @field_validator("namespace", "name")
    @classmethod
    def validate_repository_segment(cls, value: str) -> str:
        """Ensure repository identity values remain single segments."""

        if "/" in value or "\\" in value:
            raise ValueError("Repository namespace and name must not contain slashes.")

        if value in {".", ".."}:
            raise ValueError("Repository namespace and name must not be '.' or '..'.")

        return value

    @computed_field
    @property
    def full_name(self) -> str:
        """Return the namespace/name repository identifier."""

        return f"{self.namespace}/{self.name}"

    @computed_field
    @property
    def key(self) -> str:
        """Return the stable provider-qualified repository key."""

        return f"{self.provider}:{self.full_name}"


class KnowledgeActor(StrictKnowledgeModel):
    """An actor associated with a project knowledge item."""

    identifier: str = Field(min_length=1)
    actor_type: KnowledgeActorType = KnowledgeActorType.UNKNOWN
    display_name: str | None = None
    url: str | None = None

    @field_validator("display_name", "url")
    @classmethod
    def reject_empty_optional_strings(cls, value: str | None) -> str | None:
        """Reject optional actor strings that contain no content."""

        if value == "":
            raise ValueError("Optional actor strings must not be empty.")

        return value


class KnowledgeItemReference(StrictKnowledgeModel):
    """A compact stable reference to a project knowledge item."""

    repository: KnowledgeRepositoryRef
    item_type: KnowledgeItemType
    external_id: str = Field(min_length=1)
    source_kind: KnowledgeSourceKind
    title: str | None = None
    url: str | None = None

    @field_validator("title", "url")
    @classmethod
    def reject_empty_optional_strings(cls, value: str | None) -> str | None:
        """Reject optional reference strings that contain no content."""

        if value == "":
            raise ValueError("Optional reference strings must not be empty.")

        return value

    @computed_field
    @property
    def key(self) -> str:
        """Return the stable repository and item identifier."""

        return f"{self.repository.key}:{self.item_type.value}:{self.external_id}"


class KnowledgeItem(StrictKnowledgeModel):
    """A normalized historical project item."""

    repository: KnowledgeRepositoryRef
    item_type: KnowledgeItemType
    external_id: str = Field(min_length=1)
    source_kind: KnowledgeSourceKind
    state: KnowledgeItemState = KnowledgeItemState.UNKNOWN
    title: str = Field(min_length=1)
    body: str | None = None
    summary: str | None = None
    url: str | None = None
    author: KnowledgeActor | None = None
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    affected_paths: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    decision_significance: DecisionSignificance = DecisionSignificance.NONE

    @field_validator("body", "summary", "url")
    @classmethod
    def reject_empty_optional_strings(cls, value: str | None) -> str | None:
        """Reject optional item strings that contain no content."""

        if value == "":
            raise ValueError("Optional item strings must not be empty.")

        return value

    @field_validator("created_at", "updated_at", "closed_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        """Require aware item timestamps and normalize them to UTC."""

        if value is None:
            return None

        return _normalize_aware_datetime(value, "Knowledge item timestamps")

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[str]) -> list[str]:
        """Require unique, non-empty labels."""

        return _validate_unique_non_empty_strings(labels, "Labels")

    @field_validator("affected_paths")
    @classmethod
    def normalize_affected_paths(cls, paths: list[str]) -> list[str]:
        """Normalize and validate repository-relative affected paths."""

        return _normalize_unique_paths(paths)

    @field_validator("components")
    @classmethod
    def validate_components(cls, components: list[str]) -> list[str]:
        """Require unique, non-empty component names."""

        return _validate_unique_non_empty_strings(components, "Components")

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> Self:
        """Ensure item lifecycle timestamps are chronological."""

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at.")

        if self.closed_at is not None and self.closed_at < self.created_at:
            raise ValueError("closed_at must not be earlier than created_at.")

        return self

    @computed_field
    @property
    def key(self) -> str:
        """Return the stable repository and item identifier."""

        return f"{self.repository.key}:{self.item_type.value}:{self.external_id}"

    def to_reference(self) -> KnowledgeItemReference:
        """Return a compact reference preserving stable source identity."""

        return KnowledgeItemReference(
            repository=self.repository,
            item_type=self.item_type,
            external_id=self.external_id,
            source_kind=self.source_kind,
            title=self.title,
            url=self.url,
        )


class DecisionEvidence(StrictKnowledgeModel):
    """One exact source supporting or challenging a project decision."""

    source: KnowledgeItemReference
    kind: DecisionEvidenceKind
    relationship: DecisionEvidenceRelationship
    strength: DecisionEvidenceStrength
    summary: str = Field(min_length=1)
    excerpt: str | None = None
    source_anchor: str | None = None
    author: KnowledgeActor | None = None
    occurred_at: datetime | None = None

    @field_validator("excerpt", "source_anchor")
    @classmethod
    def reject_empty_optional_strings(cls, value: str | None) -> str | None:
        """Reject optional evidence strings that contain no content."""

        if value == "":
            raise ValueError("Optional evidence strings must not be empty.")

        return value

    @field_validator("occurred_at")
    @classmethod
    def normalize_occurred_at(cls, value: datetime | None) -> datetime | None:
        """Require an aware evidence timestamp and normalize it to UTC."""

        if value is None:
            return None

        return _normalize_aware_datetime(value, "occurred_at")


class DecisionRecord(StrictKnowledgeModel):
    """An evidence-backed candidate or established project decision."""

    repository: KnowledgeRepositoryRef
    decision_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    status: DecisionStatus = DecisionStatus.CANDIDATE
    significance: DecisionSignificance = DecisionSignificance.MEDIUM
    rationale: str | None = None
    affected_paths: list[str] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    evidence: list[DecisionEvidence] = Field(min_length=1)
    recorded_at: datetime

    @field_validator("rationale")
    @classmethod
    def reject_empty_rationale(cls, rationale: str | None) -> str | None:
        """Reject a supplied rationale that contains no content."""

        if rationale == "":
            raise ValueError("rationale must not be empty when supplied.")

        return rationale

    @field_validator("affected_paths")
    @classmethod
    def normalize_affected_paths(cls, paths: list[str]) -> list[str]:
        """Normalize and validate repository-relative affected paths."""

        return _normalize_unique_paths(paths)

    @field_validator("components")
    @classmethod
    def validate_components(cls, components: list[str]) -> list[str]:
        """Require unique, non-empty component names."""

        return _validate_unique_non_empty_strings(components, "Components")

    @field_validator("recorded_at")
    @classmethod
    def normalize_recorded_at(cls, value: datetime) -> datetime:
        """Require an aware decision timestamp and normalize it to UTC."""

        return _normalize_aware_datetime(value, "recorded_at")

    @model_validator(mode="after")
    def validate_evidence(self) -> Self:
        """Validate evidence identity, uniqueness, and required strength."""

        if any(item.source.repository != self.repository for item in self.evidence):
            raise ValueError("Every evidence source must belong to the decision repository.")

        evidence_keys = [
            (
                item.source.key,
                item.kind,
                item.relationship,
                item.source_anchor,
            )
            for item in self.evidence
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("Decision evidence entries must not contain duplicates.")

        statuses_requiring_strong_evidence = {
            DecisionStatus.CONFIRMED,
            DecisionStatus.SUPERSEDED,
            DecisionStatus.REJECTED,
            DecisionStatus.EXCEPTION,
        }
        qualifying_strengths = {
            DecisionEvidenceStrength.STRONG,
            DecisionEvidenceStrength.AUTHORITATIVE,
        }
        has_qualifying_evidence = any(
            item.strength in qualifying_strengths
            and item.relationship != DecisionEvidenceRelationship.CONTRADICTS
            for item in self.evidence
        )

        if self.status in statuses_requiring_strong_evidence and not has_qualifying_evidence:
            raise ValueError(
                "Established decisions require strong or authoritative "
                "non-contradicting evidence."
            )

        return self

    @computed_field
    @property
    def key(self) -> str:
        """Return the stable repository-qualified decision identifier."""

        return f"{self.repository.key}:decision:{self.decision_id}"
