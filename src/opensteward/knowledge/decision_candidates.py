"""Conservative documentary decision-candidate extraction."""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, computed_field, field_validator, model_validator

from opensteward.knowledge.models import (
    DecisionEvidence,
    DecisionEvidenceKind,
    DecisionEvidenceRelationship,
    DecisionEvidenceStrength,
    DecisionRecord,
    DecisionStatus,
    KnowledgeItem,
    KnowledgeItemReference,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    StrictKnowledgeModel,
)

MAX_KNOWLEDGE_DECISION_CANDIDATES = 500

_DECISION_CANDIDATE_ID_PREFIX = "candidate"
_DECISION_CANDIDATE_RULE_VERSION = "v1"
_DECISION_CANDIDATE_DIGEST_LENGTH = 24

_UNSUPPORTED_EXPLANATION = (
    "This knowledge-item type is not an explicit documentary decision source."
)
_INELIGIBLE_RELEASE_EXPLANATION = (
    "Release-note decision candidates require a published or active source state."
)
_CANDIDATE_LIMIT_EXPLANATION = (
    "Decision candidate was omitted by the configured candidate safety limit."
)
_ADR_ACCEPTED_SUMMARY = (
    "Repository ADR provides formal documentary evidence for this decision candidate."
)
_ADR_DOCUMENTARY_SUMMARY = (
    "Repository ADR provides documentary evidence for this decision candidate."
)
_MAINTAINER_SUMMARY = (
    "Maintainer decision item provides explicit decision evidence."
)
_RELEASE_SUMMARY = (
    "Published release material provides commitment evidence for this decision candidate."
)
_ESTABLISHED_DOCUMENT_STATES = {
    KnowledgeItemState.ACTIVE,
    KnowledgeItemState.PUBLISHED,
    KnowledgeItemState.SUPERSEDED,
}
_ELIGIBLE_RELEASE_STATES = {
    KnowledgeItemState.ACTIVE,
    KnowledgeItemState.PUBLISHED,
}
_DOCUMENTARY_ITEM_TYPES = {
    KnowledgeItemType.ADR,
    KnowledgeItemType.MAINTAINER_DECISION,
    KnowledgeItemType.RELEASE_NOTE,
}


class KnowledgeDecisionCandidateRule(StrEnum):
    """Fixed rules that admit documentary decision candidates."""

    ADR_DOCUMENT = "adr_document"
    MAINTAINER_DECISION = "maintainer_decision"
    RELEASE_COMMITMENT = "release_commitment"


class KnowledgeDecisionCandidateSkipReason(StrEnum):
    """Reasons a source item did not produce a candidate."""

    UNSUPPORTED_ITEM_TYPE = "unsupported_item_type"
    INELIGIBLE_STATE = "ineligible_state"
    CANDIDATE_LIMIT = "candidate_limit"


class KnowledgeDecisionCandidateExtractionOptions(StrictKnowledgeModel):
    """Safety bounds for documentary candidate extraction."""

    max_candidates: int = Field(
        default=100,
        ge=0,
        le=MAX_KNOWLEDGE_DECISION_CANDIDATES,
    )


class KnowledgeDecisionCandidateSkippedSource(StrictKnowledgeModel):
    """One fully classified source that did not produce a candidate."""

    source: KnowledgeItemReference
    reason: KnowledgeDecisionCandidateSkipReason
    explanation: str = Field(min_length=1)

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the skipped source item key."""

        return self.source.key


class KnowledgeDecisionCandidateExtractionStats(StrictKnowledgeModel):
    """Accounting for one complete source classification pass."""

    source_items_seen: int = Field(ge=0)
    eligible_source_items: int = Field(ge=0)
    candidates_created: int = Field(ge=0)
    skipped_sources: int = Field(ge=0)
    candidate_limit_reached: bool

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        """Require internally consistent extraction counts."""

        if self.eligible_source_items > self.source_items_seen:
            raise ValueError(
                "eligible_source_items must not exceed source_items_seen."
            )
        if self.candidates_created > self.eligible_source_items:
            raise ValueError(
                "candidates_created must not exceed eligible_source_items."
            )
        if self.skipped_sources != (
            self.source_items_seen - self.candidates_created
        ):
            raise ValueError(
                "skipped_sources must equal source_items_seen minus "
                "candidates_created."
            )
        expected_limit_reached = (
            self.eligible_source_items > self.candidates_created
        )
        if self.candidate_limit_reached != expected_limit_reached:
            raise ValueError(
                "candidate_limit_reached must identify omitted eligible sources."
            )
        return self


class KnowledgeDecisionCandidateExtractionResult(StrictKnowledgeModel):
    """Deterministic candidates, skipped sources, and extraction accounting."""

    repository: KnowledgeRepositoryRef
    recorded_at: datetime
    candidates: list[DecisionRecord]
    skipped_sources: list[KnowledgeDecisionCandidateSkippedSource]
    stats: KnowledgeDecisionCandidateExtractionStats

    @field_validator("recorded_at")
    @classmethod
    def normalize_recorded_at(cls, recorded_at: datetime) -> datetime:
        """Require an aware extraction time and normalize it to UTC."""

        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware.")
        return recorded_at.astimezone(UTC)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate candidate identity, ordering, provenance, and accounting."""

        self._validate_candidates()
        self._validate_skipped_sources()
        self._validate_source_accounting()
        return self

    def _validate_candidates(self) -> None:
        candidate_keys = [candidate.key for candidate in self.candidates]
        decision_ids = [candidate.decision_id for candidate in self.candidates]
        if len(candidate_keys) != len(set(candidate_keys)):
            raise ValueError("Decision candidate keys must be unique.")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("Decision candidate IDs must be unique.")
        if candidate_keys != sorted(candidate_keys):
            raise ValueError(
                "Decision candidates must use candidate-key ascending order."
            )

        for candidate in self.candidates:
            if candidate.repository != self.repository:
                raise ValueError(
                    "Every decision candidate must belong to the result repository."
                )
            if candidate.status != DecisionStatus.CANDIDATE:
                raise ValueError(
                    "Extracted decision records must have CANDIDATE status."
                )
            if candidate.recorded_at != self.recorded_at:
                raise ValueError(
                    "Every decision candidate must use the result recorded_at."
                )
            if len(candidate.evidence) != 1:
                raise ValueError(
                    "Every decision candidate must contain exactly one evidence entry."
                )

            evidence = candidate.evidence[0]
            if evidence.source.repository != self.repository:
                raise ValueError(
                    "Every candidate evidence source must belong to the result "
                    "repository."
                )
            if evidence.relationship != DecisionEvidenceRelationship.SUPPORTS:
                raise ValueError(
                    "Decision candidate evidence must use the SUPPORTS relationship."
                )
            if evidence.source.item_type not in _DOCUMENTARY_ITEM_TYPES:
                raise ValueError(
                    "Decision candidate evidence must use a documentary source type."
                )

    def _validate_skipped_sources(self) -> None:
        if any(
            skipped.source.repository != self.repository
            for skipped in self.skipped_sources
        ):
            raise ValueError(
                "Every skipped source must belong to the result repository."
            )

        skipped_keys = [skipped.item_key for skipped in self.skipped_sources]
        if len(skipped_keys) != len(set(skipped_keys)):
            raise ValueError("Skipped source item keys must be unique.")
        if skipped_keys != sorted(skipped_keys):
            raise ValueError(
                "Skipped sources must use item-key ascending order."
            )

    def _validate_source_accounting(self) -> None:
        candidate_source_keys = [
            candidate.evidence[0].source.key
            for candidate in self.candidates
        ]
        skipped_source_keys = [
            skipped.item_key
            for skipped in self.skipped_sources
        ]
        if set(candidate_source_keys) & set(skipped_source_keys):
            raise ValueError(
                "Candidate and skipped source item keys must be mutually exclusive."
            )

        represented_keys = [*candidate_source_keys, *skipped_source_keys]
        if len(represented_keys) != len(set(represented_keys)):
            raise ValueError("Every represented source item key must be unique.")
        if len(represented_keys) != self.stats.source_items_seen:
            raise ValueError(
                "Represented source item keys must equal stats.source_items_seen."
            )
        if len(self.candidates) != self.stats.candidates_created:
            raise ValueError(
                "Candidate count must equal stats.candidates_created."
            )
        if len(self.skipped_sources) != self.stats.skipped_sources:
            raise ValueError(
                "Skipped count must equal stats.skipped_sources."
            )

        limit_skipped = sum(
            skipped.reason == KnowledgeDecisionCandidateSkipReason.CANDIDATE_LIMIT
            for skipped in self.skipped_sources
        )
        if self.stats.eligible_source_items != (
            len(self.candidates) + limit_skipped
        ):
            raise ValueError(
                "Eligible source accounting must include candidates and "
                "candidate-limit skips."
            )

    @computed_field
    @property
    def candidate_count(self) -> int:
        """Return the number of extracted candidates."""

        return len(self.candidates)

    @computed_field
    @property
    def skipped_count(self) -> int:
        """Return the number of classified skipped sources."""

        return len(self.skipped_sources)

    @computed_field
    @property
    def complete(self) -> bool:
        """Return whether no eligible source was omitted by the safety limit."""

        return not self.stats.candidate_limit_reached


@dataclass(frozen=True)
class _EvidenceRule:
    rule: KnowledgeDecisionCandidateRule
    kind: DecisionEvidenceKind
    strength: DecisionEvidenceStrength
    summary: str


@dataclass(frozen=True)
class _EligibleSource:
    item: KnowledgeItem
    evidence_rule: _EvidenceRule


def _candidate_id(item: KnowledgeItem) -> str:
    identity = (
        "opensteward:decision-candidate:"
        f"{_DECISION_CANDIDATE_RULE_VERSION}:{item.key}"
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return (
        f"{_DECISION_CANDIDATE_ID_PREFIX}-"
        f"{digest[:_DECISION_CANDIDATE_DIGEST_LENGTH]}"
    )


def _classify_adr(item: KnowledgeItem) -> _EligibleSource:
    if item.state in _ESTABLISHED_DOCUMENT_STATES:
        rule = _EvidenceRule(
            rule=KnowledgeDecisionCandidateRule.ADR_DOCUMENT,
            kind=DecisionEvidenceKind.ACCEPTED_DOCUMENT,
            strength=DecisionEvidenceStrength.AUTHORITATIVE,
            summary=_ADR_ACCEPTED_SUMMARY,
        )
    else:
        rule = _EvidenceRule(
            rule=KnowledgeDecisionCandidateRule.ADR_DOCUMENT,
            kind=DecisionEvidenceKind.OTHER,
            strength=DecisionEvidenceStrength.MODERATE,
            summary=_ADR_DOCUMENTARY_SUMMARY,
        )
    return _EligibleSource(item=item, evidence_rule=rule)


def _classify_maintainer_decision(item: KnowledgeItem) -> _EligibleSource:
    strength = (
        DecisionEvidenceStrength.STRONG
        if item.state in _ESTABLISHED_DOCUMENT_STATES
        else DecisionEvidenceStrength.MODERATE
    )
    return _EligibleSource(
        item=item,
        evidence_rule=_EvidenceRule(
            rule=KnowledgeDecisionCandidateRule.MAINTAINER_DECISION,
            kind=DecisionEvidenceKind.MAINTAINER_STATEMENT,
            strength=strength,
            summary=_MAINTAINER_SUMMARY,
        ),
    )


def _classify_source(
    item: KnowledgeItem,
) -> tuple[_EligibleSource | None, KnowledgeDecisionCandidateSkippedSource | None]:
    if item.item_type == KnowledgeItemType.ADR:
        return _classify_adr(item), None
    if item.item_type == KnowledgeItemType.MAINTAINER_DECISION:
        return _classify_maintainer_decision(item), None
    if item.item_type == KnowledgeItemType.RELEASE_NOTE:
        if item.state in _ELIGIBLE_RELEASE_STATES:
            return (
                _EligibleSource(
                    item=item,
                    evidence_rule=_EvidenceRule(
                        rule=KnowledgeDecisionCandidateRule.RELEASE_COMMITMENT,
                        kind=DecisionEvidenceKind.RELEASE_COMMITMENT,
                        strength=DecisionEvidenceStrength.STRONG,
                        summary=_RELEASE_SUMMARY,
                    ),
                ),
                None,
            )
        return None, KnowledgeDecisionCandidateSkippedSource(
            source=item.to_reference(),
            reason=KnowledgeDecisionCandidateSkipReason.INELIGIBLE_STATE,
            explanation=_INELIGIBLE_RELEASE_EXPLANATION,
        )
    return None, KnowledgeDecisionCandidateSkippedSource(
        source=item.to_reference(),
        reason=KnowledgeDecisionCandidateSkipReason.UNSUPPORTED_ITEM_TYPE,
        explanation=_UNSUPPORTED_EXPLANATION,
    )


def _build_candidate(
    source: _EligibleSource,
    *,
    repository: KnowledgeRepositoryRef,
    recorded_at: datetime,
) -> DecisionRecord:
    item = source.item
    evidence_rule = source.evidence_rule
    evidence = DecisionEvidence(
        source=item.to_reference(),
        kind=evidence_rule.kind,
        relationship=DecisionEvidenceRelationship.SUPPORTS,
        strength=evidence_rule.strength,
        summary=evidence_rule.summary,
        excerpt=None,
        source_anchor=None,
        author=item.author,
        occurred_at=item.updated_at,
    )
    return DecisionRecord(
        repository=repository,
        decision_id=_candidate_id(item),
        statement=item.title,
        status=DecisionStatus.CANDIDATE,
        significance=item.decision_significance,
        rationale=item.summary,
        affected_paths=list(item.affected_paths),
        components=list(item.components),
        evidence=[evidence],
        recorded_at=recorded_at,
    )


def extract_knowledge_decision_candidates(
    repository: KnowledgeRepositoryRef,
    items: list[KnowledgeItem],
    *,
    recorded_at: datetime,
    options: KnowledgeDecisionCandidateExtractionOptions | None = None,
) -> KnowledgeDecisionCandidateExtractionResult:
    """Extract bounded documentary candidates and classify every source item."""

    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("recorded_at must be timezone-aware.")
    normalized_recorded_at = recorded_at.astimezone(UTC)

    if any(item.repository != repository for item in items):
        raise ValueError(
            "Every decision-candidate source item must belong to the repository."
        )
    item_keys = [item.key for item in items]
    if len(item_keys) != len(set(item_keys)):
        raise ValueError("Decision-candidate source item keys must be unique.")

    effective_options = (
        options
        if options is not None
        else KnowledgeDecisionCandidateExtractionOptions()
    )
    eligible_sources: list[_EligibleSource] = []
    skipped_sources: list[KnowledgeDecisionCandidateSkippedSource] = []
    for item in sorted(items, key=lambda source: source.key):
        eligible, skipped = _classify_source(item)
        if eligible is not None:
            eligible_sources.append(eligible)
        else:
            if skipped is None:
                raise AssertionError("Every ineligible source must have a skip reason.")
            skipped_sources.append(skipped)

    selected_sources = eligible_sources[:effective_options.max_candidates]
    for source in eligible_sources[effective_options.max_candidates:]:
        skipped_sources.append(
            KnowledgeDecisionCandidateSkippedSource(
                source=source.item.to_reference(),
                reason=KnowledgeDecisionCandidateSkipReason.CANDIDATE_LIMIT,
                explanation=_CANDIDATE_LIMIT_EXPLANATION,
            )
        )

    candidates = [
        _build_candidate(
            source,
            repository=repository,
            recorded_at=normalized_recorded_at,
        )
        for source in selected_sources
    ]
    candidates.sort(key=lambda candidate: candidate.key)
    skipped_sources.sort(key=lambda skipped: skipped.item_key)

    candidates_created = len(candidates)
    eligible_count = len(eligible_sources)
    stats = KnowledgeDecisionCandidateExtractionStats(
        source_items_seen=len(items),
        eligible_source_items=eligible_count,
        candidates_created=candidates_created,
        skipped_sources=len(skipped_sources),
        candidate_limit_reached=eligible_count > candidates_created,
    )
    return KnowledgeDecisionCandidateExtractionResult(
        repository=repository,
        recorded_at=normalized_recorded_at,
        candidates=candidates,
        skipped_sources=skipped_sources,
        stats=stats,
    )
