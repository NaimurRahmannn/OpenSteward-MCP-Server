"""Explicit pull-request outcome evidence and conservative decision resolution."""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import Field, computed_field, field_validator, model_validator

from opensteward.knowledge.decision_candidates import (
    KnowledgeDecisionCandidateExtractionResult,
)
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

MAX_KNOWLEDGE_DECISION_OUTCOME_LINKS = 2_000

_DOCUMENTARY_SOURCE_TYPES = {
    KnowledgeItemType.ADR,
    KnowledgeItemType.MAINTAINER_DECISION,
    KnowledgeItemType.RELEASE_NOTE,
}
_COMPLETED_PULL_REQUEST_STATES = {
    KnowledgeItemState.MERGED,
    KnowledgeItemState.REJECTED,
}
_ESTABLISHED_DOCUMENTARY_STATES = {
    KnowledgeItemState.ACTIVE,
    KnowledgeItemState.PUBLISHED,
}
_ESTABLISHED_EVIDENCE_STRENGTHS = {
    DecisionEvidenceStrength.STRONG,
    DecisionEvidenceStrength.AUTHORITATIVE,
}
_RESOLVED_STATUSES = {
    DecisionStatus.CANDIDATE,
    DecisionStatus.CONFIRMED,
    DecisionStatus.REJECTED,
    DecisionStatus.SUPERSEDED,
}


class KnowledgeDecisionResolutionError(ValueError):
    """Raised when local decision-resolution inputs are inconsistent."""


class KnowledgeDecisionOutcomeEffect(StrEnum):
    """Caller-asserted effect of a pull-request outcome."""

    CONFIRMS = "confirms"
    REJECTS = "rejects"


class KnowledgeDecisionResolutionReason(StrEnum):
    """Exact reason for one resolved decision status."""

    DOCUMENTARY_SOURCE_ESTABLISHED = "documentary_source_established"
    DOCUMENTARY_SOURCE_SUPERSEDED = "documentary_source_superseded"
    EXPLICIT_OUTCOME_CONFIRMATION = "explicit_outcome_confirmation"
    EXPLICIT_OUTCOME_REJECTION = "explicit_outcome_rejection"
    CONFLICTING_OUTCOME_EVIDENCE = "conflicting_outcome_evidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


_REASON_STATUSES = {
    KnowledgeDecisionResolutionReason.DOCUMENTARY_SOURCE_ESTABLISHED: (
        DecisionStatus.CONFIRMED
    ),
    KnowledgeDecisionResolutionReason.DOCUMENTARY_SOURCE_SUPERSEDED: (
        DecisionStatus.SUPERSEDED
    ),
    KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_CONFIRMATION: (
        DecisionStatus.CONFIRMED
    ),
    KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_REJECTION: (
        DecisionStatus.REJECTED
    ),
    KnowledgeDecisionResolutionReason.CONFLICTING_OUTCOME_EVIDENCE: (
        DecisionStatus.CANDIDATE
    ),
    KnowledgeDecisionResolutionReason.INSUFFICIENT_EVIDENCE: (
        DecisionStatus.CANDIDATE
    ),
}


class KnowledgeDecisionResolutionOptions(StrictKnowledgeModel):
    """Safety bounds for one decision-resolution operation."""

    max_outcome_links: int = Field(
        default=1_000,
        ge=0,
        le=MAX_KNOWLEDGE_DECISION_OUTCOME_LINKS,
    )


class KnowledgeDecisionOutcomeLink(StrictKnowledgeModel):
    """One explicit caller assertion linking a decision and PR outcome."""

    decision_id: str = Field(min_length=1)
    source: KnowledgeItemReference
    effect: KnowledgeDecisionOutcomeEffect
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_pull_request_source(self) -> Self:
        """Require a pull-request reference without rewriting caller text."""

        if self.source.item_type != KnowledgeItemType.PULL_REQUEST:
            raise ValueError("Decision outcome links require a pull-request source.")
        return self

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the linked pull-request source key."""

        return self.source.key

    @computed_field
    @property
    def key(self) -> str:
        """Return the unique decision and pull-request pair."""

        return f"{self.decision_id}:{self.source.key}"


class KnowledgeDecisionResolution(StrictKnowledgeModel):
    """One resolved status and its exact resolution provenance."""

    decision_id: str = Field(min_length=1)
    candidate_key: str = Field(min_length=1)
    status: DecisionStatus
    reason: KnowledgeDecisionResolutionReason
    outcome_source_keys: list[str]

    @field_validator("outcome_source_keys")
    @classmethod
    def validate_outcome_source_keys(cls, source_keys: list[str]) -> list[str]:
        """Require non-empty, unique, ascending outcome source keys."""

        if any(not source_key for source_key in source_keys):
            raise ValueError("Outcome source keys must be non-empty.")
        if len(source_keys) != len(set(source_keys)):
            raise ValueError("Outcome source keys must be unique.")
        if source_keys != sorted(source_keys):
            raise ValueError("Outcome source keys must use ascending order.")
        return source_keys

    @model_validator(mode="after")
    def validate_status_and_reason(self) -> Self:
        """Require one supported status and its corresponding exact reason."""

        if self.status not in _RESOLVED_STATUSES:
            raise ValueError("Decision resolution status is not supported.")
        if _REASON_STATUSES[self.reason] != self.status:
            raise ValueError(
                "Decision resolution reason must match the resolved status."
            )
        return self

    @computed_field
    @property
    def outcome_count(self) -> int:
        """Return the number of linked pull-request outcomes."""

        return len(self.outcome_source_keys)


class KnowledgeDecisionResolutionStats(StrictKnowledgeModel):
    """Accounting for candidates, links, evidence, and resolved statuses."""

    candidates_seen: int = Field(ge=0)
    outcome_links_seen: int = Field(ge=0)
    outcome_evidence_added: int = Field(ge=0)
    confirmed_decisions: int = Field(ge=0)
    rejected_decisions: int = Field(ge=0)
    superseded_decisions: int = Field(ge=0)
    remaining_candidates: int = Field(ge=0)
    conflicting_decisions: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_accounting(self) -> Self:
        """Require exact evidence and status accounting."""

        if self.outcome_evidence_added != self.outcome_links_seen:
            raise ValueError(
                "outcome_evidence_added must equal outcome_links_seen."
            )
        status_total = (
            self.confirmed_decisions
            + self.rejected_decisions
            + self.superseded_decisions
            + self.remaining_candidates
        )
        if status_total != self.candidates_seen:
            raise ValueError(
                "Decision status counts must sum to candidates_seen."
            )
        if self.conflicting_decisions > self.remaining_candidates:
            raise ValueError(
                "conflicting_decisions must not exceed remaining_candidates."
            )
        return self


class KnowledgeDecisionResolutionResult(StrictKnowledgeModel):
    """Resolved decisions, explicit links, provenance, and complete accounting."""

    repository: KnowledgeRepositoryRef
    recorded_at: datetime
    source_extraction_complete: bool
    decisions: list[DecisionRecord]
    resolutions: list[KnowledgeDecisionResolution]
    outcome_links: list[KnowledgeDecisionOutcomeLink]
    stats: KnowledgeDecisionResolutionStats

    @field_validator("recorded_at")
    @classmethod
    def normalize_recorded_at(cls, recorded_at: datetime) -> datetime:
        """Require an aware resolution time and normalize it to UTC."""

        if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
            raise ValueError("recorded_at must be timezone-aware.")
        return recorded_at.astimezone(UTC)

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate decision, resolution, link, and statistics consistency."""

        decisions_by_id = self._validate_decisions()
        resolutions_by_id = self._validate_resolutions(decisions_by_id)
        links_by_decision = self._validate_outcome_links(decisions_by_id)
        self._validate_link_coverage(resolutions_by_id, links_by_decision)
        self._validate_accounting()
        return self

    def _validate_decisions(self) -> dict[str, DecisionRecord]:
        if any(decision.repository != self.repository for decision in self.decisions):
            raise ValueError(
                "Every resolved decision must belong to the result repository."
            )
        if any(
            decision.recorded_at != self.recorded_at
            for decision in self.decisions
        ):
            raise ValueError(
                "Every resolved decision must use the result recorded_at."
            )
        if any(decision.status not in _RESOLVED_STATUSES for decision in self.decisions):
            raise ValueError("Resolved decisions contain an unsupported status.")
        if any(not decision.evidence for decision in self.decisions):
            raise ValueError(
                "Every resolved decision must contain at least one evidence record."
            )
        if any(
            evidence.source.repository != self.repository
            for decision in self.decisions
            for evidence in decision.evidence
        ):
            raise ValueError(
                "Every resolved decision evidence source must belong to the "
                "result repository."
            )

        decision_keys = [decision.key for decision in self.decisions]
        decision_ids = [decision.decision_id for decision in self.decisions]
        if len(decision_keys) != len(set(decision_keys)):
            raise ValueError("Resolved decision keys must be unique.")
        if len(decision_ids) != len(set(decision_ids)):
            raise ValueError("Resolved decision IDs must be unique.")
        if decision_keys != sorted(decision_keys):
            raise ValueError(
                "Resolved decisions must use decision-key ascending order."
            )
        return {
            decision.decision_id: decision
            for decision in self.decisions
        }

    def _validate_resolutions(
        self,
        decisions_by_id: dict[str, DecisionRecord],
    ) -> dict[str, KnowledgeDecisionResolution]:
        resolution_ids = [
            resolution.decision_id
            for resolution in self.resolutions
        ]
        if len(resolution_ids) != len(set(resolution_ids)):
            raise ValueError("Resolution decision IDs must be unique.")
        if resolution_ids != sorted(resolution_ids):
            raise ValueError("Resolutions must use decision-ID ascending order.")
        if set(resolution_ids) != set(decisions_by_id):
            raise ValueError("Exactly one resolution must exist per decision.")

        for resolution in self.resolutions:
            decision = decisions_by_id[resolution.decision_id]
            if resolution.status != decision.status:
                raise ValueError(
                    "Resolution status must match the associated decision status."
                )
            if resolution.candidate_key != decision.key:
                raise ValueError(
                    "Resolution candidate_key must match the associated decision key."
                )
        return {
            resolution.decision_id: resolution
            for resolution in self.resolutions
        }

    def _validate_outcome_links(
        self,
        decisions_by_id: dict[str, DecisionRecord],
    ) -> dict[str, list[str]]:
        if any(
            link.source.repository != self.repository
            for link in self.outcome_links
        ):
            raise ValueError(
                "Every outcome link must belong to the result repository."
            )

        link_keys = [link.key for link in self.outcome_links]
        if len(link_keys) != len(set(link_keys)):
            raise ValueError("Outcome link keys must be unique.")
        expected_order = sorted(
            self.outcome_links,
            key=lambda link: (link.decision_id, link.source.key),
        )
        if self.outcome_links != expected_order:
            raise ValueError(
                "Outcome links must use decision-ID and source-key ascending order."
            )
        if any(
            link.decision_id not in decisions_by_id
            for link in self.outcome_links
        ):
            raise ValueError(
                "Every outcome link must identify a resolved decision."
            )

        links_by_decision = {
            decision_id: []
            for decision_id in decisions_by_id
        }
        for link in self.outcome_links:
            links_by_decision[link.decision_id].append(link.source.key)
        return links_by_decision

    @staticmethod
    def _validate_link_coverage(
        resolutions_by_id: dict[str, KnowledgeDecisionResolution],
        links_by_decision: dict[str, list[str]],
    ) -> None:
        for decision_id, resolution in resolutions_by_id.items():
            if resolution.outcome_source_keys != links_by_decision[decision_id]:
                raise ValueError(
                    "Resolution outcome source keys must match associated links."
                )

    def _validate_accounting(self) -> None:
        if len(self.decisions) != self.stats.candidates_seen:
            raise ValueError(
                "Decision count must equal stats.candidates_seen."
            )
        if len(self.outcome_links) != self.stats.outcome_links_seen:
            raise ValueError(
                "Outcome link count must equal stats.outcome_links_seen."
            )
        if sum(
            resolution.outcome_count
            for resolution in self.resolutions
        ) != self.stats.outcome_evidence_added:
            raise ValueError(
                "Resolution outcome counts must equal stats.outcome_evidence_added."
            )

        expected_counts = {
            DecisionStatus.CONFIRMED: self.stats.confirmed_decisions,
            DecisionStatus.REJECTED: self.stats.rejected_decisions,
            DecisionStatus.SUPERSEDED: self.stats.superseded_decisions,
            DecisionStatus.CANDIDATE: self.stats.remaining_candidates,
        }
        for status, expected_count in expected_counts.items():
            actual_count = sum(
                decision.status == status
                for decision in self.decisions
            )
            if actual_count != expected_count:
                raise ValueError(
                    "Decision status counts must match resolved decisions."
                )

        conflict_count = sum(
            resolution.reason
            == KnowledgeDecisionResolutionReason.CONFLICTING_OUTCOME_EVIDENCE
            for resolution in self.resolutions
        )
        if conflict_count != self.stats.conflicting_decisions:
            raise ValueError(
                "Conflict count must match conflicting decision resolutions."
            )

    @computed_field
    @property
    def decision_count(self) -> int:
        """Return the number of resolved decision records."""

        return len(self.decisions)

    def _count_status(self, status: DecisionStatus) -> int:
        return sum(decision.status == status for decision in self.decisions)

    @computed_field
    @property
    def confirmed_count(self) -> int:
        """Return the number of confirmed decisions."""

        return self._count_status(DecisionStatus.CONFIRMED)

    @computed_field
    @property
    def rejected_count(self) -> int:
        """Return the number of rejected decisions."""

        return self._count_status(DecisionStatus.REJECTED)

    @computed_field
    @property
    def superseded_count(self) -> int:
        """Return the number of superseded decisions."""

        return self._count_status(DecisionStatus.SUPERSEDED)

    @computed_field
    @property
    def candidate_count(self) -> int:
        """Return the number of unresolved candidates."""

        return self._count_status(DecisionStatus.CANDIDATE)

    @computed_field
    @property
    def has_conflicts(self) -> bool:
        """Return whether any candidate retained conflicting outcomes."""

        return self.stats.conflicting_decisions > 0

    @computed_field
    @property
    def complete(self) -> bool:
        """Return whether extraction and resolution both have full coverage."""

        return (
            self.source_extraction_complete
            and self.stats.remaining_candidates == 0
        )


def _build_item_lookup(
    extraction: KnowledgeDecisionCandidateExtractionResult,
    items: list[KnowledgeItem],
) -> dict[str, KnowledgeItem]:
    if any(item.repository != extraction.repository for item in items):
        raise KnowledgeDecisionResolutionError(
            "Every resolution source item must belong to the extraction repository."
        )
    item_keys = [item.key for item in items]
    if len(item_keys) != len(set(item_keys)):
        raise KnowledgeDecisionResolutionError(
            "Resolution source item keys must be unique."
        )
    return {item.key: item for item in items}


def _validate_candidates(
    extraction: KnowledgeDecisionCandidateExtractionResult,
    items_by_key: dict[str, KnowledgeItem],
) -> dict[str, tuple[DecisionRecord, KnowledgeItem]]:
    decision_ids = [
        candidate.decision_id
        for candidate in extraction.candidates
    ]
    if len(decision_ids) != len(set(decision_ids)):
        raise KnowledgeDecisionResolutionError(
            "Extracted candidate decision IDs must be unique."
        )

    for candidate in extraction.candidates:
        if candidate.repository != extraction.repository:
            raise KnowledgeDecisionResolutionError(
                "Every extracted candidate must belong to the extraction repository."
            )
        if candidate.status != DecisionStatus.CANDIDATE:
            raise KnowledgeDecisionResolutionError(
                "Resolution accepts only CANDIDATE decision records."
            )
        if len(candidate.evidence) != 1:
            raise KnowledgeDecisionResolutionError(
                "Every extracted candidate must contain exactly one documentary "
                "evidence record."
            )

        source_reference = candidate.evidence[0].source
        if source_reference.item_type not in _DOCUMENTARY_SOURCE_TYPES:
            raise KnowledgeDecisionResolutionError(
                "Candidate evidence must use a documentary source type."
            )

    candidates_by_id: dict[str, tuple[DecisionRecord, KnowledgeItem]] = {}
    for candidate in extraction.candidates:
        source_reference = candidate.evidence[0].source
        source_item = items_by_key.get(source_reference.key)
        if source_item is None:
            raise KnowledgeDecisionResolutionError(
                "Candidate documentary source item is unavailable."
            )
        if source_reference != source_item.to_reference():
            raise KnowledgeDecisionResolutionError(
                "Candidate documentary source reference does not match the "
                "authoritative item."
            )
        candidates_by_id[candidate.decision_id] = (candidate, source_item)
    return candidates_by_id


def _validate_outcome_links(
    outcome_links: list[KnowledgeDecisionOutcomeLink],
    candidates_by_id: dict[str, tuple[DecisionRecord, KnowledgeItem]],
    items_by_key: dict[str, KnowledgeItem],
) -> dict[str, list[tuple[KnowledgeDecisionOutcomeLink, KnowledgeItem]]]:
    link_keys = [link.key for link in outcome_links]
    if len(link_keys) != len(set(link_keys)):
        raise KnowledgeDecisionResolutionError(
            "Decision outcome link keys must be unique."
        )

    if any(link.decision_id not in candidates_by_id for link in outcome_links):
        raise KnowledgeDecisionResolutionError(
            "Decision outcome link identifies an unknown candidate."
        )
    if any(link.source.key not in items_by_key for link in outcome_links):
        raise KnowledgeDecisionResolutionError(
            "Decision outcome source item is unavailable."
        )
    if any(
        link.source != items_by_key[link.source.key].to_reference()
        for link in outcome_links
    ):
        raise KnowledgeDecisionResolutionError(
            "Decision outcome source reference does not match the authoritative item."
        )
    if any(
        items_by_key[link.source.key].item_type
        != KnowledgeItemType.PULL_REQUEST
        or items_by_key[link.source.key].state
        not in _COMPLETED_PULL_REQUEST_STATES
        for link in outcome_links
    ):
        raise KnowledgeDecisionResolutionError(
            "Decision outcome source must be a merged or rejected pull request."
        )

    links_by_decision = {
        decision_id: []
        for decision_id in candidates_by_id
    }
    for link in outcome_links:
        source_item = items_by_key[link.source.key]
        links_by_decision[link.decision_id].append((link, source_item))

    for links in links_by_decision.values():
        links.sort(key=lambda entry: entry[1].key)
    return links_by_decision


def _build_outcome_evidence(
    link: KnowledgeDecisionOutcomeLink,
    source_item: KnowledgeItem,
) -> DecisionEvidence:
    kind = (
        DecisionEvidenceKind.MERGE_OUTCOME
        if source_item.state == KnowledgeItemState.MERGED
        else DecisionEvidenceKind.REJECTION_OUTCOME
    )
    return DecisionEvidence(
        source=source_item.to_reference(),
        kind=kind,
        relationship=DecisionEvidenceRelationship.SUPPORTS,
        strength=DecisionEvidenceStrength.STRONG,
        summary=link.explanation,
        excerpt=None,
        source_anchor=None,
        author=source_item.author,
        occurred_at=source_item.closed_at or source_item.updated_at,
    )


def _resolve_status(
    documentary_source: KnowledgeItem,
    documentary_evidence: DecisionEvidence,
    links: list[tuple[KnowledgeDecisionOutcomeLink, KnowledgeItem]],
) -> tuple[DecisionStatus, KnowledgeDecisionResolutionReason]:
    if documentary_source.state == KnowledgeItemState.SUPERSEDED:
        return (
            DecisionStatus.SUPERSEDED,
            KnowledgeDecisionResolutionReason.DOCUMENTARY_SOURCE_SUPERSEDED,
        )

    effects = {link.effect for link, _ in links}
    if len(effects) > 1:
        return (
            DecisionStatus.CANDIDATE,
            KnowledgeDecisionResolutionReason.CONFLICTING_OUTCOME_EVIDENCE,
        )
    if effects == {KnowledgeDecisionOutcomeEffect.REJECTS}:
        return (
            DecisionStatus.REJECTED,
            KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_REJECTION,
        )
    if effects == {KnowledgeDecisionOutcomeEffect.CONFIRMS}:
        return (
            DecisionStatus.CONFIRMED,
            KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_CONFIRMATION,
        )
    if documentary_source.state in _ESTABLISHED_DOCUMENTARY_STATES:
        if documentary_evidence.strength not in _ESTABLISHED_EVIDENCE_STRENGTHS:
            raise KnowledgeDecisionResolutionError(
                "Established documentary sources require strong or authoritative "
                "original evidence."
            )
        return (
            DecisionStatus.CONFIRMED,
            KnowledgeDecisionResolutionReason.DOCUMENTARY_SOURCE_ESTABLISHED,
        )
    return (
        DecisionStatus.CANDIDATE,
        KnowledgeDecisionResolutionReason.INSUFFICIENT_EVIDENCE,
    )


def _resolve_candidate(
    candidate: DecisionRecord,
    documentary_source: KnowledgeItem,
    links: list[tuple[KnowledgeDecisionOutcomeLink, KnowledgeItem]],
    *,
    recorded_at: datetime,
) -> tuple[DecisionRecord, KnowledgeDecisionResolution]:
    documentary_evidence = candidate.evidence[0].model_copy(deep=True)
    status, reason = _resolve_status(
        documentary_source,
        documentary_evidence,
        links,
    )
    outcome_evidence = [
        _build_outcome_evidence(link, source_item)
        for link, source_item in links
    ]
    decision = DecisionRecord(
        repository=candidate.repository,
        decision_id=candidate.decision_id,
        statement=candidate.statement,
        status=status,
        significance=candidate.significance,
        rationale=candidate.rationale,
        affected_paths=list(candidate.affected_paths),
        components=list(candidate.components),
        evidence=[documentary_evidence, *outcome_evidence],
        recorded_at=recorded_at,
    )
    resolution = KnowledgeDecisionResolution(
        decision_id=candidate.decision_id,
        candidate_key=decision.key,
        status=status,
        reason=reason,
        outcome_source_keys=[
            source_item.key
            for _, source_item in links
        ],
    )
    return decision, resolution


def _build_stats(
    decisions: list[DecisionRecord],
    resolutions: list[KnowledgeDecisionResolution],
    outcome_link_count: int,
) -> KnowledgeDecisionResolutionStats:
    return KnowledgeDecisionResolutionStats(
        candidates_seen=len(decisions),
        outcome_links_seen=outcome_link_count,
        outcome_evidence_added=sum(
            resolution.outcome_count
            for resolution in resolutions
        ),
        confirmed_decisions=sum(
            decision.status == DecisionStatus.CONFIRMED
            for decision in decisions
        ),
        rejected_decisions=sum(
            decision.status == DecisionStatus.REJECTED
            for decision in decisions
        ),
        superseded_decisions=sum(
            decision.status == DecisionStatus.SUPERSEDED
            for decision in decisions
        ),
        remaining_candidates=sum(
            decision.status == DecisionStatus.CANDIDATE
            for decision in decisions
        ),
        conflicting_decisions=sum(
            resolution.reason
            == KnowledgeDecisionResolutionReason.CONFLICTING_OUTCOME_EVIDENCE
            for resolution in resolutions
        ),
    )


def resolve_knowledge_decision_candidates(
    extraction: KnowledgeDecisionCandidateExtractionResult,
    items: list[KnowledgeItem],
    outcome_links: list[KnowledgeDecisionOutcomeLink],
    *,
    recorded_at: datetime,
    options: KnowledgeDecisionResolutionOptions | None = None,
) -> KnowledgeDecisionResolutionResult:
    """Resolve documentary candidates using only explicit completed-PR links."""

    if recorded_at.tzinfo is None or recorded_at.utcoffset() is None:
        raise ValueError("recorded_at must be timezone-aware.")
    normalized_recorded_at = recorded_at.astimezone(UTC)
    effective_options = options or KnowledgeDecisionResolutionOptions()
    if len(outcome_links) > effective_options.max_outcome_links:
        raise KnowledgeDecisionResolutionError(
            "Decision outcome link count exceeds the configured safety limit."
        )

    items_by_key = _build_item_lookup(extraction, items)
    candidates_by_id = _validate_candidates(extraction, items_by_key)
    links_by_decision = _validate_outcome_links(
        outcome_links,
        candidates_by_id,
        items_by_key,
    )

    decisions: list[DecisionRecord] = []
    resolutions: list[KnowledgeDecisionResolution] = []
    for decision_id, (candidate, documentary_source) in candidates_by_id.items():
        decision, resolution = _resolve_candidate(
            candidate,
            documentary_source,
            links_by_decision[decision_id],
            recorded_at=normalized_recorded_at,
        )
        decisions.append(decision)
        resolutions.append(resolution)

    decisions.sort(key=lambda decision: decision.key)
    resolutions.sort(key=lambda resolution: resolution.decision_id)
    ordered_links = sorted(
        outcome_links,
        key=lambda link: (link.decision_id, link.source.key),
    )
    stats = _build_stats(decisions, resolutions, len(ordered_links))
    return KnowledgeDecisionResolutionResult(
        repository=extraction.repository,
        recorded_at=normalized_recorded_at,
        source_extraction_complete=extraction.complete,
        decisions=decisions,
        resolutions=resolutions,
        outcome_links=ordered_links,
        stats=stats,
    )
