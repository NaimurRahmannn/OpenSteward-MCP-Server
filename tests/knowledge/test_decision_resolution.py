"""Tests for explicit PR-outcome evidence and decision resolution."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.knowledge import (
    MAX_KNOWLEDGE_DECISION_OUTCOME_LINKS,
    DecisionEvidenceKind,
    DecisionEvidenceRelationship,
    DecisionEvidenceStrength,
    DecisionSignificance,
    DecisionStatus,
    KnowledgeActor,
    KnowledgeActorType,
    KnowledgeDecisionCandidateExtractionOptions,
    KnowledgeDecisionCandidateExtractionResult,
    KnowledgeDecisionCandidateExtractionStats,
    KnowledgeDecisionOutcomeEffect,
    KnowledgeDecisionOutcomeLink,
    KnowledgeDecisionResolution,
    KnowledgeDecisionResolutionError,
    KnowledgeDecisionResolutionOptions,
    KnowledgeDecisionResolutionReason,
    KnowledgeDecisionResolutionResult,
    KnowledgeDecisionResolutionStats,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
    extract_knowledge_decision_candidates,
    resolve_knowledge_decision_candidates,
)

CREATED_AT = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
UPDATED_AT = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
PR_UPDATED_AT = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
PR_CLOSED_AT = datetime(2026, 1, 6, 11, 0, tzinfo=UTC)
EXTRACTED_AT = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
RESOLVED_AT = datetime(2026, 3, 1, 15, 0, tzinfo=UTC)
NON_UTC_RESOLVED_AT = datetime(
    2026,
    3,
    1,
    21,
    0,
    tzinfo=timezone(timedelta(hours=6)),
)
REPOSITORY = KnowledgeRepositoryRef(
    provider="github",
    namespace="acme",
    name="framework",
)
OTHER_REPOSITORY = KnowledgeRepositoryRef(
    provider="gitlab",
    namespace="other",
    name="project",
)
AUTHOR = KnowledgeActor(
    identifier="maintainer",
    actor_type=KnowledgeActorType.USER,
    display_name="A Maintainer",
)
PR_AUTHOR = KnowledgeActor(
    identifier="contributor",
    actor_type=KnowledgeActorType.USER,
    display_name="A Contributor",
)


def make_document(
    *,
    external_id: str = "adr-1",
    repository: KnowledgeRepositoryRef = REPOSITORY,
    item_type: KnowledgeItemType = KnowledgeItemType.ADR,
    state: KnowledgeItemState = KnowledgeItemState.UNKNOWN,
    title: str = "Do not introduce runtime reflection",
    body: str | None = "Runtime reflection is not permitted.",
    summary: str | None = "Prefer static behavior.",
    labels: list[str] | None = None,
    components: list[str] | None = None,
    affected_paths: list[str] | None = None,
    significance: DecisionSignificance = DecisionSignificance.HIGH,
) -> KnowledgeItem:
    """Build one documentary candidate source."""

    return KnowledgeItem(
        repository=repository,
        item_type=item_type,
        external_id=external_id,
        source_kind=KnowledgeSourceKind.REPOSITORY_FILE,
        state=state,
        title=title,
        body=body,
        summary=summary,
        url=f"https://example.test/docs/{external_id}",
        author=AUTHOR,
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
        labels=[] if labels is None else labels,
        components=["Runtime"] if components is None else components,
        affected_paths=(
            ["src/opensteward/runtime.py"]
            if affected_paths is None
            else affected_paths
        ),
        decision_significance=significance,
    )


def make_pull_request(
    *,
    external_id: str = "101",
    repository: KnowledgeRepositoryRef = REPOSITORY,
    state: KnowledgeItemState = KnowledgeItemState.MERGED,
    title: str = "Avoid runtime reflection",
    body: str | None = "Implements the selected approach.",
    labels: list[str] | None = None,
    components: list[str] | None = None,
    affected_paths: list[str] | None = None,
    updated_at: datetime = PR_UPDATED_AT,
    closed_at: datetime | None = PR_CLOSED_AT,
) -> KnowledgeItem:
    """Build one normalized pull-request outcome source."""

    return KnowledgeItem(
        repository=repository,
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id=external_id,
        source_kind=KnowledgeSourceKind.GITHUB,
        state=state,
        title=title,
        body=body,
        url=f"https://example.test/pulls/{external_id}",
        author=PR_AUTHOR,
        created_at=CREATED_AT,
        updated_at=updated_at,
        closed_at=closed_at,
        labels=[] if labels is None else labels,
        components=[] if components is None else components,
        affected_paths=[] if affected_paths is None else affected_paths,
    )


def extract_documents(
    *documents: KnowledgeItem,
    max_candidates: int = 100,
) -> KnowledgeDecisionCandidateExtractionResult:
    """Run the real Part 4.5A extractor."""

    return extract_knowledge_decision_candidates(
        REPOSITORY,
        list(documents),
        recorded_at=EXTRACTED_AT,
        options=KnowledgeDecisionCandidateExtractionOptions(
            max_candidates=max_candidates
        ),
    )


def make_link(
    extraction: KnowledgeDecisionCandidateExtractionResult,
    source: KnowledgeItem,
    *,
    candidate_index: int = 0,
    effect: KnowledgeDecisionOutcomeEffect = (
        KnowledgeDecisionOutcomeEffect.CONFIRMS
    ),
    explanation: str = "The pull-request outcome explicitly supports this resolution.",
) -> KnowledgeDecisionOutcomeLink:
    """Build one explicit link to an extracted candidate."""

    return KnowledgeDecisionOutcomeLink(
        decision_id=extraction.candidates[candidate_index].decision_id,
        source=source.to_reference(),
        effect=effect,
        explanation=explanation,
    )


def resolve(
    extraction: KnowledgeDecisionCandidateExtractionResult,
    items: list[KnowledgeItem],
    links: list[KnowledgeDecisionOutcomeLink] | None = None,
    *,
    recorded_at: datetime = RESOLVED_AT,
    max_outcome_links: int = 1_000,
) -> KnowledgeDecisionResolutionResult:
    """Resolve candidates with concise test defaults."""

    return resolve_knowledge_decision_candidates(
        extraction,
        items,
        [] if links is None else links,
        recorded_at=recorded_at,
        options=KnowledgeDecisionResolutionOptions(
            max_outcome_links=max_outcome_links
        ),
    )


def result_payload(
    result: KnowledgeDecisionResolutionResult,
) -> dict[str, object]:
    """Return declared result fields for validation mutations."""

    return {
        field_name: getattr(result, field_name)
        for field_name in KnowledgeDecisionResolutionResult.model_fields
    }


def malformed_extraction(
    candidates: list[Any],
) -> KnowledgeDecisionCandidateExtractionResult:
    """Construct a defensive-input fixture that bypasses 4.5A result checks."""

    count = len(candidates)
    return KnowledgeDecisionCandidateExtractionResult.model_construct(
        repository=REPOSITORY,
        recorded_at=EXTRACTED_AT,
        candidates=candidates,
        skipped_sources=[],
        stats=KnowledgeDecisionCandidateExtractionStats(
            source_items_seen=count,
            eligible_source_items=count,
            candidates_created=count,
            skipped_sources=0,
            candidate_limit_reached=False,
        ),
    )


def test_public_enums_and_default_options_are_exact() -> None:
    assert [effect.value for effect in KnowledgeDecisionOutcomeEffect] == [
        "confirms",
        "rejects",
    ]
    assert [reason.value for reason in KnowledgeDecisionResolutionReason] == [
        "documentary_source_established",
        "documentary_source_superseded",
        "explicit_outcome_confirmation",
        "explicit_outcome_rejection",
        "conflicting_outcome_evidence",
        "insufficient_evidence",
    ]
    assert KnowledgeDecisionResolutionOptions().max_outcome_links == 1_000
    assert MAX_KNOWLEDGE_DECISION_OUTCOME_LINKS == 2_000


def test_options_accept_bounds_and_reject_out_of_range() -> None:
    assert KnowledgeDecisionResolutionOptions(
        max_outcome_links=0
    ).max_outcome_links == 0
    assert KnowledgeDecisionResolutionOptions(
        max_outcome_links=MAX_KNOWLEDGE_DECISION_OUTCOME_LINKS
    ).max_outcome_links == MAX_KNOWLEDGE_DECISION_OUTCOME_LINKS

    for value in (-1, MAX_KNOWLEDGE_DECISION_OUTCOME_LINKS + 1):
        with pytest.raises(ValidationError):
            KnowledgeDecisionResolutionOptions(max_outcome_links=value)


def test_outcome_link_validates_source_text_and_computed_keys() -> None:
    extraction = extract_documents(make_document())
    pull_request = make_pull_request()
    link = make_link(
        extraction,
        pull_request,
        explanation="  Caller supplied explanation.  ",
    )

    assert link.explanation == "Caller supplied explanation."
    assert link.item_key == link.source.key
    assert link.key == f"{link.decision_id}:{link.source.key}"
    assert link.effect.value not in link.key

    for updates in (
        {"decision_id": " "},
        {"explanation": " "},
    ):
        payload = {
            "decision_id": extraction.candidates[0].decision_id,
            "source": pull_request.to_reference(),
            "effect": KnowledgeDecisionOutcomeEffect.CONFIRMS,
            "explanation": "Explanation",
            **updates,
        }
        with pytest.raises(ValidationError):
            KnowledgeDecisionOutcomeLink.model_validate(payload)

    with pytest.raises(ValidationError, match="pull-request"):
        KnowledgeDecisionOutcomeLink(
            decision_id=extraction.candidates[0].decision_id,
            source=make_document().to_reference(),
            effect=KnowledgeDecisionOutcomeEffect.CONFIRMS,
            explanation="Explanation",
        )


@pytest.mark.parametrize(
    ("reason", "status"),
    [
        (
            KnowledgeDecisionResolutionReason.DOCUMENTARY_SOURCE_ESTABLISHED,
            DecisionStatus.CONFIRMED,
        ),
        (
            KnowledgeDecisionResolutionReason.DOCUMENTARY_SOURCE_SUPERSEDED,
            DecisionStatus.SUPERSEDED,
        ),
        (
            KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_CONFIRMATION,
            DecisionStatus.CONFIRMED,
        ),
        (
            KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_REJECTION,
            DecisionStatus.REJECTED,
        ),
        (
            KnowledgeDecisionResolutionReason.CONFLICTING_OUTCOME_EVIDENCE,
            DecisionStatus.CANDIDATE,
        ),
        (
            KnowledgeDecisionResolutionReason.INSUFFICIENT_EVIDENCE,
            DecisionStatus.CANDIDATE,
        ),
    ],
)
def test_resolution_accepts_every_reason_status_pair(
    reason: KnowledgeDecisionResolutionReason,
    status: DecisionStatus,
) -> None:
    resolution = KnowledgeDecisionResolution(
        decision_id="decision-1",
        candidate_key="repository:decision:decision-1",
        status=status,
        reason=reason,
        outcome_source_keys=["pull:1", "pull:2"],
    )

    assert resolution.outcome_count == 2


def test_resolution_rejects_status_reason_and_source_key_errors() -> None:
    base = {
        "decision_id": "decision-1",
        "candidate_key": "repository:decision:decision-1",
        "status": DecisionStatus.CANDIDATE,
        "reason": KnowledgeDecisionResolutionReason.INSUFFICIENT_EVIDENCE,
        "outcome_source_keys": [],
    }
    with pytest.raises(ValidationError, match="not supported"):
        KnowledgeDecisionResolution.model_validate(
            {**base, "status": DecisionStatus.EXCEPTION}
        )
    with pytest.raises(ValidationError, match="match"):
        KnowledgeDecisionResolution.model_validate(
            {**base, "status": DecisionStatus.CONFIRMED}
        )
    with pytest.raises(ValidationError, match="unique"):
        KnowledgeDecisionResolution.model_validate(
            {**base, "outcome_source_keys": ["pull:1", "pull:1"]}
        )
    with pytest.raises(ValidationError, match="ascending"):
        KnowledgeDecisionResolution.model_validate(
            {**base, "outcome_source_keys": ["pull:2", "pull:1"]}
        )
    with pytest.raises(ValidationError, match="non-empty"):
        KnowledgeDecisionResolution.model_validate(
            {**base, "outcome_source_keys": [" "]}
        )


def test_stats_validate_evidence_status_and_conflict_accounting() -> None:
    valid = {
        "candidates_seen": 1,
        "outcome_links_seen": 1,
        "outcome_evidence_added": 1,
        "confirmed_decisions": 1,
        "rejected_decisions": 0,
        "superseded_decisions": 0,
        "remaining_candidates": 0,
        "conflicting_decisions": 0,
    }
    for updates in (
        {"outcome_evidence_added": 0},
        {"confirmed_decisions": 0},
        {"confirmed_decisions": 0, "remaining_candidates": 1, "conflicting_decisions": 2},
    ):
        with pytest.raises(ValidationError):
            KnowledgeDecisionResolutionStats.model_validate(
                {**valid, **updates}
            )


def test_empty_resolution_normalizes_time_and_rejects_extras() -> None:
    extraction = extract_documents()
    result = resolve(
        extraction,
        [],
        recorded_at=NON_UTC_RESOLVED_AT,
    )

    assert result.recorded_at == RESOLVED_AT
    assert result.decisions == []
    assert result.resolutions == []
    assert result.outcome_links == []
    assert result.stats.candidates_seen == 0
    assert result.complete is True

    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_knowledge_decision_candidates(
            extraction,
            [],
            [],
            recorded_at=datetime(2026, 3, 1, 15, 0),
        )

    payload = result_payload(result)
    payload["provider"] = "github"
    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeDecisionResolutionResult.model_validate(payload)


def test_outcome_link_limit_is_enforced_before_other_inputs() -> None:
    document = make_document()
    extraction = extract_documents(document)
    pull_request = make_pull_request()
    link = make_link(extraction, pull_request)

    with pytest.raises(
        KnowledgeDecisionResolutionError,
        match="configured safety limit",
    ):
        resolve_knowledge_decision_candidates(
            extraction,
            [make_document(repository=OTHER_REPOSITORY)],
            [link],
            recorded_at=RESOLVED_AT,
            options=KnowledgeDecisionResolutionOptions(max_outcome_links=0),
        )


def test_cross_repository_and_duplicate_items_are_rejected() -> None:
    document = make_document()
    extraction = extract_documents(document)
    foreign = make_pull_request(repository=OTHER_REPOSITORY)

    with pytest.raises(KnowledgeDecisionResolutionError, match="repository"):
        resolve(extraction, [document, foreign])
    with pytest.raises(KnowledgeDecisionResolutionError, match="keys must be unique"):
        resolve(extraction, [document, document])


def test_duplicate_candidate_ids_are_rejected_defensively() -> None:
    document = make_document()
    candidate = extract_documents(document).candidates[0]
    extraction = malformed_extraction([candidate, candidate])

    with pytest.raises(
        KnowledgeDecisionResolutionError,
        match="decision IDs must be unique",
    ):
        resolve(extraction, [document])


def test_candidate_shape_is_validated_before_source_lookup() -> None:
    document = make_document(state=KnowledgeItemState.ACTIVE)
    candidate = extract_documents(document).candidates[0]
    non_candidate = candidate.model_copy(
        update={"status": DecisionStatus.CONFIRMED}
    )
    extraction = malformed_extraction([non_candidate])

    with pytest.raises(KnowledgeDecisionResolutionError, match="only CANDIDATE"):
        resolve(extraction, [])

    second_evidence = candidate.evidence[0].model_copy(
        update={"source_anchor": "second"}
    )
    extraction = malformed_extraction(
        [
            candidate.model_copy(
                update={"evidence": [candidate.evidence[0], second_evidence]}
            )
        ]
    )
    with pytest.raises(KnowledgeDecisionResolutionError, match="exactly one"):
        resolve(extraction, [document])

    issue = make_document(item_type=KnowledgeItemType.ISSUE)
    unsupported_evidence = candidate.evidence[0].model_copy(
        update={"source": issue.to_reference()}
    )
    extraction = malformed_extraction(
        [candidate.model_copy(update={"evidence": [unsupported_evidence]})]
    )
    with pytest.raises(KnowledgeDecisionResolutionError, match="documentary"):
        resolve(extraction, [issue])


def test_documentary_source_must_exist_and_match_exact_reference() -> None:
    document = make_document()
    extraction = extract_documents(document)

    with pytest.raises(KnowledgeDecisionResolutionError, match="unavailable"):
        resolve(extraction, [])

    changed = make_document(title="Changed authoritative title")
    with pytest.raises(KnowledgeDecisionResolutionError, match="does not match"):
        resolve(extraction, [changed])


def test_duplicate_unknown_and_skipped_candidate_links_are_rejected() -> None:
    document = make_document()
    pull_request = make_pull_request()
    extraction = extract_documents(document)
    first = make_link(extraction, pull_request)
    duplicate_effect = make_link(
        extraction,
        pull_request,
        effect=KnowledgeDecisionOutcomeEffect.REJECTS,
    )

    with pytest.raises(KnowledgeDecisionResolutionError, match="keys must be unique"):
        resolve(
            extraction,
            [document, pull_request],
            [first, duplicate_effect],
        )

    unknown = first.model_copy(update={"decision_id": "unknown-decision"})
    with pytest.raises(KnowledgeDecisionResolutionError, match="unknown candidate"):
        resolve(extraction, [document, pull_request], [unknown])

    limited = extract_documents(
        make_document(external_id="adr-1"),
        make_document(external_id="adr-2"),
        max_candidates=1,
    )
    skipped_target = first.model_copy(
        update={"decision_id": "candidate-for-skipped-source"}
    )
    with pytest.raises(KnowledgeDecisionResolutionError, match="unknown candidate"):
        resolve(
            limited,
            [
                make_document(external_id="adr-1"),
                make_document(external_id="adr-2"),
                pull_request,
            ],
            [skipped_target],
        )


def test_outcome_source_must_exist_and_match_exact_reference() -> None:
    document = make_document()
    pull_request = make_pull_request()
    extraction = extract_documents(document)
    link = make_link(extraction, pull_request)

    with pytest.raises(KnowledgeDecisionResolutionError, match="unavailable"):
        resolve(extraction, [document], [link])

    changed = make_pull_request(title="Changed authoritative title")
    with pytest.raises(KnowledgeDecisionResolutionError, match="does not match"):
        resolve(extraction, [document, changed], [link])


@pytest.mark.parametrize(
    "state",
    [KnowledgeItemState.OPEN, KnowledgeItemState.CLOSED],
)
def test_incomplete_pull_request_states_are_rejected(
    state: KnowledgeItemState,
) -> None:
    document = make_document()
    pull_request = make_pull_request(state=state)
    extraction = extract_documents(document)
    link = make_link(extraction, pull_request)

    with pytest.raises(
        KnowledgeDecisionResolutionError,
        match="merged or rejected",
    ):
        resolve(extraction, [document, pull_request], [link])


def test_malformed_non_pr_outcome_source_is_rejected() -> None:
    document = make_document()
    issue = make_document(item_type=KnowledgeItemType.ISSUE)
    extraction = extract_documents(document)
    malformed_link = KnowledgeDecisionOutcomeLink.model_construct(
        decision_id=extraction.candidates[0].decision_id,
        source=issue.to_reference(),
        effect=KnowledgeDecisionOutcomeEffect.CONFIRMS,
        explanation="Explicit assertion",
    )

    with pytest.raises(
        KnowledgeDecisionResolutionError,
        match="pull request",
    ):
        resolve(extraction, [document, issue], [malformed_link])


@pytest.mark.parametrize(
    "state",
    [KnowledgeItemState.MERGED, KnowledgeItemState.REJECTED],
)
def test_completed_pull_request_states_are_accepted(
    state: KnowledgeItemState,
) -> None:
    document = make_document()
    pull_request = make_pull_request(state=state)
    extraction = extract_documents(document)

    result = resolve(
        extraction,
        [document, pull_request],
        [make_link(extraction, pull_request)],
    )

    assert result.confirmed_count == 1


def test_naive_time_is_rejected_before_resolution_consistency_checks() -> None:
    document = make_document(repository=OTHER_REPOSITORY)
    extraction = extract_documents()

    with pytest.raises(ValueError, match="timezone-aware"):
        resolve_knowledge_decision_candidates(
            extraction,
            [document],
            [],
            recorded_at=datetime(2026, 3, 1, 15, 0),
        )


def test_resolution_does_not_mutate_any_inputs() -> None:
    document = make_document()
    pull_request = make_pull_request()
    extraction = extract_documents(document)
    link = make_link(extraction, pull_request)
    items = [pull_request, document]
    links = [link]
    extraction_before = extraction.model_dump()
    candidate_before = extraction.candidates[0].model_dump()
    item_payloads = [item.model_dump() for item in items]
    link_payloads = [item.model_dump() for item in links]

    resolve(extraction, items, links)

    assert extraction.model_dump() == extraction_before
    assert extraction.candidates[0].model_dump() == candidate_before
    assert [item.model_dump() for item in items] == item_payloads
    assert [item.model_dump() for item in links] == link_payloads


def test_unlinked_pr_never_creates_evidence_from_similarity() -> None:
    document = make_document(
        external_id="same-id",
        title="Shared title",
        labels=["shared-label"],
        components=["Shared"],
        affected_paths=["src/shared.py"],
    )
    pull_request = make_pull_request(
        external_id="same-id",
        title="Shared title",
        labels=["shared-label"],
        components=["Shared"],
        affected_paths=["src/shared.py"],
    )
    extraction = extract_documents(document)

    result = resolve(extraction, [document, pull_request])

    assert result.outcome_links == []
    assert len(result.decisions[0].evidence) == 1
    assert result.decisions[0].status == DecisionStatus.CANDIDATE


def test_only_explicit_target_receives_evidence() -> None:
    first = make_document(external_id="adr-1")
    second = make_document(external_id="adr-2")
    pull_request = make_pull_request()
    extraction = extract_documents(first, second)
    target_index = next(
        index
        for index, candidate in enumerate(extraction.candidates)
        if candidate.evidence[0].source.key == second.key
    )
    link = make_link(
        extraction,
        pull_request,
        candidate_index=target_index,
    )

    result = resolve(extraction, [first, second, pull_request], [link])
    by_id = {decision.decision_id: decision for decision in result.decisions}

    assert len(by_id[link.decision_id].evidence) == 2
    assert sum(len(decision.evidence) == 1 for decision in result.decisions) == 1


def test_one_pr_may_link_to_multiple_decisions() -> None:
    first = make_document(external_id="adr-1")
    second = make_document(external_id="adr-2")
    pull_request = make_pull_request()
    extraction = extract_documents(first, second)
    links = [
        make_link(extraction, pull_request, candidate_index=index)
        for index in range(2)
    ]

    result = resolve(extraction, [first, second, pull_request], links)

    assert result.stats.outcome_links_seen == 2
    assert all(len(decision.evidence) == 2 for decision in result.decisions)


def test_one_decision_may_receive_multiple_distinct_pr_links() -> None:
    document = make_document()
    first = make_pull_request(external_id="101")
    second = make_pull_request(
        external_id="102",
        state=KnowledgeItemState.REJECTED,
    )
    extraction = extract_documents(document)
    links = [
        make_link(extraction, first),
        make_link(extraction, second),
    ]

    result = resolve(extraction, [document, first, second], links)

    assert result.resolutions[0].outcome_count == 2
    assert len(result.decisions[0].evidence) == 3


@pytest.mark.parametrize(
    ("state", "expected_kind"),
    [
        (KnowledgeItemState.MERGED, DecisionEvidenceKind.MERGE_OUTCOME),
        (KnowledgeItemState.REJECTED, DecisionEvidenceKind.REJECTION_OUTCOME),
    ],
)
def test_outcome_evidence_shape(
    state: KnowledgeItemState,
    expected_kind: DecisionEvidenceKind,
) -> None:
    document = make_document()
    pull_request = make_pull_request(state=state)
    extraction = extract_documents(document)
    explanation = "Caller says this exact outcome confirms the decision."
    link = make_link(
        extraction,
        pull_request,
        explanation=explanation,
    )

    result = resolve(extraction, [document, pull_request], [link])
    evidence = result.decisions[0].evidence[1]

    assert evidence.kind == expected_kind
    assert evidence.relationship == DecisionEvidenceRelationship.SUPPORTS
    assert evidence.strength == DecisionEvidenceStrength.STRONG
    assert evidence.summary == explanation
    assert evidence.excerpt is None
    assert evidence.source_anchor is None
    assert evidence.author == pull_request.author
    assert evidence.occurred_at == pull_request.closed_at
    assert evidence.source == pull_request.to_reference()


def test_outcome_evidence_time_falls_back_to_updated_at() -> None:
    document = make_document()
    pull_request = make_pull_request(closed_at=None)
    extraction = extract_documents(document)

    result = resolve(
        extraction,
        [document, pull_request],
        [make_link(extraction, pull_request)],
    )

    assert result.decisions[0].evidence[1].occurred_at == pull_request.updated_at


def test_documentary_evidence_stays_first_and_outcomes_sort_by_source_key() -> None:
    document = make_document()
    later = make_pull_request(external_id="200")
    earlier = make_pull_request(external_id="100")
    extraction = extract_documents(document)
    links = [
        make_link(extraction, later),
        make_link(extraction, earlier),
    ]

    result = resolve(extraction, [later, document, earlier], links)
    evidence = result.decisions[0].evidence

    assert evidence[0] == extraction.candidates[0].evidence[0]
    assert [item.source.key for item in evidence[1:]] == sorted(
        [earlier.key, later.key]
    )


@pytest.mark.parametrize(
    "effect",
    [
        KnowledgeDecisionOutcomeEffect.CONFIRMS,
        KnowledgeDecisionOutcomeEffect.REJECTS,
    ],
)
def test_superseded_documentary_source_has_highest_priority(
    effect: KnowledgeDecisionOutcomeEffect,
) -> None:
    document = make_document(state=KnowledgeItemState.SUPERSEDED)
    pull_request = make_pull_request()
    extraction = extract_documents(document)
    link = make_link(extraction, pull_request, effect=effect)

    result = resolve(extraction, [document, pull_request], [link])

    assert result.decisions[0].status == DecisionStatus.SUPERSEDED
    assert result.resolutions[0].reason == (
        KnowledgeDecisionResolutionReason.DOCUMENTARY_SOURCE_SUPERSEDED
    )
    assert len(result.decisions[0].evidence) == 2


def test_conflicting_effects_remain_candidate_without_timestamp_or_state_preference() -> None:
    document = make_document()
    merged_old = make_pull_request(
        external_id="101",
        state=KnowledgeItemState.MERGED,
        updated_at=PR_UPDATED_AT,
        closed_at=PR_CLOSED_AT,
    )
    rejected_new = make_pull_request(
        external_id="102",
        state=KnowledgeItemState.REJECTED,
        updated_at=PR_UPDATED_AT + timedelta(days=30),
        closed_at=PR_CLOSED_AT + timedelta(days=30),
    )
    extraction = extract_documents(document)
    links = [
        make_link(
            extraction,
            merged_old,
            effect=KnowledgeDecisionOutcomeEffect.CONFIRMS,
        ),
        make_link(
            extraction,
            rejected_new,
            effect=KnowledgeDecisionOutcomeEffect.REJECTS,
        ),
    ]

    result = resolve(extraction, [document, merged_old, rejected_new], links)

    assert result.decisions[0].status == DecisionStatus.CANDIDATE
    assert result.resolutions[0].reason == (
        KnowledgeDecisionResolutionReason.CONFLICTING_OUTCOME_EVIDENCE
    )
    assert result.stats.conflicting_decisions == 1
    assert result.has_conflicts is True


@pytest.mark.parametrize(
    ("effect", "expected_status", "expected_reason"),
    [
        (
            KnowledgeDecisionOutcomeEffect.REJECTS,
            DecisionStatus.REJECTED,
            KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_REJECTION,
        ),
        (
            KnowledgeDecisionOutcomeEffect.CONFIRMS,
            DecisionStatus.CONFIRMED,
            KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_CONFIRMATION,
        ),
    ],
)
def test_uniform_explicit_effect_controls_resolution(
    effect: KnowledgeDecisionOutcomeEffect,
    expected_status: DecisionStatus,
    expected_reason: KnowledgeDecisionResolutionReason,
) -> None:
    document = make_document()
    merged = make_pull_request(external_id="101")
    rejected = make_pull_request(
        external_id="102",
        state=KnowledgeItemState.REJECTED,
    )
    extraction = extract_documents(document)
    links = [
        make_link(extraction, merged, effect=effect),
        make_link(extraction, rejected, effect=effect),
    ]

    result = resolve(extraction, [document, merged, rejected], links)

    assert result.decisions[0].status == expected_status
    assert result.resolutions[0].reason == expected_reason


@pytest.mark.parametrize(
    "state",
    [KnowledgeItemState.ACTIVE, KnowledgeItemState.PUBLISHED],
)
def test_established_documentary_source_confirms_without_links(
    state: KnowledgeItemState,
) -> None:
    document = make_document(state=state)
    extraction = extract_documents(document)

    result = resolve(extraction, [document])

    assert result.decisions[0].status == DecisionStatus.CONFIRMED
    assert result.resolutions[0].reason == (
        KnowledgeDecisionResolutionReason.DOCUMENTARY_SOURCE_ESTABLISHED
    )


def test_established_source_with_weak_original_evidence_raises() -> None:
    document = make_document(state=KnowledgeItemState.ACTIVE)
    candidate = extract_documents(document).candidates[0]
    weak_evidence = candidate.evidence[0].model_copy(
        update={"strength": DecisionEvidenceStrength.MODERATE}
    )
    extraction = malformed_extraction(
        [candidate.model_copy(update={"evidence": [weak_evidence]})]
    )

    with pytest.raises(
        KnowledgeDecisionResolutionError,
        match="strong or authoritative",
    ):
        resolve(extraction, [document])


@pytest.mark.parametrize(
    "state",
    [
        KnowledgeItemState.UNKNOWN,
        KnowledgeItemState.DRAFT,
        KnowledgeItemState.ARCHIVED,
    ],
)
def test_non_established_source_without_links_remains_candidate(
    state: KnowledgeItemState,
) -> None:
    document = make_document(state=state)
    extraction = extract_documents(document)

    result = resolve(extraction, [document])

    assert result.decisions[0].status == DecisionStatus.CANDIDATE
    assert result.resolutions[0].reason == (
        KnowledgeDecisionResolutionReason.INSUFFICIENT_EVIDENCE
    )
    assert result.decisions[0].status not in {
        DecisionStatus.EXCEPTION,
        DecisionStatus.UNKNOWN,
    }


def test_rejected_and_merged_pr_effects_are_never_inferred() -> None:
    negative = make_document(title="Do not add reflection")
    rejected = make_pull_request(state=KnowledgeItemState.REJECTED)
    extraction = extract_documents(negative)
    confirms = make_link(
        extraction,
        rejected,
        effect=KnowledgeDecisionOutcomeEffect.CONFIRMS,
    )
    confirmed = resolve(extraction, [negative, rejected], [confirms])
    assert confirmed.decisions[0].status == DecisionStatus.CONFIRMED

    positive = make_document(title="Add reflection")
    positive_extraction = extract_documents(positive)
    rejects = make_link(
        positive_extraction,
        rejected,
        effect=KnowledgeDecisionOutcomeEffect.REJECTS,
    )
    rejected_result = resolve(
        positive_extraction,
        [positive, rejected],
        [rejects],
    )
    assert rejected_result.decisions[0].status == DecisionStatus.REJECTED

    merged = make_pull_request(external_id="102")
    merged_rejects = make_link(
        positive_extraction,
        merged,
        effect=KnowledgeDecisionOutcomeEffect.REJECTS,
    )
    merged_result = resolve(
        positive_extraction,
        [positive, merged],
        [merged_rejects],
    )
    assert merged_result.decisions[0].status == DecisionStatus.REJECTED


def test_decision_reconstruction_preserves_identity_and_explicit_fields() -> None:
    document = make_document(
        title="Preserve exact statement",
        summary="Preserve rationale",
        significance=DecisionSignificance.CRITICAL,
        affected_paths=["src/a.py", "src/b.py"],
        components=["API", "Runtime"],
    )
    pull_request = make_pull_request()
    extraction = extract_documents(document)
    candidate = extraction.candidates[0]

    result = resolve(
        extraction,
        [document, pull_request],
        [make_link(extraction, pull_request)],
        recorded_at=NON_UTC_RESOLVED_AT,
    )
    decision = result.decisions[0]

    assert decision.decision_id == candidate.decision_id
    assert decision.key == candidate.key
    assert decision.statement == candidate.statement
    assert decision.significance == candidate.significance
    assert decision.rationale == candidate.rationale
    assert decision.affected_paths == candidate.affected_paths
    assert decision.components == candidate.components
    assert decision.status == DecisionStatus.CONFIRMED
    assert decision.recorded_at == RESOLVED_AT
    assert decision.recorded_at != candidate.recorded_at
    assert decision.evidence is not candidate.evidence
    assert decision.evidence[0] is not candidate.evidence[0]


def test_output_order_is_independent_of_every_input_order() -> None:
    documents = [
        make_document(external_id="adr-3"),
        make_document(external_id="adr-1"),
        make_document(external_id="adr-2"),
    ]
    pull_requests = [
        make_pull_request(external_id="103"),
        make_pull_request(external_id="101"),
        make_pull_request(external_id="102"),
    ]
    extraction = extract_documents(*documents)
    links = [
        make_link(
            extraction,
            pull_request,
            candidate_index=index,
        )
        for index, pull_request in enumerate(pull_requests)
    ]

    first = resolve(
        extraction,
        [*documents, *pull_requests],
        links,
    )
    reversed_extraction = extraction.model_copy(
        update={"candidates": list(reversed(extraction.candidates))}
    )
    second = resolve(
        reversed_extraction,
        [*reversed(pull_requests), *reversed(documents)],
        list(reversed(links)),
    )

    assert first == second
    assert [decision.key for decision in first.decisions] == sorted(
        decision.key for decision in first.decisions
    )
    assert [resolution.decision_id for resolution in first.resolutions] == sorted(
        resolution.decision_id for resolution in first.resolutions
    )
    assert first.outcome_links == sorted(
        first.outcome_links,
        key=lambda link: (link.decision_id, link.source.key),
    )
    assert all(
        resolution.outcome_source_keys
        == sorted(resolution.outcome_source_keys)
        for resolution in first.resolutions
    )


def test_result_requires_exact_resolution_identity_and_coverage() -> None:
    document = make_document(state=KnowledgeItemState.ACTIVE)
    result = resolve(extract_documents(document), [document])
    resolution = result.resolutions[0]

    payload = result_payload(result)
    payload["resolutions"] = []
    with pytest.raises(ValidationError, match="Exactly one"):
        KnowledgeDecisionResolutionResult.model_validate(payload)

    payload = result_payload(result)
    payload["resolutions"] = [
        resolution.model_copy(update={"decision_id": "unknown"})
    ]
    with pytest.raises(ValidationError, match="Exactly one"):
        KnowledgeDecisionResolutionResult.model_validate(payload)

    payload = result_payload(result)
    payload["resolutions"] = [
        resolution.model_copy(
            update={
                "status": DecisionStatus.REJECTED,
                "reason": (
                    KnowledgeDecisionResolutionReason.EXPLICIT_OUTCOME_REJECTION
                ),
            }
        )
    ]
    with pytest.raises(ValidationError, match="status"):
        KnowledgeDecisionResolutionResult.model_validate(payload)

    payload = result_payload(result)
    payload["resolutions"] = [
        resolution.model_copy(update={"candidate_key": "wrong-key"})
    ]
    with pytest.raises(ValidationError, match="candidate_key"):
        KnowledgeDecisionResolutionResult.model_validate(payload)


def test_result_requires_exact_outcome_link_identity_and_source_coverage() -> None:
    document = make_document()
    pull_request = make_pull_request()
    extraction = extract_documents(document)
    result = resolve(
        extraction,
        [document, pull_request],
        [make_link(extraction, pull_request)],
    )

    payload = result_payload(result)
    payload["outcome_links"] = [
        result.outcome_links[0].model_copy(
            update={"decision_id": "unknown"}
        )
    ]
    with pytest.raises(ValidationError, match="identify"):
        KnowledgeDecisionResolutionResult.model_validate(payload)

    payload = result_payload(result)
    payload["resolutions"] = [
        result.resolutions[0].model_copy(update={"outcome_source_keys": []})
    ]
    with pytest.raises(ValidationError, match="associated links"):
        KnowledgeDecisionResolutionResult.model_validate(payload)


def test_result_rejects_duplicate_decision_and_link_identities() -> None:
    document = make_document(state=KnowledgeItemState.ACTIVE)
    result = resolve(extract_documents(document), [document])
    decision = result.decisions[0]
    resolution = result.resolutions[0]
    duplicate_stats = KnowledgeDecisionResolutionStats(
        candidates_seen=2,
        outcome_links_seen=0,
        outcome_evidence_added=0,
        confirmed_decisions=2,
        rejected_decisions=0,
        superseded_decisions=0,
        remaining_candidates=0,
        conflicting_decisions=0,
    )

    payload = result_payload(result)
    payload["decisions"] = [decision, decision]
    payload["resolutions"] = [resolution, resolution]
    payload["stats"] = duplicate_stats
    with pytest.raises(ValidationError, match="unique"):
        KnowledgeDecisionResolutionResult.model_validate(payload)

    pull_request = make_pull_request()
    extraction = extract_documents(make_document())
    linked = resolve(
        extraction,
        [make_document(), pull_request],
        [make_link(extraction, pull_request)],
    )
    payload = result_payload(linked)
    payload["outcome_links"] = [
        linked.outcome_links[0],
        linked.outcome_links[0],
    ]
    payload["stats"] = KnowledgeDecisionResolutionStats(
        candidates_seen=1,
        outcome_links_seen=2,
        outcome_evidence_added=2,
        confirmed_decisions=1,
        rejected_decisions=0,
        superseded_decisions=0,
        remaining_candidates=0,
        conflicting_decisions=0,
    )
    with pytest.raises(ValidationError, match="unique"):
        KnowledgeDecisionResolutionResult.model_validate(payload)


def test_result_validates_repository_time_evidence_and_order() -> None:
    first = make_document(
        external_id="adr-1",
        state=KnowledgeItemState.ACTIVE,
    )
    second = make_document(
        external_id="adr-2",
        state=KnowledgeItemState.ACTIVE,
    )
    result = resolve(extract_documents(first, second), [first, second])

    payload = result_payload(result)
    payload["decisions"] = list(reversed(result.decisions))
    with pytest.raises(ValidationError, match="ascending"):
        KnowledgeDecisionResolutionResult.model_validate(payload)

    foreign_decision = result.decisions[0].model_copy(
        update={"repository": OTHER_REPOSITORY}
    )
    payload = result_payload(result)
    payload["decisions"] = [foreign_decision, result.decisions[1]]
    with pytest.raises(ValidationError):
        KnowledgeDecisionResolutionResult.model_validate(payload)

    wrong_time = result.decisions[0].model_copy(
        update={"recorded_at": RESOLVED_AT + timedelta(seconds=1)}
    )
    payload = result_payload(result)
    payload["decisions"] = [wrong_time, result.decisions[1]]
    with pytest.raises(ValidationError, match="recorded_at"):
        KnowledgeDecisionResolutionResult.model_validate(payload)


def test_result_stats_match_statuses_links_and_conflicts() -> None:
    document = make_document(state=KnowledgeItemState.ACTIVE)
    result = resolve(extract_documents(document), [document])
    payload = result_payload(result)
    payload["stats"] = KnowledgeDecisionResolutionStats(
        candidates_seen=1,
        outcome_links_seen=0,
        outcome_evidence_added=0,
        confirmed_decisions=0,
        rejected_decisions=0,
        superseded_decisions=0,
        remaining_candidates=1,
        conflicting_decisions=0,
    )
    with pytest.raises(ValidationError, match="status counts"):
        KnowledgeDecisionResolutionResult.model_validate(payload)

    unknown = make_document()
    first = make_pull_request(external_id="101")
    second = make_pull_request(
        external_id="102",
        state=KnowledgeItemState.REJECTED,
    )
    extraction = extract_documents(unknown)
    conflict = resolve(
        extraction,
        [unknown, first, second],
        [
            make_link(
                extraction,
                first,
                effect=KnowledgeDecisionOutcomeEffect.CONFIRMS,
            ),
            make_link(
                extraction,
                second,
                effect=KnowledgeDecisionOutcomeEffect.REJECTS,
            ),
        ],
    )
    payload = result_payload(conflict)
    payload["stats"] = conflict.stats.model_copy(
        update={"conflicting_decisions": 0}
    )
    with pytest.raises(ValidationError, match="Conflict count"):
        KnowledgeDecisionResolutionResult.model_validate(payload)


def test_computed_counts_and_completeness_cover_all_statuses() -> None:
    documents = [
        make_document(
            external_id="active",
            state=KnowledgeItemState.ACTIVE,
        ),
        make_document(
            external_id="rejected",
            state=KnowledgeItemState.UNKNOWN,
        ),
        make_document(
            external_id="superseded",
            state=KnowledgeItemState.SUPERSEDED,
        ),
        make_document(
            external_id="candidate",
            state=KnowledgeItemState.DRAFT,
        ),
    ]
    pull_request = make_pull_request()
    extraction = extract_documents(*documents)
    rejected_index = next(
        index
        for index, candidate in enumerate(extraction.candidates)
        if candidate.evidence[0].source.external_id == "rejected"
    )
    link = make_link(
        extraction,
        pull_request,
        candidate_index=rejected_index,
        effect=KnowledgeDecisionOutcomeEffect.REJECTS,
    )

    result = resolve(extraction, [*documents, pull_request], [link])

    assert result.decision_count == 4
    assert result.confirmed_count == 1
    assert result.rejected_count == 1
    assert result.superseded_count == 1
    assert result.candidate_count == 1
    assert result.has_conflicts is False
    assert result.complete is False
    assert result.stats.remaining_candidates == 1


def test_source_extraction_incompleteness_is_preserved() -> None:
    first = make_document(
        external_id="adr-1",
        state=KnowledgeItemState.ACTIVE,
    )
    second = make_document(
        external_id="adr-2",
        state=KnowledgeItemState.ACTIVE,
    )
    extraction = extract_documents(first, second, max_candidates=1)

    result = resolve(extraction, [first, second])

    assert extraction.complete is False
    assert result.source_extraction_complete is False
    assert result.stats.remaining_candidates == 0
    assert result.complete is False


def test_complete_true_requires_complete_extraction_and_no_candidates() -> None:
    established = make_document(state=KnowledgeItemState.ACTIVE)
    established_result = resolve(
        extract_documents(established),
        [established],
    )
    unresolved = make_document(state=KnowledgeItemState.UNKNOWN)
    unresolved_result = resolve(
        extract_documents(unresolved),
        [unresolved],
    )

    assert established_result.source_extraction_complete is True
    assert established_result.complete is True
    assert unresolved_result.source_extraction_complete is True
    assert unresolved_result.complete is False


def test_json_serialization_includes_evidence_effects_reasons_and_counts() -> None:
    document = make_document()
    pull_request = make_pull_request(state=KnowledgeItemState.REJECTED)
    extraction = extract_documents(document)
    result = resolve(
        extraction,
        [document, pull_request],
        [
            make_link(
                extraction,
                pull_request,
                effect=KnowledgeDecisionOutcomeEffect.REJECTS,
                explanation="The outcome explicitly rejects this candidate.",
            )
        ],
        recorded_at=NON_UTC_RESOLVED_AT,
    )
    payload = result.model_dump(mode="json")

    assert payload["recorded_at"] == "2026-03-01T15:00:00Z"
    assert payload["outcome_links"][0]["effect"] == "rejects"
    assert payload["resolutions"][0]["reason"] == "explicit_outcome_rejection"
    assert payload["resolutions"][0]["status"] == "rejected"
    assert payload["decisions"][0]["status"] == "rejected"
    assert payload["decisions"][0]["evidence"][1]["kind"] == "rejection_outcome"
    assert payload["decisions"][0]["evidence"][1]["strength"] == "strong"
    assert payload["stats"]["rejected_decisions"] == 1
    assert payload["decision_count"] == 1
    assert payload["rejected_count"] == 1
    assert payload["has_conflicts"] is False
    assert payload["complete"] is True


def test_existing_decision_record_validation_remains_active() -> None:
    document = make_document()
    pull_request = make_pull_request()
    extraction = extract_documents(document)
    result = resolve(
        extraction,
        [document, pull_request],
        [make_link(extraction, pull_request)],
    )

    assert result.decisions[0].status == DecisionStatus.CONFIRMED
    assert any(
        evidence.strength == DecisionEvidenceStrength.STRONG
        for evidence in result.decisions[0].evidence
    )
