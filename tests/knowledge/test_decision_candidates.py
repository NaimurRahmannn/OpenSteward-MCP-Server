"""Tests for conservative documentary decision-candidate extraction."""

import hashlib
import re
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.knowledge import (
    MAX_KNOWLEDGE_DECISION_CANDIDATES,
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
    KnowledgeDecisionCandidateRule,
    KnowledgeDecisionCandidateSkippedSource,
    KnowledgeDecisionCandidateSkipReason,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
    extract_knowledge_decision_candidates,
)

CREATED_AT = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
UPDATED_AT = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
RECORDED_AT = datetime(2026, 2, 1, 15, 0, tzinfo=UTC)
NON_UTC_RECORDED_AT = datetime(
    2026,
    2,
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

ADR_ACCEPTED_SUMMARY = (
    "Repository ADR provides formal documentary evidence for this decision candidate."
)
ADR_DOCUMENTARY_SUMMARY = (
    "Repository ADR provides documentary evidence for this decision candidate."
)
MAINTAINER_SUMMARY = (
    "Maintainer decision item provides explicit decision evidence."
)
RELEASE_SUMMARY = (
    "Published release material provides commitment evidence for this decision candidate."
)
UNSUPPORTED_EXPLANATION = (
    "This knowledge-item type is not an explicit documentary decision source."
)
INELIGIBLE_RELEASE_EXPLANATION = (
    "Release-note decision candidates require a published or active source state."
)
CANDIDATE_LIMIT_EXPLANATION = (
    "Decision candidate was omitted by the configured candidate safety limit."
)


def make_item(
    *,
    external_id: str = "1",
    repository: KnowledgeRepositoryRef = REPOSITORY,
    item_type: KnowledgeItemType = KnowledgeItemType.ADR,
    state: KnowledgeItemState = KnowledgeItemState.UNKNOWN,
    title: str = "Adopt typed repository configuration",
    body: str | None = "The repository uses typed configuration.",
    summary: str | None = "Typed configuration reduces ambiguity.",
    significance: DecisionSignificance = DecisionSignificance.HIGH,
    affected_paths: list[str] | None = None,
    components: list[str] | None = None,
    author: KnowledgeActor | None = AUTHOR,
    created_at: datetime = CREATED_AT,
    updated_at: datetime = UPDATED_AT,
) -> KnowledgeItem:
    """Build one complete provider-independent source item."""

    return KnowledgeItem(
        repository=repository,
        item_type=item_type,
        external_id=external_id,
        source_kind=KnowledgeSourceKind.REPOSITORY_FILE,
        state=state,
        title=title,
        body=body,
        summary=summary,
        url=f"https://example.test/items/{external_id}",
        author=author,
        created_at=created_at,
        updated_at=updated_at,
        affected_paths=(
            ["src/opensteward/settings.py"]
            if affected_paths is None
            else affected_paths
        ),
        components=["Settings"] if components is None else components,
        decision_significance=significance,
    )


def extract(
    *items: KnowledgeItem,
    repository: KnowledgeRepositoryRef = REPOSITORY,
    recorded_at: datetime = RECORDED_AT,
    max_candidates: int = 100,
) -> KnowledgeDecisionCandidateExtractionResult:
    """Extract candidates using concise test defaults."""

    return extract_knowledge_decision_candidates(
        repository,
        list(items),
        recorded_at=recorded_at,
        options=KnowledgeDecisionCandidateExtractionOptions(
            max_candidates=max_candidates
        ),
    )


def result_payload(
    result: KnowledgeDecisionCandidateExtractionResult,
) -> dict[str, object]:
    """Return declared result fields for validation mutations."""

    return {
        field_name: getattr(result, field_name)
        for field_name in KnowledgeDecisionCandidateExtractionResult.model_fields
    }


def candidate_source_keys(
    result: KnowledgeDecisionCandidateExtractionResult,
) -> list[str]:
    """Return candidate evidence source keys."""

    return [candidate.evidence[0].source.key for candidate in result.candidates]


def test_public_enums_and_default_options_are_exact() -> None:
    assert list(KnowledgeDecisionCandidateRule) == [
        KnowledgeDecisionCandidateRule.ADR_DOCUMENT,
        KnowledgeDecisionCandidateRule.MAINTAINER_DECISION,
        KnowledgeDecisionCandidateRule.RELEASE_COMMITMENT,
    ]
    assert [rule.value for rule in KnowledgeDecisionCandidateRule] == [
        "adr_document",
        "maintainer_decision",
        "release_commitment",
    ]
    assert list(KnowledgeDecisionCandidateSkipReason) == [
        KnowledgeDecisionCandidateSkipReason.UNSUPPORTED_ITEM_TYPE,
        KnowledgeDecisionCandidateSkipReason.INELIGIBLE_STATE,
        KnowledgeDecisionCandidateSkipReason.CANDIDATE_LIMIT,
    ]
    assert KnowledgeDecisionCandidateExtractionOptions().max_candidates == 100
    assert MAX_KNOWLEDGE_DECISION_CANDIDATES == 500


def test_options_accept_zero_and_maximum_and_reject_out_of_bounds() -> None:
    assert KnowledgeDecisionCandidateExtractionOptions(
        max_candidates=0
    ).max_candidates == 0
    assert KnowledgeDecisionCandidateExtractionOptions(
        max_candidates=MAX_KNOWLEDGE_DECISION_CANDIDATES
    ).max_candidates == MAX_KNOWLEDGE_DECISION_CANDIDATES

    for value in (-1, MAX_KNOWLEDGE_DECISION_CANDIDATES + 1):
        with pytest.raises(ValidationError):
            KnowledgeDecisionCandidateExtractionOptions(max_candidates=value)


def test_skipped_source_computes_item_key_and_requires_explanation() -> None:
    item = make_item(item_type=KnowledgeItemType.ISSUE)
    skipped = KnowledgeDecisionCandidateSkippedSource(
        source=item.to_reference(),
        reason=KnowledgeDecisionCandidateSkipReason.UNSUPPORTED_ITEM_TYPE,
        explanation=UNSUPPORTED_EXPLANATION,
    )

    assert skipped.item_key == skipped.source.key
    with pytest.raises(ValidationError):
        KnowledgeDecisionCandidateSkippedSource(
            source=item.to_reference(),
            reason=KnowledgeDecisionCandidateSkipReason.UNSUPPORTED_ITEM_TYPE,
            explanation=" ",
        )


@pytest.mark.parametrize(
    "updates",
    [
        {"source_items_seen": 0, "eligible_source_items": 1},
        {"eligible_source_items": 0, "candidates_created": 1},
        {"source_items_seen": 2, "skipped_sources": 0},
        {"eligible_source_items": 1, "candidate_limit_reached": True},
    ],
)
def test_stats_reject_inconsistent_accounting(
    updates: dict[str, object],
) -> None:
    values: dict[str, object] = {
        "source_items_seen": 1,
        "eligible_source_items": 1,
        "candidates_created": 1,
        "skipped_sources": 0,
        "candidate_limit_reached": False,
    }
    values.update(updates)

    with pytest.raises(ValidationError):
        KnowledgeDecisionCandidateExtractionStats.model_validate(values)


def test_result_rejects_naive_time_normalizes_utc_and_forbids_extras() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        extract_knowledge_decision_candidates(
            REPOSITORY,
            [],
            recorded_at=datetime(2026, 2, 1, 15, 0),
        )

    result = extract(recorded_at=NON_UTC_RECORDED_AT)
    assert result.recorded_at == RECORDED_AT
    assert result.recorded_at.tzinfo == UTC

    payload = result_payload(result)
    payload["rule_version"] = "v1"
    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_empty_input_produces_empty_complete_result() -> None:
    result = extract()

    assert result.repository == REPOSITORY
    assert result.candidates == []
    assert result.skipped_sources == []
    assert result.stats == KnowledgeDecisionCandidateExtractionStats(
        source_items_seen=0,
        eligible_source_items=0,
        candidates_created=0,
        skipped_sources=0,
        candidate_limit_reached=False,
    )
    assert result.candidate_count == 0
    assert result.skipped_count == 0
    assert result.complete is True


def test_cross_repository_and_duplicate_sources_are_rejected() -> None:
    foreign = make_item(repository=OTHER_REPOSITORY)
    with pytest.raises(ValueError, match="belong to the repository"):
        extract(foreign)

    duplicate = make_item()
    with pytest.raises(ValueError, match="keys must be unique"):
        extract(duplicate, duplicate)


def test_naive_time_is_rejected_before_source_validation() -> None:
    foreign = make_item(repository=OTHER_REPOSITORY)

    with pytest.raises(ValueError, match="timezone-aware"):
        extract_knowledge_decision_candidates(
            REPOSITORY,
            [foreign],
            recorded_at=datetime(2026, 2, 1, 15, 0),
        )


def test_extraction_does_not_mutate_input_list_or_items() -> None:
    first = make_item(external_id="2")
    second = make_item(
        external_id="1",
        item_type=KnowledgeItemType.ISSUE,
    )
    items = [first, second]
    list_before = list(items)
    item_payloads = [item.model_dump() for item in items]

    extract_knowledge_decision_candidates(
        REPOSITORY,
        items,
        recorded_at=RECORDED_AT,
    )

    assert items == list_before
    assert [item.model_dump() for item in items] == item_payloads


def test_input_order_does_not_change_output_or_ids() -> None:
    items = [
        make_item(external_id="3"),
        make_item(external_id="1", item_type=KnowledgeItemType.ISSUE),
        make_item(
            external_id="2",
            item_type=KnowledgeItemType.MAINTAINER_DECISION,
        ),
    ]

    forward = extract(*items)
    reverse = extract(*reversed(items))

    assert forward == reverse
    assert [candidate.decision_id for candidate in forward.candidates] == [
        candidate.decision_id for candidate in reverse.candidates
    ]


def test_candidate_id_uses_exact_stable_sha256_identity() -> None:
    item = make_item(external_id="1")
    candidate = extract(item).candidates[0]
    identity = (
        "opensteward:decision-candidate:"
        "v1:github:acme/framework:adr:1"
    )
    expected_digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]

    assert expected_digest == "93f29087fb589c8ba4559a1a"
    assert candidate.decision_id == f"candidate-{expected_digest}"
    assert re.fullmatch(
        r"candidate-[0-9a-f]{24}",
        candidate.decision_id,
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"title": "A different title"},
        {"body": "A different body"},
        {"summary": "A different summary"},
        {"significance": DecisionSignificance.LOW},
        {
            "created_at": CREATED_AT + timedelta(days=10),
            "updated_at": UPDATED_AT + timedelta(days=10),
        },
    ],
)
def test_candidate_id_ignores_mutable_source_content(
    updates: dict[str, Any],
) -> None:
    original = extract(make_item()).candidates[0].decision_id
    changed = extract(make_item(**updates)).candidates[0].decision_id

    assert changed == original


def test_different_source_keys_produce_different_ids() -> None:
    result = extract(
        make_item(external_id="1"),
        make_item(external_id="2"),
    )

    assert len({candidate.decision_id for candidate in result.candidates}) == 2


@pytest.mark.parametrize(
    ("state", "expected_kind", "expected_strength", "expected_summary"),
    [
        (
            KnowledgeItemState.UNKNOWN,
            DecisionEvidenceKind.OTHER,
            DecisionEvidenceStrength.MODERATE,
            ADR_DOCUMENTARY_SUMMARY,
        ),
        (
            KnowledgeItemState.DRAFT,
            DecisionEvidenceKind.OTHER,
            DecisionEvidenceStrength.MODERATE,
            ADR_DOCUMENTARY_SUMMARY,
        ),
        (
            KnowledgeItemState.ACTIVE,
            DecisionEvidenceKind.ACCEPTED_DOCUMENT,
            DecisionEvidenceStrength.AUTHORITATIVE,
            ADR_ACCEPTED_SUMMARY,
        ),
        (
            KnowledgeItemState.PUBLISHED,
            DecisionEvidenceKind.ACCEPTED_DOCUMENT,
            DecisionEvidenceStrength.AUTHORITATIVE,
            ADR_ACCEPTED_SUMMARY,
        ),
        (
            KnowledgeItemState.SUPERSEDED,
            DecisionEvidenceKind.ACCEPTED_DOCUMENT,
            DecisionEvidenceStrength.AUTHORITATIVE,
            ADR_ACCEPTED_SUMMARY,
        ),
    ],
)
def test_adr_state_matrix(
    state: KnowledgeItemState,
    expected_kind: DecisionEvidenceKind,
    expected_strength: DecisionEvidenceStrength,
    expected_summary: str,
) -> None:
    result = extract(make_item(state=state))
    evidence = result.candidates[0].evidence[0]

    assert result.candidate_count == 1
    assert evidence.kind == expected_kind
    assert evidence.strength == expected_strength
    assert evidence.summary == expected_summary


def test_adr_body_is_not_interpreted_for_status() -> None:
    result = extract(
        make_item(
            state=KnowledgeItemState.DRAFT,
            body="# Status\nAccepted and superseded",
        )
    )
    evidence = result.candidates[0].evidence[0]

    assert evidence.kind == DecisionEvidenceKind.OTHER
    assert evidence.strength == DecisionEvidenceStrength.MODERATE


@pytest.mark.parametrize(
    ("state", "expected_strength"),
    [
        (KnowledgeItemState.UNKNOWN, DecisionEvidenceStrength.MODERATE),
        (KnowledgeItemState.DRAFT, DecisionEvidenceStrength.MODERATE),
        (KnowledgeItemState.ACTIVE, DecisionEvidenceStrength.STRONG),
        (KnowledgeItemState.PUBLISHED, DecisionEvidenceStrength.STRONG),
        (KnowledgeItemState.SUPERSEDED, DecisionEvidenceStrength.STRONG),
    ],
)
def test_maintainer_decision_state_matrix(
    state: KnowledgeItemState,
    expected_strength: DecisionEvidenceStrength,
) -> None:
    result = extract(
        make_item(
            item_type=KnowledgeItemType.MAINTAINER_DECISION,
            state=state,
        )
    )
    evidence = result.candidates[0].evidence[0]

    assert result.candidate_count == 1
    assert evidence.kind == DecisionEvidenceKind.MAINTAINER_STATEMENT
    assert evidence.strength == expected_strength
    assert evidence.summary == MAINTAINER_SUMMARY


@pytest.mark.parametrize(
    "state",
    [KnowledgeItemState.PUBLISHED, KnowledgeItemState.ACTIVE],
)
def test_eligible_release_notes_create_commitment_evidence(
    state: KnowledgeItemState,
) -> None:
    result = extract(
        make_item(
            item_type=KnowledgeItemType.RELEASE_NOTE,
            state=state,
        )
    )
    evidence = result.candidates[0].evidence[0]

    assert evidence.kind == DecisionEvidenceKind.RELEASE_COMMITMENT
    assert evidence.strength == DecisionEvidenceStrength.STRONG
    assert evidence.summary == RELEASE_SUMMARY


@pytest.mark.parametrize(
    "state",
    [
        KnowledgeItemState.DRAFT,
        KnowledgeItemState.UNKNOWN,
        KnowledgeItemState.ARCHIVED,
    ],
)
def test_ineligible_release_notes_are_skipped(
    state: KnowledgeItemState,
) -> None:
    result = extract(
        make_item(
            item_type=KnowledgeItemType.RELEASE_NOTE,
            state=state,
        )
    )
    skipped = result.skipped_sources[0]

    assert result.candidates == []
    assert skipped.reason == KnowledgeDecisionCandidateSkipReason.INELIGIBLE_STATE
    assert skipped.explanation == INELIGIBLE_RELEASE_EXPLANATION
    assert result.stats.eligible_source_items == 0
    assert result.stats.candidate_limit_reached is False


@pytest.mark.parametrize(
    "item_type",
    [
        KnowledgeItemType.ISSUE,
        KnowledgeItemType.PULL_REQUEST,
        KnowledgeItemType.DISCUSSION,
        KnowledgeItemType.DOCUMENTATION,
    ],
)
def test_unsupported_item_types_are_skipped(
    item_type: KnowledgeItemType,
) -> None:
    result = extract(make_item(item_type=item_type))
    skipped = result.skipped_sources[0]

    assert result.candidates == []
    assert skipped.reason == (
        KnowledgeDecisionCandidateSkipReason.UNSUPPORTED_ITEM_TYPE
    )
    assert skipped.explanation == UNSUPPORTED_EXPLANATION
    assert result.stats.eligible_source_items == 0
    assert result.stats.candidate_limit_reached is False
    assert result.complete is True


def test_candidate_copies_only_explicit_source_fields() -> None:
    item = make_item(
        title="Keep this exact title",
        summary="Keep this explicit rationale",
        body="Do not summarize this body.",
        significance=DecisionSignificance.CRITICAL,
        affected_paths=["src/a.py", "src/b.py"],
        components=["API", "Runtime"],
    )
    candidate = extract(
        item,
        recorded_at=NON_UTC_RECORDED_AT,
    ).candidates[0]
    evidence = candidate.evidence[0]

    assert candidate.statement == item.title
    assert candidate.status == DecisionStatus.CANDIDATE
    assert candidate.significance == item.decision_significance
    assert candidate.rationale == item.summary
    assert candidate.affected_paths == item.affected_paths
    assert candidate.affected_paths is not item.affected_paths
    assert candidate.components == item.components
    assert candidate.components is not item.components
    assert candidate.recorded_at == RECORDED_AT
    assert len(candidate.evidence) == 1
    assert evidence.source == item.to_reference()
    assert evidence.relationship == DecisionEvidenceRelationship.SUPPORTS
    assert evidence.excerpt is None
    assert evidence.source_anchor is None
    assert evidence.author == item.author
    assert evidence.occurred_at == item.updated_at


def test_candidate_does_not_infer_missing_optional_fields() -> None:
    item = make_item(
        summary=None,
        body=None,
        significance=DecisionSignificance.NONE,
        affected_paths=[],
        components=[],
        author=None,
    )
    candidate = extract(item).candidates[0]

    assert candidate.rationale is None
    assert candidate.significance == DecisionSignificance.NONE
    assert candidate.affected_paths == []
    assert candidate.components == []
    assert candidate.evidence[0].author is None
    assert candidate.evidence[0].excerpt is None


def test_limit_equal_to_eligible_count_creates_every_candidate() -> None:
    items = [make_item(external_id=str(index)) for index in range(3)]
    result = extract(*items, max_candidates=3)

    assert result.candidate_count == 3
    assert result.skipped_sources == []
    assert result.stats.candidate_limit_reached is False
    assert result.complete is True


def test_partial_limit_selects_first_source_keys_and_records_every_skip() -> None:
    items = [
        make_item(external_id="3"),
        make_item(external_id="1"),
        make_item(external_id="2"),
    ]
    result = extract(*items, max_candidates=2)
    expected_selected = sorted(item.key for item in items)[:2]
    expected_skipped = sorted(item.key for item in items)[2:]

    assert sorted(candidate_source_keys(result)) == expected_selected
    assert [skipped.item_key for skipped in result.skipped_sources] == (
        expected_skipped
    )
    assert all(
        skipped.reason == KnowledgeDecisionCandidateSkipReason.CANDIDATE_LIMIT
        for skipped in result.skipped_sources
    )
    assert all(
        skipped.explanation == CANDIDATE_LIMIT_EXPLANATION
        for skipped in result.skipped_sources
    )
    assert result.stats.candidate_limit_reached is True
    assert result.complete is False


def test_zero_limit_classifies_and_limits_every_eligible_source() -> None:
    eligible = [
        make_item(external_id="2"),
        make_item(
            external_id="1",
            item_type=KnowledgeItemType.MAINTAINER_DECISION,
        ),
    ]
    result = extract(*eligible, max_candidates=0)

    assert result.candidates == []
    assert [skipped.item_key for skipped in result.skipped_sources] == sorted(
        item.key for item in eligible
    )
    assert all(
        skipped.reason == KnowledgeDecisionCandidateSkipReason.CANDIDATE_LIMIT
        for skipped in result.skipped_sources
    )
    assert result.stats.eligible_source_items == 2
    assert result.stats.candidate_limit_reached is True
    assert result.complete is False


def test_zero_limit_is_complete_without_eligible_sources() -> None:
    items = [
        make_item(item_type=KnowledgeItemType.ISSUE),
        make_item(
            external_id="2",
            item_type=KnowledgeItemType.RELEASE_NOTE,
            state=KnowledgeItemState.DRAFT,
        ),
    ]
    result = extract(*items, max_candidates=0)

    assert result.stats.eligible_source_items == 0
    assert result.stats.candidates_created == 0
    assert result.stats.skipped_sources == 2
    assert result.stats.candidate_limit_reached is False
    assert result.complete is True


def test_output_order_counts_and_stats_cover_every_source() -> None:
    items = [
        make_item(external_id="5"),
        make_item(external_id="1", item_type=KnowledgeItemType.ISSUE),
        make_item(
            external_id="4",
            item_type=KnowledgeItemType.MAINTAINER_DECISION,
        ),
        make_item(
            external_id="2",
            item_type=KnowledgeItemType.RELEASE_NOTE,
            state=KnowledgeItemState.DRAFT,
        ),
        make_item(external_id="3"),
    ]
    result = extract(*items, max_candidates=2)

    assert [candidate.key for candidate in result.candidates] == sorted(
        candidate.key for candidate in result.candidates
    )
    assert [skipped.item_key for skipped in result.skipped_sources] == sorted(
        skipped.item_key for skipped in result.skipped_sources
    )
    assert result.candidate_count == 2
    assert result.skipped_count == 3
    assert result.stats == KnowledgeDecisionCandidateExtractionStats(
        source_items_seen=5,
        eligible_source_items=3,
        candidates_created=2,
        skipped_sources=3,
        candidate_limit_reached=True,
    )


def test_result_rejects_foreign_and_non_candidate_records() -> None:
    valid = extract(make_item(state=KnowledgeItemState.ACTIVE))
    foreign = extract(
        make_item(repository=OTHER_REPOSITORY),
        repository=OTHER_REPOSITORY,
    ).candidates[0]
    payload = result_payload(valid)
    payload["candidates"] = [foreign]
    with pytest.raises(ValidationError, match="result repository"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)

    payload = result_payload(valid)
    payload["candidates"] = [
        valid.candidates[0].model_copy(
            update={"status": DecisionStatus.CONFIRMED}
        )
    ]
    with pytest.raises(ValidationError, match="CANDIDATE"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_result_rejects_multiple_evidence_and_mismatched_recorded_at() -> None:
    valid = extract(make_item(state=KnowledgeItemState.ACTIVE))
    candidate = valid.candidates[0]
    second_evidence = candidate.evidence[0].model_copy(
        update={"source_anchor": "second"}
    )

    payload = result_payload(valid)
    payload["candidates"] = [
        candidate.model_copy(
            update={"evidence": [candidate.evidence[0], second_evidence]}
        )
    ]
    with pytest.raises(ValidationError, match="exactly one"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)

    payload = result_payload(valid)
    payload["candidates"] = [
        candidate.model_copy(
            update={"recorded_at": RECORDED_AT + timedelta(seconds=1)}
        )
    ]
    with pytest.raises(ValidationError, match="result recorded_at"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_result_rejects_duplicate_candidate_keys_and_ids() -> None:
    valid = extract(make_item())
    candidate = valid.candidates[0]
    duplicate_stats = KnowledgeDecisionCandidateExtractionStats(
        source_items_seen=2,
        eligible_source_items=2,
        candidates_created=2,
        skipped_sources=0,
        candidate_limit_reached=False,
    )

    for duplicate in (
        candidate,
        candidate.model_copy(update={"decision_id": candidate.decision_id}),
    ):
        payload = result_payload(valid)
        payload["candidates"] = [candidate, duplicate]
        payload["stats"] = duplicate_stats
        with pytest.raises(ValidationError, match="unique"):
            KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_result_rejects_duplicate_and_overlapping_skipped_sources() -> None:
    skipped_result = extract(make_item(item_type=KnowledgeItemType.ISSUE))
    skipped = skipped_result.skipped_sources[0]
    payload = result_payload(skipped_result)
    payload["skipped_sources"] = [skipped, skipped]
    payload["stats"] = KnowledgeDecisionCandidateExtractionStats(
        source_items_seen=2,
        eligible_source_items=0,
        candidates_created=0,
        skipped_sources=2,
        candidate_limit_reached=False,
    )
    with pytest.raises(ValidationError, match="unique"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)

    candidate_result = extract(make_item())
    overlap = KnowledgeDecisionCandidateSkippedSource(
        source=candidate_result.candidates[0].evidence[0].source,
        reason=KnowledgeDecisionCandidateSkipReason.UNSUPPORTED_ITEM_TYPE,
        explanation=UNSUPPORTED_EXPLANATION,
    )
    payload = result_payload(candidate_result)
    payload["skipped_sources"] = [overlap]
    payload["stats"] = KnowledgeDecisionCandidateExtractionStats(
        source_items_seen=2,
        eligible_source_items=1,
        candidates_created=1,
        skipped_sources=1,
        candidate_limit_reached=False,
    )
    with pytest.raises(ValidationError, match="mutually exclusive"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_result_rejects_unaccounted_source_and_wrong_list_counts() -> None:
    valid = extract(make_item())
    payload = result_payload(valid)
    payload["stats"] = KnowledgeDecisionCandidateExtractionStats(
        source_items_seen=2,
        eligible_source_items=1,
        candidates_created=1,
        skipped_sources=1,
        candidate_limit_reached=False,
    )
    with pytest.raises(ValidationError, match="source_items_seen"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)

    limited = extract(make_item(), max_candidates=0)
    payload = result_payload(limited)
    payload["skipped_sources"] = []
    with pytest.raises(ValidationError):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_result_rejects_invalid_evidence_repository_type_and_relationship() -> None:
    valid = extract(make_item(state=KnowledgeItemState.ACTIVE))
    candidate = valid.candidates[0]
    evidence = candidate.evidence[0]
    invalid_evidence = [
        evidence.model_copy(
            update={
                "source": make_item(
                    repository=OTHER_REPOSITORY
                ).to_reference()
            }
        ),
        evidence.model_copy(
            update={
                "source": make_item(
                    item_type=KnowledgeItemType.ISSUE
                ).to_reference()
            }
        ),
        evidence.model_copy(
            update={
                "relationship": DecisionEvidenceRelationship.CONTRADICTS
            }
        ),
    ]
    expected_messages = [
        "decision repository",
        "documentary source type",
        "SUPPORTS",
    ]

    for invalid, expected_message in zip(
        invalid_evidence,
        expected_messages,
        strict=True,
    ):
        payload = result_payload(valid)
        payload["candidates"] = [
            candidate.model_copy(update={"evidence": [invalid]})
        ]
        with pytest.raises(ValidationError, match=expected_message):
            KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_result_rejects_incorrect_candidate_and_skipped_order() -> None:
    candidates = extract(
        make_item(external_id="1"),
        make_item(external_id="2"),
    )
    payload = result_payload(candidates)
    payload["candidates"] = list(reversed(candidates.candidates))
    with pytest.raises(ValidationError, match="candidate-key ascending"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)

    skipped = extract(
        make_item(external_id="1", item_type=KnowledgeItemType.ISSUE),
        make_item(external_id="2", item_type=KnowledgeItemType.DISCUSSION),
    )
    payload = result_payload(skipped)
    payload["skipped_sources"] = list(reversed(skipped.skipped_sources))
    with pytest.raises(ValidationError, match="item-key ascending"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_result_rejects_eligible_stats_inconsistent_with_limit_skips() -> None:
    result = extract(make_item())
    payload = result_payload(result)
    payload["stats"] = KnowledgeDecisionCandidateExtractionStats(
        source_items_seen=2,
        eligible_source_items=1,
        candidates_created=1,
        skipped_sources=1,
        candidate_limit_reached=False,
    )
    payload["skipped_sources"] = [
        KnowledgeDecisionCandidateSkippedSource(
            source=make_item(
                external_id="unsupported",
                item_type=KnowledgeItemType.ISSUE,
            ).to_reference(),
            reason=KnowledgeDecisionCandidateSkipReason.CANDIDATE_LIMIT,
            explanation=CANDIDATE_LIMIT_EXPLANATION,
        )
    ]

    with pytest.raises(ValidationError, match="Eligible source accounting"):
        KnowledgeDecisionCandidateExtractionResult.model_validate(payload)


def test_json_serialization_includes_candidates_evidence_stats_and_computed() -> None:
    result = extract(
        make_item(state=KnowledgeItemState.ACTIVE),
        recorded_at=NON_UTC_RECORDED_AT,
    )
    payload = result.model_dump(mode="json")
    candidate = payload["candidates"][0]
    evidence = candidate["evidence"][0]

    assert payload["recorded_at"] == "2026-02-01T15:00:00Z"
    assert payload["candidate_count"] == 1
    assert payload["skipped_count"] == 0
    assert payload["complete"] is True
    assert payload["stats"] == {
        "source_items_seen": 1,
        "eligible_source_items": 1,
        "candidates_created": 1,
        "skipped_sources": 0,
        "candidate_limit_reached": False,
    }
    assert candidate["key"].endswith(
        f":decision:{result.candidates[0].decision_id}"
    )
    assert candidate["status"] == "candidate"
    assert evidence["kind"] == "accepted_document"
    assert evidence["relationship"] == "supports"
    assert evidence["strength"] == "authoritative"
    assert evidence["source"]["item_type"] == "adr"
    assert evidence["occurred_at"] == "2026-01-02T12:00:00Z"


def test_existing_decision_record_candidate_validation_remains_compatible() -> None:
    candidate = extract(make_item()).candidates[0]

    assert candidate.status == DecisionStatus.CANDIDATE
    assert candidate.evidence[0].strength == DecisionEvidenceStrength.MODERATE
