"""Tests for unified project knowledge and decision models."""

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from opensteward.knowledge import (
    DecisionEvidence,
    DecisionEvidenceKind,
    DecisionEvidenceRelationship,
    DecisionEvidenceStrength,
    DecisionRecord,
    DecisionSignificance,
    DecisionStatus,
    KnowledgeActor,
    KnowledgeActorType,
    KnowledgeItem,
    KnowledgeItemReference,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

CREATED_AT = datetime(2025, 1, 10, 9, 0, tzinfo=UTC)
UPDATED_AT = datetime(2025, 1, 11, 10, 30, tzinfo=UTC)
NON_UTC_TIME = datetime(2025, 1, 10, 15, 0, tzinfo=timezone(timedelta(hours=6)))


def make_repository(**overrides: str) -> KnowledgeRepositoryRef:
    values = {
        "provider": "github",
        "namespace": "acme",
        "name": "framework",
    }
    values.update(overrides)
    return KnowledgeRepositoryRef(**values)


def make_reference(
    repository: KnowledgeRepositoryRef | None = None,
    external_id: str = "398",
) -> KnowledgeItemReference:
    return KnowledgeItemReference(
        repository=repository or make_repository(),
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id=external_id,
        source_kind=KnowledgeSourceKind.GITHUB,
        title="Adopt typed configuration",
        url="https://example.test/acme/framework/pull/398",
    )


def make_evidence(
    *,
    repository: KnowledgeRepositoryRef | None = None,
    strength: DecisionEvidenceStrength = DecisionEvidenceStrength.MODERATE,
    relationship: DecisionEvidenceRelationship = DecisionEvidenceRelationship.SUPPORTS,
    external_id: str = "398",
    source_anchor: str | None = "review-12",
) -> DecisionEvidence:
    return DecisionEvidence(
        source=make_reference(repository, external_id),
        kind=DecisionEvidenceKind.MAINTAINER_STATEMENT,
        relationship=relationship,
        strength=strength,
        summary="Maintainers agreed on the configuration format.",
        excerpt="Let's use a typed configuration model.",
        source_anchor=source_anchor,
        occurred_at=UPDATED_AT,
    )


def make_decision(**overrides: object) -> DecisionRecord:
    repository = overrides.get("repository")
    if not isinstance(repository, KnowledgeRepositoryRef):
        repository = make_repository()

    values: dict[str, object] = {
        "repository": repository,
        "decision_id": "typed-config",
        "statement": "Repository configuration uses typed models.",
        "evidence": [make_evidence(repository=repository)],
        "recorded_at": UPDATED_AT,
    }
    values.update(overrides)
    return DecisionRecord.model_validate(values)


def test_repository_ref_builds_full_name_and_key() -> None:
    repository = make_repository()

    assert repository.full_name == "acme/framework"
    assert repository.key == "github:acme/framework"


def test_repository_ref_normalizes_provider_to_lowercase() -> None:
    repository = make_repository(provider=" GitHub-App ")

    assert repository.provider == "github-app"


@pytest.mark.parametrize(
    "provider",
    ["", "123github", "_github", "-github", "git hub", "git.hub", "git/hub"],
)
def test_repository_ref_rejects_invalid_provider(provider: str) -> None:
    with pytest.raises(ValidationError):
        make_repository(provider=provider)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("namespace", "acme/tools"),
        ("namespace", "acme\\tools"),
        ("name", "framework/core"),
        ("name", "framework\\core"),
    ],
)
def test_repository_ref_rejects_slash_containing_segments(
    field_name: str,
    value: str,
) -> None:
    with pytest.raises(ValidationError):
        make_repository(**{field_name: value})


@pytest.mark.parametrize("field_name", ["namespace", "name"])
@pytest.mark.parametrize("value", [".", ".."])
def test_repository_ref_rejects_dot_segments(field_name: str, value: str) -> None:
    with pytest.raises(ValidationError):
        make_repository(**{field_name: value})


def test_knowledge_item_builds_and_serializes_complete_issue() -> None:
    item = KnowledgeItem(
        repository=make_repository(),
        item_type=KnowledgeItemType.ISSUE,
        external_id="217",
        source_kind=KnowledgeSourceKind.GITHUB,
        state=KnowledgeItemState.CLOSED,
        title="Support repository configuration",
        body="Configuration should be repository-local.",
        summary="Add repository-local configuration.",
        url="https://example.test/acme/framework/issues/217",
        author=KnowledgeActor(
            identifier="maintainer",
            actor_type=KnowledgeActorType.USER,
            display_name="A Maintainer",
            url="https://example.test/maintainer",
        ),
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
        closed_at=UPDATED_AT,
        labels=["Feature", "Configuration"],
        affected_paths=["src/opensteward/settings.py"],
        components=["Settings"],
        decision_significance=DecisionSignificance.HIGH,
    )

    data = item.model_dump(mode="json")

    assert data["item_type"] == "issue"
    assert data["state"] == "closed"
    assert data["author"]["actor_type"] == "user"
    assert data["created_at"] == "2025-01-10T09:00:00Z"
    assert data["decision_significance"] == "high"
    assert data["key"] == "github:acme/framework:issue:217"
    assert data["repository"]["key"] == "github:acme/framework"


def test_knowledge_item_key_has_stable_format() -> None:
    item = KnowledgeItem(
        repository=make_repository(),
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="398",
        source_kind=KnowledgeSourceKind.GITHUB,
        title="Adopt typed configuration",
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
    )

    assert item.key == "github:acme/framework:pull_request:398"


def test_knowledge_item_to_reference_preserves_required_fields() -> None:
    item = KnowledgeItem(
        repository=make_repository(),
        item_type=KnowledgeItemType.DOCUMENTATION,
        external_id="docs/configuration.md",
        source_kind=KnowledgeSourceKind.REPOSITORY_FILE,
        title="Configuration guide",
        url="https://example.test/acme/framework/blob/main/docs/configuration.md",
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
    )

    reference = item.to_reference()

    assert reference.repository == item.repository
    assert reference.item_type == item.item_type
    assert reference.external_id == item.external_id
    assert reference.source_kind == item.source_kind
    assert reference.title == item.title
    assert reference.url == item.url


def test_knowledge_item_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        KnowledgeItem.model_validate(
            {
                "repository": make_repository(),
                "item_type": "issue",
                "external_id": "217",
                "source_kind": "github",
                "title": "Configuration support",
                "created_at": CREATED_AT,
                "updated_at": UPDATED_AT,
                "score": 0.9,
            }
        )


def test_knowledge_models_strip_whitespace() -> None:
    item = KnowledgeItem(
        repository=make_repository(namespace=" acme ", name=" framework "),
        item_type=KnowledgeItemType.ISSUE,
        external_id=" 217 ",
        source_kind=KnowledgeSourceKind.GITHUB,
        title=" Configuration support ",
        labels=[" Feature "],
        components=[" Settings "],
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
    )

    assert item.repository.full_name == "acme/framework"
    assert item.external_id == "217"
    assert item.title == "Configuration support"
    assert item.labels == ["Feature"]
    assert item.components == ["Settings"]


@pytest.mark.parametrize("field_name", ["created_at", "updated_at", "closed_at"])
def test_knowledge_item_rejects_naive_timestamps(field_name: str) -> None:
    values: dict[str, object] = {
        "repository": make_repository(),
        "item_type": KnowledgeItemType.ISSUE,
        "external_id": "217",
        "source_kind": KnowledgeSourceKind.GITHUB,
        "title": "Configuration support",
        "created_at": CREATED_AT,
        "updated_at": UPDATED_AT,
        "closed_at": UPDATED_AT,
    }
    values[field_name] = datetime(2025, 1, 10, 9, 0)

    with pytest.raises(ValidationError, match="timezone-aware"):
        KnowledgeItem.model_validate(values)


def test_knowledge_item_normalizes_aware_timestamps_to_utc() -> None:
    item = KnowledgeItem(
        repository=make_repository(),
        item_type=KnowledgeItemType.ISSUE,
        external_id="217",
        source_kind=KnowledgeSourceKind.GITHUB,
        title="Configuration support",
        created_at=NON_UTC_TIME,
        updated_at=NON_UTC_TIME,
        closed_at=NON_UTC_TIME,
    )

    assert item.created_at == CREATED_AT
    assert item.updated_at == CREATED_AT
    assert item.closed_at == CREATED_AT
    assert item.created_at.tzinfo is UTC


def test_knowledge_item_rejects_updated_at_before_created_at() -> None:
    with pytest.raises(ValidationError, match="updated_at"):
        KnowledgeItem(
            repository=make_repository(),
            item_type=KnowledgeItemType.ISSUE,
            external_id="217",
            source_kind=KnowledgeSourceKind.GITHUB,
            title="Configuration support",
            created_at=UPDATED_AT,
            updated_at=CREATED_AT,
        )


def test_knowledge_item_rejects_closed_at_before_created_at() -> None:
    with pytest.raises(ValidationError, match="closed_at"):
        KnowledgeItem(
            repository=make_repository(),
            item_type=KnowledgeItemType.ISSUE,
            external_id="217",
            source_kind=KnowledgeSourceKind.GITHUB,
            title="Configuration support",
            created_at=UPDATED_AT,
            updated_at=UPDATED_AT,
            closed_at=CREATED_AT,
        )


def test_knowledge_item_rejects_duplicate_labels_case_insensitively() -> None:
    with pytest.raises(ValidationError, match="case-insensitively"):
        KnowledgeItem(
            repository=make_repository(),
            item_type=KnowledgeItemType.ISSUE,
            external_id="217",
            source_kind=KnowledgeSourceKind.GITHUB,
            title="Configuration support",
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
            labels=["Feature", "feature"],
        )


def test_knowledge_item_rejects_duplicate_components_case_insensitively() -> None:
    with pytest.raises(ValidationError, match="case-insensitively"):
        KnowledgeItem(
            repository=make_repository(),
            item_type=KnowledgeItemType.ISSUE,
            external_id="217",
            source_kind=KnowledgeSourceKind.GITHUB,
            title="Configuration support",
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
            components=["Settings", "settings"],
        )


def test_knowledge_item_normalizes_repository_paths() -> None:
    item = KnowledgeItem(
        repository=make_repository(),
        item_type=KnowledgeItemType.ISSUE,
        external_id="217",
        source_kind=KnowledgeSourceKind.GITHUB,
        title="Configuration support",
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
        affected_paths=[" .\\src\\opensteward\\settings.py ", "././docs/configuration.md"],
    )

    assert item.affected_paths == [
        "src/opensteward/settings.py",
        "docs/configuration.md",
    ]


def test_knowledge_item_rejects_duplicate_normalized_paths() -> None:
    with pytest.raises(ValidationError, match="unique after normalization"):
        KnowledgeItem(
            repository=make_repository(),
            item_type=KnowledgeItemType.ISSUE,
            external_id="217",
            source_kind=KnowledgeSourceKind.GITHUB,
            title="Configuration support",
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
            affected_paths=["src/settings.py", ".\\src\\settings.py"],
        )


@pytest.mark.parametrize(
    "path",
    [
        "",
        "/src/settings.py",
        "C:\\src\\settings.py",
        "../src/settings.py",
        "src/../settings.py",
        "src/./settings.py",
        "src//settings.py",
        "src/settings.py/",
    ],
)
def test_knowledge_item_rejects_unsafe_repository_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        KnowledgeItem(
            repository=make_repository(),
            item_type=KnowledgeItemType.ISSUE,
            external_id="217",
            source_kind=KnowledgeSourceKind.GITHUB,
            title="Configuration support",
            created_at=CREATED_AT,
            updated_at=UPDATED_AT,
            affected_paths=[path],
        )


@pytest.mark.parametrize("field_name", ["body", "summary", "url"])
def test_knowledge_item_rejects_empty_optional_strings(field_name: str) -> None:
    values: dict[str, object] = {
        "repository": make_repository(),
        "item_type": KnowledgeItemType.ISSUE,
        "external_id": "217",
        "source_kind": KnowledgeSourceKind.GITHUB,
        "title": "Configuration support",
        "created_at": CREATED_AT,
        "updated_at": UPDATED_AT,
        field_name: "   ",
    }

    with pytest.raises(ValidationError):
        KnowledgeItem.model_validate(values)


def test_candidate_decision_accepts_moderate_evidence() -> None:
    decision = make_decision()

    assert decision.status == DecisionStatus.CANDIDATE
    assert decision.evidence[0].strength == DecisionEvidenceStrength.MODERATE


@pytest.mark.parametrize(
    "strength",
    [DecisionEvidenceStrength.WEAK, DecisionEvidenceStrength.MODERATE],
)
def test_confirmed_decision_rejects_only_insufficient_evidence(
    strength: DecisionEvidenceStrength,
) -> None:
    repository = make_repository()

    with pytest.raises(ValidationError, match="strong or authoritative"):
        make_decision(
            repository=repository,
            status=DecisionStatus.CONFIRMED,
            evidence=[make_evidence(repository=repository, strength=strength)],
        )


def test_confirmed_decision_accepts_strong_supporting_evidence() -> None:
    repository = make_repository()
    decision = make_decision(
        repository=repository,
        status=DecisionStatus.CONFIRMED,
        evidence=[
            make_evidence(
                repository=repository,
                strength=DecisionEvidenceStrength.STRONG,
            )
        ],
    )

    assert decision.status == DecisionStatus.CONFIRMED


@pytest.mark.parametrize(
    "relationship",
    [
        DecisionEvidenceRelationship.SUPPORTS,
        DecisionEvidenceRelationship.SUPERSEDES,
        DecisionEvidenceRelationship.DEFINES_EXCEPTION,
    ],
)
def test_authoritative_non_contradicting_evidence_satisfies_confirmation(
    relationship: DecisionEvidenceRelationship,
) -> None:
    repository = make_repository()
    decision = make_decision(
        repository=repository,
        status=DecisionStatus.CONFIRMED,
        evidence=[
            make_evidence(
                repository=repository,
                strength=DecisionEvidenceStrength.AUTHORITATIVE,
                relationship=relationship,
            )
        ],
    )

    assert decision.status == DecisionStatus.CONFIRMED


def test_strong_contradicting_evidence_does_not_satisfy_confirmation() -> None:
    repository = make_repository()

    with pytest.raises(ValidationError, match="non-contradicting"):
        make_decision(
            repository=repository,
            status=DecisionStatus.CONFIRMED,
            evidence=[
                make_evidence(
                    repository=repository,
                    strength=DecisionEvidenceStrength.STRONG,
                    relationship=DecisionEvidenceRelationship.CONTRADICTS,
                )
            ],
        )


def test_decision_rejects_evidence_from_different_repository() -> None:
    repository = make_repository()
    other_repository = make_repository(namespace="other")

    with pytest.raises(ValidationError, match="decision repository"):
        make_decision(
            repository=repository,
            evidence=[make_evidence(repository=other_repository)],
        )


def test_decision_rejects_duplicate_evidence_entries() -> None:
    repository = make_repository()
    evidence = make_evidence(repository=repository)

    with pytest.raises(ValidationError, match="duplicates"):
        make_decision(repository=repository, evidence=[evidence, evidence])


def test_decision_rejects_duplicate_affected_paths_after_normalization() -> None:
    with pytest.raises(ValidationError, match="unique after normalization"):
        make_decision(affected_paths=["src/settings.py", ".\\src\\settings.py"])


def test_decision_rejects_duplicate_components_case_insensitively() -> None:
    with pytest.raises(ValidationError, match="case-insensitively"):
        make_decision(components=["Settings", "settings"])


def test_decision_rejects_naive_recorded_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_decision(recorded_at=datetime(2025, 1, 11, 10, 30))


def test_evidence_rejects_naive_occurred_at() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        DecisionEvidence(
            source=make_reference(),
            kind=DecisionEvidenceKind.MERGE_OUTCOME,
            relationship=DecisionEvidenceRelationship.SUPPORTS,
            strength=DecisionEvidenceStrength.STRONG,
            summary="The accepted pull request was merged.",
            occurred_at=datetime(2025, 1, 11, 10, 30),
        )


def test_decision_key_has_stable_format() -> None:
    decision = make_decision()

    assert decision.key == "github:acme/framework:decision:typed-config"


def test_decision_record_serializes_enums_as_strings_in_json_mode() -> None:
    repository = make_repository()
    decision = make_decision(
        repository=repository,
        status=DecisionStatus.CONFIRMED,
        significance=DecisionSignificance.HIGH,
        evidence=[
            make_evidence(
                repository=repository,
                strength=DecisionEvidenceStrength.STRONG,
                relationship=DecisionEvidenceRelationship.SUPPORTS,
            )
        ],
    )

    data = decision.model_dump(mode="json")

    assert data["status"] == "confirmed"
    assert data["significance"] == "high"
    assert data["evidence"][0]["kind"] == "maintainer_statement"
    assert data["evidence"][0]["relationship"] == "supports"
    assert data["evidence"][0]["strength"] == "strong"


@pytest.mark.parametrize("model_name", ["evidence", "record"])
def test_decision_models_reject_unknown_fields(model_name: str) -> None:
    if model_name == "evidence":
        values = {
            "source": make_reference(),
            "kind": DecisionEvidenceKind.MERGE_OUTCOME,
            "relationship": DecisionEvidenceRelationship.SUPPORTS,
            "strength": DecisionEvidenceStrength.STRONG,
            "summary": "The accepted pull request was merged.",
            "unknown": True,
        }
        model = DecisionEvidence
    else:
        values = {
            "repository": make_repository(),
            "decision_id": "typed-config",
            "statement": "Repository configuration uses typed models.",
            "evidence": [make_evidence()],
            "recorded_at": UPDATED_AT,
            "unknown": True,
        }
        model = DecisionRecord

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        model.model_validate(values)
