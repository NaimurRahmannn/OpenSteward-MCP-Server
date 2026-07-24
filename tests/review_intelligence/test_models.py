"""Tests for provider-independent review-intelligence models."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.knowledge import (
    KnowledgeItemReference,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)
from opensteward.review_intelligence import (
    REVIEW_COST_CHANGE_DISPERSION_WEIGHT,
    REVIEW_COST_CHANGE_SIZE_WEIGHT,
    REVIEW_COST_HISTORICAL_COMPLEXITY_WEIGHT,
    REVIEW_COST_RISK_SENSITIVE_PATHS_WEIGHT,
    REVIEW_COST_VALIDATION_GAPS_WEIGHT,
    ReviewCostAssessment,
    ReviewCostAssessmentInput,
    ReviewCostChangedFile,
    ReviewCostChangeType,
    ReviewCostHistoricalContext,
    ReviewCostLevel,
    ReviewCostSignal,
    ReviewCostSignalContribution,
)

REPOSITORY = KnowledgeRepositoryRef(
    provider="github",
    namespace="acme",
    name="framework",
)
PULL_REQUEST = KnowledgeItemReference(
    repository=REPOSITORY,
    item_type=KnowledgeItemType.PULL_REQUEST,
    external_id="17",
    source_kind=KnowledgeSourceKind.GITHUB,
    title="Parser update",
)
ASSESSED_AT = datetime(2026, 6, 1, tzinfo=UTC)
WEIGHTS = [
    REVIEW_COST_CHANGE_SIZE_WEIGHT,
    REVIEW_COST_CHANGE_DISPERSION_WEIGHT,
    REVIEW_COST_RISK_SENSITIVE_PATHS_WEIGHT,
    REVIEW_COST_VALIDATION_GAPS_WEIGHT,
    REVIEW_COST_HISTORICAL_COMPLEXITY_WEIGHT,
]


def changed_file(path: str = "src/app.py", **updates: Any) -> ReviewCostChangedFile:
    payload: dict[str, Any] = {
        "path": path,
        "change_type": ReviewCostChangeType.MODIFIED,
        "additions": 10,
        "deletions": 5,
    }
    payload.update(updates)
    return ReviewCostChangedFile(**payload)


def assessment_input(**updates: Any) -> ReviewCostAssessmentInput:
    payload: dict[str, Any] = {
        "repository": REPOSITORY,
        "pull_request": PULL_REQUEST,
        "files": [changed_file()],
        "commit_count": 2,
        "preferred_diff_size": 500,
        "policy_present": True,
        "protected_changed_paths": [],
        "draft": False,
        "merge_conflict": False,
        "required_checks_total": 0,
        "required_checks_passed": 0,
        "required_checks_failed": 0,
        "required_checks_pending": 0,
        "approval_count": 0,
        "changes_requested_count": 0,
        "historical_context": ReviewCostHistoricalContext(
            related_match_count=0,
            rejected_or_superseded_count=0,
            high_significance_count=0,
            unresolved_count=0,
            source_history_complete=True,
            ranking_coverage_complete=True,
            result_truncated=False,
        ),
    }
    payload.update(updates)
    return ReviewCostAssessmentInput(**payload)


def contributions(score: int) -> list[ReviewCostSignalContribution]:
    return [
        ReviewCostSignalContribution(
            signal=signal,
            signal_score=score,
            weight_percent=weight,
            weighted_basis_points=score * weight,
            explanation=f"{signal.value} explanation",
            evidence=[f"{signal.value} evidence"],
        )
        for signal, weight in zip(ReviewCostSignal, WEIGHTS, strict=True)
    ]


def test_changed_file_normalizes_paths_and_computes_structure() -> None:
    file = changed_file(
        "./src\\parser\\registry.py",
        previous_path="./old\\registry.py",
    )

    assert file.path == "src/parser/registry.py"
    assert file.previous_path == "old/registry.py"
    assert file.changes == 15
    assert file.directory == "src/parser"
    assert file.top_level_component == "src"
    assert changed_file("README.md").directory == "."
    assert changed_file("README.md").top_level_component == "(root)"


@pytest.mark.parametrize(
    "path",
    ["", "/absolute.py", "C:\\absolute.py", "src//app.py", "src/../app.py"],
)
def test_changed_file_rejects_unsafe_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        changed_file(path)
    with pytest.raises(ValidationError):
        changed_file("src/app.py", previous_path=path)


def test_changed_file_previous_path_and_strictness() -> None:
    with pytest.raises(ValidationError, match="must differ"):
        changed_file("src/app.py", previous_path="./src/app.py")
    with pytest.raises(ValidationError):
        ReviewCostChangedFile(
            path="src/app.py",
            change_type="modified",
            additions=1,
            deletions=1,
            extra=True,
        )


def test_historical_context_classified_counts_are_bounded() -> None:
    with pytest.raises(ValidationError, match="must not exceed"):
        ReviewCostHistoricalContext(
            related_match_count=1,
            rejected_or_superseded_count=2,
            high_significance_count=0,
            unresolved_count=0,
            source_history_complete=True,
            ranking_coverage_complete=True,
            result_truncated=False,
        )


def test_assessment_input_validates_identity_files_paths_and_checks() -> None:
    selected = assessment_input(
        files=[
            changed_file(
                "src/new.py",
                previous_path="src/old.py",
            )
        ],
        protected_changed_paths=["./src\\old.py"],
        required_checks_total=3,
        required_checks_passed=1,
        required_checks_failed=1,
        required_checks_pending=1,
    )
    assert selected.protected_changed_paths == ["src/old.py"]
    assert selected.additions == 10
    assert selected.deletions == 5
    assert selected.total_changes == 15
    assert selected.changed_file_count == 1
    assert selected.distinct_directory_count == 1
    assert selected.distinct_component_count == 1

    other_repository = REPOSITORY.model_copy(update={"name": "other"})
    with pytest.raises(ValidationError, match="belong"):
        assessment_input(
            pull_request=PULL_REQUEST.model_copy(
                update={"repository": other_repository}
            )
        )
    with pytest.raises(ValidationError, match="pull-request"):
        assessment_input(
            pull_request=PULL_REQUEST.model_copy(
                update={"item_type": KnowledgeItemType.ISSUE}
            )
        )
    with pytest.raises(ValidationError, match="unique"):
        assessment_input(files=[changed_file(), changed_file()])
    with pytest.raises(ValidationError, match="identify"):
        assessment_input(protected_changed_paths=["other.py"])
    with pytest.raises(ValidationError, match="sum"):
        assessment_input(
            required_checks_total=2,
            required_checks_passed=1,
        )


def test_contribution_validates_basis_points_evidence_and_bounds() -> None:
    contribution = contributions(25)[0]
    assert contribution.weighted_score == 6.25

    payload = contribution.model_dump(exclude_computed_fields=True)
    with pytest.raises(ValidationError, match="must equal"):
        ReviewCostSignalContribution(
            **(payload | {"weighted_basis_points": 1})
        )
    for evidence in ([""], ["same", "same"]):
        with pytest.raises(ValidationError):
            ReviewCostSignalContribution(**(payload | {"evidence": evidence}))


@pytest.mark.parametrize(
    ("score", "level"),
    [
        (24, ReviewCostLevel.LOW),
        (25, ReviewCostLevel.MODERATE),
        (49, ReviewCostLevel.MODERATE),
        (50, ReviewCostLevel.HIGH),
        (74, ReviewCostLevel.HIGH),
        (75, ReviewCostLevel.CRITICAL),
    ],
)
def test_assessment_score_level_boundaries(score: int, level: ReviewCostLevel) -> None:
    assessment = ReviewCostAssessment(
        repository=REPOSITORY,
        pull_request=PULL_REQUEST,
        assessed_at=ASSESSED_AT,
        contributions=contributions(score),
        reducers=[],
        warnings=[],
    )
    assert assessment.total_weighted_basis_points == score * 100
    assert assessment.score == score
    assert assessment.level == level


def test_assessment_validates_order_weights_time_and_text_lists() -> None:
    non_utc = datetime(
        2026,
        6,
        1,
        6,
        tzinfo=timezone(timedelta(hours=6)),
    )
    assessment = ReviewCostAssessment(
        repository=REPOSITORY,
        pull_request=PULL_REQUEST,
        assessed_at=non_utc,
        contributions=contributions(10),
        reducers=["All checks passed."],
        warnings=[],
    )
    assert assessment.assessed_at == ASSESSED_AT
    assert assessment.assessed_at.tzinfo is UTC

    with pytest.raises(ValidationError, match="exact signal order"):
        ReviewCostAssessment(
            repository=REPOSITORY,
            pull_request=PULL_REQUEST,
            assessed_at=ASSESSED_AT,
            contributions=list(reversed(contributions(10))),
            reducers=[],
            warnings=[],
        )
    invalid_weight = contributions(10)
    invalid_weight[0] = invalid_weight[0].model_copy(
        update={"weight_percent": 24, "weighted_basis_points": 240}
    )
    with pytest.raises(ValidationError, match="weights"):
        ReviewCostAssessment(
            repository=REPOSITORY,
            pull_request=PULL_REQUEST,
            assessed_at=ASSESSED_AT,
            contributions=invalid_weight,
            reducers=[],
            warnings=[],
        )
    for field_name in ("reducers", "warnings"):
        with pytest.raises(ValidationError):
            ReviewCostAssessment(
                repository=REPOSITORY,
                pull_request=PULL_REQUEST,
                assessed_at=ASSESSED_AT,
                contributions=contributions(10),
                reducers=["same", "same"] if field_name == "reducers" else [],
                warnings=["same", "same"] if field_name == "warnings" else [],
            )


def test_primary_drivers_completeness_and_json_serialization() -> None:
    selected = contributions(0)
    selected[0] = ReviewCostSignalContribution(
        signal=ReviewCostSignal.CHANGE_SIZE,
        signal_score=40,
        weight_percent=25,
        weighted_basis_points=1000,
        explanation="size",
        evidence=[],
    )
    selected[2] = ReviewCostSignalContribution(
        signal=ReviewCostSignal.RISK_SENSITIVE_PATHS,
        signal_score=40,
        weight_percent=25,
        weighted_basis_points=1000,
        explanation="risk",
        evidence=[],
    )
    assessment = ReviewCostAssessment(
        repository=REPOSITORY,
        pull_request=PULL_REQUEST,
        assessed_at=ASSESSED_AT,
        contributions=selected,
        reducers=[],
        warnings=[],
    )
    assert assessment.primary_drivers == [
        ReviewCostSignal.CHANGE_SIZE,
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    ]
    assert assessment.complete is True
    data = assessment.model_dump(mode="json")
    assert data["level"] == "low"
    assert data["primary_drivers"] == ["change_size", "risk_sensitive_paths"]
    incomplete = assessment.model_copy(update={"warnings": ["incomplete"]})
    assert incomplete.complete is False
