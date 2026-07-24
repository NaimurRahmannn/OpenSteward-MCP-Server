"""Boundary tests for deterministic review-cost scoring."""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import pytest

from opensteward.knowledge import (
    KnowledgeItemReference,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)
from opensteward.review_intelligence import (
    ReviewCostAssessmentInput,
    ReviewCostAssessmentOptions,
    ReviewCostAssessmentService,
    ReviewCostChangedFile,
    ReviewCostChangeType,
    ReviewCostHistoricalContext,
    ReviewCostSignal,
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
)
ASSESSED_AT = datetime(2026, 6, 1, tzinfo=UTC)


def file(
    path: str,
    *,
    changes: int = 1,
) -> ReviewCostChangedFile:
    return ReviewCostChangedFile(
        path=path,
        change_type=ReviewCostChangeType.MODIFIED,
        additions=changes,
        deletions=0,
    )


def input_model(**updates: Any) -> ReviewCostAssessmentInput:
    payload: dict[str, Any] = {
        "repository": REPOSITORY,
        "pull_request": PULL_REQUEST,
        "files": [],
        "commit_count": 0,
        "preferred_diff_size": 100,
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


def score(
    selected_input: ReviewCostAssessmentInput,
    signal: ReviewCostSignal,
    *,
    evidence_limit: int = 20,
) -> tuple[int, list[str], list[str]]:
    result = ReviewCostAssessmentService().assess(
        selected_input,
        assessed_at=ASSESSED_AT,
        options=ReviewCostAssessmentOptions(
            max_evidence_items_per_signal=evidence_limit
        ),
    )
    contribution = next(
        item for item in result.contributions if item.signal == signal
    )
    return contribution.signal_score, contribution.evidence, result.reducers


@pytest.mark.parametrize(
    ("changes", "expected"),
    [(0, 0), (25, 10), (50, 25), (100, 50), (200, 75), (201, 100)],
)
def test_change_line_boundaries(changes: int, expected: int) -> None:
    files = [] if changes == 0 else [file("docs/readme.md", changes=changes)]
    actual, evidence, _ = score(
        input_model(files=files),
        ReviewCostSignal.CHANGE_SIZE,
    )
    assert actual == expected
    assert evidence[0] == f"{changes} changed lines."


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, 0),
        (1, 10),
        (3, 10),
        (4, 30),
        (10, 30),
        (11, 60),
        (25, 60),
        (26, 80),
        (50, 80),
        (51, 100),
    ],
)
def test_changed_file_boundaries(count: int, expected: int) -> None:
    actual, _, _ = score(
        input_model(
            files=[file(f"docs/file-{index}.md", changes=0) for index in range(count)]
        ),
        ReviewCostSignal.CHANGE_SIZE,
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("count", "expected"),
    [(0, 0), (1, 10), (2, 10), (3, 30), (5, 30), (6, 60), (10, 60), (11, 80), (20, 80), (21, 100)],
)
def test_commit_boundaries(count: int, expected: int) -> None:
    actual, _, _ = score(
        input_model(commit_count=count),
        ReviewCostSignal.CHANGE_SIZE,
    )
    assert actual == expected


def test_change_size_uses_maximum_subscore() -> None:
    actual, _, _ = score(
        input_model(files=[file("docs/a.md", changes=1)], commit_count=21),
        ReviewCostSignal.CHANGE_SIZE,
    )
    assert actual == 100


@pytest.mark.parametrize(
    ("components", "expected"),
    [(0, 0), (1, 10), (2, 30), (3, 60), (4, 60), (5, 80), (8, 80), (9, 100)],
)
def test_component_dispersion_boundaries(components: int, expected: int) -> None:
    files = [
        file(f"component-{index}/file.md", changes=0)
        for index in range(components)
    ]
    actual, _, _ = score(
        input_model(files=files),
        ReviewCostSignal.CHANGE_DISPERSION,
    )
    assert actual == expected


@pytest.mark.parametrize(
    ("directories", "expected"),
    [(0, 0), (1, 10), (3, 10), (4, 30), (6, 30), (7, 60), (12, 60), (13, 80), (25, 80), (26, 100)],
)
def test_directory_dispersion_boundaries(directories: int, expected: int) -> None:
    files = [
        file(f"root/dir-{index}/file.md", changes=0)
        for index in range(directories)
    ]
    actual, evidence, _ = score(
        input_model(files=files),
        ReviewCostSignal.CHANGE_DISPERSION,
    )
    assert actual == expected
    assert evidence[-1] == f"{directories} directories."


def test_root_file_dispersion_and_maximum_subscore() -> None:
    actual, evidence, _ = score(
        input_model(
            files=[
                file("README.md", changes=0),
                file("a/file.md", changes=0),
                file("b/file.md", changes=0),
            ]
        ),
        ReviewCostSignal.CHANGE_DISPERSION,
    )
    assert actual == 60
    assert evidence == ["3 top-level components.", "3 directories."]


@pytest.mark.parametrize(
    "token",
    [
        "auth",
        "authentication",
        "authorization",
        "oauth",
        "token",
        "tokens",
        "secret",
        "secrets",
        "permission",
        "permissions",
        "crypto",
        "cryptography",
        "security",
        "access_control",
    ],
)
def test_every_security_token_is_classified(token: str) -> None:
    actual, evidence, _ = score(
        input_model(files=[file(f"src/{token}/service.py")]),
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    )
    assert actual == 90
    assert "security_sensitive" in " ".join(evidence)


@pytest.mark.parametrize("stem", ["token_store.py", "secret-store.py"])
def test_security_stem_splitting(stem: str) -> None:
    actual, _, _ = score(
        input_model(files=[file(f"src/{stem}")]),
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    )
    assert actual == 90


def test_security_does_not_use_unrestricted_substrings() -> None:
    actual, evidence, _ = score(
        input_model(files=[file("documentation/authors.md")]),
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    )
    assert actual == 0
    assert evidence == []


@pytest.mark.parametrize(
    ("path", "category", "base"),
    [
        ("db/migrate/001.sql", "database_migration", 85),
        ("database/migrations/001.sql", "database_migration", 85),
        ("alembic/versions/001.py", "database_migration", 85),
        (".github/workflows/ci.yml", "automation_or_deployment", 75),
        ("Dockerfile", "automation_or_deployment", 75),
        ("terraform/main.tf", "automation_or_deployment", 75),
        ("pyproject.toml", "dependency_manifest", 60),
        ("requirements-dev.txt", "dependency_manifest", 60),
    ],
)
def test_risk_category_patterns(path: str, category: str, base: int) -> None:
    actual, evidence, _ = score(
        input_model(files=[file(path)]),
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    )
    assert actual == base
    assert category in " ".join(evidence)


@pytest.mark.parametrize(
    "path",
    ["tests/app.py", "src/test_app.py", "src/app_test.py", "src/app.spec.ts", "src/app.test.ts"],
)
def test_test_paths_are_not_direct_risk(path: str) -> None:
    actual, _, _ = score(
        input_model(files=[file(path)]),
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    )
    assert actual == 0


@pytest.mark.parametrize("path", ["docs/guide.py", "README.md", "guide.rst"])
def test_documentation_paths_are_not_production_or_direct_risk(path: str) -> None:
    selected = input_model(files=[file(path)])
    risk, _, _ = score(selected, ReviewCostSignal.RISK_SENSITIVE_PATHS)
    validation, _, _ = score(selected, ReviewCostSignal.VALIDATION_GAPS)
    assert risk == 0
    assert validation == 0


def test_protected_and_multiple_risk_categories_increment_and_cap() -> None:
    selected = input_model(
        files=[
            file("security/migrations/pyproject.toml"),
            file(".github/workflows/deploy.yml"),
        ],
        protected_changed_paths=["security/migrations/pyproject.toml"],
    )
    actual, evidence, _ = score(
        selected,
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    )
    assert actual == 100
    assert evidence[:5] == [
        "1 changed paths classified as protected.",
        "1 changed paths classified as security_sensitive.",
        "1 changed paths classified as database_migration.",
        "1 changed paths classified as automation_or_deployment.",
        "1 changed paths classified as dependency_manifest.",
    ]


def test_rename_previous_path_is_fully_classified() -> None:
    renamed = ReviewCostChangedFile(
        path="src/new_name.py",
        previous_path="security/auth.py",
        change_type=ReviewCostChangeType.RENAMED,
        additions=1,
        deletions=1,
    )

    actual, evidence, _ = score(
        input_model(
            files=[renamed],
            protected_changed_paths=["security/auth.py"],
        ),
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    )

    assert actual == 100
    assert evidence[:2] == [
        "1 changed paths classified as protected.",
        "1 changed paths classified as security_sensitive.",
    ]
    assert "protected: security/auth.py" in evidence


def test_risk_evidence_bound_does_not_change_score() -> None:
    selected = input_model(
        files=[file("security/auth/token.py")],
        protected_changed_paths=["security/auth/token.py"],
    )
    full, _, _ = score(
        selected,
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
    )
    bounded, evidence, _ = score(
        selected,
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
        evidence_limit=1,
    )
    assert bounded == full == 100
    assert evidence == ["1 changed paths classified as protected."]


@pytest.mark.parametrize(
    ("updates", "minimum"),
    [
        ({"merge_conflict": True}, 100),
        ({"required_checks_total": 1, "required_checks_failed": 1}, 90),
        ({"draft": True}, 70),
        ({"required_checks_total": 1, "required_checks_pending": 1}, 60),
        ({"changes_requested_count": 1}, 60),
    ],
)
def test_each_validation_gap(updates: dict[str, Any], minimum: int) -> None:
    actual, evidence, _ = score(
        input_model(files=[file("src/app.py")], **updates),
        ReviewCostSignal.VALIDATION_GAPS,
    )
    assert actual >= minimum
    assert evidence


def test_validation_multiple_gaps_cap_and_exact_reducers() -> None:
    selected = input_model(
        files=[file("src/app.py"), file("tests/test_app.py")],
        required_checks_total=2,
        required_checks_passed=2,
        approval_count=1,
    )
    actual, evidence, reducers = score(
        selected,
        ReviewCostSignal.VALIDATION_GAPS,
    )
    assert actual == 0
    assert reducers == [
        "All required checks passed.",
        "Test changes accompany production changes.",
        "At least one approval is present without an active changes-requested review.",
    ]
    assert evidence[-1].startswith("Reduced 10 points")


def test_validation_reducers_clamp_zero() -> None:
    actual, _, _ = score(
        input_model(
            files=[file("docs/readme.md")],
            required_checks_total=1,
            required_checks_passed=1,
            approval_count=1,
        ),
        ReviewCostSignal.VALIDATION_GAPS,
    )
    assert actual == 0


@pytest.mark.parametrize(
    ("matches", "expected"),
    [(0, 0), (1, 15), (4, 15), (5, 35), (9, 35), (10, 50)],
)
def test_history_match_count_bands(matches: int, expected: int) -> None:
    context = ReviewCostHistoricalContext(
        related_match_count=matches,
        rejected_or_superseded_count=0,
        high_significance_count=0,
        unresolved_count=0,
        source_history_complete=True,
        ranking_coverage_complete=True,
        result_truncated=False,
    )
    actual, _, _ = score(
        input_model(historical_context=context),
        ReviewCostSignal.HISTORICAL_COMPLEXITY,
    )
    assert actual == expected


def test_history_factors_increment_cap_and_no_context() -> None:
    context = ReviewCostHistoricalContext(
        related_match_count=10,
        rejected_or_superseded_count=1,
        high_significance_count=1,
        unresolved_count=1,
        source_history_complete=True,
        ranking_coverage_complete=True,
        result_truncated=False,
    )
    actual, evidence, _ = score(
        input_model(historical_context=context),
        ReviewCostSignal.HISTORICAL_COMPLEXITY,
    )
    assert actual == 95
    assert evidence == [
        "10 related historical matches.",
        "1 rejected or superseded matches.",
        "1 high-significance matches.",
        "1 unresolved matches.",
    ]
    no_context, no_evidence, _ = score(
        input_model(historical_context=None),
        ReviewCostSignal.HISTORICAL_COMPLEXITY,
    )
    assert no_context == 0
    assert no_evidence == []


def test_warning_order_contributions_immutability_and_determinism() -> None:
    selected = input_model(
        policy_present=False,
        merge_conflict=None,
        historical_context=ReviewCostHistoricalContext(
            related_match_count=0,
            rejected_or_superseded_count=0,
            high_significance_count=0,
            unresolved_count=0,
            source_history_complete=False,
            ranking_coverage_complete=False,
            result_truncated=True,
        ),
    )
    before = deepcopy(selected.model_dump())
    service = ReviewCostAssessmentService()
    first = service.assess(selected, assessed_at=ASSESSED_AT)
    second = service.assess(selected, assessed_at=ASSESSED_AT)

    assert [item.signal for item in first.contributions] == list(ReviewCostSignal)
    assert first.warnings == [
        (
            "Repository policy was unavailable; the default preferred diff "
            "size and no protected paths were used."
        ),
        "Pull-request merge-conflict state was unavailable.",
        "Historical source collection was incomplete.",
        "Related-work ranking coverage was incomplete.",
        "Related-work results were truncated by the configured final result limit.",
    ]
    assert first.complete is False
    assert first == second
    assert selected.model_dump() == before


def test_unavailable_history_warning_is_exact_and_complete_case_has_none() -> None:
    unavailable = ReviewCostAssessmentService().assess(
        input_model(historical_context=None),
        assessed_at=ASSESSED_AT,
    )
    assert unavailable.warnings == [
        "Historical related-work context was unavailable."
    ]
    complete = ReviewCostAssessmentService().assess(
        input_model(files=[file("docs/readme.md")]),
        assessed_at=ASSESSED_AT,
    )
    assert complete.warnings == []
    assert complete.complete is True
