"""Tests for contribution policy input models."""

import pytest
from pydantic import ValidationError

from opensteward.policy import (
    ContributionCategory,
    ContributionPolicyInput,
)


def create_valid_input() -> ContributionPolicyInput:
    """Create a valid contribution input for tests."""

    return ContributionPolicyInput(
        changed_files=[
            "src/security/auth.py",
            "tests/test_auth.py",
        ],
        additions=120,
        deletions=30,
        categories=[
            ContributionCategory.SECURITY,
            ContributionCategory.BUG_FIX,
        ],
        linked_issue_numbers=[421],
        tests_changed=True,
        current_approvals=1,
    )


def test_contribution_input_accepts_valid_facts() -> None:
    contribution = create_valid_input()

    assert contribution.changed_files == [
        "src/security/auth.py",
        "tests/test_auth.py",
    ]
    assert contribution.additions == 120
    assert contribution.deletions == 30
    assert contribution.diff_lines == 150
    assert contribution.has_linked_issue is True
    assert contribution.tests_changed is True
    assert contribution.current_approvals == 1


def test_contribution_input_has_safe_defaults() -> None:
    contribution = ContributionPolicyInput(
        changed_files=["src/application.py"],
    )

    assert contribution.additions == 0
    assert contribution.deletions == 0
    assert contribution.categories == []
    assert contribution.linked_issue_numbers == []
    assert contribution.has_linked_issue is False
    assert contribution.tests_changed is False
    assert contribution.current_approvals == 0


def test_changed_files_are_normalized() -> None:
    contribution = ContributionPolicyInput(
        changed_files=[
            ".\\src\\security\\auth.py",
            "./tests/test_auth.py",
        ],
    )

    assert contribution.changed_files == [
        "src/security/auth.py",
        "tests/test_auth.py",
    ]


def test_duplicate_normalized_files_are_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="must not contain duplicate paths",
    ):
        ContributionPolicyInput(
            changed_files=[
                "src/security/auth.py",
                "./src/security/auth.py",
            ],
        )


@pytest.mark.parametrize(
    "path",
    [
        "/src/security/auth.py",
        "../security/auth.py",
        "src/../security/auth.py",
        "src//security/auth.py",
        "",
    ],
)
def test_unsafe_changed_files_are_rejected(
    path: str,
) -> None:
    with pytest.raises(ValidationError):
        ContributionPolicyInput(
            changed_files=[path],
        )


def test_at_least_one_changed_file_is_required() -> None:
    with pytest.raises(ValidationError):
        ContributionPolicyInput(
            changed_files=[],
        )


def test_duplicate_categories_are_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="must not contain duplicates",
    ):
        ContributionPolicyInput(
            changed_files=["src/application.py"],
            categories=[
                ContributionCategory.BUG_FIX,
                ContributionCategory.BUG_FIX,
            ],
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("additions", -1),
        ("deletions", -1),
        ("current_approvals", -1),
        ("current_approvals", 101),
    ],
)
def test_invalid_numeric_values_are_rejected(
    field_name: str,
    invalid_value: int,
) -> None:
    values = {
        "changed_files": ["src/application.py"],
        field_name: invalid_value,
    }

    with pytest.raises(ValidationError):
        ContributionPolicyInput.model_validate(values)


@pytest.mark.parametrize(
    "issue_numbers",
    [
        [0],
        [-1],
        [421, 421],
    ],
)
def test_invalid_linked_issue_numbers_are_rejected(
    issue_numbers: list[int],
) -> None:
    with pytest.raises(ValidationError):
        ContributionPolicyInput(
            changed_files=["src/application.py"],
            linked_issue_numbers=issue_numbers,
        )




def test_contribution_input_serializes_computed_fields() -> None:
    contribution = create_valid_input()

    data = contribution.model_dump(mode="json")

    assert data["diff_lines"] == 150
    assert data["has_linked_issue"] is True
    assert data["categories"] == [
        "security",
        "bug_fix",
    ]


def test_contribution_input_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ContributionPolicyInput.model_validate(
            {
                "changed_files": ["src/application.py"],
                "unknown_fact": True,
            }
        )