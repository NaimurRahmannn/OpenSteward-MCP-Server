"""Validated contribution facts consumed by the policy engine."""

from pydantic import Field, computed_field, field_validator

from opensteward.policy.models import (
    ContributionCategory,
    StrictPolicyModel,
)
from opensteward.policy.path_matcher import normalize_repository_path


class ContributionPolicyInput(StrictPolicyModel):
    """Repository facts used when evaluating contribution policy."""

    changed_files: list[str] = Field(
        min_length=1,
        max_length=10_000,
    )

    additions: int = Field(
        default=0,
        ge=0,
        le=10_000_000,
    )

    deletions: int = Field(
        default=0,
        ge=0,
        le=10_000_000,
    )

    categories: list[ContributionCategory] = Field(
        default_factory=list,
    )

    linked_issue_numbers: list[int] = Field(
        default_factory=list,
    )

    tests_changed: bool = False

    current_approvals: int = Field(
        default=0,
        ge=0,
        le=100,
    )

    @field_validator("changed_files")
    @classmethod
    def normalize_changed_files(
        cls,
        paths: list[str],
    ) -> list[str]:
        """Normalize changed files and reject duplicate paths."""

        normalized_paths = [
            normalize_repository_path(path)
            for path in paths
        ]

        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError(
                "Changed files must not contain duplicate paths."
            )

        return normalized_paths

    @field_validator("categories")
    @classmethod
    def reject_duplicate_categories(
        cls,
        categories: list[ContributionCategory],
    ) -> list[ContributionCategory]:
        """Reject repeated contribution categories."""

        if len(categories) != len(set(categories)):
            raise ValueError(
                "Contribution categories must not contain duplicates."
            )

        return categories

    @field_validator("linked_issue_numbers")
    @classmethod
    def validate_linked_issue_numbers(
        cls,
        issue_numbers: list[int],
    ) -> list[int]:
        """Require positive and unique issue numbers."""

        if any(number <= 0 for number in issue_numbers):
            raise ValueError(
                "Linked issue numbers must be positive integers."
            )

        if len(issue_numbers) != len(set(issue_numbers)):
            raise ValueError(
                "Linked issue numbers must not contain duplicates."
            )

        return issue_numbers

    @computed_field
    @property
    def diff_lines(self) -> int:
        """Return the total number of added and deleted lines."""

        return self.additions + self.deletions

    @computed_field
    @property
    def has_linked_issue(self) -> bool:
        """Return whether the contribution references an issue."""

        return bool(self.linked_issue_numbers)