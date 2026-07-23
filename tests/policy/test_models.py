"""Tests for typed OpenSteward repository policies."""

import pytest
from pydantic import ValidationError

from opensteward.policy import (
    ContributionCategory,
    RepositoryPolicy,
    RiskLevel,
)


def test_repository_policy_has_safe_defaults() -> None:
    policy = RepositoryPolicy()

    assert policy.version == 1

    assert policy.contributions.ai_assistance.allowed is True
    assert (
        policy.contributions.ai_assistance.disclosure_required
        is False
    )

    assert (
        policy.pull_requests.preferred_maximum_diff_lines
        == 500
    )

    assert policy.pull_requests.tests_required_for == [
        ContributionCategory.BUG_FIX,
        ContributionCategory.OBSERVABLE_BEHAVIOR,
        ContributionCategory.PUBLIC_API,
    ]

    assert (
        policy.review.maximum_pending_reviews_per_reviewer
        == 8
    )

    assert policy.review.required_approvals.default == 1
    assert policy.review.required_approvals.public_api == 2
    assert policy.review.required_approvals.security == 2

    assert policy.automation.publish_comments is False
    assert policy.automation.apply_labels is False
    assert policy.automation.require_human_approval is True


def test_repository_policy_accepts_complete_configuration() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "version": 1,
            "contributions": {
                "ai_assistance": {
                    "allowed": True,
                    "disclosure_required": True,
                    "human_attestation_required": True,
                }
            },
            "pull_requests": {
                "linked_issue_required_for": [
                    "public_api",
                    "architecture",
                ],
                "tests_required_for": [
                    "bug_fix",
                    "public_api",
                ],
                "preferred_maximum_diff_lines": 750,
            },
            "protected_paths": [
                {
                    "pattern": "src/security/**",
                    "risk": "critical",
                    "human_review_required": True,
                },
                {
                    "pattern": "docs/**",
                    "risk": "low",
                    "human_review_required": False,
                },
            ],
            "review": {
                "maximum_pending_reviews_per_reviewer": 10,
                "required_approvals": {
                    "default": 1,
                    "public_api": 2,
                    "security": 3,
                },
            },
            "automation": {
                "publish_check_runs": False,
                "publish_comments": False,
                "apply_labels": False,
                "require_human_approval": True,
            },
        }
    )

    assert policy.pull_requests.preferred_maximum_diff_lines == 750
    assert policy.protected_paths[0].risk == RiskLevel.CRITICAL
    assert policy.protected_paths[1].risk == RiskLevel.LOW
    assert policy.review.required_approvals.security == 3


def test_repository_policy_serializes_enums_as_strings() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "protected_paths": [
                {
                    "pattern": "src/security/**",
                    "risk": "critical",
                }
            ]
        }
    )

    data = policy.model_dump(mode="json")

    assert data["version"] == 1
    assert data["protected_paths"][0]["risk"] == "critical"
    assert (
        data["pull_requests"]["tests_required_for"][0]
        == "bug_fix"
    )


def test_repository_policy_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RepositoryPolicy.model_validate(
            {
                "version": 1,
                "unknown_setting": True,
            }
        )


@pytest.mark.parametrize("version", [0, 2, 10])
def test_repository_policy_rejects_unsupported_versions(
    version: int,
) -> None:
    with pytest.raises(ValidationError):
        RepositoryPolicy.model_validate(
            {
                "version": version,
            }
        )


@pytest.mark.parametrize("maximum_lines", [0, -1, 100_001])
def test_pull_request_policy_rejects_invalid_diff_limits(
    maximum_lines: int,
) -> None:
    with pytest.raises(ValidationError):
        RepositoryPolicy.model_validate(
            {
                "pull_requests": {
                    "preferred_maximum_diff_lines": maximum_lines,
                }
            }
        )


def test_policy_rejects_duplicate_contribution_categories() -> None:
    with pytest.raises(
        ValidationError,
        match="must not contain duplicates",
    ):
        RepositoryPolicy.model_validate(
            {
                "pull_requests": {
                    "tests_required_for": [
                        "bug_fix",
                        "bug_fix",
                    ]
                }
            }
        )


def test_policy_rejects_absolute_protected_path() -> None:
    with pytest.raises(
        ValidationError,
        match="repository-relative",
    ):
        RepositoryPolicy.model_validate(
            {
                "protected_paths": [
                    {
                        "pattern": "/src/security/**",
                        "risk": "critical",
                    }
                ]
            }
        )


def test_policy_rejects_duplicate_protected_paths() -> None:
    with pytest.raises(
        ValidationError,
        match="must be unique",
    ):
        RepositoryPolicy.model_validate(
            {
                "protected_paths": [
                    {
                        "pattern": "src/security/**",
                        "risk": "high",
                    },
                    {
                        "pattern": "src/security/**",
                        "risk": "critical",
                    },
                ]
            }
        )


def test_sensitive_approvals_cannot_be_lower_than_default() -> None:
    with pytest.raises(
        ValidationError,
        match="Security approvals cannot be lower",
    ):
        RepositoryPolicy.model_validate(
            {
                "review": {
                    "required_approvals": {
                        "default": 2,
                        "public_api": 2,
                        "security": 1,
                    }
                }
            }
        )
def test_protected_path_pattern_is_normalized() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "protected_paths": [
                {
                    "pattern": ".\\src\\security\\**",
                    "risk": "critical",
                }
            ]
        }
    )

    assert policy.protected_paths[0].pattern == "src/security/**"


@pytest.mark.parametrize(
    "pattern",
    [
        "../security/**",
        "src/../security/**",
        "src//security/**",
        "./../security/**",
    ],
)
def test_policy_rejects_unsafe_protected_path_patterns(
    pattern: str,
) -> None:
    with pytest.raises(ValidationError):
        RepositoryPolicy.model_validate(
            {
                "protected_paths": [
                    {
                        "pattern": pattern,
                        "risk": "critical",
                    }
                ]
            }
        )