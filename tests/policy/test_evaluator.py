"""Tests for repository contribution policy evaluation."""

import pytest

from opensteward.policy import (
    ContributionCategory,
    ContributionPolicyInput,
    FindingSeverity,
    PolicyFindingStatus,
    PolicyRule,
    ProtectedPathRule,
    RepositoryPolicy,
    RiskLevel,
    evaluate_contribution_policy,
)


def find_finding(
    result,
    rule: PolicyRule,
):
    """Return the first finding for a rule."""

    return next(
        finding
        for finding in result.findings
        if finding.rule == rule
    )


def test_compliant_contribution_passes_policy() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "pull_requests": {
                "linked_issue_required_for": [
                    "public_api",
                ],
                "tests_required_for": [
                    "public_api",
                ],
            },
            "review": {
                "required_approvals": {
                    "default": 1,
                    "public_api": 2,
                    "security": 2,
                }
            },
        }
    )

    contribution = ContributionPolicyInput(
        changed_files=[
            "src/api/client.py",
            "tests/test_client.py",
        ],
        additions=100,
        deletions=20,
        categories=[
            ContributionCategory.PUBLIC_API,
        ],
        linked_issue_numbers=[421],
        tests_changed=True,
        current_approvals=2,
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    assert result.compliant is True
    assert result.required_approvals == 2
    assert result.remaining_approvals == 0
    assert result.highest_protected_path_risk is None


def test_large_diff_creates_warning_not_failure() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "pull_requests": {
                "preferred_maximum_diff_lines": 100,
                "tests_required_for": [],
            },
            "review": {
                "required_approvals": {
                    "default": 0,
                    "public_api": 0,
                    "security": 0,
                }
            },
        }
    )

    contribution = ContributionPolicyInput(
        changed_files=["src/application.py"],
        additions=150,
        deletions=20,
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    finding = find_finding(
        result,
        PolicyRule.PREFERRED_DIFF_SIZE,
    )

    assert finding.status == PolicyFindingStatus.WARNING
    assert finding.severity == FindingSeverity.MEDIUM
    assert result.compliant is True


def test_missing_required_tests_fails_policy() -> None:
    policy = RepositoryPolicy()

    contribution = ContributionPolicyInput(
        changed_files=["src/application.py"],
        categories=[
            ContributionCategory.BUG_FIX,
        ],
        tests_changed=False,
        current_approvals=1,
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    finding = find_finding(
        result,
        PolicyRule.REQUIRED_TESTS,
    )

    assert finding.status == PolicyFindingStatus.FAILED
    assert finding.severity == FindingSeverity.HIGH
    assert result.compliant is False


def test_missing_required_linked_issue_fails_policy() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "pull_requests": {
                "linked_issue_required_for": [
                    "architecture",
                ],
                "tests_required_for": [],
            }
        }
    )

    contribution = ContributionPolicyInput(
        changed_files=["src/architecture.py"],
        categories=[
            ContributionCategory.ARCHITECTURE,
        ],
        current_approvals=1,
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    finding = find_finding(
        result,
        PolicyRule.LINKED_ISSUE,
    )

    assert finding.status == PolicyFindingStatus.FAILED
    assert result.compliant is False


def test_protected_path_returns_structured_evidence() -> None:
    policy = RepositoryPolicy(
        protected_paths=[
            ProtectedPathRule(
                pattern="src/security/**",
                risk=RiskLevel.CRITICAL,
                human_review_required=True,
            )
        ]
    )

    contribution = ContributionPolicyInput(
        changed_files=["src/security/auth.py"],
        current_approvals=1,
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    findings = [
        finding
        for finding in result.findings
        if finding.rule == PolicyRule.PROTECTED_PATH
    ]

    assert len(findings) == 1
    assert findings[0].status == PolicyFindingStatus.WARNING
    assert findings[0].severity == FindingSeverity.CRITICAL
    assert "path:src/security/auth.py" in findings[0].evidence
    assert result.requires_human_review is True
    assert (
        result.highest_protected_path_risk
        == RiskLevel.CRITICAL
    )


def test_highest_protected_path_risk_is_selected() -> None:
    policy = RepositoryPolicy(
        protected_paths=[
            ProtectedPathRule(
                pattern="src/**",
                risk=RiskLevel.MEDIUM,
            ),
            ProtectedPathRule(
                pattern="src/security/**",
                risk=RiskLevel.CRITICAL,
            ),
        ]
    )

    contribution = ContributionPolicyInput(
        changed_files=["src/security/auth.py"],
        current_approvals=1,
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    assert (
        result.highest_protected_path_risk
        == RiskLevel.CRITICAL
    )


@pytest.mark.parametrize(
    ("categories", "expected_approvals"),
    [
        ([], 1),
        ([ContributionCategory.BUG_FIX], 1),
        ([ContributionCategory.PUBLIC_API], 2),
        ([ContributionCategory.SECURITY], 3),
        (
            [
                ContributionCategory.PUBLIC_API,
                ContributionCategory.SECURITY,
            ],
            3,
        ),
    ],
)
def test_effective_approval_requirement(
    categories: list[ContributionCategory],
    expected_approvals: int,
) -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "pull_requests": {
                "tests_required_for": [],
            },
            "review": {
                "required_approvals": {
                    "default": 1,
                    "public_api": 2,
                    "security": 3,
                }
            },
        }
    )

    contribution = ContributionPolicyInput(
        changed_files=["src/application.py"],
        categories=categories,
        current_approvals=expected_approvals,
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    assert result.required_approvals == expected_approvals
    assert result.remaining_approvals == 0


def test_missing_approval_fails_policy() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "pull_requests": {
                "tests_required_for": [],
            },
            "review": {
                "required_approvals": {
                    "default": 1,
                    "public_api": 2,
                    "security": 3,
                }
            },
        }
    )

    contribution = ContributionPolicyInput(
        changed_files=["src/security/auth.py"],
        categories=[
            ContributionCategory.SECURITY,
        ],
        current_approvals=1,
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    finding = find_finding(
        result,
        PolicyRule.REQUIRED_APPROVALS,
    )

    assert result.required_approvals == 3
    assert result.remaining_approvals == 2
    assert finding.status == PolicyFindingStatus.FAILED
    assert result.compliant is False


def test_evaluation_serializes_to_json() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "pull_requests": {
                "tests_required_for": [],
            },
            "review": {
                "required_approvals": {
                    "default": 0,
                    "public_api": 0,
                    "security": 0,
                }
            },
        }
    )

    contribution = ContributionPolicyInput(
        changed_files=["src/application.py"],
    )

    result = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    data = result.model_dump(mode="json")

    assert data["compliant"] is True
    assert data["required_approvals"] == 0
    assert data["remaining_approvals"] == 0
    assert data["highest_protected_path_risk"] is None
    assert isinstance(data["findings"], list)