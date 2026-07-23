"""Tests for maintainer-facing policy packets."""

from opensteward.policy import (
    ContributionPolicyInput,
    MaintainerRecommendation,
    PolicyRule,
    RepositoryPolicy,
    build_maintainer_policy_packet,
    evaluate_contribution_policy,
)


def create_security_policy() -> RepositoryPolicy:
    """Create a representative security policy."""

    return RepositoryPolicy.model_validate(
        {
            "pull_requests": {
                "linked_issue_required_for": [
                    "security",
                ],
                "tests_required_for": [
                    "security",
                ],
                "preferred_maximum_diff_lines": 500,
            },
            "protected_paths": [
                {
                    "pattern": "src/security/**",
                    "risk": "critical",
                    "human_review_required": True,
                }
            ],
            "review": {
                "required_approvals": {
                    "default": 1,
                    "public_api": 2,
                    "security": 3,
                }
            },
        }
    )


def test_packet_summarizes_blocking_requirements() -> None:
    contribution = ContributionPolicyInput(
        changed_files=[
            "src/security/auth.py",
        ],
        additions=600,
        deletions=100,
        categories=[
            "security",
        ],
        linked_issue_numbers=[],
        tests_changed=False,
        current_approvals=1,
    )

    evaluation = evaluate_contribution_policy(
        policy=create_security_policy(),
        contribution=contribution,
    )

    packet = build_maintainer_policy_packet(evaluation)

    assert (
        packet.recommendation
        == MaintainerRecommendation.REQUEST_CHANGES
    )

    assert packet.ready_for_detailed_review is False

    assert {
        item.rule
        for item in packet.blocking_requirements
    } == {
        PolicyRule.REQUIRED_TESTS,
        PolicyRule.LINKED_ISSUE,
        PolicyRule.REQUIRED_APPROVALS,
    }

    assert {
        item.rule
        for item in packet.warnings
    } == {
        PolicyRule.PREFERRED_DIFF_SIZE,
        PolicyRule.PROTECTED_PATH,
    }

    assert "3 blocking requirements" in packet.summary
    assert "2 warnings" in packet.summary

    assert (
        packet.approval_summary
        == (
            "Approval requirement not met: 1 recorded, "
            "3 required, and 2 remaining."
        )
    )


def test_packet_uses_review_with_caution_for_warning_only() -> None:
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

    evaluation = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    packet = build_maintainer_policy_packet(evaluation)

    assert (
        packet.recommendation
        == MaintainerRecommendation.REVIEW_WITH_CAUTION
    )

    assert packet.ready_for_detailed_review is True
    assert packet.blocking_requirements == []
    assert len(packet.warnings) == 1
    assert packet.warnings[0].rule == PolicyRule.PREFERRED_DIFF_SIZE


def test_packet_marks_clean_contribution_ready() -> None:
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
        additions=20,
        deletions=5,
    )

    evaluation = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    packet = build_maintainer_policy_packet(evaluation)

    assert (
        packet.recommendation
        == MaintainerRecommendation.READY_FOR_REVIEW
    )

    assert packet.ready_for_detailed_review is True
    assert packet.blocking_requirements == []
    assert packet.warnings == []
    assert packet.suggested_next_actions == []

    assert (
        packet.approval_summary
        == "No human approvals are required by the active policy."
    )


def test_packet_deduplicates_suggested_actions() -> None:
    policy = RepositoryPolicy.model_validate(
        {
            "pull_requests": {
                "tests_required_for": [],
            },
            "protected_paths": [
                {
                    "pattern": "src/security/**",
                    "risk": "critical",
                    "human_review_required": True,
                }
            ],
            "review": {
                "required_approvals": {
                    "default": 1,
                    "public_api": 1,
                    "security": 1,
                }
            },
        }
    )

    contribution = ContributionPolicyInput(
        changed_files=[
            "src/security/auth.py",
            "src/security/token.py",
        ],
        current_approvals=1,
    )

    evaluation = evaluate_contribution_policy(
        policy=policy,
        contribution=contribution,
    )

    packet = build_maintainer_policy_packet(evaluation)

    protected_action = (
        "Route the contribution to a maintainer familiar with "
        "this protected area."
    )

    assert (
        packet.suggested_next_actions.count(protected_action)
        == 1
    )


def test_packet_serializes_to_json_values() -> None:
    contribution = ContributionPolicyInput(
        changed_files=[
            "src/security/auth.py",
        ],
        categories=[
            "security",
        ],
        current_approvals=3,
        tests_changed=True,
        linked_issue_numbers=[421],
    )

    evaluation = evaluate_contribution_policy(
        policy=create_security_policy(),
        contribution=contribution,
    )

    packet = build_maintainer_policy_packet(evaluation)
    data = packet.model_dump(mode="json")

    assert data["recommendation"] == "review_with_caution"
    assert data["ready_for_detailed_review"] is True
    assert isinstance(data["blocking_requirements"], list)
    assert isinstance(data["warnings"], list)
    assert isinstance(data["suggested_next_actions"], list)