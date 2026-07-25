"""Behavior tests for deterministic attention and brief services."""

from copy import deepcopy

import pytest

from opensteward.maintainer_brief import (
    MaintainerAttentionReasonKind,
    MaintainerAttentionRecommendation,
    MaintainerAttentionService,
    MaintainerBriefService,
    MaintainerReviewRoute,
)
from tests.maintainer_brief import brief_input


@pytest.mark.parametrize(
    ("readiness", "kind", "explanation"),
    [
        (
            {
                "policy_blocked": True,
                "policy_attention_required": True,
                "policy_finding_count": 1,
            },
            MaintainerAttentionReasonKind.POLICY_BLOCKER,
            "Repository policy reports blocking findings.",
        ),
        (
            {"merge_conflict": True},
            MaintainerAttentionReasonKind.MERGE_CONFLICT,
            "The pull request has a merge conflict.",
        ),
        (
            {
                "required_checks_total": 1,
                "required_checks_passed": 0,
                "required_checks_failed": 1,
            },
            MaintainerAttentionReasonKind.FAILED_REQUIRED_CHECKS,
            "1 required checks failed.",
        ),
        (
            {"changes_requested_count": 2},
            MaintainerAttentionReasonKind.CHANGES_REQUESTED,
            "2 active changes-requested reviews remain.",
        ),
        (
            {"draft": True},
            MaintainerAttentionReasonKind.DRAFT_PULL_REQUEST,
            "The pull request is still marked as draft.",
        ),
        (
            {
                "required_checks_total": 1,
                "required_checks_passed": 0,
                "required_checks_pending": 1,
            },
            MaintainerAttentionReasonKind.PENDING_REQUIRED_CHECKS,
            "1 required checks are pending.",
        ),
    ],
)
def test_blocking_reasons_are_exact_and_take_precedence(
    readiness: dict[str, object],
    kind: MaintainerAttentionReasonKind,
    explanation: str,
) -> None:
    selected = brief_input(score=80, readiness_updates=readiness)
    attention = MaintainerAttentionService().assess(selected)
    reason = next(item for item in attention.reasons if item.kind == kind)

    assert reason.explanation == explanation
    assert reason.blocking is True
    assert (
        attention.recommendation
        == MaintainerAttentionRecommendation.AUTHOR_ACTION_FIRST
    )


@pytest.mark.parametrize(
    ("path", "route", "kind"),
    [
        (
            {"security_sensitive_path_count": 1},
            MaintainerReviewRoute.SECURITY,
            MaintainerAttentionReasonKind.SECURITY_SENSITIVE_PATHS,
        ),
        (
            {"database_migration_path_count": 1},
            MaintainerReviewRoute.DATABASE,
            MaintainerAttentionReasonKind.DATABASE_MIGRATION,
        ),
        (
            {"automation_or_deployment_path_count": 1},
            MaintainerReviewRoute.RELEASE_OR_DEPLOYMENT,
            MaintainerAttentionReasonKind.AUTOMATION_OR_DEPLOYMENT,
        ),
        (
            {"protected_path_count": 1},
            MaintainerReviewRoute.ARCHITECTURE,
            MaintainerAttentionReasonKind.PROTECTED_PATHS,
        ),
    ],
)
def test_specialist_routes_are_structured_without_general(
    path: dict[str, int],
    route: MaintainerReviewRoute,
    kind: MaintainerAttentionReasonKind,
) -> None:
    attention = MaintainerAttentionService().assess(
        brief_input(path_updates=path)
    )

    assert route in attention.routes
    assert MaintainerReviewRoute.GENERAL not in attention.routes
    assert kind in [reason.kind for reason in attention.reasons]
    assert (
        attention.recommendation
        == MaintainerAttentionRecommendation.HIGH_PRIORITY_REVIEW
    )


def test_review_cost_and_history_recommendation_precedence() -> None:
    critical = MaintainerAttentionService().assess(brief_input(score=80))
    assert (
        critical.recommendation
        == MaintainerAttentionRecommendation.IMMEDIATE_REVIEW
    )
    assert (
        critical.reasons[0].kind
        == MaintainerAttentionReasonKind.CRITICAL_REVIEW_COST
    )

    high = MaintainerAttentionService().assess(brief_input(score=60))
    assert (
        high.recommendation
        == MaintainerAttentionRecommendation.HIGH_PRIORITY_REVIEW
    )

    historical = MaintainerAttentionService().assess(
        brief_input(
            related_count=1,
            history_updates={
                "rejected_or_superseded_count": 1,
                "high_significance_count": 1,
            },
        )
    )
    assert (
        historical.recommendation
        == MaintainerAttentionRecommendation.IMMEDIATE_REVIEW
    )
    assert historical.routes == [MaintainerReviewRoute.ARCHITECTURE]


def test_unresolved_history_routes_architecture_at_three() -> None:
    attention = MaintainerAttentionService().assess(
        brief_input(
            related_count=3,
            history_updates={"unresolved_count": 3},
        )
    )
    assert attention.routes == [MaintainerReviewRoute.ARCHITECTURE]
    assert (
        attention.recommendation
        == MaintainerAttentionRecommendation.HIGH_PRIORITY_REVIEW
    )


def test_routine_brief_actions_warning_order_and_determinism() -> None:
    selected = brief_input()
    before = deepcopy(selected.model_dump())
    service = MaintainerBriefService()

    first = service.build(selected)
    second = service.build(selected)

    assert first == second
    assert first.recommended_actions == ["Proceed with normal maintainer review."]
    assert first.attention.reasons == []
    assert first.attention.routes == [MaintainerReviewRoute.GENERAL]
    assert selected.model_dump() == before


def test_brief_builds_every_ordered_action_and_stable_warnings() -> None:
    selected = brief_input(
        score=80,
        related_count=1,
        readiness_updates={
            "policy_present": False,
            "policy_blocked": True,
            "policy_attention_required": True,
            "policy_finding_count": 1,
            "draft": True,
            "merge_conflict": True,
            "required_checks_total": 2,
            "required_checks_passed": 0,
            "required_checks_failed": 1,
            "required_checks_pending": 1,
            "changes_requested_count": 1,
        },
        path_updates={
            "changed_path_count": 4,
            "security_sensitive_path_count": 1,
            "database_migration_path_count": 1,
            "automation_or_deployment_path_count": 1,
            "protected_path_count": 1,
        },
        history_updates={
            "rejected_or_superseded_count": 1,
            "unresolved_count": 1,
            "source_history_complete": False,
        },
        review_warnings=[
            (
                "Repository policy was unavailable; the default preferred diff "
                "size and no protected paths were used."
            )
        ],
    )

    brief = MaintainerBriefService().build(selected)

    assert brief.recommended_actions == [
        "Address blocking repository-policy findings before maintainer review.",
        "Resolve merge conflicts and refresh the assessment.",
        "Fix failed required checks before maintainer review.",
        (
            "Address active changes-requested reviews before requesting another "
            "maintainer pass."
        ),
        "Mark the pull request ready after the proposed change is complete.",
        "Wait for required checks to complete before maintainer review.",
        "Request a security-focused review.",
        "Request a database or migration review.",
        "Request a deployment or release-engineering review.",
        "Compare the proposal with related architectural decisions.",
        "Consider splitting the pull request into smaller reviewable changes.",
        (
            "Explain how the proposal differs from rejected or superseded "
            "related work."
        ),
        "Resolve or acknowledge unresolved related historical work.",
        "Inspect the reported coverage warnings before relying on this brief.",
    ]
    assert brief.warnings == [
        (
            "Repository policy was unavailable; the default preferred diff "
            "size and no protected paths were used."
        ),
        "Historical source collection was incomplete.",
    ]
