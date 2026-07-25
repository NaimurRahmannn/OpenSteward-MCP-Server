"""Tests for strict provider-independent maintainer-brief models."""

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from opensteward.maintainer_brief import (
    MaintainerAttentionAssessment,
    MaintainerAttentionReason,
    MaintainerAttentionReasonKind,
    MaintainerAttentionRecommendation,
    MaintainerBrief,
    MaintainerBriefService,
    MaintainerPathRiskSummary,
    MaintainerReadinessSummary,
    MaintainerReviewRoute,
)
from tests.maintainer_brief import (
    ASSESSED_AT,
    PULL_REQUEST,
    REPOSITORY,
    brief_input,
)


def test_readiness_accounting_policy_and_computed_states() -> None:
    readiness = MaintainerReadinessSummary(
        policy_present=True,
        policy_blocked=False,
        policy_attention_required=True,
        policy_finding_count=0,
        draft=False,
        merge_conflict=False,
        required_checks_total=2,
        required_checks_passed=2,
        required_checks_failed=0,
        required_checks_pending=0,
        approval_count=1,
        changes_requested_count=0,
    )

    assert readiness.all_required_checks_passed is True
    assert readiness.has_readiness_blocker is False
    with pytest.raises(ValidationError, match="subcounts"):
        readiness.model_copy(
            update={"required_checks_failed": 1}
        ).model_validate(
            {
                **readiness.model_dump(exclude_computed_fields=True),
                "required_checks_failed": 1,
            }
        )
    with pytest.raises(ValidationError, match="requires"):
        MaintainerReadinessSummary(
            **{
                **readiness.model_dump(exclude_computed_fields=True),
                "policy_blocked": True,
                "policy_attention_required": False,
                "policy_finding_count": 1,
            }
        )


def test_path_risk_bounds_and_computed_states() -> None:
    summary = MaintainerPathRiskSummary(
        changed_path_count=4,
        protected_path_count=1,
        security_sensitive_path_count=1,
        database_migration_path_count=0,
        automation_or_deployment_path_count=1,
        dependency_manifest_path_count=1,
    )

    assert summary.specialist_risk_count == 3
    assert summary.has_specialist_risk is True
    with pytest.raises(ValidationError, match="must not exceed"):
        MaintainerPathRiskSummary(
            changed_path_count=1,
            protected_path_count=2,
            security_sensitive_path_count=0,
            database_migration_path_count=0,
            automation_or_deployment_path_count=0,
            dependency_manifest_path_count=0,
        )


@pytest.mark.parametrize(
    ("kind", "blocking"),
    [
        (MaintainerAttentionReasonKind.POLICY_BLOCKER, True),
        (MaintainerAttentionReasonKind.HIGH_REVIEW_COST, False),
    ],
)
def test_reason_blocking_is_exact(
    kind: MaintainerAttentionReasonKind,
    blocking: bool,
) -> None:
    reason = MaintainerAttentionReason(
        kind=kind,
        explanation="Evidence.",
        blocking=blocking,
    )
    assert reason.blocking is blocking
    with pytest.raises(ValidationError, match="blocking flag"):
        MaintainerAttentionReason(
            kind=kind,
            explanation="Evidence.",
            blocking=not blocking,
        )


def test_attention_identity_order_routes_and_utc_normalization() -> None:
    offset = timezone(timedelta(hours=6))
    assessment = MaintainerAttentionAssessment(
        repository=REPOSITORY,
        pull_request=PULL_REQUEST,
        assessed_at=datetime(2026, 6, 1, 6, tzinfo=offset),
        recommendation=MaintainerAttentionRecommendation.HIGH_PRIORITY_REVIEW,
        routes=[
            MaintainerReviewRoute.SECURITY,
            MaintainerReviewRoute.ARCHITECTURE,
        ],
        reasons=[
            MaintainerAttentionReason(
                kind=MaintainerAttentionReasonKind.PROTECTED_PATHS,
                explanation="1 protected paths changed.",
                blocking=False,
            ),
            MaintainerAttentionReason(
                kind=MaintainerAttentionReasonKind.SECURITY_SENSITIVE_PATHS,
                explanation="1 security-sensitive paths changed.",
                blocking=False,
            ),
        ],
        warnings=[],
    )

    assert assessment.assessed_at == ASSESSED_AT
    assert assessment.has_specialist_route is True
    assert assessment.ready_for_review is True
    assert assessment.requires_author_action is False
    assert assessment.complete is True
    with pytest.raises(ValidationError, match="exact route order"):
        MaintainerAttentionAssessment(
            **{
                **assessment.model_dump(exclude_computed_fields=True),
                "routes": [
                    MaintainerReviewRoute.ARCHITECTURE,
                    MaintainerReviewRoute.SECURITY,
                ],
            }
        )
    with pytest.raises(ValidationError, match="only route"):
        MaintainerAttentionAssessment(
            **{
                **assessment.model_dump(exclude_computed_fields=True),
                "routes": [
                    MaintainerReviewRoute.SECURITY,
                    MaintainerReviewRoute.GENERAL,
                ],
            }
        )


def test_recommendation_consistency_and_strictness() -> None:
    selected = brief_input()
    routine = MaintainerAttentionAssessment(
        repository=REPOSITORY,
        pull_request=PULL_REQUEST,
        assessed_at=ASSESSED_AT,
        recommendation=MaintainerAttentionRecommendation.ROUTINE_REVIEW,
        routes=[MaintainerReviewRoute.GENERAL],
        reasons=[],
        warnings=[],
    )
    assert routine.ready_for_review is True
    with pytest.raises(ValidationError):
        MaintainerAttentionAssessment(
            **{
                **routine.model_dump(exclude_computed_fields=True),
                "extra": True,
            }
        )
    with pytest.raises(ValidationError, match="blocking reason"):
        MaintainerAttentionAssessment(
            **{
                **routine.model_dump(exclude_computed_fields=True),
                "recommendation": (
                    MaintainerAttentionRecommendation.AUTHOR_ACTION_FIRST
                ),
            }
        )
    with pytest.raises(ValidationError, match="forbids elevated"):
        MaintainerAttentionAssessment(
            **{
                **routine.model_dump(exclude_computed_fields=True),
                "reasons": [
                    MaintainerAttentionReason(
                        kind=MaintainerAttentionReasonKind.HIGH_REVIEW_COST,
                        explanation="High.",
                        blocking=False,
                    )
                ],
            }
        )
    assert selected.repository == REPOSITORY


def test_brief_exact_warnings_computed_fields_and_json_enums() -> None:
    selected = brief_input(
        review_warnings=["shared warning"],
        readiness_updates={
            "policy_present": False,
            "merge_conflict": None,
        },
        history_updates={"source_history_complete": False},
    )
    brief = MaintainerBriefService().build(selected)

    assert brief.warnings == [
        "shared warning",
        (
            "Repository policy was unavailable; the default preferred diff "
            "size and no protected paths were used."
        ),
        "Pull-request merge-conflict state was unavailable.",
        "Historical source collection was incomplete.",
    ]
    assert brief.recommendation == MaintainerAttentionRecommendation.ROUTINE_REVIEW
    assert brief.routes == [MaintainerReviewRoute.GENERAL]
    assert brief.review_cost_score == 10
    assert brief.related_match_count == 0
    assert brief.complete is False
    data = brief.model_dump(mode="json")
    assert data["recommendation"] == "routine_review"
    assert data["routes"] == ["general"]

    with pytest.raises(ValidationError, match="Warnings must equal"):
        MaintainerBrief(
            **{
                **brief.model_dump(exclude={"warnings"}, exclude_computed_fields=True),
                "warnings": [],
            }
        )
