"""Tests for the offline maintainer-brief behavioral benchmark."""

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.maintainer_brief import (
    MAX_MAINTAINER_EVALUATION_CASES,
    MaintainerAttentionReasonKind,
    MaintainerAttentionRecommendation,
    MaintainerBriefEvaluationCase,
    MaintainerBriefEvaluationExpectation,
    MaintainerBriefEvaluationReport,
    MaintainerBriefService,
    MaintainerReviewRoute,
    evaluate_maintainer_brief_cases,
)
from tests.maintainer_brief import brief_input


def case(
    case_id: str,
    recommendation: MaintainerAttentionRecommendation,
    *,
    input_updates: dict[str, Any] | None = None,
    routes: list[MaintainerReviewRoute] | None = None,
    reasons: list[MaintainerAttentionReasonKind] | None = None,
    actions: list[str] | None = None,
) -> MaintainerBriefEvaluationCase:
    return MaintainerBriefEvaluationCase(
        case_id=case_id,
        description=f"Representative behavior for {case_id}.",
        brief_input=brief_input(**(input_updates or {})),
        expectation=MaintainerBriefEvaluationExpectation(
            recommendation=recommendation,
            required_routes=routes or [],
            required_reason_kinds=reasons or [],
            required_actions=actions or [],
        ),
    )


def representative_cases() -> list[MaintainerBriefEvaluationCase]:
    author = MaintainerAttentionRecommendation.AUTHOR_ACTION_FIRST
    high = MaintainerAttentionRecommendation.HIGH_PRIORITY_REVIEW
    immediate = MaintainerAttentionRecommendation.IMMEDIATE_REVIEW
    routine = MaintainerAttentionRecommendation.ROUTINE_REVIEW
    return [
        case(
            "01-docs-routine",
            routine,
            routes=[MaintainerReviewRoute.GENERAL],
            actions=["Proceed with normal maintainer review."],
        ),
        case(
            "02-draft",
            author,
            input_updates={"readiness_updates": {"draft": True}},
            reasons=[MaintainerAttentionReasonKind.DRAFT_PULL_REQUEST],
            actions=[
                "Mark the pull request ready after the proposed change is complete."
            ],
        ),
        case(
            "03-merge-conflict",
            author,
            input_updates={"readiness_updates": {"merge_conflict": True}},
            reasons=[MaintainerAttentionReasonKind.MERGE_CONFLICT],
        ),
        case(
            "04-failed-check",
            author,
            input_updates={
                "readiness_updates": {
                    "required_checks_total": 1,
                    "required_checks_passed": 0,
                    "required_checks_failed": 1,
                }
            },
            reasons=[MaintainerAttentionReasonKind.FAILED_REQUIRED_CHECKS],
        ),
        case(
            "05-changes-requested",
            author,
            input_updates={
                "readiness_updates": {"changes_requested_count": 1}
            },
            reasons=[MaintainerAttentionReasonKind.CHANGES_REQUESTED],
        ),
        case(
            "06-high-large",
            high,
            input_updates={"score": 60},
            reasons=[MaintainerAttentionReasonKind.HIGH_REVIEW_COST],
            actions=[
                "Consider splitting the pull request into smaller reviewable changes."
            ],
        ),
        case(
            "07-critical",
            immediate,
            input_updates={"score": 80},
            reasons=[MaintainerAttentionReasonKind.CRITICAL_REVIEW_COST],
        ),
        case(
            "08-security-high",
            immediate,
            input_updates={
                "score": 60,
                "path_updates": {"security_sensitive_path_count": 1},
            },
            routes=[MaintainerReviewRoute.SECURITY],
            reasons=[
                MaintainerAttentionReasonKind.HIGH_REVIEW_COST,
                MaintainerAttentionReasonKind.SECURITY_SENSITIVE_PATHS,
            ],
            actions=["Request a security-focused review."],
        ),
        case(
            "09-database",
            high,
            input_updates={
                "path_updates": {"database_migration_path_count": 1}
            },
            routes=[MaintainerReviewRoute.DATABASE],
            actions=["Request a database or migration review."],
        ),
        case(
            "10-deployment",
            high,
            input_updates={
                "path_updates": {
                    "automation_or_deployment_path_count": 1
                }
            },
            routes=[MaintainerReviewRoute.RELEASE_OR_DEPLOYMENT],
            actions=["Request a deployment or release-engineering review."],
        ),
        case(
            "11-rejected-significant",
            immediate,
            input_updates={
                "related_count": 1,
                "history_updates": {
                    "rejected_or_superseded_count": 1,
                    "high_significance_count": 1,
                },
            },
            routes=[MaintainerReviewRoute.ARCHITECTURE],
            reasons=[
                MaintainerAttentionReasonKind.REJECTED_OR_SUPERSEDED_HISTORY,
                MaintainerAttentionReasonKind.HIGH_SIGNIFICANCE_HISTORY,
            ],
        ),
        case(
            "12-incomplete",
            routine,
            input_updates={
                "history_updates": {"source_history_complete": False}
            },
            reasons=[MaintainerAttentionReasonKind.INCOMPLETE_EVIDENCE],
            actions=[
                "Inspect the reported coverage warnings before relying on this brief."
            ],
        ),
        case(
            "13-unresolved",
            high,
            input_updates={
                "related_count": 3,
                "history_updates": {"unresolved_count": 3},
            },
            routes=[MaintainerReviewRoute.ARCHITECTURE],
            reasons=[MaintainerAttentionReasonKind.UNRESOLVED_HISTORY],
        ),
        case(
            "14-approved-tested-routine",
            routine,
            routes=[MaintainerReviewRoute.GENERAL],
        ),
    ]


def test_representative_cases_pass_at_ten_thousand_basis_points() -> None:
    cases = representative_cases()
    before = deepcopy(cases)

    report = evaluate_maintainer_brief_cases(cases)

    assert len(cases) == 14
    assert report.total_cases == report.passed_cases == 14
    assert report.failed_cases == 0
    assert report.pass_rate_basis_points == 10_000
    assert cases == before
    assert [result.case_id for result in report.results] == sorted(
        item.case_id for item in cases
    )


def test_expectations_are_strict_unique_and_ordered() -> None:
    with pytest.raises(ValidationError):
        MaintainerBriefEvaluationExpectation(
            recommendation="routine_review",
            required_routes=[],
            required_reason_kinds=[],
            required_actions=[],
            extra=True,
        )
    with pytest.raises(ValidationError, match="unique"):
        MaintainerBriefEvaluationExpectation(
            recommendation="routine_review",
            required_routes=[
                MaintainerReviewRoute.GENERAL,
                MaintainerReviewRoute.GENERAL,
            ],
            required_reason_kinds=[],
            required_actions=[],
        )
    with pytest.raises(ValidationError, match="exact enum order"):
        MaintainerBriefEvaluationExpectation(
            recommendation="high_priority_review",
            required_routes=[
                MaintainerReviewRoute.ARCHITECTURE,
                MaintainerReviewRoute.SECURITY,
            ],
            required_reason_kinds=[],
            required_actions=[],
        )


def test_duplicate_and_excessive_cases_are_rejected() -> None:
    selected = representative_cases()[0]
    with pytest.raises(ValueError, match="IDs must be unique"):
        evaluate_maintainer_brief_cases([selected, selected])
    with pytest.raises(ValueError, match="case limit"):
        evaluate_maintainer_brief_cases(
            [selected] * (MAX_MAINTAINER_EVALUATION_CASES + 1)
        )


def test_mismatches_fail_and_extra_actual_evidence_is_allowed() -> None:
    selected = representative_cases()[7]
    expectation = selected.expectation.model_copy(
        update={
            "recommendation": MaintainerAttentionRecommendation.ROUTINE_REVIEW,
            "required_routes": [
                MaintainerReviewRoute.SECURITY,
                MaintainerReviewRoute.DATABASE,
            ],
            "required_reason_kinds": [
                MaintainerAttentionReasonKind.HIGH_REVIEW_COST,
                MaintainerAttentionReasonKind.DATABASE_MIGRATION,
            ],
            "required_actions": [
                "Request a security-focused review.",
                "request a database or migration review.",
            ],
        }
    )
    mismatched = selected.model_copy(update={"expectation": expectation})

    result = evaluate_maintainer_brief_cases([mismatched]).results[0]

    assert result.passed is False
    assert result.missing_routes == [MaintainerReviewRoute.DATABASE]
    assert result.missing_reason_kinds == [
        MaintainerAttentionReasonKind.DATABASE_MIGRATION
    ]
    assert result.missing_actions == [
        "request a database or migration review."
    ]

    passing = evaluate_maintainer_brief_cases([selected]).results[0]
    assert passing.passed is True


def test_service_is_invoked_once_per_sorted_case() -> None:
    class RecordingService:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.delegate = MaintainerBriefService()

        def build(self, selected_input):
            self.calls.append(selected_input.pull_request.external_id)
            return self.delegate.build(selected_input)

    service = RecordingService()
    cases = list(reversed(representative_cases()[:2]))

    report = evaluate_maintainer_brief_cases(cases, service=service)

    assert len(service.calls) == 2
    assert [result.case_id for result in report.results] == [
        "01-docs-routine",
        "02-draft",
    ]


def test_empty_report_and_half_up_pass_rate() -> None:
    empty = evaluate_maintainer_brief_cases([])
    assert empty.pass_rate_basis_points == 10_000

    results = evaluate_maintainer_brief_cases(representative_cases()[:3]).results
    one_failed = results[0].model_copy(
        update={
            "passed": False,
            "expected_recommendation": (
                MaintainerAttentionRecommendation.IMMEDIATE_REVIEW
            ),
        }
    )
    report = MaintainerBriefEvaluationReport(
        total_cases=3,
        passed_cases=2,
        failed_cases=1,
        results=[one_failed, *results[1:]],
    )
    assert report.pass_rate_basis_points == 6667
