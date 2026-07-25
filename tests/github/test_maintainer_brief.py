"""Tests for GitHub adaptation of the provider-neutral maintainer brief."""

from copy import deepcopy
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.github import (
    GitHubCheckRunConclusion,
    GitHubContributionInputOptions,
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubMaintainerBriefError,
    GitHubMaintainerBriefRequest,
    GitHubMaintainerBriefService,
    GitHubRepositoryRef,
    GitHubReviewCostRequest,
    GitHubReviewCostService,
)
from opensteward.knowledge import KnowledgeRelatedWorkOptions
from opensteward.maintainer_brief import (
    MaintainerAttentionReasonKind,
    MaintainerAttentionRecommendation,
    MaintainerBrief,
    MaintainerBriefInput,
    MaintainerBriefService,
    MaintainerReviewRoute,
)
from opensteward.policy import ContributionCategory
from opensteward.review_intelligence import ReviewCostAssessmentOptions
from tests.github.test_review_cost import (
    ASSESSED_AT,
    REPOSITORY,
    RecordingFinder,
    RecordingReviewCostAssessor,
    assessment_result,
    check_run,
    completed_review_cost_result,
    related_result,
)
from tests.github.test_review_cost import (
    RecordingAssessor as RecordingPullRequestAssessor,
)
from tests.github.test_review_cost import (
    request as review_cost_request,
)


def request(**updates: Any) -> GitHubMaintainerBriefRequest:
    payload: dict[str, Any] = {
        "installation_id": 41,
        "repository": REPOSITORY,
        "pull_number": 17,
    }
    payload.update(updates)
    return GitHubMaintainerBriefRequest(**payload)


class RecordingAssessor:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.calls: list[GitHubReviewCostRequest] = []

    async def assess(self, selected: GitHubReviewCostRequest):
        self.calls.append(selected)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class RecordingBuilder:
    def __init__(self) -> None:
        self.calls: list[MaintainerBriefInput] = []
        self.delegate = MaintainerBriefService()

    def build(self, selected: MaintainerBriefInput) -> MaintainerBrief:
        self.calls.append(selected)
        return self.delegate.build(selected)


async def completed_maintainer_brief_result():
    """Build one real result for MCP serialization integration tests."""

    phase_five = await completed_review_cost_result()
    return await GitHubMaintainerBriefService(
        review_cost_assessor=RecordingAssessor(phase_five),
        brief_builder=MaintainerBriefService(),
    ).build(request())


def test_request_defaults_validation_and_review_cost_conversion() -> None:
    selected = request(
        policy_path="./config\\steward.yml",
        explicit_categories=[ContributionCategory.SECURITY],
        conversion_options=GitHubContributionInputOptions(
            require_complete_file_list=False
        ),
        related_work_options=KnowledgeRelatedWorkOptions(max_results=3),
        review_cost_options=ReviewCostAssessmentOptions(
            max_evidence_items_per_signal=4
        ),
    )
    before = deepcopy(selected.model_dump())

    converted = selected.to_review_cost_request()

    assert converted.installation_id == 41
    assert converted.repository == REPOSITORY
    assert converted.pull_number == 17
    assert converted.policy_path == "config/steward.yml"
    assert converted.explicit_categories == [ContributionCategory.SECURITY]
    assert converted.conversion_options == selected.conversion_options
    assert converted.conversion_options is selected.conversion_options
    assert converted.snapshot_options == GitHubHistoricalKnowledgeSnapshotOptions()
    assert converted.snapshot_options is selected.snapshot_options
    assert converted.related_work_options == selected.related_work_options
    assert converted.related_work_options is selected.related_work_options
    assert converted.review_cost_options == selected.review_cost_options
    assert converted.review_cost_options is selected.review_cost_options
    assert converted.explicit_categories is not selected.explicit_categories
    assert selected.model_dump() == before

    defaults = request()
    assert defaults.policy_path == ".opensteward.yml"
    assert defaults.explicit_categories == []
    assert defaults.conversion_options == GitHubContributionInputOptions()
    assert defaults.snapshot_options == GitHubHistoricalKnowledgeSnapshotOptions()
    assert defaults.related_work_options == KnowledgeRelatedWorkOptions()
    assert defaults.review_cost_options == ReviewCostAssessmentOptions()
    with pytest.raises(ValidationError):
        request(installation_id=0)
    with pytest.raises(ValidationError):
        request(pull_number=0)
    with pytest.raises(ValidationError, match="unique"):
        request(
            explicit_categories=[
                ContributionCategory.SECURITY,
                ContributionCategory.SECURITY,
            ]
        )


@pytest.mark.asyncio
async def test_service_reuses_phase_five_evidence_once_without_mutation() -> None:
    phase_five = await completed_review_cost_result()
    assessor = RecordingAssessor(phase_five)
    builder = RecordingBuilder()
    snapshot_before = deepcopy(
        phase_five.pull_request_assessment.snapshot.model_dump()
    )
    policy_before = deepcopy(
        phase_five.pull_request_assessment.repository_policy.model_dump()
    )

    result = await GitHubMaintainerBriefService(
        review_cost_assessor=assessor,
        brief_builder=builder,
    ).build(request())

    assert len(assessor.calls) == len(builder.calls) == 1
    assert assessor.calls[0] == request().to_review_cost_request()
    adapted = builder.calls[0]
    assert adapted.generated_at == phase_five.review_cost.assessed_at == ASSESSED_AT
    assert adapted.review_cost is phase_five.review_cost
    assert adapted.related_work is phase_five.related_work.related_work
    assert adapted.repository == phase_five.review_cost.repository
    assert adapted.pull_request == phase_five.review_cost.pull_request
    assert adapted.path_risk.changed_path_count == 3
    assert adapted.path_risk.protected_path_count == 1
    assert adapted.path_risk.security_sensitive_path_count == 3
    assert adapted.path_risk.database_migration_path_count == 0
    assert adapted.path_risk.automation_or_deployment_path_count == 0
    assert adapted.readiness.policy_present is True
    assert adapted.readiness.policy_blocked is False
    assert adapted.readiness.policy_attention_required is True
    assert adapted.readiness.policy_finding_count == 0
    assert adapted.readiness.draft is False
    assert adapted.readiness.merge_conflict is True
    assert adapted.readiness.required_checks_total == 0
    assert adapted.readiness.approval_count == 1
    assert adapted.readiness.changes_requested_count == 0
    assert adapted.historical_context.related_match_count == 1
    assert adapted.historical_context.rejected_or_superseded_count == 1
    assert adapted.historical_context.high_significance_count == 1
    assert adapted.historical_context.source_history_complete is True
    assert adapted.historical_context.ranking_coverage_complete is True
    assert adapted.historical_context.result_truncated is False
    assert (
        result.recommendation
        == MaintainerAttentionRecommendation.AUTHOR_ACTION_FIRST
    )
    assert result.routes == [
        MaintainerReviewRoute.SECURITY,
        MaintainerReviewRoute.ARCHITECTURE,
    ]
    assert phase_five.pull_request_assessment.snapshot.model_dump() == snapshot_before
    assert (
        phase_five.pull_request_assessment.repository_policy.model_dump()
        == policy_before
    )


@pytest.mark.asyncio
async def test_failed_required_check_reaches_maintainer_attention_routing() -> None:
    events: list[str] = []
    assessment = assessment_result(
        mergeable=True,
        mergeable_state="clean",
        required_checks=["tests"],
        check_runs=[
            check_run(
                check_id=1,
                name="tests",
                conclusion=GitHubCheckRunConclusion.FAILURE,
            )
        ],
    )
    related = await related_result(assessment)
    review_cost = await GitHubReviewCostService(
        pull_request_assessor=RecordingPullRequestAssessor(
            assessment,
            events,
        ),
        related_work_finder=RecordingFinder(
            related,
            events,
        ),
        review_cost_assessor=RecordingReviewCostAssessor(
            events
        ),
        clock=lambda: ASSESSED_AT,
    ).assess(review_cost_request())
    builder = RecordingBuilder()

    result = await GitHubMaintainerBriefService(
        review_cost_assessor=RecordingAssessor(
            review_cost
        ),
        brief_builder=builder,
    ).build(request())

    readiness = builder.calls[0].readiness
    assert readiness.required_checks_total == 1
    assert readiness.required_checks_passed == 0
    assert readiness.required_checks_failed == 1
    assert readiness.required_checks_pending == 0
    assert result.recommendation == (
        MaintainerAttentionRecommendation.AUTHOR_ACTION_FIRST
    )
    assert any(
        reason.kind
        == MaintainerAttentionReasonKind.FAILED_REQUIRED_CHECKS
        and reason.blocking
        for reason in result.brief.attention.reasons
    )
    assert (
        "Fix failed required checks before maintainer review."
        in result.brief.recommended_actions
    )


class SentinelError(RuntimeError):
    """Distinct dependency error."""


@pytest.mark.asyncio
async def test_review_cost_errors_propagate_and_stop_builder() -> None:
    error = SentinelError("review cost failed")
    assessor = RecordingAssessor(error)
    builder = RecordingBuilder()

    with pytest.raises(SentinelError) as exc_info:
        await GitHubMaintainerBriefService(
            review_cost_assessor=assessor,
            brief_builder=builder,
        ).build(request())

    assert exc_info.value is error
    assert len(assessor.calls) == 1
    assert builder.calls == []


@pytest.mark.asyncio
async def test_result_identity_is_validated_before_brief_builder() -> None:
    phase_five = await completed_review_cost_result()
    invalid = phase_five.model_copy(
        update={
            "repository": GitHubRepositoryRef(owner="other", name="repository")
        }
    )
    builder = RecordingBuilder()

    with pytest.raises(GitHubMaintainerBriefError, match="another repository"):
        await GitHubMaintainerBriefService(
            review_cost_assessor=RecordingAssessor(invalid),
            brief_builder=builder,
        ).build(request())

    assert builder.calls == []


@pytest.mark.asyncio
async def test_result_serialization_is_structured_and_credential_redacted() -> None:
    phase_five = await completed_review_cost_result()
    result = await GitHubMaintainerBriefService(
        review_cost_assessor=RecordingAssessor(phase_five),
        brief_builder=MaintainerBriefService(),
    ).build(request())

    data = result.model_dump(mode="json")

    assert data["pull_request"]["title"]
    assert data["brief"]["readiness"]
    assert data["brief"]["related_work"]["matches"]
    assert data["brief"]["review_cost"]["contributions"]
    assert data["recommendation"] == "author_action_first"
    assert data["routes"] == ["security", "architecture"]
    assert data["brief"]["attention"]["reasons"]
    assert data["brief"]["recommended_actions"]
    assert "complete" in data
    assessment = data["pull_request_assessment"]
    assert set(assessment) == {
        "read_only",
        "summary",
        "policy",
        "conversion",
        "packet",
        "evaluation",
    }
    assert "installation_id" not in assessment
    assert "snapshot" not in assessment
    assert "repository_policy" not in assessment
    serialized = str(data).casefold()
    for secret in (
        "installation_id",
        "installation_token",
        "private_key",
        "token_scope",
        "authorization",
    ):
        assert secret not in serialized
