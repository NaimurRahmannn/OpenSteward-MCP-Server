"""Tests for GitHub evidence-derived review-cost orchestration."""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.github import (
    GitHubApprovalCountSource,
    GitHubChecksState,
    GitHubChecksSummary,
    GitHubContributionInputOptions,
    GitHubContributionInputResult,
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubIssueLinkageScope,
    GitHubPullRequestActor,
    GitHubPullRequestAssessmentPolicy,
    GitHubPullRequestAssessmentRequest,
    GitHubPullRequestAssessmentResult,
    GitHubPullRequestAssessmentSummary,
    GitHubPullRequestBranch,
    GitHubPullRequestDetails,
    GitHubPullRequestFile,
    GitHubPullRequestReview,
    GitHubPullRequestReviewState,
    GitHubPullRequestSnapshot,
    GitHubRelatedWorkQuery,
    GitHubRelatedWorkRequest,
    GitHubRelatedWorkResult,
    GitHubRelatedWorkSnapshotSummary,
    GitHubRepositoryRef,
    GitHubReviewCostError,
    GitHubReviewCostRequest,
    GitHubReviewCostResult,
    GitHubReviewCostService,
    knowledge_repository_from_github,
)
from opensteward.knowledge import (
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRelatedWorkOptions,
    KnowledgeRelatedWorkService,
    KnowledgeSourceKind,
)
from opensteward.policy import (
    ContributionCategory,
    ContributionPolicyInput,
    MaintainerPolicyPacket,
    MaintainerRecommendation,
    PolicyEvaluationResult,
    PolicySource,
    ProtectedPathRule,
    RepositoryPolicy,
    RiskLevel,
)
from opensteward.review_intelligence import (
    ReviewCostAssessment,
    ReviewCostAssessmentInput,
    ReviewCostAssessmentOptions,
    ReviewCostAssessmentService,
    ReviewCostChangeType,
)

REPOSITORY = GitHubRepositoryRef(owner="acme", name="framework")
OTHER_REPOSITORY = GitHubRepositoryRef(owner="other", name="project")
KNOWLEDGE_REPOSITORY = knowledge_repository_from_github(REPOSITORY)
BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
ASSESSED_AT = datetime(2026, 6, 1, 10, 0, tzinfo=UTC)
RELATED_AT = datetime(2026, 5, 31, 10, 0, tzinfo=UTC)


def checks() -> GitHubChecksSummary:
    return GitHubChecksSummary(
        state=GitHubChecksState.SUCCESS,
        total_count=1,
        pending_count=0,
        success_count=1,
        failure_count=0,
        neutral_count=0,
        skipped_count=0,
    )


def assessment_result(
    *,
    mergeable: bool | None = False,
    mergeable_state: str | None = "dirty",
    policy_present: bool = True,
) -> GitHubPullRequestAssessmentResult:
    """Build a complete typed assessment exposing source evidence."""

    actor = GitHubPullRequestActor(
        id=1,
        login="author",
        account_type="User",
    )
    reviewer = GitHubPullRequestActor(
        id=2,
        login="reviewer",
        account_type="User",
    )
    details = GitHubPullRequestDetails(
        id=100,
        number=17,
        title="Refactor parser",
        body="Moves authentication parsing into a registry.",
        state="open",
        draft=False,
        merged=False,
        mergeable=mergeable,
        mergeable_state=mergeable_state,
        html_url="https://github.com/acme/framework/pull/17",
        labels=["architecture", "security"],
        author=actor,
        base=GitHubPullRequestBranch(
            ref="main",
            sha=BASE_SHA,
            repository_full_name="acme/framework",
        ),
        head=GitHubPullRequestBranch(
            ref="feature",
            sha=HEAD_SHA,
            repository_full_name="author/framework",
        ),
        additions=40,
        deletions=5,
        changed_files_count=2,
        commits_count=4,
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
        updated_at=datetime(2026, 5, 30, tzinfo=UTC),
    )
    files = [
        GitHubPullRequestFile(
            filename="src/security/auth.py",
            previous_filename="src/legacy/auth.py",
            status="RENAMED",
            additions=30,
            deletions=5,
            changes=35,
        ),
        GitHubPullRequestFile(
            filename="tests/test_auth.py",
            status="modified",
            additions=10,
            deletions=0,
            changes=10,
        ),
    ]
    approval = GitHubPullRequestReview(
        id=10,
        reviewer=reviewer,
        state=GitHubPullRequestReviewState.APPROVED,
        submitted_at=datetime(2026, 5, 30, tzinfo=UTC),
        commit_id=HEAD_SHA,
        on_head_commit=True,
    )
    snapshot = GitHubPullRequestSnapshot(
        repository=REPOSITORY,
        pull_request=details,
        files=files,
        reviews=[approval],
        effective_reviews=[approval],
        check_runs=[],
        checks=checks(),
    )
    policy = RepositoryPolicy(
        protected_paths=[
            ProtectedPathRule(
                pattern="src/security/**",
                risk=RiskLevel.CRITICAL,
            )
        ]
    )
    summary = GitHubPullRequestAssessmentSummary(
        repository=REPOSITORY,
        pull_number=17,
        title=details.title,
        html_url=details.html_url,
        author_login="author",
        state="open",
        draft=False,
        merged=False,
        mergeable=mergeable,
        mergeable_state=mergeable_state,
        base_ref="main",
        base_sha=BASE_SHA,
        head_ref="feature",
        head_sha=HEAD_SHA,
        additions=40,
        deletions=5,
        diff_lines=45,
        changed_files_reported=2,
        files_collected=2,
        files_complete=True,
        review_history_count=1,
        effective_review_count=1,
        human_approval_count=1,
        head_commit_human_approval_count=1,
        human_changes_requested_count=0,
        checks=checks(),
    )
    source = (
        PolicySource.GITHUB_REPOSITORY
        if policy_present
        else PolicySource.DEFAULT
    )
    policy_summary = GitHubPullRequestAssessmentPolicy(
        source=source,
        source_reference="github:acme/framework@base:.opensteward.yml",
        used_defaults=not policy_present,
        policy_version=1,
        trusted_ref=BASE_SHA,
        requested_path=".opensteward.yml",
        policy_file_present=policy_present,
        policy_file_sha="policy-sha" if policy_present else None,
        policy_file_html_url=(
            "https://github.com/acme/framework/blob/base/.opensteward.yml"
            if policy_present
            else None
        ),
    )
    conversion = GitHubContributionInputResult(
        contribution=ContributionPolicyInput(
            changed_files=[
                "src/security/auth.py",
                "src/legacy/auth.py",
                "tests/test_auth.py",
            ],
            additions=40,
            deletions=5,
            categories=[ContributionCategory.ARCHITECTURE],
            tests_changed=True,
            current_approvals=1,
        ),
        affected_paths=[
            "src/security/auth.py",
            "src/legacy/auth.py",
            "tests/test_auth.py",
        ],
        test_file_matches=["tests/test_auth.py"],
        category_evidence=[],
        linked_issue_evidence=[],
        approval_source=GitHubApprovalCountSource.HEAD_COMMIT,
        issue_linkage_scope=GitHubIssueLinkageScope.BODY_CLOSING_KEYWORDS_ONLY,
        checks_state=GitHubChecksState.SUCCESS,
        files_complete=True,
        warnings=[],
    )
    evaluation = PolicyEvaluationResult(
        compliant=True,
        requires_human_review=True,
        required_approvals=1,
        current_approvals=1,
        remaining_approvals=0,
        highest_protected_path_risk=RiskLevel.CRITICAL,
        findings=[],
    )
    packet = MaintainerPolicyPacket(
        recommendation=MaintainerRecommendation.READY_FOR_REVIEW,
        ready_for_detailed_review=True,
        summary="Ready.",
        approval_summary="Approval requirement met.",
        blocking_requirements=[],
        warnings=[],
        suggested_next_actions=[],
        passed_checks=0,
        informational_checks=0,
    )
    return GitHubPullRequestAssessmentResult(
        installation_id=41,
        snapshot=snapshot,
        repository_policy=policy,
        summary=summary,
        policy=policy_summary,
        conversion=conversion,
        packet=packet,
        evaluation=evaluation,
    )


def expected_query(
    assessment: GitHubPullRequestAssessmentResult,
) -> GitHubRelatedWorkQuery:
    details = assessment.snapshot.pull_request
    return GitHubRelatedWorkQuery(
        text=f"{details.title}\n\n{details.body}",
        labels=["architecture", "security"],
        affected_paths=[
            "src/security/auth.py",
            "src/legacy/auth.py",
            "tests/test_auth.py",
        ],
        components=[],
        item_types=[
            KnowledgeItemType.ISSUE,
            KnowledgeItemType.PULL_REQUEST,
            KnowledgeItemType.ADR,
        ],
        states=[],
    )


async def related_result(
    assessment: GitHubPullRequestAssessmentResult,
    *,
    warnings: list[str] | None = None,
) -> GitHubRelatedWorkResult:
    query = expected_query(assessment).to_knowledge_query(KNOWLEDGE_REPOSITORY)
    item = KnowledgeItem(
        repository=KNOWLEDGE_REPOSITORY,
        item_type=KnowledgeItemType.ISSUE,
        external_id="12",
        source_kind=KnowledgeSourceKind.GITHUB,
        state=KnowledgeItemState.REJECTED,
        title="Authentication parser design",
        body="Registry architecture was rejected.",
        summary="Rejected parser design.",
        url="https://github.com/acme/framework/issues/12",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 2, 1, tzinfo=UTC),
        closed_at=datetime(2025, 2, 1, tzinfo=UTC),
        labels=["architecture", "security"],
        affected_paths=["src/security/auth.py"],
        components=["security"],
        decision_significance=DecisionSignificance.HIGH,
    )
    related = await KnowledgeRelatedWorkService().find(
        query,
        [item],
        as_of=RELATED_AT,
        options=KnowledgeRelatedWorkOptions(),
    )
    snapshot = GitHubRelatedWorkSnapshotSummary(
        repository=REPOSITORY,
        knowledge_repository=KNOWLEDGE_REPOSITORY,
        requested_ref=BASE_SHA,
        resolved_commit_sha=BASE_SHA,
        adr_tree_sha="c" * 40,
        collected_at=RELATED_AT,
        adr_snapshot_commit_date=RELATED_AT,
        complete=True,
        total_count=1,
        issue_count=1,
        pull_request_count=0,
        adr_count=0,
        warnings=warnings or [],
    )
    return GitHubRelatedWorkResult(
        repository=REPOSITORY,
        snapshot=snapshot,
        related_work=related,
        warnings=warnings or [],
    )


def request(**updates: Any) -> GitHubReviewCostRequest:
    payload: dict[str, Any] = {
        "installation_id": 41,
        "repository": REPOSITORY,
        "pull_number": 17,
    }
    payload.update(updates)
    return GitHubReviewCostRequest(**payload)


class RecordingAssessor:
    def __init__(self, outcome: object, events: list[str]) -> None:
        self.outcome = outcome
        self.events = events
        self.calls: list[GitHubPullRequestAssessmentRequest] = []

    async def assess(
        self,
        selected_request: GitHubPullRequestAssessmentRequest,
    ) -> GitHubPullRequestAssessmentResult:
        self.events.append("assessment")
        self.calls.append(selected_request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        assert isinstance(self.outcome, GitHubPullRequestAssessmentResult)
        return self.outcome


class RecordingFinder:
    def __init__(self, outcome: object, events: list[str]) -> None:
        self.outcome = outcome
        self.events = events
        self.calls: list[GitHubRelatedWorkRequest] = []

    async def find(
        self,
        selected_request: GitHubRelatedWorkRequest,
    ) -> GitHubRelatedWorkResult:
        self.events.append("related")
        self.calls.append(selected_request)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        assert isinstance(self.outcome, GitHubRelatedWorkResult)
        return self.outcome


class RecordingReviewCostAssessor:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.calls: list[
            tuple[ReviewCostAssessmentInput, datetime, ReviewCostAssessmentOptions | None]
        ] = []

    def assess(
        self,
        assessment_input: ReviewCostAssessmentInput,
        *,
        assessed_at: datetime,
        options: ReviewCostAssessmentOptions | None = None,
    ) -> ReviewCostAssessment:
        self.events.append("review-cost")
        self.calls.append((assessment_input, assessed_at, options))
        return ReviewCostAssessmentService().assess(
            assessment_input,
            assessed_at=assessed_at,
            options=options,
        )


async def completed_review_cost_result() -> GitHubReviewCostResult:
    """Build one fully validated result for cross-boundary serialization tests."""

    events: list[str] = []
    assessment = assessment_result()
    related = await related_result(assessment)
    return await GitHubReviewCostService(
        pull_request_assessor=RecordingAssessor(assessment, events),
        related_work_finder=RecordingFinder(related, events),
        review_cost_assessor=RecordingReviewCostAssessor(events),
        clock=lambda: ASSESSED_AT,
    ).assess(request())


class SentinelError(RuntimeError):
    """Distinct dependency failure."""


def test_request_defaults_bounds_and_explicit_options() -> None:
    selected = request()
    assert selected.policy_path == ".opensteward.yml"
    assert selected.explicit_categories == []
    assert selected.conversion_options == GitHubContributionInputOptions()
    assert selected.snapshot_options == GitHubHistoricalKnowledgeSnapshotOptions()
    assert selected.related_work_options == KnowledgeRelatedWorkOptions()
    assert selected.review_cost_options == ReviewCostAssessmentOptions()

    with pytest.raises(ValidationError):
        request(installation_id=0)
    with pytest.raises(ValidationError):
        request(pull_number=0)

    options = ReviewCostAssessmentOptions(max_evidence_items_per_signal=3)
    explicit = request(
        explicit_categories=[ContributionCategory.SECURITY],
        review_cost_options=options,
    )
    assert explicit.explicit_categories == [ContributionCategory.SECURITY]
    assert explicit.review_cost_options == options


@pytest.mark.asyncio
async def test_service_derives_all_evidence_in_exact_sequence_without_mutation() -> None:
    events: list[str] = []
    assessment = assessment_result()
    related = await related_result(assessment, warnings=["history warning"])
    assessor = RecordingAssessor(assessment, events)
    finder = RecordingFinder(related, events)
    domain = RecordingReviewCostAssessor(events)
    selected_request = request(
        explicit_categories=[ContributionCategory.ARCHITECTURE],
        review_cost_options=ReviewCostAssessmentOptions(
            max_evidence_items_per_signal=7
        ),
    )
    before = (
        deepcopy(selected_request.model_dump()),
        deepcopy(assessment.model_dump()),
        deepcopy(related.model_dump()),
    )
    clock_calls = 0

    def clock() -> datetime:
        nonlocal clock_calls
        clock_calls += 1
        events.append("clock")
        return ASSESSED_AT

    result = await GitHubReviewCostService(
        pull_request_assessor=assessor,
        related_work_finder=finder,
        review_cost_assessor=domain,
        clock=clock,
    ).assess(selected_request)

    assert events == ["assessment", "related", "clock", "review-cost"]
    assert clock_calls == 1
    assert len(assessor.calls) == len(finder.calls) == len(domain.calls) == 1
    assessment_request = assessor.calls[0]
    assert assessment_request.installation_id == selected_request.installation_id
    assert assessment_request.repository == REPOSITORY
    assert assessment_request.pull_number == 17
    assert assessment_request.explicit_categories == [
        ContributionCategory.ARCHITECTURE
    ]

    related_request = finder.calls[0]
    assert related_request.git_ref == BASE_SHA
    assert related_request.query == expected_query(assessment)
    assert related_request.snapshot_options == selected_request.snapshot_options
    assert related_request.related_work_options == selected_request.related_work_options

    domain_input, assessed_at, domain_options = domain.calls[0]
    assert [file.change_type for file in domain_input.files] == [
        ReviewCostChangeType.RENAMED,
        ReviewCostChangeType.MODIFIED,
    ]
    assert [(file.additions, file.deletions) for file in domain_input.files] == [
        (30, 5),
        (10, 0),
    ]
    assert domain_input.commit_count == 4
    assert domain_input.preferred_diff_size == 500
    assert domain_input.policy_present is True
    assert domain_input.protected_changed_paths == ["src/security/auth.py"]
    assert domain_input.draft is False
    assert domain_input.merge_conflict is True
    assert domain_input.required_checks_total == 0
    assert domain_input.approval_count == 1
    assert domain_input.changes_requested_count == 0
    assert domain_input.historical_context is not None
    assert domain_input.historical_context.related_match_count == 1
    assert domain_input.historical_context.rejected_or_superseded_count == 1
    assert domain_input.historical_context.high_significance_count == 1
    assert assessed_at == ASSESSED_AT
    assert domain_options is selected_request.review_cost_options

    assert result.pull_request.base_sha == BASE_SHA
    assert result.pull_request.head_sha == HEAD_SHA
    assert result.pull_request.changed_files == 2
    assert result.pull_request.commits == 4
    assert result.review_cost is not None
    assert result.warnings == ["history warning"]
    assert selected_request.model_dump() == before[0]
    assert assessment.model_dump() == before[1]
    assert related.model_dump() == before[2]


@pytest.mark.asyncio
async def test_missing_policy_uses_safe_fallback_and_no_protected_paths() -> None:
    events: list[str] = []
    assessment = assessment_result(policy_present=False)
    related = await related_result(assessment)
    domain = RecordingReviewCostAssessor(events)
    result = await GitHubReviewCostService(
        pull_request_assessor=RecordingAssessor(assessment, events),
        related_work_finder=RecordingFinder(related, events),
        review_cost_assessor=domain,
        clock=lambda: ASSESSED_AT,
    ).assess(request())

    adapted = domain.calls[0][0]
    assert adapted.preferred_diff_size == 700
    assert adapted.policy_present is False
    assert adapted.protected_changed_paths == []
    assert result.review_cost.warnings[0].startswith(
        "Repository policy was unavailable"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mergeable", "state", "expected"),
    [
        (True, "clean", False),
        (False, "dirty", True),
        (None, "dirty", True),
        (None, "clean", False),
        (None, "unknown", None),
    ],
)
async def test_mergeability_mapping(
    mergeable: bool | None,
    state: str,
    expected: bool | None,
) -> None:
    events: list[str] = []
    assessment = assessment_result(
        mergeable=mergeable,
        mergeable_state=state,
    )
    related = await related_result(assessment)
    domain = RecordingReviewCostAssessor(events)
    await GitHubReviewCostService(
        pull_request_assessor=RecordingAssessor(assessment, events),
        related_work_finder=RecordingFinder(related, events),
        review_cost_assessor=domain,
        clock=lambda: ASSESSED_AT,
    ).assess(request())
    assert domain.calls[0][0].merge_conflict is expected


@pytest.mark.asyncio
async def test_dependency_errors_propagate_and_stop_later_services() -> None:
    events: list[str] = []
    finder = RecordingFinder(SentinelError("must not run"), events)
    domain = RecordingReviewCostAssessor(events)
    with pytest.raises(SentinelError, match="assessment"):
        await GitHubReviewCostService(
            pull_request_assessor=RecordingAssessor(
                SentinelError("assessment"),
                events,
            ),
            related_work_finder=finder,
            review_cost_assessor=domain,
        ).assess(request())
    assert finder.calls == []
    assert domain.calls == []

    assessment = assessment_result()
    with pytest.raises(SentinelError, match="related"):
        await GitHubReviewCostService(
            pull_request_assessor=RecordingAssessor(assessment, events),
            related_work_finder=RecordingFinder(
                SentinelError("related"),
                events,
            ),
            review_cost_assessor=domain,
        ).assess(request())
    assert domain.calls == []


@pytest.mark.asyncio
async def test_naive_clock_is_rejected_before_domain_assessor() -> None:
    events: list[str] = []
    assessment = assessment_result()
    related = await related_result(assessment)
    domain = RecordingReviewCostAssessor(events)
    with pytest.raises(GitHubReviewCostError, match="timezone-aware"):
        await GitHubReviewCostService(
            pull_request_assessor=RecordingAssessor(assessment, events),
            related_work_finder=RecordingFinder(related, events),
            review_cost_assessor=domain,
            clock=lambda: datetime(2026, 6, 1),
        ).assess(request())
    assert domain.calls == []


@pytest.mark.asyncio
async def test_consistency_errors_use_dedicated_type() -> None:
    events: list[str] = []
    assessment = assessment_result()
    invalid_assessment = assessment.model_copy(
        update={
            "summary": assessment.summary.model_copy(
                update={"repository": OTHER_REPOSITORY}
            )
        }
    )
    with pytest.raises(GitHubReviewCostError, match="another repository"):
        await GitHubReviewCostService(
            pull_request_assessor=RecordingAssessor(invalid_assessment, events),
            related_work_finder=RecordingFinder(SentinelError(), events),
            review_cost_assessor=RecordingReviewCostAssessor(events),
        ).assess(request())

    related = await related_result(assessment)
    invalid_related = related.model_copy(
        update={
            "snapshot": related.snapshot.model_copy(
                update={"requested_ref": "wrong"}
            )
        }
    )
    with pytest.raises(GitHubReviewCostError, match="base SHA"):
        await GitHubReviewCostService(
            pull_request_assessor=RecordingAssessor(assessment, events),
            related_work_finder=RecordingFinder(invalid_related, events),
            review_cost_assessor=RecordingReviewCostAssessor(events),
        ).assess(request())


@pytest.mark.asyncio
async def test_result_serialization_is_concise_and_excludes_credentials() -> None:
    events: list[str] = []
    assessment = assessment_result()
    related = await related_result(assessment)
    result = await GitHubReviewCostService(
        pull_request_assessor=RecordingAssessor(assessment, events),
        related_work_finder=RecordingFinder(related, events),
        review_cost_assessor=RecordingReviewCostAssessor(events),
        clock=lambda: ASSESSED_AT,
    ).assess(request())

    data = result.model_dump(mode="json")

    assert data["pull_request"]["pull_number"] == 17
    assert data["review_cost"]["contributions"]
    assert data["score"] == result.score
    assert data["level"] == result.level.value
    serialized = str(data).casefold()
    for sensitive_text in (
        "installation_id",
        "installation token",
        "installation_token",
        "private key",
        "private_key",
        "token scope",
        "token_scope",
    ):
        assert sensitive_text not in serialized
    assert result.complete is True


def test_assessment_contract_exposes_authoritative_review_cost_evidence() -> None:
    result = assessment_result()

    assert result.snapshot.pull_request.labels == [
        "architecture",
        "security",
    ]
    assert result.snapshot.pull_request.commits_count == 4
    assert (
        result.repository_policy.pull_requests.preferred_maximum_diff_lines
        == 500
    )
    assert result.policy.policy_file_present is True

    data = result.model_dump(mode="json")
    assert data["installation_id"] == 41
    assert "snapshot" not in data
    assert "repository_policy" not in data
    assert {
        "read_only",
        "installation_id",
        "summary",
        "policy",
        "conversion",
        "packet",
        "evaluation",
    } == set(data)

    schema = result.model_json_schema()
    assert set(schema["properties"]) == set(data)
    assert "installation_id" in schema["required"]


@pytest.mark.asyncio
async def test_review_cost_nested_assessment_is_public_and_redacted() -> None:
    result = await completed_review_cost_result()

    data = result.model_dump(mode="json")
    assessment = data["pull_request_assessment"]
    assert {
        "read_only",
        "summary",
        "policy",
        "conversion",
        "packet",
        "evaluation",
    } == set(assessment)
    assert "installation_id" not in assessment
    assert "snapshot" not in assessment
    assert "repository_policy" not in assessment
    assert {
        "repository",
        "pull_request",
        "pull_request_assessment",
        "related_work",
        "review_cost",
        "warnings",
        "score",
        "level",
        "complete",
    } == set(data)
    serialized = str(data).casefold()
    assert "installation_id" not in serialized
    assert "snapshot" not in assessment
    assert "repository_policy" not in serialized
