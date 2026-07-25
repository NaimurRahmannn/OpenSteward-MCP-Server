"""Shared fixtures for maintainer-brief tests."""

import asyncio
from datetime import UTC, datetime
from typing import Any

from opensteward.knowledge import (
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemReference,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalQuery,
    KnowledgeRelatedWorkResult,
    KnowledgeRelatedWorkService,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)
from opensteward.maintainer_brief import (
    MaintainerBriefInput,
    MaintainerPathRiskSummary,
    MaintainerReadinessSummary,
)
from opensteward.review_intelligence import (
    REVIEW_COST_CHANGE_DISPERSION_WEIGHT,
    REVIEW_COST_CHANGE_SIZE_WEIGHT,
    REVIEW_COST_HISTORICAL_COMPLEXITY_WEIGHT,
    REVIEW_COST_RISK_SENSITIVE_PATHS_WEIGHT,
    REVIEW_COST_VALIDATION_GAPS_WEIGHT,
    ReviewCostAssessment,
    ReviewCostHistoricalContext,
    ReviewCostSignal,
    ReviewCostSignalContribution,
)

REPOSITORY = KnowledgeRepositoryRef(
    provider="github",
    namespace="acme",
    name="framework",
)
PULL_REQUEST = KnowledgeItemReference(
    repository=REPOSITORY,
    item_type=KnowledgeItemType.PULL_REQUEST,
    external_id="17",
    source_kind=KnowledgeSourceKind.GITHUB,
    title="Parser update",
)
ASSESSED_AT = datetime(2026, 6, 1, tzinfo=UTC)
_WEIGHTS = [
    REVIEW_COST_CHANGE_SIZE_WEIGHT,
    REVIEW_COST_CHANGE_DISPERSION_WEIGHT,
    REVIEW_COST_RISK_SENSITIVE_PATHS_WEIGHT,
    REVIEW_COST_VALIDATION_GAPS_WEIGHT,
    REVIEW_COST_HISTORICAL_COMPLEXITY_WEIGHT,
]


def review_cost(
    score: int = 10,
    warnings: list[str] | None = None,
) -> ReviewCostAssessment:
    contributions = [
        ReviewCostSignalContribution(
            signal=signal,
            signal_score=score,
            weight_percent=weight,
            weighted_basis_points=score * weight,
            explanation=f"{signal.value} explanation",
            evidence=[f"{signal.value} evidence"],
        )
        for signal, weight in zip(ReviewCostSignal, _WEIGHTS, strict=True)
    ]
    return ReviewCostAssessment(
        repository=REPOSITORY,
        pull_request=PULL_REQUEST,
        assessed_at=ASSESSED_AT,
        contributions=contributions,
        reducers=[],
        warnings=warnings or [],
    )


def related_work(
    count: int = 0,
    warnings: list[str] | None = None,
) -> KnowledgeRelatedWorkResult:
    query = KnowledgeLexicalQuery(repository=REPOSITORY, text="parser")
    items = [
        KnowledgeItem(
            repository=REPOSITORY,
            item_type=KnowledgeItemType.ISSUE,
            external_id=str(index + 1),
            source_kind=KnowledgeSourceKind.GITHUB,
            state=KnowledgeItemState.OPEN,
            title=f"Parser issue {index + 1}",
            body="Parser architecture and implementation.",
            summary="Parser work.",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            updated_at=datetime(2025, 1, 2, tzinfo=UTC),
            labels=[],
            affected_paths=[],
            components=[],
            decision_significance=DecisionSignificance.MEDIUM,
        )
        for index in range(count)
    ]
    result = asyncio.run(
        KnowledgeRelatedWorkService().find(
            query,
            items,
            as_of=ASSESSED_AT,
        )
    )
    if warnings:
        result = result.model_copy(update={"warnings": list(warnings)})
    return result


def brief_input(
    *,
    score: int = 10,
    related_count: int = 0,
    readiness_updates: dict[str, Any] | None = None,
    path_updates: dict[str, Any] | None = None,
    history_updates: dict[str, Any] | None = None,
    review_warnings: list[str] | None = None,
    related_warnings: list[str] | None = None,
) -> MaintainerBriefInput:
    readiness: dict[str, Any] = {
        "policy_present": True,
        "policy_blocked": False,
        "policy_attention_required": False,
        "policy_finding_count": 0,
        "draft": False,
        "merge_conflict": False,
        "required_checks_total": 1,
        "required_checks_passed": 1,
        "required_checks_failed": 0,
        "required_checks_pending": 0,
        "approval_count": 1,
        "changes_requested_count": 0,
    }
    readiness.update(readiness_updates or {})
    path_risk: dict[str, Any] = {
        "changed_path_count": 1,
        "protected_path_count": 0,
        "security_sensitive_path_count": 0,
        "database_migration_path_count": 0,
        "automation_or_deployment_path_count": 0,
        "dependency_manifest_path_count": 0,
    }
    path_risk.update(path_updates or {})
    history: dict[str, Any] = {
        "related_match_count": related_count,
        "rejected_or_superseded_count": 0,
        "high_significance_count": 0,
        "unresolved_count": 0,
        "source_history_complete": True,
        "ranking_coverage_complete": True,
        "result_truncated": False,
    }
    history.update(history_updates or {})
    return MaintainerBriefInput(
        repository=REPOSITORY,
        pull_request=PULL_REQUEST,
        generated_at=ASSESSED_AT,
        readiness=MaintainerReadinessSummary(**readiness),
        path_risk=MaintainerPathRiskSummary(**path_risk),
        historical_context=ReviewCostHistoricalContext(**history),
        review_cost=review_cost(score, review_warnings),
        related_work=related_work(related_count, related_warnings),
    )
