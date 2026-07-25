"""Deterministic maintainer-attention and brief services."""

from opensteward.maintainer_brief.models import (
    MaintainerAttentionAssessment,
    MaintainerAttentionReason,
    MaintainerAttentionReasonKind,
    MaintainerAttentionRecommendation,
    MaintainerBrief,
    MaintainerBriefInput,
    MaintainerReviewRoute,
)
from opensteward.review_intelligence import ReviewCostLevel, ReviewCostSignal

_MISSING_POLICY_WARNING = (
    "Repository policy was unavailable; the default preferred diff size and "
    "no protected paths were used."
)
_UNKNOWN_MERGE_WARNING = "Pull-request merge-conflict state was unavailable."
_INCOMPLETE_HISTORY_WARNING = "Historical source collection was incomplete."
_INCOMPLETE_RANKING_WARNING = "Related-work ranking coverage was incomplete."
_TRUNCATED_RELATED_WARNING = (
    "Related-work results were truncated by the configured final result limit."
)


def _reason(
    kind: MaintainerAttentionReasonKind,
    explanation: str,
) -> MaintainerAttentionReason:
    return MaintainerAttentionReason(
        kind=kind,
        explanation=explanation,
        blocking=kind
        in {
            MaintainerAttentionReasonKind.POLICY_BLOCKER,
            MaintainerAttentionReasonKind.MERGE_CONFLICT,
            MaintainerAttentionReasonKind.FAILED_REQUIRED_CHECKS,
            MaintainerAttentionReasonKind.CHANGES_REQUESTED,
            MaintainerAttentionReasonKind.DRAFT_PULL_REQUEST,
            MaintainerAttentionReasonKind.PENDING_REQUIRED_CHECKS,
        },
    )


class MaintainerAttentionService:
    """Derive deterministic queue priority, routes, reasons, and warnings."""

    def assess(
        self,
        brief_input: MaintainerBriefInput,
    ) -> MaintainerAttentionAssessment:
        readiness = brief_input.readiness
        path_risk = brief_input.path_risk
        history = brief_input.historical_context
        review_cost = brief_input.review_cost
        reasons: list[MaintainerAttentionReason] = []

        if readiness.policy_blocked:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.POLICY_BLOCKER,
                    "Repository policy reports blocking findings.",
                )
            )
        if readiness.merge_conflict is True:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.MERGE_CONFLICT,
                    "The pull request has a merge conflict.",
                )
            )
        if readiness.required_checks_failed:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.FAILED_REQUIRED_CHECKS,
                    f"{readiness.required_checks_failed} required checks failed.",
                )
            )
        if readiness.changes_requested_count:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.CHANGES_REQUESTED,
                    f"{readiness.changes_requested_count} active "
                    "changes-requested reviews remain.",
                )
            )
        if readiness.draft:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.DRAFT_PULL_REQUEST,
                    "The pull request is still marked as draft.",
                )
            )
        if readiness.required_checks_pending:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.PENDING_REQUIRED_CHECKS,
                    f"{readiness.required_checks_pending} required checks are pending.",
                )
            )
        if review_cost.level == ReviewCostLevel.CRITICAL:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.CRITICAL_REVIEW_COST,
                    f"Review cost is critical at {review_cost.score}/100.",
                )
            )
        elif review_cost.level == ReviewCostLevel.HIGH:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.HIGH_REVIEW_COST,
                    f"Review cost is high at {review_cost.score}/100.",
                )
            )
        if path_risk.protected_path_count:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.PROTECTED_PATHS,
                    f"{path_risk.protected_path_count} protected paths changed.",
                )
            )
        if path_risk.security_sensitive_path_count:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.SECURITY_SENSITIVE_PATHS,
                    f"{path_risk.security_sensitive_path_count} "
                    "security-sensitive paths changed.",
                )
            )
        if path_risk.database_migration_path_count:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.DATABASE_MIGRATION,
                    f"{path_risk.database_migration_path_count} "
                    "database-migration paths changed.",
                )
            )
        if path_risk.automation_or_deployment_path_count:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.AUTOMATION_OR_DEPLOYMENT,
                    f"{path_risk.automation_or_deployment_path_count} "
                    "automation or deployment paths changed.",
                )
            )
        if history.rejected_or_superseded_count:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.REJECTED_OR_SUPERSEDED_HISTORY,
                    f"{history.rejected_or_superseded_count} related items are "
                    "rejected or superseded.",
                )
            )
        if history.high_significance_count:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.HIGH_SIGNIFICANCE_HISTORY,
                    f"{history.high_significance_count} related items have high "
                    "or critical significance.",
                )
            )
        if history.unresolved_count:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.UNRESOLVED_HISTORY,
                    f"{history.unresolved_count} related items remain unresolved.",
                )
            )
        incomplete = (
            not readiness.policy_present
            or readiness.merge_conflict is None
            or not history.source_history_complete
            or not history.ranking_coverage_complete
            or history.result_truncated
        )
        if incomplete:
            reasons.append(
                _reason(
                    MaintainerAttentionReasonKind.INCOMPLETE_EVIDENCE,
                    "The brief has incomplete policy, mergeability, source-history, "
                    "ranking, or result coverage.",
                )
            )

        routes: list[MaintainerReviewRoute] = []
        if path_risk.security_sensitive_path_count:
            routes.append(MaintainerReviewRoute.SECURITY)
        if path_risk.database_migration_path_count:
            routes.append(MaintainerReviewRoute.DATABASE)
        if path_risk.automation_or_deployment_path_count:
            routes.append(MaintainerReviewRoute.RELEASE_OR_DEPLOYMENT)
        if (
            path_risk.protected_path_count
            or history.rejected_or_superseded_count
            or history.high_significance_count
            or history.unresolved_count >= 3
        ):
            routes.append(MaintainerReviewRoute.ARCHITECTURE)
        if not routes:
            routes = [MaintainerReviewRoute.GENERAL]

        if any(reason.blocking for reason in reasons):
            recommendation = MaintainerAttentionRecommendation.AUTHOR_ACTION_FIRST
        elif (
            review_cost.level == ReviewCostLevel.CRITICAL
            or (
                path_risk.security_sensitive_path_count > 0
                and review_cost.level in {ReviewCostLevel.HIGH, ReviewCostLevel.CRITICAL}
            )
            or (
                history.rejected_or_superseded_count > 0
                and history.high_significance_count > 0
            )
        ):
            recommendation = MaintainerAttentionRecommendation.IMMEDIATE_REVIEW
        elif (
            review_cost.level == ReviewCostLevel.HIGH
            or routes != [MaintainerReviewRoute.GENERAL]
            or history.rejected_or_superseded_count > 0
            or history.high_significance_count > 0
            or history.unresolved_count >= 3
        ):
            recommendation = MaintainerAttentionRecommendation.HIGH_PRIORITY_REVIEW
        else:
            recommendation = MaintainerAttentionRecommendation.ROUTINE_REVIEW

        warnings: list[str] = []
        if not readiness.policy_present:
            warnings.append(_MISSING_POLICY_WARNING)
        if readiness.merge_conflict is None:
            warnings.append(_UNKNOWN_MERGE_WARNING)
        if not history.source_history_complete:
            warnings.append(_INCOMPLETE_HISTORY_WARNING)
        if not history.ranking_coverage_complete:
            warnings.append(_INCOMPLETE_RANKING_WARNING)
        if history.result_truncated:
            warnings.append(_TRUNCATED_RELATED_WARNING)

        return MaintainerAttentionAssessment(
            repository=brief_input.repository,
            pull_request=brief_input.pull_request,
            assessed_at=brief_input.generated_at,
            recommendation=recommendation,
            routes=routes,
            reasons=reasons,
            warnings=list(dict.fromkeys(warnings)),
        )


def _recommended_actions(
    brief_input: MaintainerBriefInput,
    attention: MaintainerAttentionAssessment,
) -> list[str]:
    readiness = brief_input.readiness
    history = brief_input.historical_context
    review_cost = brief_input.review_cost
    actions: list[str] = []
    if readiness.policy_blocked:
        actions.append(
            "Address blocking repository-policy findings before maintainer review."
        )
    if readiness.merge_conflict is True:
        actions.append("Resolve merge conflicts and refresh the assessment.")
    if readiness.required_checks_failed:
        actions.append("Fix failed required checks before maintainer review.")
    if readiness.changes_requested_count:
        actions.append(
            "Address active changes-requested reviews before requesting another "
            "maintainer pass."
        )
    if readiness.draft:
        actions.append(
            "Mark the pull request ready after the proposed change is complete."
        )
    if readiness.required_checks_pending:
        actions.append(
            "Wait for required checks to complete before maintainer review."
        )
    if MaintainerReviewRoute.SECURITY in attention.routes:
        actions.append("Request a security-focused review.")
    if MaintainerReviewRoute.DATABASE in attention.routes:
        actions.append("Request a database or migration review.")
    if MaintainerReviewRoute.RELEASE_OR_DEPLOYMENT in attention.routes:
        actions.append("Request a deployment or release-engineering review.")
    if MaintainerReviewRoute.ARCHITECTURE in attention.routes:
        actions.append("Compare the proposal with related architectural decisions.")
    if (
        review_cost.level in {ReviewCostLevel.HIGH, ReviewCostLevel.CRITICAL}
        and set(review_cost.primary_drivers)
        & {ReviewCostSignal.CHANGE_SIZE, ReviewCostSignal.CHANGE_DISPERSION}
    ):
        actions.append(
            "Consider splitting the pull request into smaller reviewable changes."
        )
    if history.rejected_or_superseded_count:
        actions.append(
            "Explain how the proposal differs from rejected or superseded related work."
        )
    if history.unresolved_count:
        actions.append("Resolve or acknowledge unresolved related historical work.")
    if not attention.complete:
        actions.append(
            "Inspect the reported coverage warnings before relying on this brief."
        )
    if not actions:
        actions.append("Proceed with normal maintainer review.")
    return list(dict.fromkeys(actions))


class MaintainerBriefService:
    """Build one complete deterministic maintainer brief."""

    def __init__(
        self,
        *,
        attention_service: MaintainerAttentionService | None = None,
    ) -> None:
        self._attention_service = attention_service or MaintainerAttentionService()

    def build(self, brief_input: MaintainerBriefInput) -> MaintainerBrief:
        attention = self._attention_service.assess(brief_input)
        warnings = list(
            dict.fromkeys(
                [
                    *brief_input.review_cost.warnings,
                    *brief_input.related_work.warnings,
                    *attention.warnings,
                ]
            )
        )
        return MaintainerBrief(
            repository=brief_input.repository,
            pull_request=brief_input.pull_request,
            generated_at=brief_input.generated_at,
            readiness=brief_input.readiness,
            path_risk=brief_input.path_risk,
            historical_context=brief_input.historical_context,
            attention=attention,
            review_cost=brief_input.review_cost,
            related_work=brief_input.related_work,
            recommended_actions=_recommended_actions(brief_input, attention),
            warnings=warnings,
        )
