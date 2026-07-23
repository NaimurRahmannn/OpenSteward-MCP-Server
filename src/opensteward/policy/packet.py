"""Maintainer-facing policy packets built from detailed findings."""

from collections.abc import Iterable
from enum import StrEnum

from pydantic import Field

from opensteward.policy.models import StrictPolicyModel
from opensteward.policy.results import (
    FindingSeverity,
    PolicyEvaluationResult,
    PolicyFinding,
    PolicyFindingStatus,
    PolicyRule,
)


class MaintainerRecommendation(StrEnum):
    """Recommended maintainer action after policy evaluation."""

    READY_FOR_REVIEW = "ready_for_review"
    REVIEW_WITH_CAUTION = "review_with_caution"
    REQUEST_CHANGES = "request_changes"


class MaintainerPacketItem(StrictPolicyModel):
    """One concise item shown in a maintainer policy packet."""

    rule: PolicyRule
    severity: FindingSeverity
    message: str = Field(min_length=1)
    next_action: str | None = None


class MaintainerPolicyPacket(StrictPolicyModel):
    """Concise maintainer view of a policy evaluation."""

    recommendation: MaintainerRecommendation
    ready_for_detailed_review: bool

    summary: str = Field(min_length=1)
    approval_summary: str = Field(min_length=1)

    blocking_requirements: list[MaintainerPacketItem] = Field(
        default_factory=list,
    )

    warnings: list[MaintainerPacketItem] = Field(
        default_factory=list,
    )

    suggested_next_actions: list[str] = Field(
        default_factory=list,
    )

    passed_checks: int = Field(ge=0)
    informational_checks: int = Field(ge=0)


def _create_packet_item(
    finding: PolicyFinding,
) -> MaintainerPacketItem:
    """Convert one detailed policy finding into a packet item."""

    return MaintainerPacketItem(
        rule=finding.rule,
        severity=finding.severity,
        message=finding.message,
        next_action=finding.remediation,
    )


def _deduplicate_actions(
    actions: Iterable[str | None],
) -> list[str]:
    """Return unique, non-empty actions while preserving order."""

    unique_actions: list[str] = []
    seen: set[str] = set()

    for action in actions:
        if action is None or action in seen:
            continue

        seen.add(action)
        unique_actions.append(action)

    return unique_actions


def _format_count(
    count: int,
    singular: str,
) -> str:
    """Format a count with simple singular or plural wording."""

    label = singular if count == 1 else f"{singular}s"

    return f"{count} {label}"


def _build_summary(
    blocking_count: int,
    warning_count: int,
) -> str:
    """Build the packet's concise maintainer summary."""

    if blocking_count:
        return (
            "Request changes before detailed review: "
            f"{_format_count(blocking_count, 'blocking requirement')} "
            f"and {_format_count(warning_count, 'warning')} "
            "need attention."
        )

    if warning_count:
        return (
            "The configured requirements are satisfied, but "
            f"{_format_count(warning_count, 'warning')} "
            "needs maintainer attention."
        )

    return (
        "The configured requirements are satisfied and no policy "
        "warnings were found."
    )


def _build_approval_summary(
    evaluation: PolicyEvaluationResult,
) -> str:
    """Describe approval progress in one sentence."""

    if evaluation.required_approvals == 0:
        return (
            "No human approvals are required by the active policy."
        )

    if evaluation.remaining_approvals == 0:
        return (
            "Approval requirement met: "
            f"{evaluation.current_approvals} recorded and "
            f"{evaluation.required_approvals} required."
        )

    return (
        "Approval requirement not met: "
        f"{evaluation.current_approvals} recorded, "
        f"{evaluation.required_approvals} required, and "
        f"{evaluation.remaining_approvals} remaining."
    )


def build_maintainer_policy_packet(
    evaluation: PolicyEvaluationResult,
) -> MaintainerPolicyPacket:
    """Build a concise maintainer packet from detailed findings."""

    blocking_findings = [
        finding
        for finding in evaluation.findings
        if finding.status == PolicyFindingStatus.FAILED
    ]

    warning_findings = [
        finding
        for finding in evaluation.findings
        if finding.status == PolicyFindingStatus.WARNING
    ]

    blocking_requirements = [
        _create_packet_item(finding)
        for finding in blocking_findings
    ]

    warnings = [
        _create_packet_item(finding)
        for finding in warning_findings
    ]

    suggested_next_actions = _deduplicate_actions(
        finding.remediation
        for finding in [
            *blocking_findings,
            *warning_findings,
        ]
    )

    if blocking_findings:
        recommendation = MaintainerRecommendation.REQUEST_CHANGES
    elif warning_findings:
        recommendation = (
            MaintainerRecommendation.REVIEW_WITH_CAUTION
        )
    else:
        recommendation = MaintainerRecommendation.READY_FOR_REVIEW

    passed_checks = sum(
        finding.status == PolicyFindingStatus.PASSED
        for finding in evaluation.findings
    )

    informational_checks = sum(
        finding.status == PolicyFindingStatus.INFO
        for finding in evaluation.findings
    )

    return MaintainerPolicyPacket(
        recommendation=recommendation,
        ready_for_detailed_review=evaluation.compliant,
        summary=_build_summary(
            blocking_count=len(blocking_findings),
            warning_count=len(warning_findings),
        ),
        approval_summary=_build_approval_summary(evaluation),
        blocking_requirements=blocking_requirements,
        warnings=warnings,
        suggested_next_actions=suggested_next_actions,
        passed_checks=passed_checks,
        informational_checks=informational_checks,
    )