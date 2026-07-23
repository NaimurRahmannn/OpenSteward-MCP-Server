"""Repository contribution policy evaluation for OpenSteward."""

from collections.abc import Iterable

from opensteward.policy.inputs import ContributionPolicyInput
from opensteward.policy.models import (
    ContributionCategory,
    RepositoryPolicy,
    RiskLevel,
)
from opensteward.policy.path_matcher import match_protected_paths
from opensteward.policy.results import (
    FindingSeverity,
    PolicyEvaluationResult,
    PolicyFinding,
    PolicyFindingStatus,
    PolicyRule,
)


RISK_ORDER: dict[RiskLevel, int] = {
    RiskLevel.LOW: 1,
    RiskLevel.MEDIUM: 2,
    RiskLevel.HIGH: 3,
    RiskLevel.CRITICAL: 4,
}


RISK_TO_SEVERITY: dict[RiskLevel, FindingSeverity] = {
    RiskLevel.LOW: FindingSeverity.LOW,
    RiskLevel.MEDIUM: FindingSeverity.MEDIUM,
    RiskLevel.HIGH: FindingSeverity.HIGH,
    RiskLevel.CRITICAL: FindingSeverity.CRITICAL,
}


def _format_categories(
    categories: Iterable[ContributionCategory],
) -> str:
    """Return contribution categories in deterministic display order."""

    return ", ".join(
        sorted(category.value for category in categories)
    )


def _find_highest_risk(
    risks: Iterable[RiskLevel],
) -> RiskLevel | None:
    """Return the highest risk level from an iterable."""

    risk_values = list(risks)

    if not risk_values:
        return None

    return max(
        risk_values,
        key=RISK_ORDER.__getitem__,
    )


def _calculate_required_approvals(
    policy: RepositoryPolicy,
    contribution: ContributionPolicyInput,
) -> int:
    """Calculate the effective human approval requirement."""

    requirements = [
        policy.review.required_approvals.default,
    ]

    if ContributionCategory.PUBLIC_API in contribution.categories:
        requirements.append(
            policy.review.required_approvals.public_api
        )

    if ContributionCategory.SECURITY in contribution.categories:
        requirements.append(
            policy.review.required_approvals.security
        )

    return max(requirements)


def _evaluate_diff_size(
    policy: RepositoryPolicy,
    contribution: ContributionPolicyInput,
) -> PolicyFinding:
    """Evaluate the preferred contribution diff size."""

    preferred_maximum = (
        policy.pull_requests.preferred_maximum_diff_lines
    )

    if contribution.diff_lines > preferred_maximum:
        return PolicyFinding(
            rule=PolicyRule.PREFERRED_DIFF_SIZE,
            status=PolicyFindingStatus.WARNING,
            severity=FindingSeverity.MEDIUM,
            message=(
                f"The contribution changes {contribution.diff_lines} lines, "
                f"which exceeds the preferred maximum of "
                f"{preferred_maximum} lines."
            ),
            evidence=[
                f"additions:{contribution.additions}",
                f"deletions:{contribution.deletions}",
                f"diff_lines:{contribution.diff_lines}",
                f"preferred_maximum:{preferred_maximum}",
            ],
            remediation=(
                "Consider splitting the contribution into smaller, "
                "independently reviewable changes."
            ),
        )

    return PolicyFinding(
        rule=PolicyRule.PREFERRED_DIFF_SIZE,
        status=PolicyFindingStatus.PASSED,
        severity=FindingSeverity.INFO,
        message=(
            f"The contribution changes {contribution.diff_lines} lines "
            f"and is within the preferred maximum of "
            f"{preferred_maximum} lines."
        ),
        evidence=[
            f"diff_lines:{contribution.diff_lines}",
            f"preferred_maximum:{preferred_maximum}",
        ],
    )


def _evaluate_tests(
    policy: RepositoryPolicy,
    contribution: ContributionPolicyInput,
) -> PolicyFinding:
    """Evaluate whether required test changes are present."""

    required_categories = (
        set(contribution.categories)
        & set(policy.pull_requests.tests_required_for)
    )

    if not required_categories:
        return PolicyFinding(
            rule=PolicyRule.REQUIRED_TESTS,
            status=PolicyFindingStatus.INFO,
            severity=FindingSeverity.INFO,
            message=(
                "The contribution does not trigger a configured "
                "test-change requirement."
            ),
            evidence=[
                (
                    "contribution_categories:"
                    f"{_format_categories(contribution.categories)}"
                )
            ],
        )

    formatted_categories = _format_categories(required_categories)

    if contribution.tests_changed:
        return PolicyFinding(
            rule=PolicyRule.REQUIRED_TESTS,
            status=PolicyFindingStatus.PASSED,
            severity=FindingSeverity.INFO,
            message=(
                "Required test changes are present for the triggered "
                f"categories: {formatted_categories}."
            ),
            evidence=[
                f"required_categories:{formatted_categories}",
                "tests_changed:true",
            ],
        )

    return PolicyFinding(
        rule=PolicyRule.REQUIRED_TESTS,
        status=PolicyFindingStatus.FAILED,
        severity=FindingSeverity.HIGH,
        message=(
            "Test changes are required for the triggered categories: "
            f"{formatted_categories}."
        ),
        evidence=[
            f"required_categories:{formatted_categories}",
            "tests_changed:false",
        ],
        remediation=(
            "Add tests that demonstrate the intended behavior and protect "
            "against regression."
        ),
    )


def _evaluate_linked_issue(
    policy: RepositoryPolicy,
    contribution: ContributionPolicyInput,
) -> PolicyFinding:
    """Evaluate whether a required issue is linked."""

    required_categories = (
        set(contribution.categories)
        & set(policy.pull_requests.linked_issue_required_for)
    )

    if not required_categories:
        return PolicyFinding(
            rule=PolicyRule.LINKED_ISSUE,
            status=PolicyFindingStatus.INFO,
            severity=FindingSeverity.INFO,
            message=(
                "The contribution does not trigger a linked-issue "
                "requirement."
            ),
            evidence=[
                (
                    "contribution_categories:"
                    f"{_format_categories(contribution.categories)}"
                )
            ],
        )

    formatted_categories = _format_categories(required_categories)

    if contribution.has_linked_issue:
        linked_numbers = ",".join(
            str(number)
            for number in contribution.linked_issue_numbers
        )

        return PolicyFinding(
            rule=PolicyRule.LINKED_ISSUE,
            status=PolicyFindingStatus.PASSED,
            severity=FindingSeverity.INFO,
            message=(
                "A linked issue is present for the triggered categories: "
                f"{formatted_categories}."
            ),
            evidence=[
                f"required_categories:{formatted_categories}",
                f"linked_issue_numbers:{linked_numbers}",
            ],
        )

    return PolicyFinding(
        rule=PolicyRule.LINKED_ISSUE,
        status=PolicyFindingStatus.FAILED,
        severity=FindingSeverity.HIGH,
        message=(
            "A linked issue is required for the triggered categories: "
            f"{formatted_categories}."
        ),
        evidence=[
            f"required_categories:{formatted_categories}",
            "linked_issue_numbers:none",
        ],
        remediation=(
            "Link an existing issue or create a design discussion before "
            "requesting detailed review."
        ),
    )


def _evaluate_protected_paths(
    policy: RepositoryPolicy,
    contribution: ContributionPolicyInput,
) -> tuple[
    list[PolicyFinding],
    RiskLevel | None,
    bool,
]:
    """Evaluate changed files against protected-path rules."""

    matches = match_protected_paths(
        paths=contribution.changed_files,
        rules=policy.protected_paths,
    )

    findings: list[PolicyFinding] = []

    for match in matches:
        status = (
            PolicyFindingStatus.WARNING
            if match.human_review_required
            else PolicyFindingStatus.INFO
        )

        findings.append(
            PolicyFinding(
                rule=PolicyRule.PROTECTED_PATH,
                status=status,
                severity=RISK_TO_SEVERITY[match.risk],
                message=match.explanation,
                evidence=[
                    f"path:{match.path}",
                    f"pattern:{match.pattern}",
                    f"risk:{match.risk.value}",
                    (
                        "human_review_required:"
                        f"{str(match.human_review_required).lower()}"
                    ),
                ],
                remediation=(
                    "Route the contribution to a maintainer familiar with "
                    "this protected area."
                    if match.human_review_required
                    else None
                ),
            )
        )

    highest_risk = _find_highest_risk(
        match.risk
        for match in matches
    )

    protected_review_required = any(
        match.human_review_required
        for match in matches
    )

    return findings, highest_risk, protected_review_required


def _evaluate_approvals(
    policy: RepositoryPolicy,
    contribution: ContributionPolicyInput,
    required_approvals: int,
) -> PolicyFinding:
    """Evaluate the effective human approval requirement."""

    remaining = max(
        required_approvals - contribution.current_approvals,
        0,
    )

    if remaining == 0:
        return PolicyFinding(
            rule=PolicyRule.REQUIRED_APPROVALS,
            status=PolicyFindingStatus.PASSED,
            severity=FindingSeverity.INFO,
            message=(
                f"The contribution has {contribution.current_approvals} "
                f"approval(s), meeting the requirement of "
                f"{required_approvals}."
            ),
            evidence=[
                f"current_approvals:{contribution.current_approvals}",
                f"required_approvals:{required_approvals}",
            ],
        )

    sensitive_change = any(
        category in contribution.categories
        for category in (
            ContributionCategory.PUBLIC_API,
            ContributionCategory.SECURITY,
        )
    )

    severity = (
        FindingSeverity.HIGH
        if sensitive_change
        else FindingSeverity.MEDIUM
    )

    return PolicyFinding(
        rule=PolicyRule.REQUIRED_APPROVALS,
        status=PolicyFindingStatus.FAILED,
        severity=severity,
        message=(
            f"The contribution has {contribution.current_approvals} "
            f"approval(s), but {required_approvals} are required."
        ),
        evidence=[
            f"current_approvals:{contribution.current_approvals}",
            f"required_approvals:{required_approvals}",
            f"remaining_approvals:{remaining}",
        ],
        remediation=(
            f"Obtain {remaining} additional valid human approval(s)."
        ),
    )


def evaluate_contribution_policy(
    policy: RepositoryPolicy,
    contribution: ContributionPolicyInput,
) -> PolicyEvaluationResult:
    """Evaluate one contribution against a repository policy."""

    findings: list[PolicyFinding] = [
        _evaluate_diff_size(
            policy=policy,
            contribution=contribution,
        ),
        _evaluate_tests(
            policy=policy,
            contribution=contribution,
        ),
        _evaluate_linked_issue(
            policy=policy,
            contribution=contribution,
        ),
    ]

    (
        protected_path_findings,
        highest_protected_path_risk,
        protected_review_required,
    ) = _evaluate_protected_paths(
        policy=policy,
        contribution=contribution,
    )

    findings.extend(protected_path_findings)

    required_approvals = _calculate_required_approvals(
        policy=policy,
        contribution=contribution,
    )

    approval_finding = _evaluate_approvals(
        policy=policy,
        contribution=contribution,
        required_approvals=required_approvals,
    )

    findings.append(approval_finding)

    remaining_approvals = max(
        required_approvals - contribution.current_approvals,
        0,
    )

    compliant = not any(
        finding.status == PolicyFindingStatus.FAILED
        for finding in findings
    )

    return PolicyEvaluationResult(
        compliant=compliant,
        requires_human_review=(
            required_approvals > 0
            or protected_review_required
        ),
        required_approvals=required_approvals,
        current_approvals=contribution.current_approvals,
        remaining_approvals=remaining_approvals,
        highest_protected_path_risk=highest_protected_path_risk,
        findings=findings,
    )