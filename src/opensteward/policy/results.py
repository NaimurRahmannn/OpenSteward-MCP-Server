"""Structured results produced by the repository policy engine."""

from enum import StrEnum

from pydantic import Field

from opensteward.policy.models import RiskLevel, StrictPolicyModel


class PolicyRule(StrEnum):
    """Repository policy rules currently evaluated by OpenSteward."""

    PREFERRED_DIFF_SIZE = "preferred_diff_size"
    REQUIRED_TESTS = "required_tests"
    LINKED_ISSUE = "linked_issue"
    PROTECTED_PATH = "protected_path"
    REQUIRED_APPROVALS = "required_approvals"


class PolicyFindingStatus(StrEnum):
    """Outcome of evaluating one repository policy rule."""

    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    INFO = "info"


class FindingSeverity(StrEnum):
    """Importance of an individual policy finding."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PolicyFinding(StrictPolicyModel):
    """Explainable result from evaluating one policy rule."""

    rule: PolicyRule
    status: PolicyFindingStatus
    severity: FindingSeverity
    message: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    remediation: str | None = None


class PolicyEvaluationResult(StrictPolicyModel):
    """Complete policy evaluation for one contribution."""

    compliant: bool
    requires_human_review: bool

    required_approvals: int = Field(ge=0, le=20)
    current_approvals: int = Field(ge=0, le=100)
    remaining_approvals: int = Field(ge=0, le=20)

    highest_protected_path_risk: RiskLevel | None = None

    findings: list[PolicyFinding] = Field(default_factory=list)