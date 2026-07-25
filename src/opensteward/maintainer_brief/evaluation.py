"""Offline behavioral evaluation for deterministic maintainer briefs."""

from typing import Self

from pydantic import Field, computed_field, field_validator, model_validator

from opensteward.knowledge import StrictKnowledgeModel
from opensteward.maintainer_brief.models import (
    MAX_MAINTAINER_EVALUATION_CASES,
    MaintainerAttentionReasonKind,
    MaintainerAttentionRecommendation,
    MaintainerBriefInput,
    MaintainerReviewRoute,
)
from opensteward.maintainer_brief.service import MaintainerBriefService


def _ordered_unique(values: list[object], expected: list[object], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{name} must be unique.")
    if values != [value for value in expected if value in values]:
        raise ValueError(f"{name} must use the exact enum order.")


def _unique_actions(actions: list[str]) -> list[str]:
    if any(not action for action in actions):
        raise ValueError("Evaluation actions must be non-empty.")
    if len(actions) != len(set(actions)):
        raise ValueError("Evaluation actions must be unique.")
    return actions


class MaintainerBriefEvaluationExpectation(StrictKnowledgeModel):
    """Required subset of one deterministic brief outcome."""

    recommendation: MaintainerAttentionRecommendation
    required_routes: list[MaintainerReviewRoute] = Field(default_factory=list)
    required_reason_kinds: list[MaintainerAttentionReasonKind] = Field(
        default_factory=list
    )
    required_actions: list[str] = Field(default_factory=list)

    @field_validator("required_actions")
    @classmethod
    def validate_actions(cls, actions: list[str]) -> list[str]:
        return _unique_actions(actions)

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        _ordered_unique(
            self.required_routes,
            list(MaintainerReviewRoute),
            "Required routes",
        )
        _ordered_unique(
            self.required_reason_kinds,
            list(MaintainerAttentionReasonKind),
            "Required reason kinds",
        )
        return self


class MaintainerBriefEvaluationCase(StrictKnowledgeModel):
    """One complete offline input and expected behavior."""

    case_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    brief_input: MaintainerBriefInput
    expectation: MaintainerBriefEvaluationExpectation


class MaintainerBriefEvaluationCaseResult(StrictKnowledgeModel):
    """Comparison outcome for one evaluation case."""

    case_id: str = Field(min_length=1)
    passed: bool
    expected_recommendation: MaintainerAttentionRecommendation
    actual_recommendation: MaintainerAttentionRecommendation
    missing_routes: list[MaintainerReviewRoute]
    missing_reason_kinds: list[MaintainerAttentionReasonKind]
    missing_actions: list[str]

    @field_validator("missing_actions")
    @classmethod
    def validate_actions(cls, actions: list[str]) -> list[str]:
        return _unique_actions(actions)

    @model_validator(mode="after")
    def validate_passed(self) -> Self:
        _ordered_unique(
            self.missing_routes,
            list(MaintainerReviewRoute),
            "Missing routes",
        )
        _ordered_unique(
            self.missing_reason_kinds,
            list(MaintainerAttentionReasonKind),
            "Missing reason kinds",
        )
        expected_passed = (
            self.actual_recommendation == self.expected_recommendation
            and not self.missing_routes
            and not self.missing_reason_kinds
            and not self.missing_actions
        )
        if self.passed != expected_passed:
            raise ValueError("passed must equal the complete expectation comparison.")
        return self


class MaintainerBriefEvaluationReport(StrictKnowledgeModel):
    """Aggregate deterministic evaluation results."""

    total_cases: int = Field(ge=0)
    passed_cases: int = Field(ge=0)
    failed_cases: int = Field(ge=0)
    results: list[MaintainerBriefEvaluationCaseResult]

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if self.total_cases != len(self.results):
            raise ValueError("total_cases must equal the result count.")
        if self.passed_cases + self.failed_cases != self.total_cases:
            raise ValueError("Passed and failed case counts must equal total_cases.")
        if self.passed_cases != sum(result.passed for result in self.results):
            raise ValueError("passed_cases must equal the passed result count.")
        case_ids = [result.case_id for result in self.results]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation result case IDs must be unique.")
        if case_ids != sorted(case_ids):
            raise ValueError("Evaluation results must be sorted by case_id.")
        return self

    @computed_field
    @property
    def pass_rate_basis_points(self) -> int:
        if self.total_cases == 0:
            return 10_000
        return (
            self.passed_cases * 10_000 + self.total_cases // 2
        ) // self.total_cases


def evaluate_maintainer_brief_cases(
    cases: list[MaintainerBriefEvaluationCase],
    *,
    service: MaintainerBriefService | None = None,
) -> MaintainerBriefEvaluationReport:
    """Evaluate deterministic brief behavior without external services."""

    if len(cases) > MAX_MAINTAINER_EVALUATION_CASES:
        raise ValueError("Maintainer brief evaluation exceeds the case limit.")
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Maintainer brief evaluation case IDs must be unique.")
    selected_service = service or MaintainerBriefService()
    results: list[MaintainerBriefEvaluationCaseResult] = []
    for case in sorted(cases, key=lambda item: item.case_id):
        brief = selected_service.build(case.brief_input)
        expectation = case.expectation
        actual_reason_kinds = {reason.kind for reason in brief.attention.reasons}
        missing_routes = [
            route for route in expectation.required_routes if route not in brief.routes
        ]
        missing_reasons = [
            kind
            for kind in expectation.required_reason_kinds
            if kind not in actual_reason_kinds
        ]
        missing_actions = [
            action
            for action in expectation.required_actions
            if action not in brief.recommended_actions
        ]
        passed = (
            brief.recommendation == expectation.recommendation
            and not missing_routes
            and not missing_reasons
            and not missing_actions
        )
        results.append(
            MaintainerBriefEvaluationCaseResult(
                case_id=case.case_id,
                passed=passed,
                expected_recommendation=expectation.recommendation,
                actual_recommendation=brief.recommendation,
                missing_routes=missing_routes,
                missing_reason_kinds=missing_reasons,
                missing_actions=missing_actions,
            )
        )
    passed_cases = sum(result.passed for result in results)
    return MaintainerBriefEvaluationReport(
        total_cases=len(results),
        passed_cases=passed_cases,
        failed_cases=len(results) - passed_cases,
        results=results,
    )
