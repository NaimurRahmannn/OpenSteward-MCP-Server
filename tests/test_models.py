"""Tests for OpenSteward domain models."""

import pytest
from pydantic import ValidationError

from opensteward.models import (
    ReviewCostContribution,
    ReviewCostFactorName,
    ReviewCostFactors,
    ReviewCostLevel,
    ReviewCostResult,
)


def create_valid_factors() -> ReviewCostFactors:
    """Create valid review-cost factors for tests."""

    return ReviewCostFactors(
        change_size=80,
        component_risk=90,
        test_gap=70,
        ownership_dispersion=40,
        public_api_impact=60,
        ci_risk=20,
        reviewer_load=50,
    )


def test_review_cost_factors_accept_valid_values() -> None:
    factors = create_valid_factors()

    assert factors.change_size == 80
    assert factors.component_risk == 90
    assert factors.ci_risk == 20


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("change_size", -1),
        ("change_size", 101),
        ("component_risk", -20),
        ("test_gap", 500),
        ("reviewer_load", 101),
    ],
)
def test_review_cost_factors_reject_out_of_range_values(
    field_name: str,
    invalid_value: int,
) -> None:
    values = create_valid_factors().model_dump()
    values[field_name] = invalid_value

    with pytest.raises(ValidationError):
        ReviewCostFactors(**values)


def test_review_cost_factors_reject_unknown_fields() -> None:
    values = create_valid_factors().model_dump()
    values["unknown_factor"] = 50

    with pytest.raises(ValidationError):
        ReviewCostFactors(**values)


def test_review_cost_contribution_requires_an_explanation() -> None:
    with pytest.raises(ValidationError):
        ReviewCostContribution(
            factor=ReviewCostFactorName.CHANGE_SIZE,
            raw_score=80,
            weight=0.20,
            weighted_score=16,
            explanation="",
        )


def test_review_cost_result_serializes_as_json() -> None:
    contribution = ReviewCostContribution(
        factor=ReviewCostFactorName.COMPONENT_RISK,
        raw_score=90,
        weight=0.20,
        weighted_score=18,
        explanation="The change affects a protected component.",
    )

    result = ReviewCostResult(
        score=72,
        level=ReviewCostLevel.HIGH,
        contributions=[contribution],
        summary="The pull request is expected to require substantial review effort.",
    )

    data = result.model_dump(mode="json")

    assert data["score"] == 72
    assert data["level"] == "high"
    assert data["contributions"][0]["factor"] == "component_risk"
    assert data["contributions"][0]["weighted_score"] == 18