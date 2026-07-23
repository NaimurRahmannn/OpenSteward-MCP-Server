"""Tests for the OpenSteward review-cost engine."""

import pytest

from opensteward.models import (
    ReviewCostFactorName,
    ReviewCostFactors,
    ReviewCostLevel,
)
from opensteward.review_cost import (
    REVIEW_COST_WEIGHTS,
    calculate_review_cost,
    classify_review_cost,
)


def create_sample_factors() -> ReviewCostFactors:
    """Create a representative set of review-cost factors."""

    return ReviewCostFactors(
        change_size=80,
        component_risk=90,
        test_gap=70,
        ownership_dispersion=40,
        public_api_impact=60,
        ci_risk=20,
        reviewer_load=50,
    )


def test_review_cost_weights_total_one() -> None:
    assert sum(REVIEW_COST_WEIGHTS.values()) == pytest.approx(1.0)


def test_calculate_review_cost_returns_expected_score() -> None:
    result = calculate_review_cost(create_sample_factors())

    assert result.score == 64
    assert result.level == ReviewCostLevel.HIGH
    assert (
        result.summary
        == "This contribution is expected to require substantial maintainer attention."
    )


def test_calculate_review_cost_returns_all_contributions() -> None:
    result = calculate_review_cost(create_sample_factors())

    assert len(result.contributions) == 7

    returned_factors = {
        contribution.factor
        for contribution in result.contributions
    }

    assert returned_factors == set(ReviewCostFactorName)


def test_component_risk_contribution_is_explainable() -> None:
    result = calculate_review_cost(create_sample_factors())

    component_risk = next(
        contribution
        for contribution in result.contributions
        if contribution.factor == ReviewCostFactorName.COMPONENT_RISK
    )

    assert component_risk.raw_score == 90
    assert component_risk.weight == pytest.approx(0.20)
    assert component_risk.weighted_score == pytest.approx(18)
    assert "very high" in component_risk.explanation
    assert "90/100" in component_risk.explanation


@pytest.mark.parametrize(
    ("score", "expected_level"),
    [
        (0, ReviewCostLevel.LOW),
        (29, ReviewCostLevel.LOW),
        (30, ReviewCostLevel.MEDIUM),
        (59, ReviewCostLevel.MEDIUM),
        (60, ReviewCostLevel.HIGH),
        (79, ReviewCostLevel.HIGH),
        (80, ReviewCostLevel.CRITICAL),
        (100, ReviewCostLevel.CRITICAL),
    ],
)
def test_classify_review_cost_boundaries(
    score: int,
    expected_level: ReviewCostLevel,
) -> None:
    assert classify_review_cost(score) == expected_level


@pytest.mark.parametrize("invalid_score", [-1, 101])
def test_classify_review_cost_rejects_invalid_scores(
    invalid_score: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        classify_review_cost(invalid_score)


def test_all_zero_factors_produce_zero_score() -> None:
    factors = ReviewCostFactors(
        change_size=0,
        component_risk=0,
        test_gap=0,
        ownership_dispersion=0,
        public_api_impact=0,
        ci_risk=0,
        reviewer_load=0,
    )

    result = calculate_review_cost(factors)

    assert result.score == 0
    assert result.level == ReviewCostLevel.LOW


def test_all_maximum_factors_produce_maximum_score() -> None:
    factors = ReviewCostFactors(
        change_size=100,
        component_risk=100,
        test_gap=100,
        ownership_dispersion=100,
        public_api_impact=100,
        ci_risk=100,
        reviewer_load=100,
    )

    result = calculate_review_cost(factors)

    assert result.score == 100
    assert result.level == ReviewCostLevel.CRITICAL