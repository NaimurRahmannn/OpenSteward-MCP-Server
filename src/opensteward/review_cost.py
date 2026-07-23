"""Explainable review-cost calculation for OpenSteward."""

from opensteward.models import (
    ReviewCostContribution,
    ReviewCostFactorName,
    ReviewCostFactors,
    ReviewCostLevel,
    ReviewCostResult,
)


REVIEW_COST_WEIGHTS: dict[ReviewCostFactorName, float] = {
    ReviewCostFactorName.CHANGE_SIZE: 0.20,
    ReviewCostFactorName.COMPONENT_RISK: 0.20,
    ReviewCostFactorName.TEST_GAP: 0.15,
    ReviewCostFactorName.OWNERSHIP_DISPERSION: 0.15,
    ReviewCostFactorName.PUBLIC_API_IMPACT: 0.10,
    ReviewCostFactorName.CI_RISK: 0.10,
    ReviewCostFactorName.REVIEWER_LOAD: 0.10,
}


FACTOR_LABELS: dict[ReviewCostFactorName, str] = {
    ReviewCostFactorName.CHANGE_SIZE: "Change size",
    ReviewCostFactorName.COMPONENT_RISK: "Component risk",
    ReviewCostFactorName.TEST_GAP: "Test gap",
    ReviewCostFactorName.OWNERSHIP_DISPERSION: "Ownership dispersion",
    ReviewCostFactorName.PUBLIC_API_IMPACT: "Public API impact",
    ReviewCostFactorName.CI_RISK: "CI risk",
    ReviewCostFactorName.REVIEWER_LOAD: "Reviewer load",
}


FACTOR_MEANINGS: dict[ReviewCostFactorName, str] = {
    ReviewCostFactorName.CHANGE_SIZE: (
        "the amount and structural breadth of code that requires inspection"
    ),
    ReviewCostFactorName.COMPONENT_RISK: (
        "the sensitivity and operational importance of the affected components"
    ),
    ReviewCostFactorName.TEST_GAP: (
        "the amount of changed behavior that is not supported by updated tests"
    ),
    ReviewCostFactorName.OWNERSHIP_DISPERSION: (
        "the number of ownership boundaries involved in the change"
    ),
    ReviewCostFactorName.PUBLIC_API_IMPACT: (
        "the potential effect on users, integrations, and compatibility"
    ),
    ReviewCostFactorName.CI_RISK: (
        "the uncertainty created by failing, missing, or incomplete automated checks"
    ),
    ReviewCostFactorName.REVIEWER_LOAD: (
        "the current review pressure on maintainers familiar with the affected code"
    ),
}


SUMMARY_BY_LEVEL: dict[ReviewCostLevel, str] = {
    ReviewCostLevel.LOW: (
        "This contribution is expected to require relatively little maintainer attention."
    ),
    ReviewCostLevel.MEDIUM: (
        "This contribution is expected to require a moderate amount of maintainer attention."
    ),
    ReviewCostLevel.HIGH: (
        "This contribution is expected to require substantial maintainer attention."
    ),
    ReviewCostLevel.CRITICAL: (
        "This contribution is expected to require intensive and carefully coordinated review."
    ),
}


def classify_review_cost(score: int) -> ReviewCostLevel:
    """Convert a numeric review-cost score into a readable category."""

    if not 0 <= score <= 100:
        raise ValueError("Review-cost score must be between 0 and 100.")

    if score >= 80:
        return ReviewCostLevel.CRITICAL

    if score >= 60:
        return ReviewCostLevel.HIGH

    if score >= 30:
        return ReviewCostLevel.MEDIUM

    return ReviewCostLevel.LOW


def describe_factor(
    factor: ReviewCostFactorName,
    raw_score: int,
) -> str:
    """Create a human-readable explanation for one review-cost factor."""

    if not 0 <= raw_score <= 100:
        raise ValueError("Factor score must be between 0 and 100.")

    if raw_score >= 80:
        impact_level = "very high"
    elif raw_score >= 60:
        impact_level = "high"
    elif raw_score >= 30:
        impact_level = "moderate"
    else:
        impact_level = "low"

    label = FACTOR_LABELS[factor]
    meaning = FACTOR_MEANINGS[factor]

    return (
        f"{label} has a {impact_level} signal of {raw_score}/100. "
        f"This represents {meaning}."
    )


def calculate_review_cost(
    factors: ReviewCostFactors,
) -> ReviewCostResult:
    """Calculate an explainable weighted review-cost result."""

    contributions: list[ReviewCostContribution] = []

    for factor, weight in REVIEW_COST_WEIGHTS.items():
        raw_score = getattr(factors, factor.value)
        weighted_score = round(raw_score * weight, 2)

        contributions.append(
            ReviewCostContribution(
                factor=factor,
                raw_score=raw_score,
                weight=weight,
                weighted_score=weighted_score,
                explanation=describe_factor(
                    factor=factor,
                    raw_score=raw_score,
                ),
            )
        )

    total_weighted_score = sum(
        contribution.weighted_score
        for contribution in contributions
    )

    final_score = round(total_weighted_score)
    level = classify_review_cost(final_score)

    return ReviewCostResult(
        score=final_score,
        level=level,
        contributions=contributions,
        summary=SUMMARY_BY_LEVEL[level],
    )