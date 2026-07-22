"""Domain models used by OpenSteward."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ReviewCostLevel(StrEnum):
    """Human-readable review-cost categories."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ReviewCostFactorName(StrEnum):
    """Signals currently used to estimate review cost."""

    CHANGE_SIZE = "change_size"
    COMPONENT_RISK = "component_risk"
    TEST_GAP = "test_gap"
    OWNERSHIP_DISPERSION = "ownership_dispersion"
    PUBLIC_API_IMPACT = "public_api_impact"
    CI_RISK = "ci_risk"
    REVIEWER_LOAD = "reviewer_load"


class ReviewCostFactors(BaseModel):
    """Normalized signals used when calculating review cost.

    Each factor must be a number from 0 to 100.

    A larger number means that the factor is expected to require
    more maintainer attention.
    """

    model_config = ConfigDict(extra="forbid")

    change_size: int = Field(ge=0, le=100)
    component_risk: int = Field(ge=0, le=100)
    test_gap: int = Field(ge=0, le=100)
    ownership_dispersion: int = Field(ge=0, le=100)
    public_api_impact: int = Field(ge=0, le=100)
    ci_risk: int = Field(ge=0, le=100)
    reviewer_load: int = Field(ge=0, le=100)


class ReviewCostContribution(BaseModel):
    """Explain how one factor contributed to the final score."""

    model_config = ConfigDict(extra="forbid")

    factor: ReviewCostFactorName
    raw_score: int = Field(ge=0, le=100)
    weight: float = Field(ge=0, le=1)
    weighted_score: float = Field(ge=0, le=100)
    explanation: str = Field(min_length=1)


class ReviewCostResult(BaseModel):
    """Complete explainable review-cost result."""

    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=0, le=100)
    level: ReviewCostLevel
    contributions: list[ReviewCostContribution]
    summary: str = Field(min_length=1)