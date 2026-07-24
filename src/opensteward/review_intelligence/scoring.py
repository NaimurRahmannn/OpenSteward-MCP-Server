"""Deterministic explainable review-cost scoring."""

import re
from datetime import UTC, datetime
from pathlib import PurePosixPath

from opensteward.review_intelligence.models import (
    REVIEW_COST_CHANGE_DISPERSION_WEIGHT,
    REVIEW_COST_CHANGE_SIZE_WEIGHT,
    REVIEW_COST_HISTORICAL_COMPLEXITY_WEIGHT,
    REVIEW_COST_RISK_SENSITIVE_PATHS_WEIGHT,
    REVIEW_COST_VALIDATION_GAPS_WEIGHT,
    ReviewCostAssessment,
    ReviewCostAssessmentInput,
    ReviewCostAssessmentOptions,
    ReviewCostPathCategory,
    ReviewCostSignal,
    ReviewCostSignalContribution,
)

_RISK_CATEGORY_ORDER = (
    ReviewCostPathCategory.PROTECTED,
    ReviewCostPathCategory.SECURITY_SENSITIVE,
    ReviewCostPathCategory.DATABASE_MIGRATION,
    ReviewCostPathCategory.AUTOMATION_OR_DEPLOYMENT,
    ReviewCostPathCategory.DEPENDENCY_MANIFEST,
)
_RISK_BASE_SCORES = {
    ReviewCostPathCategory.PROTECTED: 100,
    ReviewCostPathCategory.SECURITY_SENSITIVE: 90,
    ReviewCostPathCategory.DATABASE_MIGRATION: 85,
    ReviewCostPathCategory.AUTOMATION_OR_DEPLOYMENT: 75,
    ReviewCostPathCategory.DEPENDENCY_MANIFEST: 60,
}
_SECURITY_TOKENS = {
    "auth",
    "authentication",
    "authorization",
    "oauth",
    "token",
    "tokens",
    "secret",
    "secrets",
    "permission",
    "permissions",
    "crypto",
    "cryptography",
    "security",
    "access_control",
}
_DEPLOYMENT_SEGMENTS = {
    "deploy",
    "deployment",
    "deployments",
    "k8s",
    "kubernetes",
    "helm",
    "terraform",
    "infrastructure",
}
_DEPENDENCY_BASENAMES = {
    "pyproject.toml",
    "requirements.txt",
    "uv.lock",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "go.mod",
    "go.sum",
    "cargo.toml",
    "cargo.lock",
    "gemfile",
    "gemfile.lock",
    "composer.json",
    "composer.lock",
}
_TEST_SEGMENTS = {"test", "tests", "spec", "specs", "__tests__"}
_DOCUMENTATION_SEGMENTS = {"doc", "docs", "documentation"}
_DOCUMENTATION_SUFFIXES = {".md", ".mdx", ".rst", ".adoc"}

_MISSING_POLICY_WARNING = (
    "Repository policy was unavailable; the default preferred diff size and "
    "no protected paths were used."
)
_UNKNOWN_MERGE_WARNING = "Pull-request merge-conflict state was unavailable."
_NO_HISTORY_WARNING = "Historical related-work context was unavailable."
_INCOMPLETE_HISTORY_WARNING = "Historical source collection was incomplete."
_INCOMPLETE_RANKING_WARNING = "Related-work ranking coverage was incomplete."
_TRUNCATED_RELATED_WARNING = (
    "Related-work results were truncated by the configured final result limit."
)


def _security_sensitive(path: str) -> bool:
    parts = path.casefold().split("/")
    candidates = set(parts)
    basename = parts[-1]
    stem = PurePosixPath(basename).stem
    candidates.update({basename, stem})
    for value in [*parts, stem]:
        candidates.update(re.split(r"[_-]", value))
    return bool(candidates & _SECURITY_TOKENS)


def _database_migration(path: str) -> bool:
    parts = path.casefold().split("/")
    return (
        any(part in {"migration", "migrations", "alembic"} for part in parts)
        or any(parts[index : index + 2] == ["db", "migrate"] for index in range(len(parts)))
        or any(
            parts[index : index + 2] == ["database", "migrations"]
            for index in range(len(parts))
        )
    )


def _automation_or_deployment(path: str) -> bool:
    folded = path.casefold()
    parts = folded.split("/")
    basename = parts[-1]
    return (
        parts[:2] == [".github", "workflows"]
        or basename in {"dockerfile", "docker-compose.yml", "docker-compose.yaml"}
        or any(part in _DEPLOYMENT_SEGMENTS for part in parts[:-1])
    )


def _dependency_manifest(path: str) -> bool:
    basename = path.casefold().split("/")[-1]
    return basename in _DEPENDENCY_BASENAMES or bool(
        re.fullmatch(r"requirements-.+\.txt", basename)
    )


def _test_path(path: str) -> bool:
    parts = path.casefold().split("/")
    basename = parts[-1]
    stem = PurePosixPath(basename).stem
    return (
        any(part in _TEST_SEGMENTS for part in parts[:-1])
        or basename.startswith("test_")
        or stem.endswith("_test")
        or stem.endswith(".spec")
        or stem.endswith(".test")
    )


def _documentation_path(path: str) -> bool:
    folded = path.casefold()
    parts = folded.split("/")
    return (
        any(part in _DOCUMENTATION_SEGMENTS for part in parts[:-1])
        or PurePosixPath(folded).suffix in _DOCUMENTATION_SUFFIXES
    )


def _classify_path(
    path: str,
    protected_paths: set[str],
) -> tuple[ReviewCostPathCategory, ...]:
    categories: list[ReviewCostPathCategory] = []
    if path in protected_paths:
        categories.append(ReviewCostPathCategory.PROTECTED)
    if _security_sensitive(path):
        categories.append(ReviewCostPathCategory.SECURITY_SENSITIVE)
    if _database_migration(path):
        categories.append(ReviewCostPathCategory.DATABASE_MIGRATION)
    if _automation_or_deployment(path):
        categories.append(ReviewCostPathCategory.AUTOMATION_OR_DEPLOYMENT)
    if _dependency_manifest(path):
        categories.append(ReviewCostPathCategory.DEPENDENCY_MANIFEST)
    is_test = _test_path(path)
    is_documentation = _documentation_path(path)
    if is_test:
        categories.append(ReviewCostPathCategory.TEST)
    if is_documentation:
        categories.append(ReviewCostPathCategory.DOCUMENTATION)
    if not is_test and not is_documentation:
        categories.append(ReviewCostPathCategory.PRODUCTION)
    return tuple(categories)


def _band(value: int, bands: tuple[tuple[int, int], ...]) -> int:
    if value == 0:
        return 0
    for maximum, score in bands:
        if value <= maximum:
            return score
    return 100


def _contribution(
    signal: ReviewCostSignal,
    score: int,
    weight: int,
    explanation: str,
    evidence: list[str],
    limit: int,
) -> ReviewCostSignalContribution:
    return ReviewCostSignalContribution(
        signal=signal,
        signal_score=score,
        weight_percent=weight,
        weighted_basis_points=score * weight,
        explanation=explanation,
        evidence=evidence[:limit],
    )


def _change_size(
    assessment_input: ReviewCostAssessmentInput,
    limit: int,
) -> ReviewCostSignalContribution:
    changes = assessment_input.total_changes
    preferred = assessment_input.preferred_diff_size
    if changes == 0:
        line_score = 0
    elif changes * 4 <= preferred:
        line_score = 10
    elif changes * 2 <= preferred:
        line_score = 25
    elif changes <= preferred:
        line_score = 50
    elif changes <= preferred * 2:
        line_score = 75
    else:
        line_score = 100
    file_score = _band(
        assessment_input.changed_file_count,
        ((3, 10), (10, 30), (25, 60), (50, 80)),
    )
    commit_score = _band(
        assessment_input.commit_count,
        ((2, 10), (5, 30), (10, 60), (20, 80)),
    )
    return _contribution(
        ReviewCostSignal.CHANGE_SIZE,
        max(line_score, file_score, commit_score),
        REVIEW_COST_CHANGE_SIZE_WEIGHT,
        "Change size combines changed lines, file count, and commit count.",
        [
            f"{changes} changed lines.",
            f"{assessment_input.changed_file_count} changed files.",
            f"{assessment_input.commit_count} commits.",
            f"Preferred diff size is {preferred} lines.",
        ],
        limit,
    )


def _change_dispersion(
    assessment_input: ReviewCostAssessmentInput,
    limit: int,
) -> ReviewCostSignalContribution:
    component_score = _band(
        assessment_input.distinct_component_count,
        ((1, 10), (2, 30), (4, 60), (8, 80)),
    )
    directory_score = _band(
        assessment_input.distinct_directory_count,
        ((3, 10), (6, 30), (12, 60), (25, 80)),
    )
    return _contribution(
        ReviewCostSignal.CHANGE_DISPERSION,
        max(component_score, directory_score),
        REVIEW_COST_CHANGE_DISPERSION_WEIGHT,
        "Change dispersion reflects how widely the pull request spans repository areas.",
        [
            f"{assessment_input.distinct_component_count} top-level components.",
            f"{assessment_input.distinct_directory_count} directories.",
        ],
        limit,
    )


def _risk_sensitive_paths(
    classifications: dict[str, tuple[ReviewCostPathCategory, ...]],
    limit: int,
) -> ReviewCostSignalContribution:
    counts = {
        category: sum(category in categories for categories in classifications.values())
        for category in _RISK_CATEGORY_ORDER
    }
    present = [category for category in _RISK_CATEGORY_ORDER if counts[category]]
    score = (
        min(100, max(_RISK_BASE_SCORES[category] for category in present) + 5 * (len(present) - 1))
        if present
        else 0
    )
    evidence = [
        f"{counts[category]} changed paths classified as {category.value}."
        for category in present
    ]
    for path in sorted(classifications):
        for category in present:
            if category in classifications[path]:
                evidence.append(f"{category.value}: {path}")
    return _contribution(
        ReviewCostSignal.RISK_SENSITIVE_PATHS,
        score,
        REVIEW_COST_RISK_SENSITIVE_PATHS_WEIGHT,
        "Risk-sensitive paths increase the need for specialized or cautious review.",
        evidence,
        limit,
    )


def _validation_gaps(
    assessment_input: ReviewCostAssessmentInput,
    classifications: dict[str, tuple[ReviewCostPathCategory, ...]],
    limit: int,
) -> tuple[ReviewCostSignalContribution, list[str]]:
    production_count = sum(
        ReviewCostPathCategory.PRODUCTION in categories
        for categories in classifications.values()
    )
    test_count = sum(
        ReviewCostPathCategory.TEST in categories
        for categories in classifications.values()
    )
    gaps: list[tuple[int, str]] = []
    if assessment_input.merge_conflict is True:
        gaps.append((100, "A merge conflict is present."))
    if assessment_input.required_checks_failed:
        gaps.append(
            (
                90,
                f"{assessment_input.required_checks_failed} required checks failed.",
            )
        )
    if assessment_input.draft:
        gaps.append((70, "The pull request is a draft."))
    if assessment_input.required_checks_pending:
        gaps.append(
            (
                60,
                f"{assessment_input.required_checks_pending} required checks are pending.",
            )
        )
    if assessment_input.changes_requested_count:
        gaps.append(
            (
                60,
                f"{assessment_input.changes_requested_count} active changes-requested reviews.",
            )
        )
    if assessment_input.required_checks_total == 0 and production_count:
        gaps.append((50, "Production changes have no configured required checks."))
    if production_count and test_count == 0:
        gaps.append((55, "Production changes have no accompanying test files."))

    score = min(100, max((base for base, _ in gaps), default=0) + max(0, len(gaps) - 1) * 5)
    reducers: list[str] = []
    reduction_evidence: list[str] = []
    if (
        assessment_input.required_checks_total > 0
        and assessment_input.required_checks_passed
        == assessment_input.required_checks_total
    ):
        reducers.append("All required checks passed.")
        reduction_evidence.append("Reduced 20 points because all required checks passed.")
        score -= 20
    if production_count and test_count:
        reducers.append("Test changes accompany production changes.")
        reduction_evidence.append(
            "Reduced 15 points because test changes accompany production changes."
        )
        score -= 15
    if (
        assessment_input.approval_count > 0
        and assessment_input.changes_requested_count == 0
    ):
        reducers.append(
            "At least one approval is present without an active changes-requested review."
        )
        reduction_evidence.append(
            "Reduced 10 points because an approval is present without changes requested."
        )
        score -= 10
    contribution = _contribution(
        ReviewCostSignal.VALIDATION_GAPS,
        max(0, min(100, score)),
        REVIEW_COST_VALIDATION_GAPS_WEIGHT,
        "Validation gaps estimate additional maintainer work before confident review.",
        [text for _, text in gaps] + reduction_evidence,
        limit,
    )
    return contribution, reducers


def _historical_complexity(
    assessment_input: ReviewCostAssessmentInput,
    limit: int,
) -> ReviewCostSignalContribution:
    context = assessment_input.historical_context
    if context is None:
        score = 0
        evidence: list[str] = []
    else:
        factors: list[int] = []
        if context.rejected_or_superseded_count:
            factors.append(80)
        if context.high_significance_count:
            factors.append(70)
        if context.unresolved_count:
            factors.append(50)
        if context.related_match_count >= 10:
            factors.append(50)
        elif context.related_match_count >= 5:
            factors.append(35)
        elif context.related_match_count >= 1:
            factors.append(15)
        score = (
            min(100, max(factors) + 5 * (len(factors) - 1))
            if factors
            else 0
        )
        evidence = [
            f"{context.related_match_count} related historical matches.",
            (
                f"{context.rejected_or_superseded_count} rejected or "
                "superseded matches."
            ),
            f"{context.high_significance_count} high-significance matches.",
            f"{context.unresolved_count} unresolved matches.",
        ]
    return _contribution(
        ReviewCostSignal.HISTORICAL_COMPLEXITY,
        score,
        REVIEW_COST_HISTORICAL_COMPLEXITY_WEIGHT,
        "Historical complexity reflects related rejected, important, or unresolved work.",
        evidence,
        limit,
    )


def _warnings(assessment_input: ReviewCostAssessmentInput) -> list[str]:
    warnings: list[str] = []
    if not assessment_input.policy_present:
        warnings.append(_MISSING_POLICY_WARNING)
    if assessment_input.merge_conflict is None:
        warnings.append(_UNKNOWN_MERGE_WARNING)
    context = assessment_input.historical_context
    if context is None:
        warnings.append(_NO_HISTORY_WARNING)
    else:
        if not context.source_history_complete:
            warnings.append(_INCOMPLETE_HISTORY_WARNING)
        if not context.ranking_coverage_complete:
            warnings.append(_INCOMPLETE_RANKING_WARNING)
        if context.result_truncated:
            warnings.append(_TRUNCATED_RELATED_WARNING)
    return list(dict.fromkeys(warnings))


class ReviewCostAssessmentService:
    """Calculate deterministic evidence-derived review cost."""

    def assess(
        self,
        assessment_input: ReviewCostAssessmentInput,
        *,
        assessed_at: datetime,
        options: ReviewCostAssessmentOptions | None = None,
    ) -> ReviewCostAssessment:
        """Analyze every source file and return five explainable signals."""

        if assessed_at.tzinfo is None or assessed_at.utcoffset() is None:
            raise ValueError("assessed_at must be timezone-aware.")
        normalized_time = assessed_at.astimezone(UTC)
        selected_options = options or ReviewCostAssessmentOptions()
        protected = set(assessment_input.protected_changed_paths)
        changed_paths = dict.fromkeys(
            path
            for file in assessment_input.files
            for path in (file.path, file.previous_path)
            if path is not None
        )
        classifications = {
            path: _classify_path(path, protected)
            for path in changed_paths
        }
        limit = selected_options.max_evidence_items_per_signal
        validation, reducers = _validation_gaps(
            assessment_input,
            classifications,
            limit,
        )
        contributions = [
            _change_size(assessment_input, limit),
            _change_dispersion(assessment_input, limit),
            _risk_sensitive_paths(classifications, limit),
            validation,
            _historical_complexity(assessment_input, limit),
        ]
        return ReviewCostAssessment(
            repository=assessment_input.repository,
            pull_request=assessment_input.pull_request,
            assessed_at=normalized_time,
            contributions=contributions,
            reducers=reducers,
            warnings=_warnings(assessment_input),
        )
