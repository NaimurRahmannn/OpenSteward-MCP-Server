"""Convert GitHub pull-request snapshots into policy input."""

import re
from collections.abc import Iterable
from enum import StrEnum

from pydantic import (
    Field,
    TypeAdapter,
    field_validator,
)

from opensteward.github.models import StrictGitHubModel
from opensteward.github.pull_requests import (
    GitHubChecksState,
    GitHubPullRequestSnapshot,
)
from opensteward.policy import (
    ContributionCategory,
    ContributionPolicyInput,
    matches_repository_pattern,
    normalize_repository_pattern,
)


DEFAULT_TEST_FILE_PATTERNS: tuple[str, ...] = (
    "**/tests/**",
    "**/test/**",
    "**/__tests__/**",
    "**/test_*.*",
    "**/*_test.*",
    "**/*.test.*",
    "**/*.spec.*",
)


_CLOSING_REFERENCE_PATTERN = re.compile(
    r"\b"
    r"(?P<keyword>"
    r"close[sd]?"
    r"|fix(?:e[sd])?"
    r"|resolve[sd]?"
    r")"
    r"\s*:?\s+"
    r"(?P<reference>"
    r"(?:(?P<owner>[A-Za-z0-9_.-]+)/"
    r"(?P<repository>[A-Za-z0-9_.-]+))?"
    r"#(?P<number>[1-9]\d*)"
    r")",
    re.IGNORECASE,
)


_FENCED_CODE_PATTERN = re.compile(
    r"(?s)(```|~~~).*?\1"
)

_INLINE_CODE_PATTERN = re.compile(
    r"`[^`\n]*`"
)


class GitHubContributionConversionError(ValueError):
    """Raised when a PR snapshot cannot safely become policy input."""


class GitHubApprovalCountSource(StrEnum):
    """Approval count used for policy evaluation."""

    HEAD_COMMIT = "head_commit"
    EFFECTIVE = "effective"


class GitHubCategoryEvidenceSource(StrEnum):
    """Origin of one contribution-category classification."""

    EXPLICIT = "explicit"
    PATH_PATTERN = "path_pattern"


class GitHubIssueLinkageScope(StrEnum):
    """Issue-link evidence available from the current snapshot."""

    BODY_CLOSING_KEYWORDS_ONLY = (
        "pull_request_body_closing_keywords_only"
    )


class GitHubContributionCategoryPathRule(StrictGitHubModel):
    """Repository-specific path rules for one contribution category."""

    category: ContributionCategory

    patterns: list[str] = Field(
        min_length=1,
        max_length=100,
    )

    @field_validator("patterns")
    @classmethod
    def normalize_patterns(
        cls,
        patterns: list[str],
    ) -> list[str]:
        """Normalize patterns and reject duplicates."""

        normalized = [
            normalize_repository_pattern(pattern)
            for pattern in patterns
        ]

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "Category path patterns must be unique."
            )

        return normalized


class GitHubContributionInputOptions(StrictGitHubModel):
    """Configurable evidence extraction behavior."""

    test_file_patterns: list[str] = Field(
        default_factory=lambda: list(
            DEFAULT_TEST_FILE_PATTERNS
        ),
        max_length=100,
    )

    category_path_rules: list[
        GitHubContributionCategoryPathRule
    ] = Field(
        default_factory=list,
        max_length=100,
    )

    approval_source: GitHubApprovalCountSource = (
        GitHubApprovalCountSource.HEAD_COMMIT
    )

    require_complete_file_list: bool = True

    @field_validator("test_file_patterns")
    @classmethod
    def normalize_test_patterns(
        cls,
        patterns: list[str],
    ) -> list[str]:
        """Normalize test patterns and reject duplicates."""

        normalized = [
            normalize_repository_pattern(pattern)
            for pattern in patterns
        ]

        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "Test file patterns must be unique."
            )

        return normalized


class GitHubContributionCategoryEvidence(StrictGitHubModel):
    """Evidence supporting one contribution category."""

    category: ContributionCategory
    source: GitHubCategoryEvidenceSource

    path: str | None = None
    pattern: str | None = None


class GitHubIssueReferenceEvidence(StrictGitHubModel):
    """One same-repository issue reference found in the PR body."""

    issue_number: int = Field(gt=0)

    keyword: str = Field(min_length=1)
    reference: str = Field(min_length=1)


class GitHubContributionInputResult(StrictGitHubModel):
    """Policy input together with its extraction evidence."""

    contribution: ContributionPolicyInput

    affected_paths: list[str] = Field(
        default_factory=list,
    )

    test_file_matches: list[str] = Field(
        default_factory=list,
    )

    category_evidence: list[
        GitHubContributionCategoryEvidence
    ] = Field(
        default_factory=list,
    )

    linked_issue_evidence: list[
        GitHubIssueReferenceEvidence
    ] = Field(
        default_factory=list,
    )

    approval_source: GitHubApprovalCountSource

    issue_linkage_scope: GitHubIssueLinkageScope = (
        GitHubIssueLinkageScope
        .BODY_CLOSING_KEYWORDS_ONLY
    )

    checks_state: GitHubChecksState

    files_complete: bool

    warnings: list[str] = Field(
        default_factory=list,
    )


def _collect_affected_paths(
    snapshot: GitHubPullRequestSnapshot,
) -> list[str]:
    """Return current and previous paths, preserving order.

    Renamed files contribute both their old and new paths so a file
    cannot escape protected-path policy merely by being renamed.
    """

    paths: list[str] = []
    seen: set[str] = set()

    for changed_file in snapshot.files:
        candidates = (
            changed_file.filename,
            changed_file.previous_filename,
        )

        for candidate in candidates:
            if candidate is None or candidate in seen:
                continue

            seen.add(candidate)
            paths.append(candidate)

    return paths


def _strip_markdown_code(
    content: str,
) -> str:
    """Remove common fenced and inline Markdown code regions."""

    without_fenced_code = (
        _FENCED_CODE_PATTERN.sub(
            "",
            content,
        )
    )

    return _INLINE_CODE_PATTERN.sub(
        "",
        without_fenced_code,
    )


def _extract_issue_references(
    snapshot: GitHubPullRequestSnapshot,
) -> list[GitHubIssueReferenceEvidence]:
    """Extract same-repository closing-keyword references."""

    body = snapshot.pull_request.body

    if not body:
        return []

    searchable_body = _strip_markdown_code(
        body
    )

    repository_full_name = (
        snapshot.repository.full_name.casefold()
    )

    evidence: list[
        GitHubIssueReferenceEvidence
    ] = []

    seen_issue_numbers: set[int] = set()

    for match in _CLOSING_REFERENCE_PATTERN.finditer(
        searchable_body
    ):
        owner = match.group("owner")
        repository_name = match.group(
            "repository"
        )

        if owner is not None and repository_name is not None:
            referenced_repository = (
                f"{owner}/{repository_name}"
                .casefold()
            )

            if referenced_repository != repository_full_name:
                continue

        issue_number = int(
            match.group("number")
        )

        if issue_number in seen_issue_numbers:
            continue

        seen_issue_numbers.add(
            issue_number
        )

        evidence.append(
            GitHubIssueReferenceEvidence(
                issue_number=issue_number,
                keyword=(
                    match.group("keyword")
                    .casefold()
                ),
                reference=match.group(
                    "reference"
                ),
            )
        )

    return evidence


def _normalize_explicit_categories(
    categories: Iterable[
        ContributionCategory | str
    ],
) -> list[ContributionCategory]:
    """Validate and deduplicate explicit category hints."""

    normalized = TypeAdapter(
        list[ContributionCategory]
    ).validate_python(
        list(categories)
    )

    if len(normalized) != len(set(normalized)):
        raise ValueError(
            "Explicit contribution categories must be unique."
        )

    return normalized


def _find_test_file_matches(
    affected_paths: list[str],
    *,
    patterns: list[str],
) -> list[str]:
    """Return paths matching configured test-file patterns."""

    matches: list[str] = []

    for path in affected_paths:
        if any(
            matches_repository_pattern(
                path,
                pattern,
            )
            for pattern in patterns
        ):
            matches.append(path)

    return matches


def _classify_categories(
    affected_paths: list[str],
    *,
    explicit_categories: list[
        ContributionCategory
    ],
    path_rules: list[
        GitHubContributionCategoryPathRule
    ],
) -> tuple[
    list[ContributionCategory],
    list[GitHubContributionCategoryEvidence],
]:
    """Classify categories using explicit and path-based evidence."""

    categories: list[
        ContributionCategory
    ] = []

    category_set: set[
        ContributionCategory
    ] = set()

    evidence: list[
        GitHubContributionCategoryEvidence
    ] = []

    for category in explicit_categories:
        if category not in category_set:
            category_set.add(category)
            categories.append(category)

        evidence.append(
            GitHubContributionCategoryEvidence(
                category=category,
                source=(
                    GitHubCategoryEvidenceSource
                    .EXPLICIT
                ),
            )
        )

    for rule in path_rules:
        for path in affected_paths:
            for pattern in rule.patterns:
                if not matches_repository_pattern(
                    path,
                    pattern,
                ):
                    continue

                if rule.category not in category_set:
                    category_set.add(
                        rule.category
                    )
                    categories.append(
                        rule.category
                    )

                evidence.append(
                    GitHubContributionCategoryEvidence(
                        category=rule.category,
                        source=(
                            GitHubCategoryEvidenceSource
                            .PATH_PATTERN
                        ),
                        path=path,
                        pattern=pattern,
                    )
                )

    return categories, evidence


def _select_approval_count(
    snapshot: GitHubPullRequestSnapshot,
    *,
    source: GitHubApprovalCountSource,
) -> int:
    """Select the approval count used for policy evaluation."""

    if source == GitHubApprovalCountSource.HEAD_COMMIT:
        return (
            snapshot
            .head_commit_human_approval_count
        )

    return snapshot.human_approval_count


def _build_warnings(
    snapshot: GitHubPullRequestSnapshot,
    *,
    files_complete: bool,
) -> list[str]:
    """Build evidence-quality warnings."""

    warnings: list[str] = []

    if not files_complete:
        warnings.append(
            "GitHub did not return the complete pull-request "
            "file list; path-based evidence may be incomplete."
        )

    base_repository = (
        snapshot
        .pull_request
        .base
        .repository_full_name
    )

    head_repository = (
        snapshot
        .pull_request
        .head
        .repository_full_name
    )

    if (
        base_repository is not None
        and head_repository is not None
        and (
            base_repository.casefold()
            != head_repository.casefold()
        )
    ):
        warnings.append(
            "The pull request originates from a fork; "
            "check-run visibility from the base repository "
            "may be incomplete."
        )

    return warnings


def build_contribution_policy_input_from_snapshot(
    snapshot: GitHubPullRequestSnapshot,
    *,
    explicit_categories: Iterable[
        ContributionCategory | str
    ] = (),
    options: GitHubContributionInputOptions | None = None,
) -> GitHubContributionInputResult:
    """Convert a GitHub PR snapshot into policy input.

    Semantic categories are included only when explicitly supplied or
    matched by configured repository path rules.
    """

    effective_options = (
        options
        or GitHubContributionInputOptions()
    )

    files_complete = (
        not snapshot.files_truncated
    )

    if (
        not files_complete
        and (
            effective_options
            .require_complete_file_list
        )
    ):
        raise GitHubContributionConversionError(
            "The pull-request file list is incomplete, so "
            "OpenSteward cannot safely evaluate path-based policy."
        )

    affected_paths = _collect_affected_paths(
        snapshot
    )

    if not affected_paths:
        raise GitHubContributionConversionError(
            "The pull-request snapshot contains no changed paths."
        )

    normalized_explicit_categories = (
        _normalize_explicit_categories(
            explicit_categories
        )
    )

    categories, category_evidence = (
        _classify_categories(
            affected_paths,
            explicit_categories=(
                normalized_explicit_categories
            ),
            path_rules=(
                effective_options
                .category_path_rules
            ),
        )
    )

    issue_evidence = (
        _extract_issue_references(
            snapshot
        )
    )

    test_file_matches = (
        _find_test_file_matches(
            affected_paths,
            patterns=(
                effective_options
                .test_file_patterns
            ),
        )
    )

    current_approvals = (
        _select_approval_count(
            snapshot,
            source=(
                effective_options
                .approval_source
            ),
        )
    )

    contribution = ContributionPolicyInput(
        changed_files=affected_paths,
        additions=(
            snapshot.pull_request.additions
        ),
        deletions=(
            snapshot.pull_request.deletions
        ),
        categories=categories,
        linked_issue_numbers=[
            item.issue_number
            for item in issue_evidence
        ],
        tests_changed=bool(
            test_file_matches
        ),
        current_approvals=current_approvals,
    )

    return GitHubContributionInputResult(
        contribution=contribution,
        affected_paths=affected_paths,
        test_file_matches=test_file_matches,
        category_evidence=category_evidence,
        linked_issue_evidence=issue_evidence,
        approval_source=(
            effective_options.approval_source
        ),
        checks_state=(
            snapshot.checks.state
        ),
        files_complete=files_complete,
        warnings=_build_warnings(
            snapshot,
            files_complete=files_complete,
        ),
    )