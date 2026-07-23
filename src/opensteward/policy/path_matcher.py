"""Protected repository-path matching for OpenSteward."""

from collections.abc import Iterable, Sequence
from fnmatch import fnmatchcase
from functools import lru_cache

from opensteward.policy.models import (
    ProtectedPathMatch,
    ProtectedPathRule,
)


def _normalize_repository_value(
    value: str,
    *,
    description: str,
) -> str:
    """Normalize and validate a repository-relative path or pattern."""

    normalized = value.replace("\\", "/").strip()

    while normalized.startswith("./"):
        normalized = normalized[2:]

    if not normalized:
        raise ValueError(f"{description} must not be empty.")

    if normalized.startswith("/"):
        raise ValueError(
            f"{description} must be repository-relative."
        )

    parts = normalized.split("/")

    if any(part == "" for part in parts):
        raise ValueError(
            f"{description} must not contain empty path segments."
        )

    if any(part in {".", ".."} for part in parts):
        raise ValueError(
            f"{description} must not contain '.' or '..' segments."
        )

    return normalized


def normalize_repository_path(path: str) -> str:
    """Normalize a changed-file path to repository POSIX format."""

    return _normalize_repository_value(
        path,
        description="Repository path",
    )


def normalize_repository_pattern(pattern: str) -> str:
    """Normalize a protected-path glob pattern."""

    return _normalize_repository_value(
        pattern,
        description="Repository pattern",
    )


def matches_repository_pattern(
    path: str,
    pattern: str,
) -> bool:
    """Return whether a repository path matches a glob pattern.

    Supported behavior:

    - ``*`` matches characters inside one path segment.
    - ``?`` matches one character inside one path segment.
    - ``[abc]`` uses normal shell-style character matching.
    - ``**`` matches zero or more complete path segments.
    """

    normalized_path = normalize_repository_path(path)
    normalized_pattern = normalize_repository_pattern(pattern)

    path_parts = tuple(normalized_path.split("/"))
    pattern_parts = tuple(normalized_pattern.split("/"))

    @lru_cache(maxsize=None)
    def match_parts(
        path_index: int,
        pattern_index: int,
    ) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)

        pattern_part = pattern_parts[pattern_index]

        if pattern_part == "**":
            # ``**`` may match zero path segments.
            if match_parts(path_index, pattern_index + 1):
                return True

            # Or it may consume the current segment and continue.
            if path_index < len(path_parts):
                return match_parts(
                    path_index + 1,
                    pattern_index,
                )

            return False

        if path_index == len(path_parts):
            return False

        if not fnmatchcase(
            path_parts[path_index],
            pattern_part,
        ):
            return False

        return match_parts(
            path_index + 1,
            pattern_index + 1,
        )

    return match_parts(0, 0)


def find_protected_path_matches(
    path: str,
    rules: Sequence[ProtectedPathRule],
) -> list[ProtectedPathMatch]:
    """Return all protected-path rules matching one repository path."""

    normalized_path = normalize_repository_path(path)
    matches: list[ProtectedPathMatch] = []

    for rule in rules:
        if not matches_repository_pattern(
            normalized_path,
            rule.pattern,
        ):
            continue

        review_message = (
            "Human review is required."
            if rule.human_review_required
            else "Human review is not required by this rule."
        )

        matches.append(
            ProtectedPathMatch(
                path=normalized_path,
                pattern=rule.pattern,
                risk=rule.risk,
                human_review_required=rule.human_review_required,
                explanation=(
                    f"{normalized_path} matches protected path pattern "
                    f"{rule.pattern} with {rule.risk.value} risk. "
                    f"{review_message}"
                ),
            )
        )

    return matches


def match_protected_paths(
    paths: Iterable[str],
    rules: Sequence[ProtectedPathRule],
) -> list[ProtectedPathMatch]:
    """Return protected-path matches for multiple changed files."""

    matches: list[ProtectedPathMatch] = []

    for path in paths:
        matches.extend(
            find_protected_path_matches(
                path=path,
                rules=rules,
            )
        )

    return matches