"""Tests for protected repository-path matching."""

import pytest

from opensteward.policy import (
    ProtectedPathRule,
    RiskLevel,
    find_protected_path_matches,
    match_protected_paths,
    matches_repository_pattern,
    normalize_repository_path,
)


@pytest.mark.parametrize(
    ("path", "pattern"),
    [
        ("src/security/auth.py", "src/security/auth.py"),
        ("src/security/auth.py", "src/security/*.py"),
        ("src/parser/config.py", "src/*/config.py"),
        ("tests/test_api.py", "tests/test_???.py"),
        ("src/api/client.py", "src/[a-z]pi/client.py"),
    ],
)
def test_matches_repository_pattern(
    path: str,
    pattern: str,
) -> None:
    assert matches_repository_pattern(path, pattern) is True


def test_single_star_does_not_cross_directory_boundaries() -> None:
    assert (
        matches_repository_pattern(
            "src/api/internal/config.py",
            "src/*/config.py",
        )
        is False
    )


@pytest.mark.parametrize(
    "path",
    [
        "src/security/auth.py",
        "src/security/internal/auth.py",
        "src/security/internal/tokens/loader.py",
    ],
)
def test_double_star_matches_recursive_directories(
    path: str,
) -> None:
    assert (
        matches_repository_pattern(
            path,
            "src/security/**",
        )
        is True
    )


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "docs/setup.md",
        "docs/guides/install.md",
    ],
)
def test_double_star_can_match_zero_or_more_directories(
    path: str,
) -> None:
    assert matches_repository_pattern(path, "**/*.md") is True


def test_pattern_matching_is_case_sensitive() -> None:
    assert (
        matches_repository_pattern(
            "src/Security/auth.py",
            "src/security/**",
        )
        is False
    )


def test_repository_path_normalizes_backslashes() -> None:
    normalized = normalize_repository_path(
        ".\\src\\security\\auth.py"
    )

    assert normalized == "src/security/auth.py"


@pytest.mark.parametrize(
    "path",
    [
        "/src/security/auth.py",
        "../security/auth.py",
        "src/../security/auth.py",
        "src//security/auth.py",
        "",
    ],
)
def test_repository_path_rejects_unsafe_values(
    path: str,
) -> None:
    with pytest.raises(ValueError):
        normalize_repository_path(path)


def test_find_protected_path_matches_returns_evidence() -> None:
    rules = [
        ProtectedPathRule(
            pattern="src/security/**",
            risk=RiskLevel.CRITICAL,
            human_review_required=True,
        ),
        ProtectedPathRule(
            pattern="src/**",
            risk=RiskLevel.MEDIUM,
            human_review_required=False,
        ),
    ]

    matches = find_protected_path_matches(
        path="src/security/auth.py",
        rules=rules,
    )

    assert len(matches) == 2

    critical_match = matches[0]

    assert critical_match.path == "src/security/auth.py"
    assert critical_match.pattern == "src/security/**"
    assert critical_match.risk == RiskLevel.CRITICAL
    assert critical_match.human_review_required is True
    assert "Human review is required" in critical_match.explanation


def test_find_protected_path_matches_returns_empty_list() -> None:
    rules = [
        ProtectedPathRule(
            pattern="src/security/**",
            risk=RiskLevel.CRITICAL,
        )
    ]

    matches = find_protected_path_matches(
        path="docs/setup.md",
        rules=rules,
    )

    assert matches == []


def test_match_multiple_protected_paths_preserves_order() -> None:
    rules = [
        ProtectedPathRule(
            pattern="src/security/**",
            risk=RiskLevel.CRITICAL,
        ),
        ProtectedPathRule(
            pattern="docs/**",
            risk=RiskLevel.LOW,
            human_review_required=False,
        ),
    ]

    matches = match_protected_paths(
        paths=[
            "src/security/auth.py",
            "src/application.py",
            "docs/setup.md",
        ],
        rules=rules,
    )

    assert len(matches) == 2
    assert matches[0].path == "src/security/auth.py"
    assert matches[0].risk == RiskLevel.CRITICAL
    assert matches[1].path == "docs/setup.md"
    assert matches[1].risk == RiskLevel.LOW


def test_path_match_serializes_to_json_values() -> None:
    rules = [
        ProtectedPathRule(
            pattern="src/security/**",
            risk=RiskLevel.CRITICAL,
        )
    ]

    match = find_protected_path_matches(
        path="src/security/auth.py",
        rules=rules,
    )[0]

    data = match.model_dump(mode="json")

    assert data["path"] == "src/security/auth.py"
    assert data["pattern"] == "src/security/**"
    assert data["risk"] == "critical"
    assert data["human_review_required"] is True