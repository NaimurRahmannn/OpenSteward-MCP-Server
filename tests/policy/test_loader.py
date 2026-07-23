"""Tests for loading OpenSteward repository policy files."""

from pathlib import Path

import pytest

from opensteward.policy import (
    ContributionCategory,
    PolicyLoadError,
    RiskLevel,
    load_repository_policy,
    parse_repository_policy,
)


def test_parse_empty_policy_returns_safe_defaults() -> None:
    policy = parse_repository_policy("")

    assert policy.version == 1
    assert policy.pull_requests.preferred_maximum_diff_lines == 500
    assert policy.automation.require_human_approval is True


def test_parse_whitespace_only_policy_returns_defaults() -> None:
    policy = parse_repository_policy(" \n\t ")

    assert policy.version == 1
    assert policy.protected_paths == []


def test_parse_valid_policy() -> None:
    policy = parse_repository_policy(
        """
        version: 1
        pull_requests:
          linked_issue_required_for:
            - public_api
            - architecture
          preferred_maximum_diff_lines: 750

        protected_paths:
          - pattern: src/security/**
            risk: critical
            human_review_required: true
        """,
        source="test-policy",
    )

    assert policy.version == 1
   
    assert policy.pull_requests.preferred_maximum_diff_lines == 750

    assert policy.pull_requests.linked_issue_required_for == [
        ContributionCategory.PUBLIC_API,
        ContributionCategory.ARCHITECTURE,
    ]

    assert policy.protected_paths[0].pattern == "src/security/**"
    assert policy.protected_paths[0].risk == RiskLevel.CRITICAL


def test_load_missing_policy_file_returns_defaults(
    tmp_path: Path,
) -> None:
    missing_path = tmp_path / ".opensteward.yml"

    policy = load_repository_policy(missing_path)

    assert policy.version == 1
    assert policy.pull_requests.preferred_maximum_diff_lines == 500


def test_load_policy_from_file(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / ".opensteward.yml"

    policy_path.write_text(
        """
        version: 1

        pull_requests:
          preferred_maximum_diff_lines: 900
        """,
        encoding="utf-8",
    )

    policy = load_repository_policy(policy_path)

    assert policy.pull_requests.preferred_maximum_diff_lines == 900


def test_parse_rejects_invalid_yaml() -> None:
    with pytest.raises(
        PolicyLoadError,
        match="invalid YAML syntax",
    ):
        parse_repository_policy(
            """
            version: 1
            protected_paths:
              - pattern: [
            """,
            source=".opensteward.yml",
        )


@pytest.mark.parametrize(
    "content",
    [
        "- version\n- pull_requests\n",
        "OpenSteward policy",
        "42",
    ],
)
def test_parse_requires_mapping_root(
    content: str,
) -> None:
    with pytest.raises(
        PolicyLoadError,
        match="must be a YAML mapping",
    ):
        parse_repository_policy(content)


def test_parse_wraps_policy_validation_errors() -> None:
    with pytest.raises(
        PolicyLoadError,
        match="policy validation failed",
    ) as error:
        parse_repository_policy(
            """
            version: 2
            """,
            source="repository/.opensteward.yml",
        )

    assert "repository/.opensteward.yml" in str(error.value)


def test_parse_rejects_unknown_policy_fields() -> None:
    with pytest.raises(
        PolicyLoadError,
        match="policy validation failed",
    ):
        parse_repository_policy(
            """
            version: 1
            unknown_setting: true
            """
        )


def test_parse_rejects_unsafe_python_tags() -> None:
    with pytest.raises(
        PolicyLoadError,
        match="invalid YAML syntax",
    ):
        parse_repository_policy(
            """
            !!python/object/apply:os.system
            - echo unsafe
            """
        )


def test_load_wraps_file_read_errors(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / ".opensteward.yml"
    policy_path.mkdir()

    with pytest.raises(
        PolicyLoadError,
        match="unable to read",
    ):
        load_repository_policy(policy_path)