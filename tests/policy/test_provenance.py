"""Tests for repository policy source metadata."""

from pathlib import Path

from opensteward.policy import (
    LoadedRepositoryPolicy,
    PolicySource,
    RepositoryPolicy,
    create_default_policy_result,
    load_repository_policy,
    load_repository_policy_with_metadata,
    parse_repository_policy,
    parse_repository_policy_with_metadata,
)


def test_default_policy_result_contains_metadata() -> None:
    loaded = create_default_policy_result()

    assert isinstance(loaded, LoadedRepositoryPolicy)
    assert loaded.source == PolicySource.DEFAULT
    assert loaded.used_defaults is True
    assert loaded.policy.version == 1


def test_missing_file_reports_default_source(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / ".opensteward.yml"

    loaded = load_repository_policy_with_metadata(policy_path)

    assert loaded.source == PolicySource.DEFAULT
    assert loaded.source_reference == str(policy_path)
    assert loaded.used_defaults is True


def test_empty_file_reports_default_source(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / ".opensteward.yml"
    policy_path.write_text("", encoding="utf-8")

    loaded = load_repository_policy_with_metadata(policy_path)

    assert loaded.source == PolicySource.DEFAULT
    assert loaded.source_reference == str(policy_path)
    assert loaded.used_defaults is True


def test_valid_file_reports_repository_file_source(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / ".opensteward.yml"

    policy_path.write_text(
        """
        version: 1

        pull_requests:
          preferred_maximum_diff_lines: 750
        """,
        encoding="utf-8",
    )

    loaded = load_repository_policy_with_metadata(policy_path)

    assert loaded.source == PolicySource.REPOSITORY_FILE
    assert loaded.source_reference == str(policy_path)
    assert loaded.used_defaults is False
    assert loaded.policy.pull_requests.preferred_maximum_diff_lines == 750


def test_valid_memory_content_reports_memory_source() -> None:
    loaded = parse_repository_policy_with_metadata(
        """
        version: 1

        pull_requests:
          preferred_maximum_diff_lines: 900
        """,
        source="github:acme/example/.opensteward.yml",
    )

    assert loaded.source == PolicySource.MEMORY
    assert (
        loaded.source_reference
        == "github:acme/example/.opensteward.yml"
    )
    assert loaded.used_defaults is False
    assert loaded.policy.pull_requests.preferred_maximum_diff_lines == 900


def test_metadata_serializes_to_json_values() -> None:
    loaded = parse_repository_policy_with_metadata(
        "version: 1",
        source="test-policy",
    )

    data = loaded.model_dump(mode="json")

    assert data["source"] == "memory"
    assert data["source_reference"] == "test-policy"
    assert data["used_defaults"] is False
    assert data["policy"]["version"] == 1


def test_existing_parse_function_remains_compatible() -> None:
    policy = parse_repository_policy(
        """
        version: 1

        pull_requests:
          preferred_maximum_diff_lines: 650
        """
    )

    assert isinstance(policy, RepositoryPolicy)
    assert policy.pull_requests.preferred_maximum_diff_lines == 650


def test_existing_load_function_remains_compatible(
    tmp_path: Path,
) -> None:
    policy_path = tmp_path / ".opensteward.yml"
    policy_path.write_text("version: 1", encoding="utf-8")

    policy = load_repository_policy(policy_path)

    assert isinstance(policy, RepositoryPolicy)
    assert policy.version == 1