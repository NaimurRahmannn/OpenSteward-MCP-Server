"""Tests for typed GitHub identity models."""

import pytest
from pydantic import ValidationError

from opensteward.github import (
    GitHubAccountType,
    GitHubInstallationRef,
    GitHubRepositoryRef,
)


def test_repository_ref_builds_full_name() -> None:
    repository = GitHubRepositoryRef(
        owner="acme",
        name="framework",
    )

    assert repository.owner == "acme"
    assert repository.name == "framework"
    assert repository.full_name == "acme/framework"


def test_repository_ref_strips_whitespace() -> None:
    repository = GitHubRepositoryRef(
        owner="  acme  ",
        name="  framework  ",
    )

    assert repository.full_name == "acme/framework"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("owner", "acme/example"),
        ("owner", ".."),
        ("name", "src\\example"),
        ("name", "."),
    ],
)
def test_repository_ref_rejects_invalid_segments(
    field_name: str,
    value: str,
) -> None:
    values = {
        "owner": "acme",
        "name": "framework",
    }
    values[field_name] = value

    with pytest.raises(ValidationError):
        GitHubRepositoryRef.model_validate(values)


def test_repository_ref_serializes_computed_name() -> None:
    repository = GitHubRepositoryRef(
        owner="acme",
        name="framework",
    )

    data = repository.model_dump(mode="json")

    assert data == {
        "owner": "acme",
        "name": "framework",
        "full_name": "acme/framework",
    }


def test_installation_ref_accepts_organization() -> None:
    installation = GitHubInstallationRef(
        installation_id=123456,
        account_login="acme",
        account_type=GitHubAccountType.ORGANIZATION,
    )

    assert installation.installation_id == 123456
    assert installation.account_login == "acme"
    assert (
        installation.account_type
        == GitHubAccountType.ORGANIZATION
    )


def test_installation_ref_rejects_invalid_id() -> None:
    with pytest.raises(ValidationError):
        GitHubInstallationRef(
            installation_id=0,
            account_login="acme",
            account_type=GitHubAccountType.ORGANIZATION,
        )


def test_installation_ref_rejects_slash_in_login() -> None:
    with pytest.raises(ValidationError):
        GitHubInstallationRef(
            installation_id=123456,
            account_login="acme/example",
            account_type=GitHubAccountType.ORGANIZATION,
        )