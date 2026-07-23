"""Tests for GitHub repository metadata and policy retrieval."""

import base64
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import TypeAdapter

from opensteward.github import (
    GitHubRepositoryPolicyFileError,
    GitHubRepositoryRef,
    GitHubRepositoryService,
    GitHubRestResponse,
    GitHubRestResponseError,
)
from opensteward.policy import (
    PolicyLoadError,
    PolicySource,
)


def repository_payload() -> dict[str, object]:
    """Create a representative GitHub repository response."""

    return {
        "id": 1001,
        "name": "framework",
        "full_name": "acme/framework",
        "private": True,
        "fork": False,
        "archived": False,
        "disabled": False,
        "html_url": "https://github.com/acme/framework",
        "default_branch": "main",
        "owner": {
            "id": 2001,
            "login": "acme",
            "type": "Organization",
        },
        "extra_github_field": "ignored",
    }


def policy_content_payload(
    content: bytes,
    *,
    encoding: str = "base64",
    path: str = ".opensteward.yml",
    content_type: str = "file",
    reported_size: int | None = None,
) -> dict[str, object]:
    """Create a representative repository-content response."""

    return {
        "type": content_type,
        "encoding": encoding,
        "size": (
            len(content)
            if reported_size is None
            else reported_size
        ),
        "name": path.rsplit("/", maxsplit=1)[-1],
        "path": path,
        "sha": "abc123policysha",
        "content": (
            base64.b64encode(content)
            .decode("ascii")
        ),
        "html_url": (
            "https://github.com/acme/framework/"
            f"blob/main/{path}"
        ),
    }


class FakeGitHubRestClient:
    """Queued GitHub REST client for repository-service tests."""

    def __init__(
        self,
        *outcomes: object,
    ) -> None:
        self._outcomes = list(outcomes)

        self.calls: list[dict[str, object]] = []

    async def get_json(
        self,
        path: str,
        *,
        params: object = None,
        response_type: Any = Any,
        accept: str = "application/vnd.github+json",
    ) -> GitHubRestResponse[Any]:
        self.calls.append(
            {
                "path": path,
                "params": params,
                "response_type": response_type,
                "accept": accept,
            }
        )

        outcome = self._outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        data = outcome

        if response_type is not Any:
            data = TypeAdapter(
                response_type
            ).validate_python(outcome)

        return GitHubRestResponse(
            status_code=200,
            data=data,
        )


def create_repository() -> GitHubRepositoryRef:
    """Create the repository used throughout the tests."""

    return GitHubRepositoryRef(
        owner="acme",
        name="framework",
    )


@pytest.mark.anyio
async def test_get_repository_metadata() -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    repository = await service.get_repository(
        create_repository()
    )

    assert repository.id == 1001
    assert repository.full_name == "acme/framework"
    assert repository.default_branch == "main"
    assert repository.owner.login == "acme"

    assert client.calls[0]["path"] == (
        "/repos/acme/framework"
    )


@pytest.mark.anyio
async def test_load_valid_repository_policy() -> None:
    policy_yaml = b"""
    version: 1

    pull_requests:
      preferred_maximum_diff_lines: 750
    """

    client = FakeGitHubRestClient(
        repository_payload(),
        policy_content_payload(policy_yaml),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    result = await service.load_repository_policy(
        create_repository()
    )

    assert result.policy_file_present is True
    assert result.requested_path == ".opensteward.yml"
    assert result.requested_ref == "main"

    assert result.policy_file is not None
    assert result.policy_file.sha == "abc123policysha"
    assert result.policy_file.git_ref == "main"
    assert result.policy_file.size_bytes == len(policy_yaml)

    assert (
        result.loaded_policy.source
        == PolicySource.GITHUB_REPOSITORY
    )

    assert result.loaded_policy.used_defaults is False

    assert (
        result.loaded_policy
        .policy
        .pull_requests
        .preferred_maximum_diff_lines
        == 750
    )

    assert (
        result.loaded_policy.source_reference
        == (
            "github:acme/framework"
            "@main:.opensteward.yml"
        )
    )

    assert client.calls[1]["path"] == (
        "/repos/acme/framework/"
        "contents/.opensteward.yml"
    )

    assert client.calls[1]["params"] == {
        "ref": "main",
    }


@pytest.mark.anyio
async def test_load_policy_from_custom_ref() -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
        policy_content_payload(
            b"version: 1",
        ),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    result = await service.load_repository_policy(
        create_repository(),
        git_ref="feature/security-review",
    )

    assert (
        result.requested_ref
        == "feature/security-review"
    )

    assert client.calls[1]["params"] == {
        "ref": "feature/security-review",
    }

    assert (
        result.loaded_policy.source_reference
        == (
            "github:acme/framework"
            "@feature/security-review:"
            ".opensteward.yml"
        )
    )


@pytest.mark.anyio
async def test_missing_policy_file_uses_defaults() -> None:
    missing_file_error = GitHubRestResponseError(
        "GitHub REST request failed with status 404: Not Found",
        status_code=404,
    )

    client = FakeGitHubRestClient(
        repository_payload(),
        missing_file_error,
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    result = await service.load_repository_policy(
        create_repository()
    )

    assert result.policy_file_present is False
    assert result.policy_file is None

    assert result.loaded_policy.source == PolicySource.DEFAULT
    assert result.loaded_policy.used_defaults is True

    assert (
        result.loaded_policy.source_reference
        == (
            "github:acme/framework"
            "@main:.opensteward.yml"
        )
    )


@pytest.mark.anyio
async def test_empty_policy_file_uses_defaults_but_is_present() -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
        policy_content_payload(b""),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    result = await service.load_repository_policy(
        create_repository()
    )

    assert result.policy_file_present is True
    assert result.policy_file is not None

    assert result.loaded_policy.source == PolicySource.DEFAULT
    assert result.loaded_policy.used_defaults is True


@pytest.mark.anyio
async def test_repository_not_found_is_not_treated_as_missing_policy() -> None:
    repository_error = GitHubRestResponseError(
        "GitHub REST request failed with status 404: Not Found",
        status_code=404,
    )

    client = FakeGitHubRestClient(
        repository_error,
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(
        GitHubRestResponseError,
    ) as error_info:
        await service.load_repository_policy(
            create_repository()
        )

    assert error_info.value.status_code == 404
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_contents_permission_error_is_propagated() -> None:
    permission_error = GitHubRestResponseError(
        (
            "GitHub REST request failed with status 403: "
            "Resource not accessible by integration"
        ),
        status_code=403,
    )

    client = FakeGitHubRestClient(
        repository_payload(),
        permission_error,
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(
        GitHubRestResponseError,
        match="Resource not accessible",
    ) as error_info:
        await service.load_repository_policy(
            create_repository()
        )

    assert error_info.value.status_code == 403


@pytest.mark.anyio
async def test_malformed_repository_policy_is_rejected() -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
        policy_content_payload(
            b"""
            version: 1
            protected_paths:
              - pattern: [
            """
        ),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(
        PolicyLoadError,
        match="invalid YAML syntax",
    ) as error_info:
        await service.load_repository_policy(
            create_repository()
        )

    assert (
        "github:acme/framework"
        in error_info.value.source
    )


@pytest.mark.anyio
async def test_invalid_base64_is_rejected() -> None:
    payload = policy_content_payload(
        b"version: 1",
    )

    payload["content"] = "%%%not-base64%%%"

    client = FakeGitHubRestClient(
        repository_payload(),
        payload,
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(
        GitHubRepositoryPolicyFileError,
        match="invalid Base64",
    ):
        await service.load_repository_policy(
            create_repository()
        )


@pytest.mark.anyio
async def test_non_utf8_policy_is_rejected() -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
        policy_content_payload(
            b"\xff\xfe\x00\x01",
        ),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(
        GitHubRepositoryPolicyFileError,
        match="UTF-8",
    ):
        await service.load_repository_policy(
            create_repository()
        )


@pytest.mark.anyio
async def test_directory_policy_path_is_rejected() -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
        [
            {
                "type": "file",
                "name": "policy.yml",
            }
        ],
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(
        GitHubRepositoryPolicyFileError,
        match="directory or unsupported",
    ):
        await service.load_repository_policy(
            create_repository()
        )


@pytest.mark.anyio
async def test_unsupported_content_encoding_is_rejected() -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
        policy_content_payload(
            b"version: 1",
            encoding="none",
        ),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(
        GitHubRepositoryPolicyFileError,
        match="unsupported encoding",
    ):
        await service.load_repository_policy(
            create_repository()
        )


@pytest.mark.anyio
async def test_reported_file_size_must_match_content() -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
        policy_content_payload(
            b"version: 1",
            reported_size=999,
        ),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(
        GitHubRepositoryPolicyFileError,
        match="does not match",
    ):
        await service.load_repository_policy(
            create_repository()
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    "policy_path",
    [
        "../.opensteward.yml",
        "/.opensteward.yml",
        "config/../.opensteward.yml",
        "config//.opensteward.yml",
        "",
    ],
)
async def test_unsafe_policy_paths_are_rejected_before_request(
    policy_path: str,
) -> None:
    client = FakeGitHubRestClient(
        repository_payload(),
    )

    service = GitHubRepositoryService(
        rest_client=client,  # type: ignore[arg-type]
    )

    with pytest.raises(ValueError):
        await service.load_repository_policy(
            create_repository(),
            policy_path=policy_path,
        )

    assert client.calls == []