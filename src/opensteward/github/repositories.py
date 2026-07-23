"""GitHub repository metadata and OpenSteward policy retrieval."""

import base64
import binascii
from typing import Any
from urllib.parse import quote

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
)

from opensteward.github.models import (
    GitHubRepositoryRef,
    StrictGitHubModel,
)
from opensteward.github.rest_client import (
    GitHubRestClient,
    GitHubRestResponseError,
)
from opensteward.policy import (
    DEFAULT_POLICY_FILENAME,
    LoadedRepositoryPolicy,
    PolicySource,
    create_default_policy_result,
    normalize_repository_path,
    parse_repository_policy_with_metadata,
)


MAX_REPOSITORY_POLICY_BYTES = 256 * 1024


class GitHubRepositoryPolicyFileError(ValueError):
    """Raised when a repository policy file cannot be decoded."""


class GitHubRepositoryOwner(StrictGitHubModel):
    """Minimal GitHub account information for a repository owner."""

    id: int = Field(gt=0)
    login: str = Field(min_length=1)
    type: str = Field(min_length=1)


class GitHubRepositoryMetadata(StrictGitHubModel):
    """Repository information needed by OpenSteward."""

    id: int = Field(gt=0)
    name: str = Field(min_length=1)
    full_name: str = Field(min_length=1)

    private: bool
    fork: bool
    archived: bool
    disabled: bool

    html_url: str = Field(min_length=1)
    default_branch: str = Field(min_length=1)

    owner: GitHubRepositoryOwner


class GitHubRepositoryPolicyFile(StrictGitHubModel):
    """Metadata describing a policy file retrieved from GitHub."""

    path: str = Field(min_length=1)
    sha: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    git_ref: str = Field(min_length=1)
    html_url: str | None = None


class GitHubRepositoryPolicyResult(StrictGitHubModel):
    """Repository metadata and its active OpenSteward policy."""

    repository: GitHubRepositoryMetadata

    loaded_policy: LoadedRepositoryPolicy

    requested_path: str = Field(min_length=1)
    requested_ref: str = Field(min_length=1)

    policy_file: GitHubRepositoryPolicyFile | None = None

    @computed_field
    @property
    def policy_file_present(self) -> bool:
        """Return whether the repository contained the policy file."""

        return self.policy_file is not None


class _GitHubContentFileResponse(BaseModel):
    """Validated subset of GitHub's repository-content response."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    type: str = Field(min_length=1)
    encoding: str | None = None

    size: int = Field(ge=0)

    name: str = Field(min_length=1)
    path: str = Field(min_length=1)
    sha: str = Field(min_length=1)

    content: str | None = None

    html_url: str | None = None


def _normalize_git_ref(
    value: str,
) -> str:
    """Validate a branch, tag, or commit reference."""

    normalized = value.strip()

    if not normalized:
        raise ValueError(
            "GitHub repository ref must not be empty."
        )

    if any(
        ord(character) < 32
        for character in normalized
    ):
        raise ValueError(
            "GitHub repository ref must not contain control characters."
        )

    return normalized


def _build_repository_path(
    repository: GitHubRepositoryRef,
) -> str:
    """Build the REST path for repository metadata."""

    owner = quote(
        repository.owner,
        safe="",
    )

    name = quote(
        repository.name,
        safe="",
    )

    return f"/repos/{owner}/{name}"


def _build_repository_content_path(
    repository: GitHubRepositoryRef,
    file_path: str,
) -> str:
    """Build the REST path for one repository file."""

    repository_path = _build_repository_path(
        repository
    )

    encoded_file_path = quote(
        file_path,
        safe="/",
    )

    return (
        f"{repository_path}"
        f"/contents/{encoded_file_path}"
    )


def _build_policy_source_reference(
    *,
    repository: GitHubRepositoryMetadata,
    git_ref: str,
    policy_path: str,
) -> str:
    """Build a human-readable GitHub policy source reference."""

    return (
        f"github:{repository.full_name}"
        f"@{git_ref}:{policy_path}"
    )


def _decode_policy_content(
    file_response: _GitHubContentFileResponse,
) -> str:
    """Decode and validate a GitHub repository policy file."""

    if file_response.type != "file":
        raise GitHubRepositoryPolicyFileError(
            "The configured OpenSteward policy path does not "
            "refer to a regular repository file."
        )

    if file_response.encoding != "base64":
        raise GitHubRepositoryPolicyFileError(
            "GitHub returned the OpenSteward policy using an "
            f"unsupported encoding: {file_response.encoding!r}."
        )

    if file_response.size > MAX_REPOSITORY_POLICY_BYTES:
        raise GitHubRepositoryPolicyFileError(
            "The OpenSteward policy file exceeds the maximum "
            f"supported size of {MAX_REPOSITORY_POLICY_BYTES} bytes."
        )

    if file_response.content is None:
        raise GitHubRepositoryPolicyFileError(
            "GitHub did not return content for the OpenSteward "
            "policy file."
        )

    compact_content = "".join(
        file_response.content.split()
    )

    try:
        decoded = base64.b64decode(
            compact_content,
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise GitHubRepositoryPolicyFileError(
            "GitHub returned invalid Base64 content for the "
            "OpenSteward policy file."
        ) from exc

    if len(decoded) > MAX_REPOSITORY_POLICY_BYTES:
        raise GitHubRepositoryPolicyFileError(
            "The decoded OpenSteward policy file exceeds the "
            f"maximum supported size of "
            f"{MAX_REPOSITORY_POLICY_BYTES} bytes."
        )

    if len(decoded) != file_response.size:
        raise GitHubRepositoryPolicyFileError(
            "The decoded OpenSteward policy size does not match "
            "the size reported by GitHub."
        )

    try:
        return decoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GitHubRepositoryPolicyFileError(
            "The OpenSteward policy file must use UTF-8 encoding."
        ) from exc


class GitHubRepositoryService:
    """Retrieve repository metadata and repository policy files."""

    def __init__(
        self,
        *,
        rest_client: GitHubRestClient,
    ) -> None:
        self._rest_client = rest_client

    async def get_repository(
        self,
        repository: GitHubRepositoryRef,
    ) -> GitHubRepositoryMetadata:
        """Retrieve repository metadata from GitHub."""

        response = await self._rest_client.get_json(
            _build_repository_path(repository),
            response_type=GitHubRepositoryMetadata,
        )

        return response.data

    async def load_repository_policy(
        self,
        repository: GitHubRepositoryRef,
        *,
        policy_path: str = DEFAULT_POLICY_FILENAME,
        git_ref: str | None = None,
    ) -> GitHubRepositoryPolicyResult:
        """Load a repository's OpenSteward policy from GitHub.

        A missing policy file activates OpenSteward's built-in defaults.
        Other GitHub errors, malformed content, and invalid YAML are
        propagated to the caller.
        """

        normalized_policy_path = normalize_repository_path(
            policy_path
        )

        repository_metadata = await self.get_repository(
            repository
        )

        selected_ref = _normalize_git_ref(
            git_ref
            or repository_metadata.default_branch
        )

        source_reference = (
            _build_policy_source_reference(
                repository=repository_metadata,
                git_ref=selected_ref,
                policy_path=normalized_policy_path,
            )
        )

        content_path = (
            _build_repository_content_path(
                repository,
                normalized_policy_path,
            )
        )

        try:
            response = await self._rest_client.get_json(
                content_path,
                params={
                    "ref": selected_ref,
                },
                response_type=Any,
            )
        except GitHubRestResponseError as exc:
            if exc.status_code != 404:
                raise

            return GitHubRepositoryPolicyResult(
                repository=repository_metadata,
                loaded_policy=create_default_policy_result(
                    source_reference=source_reference,
                ),
                requested_path=normalized_policy_path,
                requested_ref=selected_ref,
                policy_file=None,
            )

        if not isinstance(response.data, dict):
            raise GitHubRepositoryPolicyFileError(
                "GitHub returned a directory or unsupported value "
                "for the OpenSteward policy path."
            )

        try:
            file_response = (
                _GitHubContentFileResponse
                .model_validate(response.data)
            )
        except ValidationError as exc:
            raise GitHubRepositoryPolicyFileError(
                "GitHub returned an invalid repository-content "
                "response for the OpenSteward policy file."
            ) from exc

        policy_content = _decode_policy_content(
            file_response
        )

        loaded_policy = (
            parse_repository_policy_with_metadata(
                policy_content,
                source=source_reference,
                policy_source=(
                    PolicySource.GITHUB_REPOSITORY
                ),
            )
        )

        return GitHubRepositoryPolicyResult(
            repository=repository_metadata,
            loaded_policy=loaded_policy,
            requested_path=normalized_policy_path,
            requested_ref=selected_ref,
            policy_file=GitHubRepositoryPolicyFile(
                path=file_response.path,
                sha=file_response.sha,
                size_bytes=file_response.size,
                git_ref=selected_ref,
                html_url=file_response.html_url,
            ),
        )