"""Authenticated, read-only GitHub REST API client."""

import re
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, Generic, Protocol, TypeVar
from urllib.parse import urlsplit

import httpx
from pydantic import (
    Field,
    TypeAdapter,
    ValidationError,
)

from opensteward.github.installation_tokens import (
    GitHubInstallationToken,
    GitHubInstallationTokenScope,
)
from opensteward.github.models import StrictGitHubModel
from opensteward.github.settings import GitHubAppSettings


DEFAULT_GITHUB_ACCEPT = "application/vnd.github+json"

_LINK_PATTERN = re.compile(
    r'<([^>]+)>\s*;\s*rel="([^"]+)"'
)

ResponseData = TypeVar("ResponseData")

QueryValue = str | int | float | bool | None

QueryParams = Mapping[
    str,
    QueryValue | list[QueryValue],
]


class InstallationTokenProvider(Protocol):
    """Behavior required from an installation-token provider."""

    async def get_token(
        self,
        installation_id: int,
        *,
        scope: GitHubInstallationTokenScope | None = None,
        force_refresh: bool = False,
    ) -> GitHubInstallationToken:
        """Return an installation token."""

        ...

    def invalidate(
        self,
        installation_id: int,
        *,
        scope: GitHubInstallationTokenScope | None = None,
    ) -> None:
        """Invalidate a cached installation token."""

        ...


class GitHubRateLimitMetadata(StrictGitHubModel):
    """Rate-limit metadata returned with a GitHub response."""

    limit: int | None = Field(
        default=None,
        ge=0,
    )

    remaining: int | None = Field(
        default=None,
        ge=0,
    )

    used: int | None = Field(
        default=None,
        ge=0,
    )

    reset_at: datetime | None = None

    resource: str | None = None


class GitHubPaginationLinks(StrictGitHubModel):
    """Pagination links parsed from GitHub's Link header."""

    next: str | None = None
    previous: str | None = None
    first: str | None = None
    last: str | None = None


class GitHubRestResponse(
    StrictGitHubModel,
    Generic[ResponseData],
):
    """Validated GitHub response and relevant response metadata."""

    status_code: int = Field(
        ge=100,
        le=599,
    )

    data: ResponseData

    request_id: str | None = None

    pagination: GitHubPaginationLinks | None = None

    rate_limit: GitHubRateLimitMetadata | None = None


class GitHubRestError(RuntimeError):
    """Base error for GitHub REST client failures."""


class GitHubRestTransportError(GitHubRestError):
    """Raised when an HTTP request cannot reach GitHub."""


class GitHubRestResponseError(GitHubRestError):
    """Raised when GitHub returns an unsuccessful or invalid response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        request_id: str | None = None,
        documentation_url: str | None = None,
        rate_limit: GitHubRateLimitMetadata | None = None,
        retry_after_seconds: int | None = None,
        rate_limited: bool = False,
    ) -> None:
        self.status_code = status_code
        self.request_id = request_id
        self.documentation_url = documentation_url
        self.rate_limit = rate_limit
        self.retry_after_seconds = retry_after_seconds
        self.rate_limited = rate_limited

        super().__init__(message)


def _parse_non_negative_integer(
    value: str | None,
) -> int | None:
    """Parse a non-negative response-header integer."""

    if value is None:
        return None

    try:
        parsed = int(value)
    except ValueError:
        return None

    if parsed < 0:
        return None

    return parsed


def _parse_rate_limit_metadata(
    response: httpx.Response,
) -> GitHubRateLimitMetadata | None:
    """Parse GitHub rate-limit response headers."""

    limit = _parse_non_negative_integer(
        response.headers.get("x-ratelimit-limit")
    )

    remaining = _parse_non_negative_integer(
        response.headers.get("x-ratelimit-remaining")
    )

    used = _parse_non_negative_integer(
        response.headers.get("x-ratelimit-used")
    )

    reset_epoch = _parse_non_negative_integer(
        response.headers.get("x-ratelimit-reset")
    )

    resource = response.headers.get(
        "x-ratelimit-resource"
    )

    if (
        limit is None
        and remaining is None
        and used is None
        and reset_epoch is None
        and resource is None
    ):
        return None

    reset_at = (
        datetime.fromtimestamp(
            reset_epoch,
            tz=UTC,
        )
        if reset_epoch is not None
        else None
    )

    return GitHubRateLimitMetadata(
        limit=limit,
        remaining=remaining,
        used=used,
        reset_at=reset_at,
        resource=resource,
    )


def _parse_pagination_links(
    response: httpx.Response,
) -> GitHubPaginationLinks | None:
    """Parse GitHub pagination URLs from a Link header."""

    link_header = response.headers.get("link")

    if not link_header:
        return None

    relations: dict[str, str] = {}

    for url, relation in _LINK_PATTERN.findall(
        link_header
    ):
        relations[relation] = url

    if not relations:
        return None

    pagination = GitHubPaginationLinks(
        next=relations.get("next"),
        previous=relations.get("prev"),
        first=relations.get("first"),
        last=relations.get("last"),
    )

    if not any(
        (
            pagination.next,
            pagination.previous,
            pagination.first,
            pagination.last,
        )
    ):
        return None

    return pagination


def _extract_error_details(
    response: httpx.Response,
) -> tuple[str, str | None]:
    """Extract a safe GitHub error message and documentation URL."""

    default_message = "GitHub returned an error response."

    try:
        payload = response.json()
    except ValueError:
        return default_message, None

    if not isinstance(payload, dict):
        return default_message, None

    raw_message = payload.get("message")

    message = (
        raw_message.strip()
        if isinstance(raw_message, str)
        and raw_message.strip()
        else default_message
    )

    raw_documentation_url = payload.get(
        "documentation_url"
    )

    documentation_url = (
        raw_documentation_url.strip()
        if isinstance(raw_documentation_url, str)
        and raw_documentation_url.strip()
        else None
    )

    return message, documentation_url


def _redact_secret(
    value: str,
    *,
    secret: str,
) -> str:
    """Remove a credential if it unexpectedly appears in text."""

    if not secret:
        return value

    return value.replace(
        secret,
        "[REDACTED]",
    )


def _validate_relative_api_path(
    path: str,
) -> str:
    """Validate a GitHub API-relative path.

    Full URLs are rejected so an installation token cannot be sent to
    a caller-controlled host.
    """

    normalized = path.strip()

    if not normalized.startswith("/"):
        raise ValueError(
            "GitHub API path must begin with '/'."
        )

    if normalized.startswith("//"):
        raise ValueError(
            "GitHub API path must not begin with '//'."
        )

    if "\\" in normalized:
        raise ValueError(
            "GitHub API path must not contain backslashes."
        )

    parsed = urlsplit(normalized)

    if parsed.scheme or parsed.netloc:
        raise ValueError(
            "GitHub API path must be relative."
        )

    if parsed.query:
        raise ValueError(
            "Pass query parameters through the params argument."
        )

    if parsed.fragment:
        raise ValueError(
            "GitHub API path must not contain a fragment."
        )

    path_parts = parsed.path.split("/")

    if any(
        part in {".", ".."}
        for part in path_parts
    ):
        raise ValueError(
            "GitHub API path must not contain '.' or '..' segments."
        )

    return parsed.path


class GitHubRestClient:
    """Read-only GitHub REST client using installation authentication."""

    def __init__(
        self,
        *,
        settings: GitHubAppSettings,
        token_provider: InstallationTokenProvider,
        client: httpx.AsyncClient,
        installation_id: int,
        token_scope: GitHubInstallationTokenScope | None = None,
    ) -> None:
        if installation_id <= 0:
            raise ValueError(
                "GitHub installation ID must be positive."
            )

        self._settings = settings
        self._token_provider = token_provider
        self._client = client
        self._installation_id = installation_id

        self._token_scope = (
            token_scope.model_copy(deep=True)
            if token_scope is not None
            else GitHubInstallationTokenScope()
        )

    def _build_url(
        self,
        path: str,
    ) -> str:
        """Build a GitHub REST URL from a validated relative path."""

        validated_path = _validate_relative_api_path(
            path
        )

        return (
            f"{self._settings.api_url}"
            f"{validated_path}"
        )

    async def _send_get(
        self,
        *,
        url: str,
        params: QueryParams | None,
        token: GitHubInstallationToken,
        accept: str,
    ) -> httpx.Response:
        """Send one authenticated GET request."""

        headers = {
            "Accept": accept,
            "Authorization": (
                token.authorization_header_value()
            ),
            "X-GitHub-Api-Version": (
                self._settings.api_version
            ),
            "User-Agent": self._settings.user_agent,
        }

        try:
            return await self._client.get(
                url,
                params=params,
                headers=headers,
                timeout=(
                    self._settings
                    .request_timeout_seconds
                ),
            )
        except httpx.RequestError as exc:
            raise GitHubRestTransportError(
                "Unable to reach GitHub while making a REST request."
            ) from exc

    def _raise_response_error(
        self,
        response: httpx.Response,
        *,
        token: GitHubInstallationToken,
    ) -> None:
        """Raise a safe structured error for a GitHub response."""

        message, documentation_url = (
            _extract_error_details(response)
        )

        raw_token = token.token.get_secret_value()

        safe_message = _redact_secret(
            message,
            secret=raw_token,
        )

        request_id = response.headers.get(
            "x-github-request-id"
        )

        rate_limit = _parse_rate_limit_metadata(
            response
        )

        retry_after_seconds = (
            _parse_non_negative_integer(
                response.headers.get("retry-after")
            )
        )

        remaining = (
            rate_limit.remaining
            if rate_limit is not None
            else None
        )

        rate_limited = (
            response.status_code in {403, 429}
            and (
                remaining == 0
                or retry_after_seconds is not None
                or "rate limit" in safe_message.casefold()
            )
        )

        error_message = (
            "GitHub REST request failed with status "
            f"{response.status_code}: {safe_message}"
        )

        if request_id:
            error_message = (
                f"{error_message} "
                f"(request id: {request_id})"
            )

        raise GitHubRestResponseError(
            error_message,
            status_code=response.status_code,
            request_id=request_id,
            documentation_url=documentation_url,
            rate_limit=rate_limit,
            retry_after_seconds=retry_after_seconds,
            rate_limited=rate_limited,
        )

    async def get_json(
        self,
        path: str,
        *,
        params: QueryParams | None = None,
        response_type: Any = Any,
        accept: str = DEFAULT_GITHUB_ACCEPT,
    ) -> GitHubRestResponse[Any]:
        """Send an authenticated GET request and validate its JSON body.

        A 401 response causes one installation-token refresh and one
        retry. Other error statuses are not retried automatically.
        """

        normalized_accept = accept.strip()

        if not normalized_accept:
            raise ValueError(
                "GitHub Accept header must not be empty."
            )

        url = self._build_url(path)

        token = await self._token_provider.get_token(
            self._installation_id,
            scope=self._token_scope,
        )

        response = await self._send_get(
            url=url,
            params=params,
            token=token,
            accept=normalized_accept,
        )

        if response.status_code == 401:
            self._token_provider.invalidate(
                self._installation_id,
                scope=self._token_scope,
            )

            token = await self._token_provider.get_token(
                self._installation_id,
                scope=self._token_scope,
                force_refresh=True,
            )

            response = await self._send_get(
                url=url,
                params=params,
                token=token,
                accept=normalized_accept,
            )

        if not response.is_success:
            self._raise_response_error(
                response,
                token=token,
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise GitHubRestResponseError(
                "GitHub returned an invalid JSON response.",
                status_code=response.status_code,
                request_id=response.headers.get(
                    "x-github-request-id"
                ),
                rate_limit=(
                    _parse_rate_limit_metadata(response)
                ),
            ) from exc

        try:
            validated_data = TypeAdapter(
                response_type
            ).validate_python(payload)
        except ValidationError as exc:
            raise GitHubRestResponseError(
                "GitHub returned JSON that did not match "
                "the expected response type.",
                status_code=response.status_code,
                request_id=response.headers.get(
                    "x-github-request-id"
                ),
                rate_limit=(
                    _parse_rate_limit_metadata(response)
                ),
            ) from exc

        return GitHubRestResponse(
            status_code=response.status_code,
            data=validated_data,
            request_id=response.headers.get(
                "x-github-request-id"
            ),
            pagination=_parse_pagination_links(
                response
            ),
            rate_limit=_parse_rate_limit_metadata(
                response
            ),
        )