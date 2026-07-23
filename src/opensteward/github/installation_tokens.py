"""GitHub App installation-token creation and caching."""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, TypeAlias

import httpx
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)

from opensteward.github.app_jwt import (
    GitHubAppJwt,
    generate_github_app_jwt,
)
from opensteward.github.models import StrictGitHubModel
from opensteward.github.settings import GitHubAppSettings


INSTALLATION_TOKEN_REFRESH_MARGIN = timedelta(minutes=5)


class GitHubInstallationTokenError(RuntimeError):
    """Raised when an installation token cannot be created."""


class GitHubPermissionLevel(StrEnum):
    """Permission levels returned or accepted by GitHub."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class GitHubRepositorySelection(StrEnum):
    """Repository selection represented by an installation token."""

    ALL = "all"
    SELECTED = "selected"


class GitHubInstallationTokenScope(StrictGitHubModel):
    """Optional restrictions for an installation access token."""

    repositories: list[str] = Field(
        default_factory=list,
        max_length=500,
    )

    permissions: dict[str, GitHubPermissionLevel] = Field(
        default_factory=dict,
    )

    @field_validator("repositories")
    @classmethod
    def validate_repositories(
        cls,
        repositories: list[str],
    ) -> list[str]:
        """Normalize repository names and reject unsafe duplicates."""

        normalized: list[str] = []
        seen: set[str] = set()

        for repository in repositories:
            name = repository.strip()

            if not name:
                raise ValueError(
                    "Scoped repository names must not be empty."
                )

            if "/" in name or "\\" in name:
                raise ValueError(
                    "Scoped repositories must use repository names, "
                    "not owner/name paths."
                )

            if name in {".", ".."}:
                raise ValueError(
                    "Scoped repository names must not be '.' or '..'."
                )

            comparison_key = name.casefold()

            if comparison_key in seen:
                raise ValueError(
                    "Scoped repository names must be unique."
                )

            seen.add(comparison_key)
            normalized.append(name)

        return normalized

    @field_validator("permissions")
    @classmethod
    def validate_permissions(
        cls,
        permissions: dict[str, GitHubPermissionLevel],
    ) -> dict[str, GitHubPermissionLevel]:
        """Normalize permission names and reject empty keys."""

        normalized: dict[str, GitHubPermissionLevel] = {}

        for raw_name, level in permissions.items():
            name = raw_name.strip()

            if not name:
                raise ValueError(
                    "GitHub permission names must not be empty."
                )

            if name in normalized:
                raise ValueError(
                    "GitHub permission names must be unique."
                )

            normalized[name] = level

        return normalized

    def request_body(self) -> dict[str, Any]:
        """Build the optional GitHub token-request body."""

        body: dict[str, Any] = {}

        if self.repositories:
            body["repositories"] = self.repositories

        if self.permissions:
            body["permissions"] = {
                name: level.value
                for name, level in self.permissions.items()
            }

        return body

    def cache_components(
        self,
    ) -> tuple[
        tuple[str, ...],
        tuple[tuple[str, str], ...],
    ]:
        """Return deterministic values for the token-cache key."""

        repositories = tuple(
            sorted(
                repository.casefold()
                for repository in self.repositories
            )
        )

        permissions = tuple(
            sorted(
                (
                    name,
                    level.value,
                )
                for name, level in self.permissions.items()
            )
        )

        return repositories, permissions


class GitHubTokenRepository(BaseModel):
    """Minimal repository identity returned with a scoped token."""

    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    id: int = Field(gt=0)
    name: str = Field(min_length=1)
    full_name: str = Field(min_length=1)


def _normalize_utc_datetime(
    value: datetime,
    *,
    description: str,
) -> datetime:
    """Require a timezone-aware datetime and normalize it to UTC."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{description} must be timezone-aware."
        )

    return value.astimezone(UTC)


class GitHubInstallationToken(StrictGitHubModel):
    """A masked, short-lived GitHub App installation token."""

    token: SecretStr
    installation_id: int = Field(gt=0)
    expires_at: datetime

    permissions: dict[str, GitHubPermissionLevel] = Field(
        default_factory=dict,
    )

    repository_selection: GitHubRepositorySelection

    repositories: list[GitHubTokenRepository] = Field(
        default_factory=list,
    )

    @field_validator("token")
    @classmethod
    def validate_token(
        cls,
        token: SecretStr,
    ) -> SecretStr:
        """Reject empty tokens without assuming a token format."""

        if not token.get_secret_value().strip():
            raise ValueError(
                "GitHub installation token must not be empty."
            )

        return token

    @field_validator("expires_at")
    @classmethod
    def validate_expiration(
        cls,
        expires_at: datetime,
    ) -> datetime:
        """Require a timezone-aware expiration timestamp."""

        return _normalize_utc_datetime(
            expires_at,
            description="GitHub token expiration",
        )

    def authorization_header_value(self) -> str:
        """Return an HTTP Authorization header value.

        The returned value must not be logged or persisted.
        """

        return (
            "Bearer "
            f"{self.token.get_secret_value()}"
        )

    def is_usable(
        self,
        *,
        at: datetime,
        refresh_margin: timedelta = (
            INSTALLATION_TOKEN_REFRESH_MARGIN
        ),
    ) -> bool:
        """Return whether the token remains usable beyond the margin."""

        if refresh_margin < timedelta(0):
            raise ValueError(
                "Token refresh margin must not be negative."
            )

        normalized_time = _normalize_utc_datetime(
            at,
            description="Token usability time",
        )

        return (
            self.expires_at
            > normalized_time + refresh_margin
        )


class _InstallationTokenApiResponse(BaseModel):
    """Validated subset of GitHub's token endpoint response."""

    model_config = ConfigDict(
        extra="ignore",
    )

    token: SecretStr
    expires_at: datetime

    permissions: dict[str, GitHubPermissionLevel] = Field(
        default_factory=dict,
    )

    repository_selection: GitHubRepositorySelection

    repositories: list[GitHubTokenRepository] = Field(
        default_factory=list,
    )

    @field_validator("token")
    @classmethod
    def validate_token(
        cls,
        token: SecretStr,
    ) -> SecretStr:
        if not token.get_secret_value().strip():
            raise ValueError(
                "GitHub returned an empty installation token."
            )

        return token

    @field_validator("expires_at")
    @classmethod
    def validate_expiration(
        cls,
        expires_at: datetime,
    ) -> datetime:
        return _normalize_utc_datetime(
            expires_at,
            description="GitHub token expiration",
        )


@dataclass(frozen=True)
class _TokenCacheKey:
    """Internal cache identity for one installation-token scope."""

    installation_id: int
    repositories: tuple[str, ...]
    permissions: tuple[tuple[str, str], ...]


Clock: TypeAlias = Callable[[], datetime]

AppJwtFactory: TypeAlias = Callable[
    [GitHubAppSettings, datetime],
    GitHubAppJwt,
]


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(UTC)


def _default_app_jwt_factory(
    settings: GitHubAppSettings,
    now: datetime,
) -> GitHubAppJwt:
    """Generate the App JWT used by the token endpoint."""

    return generate_github_app_jwt(
        settings,
        now=now,
    )


def _extract_github_error_message(
    response: httpx.Response,
) -> str:
    """Extract a safe message from a GitHub error response."""

    try:
        payload = response.json()
    except ValueError:
        return "GitHub returned an unreadable error response."

    if isinstance(payload, dict):
        message = payload.get("message")

        if isinstance(message, str) and message.strip():
            return message.strip()

    return "GitHub returned an error response."


class GitHubInstallationTokenProvider:
    """Create and cache GitHub App installation access tokens."""

    def __init__(
        self,
        *,
        settings: GitHubAppSettings,
        client: httpx.AsyncClient,
        clock: Clock | None = None,
        app_jwt_factory: AppJwtFactory | None = None,
        refresh_margin: timedelta = (
            INSTALLATION_TOKEN_REFRESH_MARGIN
        ),
    ) -> None:
        if refresh_margin < timedelta(0):
            raise ValueError(
                "Token refresh margin must not be negative."
            )

        self._settings = settings
        self._client = client
        self._clock = clock or _utc_now

        self._app_jwt_factory = (
            app_jwt_factory
            or _default_app_jwt_factory
        )

        self._refresh_margin = refresh_margin

        self._cache: dict[
            _TokenCacheKey,
            GitHubInstallationToken,
        ] = {}

        self._refresh_lock = asyncio.Lock()

    def _current_time(self) -> datetime:
        """Return a normalized current time from the injected clock."""

        try:
            return _normalize_utc_datetime(
                self._clock(),
                description="GitHub token-provider clock",
            )
        except ValueError as exc:
            raise GitHubInstallationTokenError(
                str(exc)
            ) from exc

    @staticmethod
    def _build_cache_key(
        installation_id: int,
        scope: GitHubInstallationTokenScope,
    ) -> _TokenCacheKey:
        repositories, permissions = (
            scope.cache_components()
        )

        return _TokenCacheKey(
            installation_id=installation_id,
            repositories=repositories,
            permissions=permissions,
        )

    async def get_token(
        self,
        installation_id: int,
        *,
        scope: GitHubInstallationTokenScope | None = None,
        force_refresh: bool = False,
    ) -> GitHubInstallationToken:
        """Return a cached token or request a new installation token."""

        if installation_id <= 0:
            raise ValueError(
                "GitHub installation ID must be positive."
            )

        effective_scope = (
            scope
            or GitHubInstallationTokenScope()
        )

        cache_key = self._build_cache_key(
            installation_id,
            effective_scope,
        )

        current_time = self._current_time()
        cached_token = self._cache.get(cache_key)

        if (
            not force_refresh
            and cached_token is not None
            and cached_token.is_usable(
                at=current_time,
                refresh_margin=self._refresh_margin,
            )
        ):
            return cached_token

        async with self._refresh_lock:
            # Check again because another coroutine may have refreshed
            # the same token while this coroutine waited for the lock.
            current_time = self._current_time()
            cached_token = self._cache.get(cache_key)

            if (
                not force_refresh
                and cached_token is not None
                and cached_token.is_usable(
                    at=current_time,
                    refresh_margin=self._refresh_margin,
                )
            ):
                return cached_token

            token = await self._request_token(
                installation_id=installation_id,
                scope=effective_scope,
                current_time=current_time,
            )

            self._cache[cache_key] = token

            return token

    async def _request_token(
        self,
        *,
        installation_id: int,
        scope: GitHubInstallationTokenScope,
        current_time: datetime,
    ) -> GitHubInstallationToken:
        """Request one installation token from GitHub."""

        app_jwt = self._app_jwt_factory(
            self._settings,
            current_time,
        )

        url = (
            f"{self._settings.api_url}"
            f"/app/installations/{installation_id}"
            "/access_tokens"
        )

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": (
                app_jwt.authorization_header_value()
            ),
            "X-GitHub-Api-Version": (
                self._settings.api_version
            ),
            "User-Agent": self._settings.user_agent,
        }

        request_body = scope.request_body()

        try:
            if request_body:
                response = await self._client.post(
                    url,
                    headers=headers,
                    json=request_body,
                    timeout=(
                        self._settings
                        .request_timeout_seconds
                    ),
                )
            else:
                response = await self._client.post(
                    url,
                    headers=headers,
                    timeout=(
                        self._settings
                        .request_timeout_seconds
                    ),
                )
        except httpx.RequestError as exc:
            raise GitHubInstallationTokenError(
                "Unable to reach GitHub while creating an "
                f"installation token for installation "
                f"{installation_id}."
            ) from exc

        if response.status_code != 201:
            detail = _extract_github_error_message(
                response
            )

            raise GitHubInstallationTokenError(
                "GitHub rejected the installation-token request "
                f"for installation {installation_id} with status "
                f"{response.status_code}: {detail}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise GitHubInstallationTokenError(
                "GitHub returned invalid JSON while creating an "
                "installation token."
            ) from exc

        try:
            parsed = (
                _InstallationTokenApiResponse
                .model_validate(payload)
            )
        except ValidationError as exc:
            raise GitHubInstallationTokenError(
                "GitHub returned an invalid installation-token "
                "response."
            ) from exc

        if parsed.expires_at <= current_time:
            raise GitHubInstallationTokenError(
                "GitHub returned an installation token that is "
                "already expired."
            )

        return GitHubInstallationToken(
            token=parsed.token,
            installation_id=installation_id,
            expires_at=parsed.expires_at,
            permissions=parsed.permissions,
            repository_selection=(
                parsed.repository_selection
            ),
            repositories=parsed.repositories,
        )

    def invalidate(
        self,
        installation_id: int,
        *,
        scope: GitHubInstallationTokenScope | None = None,
    ) -> None:
        """Remove one cached installation token."""

        effective_scope = (
            scope
            or GitHubInstallationTokenScope()
        )

        cache_key = self._build_cache_key(
            installation_id,
            effective_scope,
        )

        self._cache.pop(cache_key, None)

    def clear(self) -> None:
        """Remove every cached installation token."""

        self._cache.clear()