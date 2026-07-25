"""Bounded application and GitHub dependency readiness checks."""

import asyncio
from collections.abc import Callable
from enum import StrEnum
from time import monotonic
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from opensteward.github.app_jwt import (
    GitHubJwtGenerationError,
    generate_github_app_jwt,
)
from opensteward.github.settings import (
    GitHubAppSettings,
    GitHubConfigurationError,
    get_github_settings,
)
from opensteward.settings import Settings

READINESS_CACHE_TTL_SECONDS = 30.0
MAX_READINESS_INSTALLATIONS = 100

GITHUB_READINESS_REQUIRED_PERMISSIONS = frozenset(
    {
        "checks",
        "contents",
        "issues",
        "pull_requests",
    }
)

_SUFFICIENT_PERMISSION_LEVELS = frozenset(
    {
        "read",
        "write",
        "admin",
    }
)


class ReadinessCheckStatus(StrEnum):
    """State of one readiness dependency check."""

    READY = "ready"
    NOT_READY = "not_ready"
    NOT_CHECKED = "not_checked"


class ReadinessStatus(StrEnum):
    """Overall application readiness state."""

    READY = "ready"
    NOT_READY = "not_ready"


class ReadinessChecks(BaseModel):
    """Individual readiness check states."""

    model_config = ConfigDict(extra="forbid")

    mcp: ReadinessCheckStatus
    mcp_authentication: ReadinessCheckStatus
    github_credentials: ReadinessCheckStatus
    github_api: ReadinessCheckStatus
    github_installations: ReadinessCheckStatus


class ReadinessAssessment(BaseModel):
    """Complete sanitized readiness result."""

    model_config = ConfigDict(extra="forbid")

    status: ReadinessStatus
    checks: ReadinessChecks
    issues: list[str] = Field(default_factory=list)


class _GitHubAppPayload(BaseModel):
    """Minimal authenticated GitHub App response."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(gt=0)


class _GitHubInstallationPayload(BaseModel):
    """Minimal GitHub App installation response."""

    model_config = ConfigDict(extra="ignore")

    id: int = Field(gt=0)
    permissions: dict[str, str]


class _GitHubReadinessResult(BaseModel):
    """Internal GitHub dependency check result."""

    model_config = ConfigDict(extra="forbid")

    credentials: ReadinessCheckStatus
    api: ReadinessCheckStatus
    installations: ReadinessCheckStatus
    issues: list[str] = Field(default_factory=list)


type GitHubSettingsFactory = Callable[[], GitHubAppSettings]
type HttpClientFactory = Callable[..., httpx.AsyncClient]
type MonotonicClock = Callable[[], float]


def _default_http_client_factory(
    **kwargs: Any,
) -> httpx.AsyncClient:
    return httpx.AsyncClient(**kwargs)


def _not_ready_github_result(
    *,
    credentials: ReadinessCheckStatus,
    api: ReadinessCheckStatus,
    installations: ReadinessCheckStatus,
    issue: str,
) -> _GitHubReadinessResult:
    return _GitHubReadinessResult(
        credentials=credentials,
        api=api,
        installations=installations,
        issues=[issue],
    )


class GitHubReadinessProbe:
    """Validate GitHub App identity, installations, and permission grants."""

    def __init__(
        self,
        *,
        settings_factory: GitHubSettingsFactory = get_github_settings,
        http_client_factory: HttpClientFactory = _default_http_client_factory,
    ) -> None:
        self._settings_factory = settings_factory
        self._http_client_factory = http_client_factory

    async def check(
        self,
        installation_ids: tuple[int, ...],
    ) -> _GitHubReadinessResult:
        """Run one bounded GitHub dependency probe."""

        settings = self._settings_factory()

        if not settings.configured:
            return _not_ready_github_result(
                credentials=ReadinessCheckStatus.NOT_READY,
                api=ReadinessCheckStatus.NOT_CHECKED,
                installations=ReadinessCheckStatus.NOT_CHECKED,
                issue="GitHub App authentication is not configured.",
            )

        if len(installation_ids) > MAX_READINESS_INSTALLATIONS:
            return _not_ready_github_result(
                credentials=ReadinessCheckStatus.NOT_CHECKED,
                api=ReadinessCheckStatus.NOT_CHECKED,
                installations=ReadinessCheckStatus.NOT_READY,
                issue=(
                    "Configured GitHub installation count exceeds the "
                    "readiness safety limit."
                ),
            )

        try:
            app_jwt = generate_github_app_jwt(settings)
        except (
            GitHubConfigurationError,
            GitHubJwtGenerationError,
        ):
            return _not_ready_github_result(
                credentials=ReadinessCheckStatus.NOT_READY,
                api=ReadinessCheckStatus.NOT_CHECKED,
                installations=ReadinessCheckStatus.NOT_CHECKED,
                issue="GitHub App credentials or private key are invalid.",
            )

        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": app_jwt.authorization_header_value(),
            "User-Agent": settings.user_agent,
            "X-GitHub-Api-Version": settings.api_version,
        }

        try:
            async with self._http_client_factory(
                follow_redirects=False,
                timeout=httpx.Timeout(
                    settings.request_timeout_seconds
                ),
            ) as client:
                app_response = await client.get(
                    f"{settings.api_url}/app",
                    headers=headers,
                )

                app_error = self._validate_app_response(
                    app_response,
                    expected_app_id=settings.app_id,
                )

                if app_error is not None:
                    return _not_ready_github_result(
                        credentials=ReadinessCheckStatus.READY,
                        api=ReadinessCheckStatus.NOT_READY,
                        installations=ReadinessCheckStatus.NOT_CHECKED,
                        issue=app_error,
                    )

                if not installation_ids:
                    return _not_ready_github_result(
                        credentials=ReadinessCheckStatus.READY,
                        api=ReadinessCheckStatus.READY,
                        installations=ReadinessCheckStatus.NOT_READY,
                        issue="No GitHub installations are authorized for MCP callers.",
                    )

                installation_issues = await asyncio.gather(
                    *(
                        self._check_installation(
                            client=client,
                            settings=settings,
                            headers=headers,
                            installation_id=installation_id,
                        )
                        for installation_id in installation_ids
                    )
                )
        except httpx.HTTPError:
            return _not_ready_github_result(
                credentials=ReadinessCheckStatus.READY,
                api=ReadinessCheckStatus.NOT_READY,
                installations=ReadinessCheckStatus.NOT_CHECKED,
                issue="GitHub API connectivity check failed.",
            )

        issues = [
            issue
            for issue in installation_issues
            if issue is not None
        ]

        return _GitHubReadinessResult(
            credentials=ReadinessCheckStatus.READY,
            api=ReadinessCheckStatus.READY,
            installations=(
                ReadinessCheckStatus.NOT_READY
                if issues
                else ReadinessCheckStatus.READY
            ),
            issues=issues,
        )

    @staticmethod
    def _validate_app_response(
        response: httpx.Response,
        *,
        expected_app_id: int | None,
    ) -> str | None:
        if response.status_code != 200:
            return (
                "GitHub rejected the App authentication readiness check "
                f"with status {response.status_code}."
            )

        try:
            payload = _GitHubAppPayload.model_validate(
                response.json()
            )
        except (ValueError, ValidationError):
            return "GitHub returned an invalid App readiness response."

        if payload.id != expected_app_id:
            return "GitHub App identity does not match configured app_id."

        return None

    async def _check_installation(
        self,
        *,
        client: httpx.AsyncClient,
        settings: GitHubAppSettings,
        headers: dict[str, str],
        installation_id: int,
    ) -> str | None:
        try:
            response = await client.get(
                (
                    f"{settings.api_url}/app/installations/"
                    f"{installation_id}"
                ),
                headers=headers,
            )
        except httpx.HTTPError:
            return (
                f"GitHub installation {installation_id} "
                "connectivity check failed."
            )

        if response.status_code != 200:
            return (
                f"GitHub installation {installation_id} is unavailable "
                f"with status {response.status_code}."
            )

        try:
            payload = _GitHubInstallationPayload.model_validate(
                response.json()
            )
        except (ValueError, ValidationError):
            return (
                f"GitHub installation {installation_id} returned "
                "an invalid readiness response."
            )

        if payload.id != installation_id:
            return (
                f"GitHub installation {installation_id} identity "
                "does not match the response."
            )

        missing_permissions = sorted(
            permission
            for permission in GITHUB_READINESS_REQUIRED_PERMISSIONS
            if payload.permissions.get(permission)
            not in _SUFFICIENT_PERMISSION_LEVELS
        )

        if missing_permissions:
            return (
                f"GitHub installation {installation_id} lacks required "
                f"permissions: {', '.join(missing_permissions)}."
            )

        return None


class ReadinessService:
    """Combine local configuration and cached GitHub dependency checks."""

    def __init__(
        self,
        *,
        settings: Settings,
        github_probe: GitHubReadinessProbe | None = None,
        cache_ttl_seconds: float = READINESS_CACHE_TTL_SECONDS,
        clock: MonotonicClock = monotonic,
    ) -> None:
        if cache_ttl_seconds < 0:
            raise ValueError(
                "Readiness cache TTL must not be negative."
            )

        self._settings = settings
        self._github_probe = github_probe or GitHubReadinessProbe()
        self._cache_ttl_seconds = cache_ttl_seconds
        self._clock = clock
        self._lock = asyncio.Lock()
        self._cached_key: tuple[bool, tuple[int, ...]] | None = None
        self._cached_until = 0.0
        self._cached_assessment: ReadinessAssessment | None = None

    async def assess(self) -> ReadinessAssessment:
        """Return one cached or freshly evaluated readiness assessment."""

        installation_ids = tuple(
            sorted(
                {
                    installation_id
                    for caller in (
                        self._settings
                        .mcp_authorized_callers
                        .values()
                    )
                    for installation_id in caller.installation_ids
                }
            )
        )
        authentication_configured = (
            self._settings.mcp_authentication_configured
        )
        cache_key = (
            authentication_configured,
            installation_ids,
        )

        async with self._lock:
            now = self._clock()

            if (
                self._cached_assessment is not None
                and self._cached_key == cache_key
                and now < self._cached_until
            ):
                return self._cached_assessment

            github = await self._github_probe.check(
                installation_ids
            )
            issues = list(github.issues)

            if not authentication_configured:
                issues.insert(
                    0,
                    "MCP caller authentication is not configured.",
                )

            checks = ReadinessChecks(
                mcp=ReadinessCheckStatus.READY,
                mcp_authentication=(
                    ReadinessCheckStatus.READY
                    if authentication_configured
                    else ReadinessCheckStatus.NOT_READY
                ),
                github_credentials=github.credentials,
                github_api=github.api,
                github_installations=github.installations,
            )
            ready = all(
                check == ReadinessCheckStatus.READY
                for check in (
                    checks.mcp,
                    checks.mcp_authentication,
                    checks.github_credentials,
                    checks.github_api,
                    checks.github_installations,
                )
            )
            assessment = ReadinessAssessment(
                status=(
                    ReadinessStatus.READY
                    if ready
                    else ReadinessStatus.NOT_READY
                ),
                checks=checks,
                issues=issues,
            )

            self._cached_key = cache_key
            self._cached_until = now + self._cache_ttl_seconds
            self._cached_assessment = assessment
            return assessment
