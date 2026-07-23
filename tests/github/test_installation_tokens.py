"""Tests for GitHub App installation-token creation and caching."""

import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from opensteward.github import (
    GitHubAppJwt,
    GitHubAppSettings,
    GitHubInstallationTokenError,
    GitHubInstallationTokenProvider,
    GitHubInstallationTokenScope,
    GitHubPermissionLevel,
    GitHubRepositorySelection,
)


FIXED_NOW = datetime(
    2026,
    7,
    23,
    12,
    0,
    0,
    tzinfo=UTC,
)


TEST_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
test-key-used-with-fake-jwt-factory
-----END RSA PRIVATE KEY-----"""


@pytest.fixture
def anyio_backend() -> str:
    """Run async tests using asyncio."""

    return "asyncio"


def create_settings() -> GitHubAppSettings:
    """Create GitHub settings for transport-level tests."""

    return GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=TEST_PRIVATE_KEY,
        api_url="https://api.github.test",
        api_version="2026-03-10",
        user_agent="OpenSteward/Test",
        request_timeout_seconds=10,
    )


def fake_app_jwt_factory(
    settings: GitHubAppSettings,
    now: datetime,
) -> GitHubAppJwt:
    """Return a deterministic App JWT without RSA signing."""

    assert settings.app_id is not None

    return GitHubAppJwt(
        token=SecretStr("test-app-jwt"),
        issuer=str(settings.app_id),
        issued_at=now - timedelta(seconds=60),
        expires_at=now + timedelta(minutes=9),
    )


def token_response(
    *,
    token: str = "ghs_test_installation_token",
    expires_at: datetime = FIXED_NOW + timedelta(hours=1),
    repository_selection: str = "selected",
) -> dict[str, object]:
    """Create a representative GitHub token response."""

    return {
        "token": token,
        "expires_at": expires_at.isoformat().replace(
            "+00:00",
            "Z",
        ),
        "permissions": {
            "contents": "read",
            "pull_requests": "read",
        },
        "repository_selection": repository_selection,
        "repositories": [
            {
                "id": 1001,
                "name": "framework",
                "full_name": "acme/framework",
                "extra_github_field": "ignored",
            }
        ],
    }


def test_scope_builds_github_request_body() -> None:
    scope = GitHubInstallationTokenScope(
        repositories=[
            "framework",
        ],
        permissions={
            "contents": GitHubPermissionLevel.READ,
            "pull_requests": GitHubPermissionLevel.READ,
        },
    )

    assert scope.request_body() == {
        "repositories": [
            "framework",
        ],
        "permissions": {
            "contents": "read",
            "pull_requests": "read",
        },
    }


def test_scope_rejects_duplicate_repository_names() -> None:
    with pytest.raises(
        ValidationError,
        match="must be unique",
    ):
        GitHubInstallationTokenScope(
            repositories=[
                "Framework",
                "framework",
            ]
        )


@pytest.mark.parametrize(
    "repository",
    [
        "",
        "acme/framework",
        "../framework",
        "src\\framework",
    ],
)
def test_scope_rejects_invalid_repository_names(
    repository: str,
) -> None:
    with pytest.raises(ValidationError):
        GitHubInstallationTokenScope(
            repositories=[repository],
        )


@pytest.mark.anyio
async def test_provider_requests_scoped_token() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            status_code=201,
            json=token_response(),
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        scope = GitHubInstallationTokenScope(
            repositories=[
                "framework",
            ],
            permissions={
                "contents": "read",
                "pull_requests": "read",
            },
        )

        token = await provider.get_token(
            987654,
            scope=scope,
        )

    assert len(captured_requests) == 1

    request = captured_requests[0]

    assert (
        str(request.url)
        == (
            "https://api.github.test/app/installations/"
            "987654/access_tokens"
        )
    )

    assert request.method == "POST"

    assert (
        request.headers["authorization"]
        == "Bearer test-app-jwt"
    )

    assert (
        request.headers["accept"]
        == "application/vnd.github+json"
    )

    assert (
        request.headers["x-github-api-version"]
        == "2026-03-10"
    )

    assert (
        request.headers["user-agent"]
        == "OpenSteward/Test"
    )

    assert json.loads(request.content) == {
        "repositories": [
            "framework",
        ],
        "permissions": {
            "contents": "read",
            "pull_requests": "read",
        },
    }

    assert token.installation_id == 987654
    assert (
        token.repository_selection
        == GitHubRepositorySelection.SELECTED
    )
    assert token.permissions == {
        "contents": GitHubPermissionLevel.READ,
        "pull_requests": GitHubPermissionLevel.READ,
    }
    assert token.repositories[0].full_name == "acme/framework"


@pytest.mark.anyio
async def test_unscoped_request_has_no_json_body() -> None:
    captured_content: list[bytes] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        captured_content.append(request.content)

        return httpx.Response(
            status_code=201,
            json=token_response(
                repository_selection="all",
            ),
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        token = await provider.get_token(987654)

    assert captured_content == [b""]

    assert (
        token.repository_selection
        == GitHubRepositorySelection.ALL
    )


@pytest.mark.anyio
async def test_provider_reuses_cached_token() -> None:
    request_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            status_code=201,
            json=token_response(),
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        first = await provider.get_token(987654)
        second = await provider.get_token(987654)

    assert request_count == 1
    assert first is second


@pytest.mark.anyio
async def test_provider_refreshes_near_expiration() -> None:
    current_time = FIXED_NOW
    request_count = 0

    def clock() -> datetime:
        return current_time

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            status_code=201,
            json=token_response(
                token=f"ghs_token_{request_count}",
                expires_at=current_time + timedelta(hours=1),
            ),
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=clock,
            app_jwt_factory=fake_app_jwt_factory,
        )

        first = await provider.get_token(987654)

        current_time = (
            FIXED_NOW
            + timedelta(minutes=56)
        )

        second = await provider.get_token(987654)

    assert request_count == 2

    assert (
        first.token.get_secret_value()
        == "ghs_token_1"
    )

    assert (
        second.token.get_secret_value()
        == "ghs_token_2"
    )


@pytest.mark.anyio
async def test_different_scopes_use_different_cache_entries() -> None:
    request_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            status_code=201,
            json=token_response(
                token=f"ghs_token_{request_count}",
            ),
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        framework_scope = GitHubInstallationTokenScope(
            repositories=["framework"],
        )

        website_scope = GitHubInstallationTokenScope(
            repositories=["website"],
        )

        await provider.get_token(
            987654,
            scope=framework_scope,
        )

        await provider.get_token(
            987654,
            scope=website_scope,
        )

    assert request_count == 2


@pytest.mark.anyio
async def test_force_refresh_bypasses_cache() -> None:
    request_count = 0

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            status_code=201,
            json=token_response(
                token=f"ghs_token_{request_count}",
            ),
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        await provider.get_token(987654)

        refreshed = await provider.get_token(
            987654,
            force_refresh=True,
        )

    assert request_count == 2

    assert (
        refreshed.token.get_secret_value()
        == "ghs_token_2"
    )


@pytest.mark.anyio
async def test_provider_reports_github_error() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=404,
            json={
                "message": "Installation not found",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        with pytest.raises(
            GitHubInstallationTokenError,
            match="Installation not found",
        ):
            await provider.get_token(987654)


@pytest.mark.anyio
async def test_provider_rejects_invalid_response() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=201,
            json={
                "expires_at": (
                    FIXED_NOW
                    + timedelta(hours=1)
                ).isoformat(),
                "repository_selection": "all",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        with pytest.raises(
            GitHubInstallationTokenError,
            match="invalid installation-token response",
        ):
            await provider.get_token(987654)


@pytest.mark.anyio
async def test_provider_reports_transport_failure() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ConnectError(
            "Connection failed",
            request=request,
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        with pytest.raises(
            GitHubInstallationTokenError,
            match="Unable to reach GitHub",
        ):
            await provider.get_token(987654)


@pytest.mark.anyio
async def test_installation_token_is_masked() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=201,
            json=token_response(),
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as client:
        provider = GitHubInstallationTokenProvider(
            settings=create_settings(),
            client=client,
            clock=lambda: FIXED_NOW,
            app_jwt_factory=fake_app_jwt_factory,
        )

        token = await provider.get_token(987654)

    raw_token = token.token.get_secret_value()

    representation = repr(token)
    serialized = token.model_dump_json()

    assert raw_token not in representation
    assert raw_token not in serialized
    assert "**********" in representation
    assert "**********" in serialized