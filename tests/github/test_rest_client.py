"""Tests for the authenticated GitHub REST client."""

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from pydantic import (
    BaseModel,
    ConfigDict,
    SecretStr,
)

from opensteward.github import (
    GitHubAppSettings,
    GitHubInstallationToken,
    GitHubInstallationTokenScope,
    GitHubRepositorySelection,
    GitHubRestClient,
    GitHubRestResponseError,
    GitHubRestTransportError,
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


class RepositoryPayload(BaseModel):
    """Minimal typed repository response used by tests."""

    model_config = ConfigDict(
        extra="ignore",
    )

    id: int
    full_name: str
    private: bool


def create_settings() -> GitHubAppSettings:
    """Create GitHub settings for REST client tests."""

    return GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=(
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "test\n"
            "-----END RSA PRIVATE KEY-----"
        ),
        api_url="https://api.github.test",
        api_version="2026-03-10",
        user_agent="OpenSteward/Test",
        request_timeout_seconds=10,
    )


def create_token(
    value: str,
) -> GitHubInstallationToken:
    """Create a usable installation token."""

    return GitHubInstallationToken(
        token=SecretStr(value),
        installation_id=987654,
        expires_at=FIXED_NOW + timedelta(hours=1),
        permissions={},
        repository_selection=(
            GitHubRepositorySelection.ALL
        ),
        repositories=[],
    )


class FakeTokenProvider:
    """Controllable token provider used by REST tests."""

    def __init__(
        self,
        *,
        normal_token: str = "ghs_first_token",
        refreshed_token: str = "ghs_refreshed_token",
    ) -> None:
        self.normal_token = normal_token
        self.refreshed_token = refreshed_token

        self.get_calls: list[dict[str, Any]] = []
        self.invalidations: list[int] = []

    async def get_token(
        self,
        installation_id: int,
        *,
        scope: GitHubInstallationTokenScope | None = None,
        force_refresh: bool = False,
    ) -> GitHubInstallationToken:
        self.get_calls.append(
            {
                "installation_id": installation_id,
                "scope": scope,
                "force_refresh": force_refresh,
            }
        )

        value = (
            self.refreshed_token
            if force_refresh
            else self.normal_token
        )

        return create_token(value)

    def invalidate(
        self,
        installation_id: int,
        *,
        scope: GitHubInstallationTokenScope | None = None,
    ) -> None:
        self.invalidations.append(
            installation_id
        )


@pytest.fixture
def anyio_backend() -> str:
    """Run asynchronous tests with asyncio."""

    return "asyncio"


@pytest.mark.anyio
async def test_client_sends_authenticated_get_request() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        captured_requests.append(request)

        return httpx.Response(
            status_code=200,
            headers={
                "X-GitHub-Request-Id": "REQUEST-123",
            },
            json={
                "id": 1001,
                "full_name": "acme/framework",
                "private": True,
                "extra_field": "ignored",
            },
        )

    transport = httpx.MockTransport(handler)
    token_provider = FakeTokenProvider()

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=token_provider,
            client=http_client,
            installation_id=987654,
        )

        response = await client.get_json(
            "/repos/acme/framework",
            params={
                "ref": "main",
            },
            response_type=RepositoryPayload,
        )

    assert len(captured_requests) == 1

    request = captured_requests[0]

    assert request.method == "GET"

    assert (
        str(request.url)
        == (
            "https://api.github.test/repos/"
            "acme/framework?ref=main"
        )
    )

    assert (
        request.headers["authorization"]
        == "Bearer ghs_first_token"
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

    assert response.status_code == 200
    assert response.request_id == "REQUEST-123"
    assert response.data.id == 1001
    assert response.data.full_name == "acme/framework"
    assert response.data.private is True


@pytest.mark.anyio
async def test_client_returns_pagination_and_rate_limit_metadata() -> None:
    reset_epoch = int(
        (
            FIXED_NOW
            + timedelta(hours=1)
        ).timestamp()
    )

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            headers={
                "Link": (
                    '<https://api.github.test/repos/'
                    'acme/framework/pulls?page=2>; rel="next", '
                    '<https://api.github.test/repos/'
                    'acme/framework/pulls?page=4>; rel="last"'
                ),
                "X-RateLimit-Limit": "5000",
                "X-RateLimit-Remaining": "4998",
                "X-RateLimit-Used": "2",
                "X-RateLimit-Reset": str(
                    reset_epoch
                ),
                "X-RateLimit-Resource": "core",
                "X-GitHub-Request-Id": "REQUEST-456",
            },
            json=[
                {
                    "number": 42,
                }
            ],
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=FakeTokenProvider(),
            client=http_client,
            installation_id=987654,
        )

        response = await client.get_json(
            "/repos/acme/framework/pulls",
            response_type=list[dict[str, int]],
        )

    assert response.data == [
        {
            "number": 42,
        }
    ]

    assert response.pagination is not None

    assert (
        response.pagination.next
        == (
            "https://api.github.test/repos/"
            "acme/framework/pulls?page=2"
        )
    )

    assert (
        response.pagination.last
        == (
            "https://api.github.test/repos/"
            "acme/framework/pulls?page=4"
        )
    )

    assert response.rate_limit is not None
    assert response.rate_limit.limit == 5000
    assert response.rate_limit.remaining == 4998
    assert response.rate_limit.used == 2
    assert response.rate_limit.resource == "core"

    assert response.rate_limit.reset_at == (
        FIXED_NOW
        + timedelta(hours=1)
    )


@pytest.mark.anyio
async def test_client_refreshes_token_once_after_401() -> None:
    received_authorization: list[str] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        authorization = request.headers[
            "authorization"
        ]

        received_authorization.append(
            authorization
        )

        if authorization == "Bearer ghs_first_token":
            return httpx.Response(
                status_code=401,
                json={
                    "message": "Bad credentials",
                },
            )

        return httpx.Response(
            status_code=200,
            json={
                "id": 1001,
                "full_name": "acme/framework",
                "private": True,
            },
        )

    transport = httpx.MockTransport(handler)
    token_provider = FakeTokenProvider()

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=token_provider,
            client=http_client,
            installation_id=987654,
        )

        response = await client.get_json(
            "/repos/acme/framework",
            response_type=RepositoryPayload,
        )

    assert response.status_code == 200

    assert received_authorization == [
        "Bearer ghs_first_token",
        "Bearer ghs_refreshed_token",
    ]

    assert token_provider.invalidations == [
        987654,
    ]

    assert [
        call["force_refresh"]
        for call in token_provider.get_calls
    ] == [
        False,
        True,
    ]


@pytest.mark.anyio
async def test_client_does_not_retry_more_than_once() -> None:
    request_count = 0
    token_provider = FakeTokenProvider()

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            status_code=401,
            headers={
                "X-GitHub-Request-Id": "AUTH-FAILED",
            },
            json={
                "message": "Bad credentials",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=token_provider,
            client=http_client,
            installation_id=987654,
        )

        with pytest.raises(
            GitHubRestResponseError,
            match="Bad credentials",
        ) as error_info:
            await client.get_json(
                "/repos/acme/framework"
            )

    assert request_count == 2
    assert error_info.value.status_code == 401
    assert error_info.value.request_id == "AUTH-FAILED"


@pytest.mark.anyio
async def test_client_reports_structured_rate_limit_error() -> None:
    reset_epoch = int(
        (
            FIXED_NOW
            + timedelta(minutes=15)
        ).timestamp()
    )

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={
                "Retry-After": "30",
                "X-RateLimit-Limit": "5000",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Used": "5000",
                "X-RateLimit-Reset": str(
                    reset_epoch
                ),
                "X-GitHub-Request-Id": "RATE-123",
            },
            json={
                "message": (
                    "API rate limit exceeded"
                ),
                "documentation_url": (
                    "https://docs.github.com/"
                    "rest/using-the-rest-api/"
                    "rate-limits-for-the-rest-api"
                ),
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=FakeTokenProvider(),
            client=http_client,
            installation_id=987654,
        )

        with pytest.raises(
            GitHubRestResponseError,
            match="rate limit exceeded",
        ) as error_info:
            await client.get_json(
                "/repos/acme/framework"
            )

    error = error_info.value

    assert error.status_code == 429
    assert error.request_id == "RATE-123"
    assert error.rate_limited is True
    assert error.retry_after_seconds == 30
    assert error.rate_limit is not None
    assert error.rate_limit.remaining == 0
    assert error.rate_limit.reset_at == (
        FIXED_NOW
        + timedelta(minutes=15)
    )


@pytest.mark.anyio
async def test_client_does_not_retry_non_401_error() -> None:
    request_count = 0
    token_provider = FakeTokenProvider()

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        return httpx.Response(
            status_code=403,
            json={
                "message": "Resource not accessible by integration",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=token_provider,
            client=http_client,
            installation_id=987654,
        )

        with pytest.raises(
            GitHubRestResponseError,
            match="Resource not accessible",
        ):
            await client.get_json(
                "/repos/acme/framework"
            )

    assert request_count == 1
    assert len(token_provider.get_calls) == 1
    assert token_provider.invalidations == []


@pytest.mark.anyio
async def test_client_redacts_token_from_error_message() -> None:
    token_value = "ghs_sensitive_installation_token"

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=400,
            json={
                "message": (
                    "Invalid token: "
                    f"{token_value}"
                ),
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=FakeTokenProvider(
                normal_token=token_value,
            ),
            client=http_client,
            installation_id=987654,
        )

        with pytest.raises(
            GitHubRestResponseError,
        ) as error_info:
            await client.get_json(
                "/repos/acme/framework"
            )

    error_message = str(error_info.value)

    assert token_value not in error_message
    assert "[REDACTED]" in error_message


@pytest.mark.anyio
async def test_client_reports_invalid_json() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=b"not-json",
            headers={
                "Content-Type": "application/json",
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=FakeTokenProvider(),
            client=http_client,
            installation_id=987654,
        )

        with pytest.raises(
            GitHubRestResponseError,
            match="invalid JSON",
        ):
            await client.get_json(
                "/repos/acme/framework"
            )


@pytest.mark.anyio
async def test_client_reports_response_validation_failure() -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            json={
                "id": "not-an-integer",
                "private": True,
            },
        )

    transport = httpx.MockTransport(handler)

    async with httpx.AsyncClient(
        transport=transport,
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=FakeTokenProvider(),
            client=http_client,
            installation_id=987654,
        )

        with pytest.raises(
            GitHubRestResponseError,
            match="expected response type",
        ):
            await client.get_json(
                "/repos/acme/framework",
                response_type=RepositoryPayload,
            )


@pytest.mark.anyio
async def test_client_reports_transport_failure() -> None:
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
    ) as http_client:
        client = GitHubRestClient(
            settings=create_settings(),
            token_provider=FakeTokenProvider(),
            client=http_client,
            installation_id=987654,
        )

        with pytest.raises(
            GitHubRestTransportError,
            match="Unable to reach GitHub",
        ):
            await client.get_json(
                "/repos/acme/framework"
            )


@pytest.mark.parametrize(
    "path",
    [
        "repos/acme/framework",
        "//malicious.example/token",
        "https://malicious.example/token",
        "/repos/acme/../framework",
        "/repos/acme/framework?page=2",
        "/repos/acme/framework#details",
        "/repos\\acme\\framework",
    ],
)
def test_client_rejects_unsafe_paths(
    path: str,
) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            status_code=200,
            json={},
        )
    )

    token_provider = FakeTokenProvider()

    async def run_request() -> None:
        async with httpx.AsyncClient(
            transport=transport,
        ) as http_client:
            client = GitHubRestClient(
                settings=create_settings(),
                token_provider=token_provider,
                client=http_client,
                installation_id=987654,
            )

            await client.get_json(path)

    with pytest.raises(ValueError):
        import anyio

        anyio.run(run_request)

    assert token_provider.get_calls == []