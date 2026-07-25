"""Tests for bounded application dependency readiness checks."""

from collections.abc import Callable

import httpx
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from opensteward.github import GitHubAppSettings
from opensteward.readiness import (
    GITHUB_READINESS_REQUIRED_PERMISSIONS,
    GitHubReadinessProbe,
    ReadinessCheckStatus,
    ReadinessService,
    ReadinessStatus,
)
from opensteward.settings import Settings

TEST_TOKEN = "readiness-test-token-with-32-characters"


@pytest.fixture(scope="module")
def private_key() -> str:
    """Create one valid ephemeral GitHub App signing key."""

    generated = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    private_pem = generated.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return private_pem.decode("ascii")


def app_settings(
    private_key: str,
) -> GitHubAppSettings:
    return GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=private_key,
        api_url="https://api.github.test",
        request_timeout_seconds=5,
    )


def server_settings(
    *,
    installation_ids: list[int] | None = None,
) -> Settings:
    return Settings(
        _env_file=None,
        mcp_authorized_callers={
            "inspector": {
                "token": TEST_TOKEN,
                "installation_ids": (
                    installation_ids
                    if installation_ids is not None
                    else [73]
                ),
            },
        },
    )


def client_factory(
    handler: Callable[[httpx.Request], httpx.Response],
):
    def create_client(
        **kwargs: object,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            **kwargs,
        )

    return create_client


def installation_payload(
    installation_id: int,
    *,
    permissions: dict[str, str] | None = None,
) -> dict[str, object]:
    return {
        "id": installation_id,
        "permissions": (
            permissions
            if permissions is not None
            else {
                permission: "read"
                for permission in GITHUB_READINESS_REQUIRED_PERMISSIONS
            }
        ),
    }


@pytest.mark.asyncio
async def test_ready_when_app_and_installations_are_usable(
    private_key: str,
) -> None:
    requested_paths: list[str] = []

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        requested_paths.append(request.url.path)
        assert request.headers["authorization"].startswith(
            "Bearer "
        )
        assert (
            request.headers["x-github-api-version"]
            == "2026-03-10"
        )

        if request.url.path == "/app":
            return httpx.Response(
                200,
                json={"id": 123456},
            )

        return httpx.Response(
            200,
            json=installation_payload(73),
        )

    service = ReadinessService(
        settings=server_settings(),
        github_probe=GitHubReadinessProbe(
            settings_factory=lambda: app_settings(private_key),
            http_client_factory=client_factory(handler),
        ),
        cache_ttl_seconds=0,
    )

    result = await service.assess()

    assert result.status == ReadinessStatus.READY
    assert result.issues == []
    assert result.checks.model_dump(mode="json") == {
        "mcp": "ready",
        "mcp_authentication": "ready",
        "github_credentials": "ready",
        "github_api": "ready",
        "github_installations": "ready",
    }
    assert requested_paths == [
        "/app",
        "/app/installations/73",
    ]


@pytest.mark.asyncio
async def test_unconfigured_dependencies_are_not_ready() -> None:
    service = ReadinessService(
        settings=Settings(
            _env_file=None,
        ),
        github_probe=GitHubReadinessProbe(
            settings_factory=lambda: GitHubAppSettings(
                _env_file=None,
            ),
        ),
        cache_ttl_seconds=0,
    )

    result = await service.assess()

    assert result.status == ReadinessStatus.NOT_READY
    assert (
        result.checks.mcp_authentication
        == ReadinessCheckStatus.NOT_READY
    )
    assert (
        result.checks.github_credentials
        == ReadinessCheckStatus.NOT_READY
    )
    assert result.checks.github_api == ReadinessCheckStatus.NOT_CHECKED
    assert (
        result.checks.github_installations
        == ReadinessCheckStatus.NOT_CHECKED
    )
    assert result.issues == [
        "MCP caller authentication is not configured.",
        "GitHub App authentication is not configured.",
    ]


@pytest.mark.asyncio
async def test_invalid_private_key_stops_before_network() -> None:
    invalid_settings = GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=(
            "-----BEGIN RSA PRIVATE KEY-----\n"
            "invalid\n"
            "-----END RSA PRIVATE KEY-----"
        ),
        api_url="https://api.github.test",
    )

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise AssertionError(
            f"Unexpected readiness request: {request.url}"
        )

    service = ReadinessService(
        settings=server_settings(),
        github_probe=GitHubReadinessProbe(
            settings_factory=lambda: invalid_settings,
            http_client_factory=client_factory(handler),
        ),
        cache_ttl_seconds=0,
    )

    result = await service.assess()

    assert result.status == ReadinessStatus.NOT_READY
    assert (
        result.checks.github_credentials
        == ReadinessCheckStatus.NOT_READY
    )
    assert result.checks.github_api == ReadinessCheckStatus.NOT_CHECKED
    assert result.issues == [
        "GitHub App credentials or private key are invalid.",
    ]


@pytest.mark.asyncio
async def test_github_api_authentication_failure_is_sanitized(
    private_key: str,
) -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            401,
            json={
                "message": "sensitive upstream detail",
            },
        )

    service = ReadinessService(
        settings=server_settings(),
        github_probe=GitHubReadinessProbe(
            settings_factory=lambda: app_settings(private_key),
            http_client_factory=client_factory(handler),
        ),
        cache_ttl_seconds=0,
    )

    result = await service.assess()

    assert result.status == ReadinessStatus.NOT_READY
    assert result.checks.github_api == ReadinessCheckStatus.NOT_READY
    assert (
        result.checks.github_installations
        == ReadinessCheckStatus.NOT_CHECKED
    )
    assert result.issues == [
        "GitHub rejected the App authentication readiness check "
        "with status 401.",
    ]
    assert "sensitive upstream detail" not in str(result)


@pytest.mark.asyncio
async def test_github_network_failure_is_not_ready(
    private_key: str,
) -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        raise httpx.ConnectError(
            "offline",
            request=request,
        )

    result = await ReadinessService(
        settings=server_settings(),
        github_probe=GitHubReadinessProbe(
            settings_factory=lambda: app_settings(private_key),
            http_client_factory=client_factory(handler),
        ),
        cache_ttl_seconds=0,
    ).assess()

    assert result.status == ReadinessStatus.NOT_READY
    assert result.checks.github_api == ReadinessCheckStatus.NOT_READY
    assert result.issues == [
        "GitHub API connectivity check failed.",
    ]


@pytest.mark.asyncio
async def test_missing_installation_permission_is_not_ready(
    private_key: str,
) -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        if request.url.path == "/app":
            return httpx.Response(
                200,
                json={"id": 123456},
            )

        return httpx.Response(
            200,
            json=installation_payload(
                73,
                permissions={
                    "contents": "read",
                    "pull_requests": "read",
                    "issues": "read",
                },
            ),
        )

    result = await ReadinessService(
        settings=server_settings(),
        github_probe=GitHubReadinessProbe(
            settings_factory=lambda: app_settings(private_key),
            http_client_factory=client_factory(handler),
        ),
        cache_ttl_seconds=0,
    ).assess()

    assert result.status == ReadinessStatus.NOT_READY
    assert (
        result.checks.github_installations
        == ReadinessCheckStatus.NOT_READY
    )
    assert result.issues == [
        "GitHub installation 73 lacks required permissions: checks.",
    ]


@pytest.mark.asyncio
async def test_unavailable_installation_is_not_ready(
    private_key: str,
) -> None:
    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        if request.url.path == "/app":
            return httpx.Response(
                200,
                json={"id": 123456},
            )

        return httpx.Response(
            404,
            json={"message": "Not Found"},
        )

    result = await ReadinessService(
        settings=server_settings(),
        github_probe=GitHubReadinessProbe(
            settings_factory=lambda: app_settings(private_key),
            http_client_factory=client_factory(handler),
        ),
        cache_ttl_seconds=0,
    ).assess()

    assert result.status == ReadinessStatus.NOT_READY
    assert result.issues == [
        "GitHub installation 73 is unavailable with status 404.",
    ]


@pytest.mark.asyncio
async def test_readiness_deduplicates_installations_and_caches_probe(
    private_key: str,
) -> None:
    request_count = 0
    current_time = 10.0

    def clock() -> float:
        return current_time

    def handler(
        request: httpx.Request,
    ) -> httpx.Response:
        nonlocal request_count
        request_count += 1

        if request.url.path == "/app":
            return httpx.Response(
                200,
                json={"id": 123456},
            )

        installation_id = int(
            request.url.path.rsplit("/", 1)[-1]
        )
        return httpx.Response(
            200,
            json=installation_payload(installation_id),
        )

    settings = Settings(
        _env_file=None,
        mcp_authorized_callers={
            "first": {
                "token": TEST_TOKEN,
                "installation_ids": [73, 74],
            },
            "second": {
                "token": "second-readiness-token-with-32-characters",
                "installation_ids": [73],
            },
        },
    )
    service = ReadinessService(
        settings=settings,
        github_probe=GitHubReadinessProbe(
            settings_factory=lambda: app_settings(private_key),
            http_client_factory=client_factory(handler),
        ),
        cache_ttl_seconds=30,
        clock=clock,
    )

    first = await service.assess()
    second = await service.assess()

    assert first == second
    assert request_count == 3

    current_time = 41.0
    await service.assess()

    assert request_count == 6
