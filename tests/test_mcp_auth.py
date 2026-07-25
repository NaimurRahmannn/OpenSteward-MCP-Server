"""Tests for MCP caller authentication and installation authorization."""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from mcp.server.auth.middleware.auth_context import auth_context_var
from mcp.server.auth.middleware.bearer_auth import AuthenticatedUser
from mcp.server.auth.provider import AccessToken
from pydantic import ValidationError

import opensteward.mcp.github_capabilities as github_capabilities
from opensteward.github import GitHubRepositoryRef
from opensteward.mcp.auth import (
    MCP_INSTALLATION_IDS_CLAIM,
    MCP_INVOKE_SCOPE,
    ConfiguredBearerTokenVerifier,
    MCPAuthorizationError,
    require_installation_access,
)
from opensteward.settings import Settings

TEST_TOKEN = "configured-test-token-with-32-characters"


def settings(
    *,
    token: str = TEST_TOKEN,
    installation_ids: list[int] | None = None,
) -> Settings:
    """Build isolated configured application settings."""

    return Settings(
        _env_file=None,
        mcp_authorized_callers={
            "test-client": {
                "token": token,
                "installation_ids": (
                    installation_ids
                    if installation_ids is not None
                    else [73]
                ),
            },
        },
    )


@contextmanager
def authenticated(
    installation_ids: list[int],
) -> Iterator[None]:
    """Install trusted MCP caller claims for one test block."""

    context_token = auth_context_var.set(
        AuthenticatedUser(
            AccessToken(
                token=TEST_TOKEN,
                client_id="test-client",
                subject="test-client",
                scopes=[MCP_INVOKE_SCOPE],
                claims={
                    MCP_INSTALLATION_IDS_CLAIM: installation_ids,
                },
            )
        )
    )

    try:
        yield
    finally:
        auth_context_var.reset(context_token)


@pytest.mark.anyio
async def test_configured_token_authenticates_caller() -> None:
    verifier = ConfiguredBearerTokenVerifier(
        settings_factory=settings,
    )

    access_token = await verifier.verify_token(TEST_TOKEN)

    assert access_token is not None
    assert access_token.client_id == "test-client"
    assert access_token.subject == "test-client"
    assert access_token.scopes == [MCP_INVOKE_SCOPE]
    assert access_token.claims == {
        MCP_INSTALLATION_IDS_CLAIM: [73],
    }


@pytest.mark.anyio
async def test_unknown_token_is_rejected() -> None:
    verifier = ConfiguredBearerTokenVerifier(
        settings_factory=settings,
    )

    assert await verifier.verify_token("unknown-token") is None


def test_installation_allowlist_accepts_authorized_installation() -> None:
    with authenticated([73]):
        require_installation_access(73)


def test_installation_allowlist_rejects_other_installation() -> None:
    with authenticated([74]), pytest.raises(
        MCPAuthorizationError,
        match="not authorized",
    ):
        require_installation_access(73)


def test_trusted_in_process_call_requires_no_transport_identity() -> None:
    require_installation_access(73)


@pytest.mark.anyio
async def test_unauthorized_tool_call_stops_before_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RunnerThatMustNotRun:
        async def assess(self, request: object) -> None:
            raise AssertionError(f"Runner received unauthorized request: {request}")

    monkeypatch.setattr(
        github_capabilities,
        "_assessment_runner",
        RunnerThatMustNotRun(),
    )

    with authenticated([74]), pytest.raises(
        MCPAuthorizationError,
        match="not authorized",
    ):
        await github_capabilities.assess_pull_request(
            installation_id=73,
            repository=GitHubRepositoryRef(
                owner="acme",
                name="framework",
            ),
            pull_number=17,
        )


def test_production_rejects_missing_caller_configuration() -> None:
    with pytest.raises(
        ValidationError,
        match="Production requires",
    ):
        Settings(
            _env_file=None,
            environment="production",
        )


def test_duplicate_caller_tokens_are_rejected() -> None:
    with pytest.raises(
        ValidationError,
        match="must be unique",
    ):
        Settings(
            _env_file=None,
            mcp_authorized_callers={
                "first": {
                    "token": TEST_TOKEN,
                    "installation_ids": [73],
                },
                "second": {
                    "token": TEST_TOKEN,
                    "installation_ids": [74],
                },
            },
        )


def test_bearer_token_is_redacted_from_settings_repr() -> None:
    representation = repr(settings())

    assert TEST_TOKEN not in representation
    assert "**********" in representation
