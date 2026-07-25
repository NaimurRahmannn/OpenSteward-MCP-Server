"""Authentication and GitHub installation authorization for MCP callers."""

from collections.abc import Callable
from secrets import compare_digest
from typing import Any

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.provider import AccessToken
from mcp.server.auth.settings import AuthSettings

from opensteward.settings import Settings, get_settings

MCP_AUTH_ISSUER_URL = "https://opensteward.local"
MCP_INVOKE_SCOPE = "opensteward:invoke"
MCP_INSTALLATION_IDS_CLAIM = "opensteward_installation_ids"


class MCPAuthorizationError(PermissionError):
    """Raised when an MCP caller lacks required authorization."""


def mcp_auth_settings() -> AuthSettings:
    """Build the MCP SDK policy for OpenSteward-issued opaque tokens."""

    return AuthSettings(
        issuer_url=MCP_AUTH_ISSUER_URL,
        resource_server_url=None,
        required_scopes=[MCP_INVOKE_SCOPE],
    )


class ConfiguredBearerTokenVerifier:
    """Verify opaque bearer tokens from application configuration."""

    def __init__(
        self,
        settings_factory: Callable[[], Settings] = get_settings,
    ) -> None:
        self._settings_factory = settings_factory

    async def verify_token(
        self,
        token: str,
    ) -> AccessToken | None:
        """Return caller identity and authorization claims for a valid token."""

        presented_token = token.encode("utf-8")

        for caller_id, caller in (
            self._settings_factory().mcp_authorized_callers.items()
        ):
            configured_token = (
                caller.token.get_secret_value().encode("utf-8")
            )

            if not compare_digest(
                presented_token,
                configured_token,
            ):
                continue

            return AccessToken(
                token=token,
                client_id=caller_id,
                subject=caller_id,
                scopes=[MCP_INVOKE_SCOPE],
                claims={
                    MCP_INSTALLATION_IDS_CLAIM: sorted(
                        caller.installation_ids
                    ),
                },
            )

        return None


def _authorized_installation_ids(
    claims: dict[str, Any] | None,
) -> frozenset[int]:
    """Extract and validate the trusted installation claim."""

    raw_installation_ids = (
        claims.get(MCP_INSTALLATION_IDS_CLAIM)
        if claims is not None
        else None
    )

    if (
        not isinstance(raw_installation_ids, list)
        or not raw_installation_ids
        or any(
            not isinstance(installation_id, int)
            or isinstance(installation_id, bool)
            or installation_id <= 0
            for installation_id in raw_installation_ids
        )
    ):
        raise MCPAuthorizationError(
            "Authenticated MCP caller has invalid authorization claims."
        )

    return frozenset(raw_installation_ids)


def require_installation_access(
    installation_id: int,
) -> None:
    """Require the current MCP caller to access one GitHub installation."""

    access_token = get_access_token()

    if access_token is None:
        # FastMCP rejects unauthenticated HTTP requests before tool execution.
        # A missing context therefore represents a trusted in-process call.
        return

    authorized_installation_ids = _authorized_installation_ids(
        access_token.claims
    )

    if installation_id not in authorized_installation_ids:
        raise MCPAuthorizationError(
            "Authenticated MCP caller is not authorized for the "
            "requested GitHub installation."
        )
