"""GitHub App JSON Web Token generation."""

from datetime import UTC, datetime, timedelta
from typing import Final, Literal

import jwt
from jwt.exceptions import PyJWTError
from pydantic import Field, SecretStr

from opensteward.github.models import StrictGitHubModel
from opensteward.github.settings import GitHubAppSettings


GITHUB_APP_JWT_ALGORITHM: Final = "RS256"

GITHUB_APP_JWT_CLOCK_SKEW: Final = timedelta(
    seconds=60,
)

GITHUB_APP_JWT_FUTURE_LIFETIME: Final = timedelta(
    minutes=9,
)


class GitHubJwtGenerationError(ValueError):
    """Raised when OpenSteward cannot create a GitHub App JWT."""


class GitHubAppJwt(StrictGitHubModel):
    """A short-lived JWT used to authenticate as a GitHub App."""

    token: SecretStr

    issuer: str = Field(
        min_length=1,
    )

    issued_at: datetime
    expires_at: datetime

    algorithm: Literal["RS256"] = "RS256"

    def authorization_header_value(self) -> str:
        """Return the value for an HTTP Authorization header.

        Callers must not log or persist the returned value.
        """

        return (
            "Bearer "
            f"{self.token.get_secret_value()}"
        )


def _normalize_current_time(
    value: datetime,
) -> datetime:
    """Normalize an aware datetime to whole UTC seconds."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise GitHubJwtGenerationError(
            "GitHub JWT generation requires a timezone-aware "
            "current time."
        )

    return (
        value.astimezone(UTC)
        .replace(microsecond=0)
    )


def generate_github_app_jwt(
    settings: GitHubAppSettings,
    *,
    now: datetime | None = None,
) -> GitHubAppJwt:
    """Create a short-lived RS256 JWT for a GitHub App.

    ``now`` is injectable so tests do not depend on the system clock.
    """

    current_time = _normalize_current_time(
        now or datetime.now(UTC)
    )

    private_key = (
        settings
        .load_private_key()
        .get_secret_value()
    )

    app_id = settings.app_id

    if app_id is None:
        raise GitHubJwtGenerationError(
            "GitHub App ID is required to generate a JWT."
        )

    issued_at = (
        current_time
        - GITHUB_APP_JWT_CLOCK_SKEW
    )

    expires_at = (
        current_time
        + GITHUB_APP_JWT_FUTURE_LIFETIME
    )

    claims = {
        "iat": int(issued_at.timestamp()),
        "exp": int(expires_at.timestamp()),
        "iss": str(app_id),
    }

    try:
        encoded_token = jwt.encode(
            payload=claims,
            key=private_key,
            algorithm=GITHUB_APP_JWT_ALGORITHM,
        )
    except (PyJWTError, TypeError, ValueError) as exc:
        raise GitHubJwtGenerationError(
            "Unable to sign the GitHub App JWT with the "
            "configured private key."
        ) from exc

    return GitHubAppJwt(
        token=SecretStr(encoded_token),
        issuer=str(app_id),
        issued_at=issued_at,
        expires_at=expires_at,
        algorithm=GITHUB_APP_JWT_ALGORITHM,
    )