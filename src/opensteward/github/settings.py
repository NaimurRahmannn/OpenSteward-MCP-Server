"""GitHub App configuration for OpenSteward."""

from functools import lru_cache
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from pydantic import (
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

from opensteward import __version__


PRIVATE_KEY_BOUNDARIES: tuple[tuple[str, str], ...] = (
    (
        "-----BEGIN RSA PRIVATE KEY-----",
        "-----END RSA PRIVATE KEY-----",
    ),
    (
        "-----BEGIN PRIVATE KEY-----",
        "-----END PRIVATE KEY-----",
    ),
)


class GitHubConfigurationError(ValueError):
    """Raised when GitHub App credentials cannot be loaded."""


class GitHubAppSettings(BaseSettings):
    """Configuration required for GitHub App authentication."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPENSTEWARD_GITHUB_",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    app_id: int | None = Field(
        default=None,
        gt=0,
    )

    private_key: SecretStr | None = None

    private_key_path: Path | None = None

    api_url: str = "https://api.github.com"

    api_version: str = "2026-03-10"

    user_agent: str = f"OpenSteward/{__version__}"

    request_timeout_seconds: float = Field(
        default=15,
        gt=0,
        le=120,
    )

    @field_validator("api_url")
    @classmethod
    def validate_api_url(
        cls,
        value: str,
    ) -> str:
        """Require a valid HTTPS GitHub API URL."""

        normalized = value.strip().rstrip("/")
        parsed = urlparse(normalized)

        if parsed.scheme != "https":
            raise ValueError(
                "GitHub API URL must use HTTPS."
            )

        if not parsed.netloc:
            raise ValueError(
                "GitHub API URL must contain a hostname."
            )

        return normalized

    @field_validator("api_version")
    @classmethod
    def validate_api_version(
        cls,
        value: str,
    ) -> str:
        """Require the YYYY-MM-DD GitHub API version format."""

        parts = value.split("-")

        if (
            len(parts) != 3
            or len(parts[0]) != 4
            or len(parts[1]) != 2
            or len(parts[2]) != 2
            or not all(part.isdigit() for part in parts)
        ):
            raise ValueError(
                "GitHub API version must use YYYY-MM-DD format."
            )

        return value

    @field_validator("user_agent")
    @classmethod
    def validate_user_agent(
        cls,
        value: str,
    ) -> str:
        """Require a non-empty GitHub User-Agent value."""

        normalized = value.strip()

        if not normalized:
            raise ValueError(
                "GitHub User-Agent must not be empty."
            )

        return normalized

    @model_validator(mode="after")
    def validate_credential_sources(self) -> Self:
        """Validate the GitHub App credential combination."""

        has_inline_key = self.private_key is not None
        has_key_path = self.private_key_path is not None
        has_key = has_inline_key or has_key_path

        if has_inline_key and has_key_path:
            raise ValueError(
                "Configure either private_key or private_key_path, "
                "not both."
            )

        if self.app_id is None and has_key:
            raise ValueError(
                "GitHub App ID is required when a private key is configured."
            )

        if self.app_id is not None and not has_key:
            raise ValueError(
                "A GitHub App private key is required when app_id "
                "is configured."
            )

        return self

    @property
    def configured(self) -> bool:
        """Return whether complete GitHub App credentials exist."""

        return (
            self.app_id is not None
            and (
                self.private_key is not None
                or self.private_key_path is not None
            )
        )

    def load_private_key(self) -> SecretStr:
        """Load and validate the configured GitHub App private key."""

        if not self.configured:
            raise GitHubConfigurationError(
                "GitHub App authentication is not configured."
            )

        if self.private_key is not None:
            raw_key = self.private_key.get_secret_value()
        else:
            assert self.private_key_path is not None

            try:
                raw_key = self.private_key_path.read_text(
                    encoding="utf-8",
                )
            except OSError as exc:
                raise GitHubConfigurationError(
                    "Unable to read the GitHub App private key from "
                    f"{self.private_key_path}: {exc}"
                ) from exc

        normalized_key = raw_key.replace("\\n", "\n").strip()

        valid_boundaries = any(
            normalized_key.startswith(begin)
            and normalized_key.endswith(end)
            for begin, end in PRIVATE_KEY_BOUNDARIES
        )

        if not valid_boundaries:
            raise GitHubConfigurationError(
                "GitHub App private key is not a valid PEM private key."
            )

        return SecretStr(normalized_key)


@lru_cache
def get_github_settings() -> GitHubAppSettings:
    """Return cached GitHub App settings."""

    return GitHubAppSettings()