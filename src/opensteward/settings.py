"""Application configuration for OpenSteward."""

from functools import cache
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

MCPInstallationId = Annotated[int, Field(gt=0)]


class MCPAuthorizedCaller(BaseModel):
    """One MCP caller credential and its GitHub installation allowlist."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    token: SecretStr

    installation_ids: frozenset[MCPInstallationId] = Field(
        min_length=1,
    )

    @field_validator("token")
    @classmethod
    def validate_token(cls, value: SecretStr) -> SecretStr:
        """Require a strong bearer token that can be parsed unambiguously."""

        raw_token = value.get_secret_value()

        if len(raw_token) < 32:
            raise ValueError(
                "MCP bearer tokens must contain at least 32 characters."
            )

        if (
            not raw_token.isascii()
            or raw_token != raw_token.strip()
            or any(character.isspace() for character in raw_token)
        ):
            raise ValueError(
                "MCP bearer tokens must contain non-whitespace ASCII characters."
            )

        return value


class Settings(BaseSettings):
    """Configuration loaded from environment variables and the .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPENSTEWARD_",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "OpenSteward"

    environment: Literal[
        "development",
        "test",
        "production",
    ] = "development"

    host: str = "127.0.0.1"

    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
    )

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = "INFO"

    mcp_authorized_callers: dict[str, MCPAuthorizedCaller] = Field(
        default_factory=dict,
    )

    @field_validator("mcp_authorized_callers")
    @classmethod
    def validate_caller_ids(
        cls,
        value: dict[str, MCPAuthorizedCaller],
    ) -> dict[str, MCPAuthorizedCaller]:
        """Require stable non-empty caller identifiers."""

        for caller_id in value:
            if not caller_id or caller_id != caller_id.strip():
                raise ValueError(
                    "MCP caller identifiers must be non-empty and trimmed."
                )

        return value

    @model_validator(mode="after")
    def validate_mcp_authentication(self) -> Self:
        """Reject ambiguous credentials and unsecured production startup."""

        token_owners: dict[str, str] = {}

        for caller_id, caller in self.mcp_authorized_callers.items():
            raw_token = caller.token.get_secret_value()
            existing_owner = token_owners.get(raw_token)

            if existing_owner is not None:
                raise ValueError(
                    "MCP bearer tokens must be unique across callers."
                )

            token_owners[raw_token] = caller_id

        if (
            self.environment == "production"
            and not self.mcp_authorized_callers
        ):
            raise ValueError(
                "Production requires at least one authorized MCP caller."
            )

        return self

    @property
    def mcp_authentication_configured(self) -> bool:
        """Return whether at least one MCP caller credential is configured."""

        return bool(self.mcp_authorized_callers)


@cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""

    return Settings()
