"""Application configuration for OpenSteward."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""

    return Settings()