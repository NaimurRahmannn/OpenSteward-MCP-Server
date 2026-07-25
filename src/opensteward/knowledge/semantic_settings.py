"""Configuration for optional local semantic scoring and Groq reranking."""

from functools import cache
from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SemanticSettings(BaseSettings):
    """Environment-backed semantic-search configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="OPENSTEWARD_",
        env_ignore_empty=True,
        case_sensitive=False,
        extra="ignore",
    )

    semantic_enabled: bool = False

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    embedding_max_tokens: int = Field(
        default=512,
        ge=64,
        le=8192,
    )

    embedding_batch_size: int = Field(
        default=4,
        ge=1,
        le=64,
    )

    embedding_threads: int = Field(
        default=2,
        ge=1,
        le=16,
    )

    embedding_cache_dir: Path | None = None

    semantic_max_documents: int = Field(
        default=100,
        ge=1,
        le=500,
    )

    groq_enabled: bool = False

    groq_model: str = "openai/gpt-oss-20b"

    groq_api_key: SecretStr | None = None

    groq_api_url: str = "https://api.groq.com/openai/v1"

    groq_max_candidates: int = Field(
        default=10,
        ge=1,
        le=25,
    )

    groq_request_timeout_seconds: float = Field(
        default=20,
        gt=0,
        le=120,
    )

    groq_max_input_characters: int = Field(
        default=24_000,
        ge=2_000,
        le=32_000,
    )

    @field_validator("embedding_model", "groq_model")
    @classmethod
    def normalize_model_name(cls, value: str) -> str:
        """Require non-empty model identifiers."""

        normalized = value.strip()

        if not normalized:
            raise ValueError("Semantic model names must not be empty.")

        return normalized

    @field_validator("groq_api_url")
    @classmethod
    def validate_groq_api_url(cls, value: str) -> str:
        """Require a normalized HTTPS Groq API base URL."""

        normalized = value.strip().rstrip("/")
        parsed = urlparse(normalized)

        if parsed.scheme != "https" or not parsed.netloc:
            raise ValueError(
                "Groq API URL must be an absolute HTTPS URL."
            )

        return normalized

    @model_validator(mode="after")
    def validate_enabled_services(self) -> Self:
        """Reject incomplete or contradictory semantic configuration."""

        if self.groq_enabled and not self.semantic_enabled:
            raise ValueError(
                "Groq reranking requires semantic scoring to be enabled."
            )

        if self.groq_enabled and self.groq_api_key is None:
            raise ValueError(
                "OPENSTEWARD_GROQ_API_KEY is required when Groq is enabled."
            )

        return self


@cache
def get_semantic_settings() -> SemanticSettings:
    """Return cached semantic-search settings."""

    return SemanticSettings()
