"""Tests for semantic-search environment configuration."""

import pytest
from pydantic import SecretStr, ValidationError

from opensteward.knowledge.semantic_settings import (
    SemanticSettings,
)


def test_semantic_settings_defaults_are_resource_bounded() -> None:
    configured = SemanticSettings(_env_file=None)

    assert configured.semantic_enabled is False
    assert (
        configured.embedding_model
        == "BAAI/bge-small-en-v1.5"
    )
    assert configured.embedding_max_tokens == 512
    assert configured.embedding_batch_size == 4
    assert configured.semantic_max_documents == 100
    assert configured.groq_enabled is False
    assert configured.groq_max_candidates == 10


def test_enabled_groq_requires_semantics_and_secret() -> None:
    with pytest.raises(
        ValidationError,
        match="semantic scoring",
    ):
        SemanticSettings(
            _env_file=None,
            semantic_enabled=False,
            groq_enabled=True,
            groq_api_key=SecretStr("secret"),
        )

    with pytest.raises(
        ValidationError,
        match="OPENSTEWARD_GROQ_API_KEY",
    ):
        SemanticSettings(
            _env_file=None,
            semantic_enabled=True,
            groq_enabled=True,
        )


def test_enabled_groq_accepts_complete_configuration() -> None:
    configured = SemanticSettings(
        _env_file=None,
        semantic_enabled=True,
        groq_enabled=True,
        groq_api_key=SecretStr("secret"),
    )

    assert configured.groq_enabled is True
    assert configured.groq_api_key is not None
    assert (
        configured.groq_api_key.get_secret_value()
        == "secret"
    )
