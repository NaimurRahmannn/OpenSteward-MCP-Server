"""Tests for GitHub App configuration."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from opensteward.github import (
    GitHubAppSettings,
    GitHubConfigurationError,
)


TEST_PRIVATE_KEY = """-----BEGIN RSA PRIVATE KEY-----
test-private-key-content
-----END RSA PRIVATE KEY-----"""


def test_github_settings_are_optional() -> None:
    settings = GitHubAppSettings(
        _env_file=None,
    )

    assert settings.configured is False
    assert settings.app_id is None
    assert settings.api_url == "https://api.github.com"
    assert settings.api_version == "2026-03-10"
    assert settings.user_agent == "OpenSteward/0.1.0"


def test_inline_private_key_is_loaded() -> None:
    settings = GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=TEST_PRIVATE_KEY,
    )

    loaded_key = settings.load_private_key()

    assert settings.configured is True
    assert loaded_key.get_secret_value() == TEST_PRIVATE_KEY


def test_escaped_newlines_are_normalized() -> None:
    escaped_key = TEST_PRIVATE_KEY.replace("\n", "\\n")

    settings = GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=escaped_key,
    )

    loaded_key = settings.load_private_key()

    assert loaded_key.get_secret_value() == TEST_PRIVATE_KEY


def test_private_key_can_be_loaded_from_file(
    tmp_path: Path,
) -> None:
    key_path = tmp_path / "opensteward.pem"
    key_path.write_text(
        TEST_PRIVATE_KEY,
        encoding="utf-8",
    )

    settings = GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key_path=key_path,
    )

    loaded_key = settings.load_private_key()

    assert loaded_key.get_secret_value() == TEST_PRIVATE_KEY


def test_settings_reject_multiple_key_sources(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValidationError,
        match="not both",
    ):
        GitHubAppSettings(
            _env_file=None,
            app_id=123456,
            private_key=TEST_PRIVATE_KEY,
            private_key_path=tmp_path / "key.pem",
        )


def test_settings_reject_app_id_without_key() -> None:
    with pytest.raises(
        ValidationError,
        match="private key is required",
    ):
        GitHubAppSettings(
            _env_file=None,
            app_id=123456,
        )


def test_settings_reject_key_without_app_id() -> None:
    with pytest.raises(
        ValidationError,
        match="App ID is required",
    ):
        GitHubAppSettings(
            _env_file=None,
            private_key=TEST_PRIVATE_KEY,
        )


def test_unconfigured_settings_cannot_load_key() -> None:
    settings = GitHubAppSettings(
        _env_file=None,
    )

    with pytest.raises(
        GitHubConfigurationError,
        match="not configured",
    ):
        settings.load_private_key()


def test_missing_private_key_file_is_reported(
    tmp_path: Path,
) -> None:
    settings = GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key_path=tmp_path / "missing.pem",
    )

    with pytest.raises(
        GitHubConfigurationError,
        match="Unable to read",
    ):
        settings.load_private_key()


def test_invalid_private_key_format_is_rejected() -> None:
    settings = GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key="not-a-private-key",
    )

    with pytest.raises(
        GitHubConfigurationError,
        match="valid PEM",
    ):
        settings.load_private_key()


def test_private_key_is_masked_in_repr() -> None:
    settings = GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=TEST_PRIVATE_KEY,
    )

    representation = repr(settings)

    assert TEST_PRIVATE_KEY not in representation
    assert "**********" in representation


def test_api_url_is_normalized() -> None:
    settings = GitHubAppSettings(
        _env_file=None,
        api_url="https://api.github.com/",
    )

    assert settings.api_url == "https://api.github.com"


@pytest.mark.parametrize(
    "api_url",
    [
        "http://api.github.com",
        "api.github.com",
        "",
    ],
)
def test_invalid_api_urls_are_rejected(
    api_url: str,
) -> None:
    with pytest.raises(ValidationError):
        GitHubAppSettings(
            _env_file=None,
            api_url=api_url,
        )


@pytest.mark.parametrize(
    "api_version",
    [
        "2026",
        "2026-3-10",
        "v2026-03-10",
        "",
    ],
)
def test_invalid_api_versions_are_rejected(
    api_version: str,
) -> None:
    with pytest.raises(ValidationError):
        GitHubAppSettings(
            _env_file=None,
            api_version=api_version,
        )