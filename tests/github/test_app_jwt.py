"""Tests for GitHub App JWT generation."""

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from opensteward.github import (
    GitHubAppSettings,
    GitHubConfigurationError,
    GitHubJwtGenerationError,
    generate_github_app_jwt,
)


FIXED_NOW = datetime(
    2026,
    7,
    23,
    12,
    0,
    0,
    tzinfo=UTC,
)


@pytest.fixture(scope="module")
def rsa_key_pair() -> tuple[str, bytes]:
    """Create an ephemeral RSA key pair for JWT tests."""

    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )

    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    public_pem = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return private_pem.decode("utf-8"), public_pem


def create_settings(
    private_key: str,
) -> GitHubAppSettings:
    """Create configured settings for JWT tests."""

    return GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=private_key,
    )


def test_generated_jwt_has_valid_signature_and_claims(
    rsa_key_pair: tuple[str, bytes],
) -> None:
    private_key, public_key = rsa_key_pair

    generated = generate_github_app_jwt(
        create_settings(private_key),
        now=FIXED_NOW,
    )

    raw_token = generated.token.get_secret_value()

    claims = jwt.decode(
        raw_token,
        public_key,
        algorithms=["RS256"],
        issuer="123456",
        options={
            "verify_exp": False,
            "verify_iat": False,
        },
    )

    assert claims == {
        "iat": int(
            (
                FIXED_NOW
                - timedelta(seconds=60)
            ).timestamp()
        ),
        "exp": int(
            (
                FIXED_NOW
                + timedelta(minutes=9)
            ).timestamp()
        ),
        "iss": "123456",
    }


def test_generated_jwt_uses_rs256_header(
    rsa_key_pair: tuple[str, bytes],
) -> None:
    private_key, _ = rsa_key_pair

    generated = generate_github_app_jwt(
        create_settings(private_key),
        now=FIXED_NOW,
    )

    header = jwt.get_unverified_header(
        generated.token.get_secret_value()
    )

    assert header["alg"] == "RS256"
    assert header["typ"] == "JWT"
    assert generated.algorithm == "RS256"


def test_generated_jwt_contains_timing_metadata(
    rsa_key_pair: tuple[str, bytes],
) -> None:
    private_key, _ = rsa_key_pair

    generated = generate_github_app_jwt(
        create_settings(private_key),
        now=FIXED_NOW,
    )

    assert generated.issuer == "123456"

    assert generated.issued_at == (
        FIXED_NOW
        - timedelta(seconds=60)
    )

    assert generated.expires_at == (
        FIXED_NOW
        + timedelta(minutes=9)
    )


def test_generated_jwt_builds_bearer_header(
    rsa_key_pair: tuple[str, bytes],
) -> None:
    private_key, _ = rsa_key_pair

    generated = generate_github_app_jwt(
        create_settings(private_key),
        now=FIXED_NOW,
    )

    raw_token = generated.token.get_secret_value()

    assert (
        generated.authorization_header_value()
        == f"Bearer {raw_token}"
    )


def test_generated_jwt_is_masked_in_representation(
    rsa_key_pair: tuple[str, bytes],
) -> None:
    private_key, _ = rsa_key_pair

    generated = generate_github_app_jwt(
        create_settings(private_key),
        now=FIXED_NOW,
    )

    raw_token = generated.token.get_secret_value()

    representation = repr(generated)
    serialized = generated.model_dump_json()

    assert raw_token not in representation
    assert raw_token not in serialized
    assert "**********" in representation
    assert "**********" in serialized


def test_generation_rejects_naive_datetime(
    rsa_key_pair: tuple[str, bytes],
) -> None:
    private_key, _ = rsa_key_pair

    naive_time = datetime(
        2026,
        7,
        23,
        12,
        0,
        0,
    )

    with pytest.raises(
        GitHubJwtGenerationError,
        match="timezone-aware",
    ):
        generate_github_app_jwt(
            create_settings(private_key),
            now=naive_time,
        )


def test_generation_requires_configured_settings() -> None:
    settings = GitHubAppSettings(
        _env_file=None,
    )

    with pytest.raises(
        GitHubConfigurationError,
        match="not configured",
    ):
        generate_github_app_jwt(
            settings,
            now=FIXED_NOW,
        )


def test_generation_rejects_invalid_rsa_key() -> None:
    invalid_private_key = """-----BEGIN RSA PRIVATE KEY-----
not-a-real-rsa-private-key
-----END RSA PRIVATE KEY-----"""

    settings = GitHubAppSettings(
        _env_file=None,
        app_id=123456,
        private_key=invalid_private_key,
    )

    with pytest.raises(
        GitHubJwtGenerationError,
        match="Unable to sign",
    ):
        generate_github_app_jwt(
            settings,
            now=FIXED_NOW,
        )