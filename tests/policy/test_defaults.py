"""Tests for OpenSteward's built-in policy defaults."""

from opensteward.policy import (
    RepositoryPolicy,
    create_default_repository_policy,
)


def test_default_policy_provider_returns_repository_policy() -> None:
    policy = create_default_repository_policy()

    assert isinstance(policy, RepositoryPolicy)
    assert policy.version == 1
    assert policy.automation.require_human_approval is True


def test_default_policy_provider_returns_fresh_instances() -> None:
    first = create_default_repository_policy()
    second = create_default_repository_policy()

    assert first is not second
    assert first == second