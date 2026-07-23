"""Built-in safe repository policy defaults."""

from opensteward.policy.models import RepositoryPolicy


DEFAULT_POLICY_REFERENCE = "OpenSteward built-in safe defaults"


def create_default_repository_policy() -> RepositoryPolicy:
    """Create a fresh instance of the built-in safe policy."""

    return RepositoryPolicy()