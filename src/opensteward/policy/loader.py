"""Load and validate OpenSteward repository policy files."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from opensteward.policy.defaults import (
    DEFAULT_POLICY_REFERENCE,
    create_default_repository_policy,
)
from opensteward.policy.models import (
    LoadedRepositoryPolicy,
    PolicySource,
    RepositoryPolicy,
)


DEFAULT_POLICY_FILENAME = ".opensteward.yml"


class PolicyLoadError(ValueError):
    """Raised when a repository policy cannot be parsed or validated."""

    def __init__(
        self,
        source: str,
        detail: str,
    ) -> None:
        self.source = source
        self.detail = detail

        super().__init__(
            f"Could not load OpenSteward policy from {source}: {detail}"
        )


def create_default_policy_result(
    *,
    source_reference: str = DEFAULT_POLICY_REFERENCE,
) -> LoadedRepositoryPolicy:
    """Create a policy result using OpenSteward's safe defaults."""

    return LoadedRepositoryPolicy(
        policy=create_default_repository_policy(),
        source=PolicySource.DEFAULT,
        source_reference=source_reference,
    )


def _parse_repository_policy_with_source(
    content: str,
    *,
    source_reference: str,
    successful_source: PolicySource,
) -> LoadedRepositoryPolicy:
    """Parse policy content using the supplied provenance information."""

    if not content.strip():
        return create_default_policy_result(
            source_reference=source_reference,
        )

    try:
        raw_policy: Any = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise PolicyLoadError(
            source=source_reference,
            detail=f"invalid YAML syntax: {exc}",
        ) from exc

    if raw_policy is None:
        return create_default_policy_result(
            source_reference=source_reference,
        )

    if not isinstance(raw_policy, dict):
        raise PolicyLoadError(
            source=source_reference,
            detail="the policy root must be a YAML mapping/object.",
        )

    try:
        policy = RepositoryPolicy.model_validate(raw_policy)
    except ValidationError as exc:
        raise PolicyLoadError(
            source=source_reference,
            detail=f"policy validation failed:\n{exc}",
        ) from exc

    return LoadedRepositoryPolicy(
        policy=policy,
        source=successful_source,
        source_reference=source_reference,
    )


def parse_repository_policy_with_metadata(
    content: str,
    *,
    source: str = "<memory>",
) -> LoadedRepositoryPolicy:
    """Parse YAML policy content and return policy provenance."""

    return _parse_repository_policy_with_source(
        content,
        source_reference=source,
        successful_source=PolicySource.MEMORY,
    )


def parse_repository_policy(
    content: str,
    *,
    source: str = "<memory>",
) -> RepositoryPolicy:
    """Parse YAML policy content and return only the validated policy."""

    return parse_repository_policy_with_metadata(
        content,
        source=source,
    ).policy


def load_repository_policy_with_metadata(
    path: str | Path = DEFAULT_POLICY_FILENAME,
) -> LoadedRepositoryPolicy:
    """Load a policy file and return policy provenance.

    A missing or empty file produces OpenSteward's safe default policy.
    """

    policy_path = Path(path)
    source_reference = str(policy_path)

    if not policy_path.exists():
        return create_default_policy_result(
            source_reference=source_reference,
        )

    try:
        content = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyLoadError(
            source=source_reference,
            detail=f"unable to read the policy file: {exc}",
        ) from exc

    return _parse_repository_policy_with_source(
        content,
        source_reference=source_reference,
        successful_source=PolicySource.REPOSITORY_FILE,
    )


def load_repository_policy(
    path: str | Path = DEFAULT_POLICY_FILENAME,
) -> RepositoryPolicy:
    """Load a policy file and return only the validated policy."""

    return load_repository_policy_with_metadata(path).policy