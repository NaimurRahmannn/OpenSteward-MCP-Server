"""Load and validate OpenSteward repository policy files"""

from pathlib import Path
from typing import Any
import yaml
from pydantic import ValidationError
from opensteward.policy.models import RepositoryPolicy

DEFAULT_POLICY_FILENAME=".opensteward.yml"

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

def parse_repository_policy(
    content: str,
    *,
    source: str = "<memory>",
) -> RepositoryPolicy:
    """Parse and validate repository policy YAML content.

    Empty content produces the safe default repository policy.

    Args:
        content: Raw YAML policy text.
        source: Human-readable source used in error messages.

    Returns:
        A validated repository policy.

    Raises:
        PolicyLoadError: When the YAML or policy configuration is invalid.
    """

    if not content.strip():
        return RepositoryPolicy()

    try:
        raw_policy: Any = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise PolicyLoadError(
            source=source,
            detail=f"invalid YAML syntax: {exc}",
        ) from exc

    if raw_policy is None:
        return RepositoryPolicy()

    if not isinstance(raw_policy, dict):
        raise PolicyLoadError(
            source=source,
            detail="the policy root must be a YAML mapping/object.",
        )

    try:
        return RepositoryPolicy.model_validate(raw_policy)
    except ValidationError as exc:
        raise PolicyLoadError(
            source=source,
            detail=f"policy validation failed:\n{exc}",
        ) from exc


def load_repository_policy(
    path: str | Path = DEFAULT_POLICY_FILENAME,
) -> RepositoryPolicy:
    """Load a repository policy from the local filesystem.

    A missing policy file produces the safe default policy.

    Args:
        path: Location of the repository policy file.

    Returns:
        A validated repository policy.

    Raises:
        PolicyLoadError: When an existing file cannot be read, parsed,
            or validated.
    """

    policy_path = Path(path)

    if not policy_path.exists():
        return RepositoryPolicy()

    try:
        content = policy_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyLoadError(
            source=str(policy_path),
            detail=f"unable to read the policy file: {exc}",
        ) from exc

    return parse_repository_policy(
        content,
        source=str(policy_path),
    )