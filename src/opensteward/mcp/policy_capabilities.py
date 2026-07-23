"""MCP capabilities for repository policy inspection and evaluation."""

import json

from pydantic import BaseModel, ConfigDict, Field

from opensteward.policy import (
    ContributionPolicyInput,
    MaintainerPolicyPacket,
    PolicyEvaluationResult,
    PolicySource,
    build_maintainer_policy_packet,
    evaluate_contribution_policy,
    load_repository_policy_with_metadata,
    parse_repository_policy_with_metadata,
)


class RepositoryPolicyEvaluationResponse(BaseModel):
    """MCP response containing policy provenance and evaluation results."""

    model_config = ConfigDict(
        extra="forbid",
    )

    policy_source: PolicySource
    source_reference: str = Field(min_length=1)
    used_defaults: bool
    policy_version: int = Field(ge=1)
    packet: MaintainerPolicyPacket
    evaluation: PolicyEvaluationResult

def evaluate_repository_policy(
    contribution: ContributionPolicyInput,
    policy_yaml: str | None = None,
) -> RepositoryPolicyEvaluationResponse:
    """Evaluate contribution facts against repository policy.

    When ``policy_yaml`` is provided, the supplied policy is evaluated.
    Otherwise, OpenSteward loads ``.opensteward.yml`` from the current
    repository and uses safe defaults when that file does not exist.

    This tool only evaluates and reports policy findings. It does not
    modify, reject, close, label, or approve a contribution.
    """

    if policy_yaml is None:
        loaded_policy = load_repository_policy_with_metadata()
    else:
        loaded_policy = parse_repository_policy_with_metadata(
            policy_yaml,
            source="mcp:policy_yaml",
        )

    evaluation = evaluate_contribution_policy(
        policy=loaded_policy.policy,
        contribution=contribution,
    )

    packet = build_maintainer_policy_packet(evaluation)

    return RepositoryPolicyEvaluationResponse(
        policy_source=loaded_policy.source,
        source_reference=loaded_policy.source_reference,
        used_defaults=loaded_policy.used_defaults,
        policy_version=loaded_policy.policy.version,
        packet=packet,
        evaluation=evaluation,
    )


def repository_policy_resource() -> str:
    """Return the active repository policy and its source metadata."""

    loaded_policy = load_repository_policy_with_metadata()

    return json.dumps(
        loaded_policy.model_dump(mode="json"),
        indent=2,
    )