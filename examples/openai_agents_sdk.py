"""Use OpenSteward from an OpenAI Agents SDK agent over Streamable HTTP."""

import asyncio
import os
from dataclasses import dataclass
from urllib.parse import urlparse

from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from agents.model_settings import ModelSettings


class ConfigurationError(ValueError):
    """Raised when the example configuration is missing or invalid."""


@dataclass(frozen=True, slots=True)
class Configuration:
    """Validated connection and pull-request settings."""

    mcp_url: str
    mcp_token: str
    installation_id: int
    repository_owner: str
    repository_name: str
    pull_number: int


def _required_text(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigurationError(f"{name} must be set.")
    return value


def _positive_integer(name: str, value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a positive integer.") from exc

    if parsed <= 0:
        raise ConfigurationError(f"{name} must be a positive integer.")
    return parsed


def _environment_or_prompt(name: str, prompt: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value

    try:
        value = input(prompt).strip()
    except EOFError as exc:
        raise ConfigurationError(
            f"{name} is not set and interactive input is unavailable."
        ) from exc

    if not value:
        raise ConfigurationError(f"{name} must not be empty.")
    return value


def _repository_segment(name: str, prompt: str) -> str:
    value = _environment_or_prompt(name, prompt)
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ConfigurationError(
            f"{name} must be one repository path segment without slashes."
        )
    return value


def load_configuration() -> Configuration:
    """Load configuration without reading the OpenAI API key directly."""

    mcp_url = _required_text("OPENSTEWARD_MCP_URL")
    parsed_url = urlparse(mcp_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ConfigurationError(
            "OPENSTEWARD_MCP_URL must be an absolute HTTP or HTTPS URL."
        )

    mcp_token = _required_text("OPENSTEWARD_MCP_TOKEN")
    if len(mcp_token) < 32 or any(character.isspace() for character in mcp_token):
        raise ConfigurationError(
            "OPENSTEWARD_MCP_TOKEN must contain at least 32 non-whitespace characters."
        )

    installation_id = _positive_integer(
        "OPENSTEWARD_INSTALLATION_ID",
        _required_text("OPENSTEWARD_INSTALLATION_ID"),
    )
    repository_owner = _repository_segment(
        "OPENSTEWARD_REPOSITORY_OWNER",
        "GitHub repository owner: ",
    )
    repository_name = _repository_segment(
        "OPENSTEWARD_REPOSITORY_NAME",
        "GitHub repository name: ",
    )
    pull_number = _positive_integer(
        "OPENSTEWARD_PULL_NUMBER",
        _environment_or_prompt(
            "OPENSTEWARD_PULL_NUMBER",
            "Pull-request number: ",
        ),
    )

    return Configuration(
        mcp_url=mcp_url,
        mcp_token=mcp_token,
        installation_id=installation_id,
        repository_owner=repository_owner,
        repository_name=repository_name,
        pull_number=pull_number,
    )


def analysis_prompt(configuration: Configuration) -> str:
    """Build the concrete task given to the agent."""

    return (
        "Analyze this pull request with OpenSteward. Call get_maintainer_brief "
        "with exactly this target:\n"
        f"- installation_id: {configuration.installation_id}\n"
        f"- repository.owner: {configuration.repository_owner}\n"
        f"- repository.name: {configuration.repository_name}\n"
        f"- pull_number: {configuration.pull_number}\n\n"
        "Base the answer only on the structured tool result. Explain readiness, "
        "policy evidence, related work, review-cost drivers, maintainer priority, "
        "specialist routes, and recommended actions. Surface every coverage or "
        "completeness warning and clearly distinguish unavailable evidence from "
        "negative evidence. Do not invent tool results, judge the contributor, "
        "or decide whether the pull request should merge."
    )


async def main() -> None:
    """Connect OpenSteward to an agent and print its final explanation."""

    configuration = load_configuration()

    async with MCPServerStreamableHttp(
        name="OpenSteward",
        params={
            "url": configuration.mcp_url,
            "headers": {
                "Authorization": f"Bearer {configuration.mcp_token}",
            },
            "timeout": 30,
        },
        cache_tools_list=True,
        client_session_timeout_seconds=180,
        use_structured_content=True,
        max_retry_attempts=3,
    ) as opensteward:
        agent = Agent(
            name="OpenSteward maintainer assistant",
            instructions=(
                "Use OpenSteward as the source of repository evidence. Prefer "
                "get_maintainer_brief for complete pull-request analysis. Explain "
                "coverage warnings instead of ignoring them. Never invent missing "
                "evidence. OpenSteward supports maintainer decisions but does not "
                "make merge decisions or judge contributor skill or trustworthiness."
            ),
            mcp_servers=[opensteward],
            model_settings=ModelSettings(tool_choice="required"),
        )

        result = await Runner.run(
            agent,
            analysis_prompt(configuration),
            max_turns=8,
        )

    print(result.final_output)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ConfigurationError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
