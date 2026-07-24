"""Tests for live GitHub related-work runtime wiring."""

from typing import Any

import pytest
from pydantic import SecretStr

import opensteward.github.runtime as runtime_module
from opensteward.github import (
    GitHubAppSettings,
    GitHubConfigurationError,
    GitHubPermissionLevel,
    GitHubRelatedWorkQuery,
    GitHubRelatedWorkRequest,
    GitHubRepositoryRef,
    LiveGitHubRelatedWorkRunner,
)

CONFIGURATION_MESSAGE = (
    "GitHub App authentication is not configured. "
    "Set OPENSTEWARD_GITHUB_APP_ID and either "
    "OPENSTEWARD_GITHUB_PRIVATE_KEY or "
    "OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH."
)


def request() -> GitHubRelatedWorkRequest:
    """Build one valid runtime request."""

    return GitHubRelatedWorkRequest(
        installation_id=29,
        repository=GitHubRepositoryRef(owner="acme", name="framework"),
        git_ref="main",
        query=GitHubRelatedWorkQuery(text="parser"),
    )


def configured_settings() -> GitHubAppSettings:
    """Build configured settings without reading the environment."""

    return GitHubAppSettings.model_construct(
        app_id=123,
        private_key=SecretStr("configured-for-construction-only"),
        private_key_path=None,
        api_url="https://api.github.com",
        api_version="2026-03-10",
        user_agent="OpenSteward/0.1.0",
        request_timeout_seconds=15.0,
    )


def unconfigured_settings() -> GitHubAppSettings:
    """Build explicitly unconfigured settings without reading the environment."""

    return GitHubAppSettings.model_construct(
        app_id=None,
        private_key=None,
        private_key_path=None,
        api_url="https://api.github.com",
        api_version="2026-03-10",
        user_agent="OpenSteward/0.1.0",
        request_timeout_seconds=15.0,
    )


class FakeAsyncClient:
    """Record async HTTP-client construction and context closure."""

    instances: list["FakeAsyncClient"] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.entered = False
        self.closed = False
        self.__class__.instances.append(self)

    async def __aenter__(self) -> "FakeAsyncClient":
        self.entered = True
        return self

    async def __aexit__(self, *args: object) -> None:
        self.closed = True


class FakeGitHubRelatedWorkService:
    """Record final service construction and delegation."""

    instances: list["FakeGitHubRelatedWorkService"] = []
    outcome: object = object()
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[GitHubRelatedWorkRequest] = []
        self.__class__.instances.append(self)

    async def find(self, selected_request: GitHubRelatedWorkRequest) -> object:
        self.calls.append(selected_request)
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.outcome


def install_runtime_doubles(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, object]]:
    """Install recording constructors for every live dependency."""

    records: dict[str, list[dict[str, Any]]] = {
        "token": [],
        "rest": [],
        "historical": [],
        "paths": [],
        "adrs": [],
        "snapshot": [],
        "knowledge": [],
    }
    FakeAsyncClient.instances = []
    FakeGitHubRelatedWorkService.instances = []
    FakeGitHubRelatedWorkService.outcome = object()
    FakeGitHubRelatedWorkService.error = None

    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", FakeAsyncClient)

    def constructor(name: str, value: object):
        def construct(**kwargs: Any) -> object:
            records[name].append(kwargs)
            return value

        return construct

    token_provider = object()
    rest_client = object()
    historical = object()
    paths = object()
    adrs = object()
    snapshot = object()
    knowledge = object()
    objects = {
        "token_provider": token_provider,
        "rest_client": rest_client,
        "historical": historical,
        "paths": paths,
        "adrs": adrs,
        "snapshot": snapshot,
        "knowledge": knowledge,
    }
    monkeypatch.setattr(
        runtime_module,
        "GitHubInstallationTokenProvider",
        constructor("token", token_provider),
    )
    monkeypatch.setattr(
        runtime_module,
        "GitHubRestClient",
        constructor("rest", rest_client),
    )
    monkeypatch.setattr(
        runtime_module,
        "GitHubHistoricalKnowledgeCollector",
        constructor("historical", historical),
    )
    monkeypatch.setattr(
        runtime_module,
        "GitHubHistoricalPullRequestPathEnricher",
        constructor("paths", paths),
    )
    monkeypatch.setattr(
        runtime_module,
        "GitHubHistoricalAdrCollector",
        constructor("adrs", adrs),
    )
    monkeypatch.setattr(
        runtime_module,
        "GitHubHistoricalKnowledgeSnapshotService",
        constructor("snapshot", snapshot),
    )
    monkeypatch.setattr(
        runtime_module,
        "KnowledgeRelatedWorkService",
        constructor("knowledge", knowledge),
    )
    monkeypatch.setattr(
        runtime_module,
        "GitHubRelatedWorkService",
        FakeGitHubRelatedWorkService,
    )
    return records, objects


@pytest.mark.asyncio
async def test_missing_configuration_uses_existing_error_and_builds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_client(**kwargs: Any) -> object:
        raise AssertionError(f"HTTP client must not be constructed: {kwargs}")

    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", forbidden_client)
    runner = LiveGitHubRelatedWorkRunner(
        settings_factory=unconfigured_settings
    )

    with pytest.raises(GitHubConfigurationError, match=CONFIGURATION_MESSAGE):
        await runner.find(request())


@pytest.mark.asyncio
async def test_configured_runtime_builds_exact_read_only_graph_and_delegates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, objects = install_runtime_doubles(monkeypatch)
    selected_request = request()
    runner = LiveGitHubRelatedWorkRunner(
        settings_factory=configured_settings
    )

    result = await runner.find(selected_request)

    assert result is FakeGitHubRelatedWorkService.outcome
    assert len(FakeAsyncClient.instances) == 1
    http_client = FakeAsyncClient.instances[0]
    assert http_client.kwargs == {"follow_redirects": False}
    assert http_client.entered is True
    assert http_client.closed is True

    assert len(records["token"]) == 1
    assert records["token"][0] == {
        "settings": configured_settings(),
        "client": http_client,
    }
    assert len(records["rest"]) == 1
    rest_kwargs = records["rest"][0]
    assert rest_kwargs["settings"] == configured_settings()
    assert rest_kwargs["token_provider"] is objects["token_provider"]
    assert rest_kwargs["client"] is http_client
    assert rest_kwargs["installation_id"] == selected_request.installation_id
    scope = rest_kwargs["token_scope"]
    assert scope.repositories == [selected_request.repository.name]
    assert scope.permissions == {
        "contents": GitHubPermissionLevel.READ,
        "issues": GitHubPermissionLevel.READ,
        "pull_requests": GitHubPermissionLevel.READ,
    }
    assert all(level != GitHubPermissionLevel.WRITE for level in scope.permissions.values())
    assert "checks" not in scope.permissions
    assert "actions" not in scope.permissions
    assert "metadata" not in scope.permissions

    rest_client = objects["rest_client"]
    assert len(records["historical"]) == 1
    assert records["historical"] == [{"rest_client": rest_client}]
    assert records["paths"] == [{"rest_client": rest_client}]
    assert records["adrs"] == [{"rest_client": rest_client}]
    assert records["snapshot"] == [
        {
            "historical_items_collector": objects["historical"],
            "path_enricher": objects["paths"],
            "adr_collector": objects["adrs"],
        }
    ]
    assert records["knowledge"] == [{}]

    service = FakeGitHubRelatedWorkService.instances[0]
    assert len(FakeGitHubRelatedWorkService.instances) == 1
    assert service.kwargs["snapshot_collector"] is objects["snapshot"]
    assert service.kwargs["related_work_finder"] is objects["knowledge"]
    assert service.calls == [selected_request]


class SentinelRuntimeError(RuntimeError):
    """Distinct final-service failure."""


@pytest.mark.asyncio
async def test_http_context_closes_when_delegated_service_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_runtime_doubles(monkeypatch)
    FakeGitHubRelatedWorkService.error = SentinelRuntimeError("failed")
    runner = LiveGitHubRelatedWorkRunner(
        settings_factory=configured_settings
    )

    with pytest.raises(SentinelRuntimeError, match="failed"):
        await runner.find(request())

    assert len(FakeGitHubRelatedWorkService.instances[0].calls) == 1
    assert FakeAsyncClient.instances[0].closed is True
