"""Tests for live GitHub review-cost runtime wiring."""

from typing import Any

import pytest
from pydantic import SecretStr

import opensteward.github.runtime as runtime_module
from opensteward.github import (
    GitHubAppSettings,
    GitHubConfigurationError,
    GitHubPermissionLevel,
    GitHubRepositoryRef,
    GitHubReviewCostRequest,
    LiveGitHubReviewCostRunner,
)

CONFIGURATION_MESSAGE = (
    "GitHub App authentication is not configured. "
    "Set OPENSTEWARD_GITHUB_APP_ID and either "
    "OPENSTEWARD_GITHUB_PRIVATE_KEY or "
    "OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH."
)


def request() -> GitHubReviewCostRequest:
    """Build one valid runtime request."""

    return GitHubReviewCostRequest(
        installation_id=29,
        repository=GitHubRepositoryRef(owner="acme", name="framework"),
        pull_number=17,
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
    """Record HTTP-client construction and context closure."""

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


class FakeGitHubReviewCostService:
    """Record final service construction and delegation."""

    instances: list["FakeGitHubReviewCostService"] = []
    outcome: object = object()
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[GitHubReviewCostRequest] = []
        self.__class__.instances.append(self)

    async def assess(self, selected_request: GitHubReviewCostRequest) -> object:
        self.calls.append(selected_request)
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.outcome


def install_runtime_doubles(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, object]]:
    """Install recording constructors for the complete live dependency graph."""

    names = (
        "token",
        "rest",
        "pull_request",
        "repository",
        "assessment",
        "historical",
        "paths",
        "adrs",
        "snapshot",
        "knowledge",
        "related",
        "review_cost",
    )
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    FakeAsyncClient.instances = []
    FakeGitHubReviewCostService.instances = []
    FakeGitHubReviewCostService.outcome = object()
    FakeGitHubReviewCostService.error = None
    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", FakeAsyncClient)

    objects = {name: object() for name in names}

    def constructor(name: str):
        def construct(**kwargs: Any) -> object:
            records[name].append(kwargs)
            return objects[name]

        return construct

    constructors = {
        "GitHubInstallationTokenProvider": "token",
        "GitHubRestClient": "rest",
        "GitHubPullRequestService": "pull_request",
        "GitHubRepositoryService": "repository",
        "GitHubPullRequestAssessmentService": "assessment",
        "GitHubHistoricalKnowledgeCollector": "historical",
        "GitHubHistoricalPullRequestPathEnricher": "paths",
        "GitHubHistoricalAdrCollector": "adrs",
        "GitHubHistoricalKnowledgeSnapshotService": "snapshot",
        "KnowledgeRelatedWorkService": "knowledge",
        "GitHubRelatedWorkService": "related",
        "ReviewCostAssessmentService": "review_cost",
    }
    for attribute, name in constructors.items():
        monkeypatch.setattr(runtime_module, attribute, constructor(name))
    monkeypatch.setattr(
        runtime_module,
        "GitHubReviewCostService",
        FakeGitHubReviewCostService,
    )
    return records, objects


@pytest.mark.asyncio
async def test_missing_configuration_uses_existing_error_and_builds_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_client(**kwargs: Any) -> object:
        raise AssertionError(f"HTTP client must not be constructed: {kwargs}")

    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", forbidden_client)
    runner = LiveGitHubReviewCostRunner(settings_factory=unconfigured_settings)

    with pytest.raises(GitHubConfigurationError, match=CONFIGURATION_MESSAGE):
        await runner.assess(request())


@pytest.mark.asyncio
async def test_runtime_builds_one_shared_read_only_graph_and_delegates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, objects = install_runtime_doubles(monkeypatch)
    selected_request = request()
    runner = LiveGitHubReviewCostRunner(settings_factory=configured_settings)

    result = await runner.assess(selected_request)

    assert result is FakeGitHubReviewCostService.outcome
    assert len(FakeAsyncClient.instances) == 1
    http_client = FakeAsyncClient.instances[0]
    assert http_client.kwargs == {"follow_redirects": False}
    assert http_client.entered is True
    assert http_client.closed is True

    assert records["token"] == [
        {"settings": configured_settings(), "client": http_client}
    ]
    rest_kwargs = records["rest"][0]
    assert len(records["rest"]) == 1
    assert rest_kwargs["settings"] == configured_settings()
    assert rest_kwargs["token_provider"] is objects["token"]
    assert rest_kwargs["client"] is http_client
    assert rest_kwargs["installation_id"] == selected_request.installation_id
    scope = rest_kwargs["token_scope"]
    assert scope.repositories == [selected_request.repository.name]
    assert scope.permissions == {
        "contents": GitHubPermissionLevel.READ,
        "pull_requests": GitHubPermissionLevel.READ,
        "checks": GitHubPermissionLevel.READ,
        "issues": GitHubPermissionLevel.READ,
    }
    assert all(
        level != GitHubPermissionLevel.WRITE
        for level in scope.permissions.values()
    )

    rest_client = objects["rest"]
    for name in ("pull_request", "repository", "historical", "paths", "adrs"):
        assert records[name] == [{"rest_client": rest_client}]
    assert records["assessment"] == [
        {
            "pull_request_loader": objects["pull_request"],
            "policy_loader": objects["repository"],
        }
    ]
    assert records["snapshot"] == [
        {
            "historical_items_collector": objects["historical"],
            "path_enricher": objects["paths"],
            "adr_collector": objects["adrs"],
        }
    ]
    assert records["knowledge"] == [{}]
    assert records["related"] == [
        {
            "snapshot_collector": objects["snapshot"],
            "related_work_finder": objects["knowledge"],
        }
    ]
    assert records["review_cost"] == [{}]

    final_service = FakeGitHubReviewCostService.instances[0]
    assert len(FakeGitHubReviewCostService.instances) == 1
    assert final_service.kwargs == {
        "pull_request_assessor": objects["assessment"],
        "related_work_finder": objects["related"],
        "review_cost_assessor": objects["review_cost"],
    }
    assert final_service.calls == [selected_request]


class SentinelRuntimeError(RuntimeError):
    """Distinct final-service failure."""


@pytest.mark.asyncio
async def test_http_context_closes_when_delegated_service_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_runtime_doubles(monkeypatch)
    FakeGitHubReviewCostService.error = SentinelRuntimeError("failed")
    runner = LiveGitHubReviewCostRunner(settings_factory=configured_settings)

    with pytest.raises(SentinelRuntimeError, match="failed"):
        await runner.assess(request())

    assert len(FakeGitHubReviewCostService.instances[0].calls) == 1
    assert FakeAsyncClient.instances[0].closed is True
