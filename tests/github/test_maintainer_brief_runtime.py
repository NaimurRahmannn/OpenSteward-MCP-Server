"""Tests for live GitHub maintainer-brief runtime wiring."""

from typing import Any

import pytest
from pydantic import SecretStr

import opensteward.github.runtime as runtime_module
from opensteward.github import (
    GitHubAppSettings,
    GitHubConfigurationError,
    GitHubMaintainerBriefRequest,
    GitHubPermissionLevel,
    GitHubRepositoryRef,
    LiveGitHubMaintainerBriefRunner,
)

CONFIGURATION_MESSAGE = (
    "GitHub App authentication is not configured. "
    "Set OPENSTEWARD_GITHUB_APP_ID and either "
    "OPENSTEWARD_GITHUB_PRIVATE_KEY or "
    "OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH."
)


def request() -> GitHubMaintainerBriefRequest:
    return GitHubMaintainerBriefRequest(
        installation_id=29,
        repository=GitHubRepositoryRef(owner="acme", name="framework"),
        pull_number=17,
    )


def configured_settings() -> GitHubAppSettings:
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


class FakeGitHubMaintainerBriefService:
    instances: list["FakeGitHubMaintainerBriefService"] = []
    outcome: object = object()
    error: Exception | None = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[GitHubMaintainerBriefRequest] = []
        self.__class__.instances.append(self)

    async def build(self, selected: GitHubMaintainerBriefRequest) -> object:
        self.calls.append(selected)
        if self.__class__.error is not None:
            raise self.__class__.error
        return self.__class__.outcome


def install_doubles(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, object]]:
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
        "review_cost_assessor",
        "review_cost_service",
        "brief",
    )
    records: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    objects = {name: object() for name in names}
    FakeAsyncClient.instances = []
    FakeGitHubMaintainerBriefService.instances = []
    FakeGitHubMaintainerBriefService.outcome = object()
    FakeGitHubMaintainerBriefService.error = None
    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", FakeAsyncClient)

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
        "ReviewCostAssessmentService": "review_cost_assessor",
        "GitHubReviewCostService": "review_cost_service",
        "MaintainerBriefService": "brief",
    }
    for attribute, name in constructors.items():
        monkeypatch.setattr(runtime_module, attribute, constructor(name))
    monkeypatch.setattr(
        runtime_module,
        "GitHubMaintainerBriefService",
        FakeGitHubMaintainerBriefService,
    )
    return records, objects


@pytest.mark.asyncio
async def test_missing_configuration_builds_no_http_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_client(**kwargs: Any) -> object:
        raise AssertionError(f"HTTP client must not be constructed: {kwargs}")

    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", forbidden_client)
    runner = LiveGitHubMaintainerBriefRunner(
        settings_factory=unconfigured_settings
    )

    with pytest.raises(GitHubConfigurationError, match=CONFIGURATION_MESSAGE):
        await runner.build(request())


@pytest.mark.asyncio
async def test_runtime_builds_one_shared_read_only_graph_and_delegates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records, objects = install_doubles(monkeypatch)
    selected = request()

    result = await LiveGitHubMaintainerBriefRunner(
        settings_factory=configured_settings
    ).build(selected)

    assert result is FakeGitHubMaintainerBriefService.outcome
    assert len(FakeAsyncClient.instances) == 1
    client = FakeAsyncClient.instances[0]
    assert client.kwargs == {"follow_redirects": False}
    assert client.entered is True
    assert client.closed is True
    assert records["token"] == [
        {"settings": configured_settings(), "client": client}
    ]
    rest = records["rest"][0]
    assert len(records["rest"]) == 1
    assert rest["installation_id"] == 29
    assert rest["token_provider"] is objects["token"]
    assert rest["client"] is client
    scope = rest["token_scope"]
    assert scope.repositories == ["framework"]
    assert scope.permissions == {
        "contents": GitHubPermissionLevel.READ,
        "pull_requests": GitHubPermissionLevel.READ,
        "checks": GitHubPermissionLevel.READ,
        "issues": GitHubPermissionLevel.READ,
    }
    assert GitHubPermissionLevel.WRITE not in scope.permissions.values()

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
    assert records["review_cost_assessor"] == [{}]
    assert records["review_cost_service"] == [
        {
            "pull_request_assessor": objects["assessment"],
            "related_work_finder": objects["related"],
            "review_cost_assessor": objects["review_cost_assessor"],
        }
    ]
    assert records["brief"] == [{}]
    final = FakeGitHubMaintainerBriefService.instances[0]
    assert final.kwargs == {
        "review_cost_assessor": objects["review_cost_service"],
        "brief_builder": objects["brief"],
    }
    assert final.calls == [selected]


class SentinelRuntimeError(RuntimeError):
    """Distinct final-service failure."""


@pytest.mark.asyncio
async def test_http_client_closes_after_delegation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_doubles(monkeypatch)
    FakeGitHubMaintainerBriefService.error = SentinelRuntimeError("failed")

    with pytest.raises(SentinelRuntimeError, match="failed"):
        await LiveGitHubMaintainerBriefRunner(
            settings_factory=configured_settings
        ).build(request())

    assert len(FakeGitHubMaintainerBriefService.instances[0].calls) == 1
    assert FakeAsyncClient.instances[0].closed is True
