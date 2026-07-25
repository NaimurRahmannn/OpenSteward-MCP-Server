"""Live GitHub runtime wiring for read-only capabilities."""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass

import httpx

from opensteward.github.assessments import (
    GitHubPullRequestAssessmentRequest,
    GitHubPullRequestAssessmentResult,
    GitHubPullRequestAssessmentService,
)
from opensteward.github.historical_adrs import GitHubHistoricalAdrCollector
from opensteward.github.historical_knowledge import (
    GitHubHistoricalKnowledgeCollector,
)
from opensteward.github.historical_paths import (
    GitHubHistoricalPullRequestPathEnricher,
)
from opensteward.github.historical_snapshot import (
    GitHubHistoricalKnowledgeSnapshotService,
)
from opensteward.github.installation_tokens import (
    GitHubInstallationTokenProvider,
    GitHubInstallationTokenScope,
    GitHubPermissionLevel,
)
from opensteward.github.maintainer_brief import (
    GitHubMaintainerBriefRequest,
    GitHubMaintainerBriefResult,
    GitHubMaintainerBriefService,
)
from opensteward.github.pull_requests import (
    GitHubPullRequestService,
)
from opensteward.github.related_work import (
    GitHubRelatedWorkRequest,
    GitHubRelatedWorkResult,
    GitHubRelatedWorkService,
)
from opensteward.github.repositories import (
    GitHubRepositoryService,
)
from opensteward.github.rest_client import (
    GitHubRestClient,
)
from opensteward.github.review_cost import (
    GitHubReviewCostRequest,
    GitHubReviewCostResult,
    GitHubReviewCostService,
)
from opensteward.github.settings import (
    GitHubAppSettings,
    GitHubConfigurationError,
    get_github_settings,
)
from opensteward.knowledge import (
    KnowledgeRelatedWorkService,
    KnowledgeSemanticScoringService,
)
from opensteward.knowledge.semantic_providers import (
    build_semantic_scorer,
)
from opensteward.knowledge.semantic_settings import (
    SemanticSettings,
    get_semantic_settings,
)
from opensteward.maintainer_brief import MaintainerBriefService
from opensteward.review_intelligence import ReviewCostAssessmentService

SettingsFactory = Callable[
    [],
    GitHubAppSettings,
]
SemanticSettingsFactory = Callable[
    [],
    SemanticSettings,
]


def _default_disabled_semantic_settings() -> SemanticSettings:
    """Return defaults without reading a developer's local .env file."""

    return SemanticSettings(
        _env_file=None,
        semantic_enabled=False,
        groq_enabled=False,
    )


def _build_related_work_finder(
    settings: SemanticSettings,
    *,
    client: httpx.AsyncClient,
) -> KnowledgeRelatedWorkService:
    """Build deterministic or configured hybrid related-work search."""

    if not settings.semantic_enabled:
        return KnowledgeRelatedWorkService()

    scorer = build_semantic_scorer(
        settings,
        client=client,
    )
    semantic_service = KnowledgeSemanticScoringService(
        scorer=scorer,
        maximum_documents=(
            settings.semantic_max_documents
        ),
    )

    return KnowledgeRelatedWorkService(
        semantic_scoring_service=semantic_service
    )


@dataclass(frozen=True, slots=True)
class _GitHubRuntimeResources:
    """Reusable GitHub transport and installation-token provider."""

    settings: GitHubAppSettings
    http_client: httpx.AsyncClient
    token_provider: GitHubInstallationTokenProvider


class _SharedGitHubRuntime:
    """Lazily own reusable GitHub resources until application shutdown."""

    def __init__(self) -> None:
        self._resources: _GitHubRuntimeResources | None = None
        self._lock = asyncio.Lock()

    @asynccontextmanager
    async def use(
        self,
        settings: GitHubAppSettings,
    ) -> AsyncIterator[_GitHubRuntimeResources]:
        """Yield shared resources without ending their process lifetime."""

        async with self._lock:
            if self._resources is None:
                http_client = httpx.AsyncClient(
                    follow_redirects=False,
                )
                token_provider = GitHubInstallationTokenProvider(
                    settings=settings,
                    client=http_client,
                )
                self._resources = _GitHubRuntimeResources(
                    settings=settings,
                    http_client=http_client,
                    token_provider=token_provider,
                )
            elif self._resources.settings != settings:
                raise RuntimeError(
                    "GitHub runtime settings changed after initialization."
                )

            resources = self._resources

        yield resources

    async def aclose(self) -> None:
        """Clear cached credentials and close the shared HTTP transport."""

        async with self._lock:
            resources = self._resources
            self._resources = None

            if resources is None:
                return

            resources.token_provider.clear()
            await resources.http_client.aclose()


_shared_github_runtime = _SharedGitHubRuntime()


async def close_live_github_runtime() -> None:
    """Close process-level GitHub resources during application shutdown."""

    await _shared_github_runtime.aclose()


class _LiveGitHubRunner:
    """Provide one persistent GitHub runtime to a live capability runner."""

    def __init__(
        self,
        *,
        settings_factory: SettingsFactory = get_github_settings,
        semantic_settings_factory: (
            SemanticSettingsFactory | None
        ) = None,
    ) -> None:
        self._settings_factory = settings_factory
        self._semantic_settings_factory = (
            semantic_settings_factory
            if semantic_settings_factory is not None
            else (
                get_semantic_settings
                if settings_factory is get_github_settings
                else _default_disabled_semantic_settings
            )
        )
        self._github_runtime = (
            _shared_github_runtime
            if settings_factory is get_github_settings
            else _SharedGitHubRuntime()
        )

    async def aclose(self) -> None:
        """Close resources owned by this runner."""

        await self._github_runtime.aclose()


class LiveGitHubPullRequestAssessmentRunner(_LiveGitHubRunner):
    """Build live GitHub dependencies and assess one pull request."""

    async def assess(
        self,
        request: GitHubPullRequestAssessmentRequest,
    ) -> GitHubPullRequestAssessmentResult:
        """Run one read-only assessment using the GitHub REST API."""

        settings = self._settings_factory()

        if not settings.configured:
            raise GitHubConfigurationError(
                "GitHub App authentication is not configured. "
                "Set OPENSTEWARD_GITHUB_APP_ID and either "
                "OPENSTEWARD_GITHUB_PRIVATE_KEY or "
                "OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH."
            )

        token_scope = (
            GitHubInstallationTokenScope(
                repositories=[
                    request.repository.name,
                ],
                permissions={
                    "contents": (
                        GitHubPermissionLevel.READ
                    ),
                    "pull_requests": (
                        GitHubPermissionLevel.READ
                    ),
                    "checks": (
                        GitHubPermissionLevel.READ
                    ),
                },
            )
        )

        async with self._github_runtime.use(settings) as runtime:
            http_client = runtime.http_client
            token_provider = runtime.token_provider

            rest_client = GitHubRestClient(
                settings=settings,
                token_provider=token_provider,
                client=http_client,
                installation_id=(
                    request.installation_id
                ),
                token_scope=token_scope,
            )

            pull_request_service = (
                GitHubPullRequestService(
                    rest_client=rest_client,
                )
            )

            repository_service = (
                GitHubRepositoryService(
                    rest_client=rest_client,
                )
            )

            assessment_service = (
                GitHubPullRequestAssessmentService(
                    pull_request_loader=(
                        pull_request_service
                    ),
                    policy_loader=(
                        repository_service
                    ),
                )
            )

            return await assessment_service.assess(
                request
            )


class LiveGitHubRelatedWorkRunner(_LiveGitHubRunner):
    """Build live GitHub dependencies and run one related-work search."""

    async def find(
        self,
        request: GitHubRelatedWorkRequest,
    ) -> GitHubRelatedWorkResult:
        """Run one bounded read-only historical related-work search."""

        settings = self._settings_factory()
        if not settings.configured:
            raise GitHubConfigurationError(
                "GitHub App authentication is not configured. "
                "Set OPENSTEWARD_GITHUB_APP_ID and either "
                "OPENSTEWARD_GITHUB_PRIVATE_KEY or "
                "OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH."
            )

        token_scope = GitHubInstallationTokenScope(
            repositories=[request.repository.name],
            permissions={
                "contents": GitHubPermissionLevel.READ,
                "issues": GitHubPermissionLevel.READ,
                "pull_requests": GitHubPermissionLevel.READ,
            },
        )

        async with self._github_runtime.use(settings) as runtime:
            http_client = runtime.http_client
            token_provider = runtime.token_provider
            rest_client = GitHubRestClient(
                settings=settings,
                token_provider=token_provider,
                client=http_client,
                installation_id=request.installation_id,
                token_scope=token_scope,
            )

            historical_collector = GitHubHistoricalKnowledgeCollector(
                rest_client=rest_client
            )
            path_enricher = GitHubHistoricalPullRequestPathEnricher(
                rest_client=rest_client
            )
            adr_collector = GitHubHistoricalAdrCollector(
                rest_client=rest_client
            )
            snapshot_service = GitHubHistoricalKnowledgeSnapshotService(
                historical_items_collector=historical_collector,
                path_enricher=path_enricher,
                adr_collector=adr_collector,
            )
            related_work_finder = _build_related_work_finder(
                self._semantic_settings_factory(),
                client=http_client,
            )
            related_work_service = GitHubRelatedWorkService(
                snapshot_collector=snapshot_service,
                related_work_finder=related_work_finder,
            )
            return await related_work_service.find(request)


class LiveGitHubReviewCostRunner(_LiveGitHubRunner):
    """Build one shared live runtime for evidence-derived review cost."""

    async def assess(
        self,
        request: GitHubReviewCostRequest,
    ) -> GitHubReviewCostResult:
        """Run one read-only review-cost assessment."""

        settings = self._settings_factory()
        if not settings.configured:
            raise GitHubConfigurationError(
                "GitHub App authentication is not configured. "
                "Set OPENSTEWARD_GITHUB_APP_ID and either "
                "OPENSTEWARD_GITHUB_PRIVATE_KEY or "
                "OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH."
            )
        token_scope = GitHubInstallationTokenScope(
            repositories=[request.repository.name],
            permissions={
                "contents": GitHubPermissionLevel.READ,
                "pull_requests": GitHubPermissionLevel.READ,
                "checks": GitHubPermissionLevel.READ,
                "issues": GitHubPermissionLevel.READ,
            },
        )
        async with self._github_runtime.use(settings) as runtime:
            http_client = runtime.http_client
            token_provider = runtime.token_provider
            rest_client = GitHubRestClient(
                settings=settings,
                token_provider=token_provider,
                client=http_client,
                installation_id=request.installation_id,
                token_scope=token_scope,
            )

            pull_request_service = GitHubPullRequestService(
                rest_client=rest_client
            )
            repository_service = GitHubRepositoryService(
                rest_client=rest_client
            )
            pull_request_assessor = GitHubPullRequestAssessmentService(
                pull_request_loader=pull_request_service,
                policy_loader=repository_service,
            )

            historical_collector = GitHubHistoricalKnowledgeCollector(
                rest_client=rest_client
            )
            path_enricher = GitHubHistoricalPullRequestPathEnricher(
                rest_client=rest_client
            )
            adr_collector = GitHubHistoricalAdrCollector(
                rest_client=rest_client
            )
            snapshot_service = GitHubHistoricalKnowledgeSnapshotService(
                historical_items_collector=historical_collector,
                path_enricher=path_enricher,
                adr_collector=adr_collector,
            )
            related_work_finder = _build_related_work_finder(
                self._semantic_settings_factory(),
                client=http_client,
            )
            related_work_service = GitHubRelatedWorkService(
                snapshot_collector=snapshot_service,
                related_work_finder=related_work_finder,
            )

            review_cost_assessor = ReviewCostAssessmentService()
            review_cost_service = GitHubReviewCostService(
                pull_request_assessor=pull_request_assessor,
                related_work_finder=related_work_service,
                review_cost_assessor=review_cost_assessor,
            )
            return await review_cost_service.assess(request)


class LiveGitHubMaintainerBriefRunner(_LiveGitHubRunner):
    """Build one shared live runtime for a structured maintainer brief."""

    async def build(
        self,
        request: GitHubMaintainerBriefRequest,
    ) -> GitHubMaintainerBriefResult:
        """Run one read-only maintainer-brief assessment."""

        settings = self._settings_factory()
        if not settings.configured:
            raise GitHubConfigurationError(
                "GitHub App authentication is not configured. "
                "Set OPENSTEWARD_GITHUB_APP_ID and either "
                "OPENSTEWARD_GITHUB_PRIVATE_KEY or "
                "OPENSTEWARD_GITHUB_PRIVATE_KEY_PATH."
            )
        token_scope = GitHubInstallationTokenScope(
            repositories=[request.repository.name],
            permissions={
                "contents": GitHubPermissionLevel.READ,
                "pull_requests": GitHubPermissionLevel.READ,
                "checks": GitHubPermissionLevel.READ,
                "issues": GitHubPermissionLevel.READ,
            },
        )
        async with self._github_runtime.use(settings) as runtime:
            http_client = runtime.http_client
            token_provider = runtime.token_provider
            rest_client = GitHubRestClient(
                settings=settings,
                token_provider=token_provider,
                client=http_client,
                installation_id=request.installation_id,
                token_scope=token_scope,
            )

            pull_request_service = GitHubPullRequestService(
                rest_client=rest_client
            )
            repository_service = GitHubRepositoryService(
                rest_client=rest_client
            )
            pull_request_assessor = GitHubPullRequestAssessmentService(
                pull_request_loader=pull_request_service,
                policy_loader=repository_service,
            )

            historical_collector = GitHubHistoricalKnowledgeCollector(
                rest_client=rest_client
            )
            path_enricher = GitHubHistoricalPullRequestPathEnricher(
                rest_client=rest_client
            )
            adr_collector = GitHubHistoricalAdrCollector(
                rest_client=rest_client
            )
            snapshot_service = GitHubHistoricalKnowledgeSnapshotService(
                historical_items_collector=historical_collector,
                path_enricher=path_enricher,
                adr_collector=adr_collector,
            )
            related_work_finder = _build_related_work_finder(
                self._semantic_settings_factory(),
                client=http_client,
            )
            related_work_service = GitHubRelatedWorkService(
                snapshot_collector=snapshot_service,
                related_work_finder=related_work_finder,
            )

            review_cost_assessor = ReviewCostAssessmentService()
            review_cost_service = GitHubReviewCostService(
                pull_request_assessor=pull_request_assessor,
                related_work_finder=related_work_service,
                review_cost_assessor=review_cost_assessor,
            )
            maintainer_brief_service = MaintainerBriefService()
            github_service = GitHubMaintainerBriefService(
                review_cost_assessor=review_cost_service,
                brief_builder=maintainer_brief_service,
            )
            return await github_service.build(request)
