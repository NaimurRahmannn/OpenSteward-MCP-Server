"""Live GitHub runtime wiring for read-only capabilities."""

from collections.abc import Callable

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
from opensteward.knowledge import KnowledgeRelatedWorkService
from opensteward.review_intelligence import ReviewCostAssessmentService

SettingsFactory = Callable[
    [],
    GitHubAppSettings,
]


class LiveGitHubPullRequestAssessmentRunner:
    """Build live GitHub dependencies and assess one pull request."""

    def __init__(
        self,
        *,
        settings_factory: SettingsFactory = (
            get_github_settings
        ),
    ) -> None:
        self._settings_factory = (
            settings_factory
        )

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

        async with httpx.AsyncClient(
            follow_redirects=False,
        ) as http_client:
            token_provider = (
                GitHubInstallationTokenProvider(
                    settings=settings,
                    client=http_client,
                )
            )

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


class LiveGitHubRelatedWorkRunner:
    """Build live GitHub dependencies and run one related-work search."""

    def __init__(
        self,
        *,
        settings_factory: SettingsFactory = get_github_settings,
    ) -> None:
        self._settings_factory = settings_factory

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

        async with httpx.AsyncClient(
            follow_redirects=False,
        ) as http_client:
            token_provider = GitHubInstallationTokenProvider(
                settings=settings,
                client=http_client,
            )
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
            related_work_finder = KnowledgeRelatedWorkService()
            related_work_service = GitHubRelatedWorkService(
                snapshot_collector=snapshot_service,
                related_work_finder=related_work_finder,
            )
            return await related_work_service.find(request)


class LiveGitHubReviewCostRunner:
    """Build one shared live runtime for evidence-derived review cost."""

    def __init__(
        self,
        *,
        settings_factory: SettingsFactory = get_github_settings,
    ) -> None:
        self._settings_factory = settings_factory

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
        async with httpx.AsyncClient(
            follow_redirects=False,
        ) as http_client:
            token_provider = GitHubInstallationTokenProvider(
                settings=settings,
                client=http_client,
            )
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
            related_work_finder = KnowledgeRelatedWorkService()
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
