"""Live GitHub runtime wiring for pull-request assessments."""

from collections.abc import Callable

import httpx

from opensteward.github.assessments import (
    GitHubPullRequestAssessmentRequest,
    GitHubPullRequestAssessmentResult,
    GitHubPullRequestAssessmentService,
)
from opensteward.github.installation_tokens import (
    GitHubInstallationTokenProvider,
    GitHubInstallationTokenScope,
    GitHubPermissionLevel,
)
from opensteward.github.pull_requests import (
    GitHubPullRequestService,
)
from opensteward.github.repositories import (
    GitHubRepositoryService,
)
from opensteward.github.rest_client import (
    GitHubRestClient,
)
from opensteward.github.settings import (
    GitHubAppSettings,
    GitHubConfigurationError,
    get_github_settings,
)


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