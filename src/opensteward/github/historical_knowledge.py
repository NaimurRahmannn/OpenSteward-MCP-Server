"""Bounded collection of historical GitHub issues and pull requests."""

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, Self
from urllib.parse import quote

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from opensteward.github.models import (
    GitHubRepositoryRef,
    StrictGitHubModel,
)
from opensteward.github.rest_client import (
    DEFAULT_GITHUB_ACCEPT,
    GitHubRestResponse,
)
from opensteward.knowledge import (
    DecisionSignificance,
    KnowledgeActor,
    KnowledgeActorType,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

GITHUB_HISTORICAL_PAGE_SIZE = 100
MAX_GITHUB_HISTORICAL_ITEMS_PER_TYPE = 500
MAX_GITHUB_HISTORICAL_SCAN_PAGES = 20


class GitHubHistoricalKnowledgeCollectionOptions(StrictGitHubModel):
    """Caller-controlled limits for bounded historical collection."""

    max_issues: int = Field(
        default=100,
        ge=0,
        le=MAX_GITHUB_HISTORICAL_ITEMS_PER_TYPE,
    )
    max_pull_requests: int = Field(
        default=100,
        ge=0,
        le=MAX_GITHUB_HISTORICAL_ITEMS_PER_TYPE,
    )


class GitHubHistoricalKnowledgeCollectionStats(StrictGitHubModel):
    """Collection statistics for one historical GitHub feed."""

    requested_limit: int = Field(ge=0)
    pages_fetched: int = Field(ge=0)
    api_items_seen: int = Field(ge=0)
    items_collected: int = Field(ge=0)
    filtered_items: int = Field(ge=0)
    duplicate_items: int = Field(ge=0)
    item_limit_reached: bool
    scan_limit_reached: bool

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Reject impossible combinations of collection statistics."""

        if self.items_collected > self.requested_limit:
            raise ValueError("items_collected must not exceed requested_limit.")

        categorized_items = (
            self.items_collected
            + self.filtered_items
            + self.duplicate_items
        )
        if categorized_items > self.api_items_seen:
            raise ValueError("Categorized items must not exceed api_items_seen.")

        if self.item_limit_reached and self.items_collected != self.requested_limit:
            raise ValueError(
                "item_limit_reached requires items_collected to equal requested_limit."
            )

        if self.item_limit_reached and self.scan_limit_reached:
            raise ValueError("Item and scan limits cannot both stop collection.")

        if (
            self.scan_limit_reached
            and self.pages_fetched != MAX_GITHUB_HISTORICAL_SCAN_PAGES
        ):
            raise ValueError(
                "scan_limit_reached requires the maximum number of scan pages."
            )

        return self


class GitHubHistoricalKnowledgeCollectionResult(StrictGitHubModel):
    """Historical GitHub items and collection statistics."""

    repository: GitHubRepositoryRef
    knowledge_repository: KnowledgeRepositoryRef
    items: list[KnowledgeItem]
    issue_stats: GitHubHistoricalKnowledgeCollectionStats
    pull_request_stats: GitHubHistoricalKnowledgeCollectionStats

    @field_validator("items")
    @classmethod
    def sort_items(cls, items: list[KnowledgeItem]) -> list[KnowledgeItem]:
        """Sort items by descending update time and ascending key."""

        key_sorted = sorted(items, key=lambda item: item.key)
        return sorted(
            key_sorted,
            key=lambda item: item.updated_at,
            reverse=True,
        )

    @model_validator(mode="after")
    def validate_items_and_stats(self) -> Self:
        """Ensure emitted items match repository and statistics metadata."""

        if any(item.repository != self.knowledge_repository for item in self.items):
            raise ValueError(
                "Every historical item must belong to the knowledge repository."
            )

        if any(
            item.source_kind != KnowledgeSourceKind.GITHUB
            for item in self.items
        ):
            raise ValueError("Every historical item must use the GitHub source kind.")

        supported_types = {
            KnowledgeItemType.ISSUE,
            KnowledgeItemType.PULL_REQUEST,
        }
        if any(item.item_type not in supported_types for item in self.items):
            raise ValueError(
                "Historical GitHub results may contain only issues and pull requests."
            )

        item_keys = [item.key for item in self.items]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Historical GitHub item keys must be unique.")

        if self.issue_stats.items_collected != self.issue_count:
            raise ValueError("Issue statistics must match the issue item count.")

        if self.pull_request_stats.items_collected != self.pull_request_count:
            raise ValueError(
                "Pull-request statistics must match the pull-request item count."
            )

        return self

    @computed_field
    @property
    def total_count(self) -> int:
        """Return the total number of collected knowledge items."""

        return len(self.items)

    @computed_field
    @property
    def issue_count(self) -> int:
        """Return the number of collected issue items."""

        return sum(
            item.item_type == KnowledgeItemType.ISSUE
            for item in self.items
        )

    @computed_field
    @property
    def pull_request_count(self) -> int:
        """Return the number of collected pull-request items."""

        return sum(
            item.item_type == KnowledgeItemType.PULL_REQUEST
            for item in self.items
        )


def knowledge_repository_from_github(
    repository: GitHubRepositoryRef,
) -> KnowledgeRepositoryRef:
    """Convert a GitHub repository identity into knowledge-domain identity."""

    return KnowledgeRepositoryRef(
        provider="github",
        namespace=repository.owner,
        name=repository.name,
    )


class GitHubJsonClient(Protocol):
    """REST-client behavior required by historical collection."""

    async def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        response_type: Any = Any,
        accept: str = DEFAULT_GITHUB_ACCEPT,
    ) -> GitHubRestResponse[Any]:
        """Retrieve and validate one GitHub JSON response."""

        ...


class _GitHubHistoricalApiModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )


class _ApiActor(_GitHubHistoricalApiModel):
    id: int = Field(gt=0)
    login: str = Field(min_length=1)
    type: str = Field(min_length=1)
    html_url: str | None = None


class _ApiLabel(_GitHubHistoricalApiModel):
    name: str = Field(min_length=1)


class _ApiIssue(_GitHubHistoricalApiModel):
    number: int = Field(gt=0)
    title: str = Field(min_length=1)
    body: str | None = None
    state: str = Field(min_length=1)
    html_url: str = Field(min_length=1)
    user: _ApiActor | None = None
    labels: list[_ApiLabel]
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    pull_request: dict[str, Any] | None = None


class _ApiPullRequest(_GitHubHistoricalApiModel):
    number: int = Field(gt=0)
    title: str = Field(min_length=1)
    body: str | None = None
    state: str = Field(min_length=1)
    html_url: str = Field(min_length=1)
    user: _ApiActor | None = None
    labels: list[_ApiLabel]
    created_at: datetime
    updated_at: datetime
    closed_at: datetime | None = None
    merged_at: datetime | None = None


def _build_repository_path(repository: GitHubRepositoryRef) -> str:
    """Build a safely encoded repository REST path."""

    owner = quote(repository.owner, safe="")
    name = quote(repository.name, safe="")
    return f"/repos/{owner}/{name}"


def _convert_actor(actor: _ApiActor | None) -> KnowledgeActor | None:
    """Convert a GitHub actor into a knowledge-domain actor."""

    if actor is None:
        return None

    actor_types = {
        "user": KnowledgeActorType.USER,
        "bot": KnowledgeActorType.BOT,
        "organization": KnowledgeActorType.ORGANIZATION,
    }

    return KnowledgeActor(
        identifier=str(actor.id),
        actor_type=actor_types.get(
            actor.type.casefold(),
            KnowledgeActorType.UNKNOWN,
        ),
        display_name=actor.login,
        url=actor.html_url,
    )


def _normalize_labels(labels: list[_ApiLabel]) -> list[str]:
    """Deduplicate label names while preserving their first spelling."""

    names: list[str] = []
    seen: set[str] = set()

    for label in labels:
        normalized = label.name.casefold()
        if normalized in seen:
            continue

        seen.add(normalized)
        names.append(label.name)

    return names


def _issue_state(state: str) -> KnowledgeItemState:
    normalized = state.casefold()
    if normalized == "closed":
        return KnowledgeItemState.CLOSED
    if normalized == "open":
        return KnowledgeItemState.OPEN
    return KnowledgeItemState.UNKNOWN


def _pull_request_state(
    state: str,
    merged_at: datetime | None,
) -> KnowledgeItemState:
    if merged_at is not None:
        return KnowledgeItemState.MERGED

    normalized = state.casefold()
    if normalized == "closed":
        return KnowledgeItemState.REJECTED
    if normalized == "open":
        return KnowledgeItemState.OPEN
    return KnowledgeItemState.UNKNOWN


def _convert_issue(
    issue: _ApiIssue,
    repository: KnowledgeRepositoryRef,
) -> KnowledgeItem:
    """Convert one GitHub issue into a knowledge item."""

    return KnowledgeItem(
        repository=repository,
        item_type=KnowledgeItemType.ISSUE,
        external_id=str(issue.number),
        source_kind=KnowledgeSourceKind.GITHUB,
        state=_issue_state(issue.state),
        title=issue.title,
        body=issue.body,
        summary=None,
        url=issue.html_url,
        author=_convert_actor(issue.user),
        created_at=issue.created_at,
        updated_at=issue.updated_at,
        closed_at=issue.closed_at,
        labels=_normalize_labels(issue.labels),
        affected_paths=[],
        components=[],
        decision_significance=DecisionSignificance.NONE,
    )


def _convert_pull_request(
    pull_request: _ApiPullRequest,
    repository: KnowledgeRepositoryRef,
) -> KnowledgeItem:
    """Convert one GitHub pull request into a knowledge item."""

    return KnowledgeItem(
        repository=repository,
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id=str(pull_request.number),
        source_kind=KnowledgeSourceKind.GITHUB,
        state=_pull_request_state(
            pull_request.state,
            pull_request.merged_at,
        ),
        title=pull_request.title,
        body=pull_request.body,
        summary=None,
        url=pull_request.html_url,
        author=_convert_actor(pull_request.user),
        created_at=pull_request.created_at,
        updated_at=pull_request.updated_at,
        closed_at=pull_request.closed_at,
        labels=_normalize_labels(pull_request.labels),
        affected_paths=[],
        components=[],
        decision_significance=DecisionSignificance.NONE,
    )


def _zero_limit_stats() -> GitHubHistoricalKnowledgeCollectionStats:
    return GitHubHistoricalKnowledgeCollectionStats(
        requested_limit=0,
        pages_fetched=0,
        api_items_seen=0,
        items_collected=0,
        filtered_items=0,
        duplicate_items=0,
        item_limit_reached=True,
        scan_limit_reached=False,
    )


def _build_stats(
    *,
    requested_limit: int,
    pages_fetched: int,
    api_items_seen: int,
    items_collected: int,
    filtered_items: int,
    duplicate_items: int,
    item_limit_reached: bool,
    scan_limit_reached: bool,
) -> GitHubHistoricalKnowledgeCollectionStats:
    return GitHubHistoricalKnowledgeCollectionStats(
        requested_limit=requested_limit,
        pages_fetched=pages_fetched,
        api_items_seen=api_items_seen,
        items_collected=items_collected,
        filtered_items=filtered_items,
        duplicate_items=duplicate_items,
        item_limit_reached=item_limit_reached,
        scan_limit_reached=scan_limit_reached,
    )


class GitHubHistoricalKnowledgeCollector:
    """Collect bounded closed-issue and pull-request history from GitHub."""

    def __init__(self, *, rest_client: GitHubJsonClient) -> None:
        self._rest_client = rest_client

    async def collect_closed_items(
        self,
        repository: GitHubRepositoryRef,
        *,
        options: GitHubHistoricalKnowledgeCollectionOptions | None = None,
    ) -> GitHubHistoricalKnowledgeCollectionResult:
        """Collect and normalize recent closed issues and pull requests."""

        selected_options = (
            options
            if options is not None
            else GitHubHistoricalKnowledgeCollectionOptions()
        )
        knowledge_repository = knowledge_repository_from_github(repository)
        repository_path = _build_repository_path(repository)

        issue_items, issue_stats = await self._collect_issues(
            repository_path=repository_path,
            knowledge_repository=knowledge_repository,
            requested_limit=selected_options.max_issues,
        )
        pull_request_items, pull_request_stats = (
            await self._collect_pull_requests(
                repository_path=repository_path,
                knowledge_repository=knowledge_repository,
                requested_limit=selected_options.max_pull_requests,
            )
        )

        return GitHubHistoricalKnowledgeCollectionResult(
            repository=repository,
            knowledge_repository=knowledge_repository,
            items=issue_items + pull_request_items,
            issue_stats=issue_stats,
            pull_request_stats=pull_request_stats,
        )

    async def _collect_issues(
        self,
        *,
        repository_path: str,
        knowledge_repository: KnowledgeRepositoryRef,
        requested_limit: int,
    ) -> tuple[
        list[KnowledgeItem],
        GitHubHistoricalKnowledgeCollectionStats,
    ]:
        if requested_limit == 0:
            return [], _zero_limit_stats()

        items: list[KnowledgeItem] = []
        seen_keys: set[str] = set()
        pages_fetched = 0
        api_items_seen = 0
        filtered_items = 0
        duplicate_items = 0
        final_page_was_full = False

        for page in range(1, MAX_GITHUB_HISTORICAL_SCAN_PAGES + 1):
            response = await self._rest_client.get_json(
                f"{repository_path}/issues",
                params={
                    "state": "closed",
                    "filter": "all",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": GITHUB_HISTORICAL_PAGE_SIZE,
                    "page": page,
                },
                response_type=list[_ApiIssue],
            )

            page_items = response.data
            pages_fetched += 1
            api_items_seen += len(page_items)
            final_page_was_full = (
                len(page_items) == GITHUB_HISTORICAL_PAGE_SIZE
            )

            for issue in page_items:
                if issue.pull_request is not None:
                    filtered_items += 1
                    continue

                item = _convert_issue(issue, knowledge_repository)
                if item.key in seen_keys:
                    duplicate_items += 1
                    continue

                seen_keys.add(item.key)
                items.append(item)

                if len(items) == requested_limit:
                    return items, _build_stats(
                        requested_limit=requested_limit,
                        pages_fetched=pages_fetched,
                        api_items_seen=api_items_seen,
                        items_collected=len(items),
                        filtered_items=filtered_items,
                        duplicate_items=duplicate_items,
                        item_limit_reached=True,
                        scan_limit_reached=False,
                    )

            if not final_page_was_full:
                break

        return items, _build_stats(
            requested_limit=requested_limit,
            pages_fetched=pages_fetched,
            api_items_seen=api_items_seen,
            items_collected=len(items),
            filtered_items=filtered_items,
            duplicate_items=duplicate_items,
            item_limit_reached=False,
            scan_limit_reached=(
                pages_fetched == MAX_GITHUB_HISTORICAL_SCAN_PAGES
                and final_page_was_full
            ),
        )

    async def _collect_pull_requests(
        self,
        *,
        repository_path: str,
        knowledge_repository: KnowledgeRepositoryRef,
        requested_limit: int,
    ) -> tuple[
        list[KnowledgeItem],
        GitHubHistoricalKnowledgeCollectionStats,
    ]:
        if requested_limit == 0:
            return [], _zero_limit_stats()

        items: list[KnowledgeItem] = []
        seen_keys: set[str] = set()
        pages_fetched = 0
        api_items_seen = 0
        duplicate_items = 0
        final_page_was_full = False

        for page in range(1, MAX_GITHUB_HISTORICAL_SCAN_PAGES + 1):
            response = await self._rest_client.get_json(
                f"{repository_path}/pulls",
                params={
                    "state": "closed",
                    "sort": "updated",
                    "direction": "desc",
                    "per_page": GITHUB_HISTORICAL_PAGE_SIZE,
                    "page": page,
                },
                response_type=list[_ApiPullRequest],
            )

            page_items = response.data
            pages_fetched += 1
            api_items_seen += len(page_items)
            final_page_was_full = (
                len(page_items) == GITHUB_HISTORICAL_PAGE_SIZE
            )

            for pull_request in page_items:
                item = _convert_pull_request(
                    pull_request,
                    knowledge_repository,
                )
                if item.key in seen_keys:
                    duplicate_items += 1
                    continue

                seen_keys.add(item.key)
                items.append(item)

                if len(items) == requested_limit:
                    return items, _build_stats(
                        requested_limit=requested_limit,
                        pages_fetched=pages_fetched,
                        api_items_seen=api_items_seen,
                        items_collected=len(items),
                        filtered_items=0,
                        duplicate_items=duplicate_items,
                        item_limit_reached=True,
                        scan_limit_reached=False,
                    )

            if not final_page_was_full:
                break

        return items, _build_stats(
            requested_limit=requested_limit,
            pages_fetched=pages_fetched,
            api_items_seen=api_items_seen,
            items_collected=len(items),
            filtered_items=0,
            duplicate_items=duplicate_items,
            item_limit_reached=False,
            scan_limit_reached=(
                pages_fetched == MAX_GITHUB_HISTORICAL_SCAN_PAGES
                and final_page_was_full
            ),
        )
