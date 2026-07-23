"""Changed-path enrichment for historical GitHub pull requests."""

import re
from collections.abc import Mapping
from typing import Any, Protocol, Self
from urllib.parse import quote

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    model_validator,
)

from opensteward.github.historical_knowledge import (
    GitHubHistoricalKnowledgeCollectionResult,
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
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

GITHUB_HISTORICAL_PATH_PAGE_SIZE = 100
MAX_GITHUB_HISTORICAL_FILES_PER_PULL_REQUEST = 3_000
MAX_GITHUB_HISTORICAL_PULL_REQUESTS_TO_ENRICH = 100
MAX_GITHUB_HISTORICAL_PATH_PAGES = (
    MAX_GITHUB_HISTORICAL_FILES_PER_PULL_REQUEST
    // GITHUB_HISTORICAL_PATH_PAGE_SIZE
)


class GitHubHistoricalPathEnrichmentError(ValueError):
    """Raised when a historical pull request cannot be safely enriched."""


class GitHubHistoricalPathEnrichmentOptions(StrictGitHubModel):
    """Caller-controlled limit for historical path enrichment."""

    max_pull_requests: int = Field(
        default=50,
        ge=0,
        le=MAX_GITHUB_HISTORICAL_PULL_REQUESTS_TO_ENRICH,
    )


class GitHubHistoricalPullRequestPathEvidence(StrictGitHubModel):
    """Completeness metadata for one enriched pull request."""

    item_key: str = Field(min_length=1)
    pull_number: int = Field(gt=0)
    pages_fetched: int = Field(ge=0)
    api_files_seen: int = Field(ge=0)
    affected_paths_collected: int = Field(ge=0)
    complete: bool
    file_limit_reached: bool

    @model_validator(mode="after")
    def validate_completeness(self) -> Self:
        """Reject contradictory path-collection metadata."""

        if self.file_limit_reached and self.complete:
            raise ValueError("File-limited path evidence cannot be complete.")

        if self.file_limit_reached and (
            self.pages_fetched != MAX_GITHUB_HISTORICAL_PATH_PAGES
            or self.api_files_seen
            < MAX_GITHUB_HISTORICAL_FILES_PER_PULL_REQUEST
        ):
            raise ValueError(
                "file_limit_reached requires all path pages and at least "
                "the maximum number of API files."
            )

        return self


class GitHubHistoricalPathEnrichmentResult(StrictGitHubModel):
    """Historical items enriched with pull-request changed paths."""

    repository: GitHubRepositoryRef
    knowledge_repository: KnowledgeRepositoryRef
    items: list[KnowledgeItem]
    pull_request_evidence: list[
        GitHubHistoricalPullRequestPathEvidence
    ]
    pull_requests_available: int = Field(ge=0)
    pull_requests_enriched: int = Field(ge=0)
    pull_requests_skipped_due_limit: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_items_and_evidence(self) -> Self:
        """Ensure items, evidence, and collection counts agree."""

        if any(item.repository != self.knowledge_repository for item in self.items):
            raise ValueError(
                "Every enriched item must belong to the knowledge repository."
            )

        if any(
            item.source_kind != KnowledgeSourceKind.GITHUB
            for item in self.items
        ):
            raise ValueError("Every enriched item must use the GitHub source kind.")

        supported_types = {
            KnowledgeItemType.ISSUE,
            KnowledgeItemType.PULL_REQUEST,
        }
        if any(item.item_type not in supported_types for item in self.items):
            raise ValueError(
                "Historical path results may contain only issues and pull requests."
            )

        item_keys = [item.key for item in self.items]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Historical path result item keys must be unique.")

        pull_request_keys = {
            item.key
            for item in self.items
            if item.item_type == KnowledgeItemType.PULL_REQUEST
        }
        evidence_keys = [
            evidence.item_key
            for evidence in self.pull_request_evidence
        ]

        if any(key not in pull_request_keys for key in evidence_keys):
            raise ValueError(
                "Every path evidence item_key must identify a pull-request item."
            )

        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("Path evidence item keys must be unique.")

        if self.pull_requests_available != self.pull_request_count:
            raise ValueError(
                "pull_requests_available must match the pull-request item count."
            )

        if self.pull_requests_enriched != len(self.pull_request_evidence):
            raise ValueError(
                "pull_requests_enriched must match the path evidence count."
            )

        if self.pull_requests_enriched > self.pull_requests_available:
            raise ValueError(
                "pull_requests_enriched must not exceed pull_requests_available."
            )

        expected_skipped = (
            self.pull_requests_available
            - self.pull_requests_enriched
        )
        if self.pull_requests_skipped_due_limit != expected_skipped:
            raise ValueError(
                "pull_requests_skipped_due_limit must match available minus enriched."
            )

        return self

    @computed_field
    @property
    def total_count(self) -> int:
        """Return the total number of historical items."""

        return len(self.items)

    @computed_field
    @property
    def issue_count(self) -> int:
        """Return the number of issue items."""

        return sum(
            item.item_type == KnowledgeItemType.ISSUE
            for item in self.items
        )

    @computed_field
    @property
    def pull_request_count(self) -> int:
        """Return the number of pull-request items."""

        return sum(
            item.item_type == KnowledgeItemType.PULL_REQUEST
            for item in self.items
        )

    @computed_field
    @property
    def complete_pull_request_count(self) -> int:
        """Return the number of completely enriched pull requests."""

        return sum(
            evidence.complete
            for evidence in self.pull_request_evidence
        )

    @computed_field
    @property
    def incomplete_pull_request_count(self) -> int:
        """Return the number of incompletely enriched pull requests."""

        return sum(
            not evidence.complete
            for evidence in self.pull_request_evidence
        )


class GitHubJsonClient(Protocol):
    """REST-client behavior required by path enrichment."""

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


class _GitHubHistoricalFileModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )

    filename: str = Field(min_length=1)
    previous_filename: str | None = None
    status: str = Field(min_length=1)


def _build_repository_path(repository: GitHubRepositoryRef) -> str:
    """Build a safely encoded repository REST path."""

    owner = quote(repository.owner, safe="")
    name = quote(repository.name, safe="")
    return f"/repos/{owner}/{name}"


def _pull_number_from_item(item: KnowledgeItem) -> int:
    """Validate and return a historical pull-request number."""

    if re.fullmatch(r"[0-9]+", item.external_id) is None:
        raise GitHubHistoricalPathEnrichmentError(
            f"Historical pull-request item has an invalid number: {item.key}."
        )

    pull_number = int(item.external_id)
    if pull_number <= 0:
        raise GitHubHistoricalPathEnrichmentError(
            f"Historical pull-request item has an invalid number: {item.key}."
        )

    return pull_number


def _append_unique_path(
    paths: list[str],
    seen_paths: set[str],
    path: str | None,
) -> bool:
    if path is None or path in seen_paths:
        return False

    seen_paths.add(path)
    paths.append(path)
    return True


def _replace_affected_paths(
    item: KnowledgeItem,
    affected_paths: list[str],
) -> KnowledgeItem:
    payload = {
        field_name: getattr(item, field_name)
        for field_name in KnowledgeItem.model_fields
    }
    payload["affected_paths"] = affected_paths
    return KnowledgeItem.model_validate(payload)


class GitHubHistoricalPullRequestPathEnricher:
    """Enrich historical pull requests with bounded changed-path evidence."""

    def __init__(self, *, rest_client: GitHubJsonClient) -> None:
        self._rest_client = rest_client

    async def enrich(
        self,
        collection: GitHubHistoricalKnowledgeCollectionResult,
        *,
        options: GitHubHistoricalPathEnrichmentOptions | None = None,
    ) -> GitHubHistoricalPathEnrichmentResult:
        """Enrich selected pull requests without mutating the collection."""

        selected_options = (
            options
            if options is not None
            else GitHubHistoricalPathEnrichmentOptions()
        )
        pull_request_indexes = [
            index
            for index, item in enumerate(collection.items)
            if item.item_type == KnowledgeItemType.PULL_REQUEST
        ]
        selected_indexes = pull_request_indexes[
            :selected_options.max_pull_requests
        ]
        result_items = list(collection.items)
        evidence_items: list[
            GitHubHistoricalPullRequestPathEvidence
        ] = []
        repository_path = _build_repository_path(collection.repository)

        for index in selected_indexes:
            enriched_item, evidence = await self._enrich_item(
                item=collection.items[index],
                repository_path=repository_path,
            )
            result_items[index] = enriched_item
            evidence_items.append(evidence)

        pull_requests_available = len(pull_request_indexes)
        pull_requests_enriched = len(evidence_items)

        return GitHubHistoricalPathEnrichmentResult(
            repository=collection.repository,
            knowledge_repository=collection.knowledge_repository,
            items=result_items,
            pull_request_evidence=evidence_items,
            pull_requests_available=pull_requests_available,
            pull_requests_enriched=pull_requests_enriched,
            pull_requests_skipped_due_limit=(
                pull_requests_available
                - pull_requests_enriched
            ),
        )

    async def _enrich_item(
        self,
        *,
        item: KnowledgeItem,
        repository_path: str,
    ) -> tuple[
        KnowledgeItem,
        GitHubHistoricalPullRequestPathEvidence,
    ]:
        pull_number = _pull_number_from_item(item)
        encoded_pull_number = quote(str(pull_number), safe="")
        path = (
            f"{repository_path}"
            f"/pulls/{encoded_pull_number}/files"
        )

        affected_paths = list(item.affected_paths)
        seen_paths = set(affected_paths)
        pages_fetched = 0
        api_files_seen = 0
        api_files_collected = 0
        affected_paths_collected = 0

        for page in range(1, MAX_GITHUB_HISTORICAL_PATH_PAGES + 1):
            response = await self._rest_client.get_json(
                path,
                params={
                    "per_page": GITHUB_HISTORICAL_PATH_PAGE_SIZE,
                    "page": page,
                },
                response_type=list[_GitHubHistoricalFileModel],
            )

            page_files = response.data
            pages_fetched += 1
            api_files_seen += len(page_files)

            remaining_capacity = (
                MAX_GITHUB_HISTORICAL_FILES_PER_PULL_REQUEST
                - api_files_collected
            )
            files_to_collect = page_files[:remaining_capacity]

            for file in files_to_collect:
                affected_paths_collected += _append_unique_path(
                    affected_paths,
                    seen_paths,
                    file.filename,
                )
                affected_paths_collected += _append_unique_path(
                    affected_paths,
                    seen_paths,
                    file.previous_filename,
                )

            api_files_collected += len(files_to_collect)

            if len(page_files) < GITHUB_HISTORICAL_PATH_PAGE_SIZE:
                enriched_item = _replace_affected_paths(
                    item,
                    affected_paths,
                )
                return enriched_item, (
                    GitHubHistoricalPullRequestPathEvidence(
                        item_key=item.key,
                        pull_number=pull_number,
                        pages_fetched=pages_fetched,
                        api_files_seen=api_files_seen,
                        affected_paths_collected=affected_paths_collected,
                        complete=True,
                        file_limit_reached=False,
                    )
                )

        enriched_item = _replace_affected_paths(
            item,
            affected_paths,
        )
        return enriched_item, GitHubHistoricalPullRequestPathEvidence(
            item_key=item.key,
            pull_number=pull_number,
            pages_fetched=pages_fetched,
            api_files_seen=api_files_seen,
            affected_paths_collected=affected_paths_collected,
            complete=False,
            file_limit_reached=True,
        )
