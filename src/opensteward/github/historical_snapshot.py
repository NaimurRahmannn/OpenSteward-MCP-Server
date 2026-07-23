"""Unified orchestration of historical GitHub knowledge collection."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol, Self

from pydantic import (
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from opensteward.github.historical_adrs import (
    GitHubHistoricalAdrCollectionOptions,
    GitHubHistoricalAdrCollectionResult,
    GitHubHistoricalAdrCollectionStats,
    GitHubHistoricalAdrFileEvidence,
    GitHubHistoricalAdrSkippedFile,
)
from opensteward.github.historical_knowledge import (
    GitHubHistoricalKnowledgeCollectionOptions,
    GitHubHistoricalKnowledgeCollectionResult,
    GitHubHistoricalKnowledgeCollectionStats,
    knowledge_repository_from_github,
)
from opensteward.github.historical_paths import (
    GitHubHistoricalPathEnrichmentOptions,
    GitHubHistoricalPathEnrichmentResult,
    GitHubHistoricalPullRequestPathEvidence,
)
from opensteward.github.models import (
    GitHubRepositoryRef,
    StrictGitHubModel,
)
from opensteward.knowledge import (
    KnowledgeItem,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

_ISSUE_ITEM_LIMIT_WARNING = (
    "Historical issue collection reached its configured item limit."
)
_ISSUE_SCAN_LIMIT_WARNING = (
    "Historical issue collection reached its scan-page safety limit."
)
_PULL_REQUEST_ITEM_LIMIT_WARNING = (
    "Historical pull-request collection reached its configured item limit."
)
_PULL_REQUEST_SCAN_LIMIT_WARNING = (
    "Historical pull-request collection reached its scan-page safety limit."
)
_PATH_SKIPPED_WARNING = (
    "Changed-path evidence was not collected for all historical pull requests."
)
_PATH_INCOMPLETE_WARNING = (
    "Changed-path evidence is incomplete for one or more historical pull requests."
)
_ADR_ITEM_LIMIT_WARNING = (
    "Historical ADR collection reached its configured item limit."
)
_ADR_SKIPPED_WARNING = (
    "Historical ADR collection skipped one or more candidate files."
)


class GitHubHistoricalKnowledgeSnapshotError(ValueError):
    """Raised when historical snapshot components are inconsistent."""


class GitHubHistoricalKnowledgeSnapshotOptions(StrictGitHubModel):
    """Nested options for each bounded historical collection stage."""

    historical_items: GitHubHistoricalKnowledgeCollectionOptions = Field(
        default_factory=GitHubHistoricalKnowledgeCollectionOptions
    )
    pull_request_paths: GitHubHistoricalPathEnrichmentOptions = Field(
        default_factory=GitHubHistoricalPathEnrichmentOptions
    )
    adrs: GitHubHistoricalAdrCollectionOptions = Field(
        default_factory=GitHubHistoricalAdrCollectionOptions
    )


class GitHubHistoricalKnowledgeSnapshotResult(StrictGitHubModel):
    """Validated historical knowledge assembled from all GitHub sources."""

    repository: GitHubRepositoryRef
    knowledge_repository: KnowledgeRepositoryRef
    collected_at: datetime
    requested_ref: str = Field(min_length=1)
    resolved_adr_commit_sha: str = Field(min_length=1)
    adr_tree_sha: str = Field(min_length=1)
    adr_snapshot_commit_date: datetime
    items: list[KnowledgeItem]
    issue_stats: GitHubHistoricalKnowledgeCollectionStats
    pull_request_stats: GitHubHistoricalKnowledgeCollectionStats
    pull_request_path_evidence: list[
        GitHubHistoricalPullRequestPathEvidence
    ]
    pull_requests_available: int = Field(ge=0)
    pull_requests_enriched: int = Field(ge=0)
    pull_requests_skipped_due_limit: int = Field(ge=0)
    adr_file_evidence: list[GitHubHistoricalAdrFileEvidence]
    adr_skipped_files: list[GitHubHistoricalAdrSkippedFile]
    adr_stats: GitHubHistoricalAdrCollectionStats
    warnings: list[str]

    @field_validator("collected_at", "adr_snapshot_commit_date")
    @classmethod
    def normalize_datetimes(cls, value: datetime) -> datetime:
        """Require aware snapshot timestamps and normalize them to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Historical snapshot timestamps must be timezone-aware.")

        return value.astimezone(UTC)

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        """Require non-empty unique warnings in caller-provided order."""

        if any(not warning for warning in warnings):
            raise ValueError("Historical snapshot warnings must not be empty.")

        if len(warnings) != len(set(warnings)):
            raise ValueError("Historical snapshot warnings must be unique.")

        return warnings

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        """Validate combined items, evidence, identity, and metadata."""

        expected_repository = knowledge_repository_from_github(self.repository)
        if self.knowledge_repository != expected_repository:
            raise ValueError(
                "knowledge_repository must match the GitHub repository identity."
            )

        if any(item.repository != self.knowledge_repository for item in self.items):
            raise ValueError(
                "Every snapshot item must belong to knowledge_repository."
            )

        supported_types = {
            KnowledgeItemType.ISSUE,
            KnowledgeItemType.PULL_REQUEST,
            KnowledgeItemType.ADR,
        }
        if any(item.item_type not in supported_types for item in self.items):
            raise ValueError(
                "Historical snapshots may contain only issues, pull requests, and ADRs."
            )

        for item in self.items:
            expected_source = (
                KnowledgeSourceKind.REPOSITORY_FILE
                if item.item_type == KnowledgeItemType.ADR
                else KnowledgeSourceKind.GITHUB
            )
            if item.source_kind != expected_source:
                raise ValueError(
                    f"Snapshot item {item.key} uses an incorrect source kind."
                )

        item_keys = [item.key for item in self.items]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Historical snapshot item keys must be unique.")

        if self.items != _sort_items(self.items):
            raise ValueError(
                "Historical snapshot items must use deterministic update order."
            )

        if self.issue_stats.items_collected != self.issue_count:
            raise ValueError("Issue statistics must match the snapshot issue count.")

        if self.pull_request_stats.items_collected != self.pull_request_count:
            raise ValueError(
                "Pull-request statistics must match the snapshot pull-request count."
            )

        if self.adr_stats.items_collected != self.adr_count:
            raise ValueError("ADR statistics must match the snapshot ADR count.")

        items_by_key = {
            item.key: item
            for item in self.items
        }
        path_evidence_keys = [
            evidence.item_key
            for evidence in self.pull_request_path_evidence
        ]
        for evidence in self.pull_request_path_evidence:
            item = items_by_key.get(evidence.item_key)
            if item is None or item.item_type != KnowledgeItemType.PULL_REQUEST:
                raise ValueError(
                    "Every path evidence item_key must identify a pull-request item."
                )

        if len(path_evidence_keys) != len(set(path_evidence_keys)):
            raise ValueError("Path evidence item keys must be unique.")

        if self.pull_requests_available != self.pull_request_count:
            raise ValueError(
                "pull_requests_available must match the snapshot pull-request count."
            )

        if self.pull_requests_enriched != len(self.pull_request_path_evidence):
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

        adr_evidence_keys = [
            evidence.item_key
            for evidence in self.adr_file_evidence
        ]
        if len(adr_evidence_keys) != len(set(adr_evidence_keys)):
            raise ValueError("ADR evidence item keys must be unique.")

        adr_evidence_paths = [
            evidence.path
            for evidence in self.adr_file_evidence
        ]
        if len(adr_evidence_paths) != len(set(adr_evidence_paths)):
            raise ValueError("ADR evidence paths must be unique.")

        for evidence in self.adr_file_evidence:
            item = items_by_key.get(evidence.item_key)
            if item is None or item.item_type != KnowledgeItemType.ADR:
                raise ValueError(
                    "Every ADR evidence item_key must identify an ADR item."
                )

            if item.external_id != evidence.path:
                raise ValueError("ADR item external_id must equal its evidence path.")

            if item.url != evidence.html_url:
                raise ValueError("ADR item URL must equal its evidence HTML URL.")

        if len(self.adr_file_evidence) != self.adr_count:
            raise ValueError("ADR evidence count must match the snapshot ADR count.")

        if len(self.adr_skipped_files) != self.adr_stats.skipped_files:
            raise ValueError(
                "Skipped ADR count must match the ADR collection statistics."
            )

        return self

    @computed_field
    @property
    def total_count(self) -> int:
        """Return the total number of historical knowledge items."""

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
    def adr_count(self) -> int:
        """Return the number of ADR items."""

        return sum(
            item.item_type == KnowledgeItemType.ADR
            for item in self.items
        )

    @computed_field
    @property
    def complete(self) -> bool:
        """Return whether every bounded historical source is complete."""

        return not (
            self.issue_stats.item_limit_reached
            or self.issue_stats.scan_limit_reached
            or self.pull_request_stats.item_limit_reached
            or self.pull_request_stats.scan_limit_reached
            or self.pull_requests_skipped_due_limit
            or any(
                not evidence.complete
                for evidence in self.pull_request_path_evidence
            )
            or self.adr_stats.tree_truncated
            or self.adr_stats.item_limit_reached
            or self.adr_stats.total_bytes_limit_reached
            or self.adr_skipped_files
        )


class HistoricalItemsCollector(Protocol):
    """Historical issue and pull-request collection boundary."""

    async def collect_closed_items(
        self,
        repository: GitHubRepositoryRef,
        *,
        options: GitHubHistoricalKnowledgeCollectionOptions | None = None,
    ) -> GitHubHistoricalKnowledgeCollectionResult:
        """Collect bounded closed issues and pull requests."""

        ...


class HistoricalPathEnricher(Protocol):
    """Historical pull-request path enrichment boundary."""

    async def enrich(
        self,
        collection: GitHubHistoricalKnowledgeCollectionResult,
        *,
        options: GitHubHistoricalPathEnrichmentOptions | None = None,
    ) -> GitHubHistoricalPathEnrichmentResult:
        """Enrich historical pull requests with changed paths."""

        ...


class HistoricalAdrCollector(Protocol):
    """Historical repository ADR collection boundary."""

    async def collect(
        self,
        repository: GitHubRepositoryRef,
        *,
        git_ref: str,
        options: GitHubHistoricalAdrCollectionOptions | None = None,
    ) -> GitHubHistoricalAdrCollectionResult:
        """Collect bounded ADRs from an exact Git reference."""

        ...


type Clock = Callable[[], datetime]


def _default_clock() -> datetime:
    return datetime.now(UTC)


def _sort_items(items: list[KnowledgeItem]) -> list[KnowledgeItem]:
    key_sorted = sorted(items, key=lambda item: item.key)
    return sorted(
        key_sorted,
        key=lambda item: item.updated_at,
        reverse=True,
    )


def _normalize_clock_value(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "Historical snapshot clock must return a timezone-aware datetime."
        )

    return value.astimezone(UTC)


def _validate_historical_result(
    result: GitHubHistoricalKnowledgeCollectionResult,
    repository: GitHubRepositoryRef,
    knowledge_repository: KnowledgeRepositoryRef,
) -> None:
    if result.repository != repository:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "Historical collection belongs to another repository."
        )

    if result.knowledge_repository != knowledge_repository:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "Historical collection uses another knowledge repository."
        )


def _validate_path_result(
    result: GitHubHistoricalPathEnrichmentResult,
    source: GitHubHistoricalKnowledgeCollectionResult,
    repository: GitHubRepositoryRef,
    knowledge_repository: KnowledgeRepositoryRef,
) -> None:
    if result.repository != repository:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "Path enrichment belongs to another repository."
        )

    if result.knowledge_repository != knowledge_repository:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "Path enrichment uses another knowledge repository."
        )

    source_keys = [item.key for item in source.items]
    enriched_keys = [item.key for item in result.items]
    if enriched_keys != source_keys:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "Path enrichment changed the historical item set or order."
        )


def _validate_adr_result(
    result: GitHubHistoricalAdrCollectionResult,
    repository: GitHubRepositoryRef,
    knowledge_repository: KnowledgeRepositoryRef,
    requested_ref: str,
) -> None:
    if result.repository != repository:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "ADR collection belongs to another repository."
        )

    if result.knowledge_repository != knowledge_repository:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "ADR collection uses another knowledge repository."
        )

    if result.requested_ref != requested_ref:
        raise GitHubHistoricalKnowledgeSnapshotError(
            "ADR collection returned another requested Git reference."
        )


def _build_warnings(
    historical: GitHubHistoricalKnowledgeCollectionResult,
    paths: GitHubHistoricalPathEnrichmentResult,
    adrs: GitHubHistoricalAdrCollectionResult,
) -> list[str]:
    warnings: list[str] = []

    if historical.issue_stats.item_limit_reached:
        warnings.append(_ISSUE_ITEM_LIMIT_WARNING)
    if historical.issue_stats.scan_limit_reached:
        warnings.append(_ISSUE_SCAN_LIMIT_WARNING)
    if historical.pull_request_stats.item_limit_reached:
        warnings.append(_PULL_REQUEST_ITEM_LIMIT_WARNING)
    if historical.pull_request_stats.scan_limit_reached:
        warnings.append(_PULL_REQUEST_SCAN_LIMIT_WARNING)
    if paths.pull_requests_skipped_due_limit > 0:
        warnings.append(_PATH_SKIPPED_WARNING)
    if any(not evidence.complete for evidence in paths.pull_request_evidence):
        warnings.append(_PATH_INCOMPLETE_WARNING)
    if adrs.stats.item_limit_reached:
        warnings.append(_ADR_ITEM_LIMIT_WARNING)
    if adrs.skipped_files:
        warnings.append(_ADR_SKIPPED_WARNING)

    for warning in adrs.warnings:
        if warning not in warnings:
            warnings.append(warning)

    return warnings


class GitHubHistoricalKnowledgeSnapshotService:
    """Assemble one validated historical knowledge snapshot."""

    def __init__(
        self,
        *,
        historical_items_collector: HistoricalItemsCollector,
        path_enricher: HistoricalPathEnricher,
        adr_collector: HistoricalAdrCollector,
        clock: Clock | None = None,
    ) -> None:
        self._historical_items_collector = historical_items_collector
        self._path_enricher = path_enricher
        self._adr_collector = adr_collector
        self._clock = clock if clock is not None else _default_clock

    async def collect(
        self,
        repository: GitHubRepositoryRef,
        *,
        git_ref: str,
        options: GitHubHistoricalKnowledgeSnapshotOptions | None = None,
    ) -> GitHubHistoricalKnowledgeSnapshotResult:
        """Collect and validate all historical GitHub knowledge sources."""

        requested_ref = git_ref.strip()
        if not requested_ref:
            raise ValueError("Git reference must not be empty.")

        collected_at = _normalize_clock_value(self._clock())
        selected_options = (
            options
            if options is not None
            else GitHubHistoricalKnowledgeSnapshotOptions()
        )
        knowledge_repository = knowledge_repository_from_github(repository)

        historical_result = (
            await self._historical_items_collector.collect_closed_items(
                repository,
                options=selected_options.historical_items,
            )
        )
        _validate_historical_result(
            historical_result,
            repository,
            knowledge_repository,
        )

        path_result = await self._path_enricher.enrich(
            historical_result,
            options=selected_options.pull_request_paths,
        )
        _validate_path_result(
            path_result,
            historical_result,
            repository,
            knowledge_repository,
        )

        adr_result = await self._adr_collector.collect(
            repository,
            git_ref=requested_ref,
            options=selected_options.adrs,
        )
        _validate_adr_result(
            adr_result,
            repository,
            knowledge_repository,
            requested_ref,
        )

        items = _sort_items([
            *path_result.items,
            *adr_result.items,
        ])
        warnings = _build_warnings(
            historical_result,
            path_result,
            adr_result,
        )

        return GitHubHistoricalKnowledgeSnapshotResult(
            repository=repository,
            knowledge_repository=knowledge_repository,
            collected_at=collected_at,
            requested_ref=requested_ref,
            resolved_adr_commit_sha=adr_result.resolved_commit_sha,
            adr_tree_sha=adr_result.tree_sha,
            adr_snapshot_commit_date=adr_result.snapshot_commit_date,
            items=items,
            issue_stats=historical_result.issue_stats,
            pull_request_stats=historical_result.pull_request_stats,
            pull_request_path_evidence=path_result.pull_request_evidence,
            pull_requests_available=path_result.pull_requests_available,
            pull_requests_enriched=path_result.pull_requests_enriched,
            pull_requests_skipped_due_limit=(
                path_result.pull_requests_skipped_due_limit
            ),
            adr_file_evidence=adr_result.file_evidence,
            adr_skipped_files=adr_result.skipped_files,
            adr_stats=adr_result.stats,
            warnings=warnings,
        )
