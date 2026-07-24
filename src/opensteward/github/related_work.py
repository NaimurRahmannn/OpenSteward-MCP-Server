"""GitHub orchestration for bounded historical related-work search."""

from datetime import UTC, datetime
from typing import Any, Protocol, Self

from pydantic import (
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_serializer,
    model_validator,
)

from opensteward.github.historical_knowledge import (
    knowledge_repository_from_github,
)
from opensteward.github.historical_snapshot import (
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubHistoricalKnowledgeSnapshotResult,
)
from opensteward.github.models import GitHubRepositoryRef, StrictGitHubModel
from opensteward.knowledge import (
    MAX_KNOWLEDGE_EXACT_PHRASES,
    MAX_KNOWLEDGE_IDENTIFIERS,
    MAX_KNOWLEDGE_QUERY_COMPONENTS,
    MAX_KNOWLEDGE_QUERY_LABELS,
    MAX_KNOWLEDGE_QUERY_PATHS,
    MAX_KNOWLEDGE_QUERY_REFERENCES,
    MAX_KNOWLEDGE_QUERY_TEXT_LENGTH,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalQuery,
    KnowledgeLexicalReference,
    KnowledgeRelatedWorkMode,
    KnowledgeRelatedWorkOptions,
    KnowledgeRelatedWorkResult,
    KnowledgeRepositoryRef,
)


class GitHubRelatedWorkError(ValueError):
    """Raised when GitHub related-work orchestration results are inconsistent."""


class GitHubRelatedWorkQuery(StrictGitHubModel):
    """GitHub-facing related-work query without duplicate repository identity."""

    text: str | None = Field(
        default=None,
        max_length=MAX_KNOWLEDGE_QUERY_TEXT_LENGTH,
    )
    exact_phrases: list[str] = Field(
        default_factory=list,
        max_length=MAX_KNOWLEDGE_EXACT_PHRASES,
    )
    identifiers: list[str] = Field(
        default_factory=list,
        max_length=MAX_KNOWLEDGE_IDENTIFIERS,
    )
    labels: list[str] = Field(
        default_factory=list,
        max_length=MAX_KNOWLEDGE_QUERY_LABELS,
    )
    components: list[str] = Field(
        default_factory=list,
        max_length=MAX_KNOWLEDGE_QUERY_COMPONENTS,
    )
    affected_paths: list[str] = Field(
        default_factory=list,
        max_length=MAX_KNOWLEDGE_QUERY_PATHS,
    )
    references: list[KnowledgeLexicalReference] = Field(
        default_factory=list,
        max_length=MAX_KNOWLEDGE_QUERY_REFERENCES,
    )
    item_types: list[KnowledgeItemType] = Field(default_factory=list)
    states: list[KnowledgeItemState] = Field(default_factory=list)

    def to_knowledge_query(
        self,
        repository: KnowledgeRepositoryRef,
    ) -> KnowledgeLexicalQuery:
        """Build the authoritative provider-neutral lexical query."""

        return KnowledgeLexicalQuery(
            repository=repository,
            text=self.text,
            exact_phrases=list(self.exact_phrases),
            identifiers=list(self.identifiers),
            labels=list(self.labels),
            components=list(self.components),
            affected_paths=list(self.affected_paths),
            references=list(self.references),
            item_types=list(self.item_types),
            states=list(self.states),
        )


class GitHubRelatedWorkRequest(StrictGitHubModel):
    """Input for one live GitHub historical related-work search."""

    installation_id: int = Field(gt=0)
    repository: GitHubRepositoryRef
    git_ref: str = Field(min_length=1)
    query: GitHubRelatedWorkQuery
    snapshot_options: GitHubHistoricalKnowledgeSnapshotOptions = Field(
        default_factory=GitHubHistoricalKnowledgeSnapshotOptions
    )
    related_work_options: KnowledgeRelatedWorkOptions = Field(
        default_factory=KnowledgeRelatedWorkOptions
    )

    @field_validator("git_ref")
    @classmethod
    def normalize_git_ref(cls, git_ref: str) -> str:
        """Strip and require one non-empty Git reference."""

        normalized = git_ref.strip()
        if not normalized:
            raise ValueError("Git reference must not be empty.")
        return normalized

    @model_validator(mode="after")
    def validate_query(self) -> Self:
        """Validate provider-neutral query semantics before orchestration."""

        self.to_knowledge_query()
        return self

    def to_knowledge_query(self) -> KnowledgeLexicalQuery:
        """Build a lexical query for this request's repository."""

        return self.query.to_knowledge_query(
            knowledge_repository_from_github(self.repository)
        )


def _validate_unique_warnings(warnings: list[str], field_name: str) -> list[str]:
    if any(not warning for warning in warnings):
        raise ValueError(f"{field_name} must contain non-empty strings.")
    if len(warnings) != len(set(warnings)):
        raise ValueError(f"{field_name} must contain unique strings.")
    return warnings


def _stable_unique_warnings(*warning_groups: list[str]) -> list[str]:
    combined: list[str] = []
    seen: set[str] = set()
    for group in warning_groups:
        for warning in group:
            if warning not in seen:
                seen.add(warning)
                combined.append(warning)
    return combined


class GitHubRelatedWorkSnapshotSummary(StrictGitHubModel):
    """Concise historical snapshot provenance and coverage."""

    repository: GitHubRepositoryRef
    knowledge_repository: KnowledgeRepositoryRef
    requested_ref: str = Field(min_length=1)
    resolved_commit_sha: str = Field(min_length=1)
    adr_tree_sha: str = Field(min_length=1)
    collected_at: datetime
    adr_snapshot_commit_date: datetime
    complete: bool
    total_count: int = Field(ge=0)
    issue_count: int = Field(ge=0)
    pull_request_count: int = Field(ge=0)
    adr_count: int = Field(ge=0)
    warnings: list[str]

    @field_validator("collected_at", "adr_snapshot_commit_date")
    @classmethod
    def normalize_datetimes(cls, value: datetime) -> datetime:
        """Require aware snapshot timestamps and normalize them to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Related-work snapshot timestamps must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        """Require ordered, unique, non-empty snapshot warnings."""

        return _validate_unique_warnings(warnings, "Snapshot warnings")

    @model_validator(mode="after")
    def validate_summary(self) -> Self:
        """Validate repository identity and concise source counts."""

        if self.knowledge_repository != knowledge_repository_from_github(
            self.repository
        ):
            raise ValueError(
                "Snapshot knowledge_repository must match the GitHub repository."
            )
        if self.total_count != (
            self.issue_count + self.pull_request_count + self.adr_count
        ):
            raise ValueError("Snapshot total_count must equal its item-type counts.")
        return self


class GitHubRelatedWorkResult(StrictGitHubModel):
    """Live GitHub related-work results with independent coverage signals."""

    model_config = ConfigDict(json_schema_mode_override="serialization")

    repository: GitHubRepositoryRef
    snapshot: GitHubRelatedWorkSnapshotSummary
    related_work: KnowledgeRelatedWorkResult
    warnings: list[str]

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        """Require unique, non-empty combined warnings."""

        return _validate_unique_warnings(warnings, "Related-work warnings")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate repository, snapshot, corpus, time, and warning identity."""

        if self.snapshot.repository != self.repository:
            raise ValueError("Snapshot repository must match the result repository.")
        if self.related_work.repository != self.snapshot.knowledge_repository:
            raise ValueError(
                "Related-work repository must match the snapshot knowledge repository."
            )
        if self.related_work.query.repository != self.snapshot.knowledge_repository:
            raise ValueError(
                "Related-work query must match the snapshot knowledge repository."
            )
        if self.related_work.as_of != self.snapshot.collected_at:
            raise ValueError("Related-work as_of must match snapshot collected_at.")
        if self.related_work.corpus_total_count != self.snapshot.total_count:
            raise ValueError(
                "Related-work corpus count must match the snapshot item count."
            )

        expected_warnings = _stable_unique_warnings(
            self.snapshot.warnings,
            self.related_work.warnings,
        )
        if self.warnings != expected_warnings:
            raise ValueError(
                "Warnings must equal stable snapshot and related-work warnings."
            )
        return self

    @model_serializer(mode="wrap")
    def serialize_result(
        self,
        _handler: Any,
        info: Any,
    ) -> dict[str, Any]:
        """Serialize declared nested fields plus public top-level coverage."""

        mode = "json" if info.mode == "json" else "python"
        serialized: dict[str, Any] = {
            "repository": self.repository.model_dump(
                mode=mode,
                exclude_computed_fields=True,
            ),
            "snapshot": self.snapshot.model_dump(
                mode=mode,
                exclude_computed_fields=True,
            ),
            "related_work": self.related_work.model_dump(
                mode=mode,
                exclude_computed_fields=True,
            ),
            "warnings": list(self.warnings),
        }
        if not info.exclude_computed_fields:
            serialized.update(
                mode=self.mode,
                returned_count=self.returned_count,
                source_history_complete=self.source_history_complete,
                ranking_coverage_complete=self.ranking_coverage_complete,
                result_truncated=self.result_truncated,
                complete=self.complete,
            )
        return serialized

    @computed_field
    @property
    def mode(self) -> KnowledgeRelatedWorkMode:
        """Return the authoritative ranking mode."""

        return self.related_work.mode

    @computed_field
    @property
    def returned_count(self) -> int:
        """Return the number of related-work matches."""

        return self.related_work.returned_count

    @computed_field
    @property
    def source_history_complete(self) -> bool:
        """Return whether bounded GitHub source collection was complete."""

        return self.snapshot.complete

    @computed_field
    @property
    def ranking_coverage_complete(self) -> bool:
        """Return whether ranking considered every lexical candidate."""

        return self.related_work.complete_ranking_coverage

    @computed_field
    @property
    def result_truncated(self) -> bool:
        """Return whether the final result limit omitted ranked candidates."""

        return self.related_work.truncated

    @computed_field
    @property
    def complete(self) -> bool:
        """Return whether collection, ranking, and returned results are complete."""

        return (
            self.source_history_complete
            and self.ranking_coverage_complete
            and not self.result_truncated
        )


class HistoricalSnapshotCollector(Protocol):
    """Internal historical snapshot collection boundary."""

    async def collect(
        self,
        repository: GitHubRepositoryRef,
        *,
        git_ref: str,
        options: GitHubHistoricalKnowledgeSnapshotOptions | None = None,
    ) -> GitHubHistoricalKnowledgeSnapshotResult:
        """Collect one bounded historical knowledge snapshot."""

        ...


class RelatedWorkFinder(Protocol):
    """Internal provider-neutral related-work search boundary."""

    async def find(
        self,
        query: KnowledgeLexicalQuery,
        items: list[KnowledgeItem],
        *,
        as_of: datetime,
        options: KnowledgeRelatedWorkOptions | None = None,
    ) -> KnowledgeRelatedWorkResult:
        """Find related work in one completed snapshot."""

        ...


class GitHubRelatedWorkRunner(Protocol):
    """Public runner boundary used by the MCP capability."""

    async def find(
        self,
        request: GitHubRelatedWorkRequest,
    ) -> GitHubRelatedWorkResult:
        """Find related work using GitHub App installation authentication."""

        ...


def _validate_snapshot_result(
    snapshot: GitHubHistoricalKnowledgeSnapshotResult,
    request: GitHubRelatedWorkRequest,
    knowledge_repository: KnowledgeRepositoryRef,
) -> None:
    if snapshot.repository != request.repository:
        raise GitHubRelatedWorkError(
            "Historical snapshot belongs to another repository."
        )
    if snapshot.knowledge_repository != knowledge_repository:
        raise GitHubRelatedWorkError(
            "Historical snapshot uses another knowledge repository."
        )
    if snapshot.requested_ref != request.git_ref:
        raise GitHubRelatedWorkError(
            "Historical snapshot returned another requested Git reference."
        )


def _validate_related_work_result(
    result: KnowledgeRelatedWorkResult,
    *,
    snapshot: GitHubHistoricalKnowledgeSnapshotResult,
    query: KnowledgeLexicalQuery,
    options: KnowledgeRelatedWorkOptions,
) -> None:
    if result.repository != snapshot.knowledge_repository:
        raise GitHubRelatedWorkError(
            "Related-work result belongs to another repository."
        )
    if result.query != query:
        raise GitHubRelatedWorkError(
            "Related-work result contains another query."
        )
    if result.options != options:
        raise GitHubRelatedWorkError(
            "Related-work result contains another options model."
        )
    if result.as_of != snapshot.collected_at:
        raise GitHubRelatedWorkError(
            "Related-work as_of differs from snapshot collected_at."
        )
    if result.corpus_total_count != snapshot.total_count:
        raise GitHubRelatedWorkError(
            "Related-work corpus count differs from snapshot item count."
        )


def _build_snapshot_summary(
    snapshot: GitHubHistoricalKnowledgeSnapshotResult,
) -> GitHubRelatedWorkSnapshotSummary:
    return GitHubRelatedWorkSnapshotSummary(
        repository=snapshot.repository,
        knowledge_repository=snapshot.knowledge_repository,
        requested_ref=snapshot.requested_ref,
        resolved_commit_sha=snapshot.resolved_adr_commit_sha,
        adr_tree_sha=snapshot.adr_tree_sha,
        collected_at=snapshot.collected_at,
        adr_snapshot_commit_date=snapshot.adr_snapshot_commit_date,
        complete=snapshot.complete,
        total_count=snapshot.total_count,
        issue_count=snapshot.issue_count,
        pull_request_count=snapshot.pull_request_count,
        adr_count=snapshot.adr_count,
        warnings=list(snapshot.warnings),
    )


class GitHubRelatedWorkService:
    """Connect one GitHub historical snapshot to related-work ranking."""

    def __init__(
        self,
        *,
        snapshot_collector: HistoricalSnapshotCollector,
        related_work_finder: RelatedWorkFinder,
    ) -> None:
        self._snapshot_collector = snapshot_collector
        self._related_work_finder = related_work_finder

    async def find(
        self,
        request: GitHubRelatedWorkRequest,
    ) -> GitHubRelatedWorkResult:
        """Collect a snapshot and find related work without mutating GitHub."""

        query = request.to_knowledge_query()
        snapshot = await self._snapshot_collector.collect(
            request.repository,
            git_ref=request.git_ref,
            options=request.snapshot_options,
        )
        _validate_snapshot_result(snapshot, request, query.repository)

        related_work = await self._related_work_finder.find(
            query,
            snapshot.items,
            as_of=snapshot.collected_at,
            options=request.related_work_options,
        )
        _validate_related_work_result(
            related_work,
            snapshot=snapshot,
            query=query,
            options=request.related_work_options,
        )

        summary = _build_snapshot_summary(snapshot)
        warnings = _stable_unique_warnings(
            summary.warnings,
            related_work.warnings,
        )
        return GitHubRelatedWorkResult(
            repository=request.repository,
            snapshot=summary,
            related_work=related_work,
            warnings=warnings,
        )
