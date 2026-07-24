"""Tests for GitHub historical related-work models and orchestration."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.github import (
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubHistoricalKnowledgeSnapshotResult,
    GitHubRelatedWorkError,
    GitHubRelatedWorkQuery,
    GitHubRelatedWorkRequest,
    GitHubRelatedWorkResult,
    GitHubRelatedWorkService,
    GitHubRelatedWorkSnapshotSummary,
    GitHubRepositoryRef,
    knowledge_repository_from_github,
)
from opensteward.knowledge import (
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalQuery,
    KnowledgeLexicalReference,
    KnowledgeRelatedWorkMode,
    KnowledgeRelatedWorkOptions,
    KnowledgeRelatedWorkResult,
    KnowledgeRelatedWorkService,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

REPOSITORY = GitHubRepositoryRef(owner="acme", name="framework")
OTHER_REPOSITORY = GitHubRepositoryRef(owner="other", name="project")
KNOWLEDGE_REPOSITORY = knowledge_repository_from_github(REPOSITORY)
OTHER_KNOWLEDGE_REPOSITORY = knowledge_repository_from_github(OTHER_REPOSITORY)
COLLECTED_AT = datetime(2026, 5, 1, 10, 0, tzinfo=UTC)
COMMIT_DATE = datetime(2026, 4, 30, 9, 0, tzinfo=UTC)
FALLBACK_WARNING = (
    "Lexical candidate retrieval reached its safety limit; semantic and hybrid "
    "ranking were skipped to avoid incomplete score fusion."
)


def make_item(
    external_id: str = "1",
    *,
    item_type: KnowledgeItemType = KnowledgeItemType.ISSUE,
    title: str = "Adopt parser registry",
    repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
) -> KnowledgeItem:
    """Build one valid historical item."""

    source_kind = (
        KnowledgeSourceKind.REPOSITORY_FILE
        if item_type == KnowledgeItemType.ADR
        else KnowledgeSourceKind.GITHUB
    )
    return KnowledgeItem(
        repository=repository,
        item_type=item_type,
        external_id=external_id,
        source_kind=source_kind,
        state=KnowledgeItemState.CLOSED,
        title=title,
        body="Parser architecture and registry details.",
        summary="Parser registry decision.",
        url=f"https://github.test/acme/framework/{external_id}",
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
        updated_at=datetime(2026, 2, 1, tzinfo=UTC),
        closed_at=datetime(2026, 2, 1, tzinfo=UTC),
        labels=["architecture"],
        affected_paths=["src/parser/registry.py"],
        components=["parser"],
        decision_significance=DecisionSignificance.HIGH,
    )


def make_snapshot(
    items: list[KnowledgeItem] | None = None,
    *,
    warnings: list[str] | None = None,
    source_complete: bool = True,
) -> GitHubHistoricalKnowledgeSnapshotResult:
    """Build a narrow snapshot double with real computed snapshot fields."""

    source_stats = SimpleNamespace(
        item_limit_reached=not source_complete,
        scan_limit_reached=False,
    )
    adr_stats = SimpleNamespace(
        tree_truncated=False,
        item_limit_reached=False,
        total_bytes_limit_reached=False,
    )
    return GitHubHistoricalKnowledgeSnapshotResult.model_construct(
        repository=REPOSITORY,
        knowledge_repository=KNOWLEDGE_REPOSITORY,
        collected_at=COLLECTED_AT,
        requested_ref="main",
        resolved_adr_commit_sha="a" * 40,
        adr_tree_sha="b" * 40,
        adr_snapshot_commit_date=COMMIT_DATE,
        items=list(items if items is not None else [make_item()]),
        issue_stats=source_stats,
        pull_request_stats=SimpleNamespace(
            item_limit_reached=False,
            scan_limit_reached=False,
        ),
        pull_request_path_evidence=[],
        pull_requests_available=0,
        pull_requests_enriched=0,
        pull_requests_skipped_due_limit=0,
        adr_file_evidence=[],
        adr_skipped_files=[],
        adr_stats=adr_stats,
        warnings=list(warnings or []),
    )


def make_request(**updates: Any) -> GitHubRelatedWorkRequest:
    """Build one valid public request."""

    payload: dict[str, Any] = {
        "installation_id": 17,
        "repository": REPOSITORY,
        "git_ref": "main",
        "query": GitHubRelatedWorkQuery(
            text="parser",
            affected_paths=["src/parser/service.py"],
        ),
    }
    payload.update(updates)
    return GitHubRelatedWorkRequest(**payload)


async def make_related_result(
    snapshot: GitHubHistoricalKnowledgeSnapshotResult,
    request: GitHubRelatedWorkRequest,
) -> KnowledgeRelatedWorkResult:
    """Run the real deterministic related-work service for a snapshot."""

    return await KnowledgeRelatedWorkService().find(
        request.to_knowledge_query(),
        snapshot.items,
        as_of=snapshot.collected_at,
        options=request.related_work_options,
    )


def make_summary(
    *,
    source_complete: bool = True,
    total_count: int = 1,
    warnings: list[str] | None = None,
) -> GitHubRelatedWorkSnapshotSummary:
    """Build one concise snapshot summary."""

    return GitHubRelatedWorkSnapshotSummary(
        repository=REPOSITORY,
        knowledge_repository=KNOWLEDGE_REPOSITORY,
        requested_ref="main",
        resolved_commit_sha="a" * 40,
        adr_tree_sha="b" * 40,
        collected_at=COLLECTED_AT,
        adr_snapshot_commit_date=COMMIT_DATE,
        complete=source_complete,
        total_count=total_count,
        issue_count=total_count,
        pull_request_count=0,
        adr_count=0,
        warnings=warnings or [],
    )


class RecordingSnapshotCollector:
    """Injected snapshot collector with exact call recording."""

    def __init__(
        self,
        outcome: GitHubHistoricalKnowledgeSnapshotResult | Exception,
        events: list[str] | None = None,
    ) -> None:
        self.outcome = outcome
        self.events = events if events is not None else []
        self.calls: list[tuple[GitHubRepositoryRef, str, object]] = []

    async def collect(
        self,
        repository: GitHubRepositoryRef,
        *,
        git_ref: str,
        options: GitHubHistoricalKnowledgeSnapshotOptions | None = None,
    ) -> GitHubHistoricalKnowledgeSnapshotResult:
        self.events.append("snapshot")
        self.calls.append((repository, git_ref, options))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class RecordingRelatedWorkFinder:
    """Injected related-work finder with exact call recording."""

    def __init__(
        self,
        outcome: KnowledgeRelatedWorkResult | Exception,
        events: list[str] | None = None,
    ) -> None:
        self.outcome = outcome
        self.events = events if events is not None else []
        self.calls: list[tuple[KnowledgeLexicalQuery, list[KnowledgeItem], datetime, object]] = []

    async def find(
        self,
        query: KnowledgeLexicalQuery,
        items: list[KnowledgeItem],
        *,
        as_of: datetime,
        options: KnowledgeRelatedWorkOptions | None = None,
    ) -> KnowledgeRelatedWorkResult:
        self.events.append("related")
        self.calls.append((query, items, as_of, options))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def test_query_wrapper_accepts_every_signal_and_preserves_domain_normalization() -> None:
    query = GitHubRelatedWorkQuery(
        text="  Parser   Registry  ",
        exact_phrases=["  parser   registry "],
        identifiers=["ParserRegistry"],
        labels=["architecture"],
        components=["parser"],
        affected_paths=["./src\\parser\\registry.py"],
        references=[
            KnowledgeLexicalReference(
                item_type=KnowledgeItemType.ISSUE,
                external_id="17",
            )
        ],
        item_types=[KnowledgeItemType.ISSUE],
        states=[KnowledgeItemState.CLOSED],
    )

    knowledge_query = query.to_knowledge_query(KNOWLEDGE_REPOSITORY)

    assert knowledge_query.repository == KNOWLEDGE_REPOSITORY
    assert knowledge_query.text == "Parser   Registry"
    assert knowledge_query.normalized_text == "parser registry"
    assert knowledge_query.exact_phrases == ["parser registry"]
    assert knowledge_query.identifiers == ["ParserRegistry"]
    assert knowledge_query.labels == ["architecture"]
    assert knowledge_query.components == ["parser"]
    assert knowledge_query.affected_paths == ["src/parser/registry.py"]
    assert knowledge_query.references == query.references
    assert knowledge_query.item_types == [KnowledgeItemType.ISSUE]
    assert knowledge_query.states == [KnowledgeItemState.CLOSED]


def test_query_wrapper_rejects_extra_fields_and_preserves_domain_validation() -> None:
    with pytest.raises(ValidationError):
        GitHubRelatedWorkQuery(text="parser", repository=REPOSITORY)

    with pytest.raises(ValidationError, match="repository-relative"):
        GitHubRelatedWorkQuery(
            affected_paths=["/absolute/path.py"]
        ).to_knowledge_query(KNOWLEDGE_REPOSITORY)

    with pytest.raises(ValidationError, match="unique"):
        GitHubRelatedWorkQuery(
            labels=["Parser", "parser"]
        ).to_knowledge_query(KNOWLEDGE_REPOSITORY)


def test_request_validates_identity_ref_defaults_and_query_before_use() -> None:
    request = make_request(git_ref="  refs/heads/main  ")

    assert request.installation_id == 17
    assert request.git_ref == "refs/heads/main"
    assert request.snapshot_options == GitHubHistoricalKnowledgeSnapshotOptions()
    assert request.related_work_options == KnowledgeRelatedWorkOptions()
    assert request.to_knowledge_query().repository == KNOWLEDGE_REPOSITORY

    with pytest.raises(ValidationError):
        make_request(installation_id=0)
    with pytest.raises(ValidationError, match="at least 1 character"):
        make_request(git_ref=" \t ")
    with pytest.raises(ValidationError, match="at least one search signal"):
        make_request(query=GitHubRelatedWorkQuery())


def test_snapshot_summary_validates_identity_counts_warnings_and_utc() -> None:
    non_utc = timezone(timedelta(hours=6))
    summary = GitHubRelatedWorkSnapshotSummary(
        repository=REPOSITORY,
        knowledge_repository=KNOWLEDGE_REPOSITORY,
        requested_ref="main",
        resolved_commit_sha="commit",
        adr_tree_sha="tree",
        collected_at=COLLECTED_AT.astimezone(non_utc),
        adr_snapshot_commit_date=COMMIT_DATE.astimezone(non_utc),
        complete=True,
        total_count=3,
        issue_count=1,
        pull_request_count=1,
        adr_count=1,
        warnings=["bounded"],
    )
    assert summary.collected_at == COLLECTED_AT
    assert summary.collected_at.tzinfo is UTC

    with pytest.raises(ValidationError, match="knowledge_repository"):
        summary.model_copy(
            update={"knowledge_repository": OTHER_KNOWLEDGE_REPOSITORY}
        ).model_validate(
            summary.model_dump(
                exclude={"knowledge_repository"},
                exclude_computed_fields=True,
            )
            | {"knowledge_repository": OTHER_KNOWLEDGE_REPOSITORY}
        )
    with pytest.raises(ValidationError, match="total_count"):
        GitHubRelatedWorkSnapshotSummary(
            **(
                summary.model_dump(
                    exclude={"total_count"},
                    exclude_computed_fields=True,
                )
                | {"total_count": 4}
            )
        )
    for warnings in ([""], ["same", "same"]):
        with pytest.raises(ValidationError):
            GitHubRelatedWorkSnapshotSummary(
                **(
                    summary.model_dump(
                        exclude={"warnings"},
                        exclude_computed_fields=True,
                    )
                    | {"warnings": warnings}
                )
            )
    with pytest.raises(ValidationError, match="timezone-aware"):
        GitHubRelatedWorkSnapshotSummary(
            **(
                summary.model_dump(
                    exclude={"collected_at"},
                    exclude_computed_fields=True,
                )
                | {"collected_at": datetime(2026, 5, 1)}
            )
        )


@pytest.mark.asyncio
async def test_result_validates_identity_time_count_and_exact_warning_order() -> None:
    request = make_request()
    snapshot = make_snapshot(warnings=["source warning"])
    related = await make_related_result(snapshot, request)
    summary = make_summary(warnings=["source warning"])
    result = GitHubRelatedWorkResult(
        repository=REPOSITORY,
        snapshot=summary,
        related_work=related,
        warnings=["source warning"],
    )
    assert result.mode == KnowledgeRelatedWorkMode.DETERMINISTIC

    invalid_cases = [
        {"repository": OTHER_REPOSITORY},
        {"snapshot": summary.model_copy(update={"repository": OTHER_REPOSITORY})},
        {
            "related_work": related.model_copy(
                update={"repository": OTHER_KNOWLEDGE_REPOSITORY}
            )
        },
        {
            "related_work": related.model_copy(
                update={
                    "query": related.query.model_copy(
                        update={"repository": OTHER_KNOWLEDGE_REPOSITORY}
                    )
                }
            )
        },
        {
            "related_work": related.model_copy(
                update={"as_of": COLLECTED_AT + timedelta(seconds=1)}
            )
        },
        {
            "related_work": related.model_copy(
                update={"corpus_total_count": 2}
            )
        },
        {"warnings": []},
        {"warnings": ["source warning", "extra"]},
    ]
    payload = result.model_dump(exclude_computed_fields=True)
    for update in invalid_cases:
        with pytest.raises(ValidationError):
            GitHubRelatedWorkResult(**(payload | update))


@pytest.mark.asyncio
async def test_result_coverage_completeness_requires_all_three_dimensions() -> None:
    request = make_request()
    snapshot = make_snapshot()
    related = await make_related_result(snapshot, request)

    complete = GitHubRelatedWorkResult(
        repository=REPOSITORY,
        snapshot=make_summary(),
        related_work=related,
        warnings=[],
    )
    assert complete.source_history_complete is True
    assert complete.ranking_coverage_complete is True
    assert complete.result_truncated is False
    assert complete.returned_count == 1
    assert complete.complete is True

    incomplete_source = complete.model_copy(
        update={"snapshot": complete.snapshot.model_copy(update={"complete": False})}
    )
    assert incomplete_source.complete is False

    two_items = [make_item("1"), make_item("2")]
    limited_request = make_request(
        related_work_options=KnowledgeRelatedWorkOptions(max_results=1)
    )
    limited_snapshot = make_snapshot(two_items)
    limited_related = await make_related_result(limited_snapshot, limited_request)
    truncated = GitHubRelatedWorkResult(
        repository=REPOSITORY,
        snapshot=make_summary(total_count=2),
        related_work=limited_related,
        warnings=[],
    )
    assert truncated.result_truncated is True
    assert truncated.complete is False

    fallback_items = [
        make_item(str(index), title=f"Parser registry history {index}")
        for index in range(101)
    ]
    fallback_snapshot = make_snapshot(fallback_items)
    fallback_related = await make_related_result(fallback_snapshot, request)
    fallback = GitHubRelatedWorkResult(
        repository=REPOSITORY,
        snapshot=make_summary(total_count=101),
        related_work=fallback_related,
        warnings=[FALLBACK_WARNING],
    )
    assert fallback.mode == KnowledgeRelatedWorkMode.LEXICAL_FALLBACK
    assert fallback.ranking_coverage_complete is False
    assert fallback.complete is False


@pytest.mark.asyncio
async def test_json_serialization_preserves_snapshot_provenance_and_evidence() -> None:
    request = make_request()
    snapshot = make_snapshot()
    related = await make_related_result(snapshot, request)
    result = GitHubRelatedWorkResult(
        repository=REPOSITORY,
        snapshot=make_summary(),
        related_work=related,
        warnings=[],
    )

    data = result.model_dump(mode="json")

    assert data["snapshot"]["requested_ref"] == "main"
    assert data["snapshot"]["resolved_commit_sha"] == "a" * 40
    assert data["snapshot"]["adr_tree_sha"] == "b" * 40
    assert data["related_work"]["matches"][0]["lexical_match"]["evidence"]
    assert "installation_id" not in data
    assert "installation_id" not in str(data)


@pytest.mark.asyncio
async def test_service_sequence_arguments_results_provenance_and_immutability() -> None:
    events: list[str] = []
    request = make_request(
        snapshot_options=GitHubHistoricalKnowledgeSnapshotOptions(),
        related_work_options=KnowledgeRelatedWorkOptions(max_results=7),
    )
    snapshot = make_snapshot(warnings=["source warning"])
    related = await make_related_result(snapshot, request)
    request_before = deepcopy(request.model_dump())
    snapshot_before = deepcopy(snapshot.model_dump())
    related_before = deepcopy(related.model_dump())
    collector = RecordingSnapshotCollector(snapshot, events)
    finder = RecordingRelatedWorkFinder(related, events)

    result = await GitHubRelatedWorkService(
        snapshot_collector=collector,
        related_work_finder=finder,
    ).find(request)

    assert events == ["snapshot", "related"]
    assert collector.calls == [
        (request.repository, request.git_ref, request.snapshot_options)
    ]
    query, passed_items, as_of, options = finder.calls[0]
    assert query == request.to_knowledge_query()
    assert passed_items is snapshot.items
    assert as_of == snapshot.collected_at
    assert options is request.related_work_options
    assert result.related_work is related
    assert result.mode == KnowledgeRelatedWorkMode.DETERMINISTIC
    assert result.snapshot.resolved_commit_sha == snapshot.resolved_adr_commit_sha
    assert result.snapshot.adr_tree_sha == snapshot.adr_tree_sha
    assert result.snapshot.adr_snapshot_commit_date == snapshot.adr_snapshot_commit_date
    assert result.snapshot.warnings == snapshot.warnings
    assert result.snapshot.warnings is not snapshot.warnings
    assert result.related_work.matches[0].lexical_match == related.matches[0].lexical_match
    assert request.model_dump() == request_before
    assert snapshot.model_dump() == snapshot_before
    assert related.model_dump() == related_before


@pytest.mark.asyncio
async def test_service_preserves_lexical_fallback_and_stably_deduplicates_warnings() -> None:
    items = [
        make_item(str(index), title=f"Parser registry history {index}")
        for index in range(101)
    ]
    request = make_request()
    snapshot = make_snapshot(
        items,
        warnings=["source bounded", FALLBACK_WARNING],
    )
    related = await make_related_result(snapshot, request)
    result = await GitHubRelatedWorkService(
        snapshot_collector=RecordingSnapshotCollector(snapshot),
        related_work_finder=RecordingRelatedWorkFinder(related),
    ).find(request)

    assert result.related_work is related
    assert result.mode == KnowledgeRelatedWorkMode.LEXICAL_FALLBACK
    assert result.warnings == ["source bounded", FALLBACK_WARNING]


class SentinelError(RuntimeError):
    """Distinct dependency failure used to prove error propagation."""


@pytest.mark.asyncio
async def test_dependency_errors_propagate_and_snapshot_failure_stops_finder() -> None:
    request = make_request()
    snapshot_error = SentinelError("snapshot failed")
    finder = RecordingRelatedWorkFinder(SentinelError("must not run"))
    with pytest.raises(SentinelError, match="snapshot failed"):
        await GitHubRelatedWorkService(
            snapshot_collector=RecordingSnapshotCollector(snapshot_error),
            related_work_finder=finder,
        ).find(request)
    assert finder.calls == []

    snapshot = make_snapshot()
    with pytest.raises(SentinelError, match="related failed"):
        await GitHubRelatedWorkService(
            snapshot_collector=RecordingSnapshotCollector(snapshot),
            related_work_finder=RecordingRelatedWorkFinder(
                SentinelError("related failed")
            ),
        ).find(request)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot_update", "message"),
    [
        (
            {"repository": OTHER_REPOSITORY},
            "Historical snapshot belongs to another repository.",
        ),
        (
            {"knowledge_repository": OTHER_KNOWLEDGE_REPOSITORY},
            "Historical snapshot uses another knowledge repository.",
        ),
        (
            {"requested_ref": "other"},
            "Historical snapshot returned another requested Git reference.",
        ),
    ],
)
async def test_service_rejects_snapshot_consistency_failures(
    snapshot_update: dict[str, object],
    message: str,
) -> None:
    request = make_request()
    snapshot = make_snapshot().model_copy(update=snapshot_update)
    finder = RecordingRelatedWorkFinder(SentinelError("must not run"))

    with pytest.raises(GitHubRelatedWorkError, match=message):
        await GitHubRelatedWorkService(
            snapshot_collector=RecordingSnapshotCollector(snapshot),
            related_work_finder=finder,
        ).find(request)
    assert finder.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result_update", "message"),
    [
        (
            {"repository": OTHER_KNOWLEDGE_REPOSITORY},
            "Related-work result belongs to another repository.",
        ),
        (
            {
                "query": KnowledgeLexicalQuery(
                    repository=KNOWLEDGE_REPOSITORY,
                    text="another query",
                )
            },
            "Related-work result contains another query.",
        ),
        (
            {"options": KnowledgeRelatedWorkOptions(max_results=1)},
            "Related-work result contains another options model.",
        ),
        (
            {"as_of": COLLECTED_AT + timedelta(seconds=1)},
            "Related-work as_of differs from snapshot collected_at.",
        ),
        (
            {"corpus_total_count": 2},
            "Related-work corpus count differs from snapshot item count.",
        ),
    ],
)
async def test_service_rejects_related_work_consistency_failures(
    result_update: dict[str, object],
    message: str,
) -> None:
    request = make_request()
    snapshot = make_snapshot()
    related = (await make_related_result(snapshot, request)).model_copy(
        update=result_update
    )

    with pytest.raises(GitHubRelatedWorkError, match=message):
        await GitHubRelatedWorkService(
            snapshot_collector=RecordingSnapshotCollector(snapshot),
            related_work_finder=RecordingRelatedWorkFinder(related),
        ).find(request)
