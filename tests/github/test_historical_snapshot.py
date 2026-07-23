"""Tests for unified historical GitHub knowledge snapshots."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.github import (
    MAX_GITHUB_HISTORICAL_SCAN_PAGES,
    GitHubHistoricalAdrCollectionOptions,
    GitHubHistoricalAdrCollectionResult,
    GitHubHistoricalAdrCollectionStats,
    GitHubHistoricalAdrFileEvidence,
    GitHubHistoricalAdrSkippedFile,
    GitHubHistoricalAdrSkipReason,
    GitHubHistoricalKnowledgeCollectionOptions,
    GitHubHistoricalKnowledgeCollectionResult,
    GitHubHistoricalKnowledgeCollectionStats,
    GitHubHistoricalKnowledgeSnapshotError,
    GitHubHistoricalKnowledgeSnapshotOptions,
    GitHubHistoricalKnowledgeSnapshotResult,
    GitHubHistoricalKnowledgeSnapshotService,
    GitHubHistoricalPathEnrichmentOptions,
    GitHubHistoricalPathEnrichmentResult,
    GitHubHistoricalPullRequestPathEvidence,
    GitHubRepositoryRef,
    knowledge_repository_from_github,
)
from opensteward.knowledge import (
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

CREATED_AT = datetime(2026, 4, 1, 9, 0, tzinfo=UTC)
ADR_UPDATED_AT = datetime(2026, 4, 2, 9, 0, tzinfo=UTC)
ISSUE_UPDATED_AT = datetime(2026, 4, 3, 9, 0, tzinfo=UTC)
PULL_REQUEST_UPDATED_AT = datetime(2026, 4, 4, 9, 0, tzinfo=UTC)
COLLECTED_AT = datetime(2026, 4, 5, 9, 0, tzinfo=UTC)
ADR_COMMIT_SHA = "snapshot-commit"
ADR_TREE_SHA = "snapshot-tree"
REPOSITORY = GitHubRepositoryRef(owner="acme", name="framework")
KNOWLEDGE_REPOSITORY = knowledge_repository_from_github(REPOSITORY)
OTHER_REPOSITORY = GitHubRepositoryRef(owner="other", name="project")
OTHER_KNOWLEDGE_REPOSITORY = knowledge_repository_from_github(OTHER_REPOSITORY)

ISSUE_ITEM_LIMIT_WARNING = (
    "Historical issue collection reached its configured item limit."
)
ISSUE_SCAN_LIMIT_WARNING = (
    "Historical issue collection reached its scan-page safety limit."
)
PULL_REQUEST_ITEM_LIMIT_WARNING = (
    "Historical pull-request collection reached its configured item limit."
)
PULL_REQUEST_SCAN_LIMIT_WARNING = (
    "Historical pull-request collection reached its scan-page safety limit."
)
PATH_SKIPPED_WARNING = (
    "Changed-path evidence was not collected for all historical pull requests."
)
PATH_INCOMPLETE_WARNING = (
    "Changed-path evidence is incomplete for one or more historical pull requests."
)
ADR_ITEM_LIMIT_WARNING = (
    "Historical ADR collection reached its configured item limit."
)
ADR_SKIPPED_WARNING = (
    "Historical ADR collection skipped one or more candidate files."
)
ADR_TIMESTAMP_WARNING = (
    "ADR created_at and updated_at use the repository snapshot commit time; "
    "per-file history was not collected."
)


def knowledge_item(
    item_type: KnowledgeItemType,
    external_id: str,
    *,
    updated_at: datetime,
    repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
    source_kind: KnowledgeSourceKind | None = None,
    affected_paths: list[str] | None = None,
) -> KnowledgeItem:
    """Create one historical knowledge item."""

    selected_source = source_kind or (
        KnowledgeSourceKind.REPOSITORY_FILE
        if item_type == KnowledgeItemType.ADR
        else KnowledgeSourceKind.GITHUB
    )
    url = (
        f"https://github.com/acme/framework/blob/{ADR_COMMIT_SHA}/{external_id}"
        if item_type == KnowledgeItemType.ADR
        else f"https://github.test/acme/framework/items/{external_id}"
    )
    return KnowledgeItem(
        repository=repository,
        item_type=item_type,
        external_id=external_id,
        source_kind=selected_source,
        state=KnowledgeItemState.UNKNOWN,
        title=f"Historical {item_type.value} {external_id}",
        body=f"Body for {external_id}",
        url=url,
        created_at=CREATED_AT,
        updated_at=updated_at,
        affected_paths=affected_paths or [],
    )


def replace_item(item: KnowledgeItem, **updates: object) -> KnowledgeItem:
    """Replace validated item fields without mutating the source item."""

    payload = {
        field_name: getattr(item, field_name)
        for field_name in KnowledgeItem.model_fields
    }
    payload.update(updates)
    return KnowledgeItem.model_validate(payload)


def collection_stats(
    items_collected: int,
    *,
    item_limit: bool = False,
    scan_limit: bool = False,
) -> GitHubHistoricalKnowledgeCollectionStats:
    """Create internally consistent issue or pull-request statistics."""

    requested_limit = (
        items_collected
        if item_limit
        else max(items_collected + 1, 1)
    )
    return GitHubHistoricalKnowledgeCollectionStats(
        requested_limit=requested_limit,
        pages_fetched=(
            MAX_GITHUB_HISTORICAL_SCAN_PAGES
            if scan_limit
            else (1 if items_collected else 0)
        ),
        api_items_seen=items_collected,
        items_collected=items_collected,
        filtered_items=0,
        duplicate_items=0,
        item_limit_reached=item_limit,
        scan_limit_reached=scan_limit,
    )


def historical_result(
    items: list[KnowledgeItem],
    *,
    repository: GitHubRepositoryRef = REPOSITORY,
    knowledge_repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
    issue_stats: GitHubHistoricalKnowledgeCollectionStats | None = None,
    pull_request_stats: GitHubHistoricalKnowledgeCollectionStats | None = None,
) -> GitHubHistoricalKnowledgeCollectionResult:
    """Create a validated historical issue and pull-request result."""

    issue_count = sum(
        item.item_type == KnowledgeItemType.ISSUE
        for item in items
    )
    pull_request_count = sum(
        item.item_type == KnowledgeItemType.PULL_REQUEST
        for item in items
    )
    return GitHubHistoricalKnowledgeCollectionResult(
        repository=repository,
        knowledge_repository=knowledge_repository,
        items=items,
        issue_stats=issue_stats or collection_stats(issue_count),
        pull_request_stats=(
            pull_request_stats
            or collection_stats(pull_request_count)
        ),
    )


def path_evidence(
    item: KnowledgeItem,
    *,
    complete: bool = True,
) -> GitHubHistoricalPullRequestPathEvidence:
    """Create changed-path evidence for one pull request."""

    return GitHubHistoricalPullRequestPathEvidence(
        item_key=item.key,
        pull_number=int(item.external_id),
        pages_fetched=1,
        api_files_seen=len(item.affected_paths),
        affected_paths_collected=len(item.affected_paths),
        complete=complete,
        file_limit_reached=False,
    )


def path_result(
    items: list[KnowledgeItem],
    *,
    evidence: list[GitHubHistoricalPullRequestPathEvidence] | None = None,
    repository: GitHubRepositoryRef = REPOSITORY,
    knowledge_repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
) -> GitHubHistoricalPathEnrichmentResult:
    """Create a validated changed-path enrichment result."""

    pull_requests = [
        item
        for item in items
        if item.item_type == KnowledgeItemType.PULL_REQUEST
    ]
    selected_evidence = (
        [path_evidence(item) for item in pull_requests]
        if evidence is None
        else evidence
    )
    return GitHubHistoricalPathEnrichmentResult(
        repository=repository,
        knowledge_repository=knowledge_repository,
        items=items,
        pull_request_evidence=selected_evidence,
        pull_requests_available=len(pull_requests),
        pull_requests_enriched=len(selected_evidence),
        pull_requests_skipped_due_limit=(
            len(pull_requests) - len(selected_evidence)
        ),
    )


def skipped_adr(
    path: str = "docs/adr/skipped.md",
    *,
    reason: GitHubHistoricalAdrSkipReason = (
        GitHubHistoricalAdrSkipReason.FILE_TOO_LARGE
    ),
) -> GitHubHistoricalAdrSkippedFile:
    """Create one skipped ADR candidate."""

    return GitHubHistoricalAdrSkippedFile(
        path=path,
        blob_sha=f"sha-{path}",
        reported_size_bytes=100,
        reason=reason,
    )


def adr_stats(
    item_count: int,
    *,
    skipped_count: int = 0,
    tree_truncated: bool = False,
    item_limit: bool = False,
    total_limit: bool = False,
) -> GitHubHistoricalAdrCollectionStats:
    """Create internally consistent ADR statistics."""

    selected = item_count + skipped_count
    candidates = selected + int(item_limit)
    return GitHubHistoricalAdrCollectionStats(
        tree_entries_seen=candidates,
        candidate_files_seen=candidates,
        selected_files=selected,
        blobs_fetched=item_count,
        items_collected=item_count,
        skipped_files=skipped_count,
        decoded_bytes=item_count,
        tree_truncated=tree_truncated,
        item_limit_reached=item_limit,
        total_bytes_limit_reached=total_limit,
    )


def adr_evidence(
    item: KnowledgeItem,
    *,
    path: str | None = None,
) -> GitHubHistoricalAdrFileEvidence:
    """Create immutable file evidence for one ADR item."""

    evidence_path = path or item.external_id
    return GitHubHistoricalAdrFileEvidence(
        item_key=item.key,
        path=evidence_path,
        blob_sha=f"sha-{evidence_path}",
        size_bytes=len((item.body or "").encode("utf-8")),
        html_url=item.url or "https://github.test/missing-url",
    )


def adr_result(
    items: list[KnowledgeItem],
    *,
    skipped_files: list[GitHubHistoricalAdrSkippedFile] | None = None,
    repository: GitHubRepositoryRef = REPOSITORY,
    knowledge_repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
    requested_ref: str = "main",
    tree_truncated: bool = False,
    item_limit: bool = False,
    total_limit: bool = False,
    warnings: list[str] | None = None,
) -> GitHubHistoricalAdrCollectionResult:
    """Create a validated historical ADR result."""

    ordered_items = sorted(items, key=lambda item: item.external_id)
    selected_skips = skipped_files or []
    return GitHubHistoricalAdrCollectionResult(
        repository=repository,
        knowledge_repository=knowledge_repository,
        requested_ref=requested_ref,
        resolved_commit_sha=ADR_COMMIT_SHA,
        snapshot_commit_date=ADR_UPDATED_AT,
        tree_sha=ADR_TREE_SHA,
        items=ordered_items,
        file_evidence=[adr_evidence(item) for item in ordered_items],
        skipped_files=selected_skips,
        stats=adr_stats(
            len(ordered_items),
            skipped_count=len(selected_skips),
            tree_truncated=tree_truncated,
            item_limit=item_limit,
            total_limit=total_limit,
        ),
        warnings=warnings or [],
    )


def sorted_items(items: list[KnowledgeItem]) -> list[KnowledgeItem]:
    """Sort items using the snapshot's documented ordering."""

    key_sorted = sorted(items, key=lambda item: item.key)
    return sorted(
        key_sorted,
        key=lambda item: item.updated_at,
        reverse=True,
    )


def direct_snapshot(
    items: list[KnowledgeItem],
    *,
    repository: GitHubRepositoryRef = REPOSITORY,
    knowledge_repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
    collected_at: datetime = COLLECTED_AT,
    adr_snapshot_commit_date: datetime = ADR_UPDATED_AT,
    issue_stats: GitHubHistoricalKnowledgeCollectionStats | None = None,
    pull_request_stats: GitHubHistoricalKnowledgeCollectionStats | None = None,
    path_evidence_items: list[
        GitHubHistoricalPullRequestPathEvidence
    ] | None = None,
    pull_requests_available: int | None = None,
    pull_requests_enriched: int | None = None,
    pull_requests_skipped_due_limit: int | None = None,
    adr_file_evidence: list[GitHubHistoricalAdrFileEvidence] | None = None,
    adr_skipped_files: list[GitHubHistoricalAdrSkippedFile] | None = None,
    adr_stats_value: GitHubHistoricalAdrCollectionStats | None = None,
    warnings: list[str] | None = None,
) -> GitHubHistoricalKnowledgeSnapshotResult:
    """Construct a snapshot directly for public-model validation tests."""

    issue_count = sum(
        item.item_type == KnowledgeItemType.ISSUE
        for item in items
    )
    pull_requests = [
        item
        for item in items
        if item.item_type == KnowledgeItemType.PULL_REQUEST
    ]
    adrs = [
        item
        for item in items
        if item.item_type == KnowledgeItemType.ADR
    ]
    selected_path_evidence = (
        [path_evidence(item) for item in pull_requests]
        if path_evidence_items is None
        else path_evidence_items
    )
    selected_adr_evidence = (
        [adr_evidence(item) for item in adrs]
        if adr_file_evidence is None
        else adr_file_evidence
    )
    selected_skips = adr_skipped_files or []
    available = (
        len(pull_requests)
        if pull_requests_available is None
        else pull_requests_available
    )
    enriched = (
        len(selected_path_evidence)
        if pull_requests_enriched is None
        else pull_requests_enriched
    )
    skipped_paths = (
        available - enriched
        if pull_requests_skipped_due_limit is None
        else pull_requests_skipped_due_limit
    )
    return GitHubHistoricalKnowledgeSnapshotResult(
        repository=repository,
        knowledge_repository=knowledge_repository,
        collected_at=collected_at,
        requested_ref="main",
        resolved_adr_commit_sha=ADR_COMMIT_SHA,
        adr_tree_sha=ADR_TREE_SHA,
        adr_snapshot_commit_date=adr_snapshot_commit_date,
        items=items,
        issue_stats=issue_stats or collection_stats(issue_count),
        pull_request_stats=(
            pull_request_stats
            or collection_stats(len(pull_requests))
        ),
        pull_request_path_evidence=selected_path_evidence,
        pull_requests_available=available,
        pull_requests_enriched=enriched,
        pull_requests_skipped_due_limit=skipped_paths,
        adr_file_evidence=selected_adr_evidence,
        adr_skipped_files=selected_skips,
        adr_stats=(
            adr_stats_value
            or adr_stats(
                len(adrs),
                skipped_count=len(selected_skips),
            )
        ),
        warnings=warnings or [],
    )


def base_components() -> tuple[
    GitHubHistoricalKnowledgeCollectionResult,
    GitHubHistoricalPathEnrichmentResult,
    GitHubHistoricalAdrCollectionResult,
]:
    """Create a complete result from each orchestration stage."""

    issue = knowledge_item(
        KnowledgeItemType.ISSUE,
        "1",
        updated_at=ISSUE_UPDATED_AT,
    )
    pull_request = knowledge_item(
        KnowledgeItemType.PULL_REQUEST,
        "2",
        updated_at=PULL_REQUEST_UPDATED_AT,
    )
    historical = historical_result([issue, pull_request])
    enriched_pull_request = replace_item(
        pull_request,
        affected_paths=["src/opensteward/app.py"],
    )
    paths = path_result([
        enriched_pull_request,
        issue,
    ])
    adr = knowledge_item(
        KnowledgeItemType.ADR,
        "docs/adr/0001-runtime.md",
        updated_at=ADR_UPDATED_AT,
    )
    return historical, paths, adr_result(
        [adr],
        warnings=[ADR_TIMESTAMP_WARNING],
    )


class FakeClock:
    """Record deterministic clock invocations."""

    def __init__(self, value: datetime, events: list[str]) -> None:
        self.value = value
        self.events = events
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        self.events.append("clock")
        return self.value


class FakeHistoricalItemsCollector:
    """Injected historical collector with call recording."""

    def __init__(
        self,
        outcome: GitHubHistoricalKnowledgeCollectionResult | Exception,
        events: list[str],
    ) -> None:
        self.outcome = outcome
        self.events = events
        self.calls: list[tuple[GitHubRepositoryRef, object]] = []

    async def collect_closed_items(
        self,
        repository: GitHubRepositoryRef,
        *,
        options: GitHubHistoricalKnowledgeCollectionOptions | None = None,
    ) -> GitHubHistoricalKnowledgeCollectionResult:
        self.events.append("historical")
        self.calls.append((repository, options))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakePathEnricher:
    """Injected path enricher with call recording."""

    def __init__(
        self,
        outcome: GitHubHistoricalPathEnrichmentResult | Exception,
        events: list[str],
    ) -> None:
        self.outcome = outcome
        self.events = events
        self.calls: list[tuple[object, object]] = []

    async def enrich(
        self,
        collection: GitHubHistoricalKnowledgeCollectionResult,
        *,
        options: GitHubHistoricalPathEnrichmentOptions | None = None,
    ) -> GitHubHistoricalPathEnrichmentResult:
        self.events.append("paths")
        self.calls.append((collection, options))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeAdrCollector:
    """Injected ADR collector with call recording."""

    def __init__(
        self,
        outcome: GitHubHistoricalAdrCollectionResult | Exception,
        events: list[str],
    ) -> None:
        self.outcome = outcome
        self.events = events
        self.calls: list[tuple[GitHubRepositoryRef, str, object]] = []

    async def collect(
        self,
        repository: GitHubRepositoryRef,
        *,
        git_ref: str,
        options: GitHubHistoricalAdrCollectionOptions | None = None,
    ) -> GitHubHistoricalAdrCollectionResult:
        self.events.append("adrs")
        self.calls.append((repository, git_ref, options))
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def create_service(
    historical: GitHubHistoricalKnowledgeCollectionResult | Exception,
    paths: GitHubHistoricalPathEnrichmentResult | Exception,
    adrs: GitHubHistoricalAdrCollectionResult | Exception,
    *,
    clock_value: datetime = COLLECTED_AT,
) -> tuple[
    GitHubHistoricalKnowledgeSnapshotService,
    FakeClock,
    FakeHistoricalItemsCollector,
    FakePathEnricher,
    FakeAdrCollector,
    list[str],
]:
    """Create a service and all recording dependencies."""

    events: list[str] = []
    clock = FakeClock(clock_value, events)
    historical_collector = FakeHistoricalItemsCollector(historical, events)
    path_enricher = FakePathEnricher(paths, events)
    adr_collector = FakeAdrCollector(adrs, events)
    service = GitHubHistoricalKnowledgeSnapshotService(
        historical_items_collector=historical_collector,
        path_enricher=path_enricher,
        adr_collector=adr_collector,
        clock=clock,
    )
    return (
        service,
        clock,
        historical_collector,
        path_enricher,
        adr_collector,
        events,
    )


def scenario_components(
    scenario: str,
) -> tuple[
    GitHubHistoricalKnowledgeCollectionResult,
    GitHubHistoricalPathEnrichmentResult,
    GitHubHistoricalAdrCollectionResult,
]:
    """Create valid components with one documented incompleteness signal."""

    historical, paths, adrs = base_components()
    if scenario == "issue_item_limit":
        historical = historical_result(
            list(historical.items),
            issue_stats=collection_stats(1, item_limit=True),
            pull_request_stats=historical.pull_request_stats,
        )
    elif scenario == "issue_scan_limit":
        historical = historical_result(
            list(historical.items),
            issue_stats=collection_stats(1, scan_limit=True),
            pull_request_stats=historical.pull_request_stats,
        )
    elif scenario == "pull_request_item_limit":
        historical = historical_result(
            list(historical.items),
            issue_stats=historical.issue_stats,
            pull_request_stats=collection_stats(1, item_limit=True),
        )
    elif scenario == "pull_request_scan_limit":
        historical = historical_result(
            list(historical.items),
            issue_stats=historical.issue_stats,
            pull_request_stats=collection_stats(1, scan_limit=True),
        )
    elif scenario == "path_skipped":
        paths = path_result(list(paths.items), evidence=[])
    elif scenario == "path_incomplete":
        pull_request = next(
            item
            for item in paths.items
            if item.item_type == KnowledgeItemType.PULL_REQUEST
        )
        paths = path_result(
            list(paths.items),
            evidence=[path_evidence(pull_request, complete=False)],
        )
    elif scenario == "adr_tree_truncated":
        adrs = adr_result(
            list(adrs.items),
            tree_truncated=True,
            warnings=[ADR_TIMESTAMP_WARNING],
        )
    elif scenario == "adr_item_limit":
        adrs = adr_result(
            list(adrs.items),
            item_limit=True,
            warnings=[ADR_TIMESTAMP_WARNING],
        )
    elif scenario == "adr_total_limit":
        total_skip = skipped_adr(
            reason=GitHubHistoricalAdrSkipReason.TOTAL_BYTES_LIMIT
        )
        adrs = adr_result(
            list(adrs.items),
            skipped_files=[total_skip],
            total_limit=True,
            warnings=[ADR_TIMESTAMP_WARNING],
        )
    elif scenario == "adr_skipped":
        adrs = adr_result(
            list(adrs.items),
            skipped_files=[skipped_adr()],
            warnings=[ADR_TIMESTAMP_WARNING],
        )
    else:
        raise AssertionError(f"Unknown scenario: {scenario}")

    return historical, paths, adrs


def test_options_create_and_preserve_nested_models() -> None:
    defaults = GitHubHistoricalKnowledgeSnapshotOptions()

    assert isinstance(
        defaults.historical_items,
        GitHubHistoricalKnowledgeCollectionOptions,
    )
    assert isinstance(
        defaults.pull_request_paths,
        GitHubHistoricalPathEnrichmentOptions,
    )
    assert isinstance(defaults.adrs, GitHubHistoricalAdrCollectionOptions)

    historical = GitHubHistoricalKnowledgeCollectionOptions(
        max_issues=3,
        max_pull_requests=4,
    )
    paths = GitHubHistoricalPathEnrichmentOptions(max_pull_requests=2)
    adrs = GitHubHistoricalAdrCollectionOptions(max_files=5)
    selected = GitHubHistoricalKnowledgeSnapshotOptions(
        historical_items=historical,
        pull_request_paths=paths,
        adrs=adrs,
    )

    assert selected.historical_items is historical
    assert selected.pull_request_paths is paths
    assert selected.adrs is adrs


def test_options_reject_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GitHubHistoricalKnowledgeSnapshotOptions.model_validate(
            {
                "include_adrs": False,
            }
        )


@pytest.mark.anyio
async def test_empty_ref_is_rejected_before_clock_and_collectors() -> None:
    historical, paths, adrs = base_components()
    service, clock, historical_fake, path_fake, adr_fake, events = create_service(
        historical,
        paths,
        adrs,
    )

    with pytest.raises(ValueError, match="must not be empty"):
        await service.collect(REPOSITORY, git_ref=" \t ")

    assert clock.calls == 0
    assert historical_fake.calls == []
    assert path_fake.calls == []
    assert adr_fake.calls == []
    assert events == []


@pytest.mark.anyio
async def test_clock_ref_normalization_sequence_and_option_forwarding() -> None:
    historical, paths, adrs = base_components()
    adrs = adr_result(
        list(adrs.items),
        requested_ref="release/v1",
        warnings=[ADR_TIMESTAMP_WARNING],
    )
    non_utc_time = datetime(
        2026,
        4,
        5,
        15,
        0,
        tzinfo=timezone(timedelta(hours=6)),
    )
    service, clock, historical_fake, path_fake, adr_fake, events = create_service(
        historical,
        paths,
        adrs,
        clock_value=non_utc_time,
    )
    options = GitHubHistoricalKnowledgeSnapshotOptions(
        historical_items=GitHubHistoricalKnowledgeCollectionOptions(
            max_issues=3,
            max_pull_requests=4,
        ),
        pull_request_paths=GitHubHistoricalPathEnrichmentOptions(
            max_pull_requests=2
        ),
        adrs=GitHubHistoricalAdrCollectionOptions(max_files=5),
    )

    result = await service.collect(
        REPOSITORY,
        git_ref="  release/v1  ",
        options=options,
    )

    assert events == ["clock", "historical", "paths", "adrs"]
    assert clock.calls == 1
    assert historical_fake.calls == [
        (REPOSITORY, options.historical_items)
    ]
    assert path_fake.calls == [
        (historical, options.pull_request_paths)
    ]
    assert adr_fake.calls == [
        (REPOSITORY, "release/v1", options.adrs)
    ]
    assert result.requested_ref == "release/v1"
    assert result.collected_at == COLLECTED_AT


@pytest.mark.anyio
async def test_naive_clock_raises_dedicated_error_before_collectors() -> None:
    historical, paths, adrs = base_components()
    service, clock, historical_fake, path_fake, adr_fake, events = create_service(
        historical,
        paths,
        adrs,
        clock_value=datetime(2026, 4, 5, 9, 0),
    )

    with pytest.raises(GitHubHistoricalKnowledgeSnapshotError, match="clock"):
        await service.collect(REPOSITORY, git_ref="main")

    assert clock.calls == 1
    assert events == ["clock"]
    assert historical_fake.calls == []
    assert path_fake.calls == []
    assert adr_fake.calls == []


@pytest.mark.anyio
async def test_service_does_not_mutate_component_results() -> None:
    historical, paths, adrs = base_components()
    before = [
        component.model_dump(mode="json")
        for component in (historical, paths, adrs)
    ]
    source_lists = (
        historical.items,
        paths.items,
        adrs.items,
    )
    service, *_ = create_service(historical, paths, adrs)

    await service.collect(REPOSITORY, git_ref="main")

    assert [
        component.model_dump(mode="json")
        for component in (historical, paths, adrs)
    ] == before
    assert historical.items is source_lists[0]
    assert paths.items is source_lists[1]
    assert adrs.items is source_lists[2]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("failing_stage", "expected_events"),
    [
        ("historical", ["clock", "historical"]),
        ("paths", ["clock", "historical", "paths"]),
        ("adrs", ["clock", "historical", "paths", "adrs"]),
    ],
)
async def test_collector_errors_propagate_and_stop_later_services(
    failing_stage: str,
    expected_events: list[str],
) -> None:
    historical, paths, adrs = base_components()
    error = RuntimeError(f"{failing_stage} failed")
    service, *_, events = create_service(
        error if failing_stage == "historical" else historical,
        error if failing_stage == "paths" else paths,
        error if failing_stage == "adrs" else adrs,
    )

    with pytest.raises(RuntimeError) as caught:
        await service.collect(REPOSITORY, git_ref="main")

    assert caught.value is error
    assert events == expected_events


@pytest.mark.anyio
@pytest.mark.parametrize(
    "mismatch",
    [
        "historical_repository",
        "historical_knowledge_repository",
        "path_repository",
        "path_knowledge_repository",
        "adr_repository",
        "adr_knowledge_repository",
    ],
)
async def test_component_identity_mismatches_raise_dedicated_error(
    mismatch: str,
) -> None:
    historical, paths, adrs = base_components()
    if mismatch == "historical_repository":
        historical = historical.model_copy(
            update={"repository": OTHER_REPOSITORY}
        )
    elif mismatch == "historical_knowledge_repository":
        historical = historical.model_copy(
            update={"knowledge_repository": OTHER_KNOWLEDGE_REPOSITORY}
        )
    elif mismatch == "path_repository":
        paths = paths.model_copy(update={"repository": OTHER_REPOSITORY})
    elif mismatch == "path_knowledge_repository":
        paths = paths.model_copy(
            update={"knowledge_repository": OTHER_KNOWLEDGE_REPOSITORY}
        )
    elif mismatch == "adr_repository":
        adrs = adrs.model_copy(update={"repository": OTHER_REPOSITORY})
    else:
        adrs = adrs.model_copy(
            update={"knowledge_repository": OTHER_KNOWLEDGE_REPOSITORY}
        )
    service, *_ = create_service(historical, paths, adrs)

    with pytest.raises(GitHubHistoricalKnowledgeSnapshotError):
        await service.collect(REPOSITORY, git_ref="main")


@pytest.mark.anyio
@pytest.mark.parametrize(
    "change",
    [
        "remove",
        "insert",
        "reorder",
        "change_key",
    ],
)
async def test_path_enrichment_must_preserve_item_key_sequence(
    change: str,
) -> None:
    historical, _, adrs = base_components()
    changed_items = list(historical.items)
    if change == "remove":
        changed_items.pop()
    elif change == "insert":
        changed_items.append(
            knowledge_item(
                KnowledgeItemType.ISSUE,
                "99",
                updated_at=CREATED_AT,
            )
        )
    elif change == "reorder":
        changed_items.reverse()
    else:
        changed_items[-1] = replace_item(
            changed_items[-1],
            external_id="changed",
        )
    paths = path_result(changed_items)
    service, *_ = create_service(historical, paths, adrs)

    with pytest.raises(
        GitHubHistoricalKnowledgeSnapshotError,
        match="item set or order",
    ):
        await service.collect(REPOSITORY, git_ref="main")


@pytest.mark.anyio
async def test_path_enrichment_may_change_only_affected_paths() -> None:
    historical, paths, adrs = base_components()
    service, *_ = create_service(historical, paths, adrs)

    result = await service.collect(REPOSITORY, git_ref="main")

    source_pull_request = next(
        item
        for item in historical.items
        if item.item_type == KnowledgeItemType.PULL_REQUEST
    )
    snapshot_pull_request = next(
        item
        for item in result.items
        if item.item_type == KnowledgeItemType.PULL_REQUEST
    )
    assert source_pull_request.affected_paths == []
    assert snapshot_pull_request.affected_paths == [
        "src/opensteward/app.py"
    ]
    assert snapshot_pull_request is paths.items[0]


@pytest.mark.anyio
async def test_adr_requested_ref_must_match_normalized_ref() -> None:
    historical, paths, adrs = base_components()
    adrs = adr_result(list(adrs.items), requested_ref="other")
    service, *_ = create_service(historical, paths, adrs)

    with pytest.raises(
        GitHubHistoricalKnowledgeSnapshotError,
        match="requested Git reference",
    ):
        await service.collect(REPOSITORY, git_ref=" main ")


@pytest.mark.anyio
async def test_combines_items_and_sorts_by_update_then_key() -> None:
    shared_update = PULL_REQUEST_UPDATED_AT
    issue = knowledge_item(
        KnowledgeItemType.ISSUE,
        "5",
        updated_at=shared_update,
    )
    pull_request = knowledge_item(
        KnowledgeItemType.PULL_REQUEST,
        "4",
        updated_at=shared_update,
    )
    historical = historical_result([pull_request, issue])
    paths = path_result(list(historical.items))
    adr = knowledge_item(
        KnowledgeItemType.ADR,
        "docs/adr/0001.md",
        updated_at=shared_update,
    )
    adrs = adr_result([adr])
    service, *_ = create_service(historical, paths, adrs)

    result = await service.collect(REPOSITORY, git_ref="main")

    assert result.items == sorted(
        [issue, pull_request, adr],
        key=lambda item: item.key,
    )
    assert result.total_count == 3
    assert result.issue_count == 1
    assert result.pull_request_count == 1
    assert result.adr_count == 1


@pytest.mark.parametrize(
    ("invalid_kind", "source_kind"),
    [
        (KnowledgeItemType.ISSUE, KnowledgeSourceKind.REPOSITORY_FILE),
        (KnowledgeItemType.PULL_REQUEST, KnowledgeSourceKind.REPOSITORY_FILE),
        (KnowledgeItemType.ADR, KnowledgeSourceKind.GITHUB),
    ],
)
def test_result_rejects_incorrect_source_kinds(
    invalid_kind: KnowledgeItemType,
    source_kind: KnowledgeSourceKind,
) -> None:
    item = knowledge_item(
        invalid_kind,
        (
            "docs/adr/invalid.md"
            if invalid_kind == KnowledgeItemType.ADR
            else "1"
        ),
        updated_at=ISSUE_UPDATED_AT,
        source_kind=source_kind,
    )

    with pytest.raises(ValidationError, match="source kind"):
        direct_snapshot([item])


def test_result_rejects_duplicate_unsupported_and_cross_repository_items() -> None:
    issue = knowledge_item(
        KnowledgeItemType.ISSUE,
        "1",
        updated_at=ISSUE_UPDATED_AT,
    )
    with pytest.raises(ValidationError, match="keys must be unique"):
        direct_snapshot(
            [issue, issue],
            issue_stats=collection_stats(2),
        )

    unsupported = knowledge_item(
        KnowledgeItemType.DOCUMENTATION,
        "README.md",
        updated_at=ISSUE_UPDATED_AT,
        source_kind=KnowledgeSourceKind.GITHUB,
    )
    with pytest.raises(ValidationError, match="only issues"):
        direct_snapshot([unsupported])

    foreign = knowledge_item(
        KnowledgeItemType.ISSUE,
        "1",
        updated_at=ISSUE_UPDATED_AT,
        repository=OTHER_KNOWLEDGE_REPOSITORY,
    )
    with pytest.raises(ValidationError, match="belong"):
        direct_snapshot([foreign])


def test_result_rejects_incorrect_manual_item_order() -> None:
    newer = knowledge_item(
        KnowledgeItemType.ISSUE,
        "1",
        updated_at=PULL_REQUEST_UPDATED_AT,
    )
    older = knowledge_item(
        KnowledgeItemType.PULL_REQUEST,
        "2",
        updated_at=ISSUE_UPDATED_AT,
    )

    with pytest.raises(ValidationError, match="update order"):
        direct_snapshot([older, newer])


@pytest.mark.anyio
async def test_service_carries_all_component_metadata_and_provenance() -> None:
    historical, paths, adrs = base_components()
    adrs = adr_result(
        list(adrs.items),
        skipped_files=[skipped_adr()],
        warnings=[ADR_TIMESTAMP_WARNING],
    )
    service, *_ = create_service(historical, paths, adrs)

    result = await service.collect(REPOSITORY, git_ref="main")

    assert result.issue_stats is historical.issue_stats
    assert result.pull_request_stats is historical.pull_request_stats
    assert result.pull_request_path_evidence == paths.pull_request_evidence
    assert result.pull_requests_available == paths.pull_requests_available
    assert result.pull_requests_enriched == paths.pull_requests_enriched
    assert (
        result.pull_requests_skipped_due_limit
        == paths.pull_requests_skipped_due_limit
    )
    assert result.adr_file_evidence == adrs.file_evidence
    assert result.adr_skipped_files == adrs.skipped_files
    assert result.adr_stats is adrs.stats
    assert result.resolved_adr_commit_sha == ADR_COMMIT_SHA
    assert result.adr_tree_sha == ADR_TREE_SHA
    assert result.adr_snapshot_commit_date == ADR_UPDATED_AT


def test_result_rejects_invalid_path_evidence_relationships() -> None:
    issue = knowledge_item(
        KnowledgeItemType.ISSUE,
        "1",
        updated_at=ISSUE_UPDATED_AT,
    )
    issue_evidence = GitHubHistoricalPullRequestPathEvidence(
        item_key=issue.key,
        pull_number=1,
        pages_fetched=1,
        api_files_seen=0,
        affected_paths_collected=0,
        complete=True,
        file_limit_reached=False,
    )
    with pytest.raises(ValidationError, match="pull-request item"):
        direct_snapshot(
            [issue],
            path_evidence_items=[issue_evidence],
            pull_requests_available=0,
            pull_requests_enriched=1,
            pull_requests_skipped_due_limit=0,
        )

    pull_request = knowledge_item(
        KnowledgeItemType.PULL_REQUEST,
        "2",
        updated_at=PULL_REQUEST_UPDATED_AT,
    )
    duplicate = path_evidence(pull_request)
    with pytest.raises(ValidationError, match="must be unique"):
        direct_snapshot(
            [pull_request],
            path_evidence_items=[duplicate, duplicate],
            pull_requests_available=2,
            pull_requests_enriched=2,
            pull_requests_skipped_due_limit=0,
        )


def test_result_rejects_invalid_adr_evidence_relationships() -> None:
    adr = knowledge_item(
        KnowledgeItemType.ADR,
        "docs/adr/one.md",
        updated_at=ADR_UPDATED_AT,
    )
    missing = knowledge_item(
        KnowledgeItemType.ADR,
        "docs/adr/missing.md",
        updated_at=ADR_UPDATED_AT,
    )
    with pytest.raises(ValidationError, match="identify an ADR item"):
        direct_snapshot(
            [adr],
            adr_file_evidence=[adr_evidence(missing)],
        )

    second = knowledge_item(
        KnowledgeItemType.ADR,
        "docs/adr/two.md",
        updated_at=ADR_UPDATED_AT,
    )
    duplicate_path = "docs/adr/one.md"
    with pytest.raises(ValidationError, match="paths must be unique"):
        direct_snapshot(
            sorted_items([adr, second]),
            adr_file_evidence=[
                adr_evidence(adr, path=duplicate_path),
                adr_evidence(second, path=duplicate_path),
            ],
        )


@pytest.mark.parametrize(
    "count_kind",
    [
        "issue",
        "pull_request",
        "adr",
        "path_available",
    ],
)
def test_result_rejects_inconsistent_collection_counts(
    count_kind: str,
) -> None:
    issue = knowledge_item(
        KnowledgeItemType.ISSUE,
        "1",
        updated_at=ISSUE_UPDATED_AT,
    )
    pull_request = knowledge_item(
        KnowledgeItemType.PULL_REQUEST,
        "2",
        updated_at=PULL_REQUEST_UPDATED_AT,
    )
    adr = knowledge_item(
        KnowledgeItemType.ADR,
        "docs/adr/one.md",
        updated_at=ADR_UPDATED_AT,
    )
    items = sorted_items([issue, pull_request, adr])
    kwargs: dict[str, Any] = {}
    if count_kind == "issue":
        kwargs["issue_stats"] = collection_stats(0)
    elif count_kind == "pull_request":
        kwargs["pull_request_stats"] = collection_stats(0)
    elif count_kind == "adr":
        kwargs["adr_stats_value"] = adr_stats(0)
    else:
        kwargs["pull_requests_available"] = 2
        kwargs["pull_requests_enriched"] = 1
        kwargs["pull_requests_skipped_due_limit"] = 1

    with pytest.raises(ValidationError):
        direct_snapshot(items, **kwargs)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("scenario", "expected_warning"),
    [
        ("issue_item_limit", ISSUE_ITEM_LIMIT_WARNING),
        ("issue_scan_limit", ISSUE_SCAN_LIMIT_WARNING),
        ("pull_request_item_limit", PULL_REQUEST_ITEM_LIMIT_WARNING),
        ("pull_request_scan_limit", PULL_REQUEST_SCAN_LIMIT_WARNING),
        ("path_skipped", PATH_SKIPPED_WARNING),
        ("path_incomplete", PATH_INCOMPLETE_WARNING),
        ("adr_item_limit", ADR_ITEM_LIMIT_WARNING),
        ("adr_skipped", ADR_SKIPPED_WARNING),
    ],
)
async def test_service_adds_each_exact_orchestration_warning(
    scenario: str,
    expected_warning: str,
) -> None:
    historical, paths, adrs = scenario_components(scenario)
    service, *_ = create_service(historical, paths, adrs)

    result = await service.collect(REPOSITORY, git_ref="main")

    assert expected_warning in result.warnings


@pytest.mark.anyio
async def test_complete_service_adds_only_existing_adr_warnings() -> None:
    historical, paths, adrs = base_components()
    service, *_ = create_service(historical, paths, adrs)

    result = await service.collect(REPOSITORY, git_ref="main")

    assert result.warnings == [ADR_TIMESTAMP_WARNING]
    assert result.complete is True


@pytest.mark.anyio
async def test_warning_order_and_deduplication_are_deterministic() -> None:
    issue = knowledge_item(
        KnowledgeItemType.ISSUE,
        "1",
        updated_at=ISSUE_UPDATED_AT,
    )
    pull_one = knowledge_item(
        KnowledgeItemType.PULL_REQUEST,
        "2",
        updated_at=PULL_REQUEST_UPDATED_AT,
        affected_paths=["src/one.py"],
    )
    pull_two = knowledge_item(
        KnowledgeItemType.PULL_REQUEST,
        "3",
        updated_at=PULL_REQUEST_UPDATED_AT - timedelta(hours=1),
    )
    historical = historical_result(
        [issue, pull_one, pull_two],
        issue_stats=collection_stats(1, item_limit=True),
        pull_request_stats=collection_stats(2, scan_limit=True),
    )
    paths = path_result(
        list(historical.items),
        evidence=[path_evidence(pull_one, complete=False)],
    )
    adr = knowledge_item(
        KnowledgeItemType.ADR,
        "docs/adr/one.md",
        updated_at=ADR_UPDATED_AT,
    )
    adrs = adr_result(
        [adr],
        skipped_files=[skipped_adr()],
        item_limit=True,
        warnings=[
            ISSUE_ITEM_LIMIT_WARNING,
            ADR_TIMESTAMP_WARNING,
        ],
    )
    service, *_ = create_service(historical, paths, adrs)

    result = await service.collect(REPOSITORY, git_ref="main")

    assert result.warnings == [
        ISSUE_ITEM_LIMIT_WARNING,
        PULL_REQUEST_SCAN_LIMIT_WARNING,
        PATH_SKIPPED_WARNING,
        PATH_INCOMPLETE_WARNING,
        ADR_ITEM_LIMIT_WARNING,
        ADR_SKIPPED_WARNING,
        ADR_TIMESTAMP_WARNING,
    ]


def test_result_rejects_empty_and_duplicate_warnings() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        direct_snapshot([], warnings=[" "])

    with pytest.raises(ValidationError, match="must be unique"):
        direct_snapshot([], warnings=["Repeated", "Repeated"])


@pytest.mark.anyio
@pytest.mark.parametrize(
    "scenario",
    [
        "issue_item_limit",
        "issue_scan_limit",
        "pull_request_item_limit",
        "pull_request_scan_limit",
        "path_skipped",
        "path_incomplete",
        "adr_tree_truncated",
        "adr_item_limit",
        "adr_total_limit",
        "adr_skipped",
    ],
)
async def test_each_incompleteness_signal_makes_snapshot_incomplete(
    scenario: str,
) -> None:
    historical, paths, adrs = scenario_components(scenario)
    service, *_ = create_service(historical, paths, adrs)

    result = await service.collect(REPOSITORY, git_ref="main")

    assert result.complete is False


@pytest.mark.anyio
async def test_zero_limits_produce_valid_incomplete_snapshot() -> None:
    historical = historical_result(
        [],
        issue_stats=collection_stats(0, item_limit=True),
        pull_request_stats=collection_stats(0, item_limit=True),
    )
    paths = path_result([])
    adrs = adr_result([], item_limit=True)
    service, _, historical_fake, path_fake, adr_fake, _ = create_service(
        historical,
        paths,
        adrs,
    )
    options = GitHubHistoricalKnowledgeSnapshotOptions(
        historical_items=GitHubHistoricalKnowledgeCollectionOptions(
            max_issues=0,
            max_pull_requests=0,
        ),
        pull_request_paths=GitHubHistoricalPathEnrichmentOptions(
            max_pull_requests=0
        ),
        adrs=GitHubHistoricalAdrCollectionOptions(max_files=0),
    )

    result = await service.collect(
        REPOSITORY,
        git_ref="main",
        options=options,
    )

    assert result.total_count == 0
    assert result.complete is False
    assert historical_fake.calls[0][1] is options.historical_items
    assert path_fake.calls[0][1] is options.pull_request_paths
    assert adr_fake.calls[0][2] is options.adrs
    assert adr_fake.calls[0][2].max_files == 0


def test_result_normalizes_datetimes_and_rejects_naive_values() -> None:
    non_utc = datetime(
        2026,
        4,
        5,
        15,
        0,
        tzinfo=timezone(timedelta(hours=6)),
    )
    result = direct_snapshot(
        [],
        collected_at=non_utc,
        adr_snapshot_commit_date=non_utc,
    )

    assert result.collected_at == COLLECTED_AT
    assert result.adr_snapshot_commit_date == COLLECTED_AT

    with pytest.raises(ValidationError, match="timezone-aware"):
        direct_snapshot(
            [],
            collected_at=datetime(2026, 4, 5, 9, 0),
        )


@pytest.mark.anyio
async def test_snapshot_serializes_nested_metadata_and_computed_fields() -> None:
    historical, paths, adrs = base_components()
    service, *_ = create_service(historical, paths, adrs)

    result = await service.collect(REPOSITORY, git_ref="main")
    payload = result.model_dump(mode="json")

    assert payload["collected_at"] == COLLECTED_AT.isoformat().replace(
        "+00:00",
        "Z",
    )
    assert payload["adr_snapshot_commit_date"] == (
        ADR_UPDATED_AT.isoformat().replace("+00:00", "Z")
    )
    assert payload["items"][0]["key"] == result.items[0].key
    assert payload["pull_request_path_evidence"][0]["complete"] is True
    assert payload["adr_file_evidence"][0]["timestamp_basis"] == (
        "snapshot_commit"
    )
    assert payload["total_count"] == 3
    assert payload["issue_count"] == 1
    assert payload["pull_request_count"] == 1
    assert payload["adr_count"] == 1
    assert payload["complete"] is True
