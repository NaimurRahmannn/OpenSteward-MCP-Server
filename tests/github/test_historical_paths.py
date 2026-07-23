"""Tests for historical pull-request changed-path enrichment."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from opensteward.github import (
    GITHUB_HISTORICAL_PATH_PAGE_SIZE,
    MAX_GITHUB_HISTORICAL_FILES_PER_PULL_REQUEST,
    MAX_GITHUB_HISTORICAL_PATH_PAGES,
    MAX_GITHUB_HISTORICAL_PULL_REQUESTS_TO_ENRICH,
    GitHubHistoricalKnowledgeCollectionResult,
    GitHubHistoricalKnowledgeCollectionStats,
    GitHubHistoricalPathEnrichmentError,
    GitHubHistoricalPathEnrichmentOptions,
    GitHubHistoricalPathEnrichmentResult,
    GitHubHistoricalPullRequestPathEnricher,
    GitHubHistoricalPullRequestPathEvidence,
    GitHubRepositoryRef,
    GitHubRestResponse,
    GitHubRestResponseError,
    knowledge_repository_from_github,
)
from opensteward.knowledge import (
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

CREATED_AT = datetime(2026, 2, 1, 9, 0, tzinfo=UTC)
UPDATED_AT = datetime(2026, 2, 5, 12, 0, tzinfo=UTC)
CLOSED_AT = datetime(2026, 2, 4, 11, 0, tzinfo=UTC)
REPOSITORY = GitHubRepositoryRef(
    owner="acme",
    name="framework",
)
KNOWLEDGE_REPOSITORY = knowledge_repository_from_github(REPOSITORY)


def file_payload(
    filename: str,
    *,
    previous_filename: str | None = None,
    status: str = "modified",
) -> dict[str, object]:
    """Create one GitHub pull-request file payload."""

    return {
        "filename": filename,
        "previous_filename": previous_filename,
        "status": status,
    }


def path_for_pull(pull_number: str) -> str:
    """Return the expected file endpoint for one pull request."""

    return f"/repos/acme/framework/pulls/{pull_number}/files"


class FakeGitHubRestClient:
    """Path-and-page-aware typed fake for path enrichment."""

    def __init__(
        self,
        responses: Mapping[tuple[str, int], object] | None = None,
    ) -> None:
        self._responses = dict(responses or {})
        self.calls: list[dict[str, object]] = []

    async def get_json(
        self,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        response_type: Any = Any,
        accept: str = "application/vnd.github+json",
    ) -> GitHubRestResponse[Any]:
        self.calls.append(
            {
                "path": path,
                "params": params,
                "response_type": response_type,
                "accept": accept,
            }
        )

        page = int(params["page"]) if params is not None else 1
        outcome = self._responses.get((path, page), [])

        if isinstance(outcome, Exception):
            raise outcome

        data = outcome
        if response_type is not Any:
            data = TypeAdapter(response_type).validate_python(outcome)

        return GitHubRestResponse(
            status_code=200,
            data=data,
        )


def knowledge_item(
    *,
    item_type: KnowledgeItemType,
    external_id: str,
    updated_at: datetime = UPDATED_AT,
    affected_paths: list[str] | None = None,
    repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
    source_kind: KnowledgeSourceKind = KnowledgeSourceKind.GITHUB,
) -> KnowledgeItem:
    """Create one historical item for enrichment tests."""

    return KnowledgeItem(
        repository=repository,
        item_type=item_type,
        external_id=external_id,
        source_kind=source_kind,
        state=KnowledgeItemState.CLOSED,
        title=f"Historical {item_type.value} {external_id}",
        created_at=CREATED_AT,
        updated_at=updated_at,
        closed_at=CLOSED_AT,
        affected_paths=affected_paths or [],
    )


def collection_stats(
    *,
    requested_limit: int,
    items_collected: int,
) -> GitHubHistoricalKnowledgeCollectionStats:
    """Create valid source collection statistics."""

    return GitHubHistoricalKnowledgeCollectionStats(
        requested_limit=requested_limit,
        pages_fetched=1,
        api_items_seen=items_collected,
        items_collected=items_collected,
        filtered_items=0,
        duplicate_items=0,
        item_limit_reached=items_collected == requested_limit,
        scan_limit_reached=False,
    )


def historical_collection(
    items: list[KnowledgeItem],
    *,
    repository: GitHubRepositoryRef = REPOSITORY,
    knowledge_repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
) -> GitHubHistoricalKnowledgeCollectionResult:
    """Create a validated historical collection."""

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
        issue_stats=collection_stats(
            requested_limit=100,
            items_collected=issue_count,
        ),
        pull_request_stats=collection_stats(
            requested_limit=100,
            items_collected=pull_request_count,
        ),
    )


async def enrich(
    client: FakeGitHubRestClient,
    collection: GitHubHistoricalKnowledgeCollectionResult,
    *,
    max_pull_requests: int = 50,
) -> GitHubHistoricalPathEnrichmentResult:
    """Run path enrichment with an explicit selection limit."""

    enricher = GitHubHistoricalPullRequestPathEnricher(
        rest_client=client,
    )
    return await enricher.enrich(
        collection,
        options=GitHubHistoricalPathEnrichmentOptions(
            max_pull_requests=max_pull_requests,
        ),
    )


def path_evidence(
    item: KnowledgeItem,
    *,
    complete: bool = True,
) -> GitHubHistoricalPullRequestPathEvidence:
    """Create result-validation evidence for one item."""

    return GitHubHistoricalPullRequestPathEvidence(
        item_key=item.key,
        pull_number=int(item.external_id),
        pages_fetched=1,
        api_files_seen=1,
        affected_paths_collected=1,
        complete=complete,
        file_limit_reached=False,
    )


def enrichment_result(
    items: list[KnowledgeItem],
    evidence: list[GitHubHistoricalPullRequestPathEvidence],
    *,
    knowledge_repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
) -> GitHubHistoricalPathEnrichmentResult:
    """Create a public enrichment result for validation tests."""

    available = sum(
        item.item_type == KnowledgeItemType.PULL_REQUEST
        for item in items
    )
    enriched = len(evidence)

    return GitHubHistoricalPathEnrichmentResult(
        repository=REPOSITORY,
        knowledge_repository=knowledge_repository,
        items=items,
        pull_request_evidence=evidence,
        pull_requests_available=available,
        pull_requests_enriched=enriched,
        pull_requests_skipped_due_limit=max(
            available - enriched,
            0,
        ),
    )


@pytest.mark.anyio
async def test_enriches_one_pull_request_with_changed_paths() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    collection = historical_collection([pull_request])
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): [
                file_payload("src/opensteward/app.py"),
                file_payload("tests/test_app.py"),
            ],
        }
    )

    result = await enrich(client, collection)

    assert result.items[0].affected_paths == [
        "src/opensteward/app.py",
        "tests/test_app.py",
    ]
    assert result.pull_request_evidence[0].affected_paths_collected == 2


@pytest.mark.anyio
async def test_preserves_issue_and_unselected_pull_request_instances() -> None:
    issue = knowledge_item(
        item_type=KnowledgeItemType.ISSUE,
        external_id="10",
        updated_at=UPDATED_AT + timedelta(days=2),
    )
    selected_pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="20",
        updated_at=UPDATED_AT + timedelta(days=1),
    )
    unselected_pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="30",
    )
    collection = historical_collection(
        [
            issue,
            selected_pull_request,
            unselected_pull_request,
        ]
    )
    client = FakeGitHubRestClient()

    result = await enrich(
        client,
        collection,
        max_pull_requests=1,
    )

    assert result.items[0] is collection.items[0]
    assert result.items[1] is not collection.items[1]
    assert result.items[2] is collection.items[2]
    assert result.pull_requests_skipped_due_limit == 1


@pytest.mark.anyio
async def test_preserves_source_order_and_does_not_mutate_collection() -> None:
    items = [
        knowledge_item(
            item_type=KnowledgeItemType.PULL_REQUEST,
            external_id="1",
            updated_at=UPDATED_AT + timedelta(days=2),
        ),
        knowledge_item(
            item_type=KnowledgeItemType.ISSUE,
            external_id="2",
            updated_at=UPDATED_AT + timedelta(days=1),
        ),
        knowledge_item(
            item_type=KnowledgeItemType.PULL_REQUEST,
            external_id="3",
        ),
    ]
    collection = historical_collection(items)
    original_dump = collection.model_dump(mode="json")
    original_instances = list(collection.items)
    client = FakeGitHubRestClient()

    result = await enrich(client, collection)

    assert [item.key for item in result.items] == [
        item.key for item in collection.items
    ]
    assert collection.model_dump(mode="json") == original_dump
    assert all(
        current is original
        for current, original in zip(
            collection.items,
            original_instances,
            strict=True,
        )
    )


@pytest.mark.anyio
async def test_renamed_file_collects_current_then_previous_path() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    collection = historical_collection([pull_request])
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): [
                file_payload(
                    "src/new_name.py",
                    previous_filename="src/old_name.py",
                    status="renamed",
                )
            ],
        }
    )

    result = await enrich(client, collection)
    evidence = result.pull_request_evidence[0]

    assert result.items[0].affected_paths == [
        "src/new_name.py",
        "src/old_name.py",
    ]
    assert evidence.api_files_seen == 1
    assert evidence.affected_paths_collected == 2


@pytest.mark.anyio
async def test_deduplicates_paths_and_preserves_existing_paths_first() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
        affected_paths=["existing/config.py"],
    )
    collection = historical_collection([pull_request])
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): [
                file_payload("existing/config.py"),
                file_payload(
                    "src/current.py",
                    previous_filename="existing/config.py",
                ),
                file_payload("src/current.py"),
            ],
        }
    )

    result = await enrich(client, collection)

    assert result.items[0].affected_paths == [
        "existing/config.py",
        "src/current.py",
    ]


@pytest.mark.anyio
async def test_uses_exact_endpoint_and_request_parameters() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="398",
    )
    collection = historical_collection([pull_request])
    client = FakeGitHubRestClient()

    await enrich(client, collection)

    assert client.calls == [
        {
            "path": "/repos/acme/framework/pulls/398/files",
            "params": {
                "per_page": GITHUB_HISTORICAL_PATH_PAGE_SIZE,
                "page": 1,
            },
            "response_type": client.calls[0]["response_type"],
            "accept": "application/vnd.github+json",
        }
    ]


@pytest.mark.anyio
async def test_repository_path_segments_are_url_encoded() -> None:
    repository = GitHubRepositoryRef(
        owner="acme org",
        name="framework tools",
    )
    knowledge_repository = knowledge_repository_from_github(repository)
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
        repository=knowledge_repository,
    )
    collection = historical_collection(
        [pull_request],
        repository=repository,
        knowledge_repository=knowledge_repository,
    )
    client = FakeGitHubRestClient()

    await enrich(client, collection)

    assert client.calls[0]["path"] == (
        "/repos/acme%20org/framework%20tools/pulls/42/files"
    )


@pytest.mark.anyio
async def test_paginates_after_full_page_then_stops_on_short_page() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    first_page = [
        file_payload(f"src/file_{index}.py")
        for index in range(GITHUB_HISTORICAL_PATH_PAGE_SIZE)
    ]
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): first_page,
            (path_for_pull("42"), 2): [
                file_payload("src/final.py"),
            ],
        }
    )

    result = await enrich(
        client,
        historical_collection([pull_request]),
    )
    evidence = result.pull_request_evidence[0]

    assert evidence.pages_fetched == 2
    assert evidence.api_files_seen == 101
    assert evidence.affected_paths_collected == 101
    assert evidence.complete is True
    assert evidence.file_limit_reached is False
    assert len(client.calls) == 2


@pytest.mark.anyio
async def test_stops_after_first_short_page() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): [
                file_payload("src/only.py"),
            ],
            (path_for_pull("42"), 2): [
                file_payload("src/not_requested.py"),
            ],
        }
    )

    result = await enrich(
        client,
        historical_collection([pull_request]),
    )

    assert result.items[0].affected_paths == ["src/only.py"]
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_empty_first_page_produces_complete_empty_evidence() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
        affected_paths=["existing/path.py"],
    )
    client = FakeGitHubRestClient()

    result = await enrich(
        client,
        historical_collection([pull_request]),
    )
    evidence = result.pull_request_evidence[0]

    assert result.items[0].affected_paths == ["existing/path.py"]
    assert evidence.pages_fetched == 1
    assert evidence.api_files_seen == 0
    assert evidence.affected_paths_collected == 0
    assert evidence.complete is True
    assert evidence.file_limit_reached is False


@pytest.mark.anyio
async def test_full_final_page_reports_file_limit_and_caps_entries() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    responses = {
        (path_for_pull("42"), page): [
            file_payload(
                f"src/page_{page}/file_{index}.py"
            )
            for index in range(GITHUB_HISTORICAL_PATH_PAGE_SIZE)
        ]
        for page in range(1, MAX_GITHUB_HISTORICAL_PATH_PAGES + 1)
    }
    responses[(path_for_pull("42"), MAX_GITHUB_HISTORICAL_PATH_PAGES + 1)] = [
        file_payload("src/must_not_be_requested.py")
    ]
    client = FakeGitHubRestClient(responses)

    result = await enrich(
        client,
        historical_collection([pull_request]),
    )
    evidence = result.pull_request_evidence[0]

    assert len(client.calls) == MAX_GITHUB_HISTORICAL_PATH_PAGES
    assert evidence.pages_fetched == MAX_GITHUB_HISTORICAL_PATH_PAGES
    assert (
        evidence.api_files_seen
        == MAX_GITHUB_HISTORICAL_FILES_PER_PULL_REQUEST
    )
    assert (
        evidence.affected_paths_collected
        == MAX_GITHUB_HISTORICAL_FILES_PER_PULL_REQUEST
    )
    assert evidence.complete is False
    assert evidence.file_limit_reached is True
    assert (
        len(result.items[0].affected_paths)
        == MAX_GITHUB_HISTORICAL_FILES_PER_PULL_REQUEST
    )


@pytest.mark.anyio
async def test_max_pull_requests_limits_endpoint_calls_and_evidence_order() -> None:
    pull_requests = [
        knowledge_item(
            item_type=KnowledgeItemType.PULL_REQUEST,
            external_id=str(number),
            updated_at=UPDATED_AT - timedelta(days=number),
        )
        for number in range(1, 4)
    ]
    collection = historical_collection(pull_requests)
    client = FakeGitHubRestClient()

    result = await enrich(
        client,
        collection,
        max_pull_requests=2,
    )

    selected_items = [
        item
        for item in collection.items
        if item.item_type == KnowledgeItemType.PULL_REQUEST
    ][:2]
    assert len(client.calls) == 2
    assert [call["path"] for call in client.calls] == [
        path_for_pull(item.external_id)
        for item in selected_items
    ]
    assert [
        evidence.item_key
        for evidence in result.pull_request_evidence
    ] == [
        item.key
        for item in selected_items
    ]
    assert result.pull_requests_enriched == 2
    assert result.pull_requests_skipped_due_limit == 1


@pytest.mark.anyio
async def test_zero_limit_performs_no_calls_and_preserves_instances() -> None:
    issue = knowledge_item(
        item_type=KnowledgeItemType.ISSUE,
        external_id="1",
        updated_at=UPDATED_AT + timedelta(days=1),
    )
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="2",
    )
    collection = historical_collection([issue, pull_request])
    client = FakeGitHubRestClient()

    result = await enrich(
        client,
        collection,
        max_pull_requests=0,
    )

    assert client.calls == []
    assert result.pull_request_evidence == []
    assert result.pull_requests_enriched == 0
    assert result.pull_requests_skipped_due_limit == 1
    assert all(
        result_item is source_item
        for result_item, source_item in zip(
            result.items,
            collection.items,
            strict=True,
        )
    )


@pytest.mark.parametrize(
    "max_pull_requests",
    [-1, MAX_GITHUB_HISTORICAL_PULL_REQUESTS_TO_ENRICH + 1],
)
def test_options_reject_out_of_bounds_values(
    max_pull_requests: int,
) -> None:
    with pytest.raises(ValidationError):
        GitHubHistoricalPathEnrichmentOptions(
            max_pull_requests=max_pull_requests
        )


@pytest.mark.anyio
@pytest.mark.parametrize("external_id", ["not-a-number", "12x", "0"])
async def test_invalid_pull_number_is_rejected_before_request(
    external_id: str,
) -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id=external_id,
    )
    collection = historical_collection([pull_request])
    client = FakeGitHubRestClient()

    with pytest.raises(
        GitHubHistoricalPathEnrichmentError,
        match=pull_request.key,
    ):
        await enrich(client, collection)

    assert client.calls == []


@pytest.mark.anyio
async def test_invalid_api_file_payload_is_rejected() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): [
                {
                    "filename": "",
                    "status": "modified",
                }
            ],
        }
    )

    with pytest.raises(ValidationError):
        await enrich(
            client,
            historical_collection([pull_request]),
        )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("field_name", "unsafe_path"),
    [
        ("filename", "../secrets.txt"),
        ("previous_filename", "/absolute/file.py"),
    ],
)
async def test_unsafe_api_paths_propagate_knowledge_validation(
    field_name: str,
    unsafe_path: str,
) -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    payload = file_payload("src/current.py")
    payload[field_name] = unsafe_path
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): [payload],
        }
    )

    with pytest.raises(ValidationError):
        await enrich(
            client,
            historical_collection([pull_request]),
        )


def test_result_computed_counts_are_correct() -> None:
    issue = knowledge_item(
        item_type=KnowledgeItemType.ISSUE,
        external_id="1",
    )
    first_pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="2",
    )
    second_pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="3",
    )
    result = enrichment_result(
        [issue, first_pull_request, second_pull_request],
        [
            path_evidence(first_pull_request),
            path_evidence(second_pull_request, complete=False),
        ],
    )

    assert result.total_count == 3
    assert result.issue_count == 1
    assert result.pull_request_count == 2
    assert result.complete_pull_request_count == 1
    assert result.incomplete_pull_request_count == 1


def test_result_rejects_duplicate_item_keys() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )

    with pytest.raises(ValidationError, match="item keys must be unique"):
        enrichment_result([pull_request, pull_request], [])


def test_result_rejects_evidence_for_issue() -> None:
    issue = knowledge_item(
        item_type=KnowledgeItemType.ISSUE,
        external_id="42",
    )
    evidence = GitHubHistoricalPullRequestPathEvidence(
        item_key=issue.key,
        pull_number=42,
        pages_fetched=1,
        api_files_seen=0,
        affected_paths_collected=0,
        complete=True,
        file_limit_reached=False,
    )

    with pytest.raises(ValidationError, match="pull-request item"):
        enrichment_result([issue], [evidence])


def test_result_rejects_duplicate_evidence_keys() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    evidence = path_evidence(pull_request)

    with pytest.raises(ValidationError, match="evidence item keys must be unique"):
        enrichment_result(
            [pull_request],
            [evidence, evidence],
        )


def test_result_rejects_cross_repository_item() -> None:
    other_repository = KnowledgeRepositoryRef(
        provider="github",
        namespace="other",
        name="framework",
    )
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
        repository=other_repository,
    )

    with pytest.raises(ValidationError, match="knowledge repository"):
        enrichment_result([pull_request], [])


@pytest.mark.anyio
async def test_extra_github_file_fields_are_ignored() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    payload = file_payload("src/app.py")
    payload.update(
        {
            "sha": "abc123",
            "additions": 10,
            "deletions": 2,
            "changes": 12,
        }
    )
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): [payload],
        }
    )

    result = await enrich(
        client,
        historical_collection([pull_request]),
    )

    assert result.items[0].affected_paths == ["src/app.py"]


@pytest.mark.anyio
async def test_result_serializes_evidence_counts_and_affected_paths() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): [
                file_payload("src/app.py"),
            ],
        }
    )

    result = await enrich(
        client,
        historical_collection([pull_request]),
    )
    data = result.model_dump(mode="json")

    assert data["items"][0]["item_type"] == "pull_request"
    assert data["items"][0]["affected_paths"] == ["src/app.py"]
    assert data["pull_request_evidence"][0]["complete"] is True
    assert data["pull_requests_available"] == 1
    assert data["pull_requests_enriched"] == 1
    assert data["total_count"] == 1
    assert data["complete_pull_request_count"] == 1


@pytest.mark.anyio
async def test_rest_errors_propagate_without_replacement() -> None:
    pull_request = knowledge_item(
        item_type=KnowledgeItemType.PULL_REQUEST,
        external_id="42",
    )
    collection = historical_collection([pull_request])
    error = GitHubRestResponseError(
        "API rate limit exceeded",
        status_code=429,
        rate_limited=True,
    )
    client = FakeGitHubRestClient(
        {
            (path_for_pull("42"), 1): error,
        }
    )

    with pytest.raises(GitHubRestResponseError) as error_info:
        await enrich(client, collection)

    assert error_info.value is error
    assert collection.items[0] is pull_request
    assert collection.items[0].affected_paths == []
