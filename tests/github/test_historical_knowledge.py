"""Tests for bounded historical GitHub knowledge collection."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import TypeAdapter, ValidationError

from opensteward.github import (
    GITHUB_HISTORICAL_PAGE_SIZE,
    MAX_GITHUB_HISTORICAL_ITEMS_PER_TYPE,
    MAX_GITHUB_HISTORICAL_SCAN_PAGES,
    GitHubHistoricalKnowledgeCollectionOptions,
    GitHubHistoricalKnowledgeCollectionResult,
    GitHubHistoricalKnowledgeCollectionStats,
    GitHubHistoricalKnowledgeCollector,
    GitHubRepositoryRef,
    GitHubRestResponse,
    knowledge_repository_from_github,
)
from opensteward.knowledge import (
    DecisionSignificance,
    KnowledgeActorType,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

CREATED_AT = datetime(2026, 1, 10, 9, 0, tzinfo=UTC)
UPDATED_AT = datetime(2026, 1, 12, 11, 0, tzinfo=UTC)
CLOSED_AT = datetime(2026, 1, 11, 10, 0, tzinfo=UTC)
ISSUES_PATH = "/repos/acme/framework/issues"
PULLS_PATH = "/repos/acme/framework/pulls"


def actor_payload(
    *,
    actor_id: int = 101,
    login: str = "maintainer",
    actor_type: str = "User",
) -> dict[str, object]:
    """Create a representative GitHub actor payload."""

    return {
        "id": actor_id,
        "login": login,
        "type": actor_type,
        "html_url": f"https://github.test/{login}",
    }


def issue_payload(
    number: int,
    *,
    updated_at: datetime = UPDATED_AT,
    user: dict[str, object] | None = None,
    labels: list[str] | None = None,
    pull_request_entry: bool = False,
) -> dict[str, object]:
    """Create one closed GitHub issue response."""

    return {
        "number": number,
        "title": f"Issue {number}",
        "body": f"Issue body {number}",
        "state": "closed",
        "html_url": f"https://github.test/acme/framework/issues/{number}",
        "user": actor_payload() if user is None else user,
        "labels": [
            {
                "name": label,
            }
            for label in (labels or [])
        ],
        "created_at": CREATED_AT.isoformat(),
        "updated_at": updated_at.isoformat(),
        "closed_at": CLOSED_AT.isoformat(),
        "pull_request": (
            {
                "url": f"https://api.github.test/pulls/{number}",
            }
            if pull_request_entry
            else None
        ),
    }


def pull_request_payload(
    number: int,
    *,
    updated_at: datetime = UPDATED_AT,
    merged: bool = False,
    user: dict[str, object] | None = None,
    labels: list[str] | None = None,
) -> dict[str, object]:
    """Create one closed GitHub pull-request response."""

    return {
        "number": number,
        "title": f"Pull request {number}",
        "body": f"Pull request body {number}",
        "state": "closed",
        "html_url": f"https://github.test/acme/framework/pull/{number}",
        "user": actor_payload() if user is None else user,
        "labels": [
            {
                "name": label,
            }
            for label in (labels or [])
        ],
        "created_at": CREATED_AT.isoformat(),
        "updated_at": updated_at.isoformat(),
        "closed_at": CLOSED_AT.isoformat(),
        "merged_at": CLOSED_AT.isoformat() if merged else None,
    }


class FakeGitHubRestClient:
    """Path-and-page-aware typed fake for historical collection."""

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


def create_repository(
    *,
    owner: str = "acme",
    name: str = "framework",
) -> GitHubRepositoryRef:
    """Create the repository used by collector tests."""

    return GitHubRepositoryRef(
        owner=owner,
        name=name,
    )


async def collect(
    client: FakeGitHubRestClient,
    *,
    max_issues: int = 0,
    max_pull_requests: int = 0,
    repository: GitHubRepositoryRef | None = None,
) -> GitHubHistoricalKnowledgeCollectionResult:
    """Run the collector with explicit feed limits."""

    collector = GitHubHistoricalKnowledgeCollector(
        rest_client=client,
    )
    return await collector.collect_closed_items(
        repository or create_repository(),
        options=GitHubHistoricalKnowledgeCollectionOptions(
            max_issues=max_issues,
            max_pull_requests=max_pull_requests,
        ),
    )


def collection_stats(
    *,
    requested_limit: int,
    items_collected: int,
) -> GitHubHistoricalKnowledgeCollectionStats:
    """Create consistent statistics for result-validation tests."""

    return GitHubHistoricalKnowledgeCollectionStats(
        requested_limit=requested_limit,
        pages_fetched=1 if requested_limit else 0,
        api_items_seen=items_collected,
        items_collected=items_collected,
        filtered_items=0,
        duplicate_items=0,
        item_limit_reached=items_collected == requested_limit,
        scan_limit_reached=False,
    )


def knowledge_item(
    *,
    item_type: KnowledgeItemType = KnowledgeItemType.ISSUE,
    external_id: str = "1",
    updated_at: datetime = UPDATED_AT,
    repository: KnowledgeRepositoryRef | None = None,
    source_kind: KnowledgeSourceKind = KnowledgeSourceKind.GITHUB,
) -> KnowledgeItem:
    """Create a knowledge item for public result validation."""

    return KnowledgeItem(
        repository=repository or knowledge_repository_from_github(create_repository()),
        item_type=item_type,
        external_id=external_id,
        source_kind=source_kind,
        state=KnowledgeItemState.CLOSED,
        title=f"Item {external_id}",
        created_at=CREATED_AT,
        updated_at=updated_at,
        closed_at=CLOSED_AT,
    )


def collection_result(
    items: list[KnowledgeItem],
) -> GitHubHistoricalKnowledgeCollectionResult:
    """Build a result with statistics matching its item types."""

    issue_count = sum(
        item.item_type == KnowledgeItemType.ISSUE
        for item in items
    )
    pull_request_count = sum(
        item.item_type == KnowledgeItemType.PULL_REQUEST
        for item in items
    )

    return GitHubHistoricalKnowledgeCollectionResult(
        repository=create_repository(),
        knowledge_repository=knowledge_repository_from_github(
            create_repository()
        ),
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


@pytest.mark.anyio
async def test_converts_closed_issue_to_knowledge_item() -> None:
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [issue_payload(42)],
        }
    )

    result = await collect(client, max_issues=10)
    item = result.items[0]

    assert item.item_type == KnowledgeItemType.ISSUE
    assert item.external_id == "42"
    assert item.source_kind == KnowledgeSourceKind.GITHUB
    assert item.state == KnowledgeItemState.CLOSED
    assert item.title == "Issue 42"
    assert item.body == "Issue body 42"
    assert item.summary is None
    assert item.url == "https://github.test/acme/framework/issues/42"
    assert item.created_at == CREATED_AT
    assert item.updated_at == UPDATED_AT
    assert item.closed_at == CLOSED_AT


@pytest.mark.anyio
async def test_converts_merged_pull_request_to_merged_state() -> None:
    client = FakeGitHubRestClient(
        {
            (PULLS_PATH, 1): [pull_request_payload(51, merged=True)],
        }
    )

    result = await collect(client, max_pull_requests=10)

    assert result.items[0].state == KnowledgeItemState.MERGED


@pytest.mark.anyio
async def test_converts_unmerged_pull_request_to_rejected_state() -> None:
    client = FakeGitHubRestClient(
        {
            (PULLS_PATH, 1): [pull_request_payload(52)],
        }
    )

    result = await collect(client, max_pull_requests=10)

    assert result.items[0].state == KnowledgeItemState.REJECTED


@pytest.mark.anyio
async def test_collection_does_not_infer_knowledge_metadata() -> None:
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [issue_payload(42)],
            (PULLS_PATH, 1): [pull_request_payload(51, merged=True)],
        }
    )

    result = await collect(
        client,
        max_issues=10,
        max_pull_requests=10,
    )

    assert all(
        item.decision_significance == DecisionSignificance.NONE
        for item in result.items
    )
    assert all(item.affected_paths == [] for item in result.items)
    assert all(item.components == [] for item in result.items)


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("github_type", "expected_type"),
    [
        ("User", KnowledgeActorType.USER),
        ("bot", KnowledgeActorType.BOT),
        ("ORGANIZATION", KnowledgeActorType.ORGANIZATION),
        ("Mannequin", KnowledgeActorType.UNKNOWN),
    ],
)
async def test_converts_github_actor_types(
    github_type: str,
    expected_type: KnowledgeActorType,
) -> None:
    user = actor_payload(
        actor_id=808,
        login="historian",
        actor_type=github_type,
    )
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [issue_payload(42, user=user)],
        }
    )

    result = await collect(client, max_issues=10)
    author = result.items[0].author

    assert author is not None
    assert author.identifier == "808"
    assert author.display_name == "historian"
    assert author.url == "https://github.test/historian"
    assert author.actor_type == expected_type


@pytest.mark.anyio
async def test_null_github_user_maps_to_none() -> None:
    payload = issue_payload(42)
    payload["user"] = None
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [payload],
        }
    )

    result = await collect(client, max_issues=10)

    assert result.items[0].author is None


@pytest.mark.anyio
async def test_deduplicates_labels_preserving_order_and_spelling() -> None:
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [
                issue_payload(
                    42,
                    labels=["bug", "Compatibility", "compatibility", "BUG"],
                )
            ],
        }
    )

    result = await collect(client, max_issues=10)

    assert result.items[0].labels == ["bug", "Compatibility"]


@pytest.mark.anyio
async def test_filters_pull_request_entries_from_issues() -> None:
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [
                issue_payload(10, pull_request_entry=True),
                issue_payload(11),
            ],
        }
    )

    result = await collect(client, max_issues=10)

    assert [item.external_id for item in result.items] == ["11"]
    assert result.issue_stats.filtered_items == 1


@pytest.mark.anyio
async def test_filtered_pull_requests_do_not_count_toward_issue_limit() -> None:
    first_page = [
        issue_payload(number, pull_request_entry=True)
        for number in range(1, GITHUB_HISTORICAL_PAGE_SIZE + 1)
    ]
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): first_page,
            (ISSUES_PATH, 2): [issue_payload(101)],
        }
    )

    result = await collect(client, max_issues=1)

    assert [item.external_id for item in result.items] == ["101"]
    assert result.issue_stats.pages_fetched == 2
    assert result.issue_stats.filtered_items == 100
    assert result.issue_stats.item_limit_reached is True


@pytest.mark.anyio
async def test_uses_exact_issue_endpoint_parameters() -> None:
    client = FakeGitHubRestClient()

    await collect(client, max_issues=10)

    assert client.calls[0]["path"] == ISSUES_PATH
    assert client.calls[0]["params"] == {
        "state": "closed",
        "filter": "all",
        "sort": "updated",
        "direction": "desc",
        "per_page": 100,
        "page": 1,
    }


@pytest.mark.anyio
async def test_uses_exact_pull_request_endpoint_parameters() -> None:
    client = FakeGitHubRestClient()

    await collect(client, max_pull_requests=10)

    assert client.calls[0]["path"] == PULLS_PATH
    assert client.calls[0]["params"] == {
        "state": "closed",
        "sort": "updated",
        "direction": "desc",
        "per_page": 100,
        "page": 1,
    }


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "item_factory", "limit_name"),
    [
        (ISSUES_PATH, issue_payload, "issues"),
        (PULLS_PATH, pull_request_payload, "pull_requests"),
    ],
)
async def test_paginates_after_full_page(
    path: str,
    item_factory: Any,
    limit_name: str,
) -> None:
    first_page = [
        item_factory(number)
        for number in range(1, GITHUB_HISTORICAL_PAGE_SIZE + 1)
    ]
    client = FakeGitHubRestClient(
        {
            (path, 1): first_page,
            (path, 2): [item_factory(101)],
        }
    )

    result = await collect(
        client,
        max_issues=101 if limit_name == "issues" else 0,
        max_pull_requests=101 if limit_name == "pull_requests" else 0,
    )
    stats = (
        result.issue_stats
        if limit_name == "issues"
        else result.pull_request_stats
    )

    assert stats.pages_fetched == 2
    assert stats.items_collected == 101


@pytest.mark.anyio
async def test_stops_on_short_page() -> None:
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [issue_payload(1), issue_payload(2)],
            (ISSUES_PATH, 2): [issue_payload(3)],
        }
    )

    result = await collect(client, max_issues=10)

    assert result.issue_stats.pages_fetched == 1
    assert len(client.calls) == 1
    assert result.issue_stats.scan_limit_reached is False


@pytest.mark.anyio
async def test_stops_when_unique_item_limit_is_reached() -> None:
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [
                issue_payload(1),
                issue_payload(2),
                issue_payload(3),
            ],
        }
    )

    result = await collect(client, max_issues=2)

    assert result.issue_count == 2
    assert result.issue_stats.api_items_seen == 3
    assert result.issue_stats.item_limit_reached is True


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("path", "item_factory", "limit_name"),
    [
        (ISSUES_PATH, issue_payload, "issues"),
        (PULLS_PATH, pull_request_payload, "pull_requests"),
    ],
)
async def test_deduplicates_repeated_keys_across_pages(
    path: str,
    item_factory: Any,
    limit_name: str,
) -> None:
    first_page = [
        item_factory(number)
        for number in range(1, GITHUB_HISTORICAL_PAGE_SIZE + 1)
    ]
    second_page = [
        item_factory(1),
        item_factory(101),
        item_factory(102),
    ]
    client = FakeGitHubRestClient(
        {
            (path, 1): first_page,
            (path, 2): second_page,
        }
    )

    result = await collect(
        client,
        max_issues=102 if limit_name == "issues" else 0,
        max_pull_requests=102 if limit_name == "pull_requests" else 0,
    )
    stats = (
        result.issue_stats
        if limit_name == "issues"
        else result.pull_request_stats
    )

    assert stats.items_collected == 102
    assert stats.duplicate_items == 1
    assert stats.item_limit_reached is True
    assert len({item.key for item in result.items}) == 102


@pytest.mark.anyio
async def test_zero_limits_skip_corresponding_endpoints() -> None:
    issue_client = FakeGitHubRestClient(
        {
            (PULLS_PATH, 1): [pull_request_payload(1)],
        }
    )
    issue_result = await collect(
        issue_client,
        max_issues=0,
        max_pull_requests=1,
    )

    assert all(call["path"] != ISSUES_PATH for call in issue_client.calls)
    assert issue_result.issue_stats.model_dump() == {
        "requested_limit": 0,
        "pages_fetched": 0,
        "api_items_seen": 0,
        "items_collected": 0,
        "filtered_items": 0,
        "duplicate_items": 0,
        "item_limit_reached": True,
        "scan_limit_reached": False,
    }

    pull_client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [issue_payload(1)],
        }
    )
    await collect(
        pull_client,
        max_issues=1,
        max_pull_requests=0,
    )

    assert all(call["path"] != PULLS_PATH for call in pull_client.calls)


@pytest.mark.parametrize("field_name", ["max_issues", "max_pull_requests"])
def test_options_reject_negative_limits(field_name: str) -> None:
    with pytest.raises(ValidationError):
        GitHubHistoricalKnowledgeCollectionOptions.model_validate(
            {
                field_name: -1,
            }
        )


@pytest.mark.parametrize("field_name", ["max_issues", "max_pull_requests"])
def test_options_reject_limits_above_safety_bound(field_name: str) -> None:
    with pytest.raises(ValidationError):
        GitHubHistoricalKnowledgeCollectionOptions.model_validate(
            {
                field_name: MAX_GITHUB_HISTORICAL_ITEMS_PER_TYPE + 1,
            }
        )


def test_builds_expected_knowledge_repository_reference() -> None:
    knowledge_repository = knowledge_repository_from_github(
        create_repository()
    )

    assert knowledge_repository == KnowledgeRepositoryRef(
        provider="github",
        namespace="acme",
        name="framework",
    )


@pytest.mark.anyio
async def test_knowledge_item_key_has_stable_format() -> None:
    client = FakeGitHubRestClient(
        {
            (PULLS_PATH, 1): [pull_request_payload(398)],
        }
    )

    result = await collect(client, max_pull_requests=1)

    assert result.items[0].key == "github:acme/framework:pull_request:398"


@pytest.mark.anyio
async def test_combines_and_sorts_final_items_deterministically() -> None:
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [
                issue_payload(9, updated_at=UPDATED_AT),
                issue_payload(1, updated_at=UPDATED_AT),
            ],
            (PULLS_PATH, 1): [
                pull_request_payload(
                    5,
                    updated_at=UPDATED_AT + timedelta(days=1),
                )
            ],
        }
    )

    result = await collect(
        client,
        max_issues=10,
        max_pull_requests=10,
    )

    assert [item.external_id for item in result.items] == ["5", "1", "9"]
    assert [
        item.item_type
        for item in result.items
    ] == [
        KnowledgeItemType.PULL_REQUEST,
        KnowledgeItemType.ISSUE,
        KnowledgeItemType.ISSUE,
    ]


@pytest.mark.anyio
async def test_result_computed_counts_are_correct() -> None:
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [issue_payload(1), issue_payload(2)],
            (PULLS_PATH, 1): [pull_request_payload(3)],
        }
    )

    result = await collect(
        client,
        max_issues=10,
        max_pull_requests=10,
    )

    assert result.total_count == 3
    assert result.issue_count == 2
    assert result.pull_request_count == 1


def test_result_rejects_duplicate_item_keys() -> None:
    item = knowledge_item()

    with pytest.raises(ValidationError, match="keys must be unique"):
        collection_result([item, item])


def test_result_rejects_item_from_another_repository() -> None:
    other_repository = KnowledgeRepositoryRef(
        provider="github",
        namespace="other",
        name="framework",
    )

    with pytest.raises(ValidationError, match="knowledge repository"):
        collection_result(
            [
                knowledge_item(repository=other_repository),
            ]
        )


def test_result_rejects_non_github_source_kind() -> None:
    with pytest.raises(ValidationError, match="GitHub source kind"):
        collection_result(
            [
                knowledge_item(source_kind=KnowledgeSourceKind.MANUAL),
            ]
        )


def test_result_rejects_unsupported_item_type() -> None:
    with pytest.raises(ValidationError, match="only issues and pull requests"):
        collection_result(
            [
                knowledge_item(item_type=KnowledgeItemType.DOCUMENTATION),
            ]
        )


@pytest.mark.anyio
async def test_collection_statistics_report_all_counters() -> None:
    first_page = [
        issue_payload(number, pull_request_entry=True)
        for number in range(1, 99)
    ]
    first_page.extend(
        [
            issue_payload(200),
            issue_payload(200),
        ]
    )
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): first_page,
            (ISSUES_PATH, 2): [issue_payload(201)],
        }
    )

    result = await collect(client, max_issues=3)
    stats = result.issue_stats

    assert stats.requested_limit == 3
    assert stats.pages_fetched == 2
    assert stats.api_items_seen == 101
    assert stats.items_collected == 2
    assert stats.filtered_items == 98
    assert stats.duplicate_items == 1
    assert stats.item_limit_reached is False
    assert stats.scan_limit_reached is False


@pytest.mark.anyio
async def test_reports_scan_page_safety_limit() -> None:
    full_duplicate_page = [
        issue_payload(1)
        for _ in range(GITHUB_HISTORICAL_PAGE_SIZE)
    ]
    responses = {
        (ISSUES_PATH, page): full_duplicate_page
        for page in range(1, MAX_GITHUB_HISTORICAL_SCAN_PAGES + 1)
    }
    client = FakeGitHubRestClient(responses)

    result = await collect(
        client,
        max_issues=MAX_GITHUB_HISTORICAL_ITEMS_PER_TYPE,
    )

    assert result.issue_stats.pages_fetched == 20
    assert result.issue_stats.api_items_seen == 2_000
    assert result.issue_stats.items_collected == 1
    assert result.issue_stats.duplicate_items == 1_999
    assert result.issue_stats.item_limit_reached is False
    assert result.issue_stats.scan_limit_reached is True


@pytest.mark.anyio
async def test_extra_github_response_fields_are_ignored() -> None:
    payload = issue_payload(42)
    payload["repository_url"] = "https://api.github.test/repos/acme/framework"
    payload["reactions"] = {
        "total_count": 3,
    }
    payload["user"]["site_admin"] = True  # type: ignore[index]
    payload["labels"][0:0] = [  # type: ignore[index]
        {
            "name": "bug",
            "color": "ff0000",
        }
    ]
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [payload],
        }
    )

    result = await collect(client, max_issues=10)

    assert result.items[0].external_id == "42"
    assert result.items[0].labels == ["bug"]


@pytest.mark.anyio
async def test_invalid_api_payload_is_not_silently_accepted() -> None:
    invalid_payload = issue_payload(42)
    invalid_payload["number"] = 0
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [invalid_payload],
        }
    )

    with pytest.raises(ValidationError):
        await collect(client, max_issues=10)


@pytest.mark.anyio
async def test_missing_required_api_field_is_rejected() -> None:
    invalid_payload = issue_payload(42)
    del invalid_payload["labels"]
    client = FakeGitHubRestClient(
        {
            (ISSUES_PATH, 1): [invalid_payload],
        }
    )

    with pytest.raises(ValidationError):
        await collect(client, max_issues=10)


def test_result_rejects_stats_that_do_not_match_item_counts() -> None:
    with pytest.raises(ValidationError, match="Issue statistics"):
        GitHubHistoricalKnowledgeCollectionResult(
            repository=create_repository(),
            knowledge_repository=knowledge_repository_from_github(
                create_repository()
            ),
            items=[knowledge_item()],
            issue_stats=collection_stats(
                requested_limit=100,
                items_collected=0,
            ),
            pull_request_stats=collection_stats(
                requested_limit=100,
                items_collected=0,
            ),
        )
