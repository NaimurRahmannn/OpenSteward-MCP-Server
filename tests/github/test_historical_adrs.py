"""Tests for repository ADR collection at an exact Git reference."""

import base64
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

import pytest
from pydantic import TypeAdapter, ValidationError

from opensteward.github import (
    DEFAULT_GITHUB_ADR_DIRECTORIES,
    GITHUB_ADR_FILE_EXTENSIONS,
    MAX_GITHUB_ADR_DIRECTORIES,
    MAX_GITHUB_ADR_FILE_BYTES,
    MAX_GITHUB_ADR_FILES,
    MAX_GITHUB_ADR_TOTAL_BYTES,
    GitHubHistoricalAdrCollectionError,
    GitHubHistoricalAdrCollectionOptions,
    GitHubHistoricalAdrCollectionResult,
    GitHubHistoricalAdrCollectionStats,
    GitHubHistoricalAdrCollector,
    GitHubHistoricalAdrFileEvidence,
    GitHubHistoricalAdrSkippedFile,
    GitHubHistoricalAdrSkipReason,
    GitHubHistoricalAdrTimestampBasis,
    GitHubRepositoryRef,
    GitHubRestResponse,
    GitHubRestResponseError,
    knowledge_repository_from_github,
)
from opensteward.knowledge import (
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

AUTHOR_DATE = datetime(2026, 3, 1, 8, 0, tzinfo=UTC)
COMMITTER_DATE = datetime(
    2026,
    3,
    2,
    14,
    30,
    tzinfo=timezone(timedelta(hours=6)),
)
COMMITTER_DATE_UTC = datetime(2026, 3, 2, 8, 30, tzinfo=UTC)
COMMIT_SHA = "abc123commit"
TREE_SHA = "def456tree"
REPOSITORY = GitHubRepositoryRef(
    owner="acme",
    name="framework",
)
KNOWLEDGE_REPOSITORY = knowledge_repository_from_github(REPOSITORY)
TREE_PATH = f"/repos/acme/framework/git/trees/{COMMIT_SHA}"


def commit_path(git_ref: str = "main") -> str:
    """Return the expected encoded commit endpoint."""

    return f"/repos/acme/framework/commits/{quote(git_ref, safe='')}"


def blob_path(blob_sha: str) -> str:
    """Return the expected blob endpoint."""

    return f"/repos/acme/framework/git/blobs/{blob_sha}"


def commit_payload(
    *,
    sha: str = COMMIT_SHA,
    author_date: datetime | None = AUTHOR_DATE,
    committer_date: datetime | None = COMMITTER_DATE,
) -> dict[str, object]:
    """Create a minimal commit response."""

    return {
        "sha": sha,
        "html_url": f"https://github.test/acme/framework/commit/{sha}",
        "commit": {
            "author": {
                "name": "Author",
                "email": "author@example.test",
                "date": (
                    author_date.isoformat()
                    if author_date is not None
                    else None
                ),
            },
            "committer": {
                "name": "Committer",
                "email": "committer@example.test",
                "date": (
                    committer_date.isoformat()
                    if committer_date is not None
                    else None
                ),
            },
        },
    }


def tree_entry(
    path: str,
    *,
    sha: str | None = None,
    size: int | None = None,
    mode: str = "100644",
    entry_type: str = "blob",
) -> dict[str, object]:
    """Create one recursive-tree entry."""

    return {
        "path": path,
        "mode": mode,
        "type": entry_type,
        "sha": sha or f"sha-{path.replace('/', '-')}",
        "size": size,
    }


def tree_payload(
    entries: list[dict[str, object]],
    *,
    truncated: bool = False,
    sha: str = TREE_SHA,
) -> dict[str, object]:
    """Create a recursive-tree response."""

    return {
        "sha": sha,
        "truncated": truncated,
        "tree": entries,
    }


def blob_payload(
    content: bytes,
    *,
    sha: str,
    encoding: str = "base64",
    reported_size: int | None = None,
    encoded_content: str | None = None,
) -> dict[str, object]:
    """Create one Git blob response."""

    return {
        "sha": sha,
        "size": len(content) if reported_size is None else reported_size,
        "encoding": encoding,
        "content": (
            base64.b64encode(content).decode("ascii")
            if encoded_content is None
            else encoded_content
        ),
    }


class FakeGitHubRestClient:
    """Path-aware typed fake for ADR collector tests."""

    def __init__(
        self,
        responses: Mapping[str, object] | None = None,
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

        outcome = self._responses[path]
        if isinstance(outcome, Exception):
            raise outcome

        data = outcome
        if response_type is not Any:
            data = TypeAdapter(response_type).validate_python(outcome)

        return GitHubRestResponse(
            status_code=200,
            data=data,
        )


def client_for_documents(
    documents: Mapping[str, bytes],
    *,
    entries: list[dict[str, object]] | None = None,
    commit: dict[str, object] | None = None,
    tree_truncated: bool = False,
) -> FakeGitHubRestClient:
    """Create a fake serving a commit, tree, and document blobs."""

    selected_entries = entries or [
        tree_entry(
            path,
            sha=f"blob-{index}",
            size=len(content),
        )
        for index, (path, content) in enumerate(
            documents.items(),
            start=1,
        )
    ]
    responses: dict[str, object] = {
        commit_path(): commit or commit_payload(),
        TREE_PATH: tree_payload(
            selected_entries,
            truncated=tree_truncated,
        ),
    }

    for entry in selected_entries:
        path = str(entry["path"])
        if path not in documents:
            continue

        sha = str(entry["sha"])
        responses[blob_path(sha)] = blob_payload(
            documents[path],
            sha=sha,
        )

    return FakeGitHubRestClient(responses)


async def collect(
    client: FakeGitHubRestClient,
    *,
    git_ref: str = "main",
    options: GitHubHistoricalAdrCollectionOptions | None = None,
    repository: GitHubRepositoryRef = REPOSITORY,
) -> GitHubHistoricalAdrCollectionResult:
    """Run the ADR collector."""

    collector = GitHubHistoricalAdrCollector(
        rest_client=client,
    )
    return await collector.collect(
        repository,
        git_ref=git_ref,
        options=options,
    )


def adr_item(
    path: str,
    *,
    repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
    item_type: KnowledgeItemType = KnowledgeItemType.ADR,
    source_kind: KnowledgeSourceKind = KnowledgeSourceKind.REPOSITORY_FILE,
    url: str | None = None,
) -> KnowledgeItem:
    """Create an ADR item for result-validation tests."""

    html_url = url or (
        f"https://github.com/acme/framework/blob/{COMMIT_SHA}/{path}"
    )
    return KnowledgeItem(
        repository=repository,
        item_type=item_type,
        external_id=path,
        source_kind=source_kind,
        state=KnowledgeItemState.UNKNOWN,
        title="ADR title",
        body="# ADR title",
        url=html_url,
        created_at=COMMITTER_DATE,
        updated_at=COMMITTER_DATE,
    )


def adr_evidence(
    item: KnowledgeItem,
    *,
    path: str | None = None,
) -> GitHubHistoricalAdrFileEvidence:
    """Create matching evidence for an ADR item."""

    evidence_path = path or item.external_id
    return GitHubHistoricalAdrFileEvidence(
        item_key=item.key,
        path=evidence_path,
        blob_sha=f"sha-{evidence_path}",
        size_bytes=len(item.body or ""),
        html_url=item.url or "https://github.test/adr",
    )


def adr_stats(
    *,
    items: int,
    skipped: int = 0,
    candidates: int | None = None,
    selected: int | None = None,
    total_limit: bool = False,
) -> GitHubHistoricalAdrCollectionStats:
    """Create internally valid ADR collection statistics."""

    candidate_count = (
        items + skipped
        if candidates is None
        else candidates
    )
    selected_count = (
        items + skipped
        if selected is None
        else selected
    )
    return GitHubHistoricalAdrCollectionStats(
        tree_entries_seen=candidate_count,
        candidate_files_seen=candidate_count,
        selected_files=selected_count,
        blobs_fetched=items,
        items_collected=items,
        skipped_files=skipped,
        decoded_bytes=items * 10,
        tree_truncated=False,
        item_limit_reached=candidate_count > selected_count,
        total_bytes_limit_reached=total_limit,
    )


def adr_result(
    items: list[KnowledgeItem],
    evidence: list[GitHubHistoricalAdrFileEvidence],
    *,
    skipped: list[GitHubHistoricalAdrSkippedFile] | None = None,
    stats: GitHubHistoricalAdrCollectionStats | None = None,
    knowledge_repository: KnowledgeRepositoryRef = KNOWLEDGE_REPOSITORY,
) -> GitHubHistoricalAdrCollectionResult:
    """Create a public ADR result for validation tests."""

    skipped_files = skipped or []
    return GitHubHistoricalAdrCollectionResult(
        repository=REPOSITORY,
        knowledge_repository=knowledge_repository,
        requested_ref="main",
        resolved_commit_sha=COMMIT_SHA,
        snapshot_commit_date=COMMITTER_DATE,
        tree_sha=TREE_SHA,
        items=items,
        file_evidence=evidence,
        skipped_files=skipped_files,
        stats=stats or adr_stats(
            items=len(items),
            skipped=len(skipped_files),
        ),
        warnings=[],
    )


def test_options_have_expected_default_directories() -> None:
    options = GitHubHistoricalAdrCollectionOptions()

    assert options.directories == list(DEFAULT_GITHUB_ADR_DIRECTORIES)
    assert GITHUB_ADR_FILE_EXTENSIONS == (".md", ".markdown")


def test_options_normalize_directory_paths() -> None:
    options = GitHubHistoricalAdrCollectionOptions(
        directories=[
            " .\\docs\\adr/// ",
            "./architecture/decisions/",
        ]
    )

    assert options.directories == [
        "docs/adr",
        "architecture/decisions",
    ]


def test_options_reject_duplicate_normalized_directories() -> None:
    with pytest.raises(ValidationError, match="unique"):
        GitHubHistoricalAdrCollectionOptions(
            directories=[
                "docs/adr",
                ".\\docs\\adr\\",
            ]
        )


@pytest.mark.parametrize(
    "directory",
    [
        "",
        "/docs/adr",
        "C:\\docs\\adr",
        "C:docs\\adr",
        "../docs/adr",
        "docs/../adr",
        "docs//adr",
        "docs/./adr",
    ],
)
def test_options_reject_unsafe_directories(directory: str) -> None:
    with pytest.raises(ValidationError):
        GitHubHistoricalAdrCollectionOptions(
            directories=[directory]
        )


def test_options_reject_too_many_directories() -> None:
    with pytest.raises(ValidationError):
        GitHubHistoricalAdrCollectionOptions(
            directories=[
                f"adr/{index}"
                for index in range(MAX_GITHUB_ADR_DIRECTORIES + 1)
            ]
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_files", -1),
        ("max_files", MAX_GITHUB_ADR_FILES + 1),
        ("max_file_bytes", 0),
        ("max_file_bytes", MAX_GITHUB_ADR_FILE_BYTES + 1),
        ("max_total_bytes", 0),
        ("max_total_bytes", MAX_GITHUB_ADR_TOTAL_BYTES + 1),
    ],
)
def test_options_reject_out_of_bounds_limits(
    field_name: str,
    value: int,
) -> None:
    with pytest.raises(ValidationError):
        GitHubHistoricalAdrCollectionOptions.model_validate(
            {
                field_name: value,
            }
        )


@pytest.mark.anyio
async def test_resolves_encoded_ref_and_prefers_committer_date() -> None:
    git_ref = "refs/heads/release candidate"
    encoded_path = commit_path(git_ref)
    client = FakeGitHubRestClient(
        {
            encoded_path: commit_payload(),
            TREE_PATH: tree_payload([]),
        }
    )

    result = await collect(client, git_ref=git_ref)

    assert client.calls[0]["path"] == encoded_path
    assert result.requested_ref == git_ref
    assert result.snapshot_commit_date == COMMITTER_DATE_UTC
    assert result.snapshot_commit_date.tzinfo is UTC


@pytest.mark.anyio
async def test_falls_back_to_author_date() -> None:
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(committer_date=None),
            TREE_PATH: tree_payload([]),
        }
    )

    result = await collect(client)

    assert result.snapshot_commit_date == AUTHOR_DATE


@pytest.mark.anyio
async def test_missing_commit_dates_raise_dedicated_error() -> None:
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(
                author_date=None,
                committer_date=None,
            ),
        }
    )

    with pytest.raises(
        GitHubHistoricalAdrCollectionError,
        match="snapshot date",
    ):
        await collect(client)

    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_uses_resolved_sha_and_exact_recursive_tree_request() -> None:
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(),
            TREE_PATH: tree_payload([]),
        }
    )

    result = await collect(client)

    assert client.calls[1]["path"] == TREE_PATH
    assert client.calls[1]["params"] == {
        "recursive": "1",
    }
    assert result.resolved_commit_sha == COMMIT_SHA
    assert result.tree_sha == TREE_SHA


@pytest.mark.anyio
async def test_extra_commit_and_tree_fields_are_ignored() -> None:
    commit = commit_payload()
    commit["parents"] = [{"sha": "parent"}]
    commit["commit"]["message"] = "ADR snapshot"  # type: ignore[index]
    tree = tree_payload([])
    tree["url"] = "https://api.github.test/tree"
    client = FakeGitHubRestClient(
        {
            commit_path(): commit,
            TREE_PATH: tree,
        }
    )

    result = await collect(client)

    assert result.total_count == 0


@pytest.mark.anyio
@pytest.mark.parametrize("invalid_response", ["commit", "tree"])
async def test_invalid_commit_or_tree_payload_is_rejected(
    invalid_response: str,
) -> None:
    commit = commit_payload()
    tree = tree_payload([])
    if invalid_response == "commit":
        commit["sha"] = ""
    else:
        del tree["tree"]

    client = FakeGitHubRestClient(
        {
            commit_path(): commit,
            TREE_PATH: tree,
        }
    )

    with pytest.raises(ValidationError):
        await collect(client)


@pytest.mark.anyio
async def test_empty_ref_is_rejected_before_request() -> None:
    client = FakeGitHubRestClient()

    with pytest.raises(ValueError, match="must not be empty"):
        await collect(client, git_ref="   ")

    assert client.calls == []


@pytest.mark.anyio
async def test_discovery_filters_by_directory_extension_type_and_mode() -> None:
    documents = {
        "docs/adr/0001-first.md": b"# First",
        "docs/adr/sub/0002-second.MARKDOWN": b"# Second",
    }
    entries = [
        tree_entry(
            "docs/adr/0001-first.md",
            sha="blob-first",
            size=len(documents["docs/adr/0001-first.md"]),
        ),
        tree_entry(
            "docs/adr/sub/0002-second.MARKDOWN",
            sha="blob-second",
            size=len(documents["docs/adr/sub/0002-second.MARKDOWN"]),
        ),
        tree_entry("docs/adrs/not-segment-match.md"),
        tree_entry("other/docs/adr/not-prefix-match.md"),
        tree_entry("docs/adr/readme.txt"),
        tree_entry(
            "docs/adr/directory.md",
            entry_type="tree",
        ),
        tree_entry(
            "docs/adr/symlink.md",
            mode="120000",
        ),
    ]
    client = client_for_documents(
        documents,
        entries=entries,
    )

    result = await collect(
        client,
        options=GitHubHistoricalAdrCollectionOptions(
            directories=["docs/adr"],
        ),
    )

    assert [item.external_id for item in result.items] == [
        "docs/adr/0001-first.md",
        "docs/adr/sub/0002-second.MARKDOWN",
    ]
    assert result.stats.tree_entries_seen == 7
    assert result.stats.candidate_files_seen == 2


@pytest.mark.anyio
async def test_candidates_deduplicate_sort_before_file_limit() -> None:
    documents = {
        "docs/adr/a-first.md": b"# First",
        "docs/adr/z-last.md": b"# Last",
    }
    entries = [
        tree_entry(
            "docs/adr/z-last.md",
            sha="blob-z",
            size=len(documents["docs/adr/z-last.md"]),
        ),
        tree_entry(
            "docs/adr/a-first.md",
            sha="blob-a",
            size=len(documents["docs/adr/a-first.md"]),
        ),
        tree_entry(
            ".\\docs\\adr\\a-first.md",
            sha="duplicate-a",
            size=10,
        ),
    ]
    client = client_for_documents(
        documents,
        entries=entries,
    )

    result = await collect(
        client,
        options=GitHubHistoricalAdrCollectionOptions(
            directories=["docs/adr"],
            max_files=1,
        ),
    )

    assert [item.external_id for item in result.items] == [
        "docs/adr/a-first.md",
    ]
    assert result.stats.candidate_files_seen == 2
    assert result.stats.selected_files == 1
    assert result.stats.item_limit_reached is True
    assert result.complete is False
    assert client.calls[2]["path"] == blob_path("blob-a")


@pytest.mark.anyio
async def test_zero_file_limit_discovers_without_fetching_blobs() -> None:
    documents = {
        "docs/adr/0001-first.md": b"# First",
    }
    client = client_for_documents(documents)

    result = await collect(
        client,
        options=GitHubHistoricalAdrCollectionOptions(
            max_files=0,
        ),
    )

    assert len(client.calls) == 2
    assert result.items == []
    assert result.stats.candidate_files_seen == 1
    assert result.stats.selected_files == 0
    assert result.stats.blobs_fetched == 0
    assert result.stats.item_limit_reached is True
    assert result.complete is False


@pytest.mark.anyio
async def test_truncated_tree_raises_before_blob_by_default() -> None:
    documents = {
        "docs/adr/0001-first.md": b"# First",
    }
    client = client_for_documents(
        documents,
        tree_truncated=True,
    )

    with pytest.raises(
        GitHubHistoricalAdrCollectionError,
        match="truncated",
    ):
        await collect(client)

    assert len(client.calls) == 2


@pytest.mark.anyio
async def test_allowed_truncated_tree_warns_and_is_incomplete() -> None:
    documents = {
        "docs/adr/0001-first.md": b"# First",
    }
    client = client_for_documents(
        documents,
        tree_truncated=True,
    )

    result = await collect(
        client,
        options=GitHubHistoricalAdrCollectionOptions(
            require_complete_tree=False,
        ),
    )

    assert result.stats.tree_truncated is True
    assert result.complete is False
    assert result.warnings == [
        (
            "GitHub returned a truncated recursive tree; "
            "ADR discovery may be incomplete."
        ),
        (
            "ADR created_at and updated_at use the repository snapshot "
            "commit time; per-file history was not collected."
        ),
    ]


@pytest.mark.anyio
async def test_valid_blob_becomes_fully_normalized_adr_item() -> None:
    path = "docs/adr/0001-use-postgres.md"
    content = b"# Use PostgreSQL\n\nDecision body.\n"
    entries = [
        tree_entry(
            path,
            sha="blob-postgres",
            size=len(content),
        )
    ]
    client = client_for_documents(
        {
            path: content,
        },
        entries=entries,
    )

    result = await collect(client)
    item = result.items[0]
    evidence = result.file_evidence[0]

    assert client.calls[2]["path"] == blob_path("blob-postgres")
    assert item.item_type == KnowledgeItemType.ADR
    assert item.source_kind == KnowledgeSourceKind.REPOSITORY_FILE
    assert item.state == KnowledgeItemState.UNKNOWN
    assert item.external_id == path
    assert item.title == "Use PostgreSQL"
    assert item.body == content.decode("utf-8").strip()
    assert item.repository == KNOWLEDGE_REPOSITORY
    assert item.created_at == COMMITTER_DATE_UTC
    assert item.updated_at == COMMITTER_DATE_UTC
    assert item.author is None
    assert item.closed_at is None
    assert item.labels == []
    assert item.affected_paths == []
    assert item.components == []
    assert item.decision_significance == DecisionSignificance.NONE
    assert item.url == (
        "https://github.com/acme/framework/blob/"
        f"{COMMIT_SHA}/{path}"
    )
    assert evidence.item_key == item.key
    assert evidence.path == path
    assert evidence.blob_sha == "blob-postgres"
    assert evidence.size_bytes == len(content)
    assert (
        evidence.timestamp_basis
        == GitHubHistoricalAdrTimestampBasis.SNAPSHOT_COMMIT
    )
    assert result.stats.decoded_bytes == len(content)
    assert result.complete is True


@pytest.mark.anyio
async def test_blobs_are_fetched_sequentially_in_path_order() -> None:
    documents = {
        "docs/adr/b.md": b"# B",
        "docs/adr/a.md": b"# A",
    }
    client = client_for_documents(documents)

    result = await collect(client)

    assert [item.external_id for item in result.items] == [
        "docs/adr/a.md",
        "docs/adr/b.md",
    ]
    expected_shas = {
        path: f"blob-{index}"
        for index, path in enumerate(documents, start=1)
    }
    assert [call["path"] for call in client.calls[2:]] == [
        blob_path(expected_shas["docs/adr/a.md"]),
        blob_path(expected_shas["docs/adr/b.md"]),
    ]


@pytest.mark.anyio
async def test_base64_whitespace_and_utf8_bom_are_handled() -> None:
    path = "docs/adr/bom.md"
    content = b"\xef\xbb\xbf# BOM title\nBody"
    entry = tree_entry(
        path,
        sha="blob-bom",
        size=len(content),
    )
    encoded = base64.b64encode(content).decode("ascii")
    spaced = f" \n{encoded[:4]}\t{encoded[4:]}\r\n "
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(),
            TREE_PATH: tree_payload([entry]),
            blob_path("blob-bom"): blob_payload(
                content,
                sha="blob-bom",
                encoded_content=spaced,
            ),
        }
    )

    result = await collect(client)

    assert result.items[0].body == "# BOM title\nBody"
    assert result.items[0].title == "BOM title"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("case_name", "entry_updates", "blob_updates", "message"),
    [
        (
            "encoding",
            {},
            {
                "encoding": "utf-8",
            },
            "unsupported blob encoding",
        ),
        (
            "base64",
            {},
            {
                "content": "%%%not-base64%%%",
            },
            "malformed Base64",
        ),
        (
            "sha",
            {},
            {
                "sha": "wrong-sha",
            },
            "SHA",
        ),
        (
            "blob_size",
            {},
            {
                "size": 999,
            },
            "blob size",
        ),
        (
            "tree_size",
            {
                "size": 999,
            },
            {},
            "tree and blob sizes",
        ),
    ],
)
async def test_blob_integrity_failures_raise_dedicated_error(
    case_name: str,
    entry_updates: dict[str, object],
    blob_updates: dict[str, object],
    message: str,
) -> None:
    del case_name
    path = "docs/adr/integrity.md"
    content = b"# Integrity"
    entry = tree_entry(
        path,
        sha="blob-integrity",
        size=len(content),
    )
    entry.update(entry_updates)
    blob = blob_payload(
        content,
        sha="blob-integrity",
    )
    blob.update(blob_updates)
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(),
            TREE_PATH: tree_payload([entry]),
            blob_path("blob-integrity"): blob,
        }
    )

    with pytest.raises(
        GitHubHistoricalAdrCollectionError,
        match=message,
    ):
        await collect(client)


@pytest.mark.anyio
async def test_extra_blob_fields_are_ignored() -> None:
    path = "docs/adr/extra.md"
    content = b"# Extra"
    entry = tree_entry(
        path,
        sha="blob-extra",
        size=len(content),
    )
    blob = blob_payload(content, sha="blob-extra")
    blob["url"] = "https://api.github.test/blob"
    blob["node_id"] = "NODE"
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(),
            TREE_PATH: tree_payload([entry]),
            blob_path("blob-extra"): blob,
        }
    )

    result = await collect(client)

    assert result.total_count == 1


@pytest.mark.anyio
async def test_invalid_blob_payload_is_not_silently_accepted() -> None:
    path = "docs/adr/invalid.md"
    entry = tree_entry(
        path,
        sha="blob-invalid",
        size=10,
    )
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(),
            TREE_PATH: tree_payload([entry]),
            blob_path("blob-invalid"): {
                "sha": "blob-invalid",
                "size": -1,
                "encoding": "base64",
                "content": "",
            },
        }
    )

    with pytest.raises(ValidationError):
        await collect(client)


@pytest.mark.anyio
async def test_known_oversized_entry_skips_without_blob_request() -> None:
    path = "docs/adr/large.md"
    entry = tree_entry(
        path,
        sha="blob-large",
        size=2,
    )
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(),
            TREE_PATH: tree_payload([entry]),
        }
    )

    result = await collect(
        client,
        options=GitHubHistoricalAdrCollectionOptions(
            max_file_bytes=1,
        ),
    )

    assert len(client.calls) == 2
    assert result.skipped_files[0].reason == (
        GitHubHistoricalAdrSkipReason.FILE_TOO_LARGE
    )
    assert result.stats.blobs_fetched == 0
    assert result.stats.decoded_bytes == 0
    assert result.complete is False
    assert result.warnings == []


@pytest.mark.anyio
async def test_unknown_size_oversized_blob_skips_after_fetch() -> None:
    path = "docs/adr/large.md"
    content = b"##"
    entry = tree_entry(
        path,
        sha="blob-large",
        size=None,
    )
    client = FakeGitHubRestClient(
        {
            commit_path(): commit_payload(),
            TREE_PATH: tree_payload([entry]),
            blob_path("blob-large"): blob_payload(
                content,
                sha="blob-large",
            ),
        }
    )

    result = await collect(
        client,
        options=GitHubHistoricalAdrCollectionOptions(
            max_file_bytes=1,
        ),
    )

    assert result.stats.blobs_fetched == 1
    assert result.skipped_files[0].reported_size_bytes == 2
    assert result.skipped_files[0].reason == (
        GitHubHistoricalAdrSkipReason.FILE_TOO_LARGE
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("content", "expected_reason"),
    [
        (
            b"\xff\xfe",
            GitHubHistoricalAdrSkipReason.NON_UTF8_CONTENT,
        ),
        (
            b" \n\t ",
            GitHubHistoricalAdrSkipReason.EMPTY_DOCUMENT,
        ),
    ],
)
async def test_skippable_blob_content_is_recorded(
    content: bytes,
    expected_reason: GitHubHistoricalAdrSkipReason,
) -> None:
    path = "docs/adr/skipped.md"
    client = client_for_documents(
        {
            path: content,
        }
    )

    result = await collect(client)

    assert result.items == []
    assert result.file_evidence == []
    assert result.skipped_files[0].reason == expected_reason
    assert result.stats.decoded_bytes == 0
    assert result.warnings == []
    assert result.complete is False


@pytest.mark.anyio
async def test_total_byte_limit_stops_and_classifies_remaining_files() -> None:
    documents = {
        "docs/adr/a.md": b"# A",
        "docs/adr/b.md": b"# B",
        "docs/adr/c.md": b"# C",
    }
    client = client_for_documents(documents)

    result = await collect(
        client,
        options=GitHubHistoricalAdrCollectionOptions(
            max_total_bytes=4,
        ),
    )

    assert [item.external_id for item in result.items] == [
        "docs/adr/a.md",
    ]
    assert [skipped.path for skipped in result.skipped_files] == [
        "docs/adr/b.md",
        "docs/adr/c.md",
    ]
    assert all(
        skipped.reason
        == GitHubHistoricalAdrSkipReason.TOTAL_BYTES_LIMIT
        for skipped in result.skipped_files
    )
    assert len(client.calls) == 3
    assert result.stats.blobs_fetched == 1
    assert result.stats.decoded_bytes == len(b"# A")
    assert result.stats.total_bytes_limit_reached is True
    assert result.complete is False


@pytest.mark.anyio
async def test_decoded_bytes_excludes_skipped_content() -> None:
    documents = {
        "docs/adr/a-valid.md": b"# Valid",
        "docs/adr/b-invalid.md": b"\xff",
    }
    client = client_for_documents(documents)

    result = await collect(client)

    assert result.stats.items_collected == 1
    assert result.stats.skipped_files == 1
    assert result.stats.selected_files == 2
    assert result.stats.blobs_fetched == 2
    assert result.stats.decoded_bytes == len(b"# Valid")


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("text", "expected_title"),
    [
        ("# First title\n# Second title", "First title"),
        ("   # Indented title", "Indented title"),
        ("# Marked title ###", "Marked title"),
        ("# Adopt C#", "Adopt C#"),
        ("# Support F#", "Support F#"),
        ("# Adopt PostgreSQL ###", "Adopt PostgreSQL"),
        ("# Adopt PostgreSQL   ###   ", "Adopt PostgreSQL"),
        ("## Not an H1", "runtime"),
        ("# ###\n# Later title", "Later title"),
        ("No heading", "runtime"),
        ("## ADR 0007-use-cache", "0007-use-cache"),
    ],
)
async def test_markdown_title_extraction(
    text: str,
    expected_title: str,
) -> None:
    filename = (
        "0007-use-cache.md"
        if expected_title == "0007-use-cache"
        else "runtime.markdown"
    )
    path = f"docs/adr/{filename}"
    client = client_for_documents(
        {
            path: text.encode("utf-8"),
        }
    )

    result = await collect(client)

    assert result.items[0].title == expected_title


@pytest.mark.anyio
async def test_html_url_percent_encodes_segments_at_resolved_sha() -> None:
    repository = GitHubRepositoryRef(
        owner="acme org",
        name="framework tools",
    )
    path = "docs/adr/decision space.md"
    content = b"# Encoded URL"
    entry = tree_entry(
        path,
        sha="blob-url",
        size=len(content),
    )
    repository_prefix = "/repos/acme%20org/framework%20tools"
    client = FakeGitHubRestClient(
        {
            f"{repository_prefix}/commits/main": commit_payload(),
            f"{repository_prefix}/git/trees/{COMMIT_SHA}": tree_payload(
                [entry]
            ),
            f"{repository_prefix}/git/blobs/blob-url": blob_payload(
                content,
                sha="blob-url",
            ),
        }
    )

    result = await collect(
        client,
        repository=repository,
    )

    assert result.items[0].url == (
        "https://github.com/acme%20org/framework%20tools/"
        f"blob/{COMMIT_SHA}/docs/adr/decision%20space.md"
    )


@pytest.mark.anyio
async def test_output_order_counts_completeness_and_timestamp_warning() -> None:
    documents = {
        "docs/adr/z.md": b"# Z",
        "docs/adr/a.md": b"# A",
    }
    client = client_for_documents(documents)

    result = await collect(client)

    assert [item.external_id for item in result.items] == [
        "docs/adr/a.md",
        "docs/adr/z.md",
    ]
    assert [evidence.path for evidence in result.file_evidence] == [
        "docs/adr/a.md",
        "docs/adr/z.md",
    ]
    assert result.total_count == 2
    assert result.stats.items_collected == 2
    assert result.stats.skipped_files == 0
    assert result.complete is True
    assert result.warnings == [
        (
            "ADR created_at and updated_at use the repository snapshot "
            "commit time; per-file history was not collected."
        )
    ]


def test_result_rejects_cross_repository_non_adr_and_wrong_source() -> None:
    other_repository = KnowledgeRepositoryRef(
        provider="github",
        namespace="other",
        name="framework",
    )

    invalid_items = [
        adr_item(
            "docs/adr/other.md",
            repository=other_repository,
        ),
        adr_item(
            "docs/adr/issue.md",
            item_type=KnowledgeItemType.ISSUE,
        ),
        adr_item(
            "docs/adr/manual.md",
            source_kind=KnowledgeSourceKind.MANUAL,
        ),
    ]
    messages = [
        "knowledge_repository",
        "only ADR",
        "repository-file",
    ]

    for item, message in zip(invalid_items, messages, strict=True):
        with pytest.raises(ValidationError, match=message):
            adr_result(
                [item],
                [adr_evidence(item)],
            )


def test_result_rejects_duplicate_items_and_missing_evidence_item() -> None:
    item = adr_item("docs/adr/a.md")
    evidence = adr_evidence(item)

    with pytest.raises(ValidationError, match="item keys must be unique"):
        adr_result(
            [item, item],
            [evidence, evidence],
            stats=adr_stats(items=2),
        )

    missing = GitHubHistoricalAdrFileEvidence(
        item_key="github:acme/framework:adr:docs/adr/missing.md",
        path="docs/adr/missing.md",
        blob_sha="missing",
        size_bytes=10,
        html_url="https://github.test/missing",
    )
    with pytest.raises(ValidationError, match="identify an item"):
        adr_result([item], [missing])


def test_result_rejects_duplicate_evidence_paths() -> None:
    first = adr_item("docs/adr/a.md")
    second = adr_item("docs/adr/b.md")
    duplicated_path = "docs/adr/a.md"

    with pytest.raises(ValidationError, match="evidence paths must be unique"):
        adr_result(
            [first, second],
            [
                adr_evidence(first, path=duplicated_path),
                adr_evidence(second, path=duplicated_path),
            ],
        )


def test_result_rejects_inconsistent_stats_and_repository_identity() -> None:
    item = adr_item("docs/adr/a.md")
    evidence = adr_evidence(item)
    inconsistent_stats = adr_stats(items=0)

    with pytest.raises(ValidationError, match="item count"):
        adr_result(
            [item],
            [evidence],
            stats=inconsistent_stats,
        )

    wrong_repository = KnowledgeRepositoryRef(
        provider="github",
        namespace="wrong",
        name="framework",
    )
    with pytest.raises(ValidationError, match="GitHub repository identity"):
        adr_result(
            [],
            [],
            knowledge_repository=wrong_repository,
            stats=adr_stats(items=0),
        )


def test_result_rejects_unsorted_items_evidence_and_skips() -> None:
    first = adr_item("docs/adr/a.md")
    second = adr_item("docs/adr/b.md")

    with pytest.raises(ValidationError, match="path-ascending"):
        adr_result(
            [second, first],
            [adr_evidence(second), adr_evidence(first)],
        )

    skipped = [
        GitHubHistoricalAdrSkippedFile(
            path="docs/adr/z.md",
            blob_sha="z",
            reported_size_bytes=10,
            reason=GitHubHistoricalAdrSkipReason.FILE_TOO_LARGE,
        ),
        GitHubHistoricalAdrSkippedFile(
            path="docs/adr/a.md",
            blob_sha="a",
            reported_size_bytes=10,
            reason=GitHubHistoricalAdrSkipReason.FILE_TOO_LARGE,
        ),
    ]
    with pytest.raises(ValidationError, match="path-ascending"):
        adr_result(
            [],
            [],
            skipped=skipped,
            stats=adr_stats(items=0, skipped=2),
        )


@pytest.mark.anyio
async def test_result_serializes_domain_values_and_computed_fields() -> None:
    path = "docs/adr/serialize.md"
    client = client_for_documents(
        {
            path: b"# Serialize",
        }
    )

    result = await collect(client)
    data = result.model_dump(mode="json")

    assert data["snapshot_commit_date"] == "2026-03-02T08:30:00Z"
    assert data["items"][0]["item_type"] == "adr"
    assert data["items"][0]["source_kind"] == "repository_file"
    assert data["items"][0]["key"] == (
        "github:acme/framework:adr:docs/adr/serialize.md"
    )
    assert data["file_evidence"][0]["timestamp_basis"] == "snapshot_commit"
    assert data["total_count"] == 1
    assert data["complete"] is True


@pytest.mark.anyio
async def test_rest_errors_propagate_unchanged() -> None:
    error = GitHubRestResponseError(
        "Resource not accessible",
        status_code=403,
    )
    client = FakeGitHubRestClient(
        {
            commit_path(): error,
        }
    )

    with pytest.raises(GitHubRestResponseError) as error_info:
        await collect(client)

    assert error_info.value is error
