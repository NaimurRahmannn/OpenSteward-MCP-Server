"""Bounded collection of repository ADRs from an exact Git reference."""

import base64
import binascii
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import PurePosixPath
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

from opensteward.github.historical_knowledge import (
    knowledge_repository_from_github,
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
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
)

DEFAULT_GITHUB_ADR_DIRECTORIES: tuple[str, ...] = (
    "adr",
    "adrs",
    "doc/adr",
    "doc/adrs",
    "docs/adr",
    "docs/adrs",
    "architecture/decisions",
    "docs/architecture/decisions",
)

GITHUB_ADR_FILE_EXTENSIONS: tuple[str, ...] = (
    ".md",
    ".markdown",
)

MAX_GITHUB_ADR_DIRECTORIES = 20
MAX_GITHUB_ADR_FILES = 100
MAX_GITHUB_ADR_FILE_BYTES = 512 * 1024
MAX_GITHUB_ADR_TOTAL_BYTES = 5 * 1024 * 1024

_TREE_TRUNCATION_WARNING = (
    "GitHub returned a truncated recursive tree; ADR discovery may be incomplete."
)
_TIMESTAMP_WARNING = (
    "ADR created_at and updated_at use the repository snapshot commit time; "
    "per-file history was not collected."
)
_REGULAR_FILE_MODES = {
    "100644",
    "100755",
}
_ASCII_WHITESPACE_PATTERN = re.compile(r"[ \t\r\n\f\v]+")


class GitHubHistoricalAdrCollectionError(ValueError):
    """Raised when local ADR collection integrity checks fail."""


class GitHubHistoricalAdrSkipReason(StrEnum):
    """Reasons an ADR candidate did not become a knowledge item."""

    FILE_TOO_LARGE = "file_too_large"
    TOTAL_BYTES_LIMIT = "total_bytes_limit"
    NON_UTF8_CONTENT = "non_utf8_content"
    EMPTY_DOCUMENT = "empty_document"


class GitHubHistoricalAdrTimestampBasis(StrEnum):
    """Timestamp provenance available for collected ADRs."""

    SNAPSHOT_COMMIT = "snapshot_commit"


def _normalize_repository_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()

    while normalized.startswith("./"):
        normalized = normalized[2:]

    if not normalized:
        raise ValueError("Repository paths must not be empty.")

    if normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError("Repository paths must be repository-relative.")

    parts = normalized.split("/")
    if any(part == "" for part in parts):
        raise ValueError("Repository paths must not contain empty segments.")

    if any(part in {".", ".."} for part in parts):
        raise ValueError("Repository paths must not contain '.' or '..' segments.")

    return normalized


def _normalize_repository_directory(directory: str) -> str:
    normalized = directory.replace("\\", "/").strip()

    while normalized.startswith("./"):
        normalized = normalized[2:]

    normalized = normalized.rstrip("/")
    return _normalize_repository_path(normalized)


class GitHubHistoricalAdrCollectionOptions(StrictGitHubModel):
    """Caller-controlled ADR discovery and content limits."""

    directories: list[str] = Field(
        default_factory=lambda: list(DEFAULT_GITHUB_ADR_DIRECTORIES),
        max_length=MAX_GITHUB_ADR_DIRECTORIES,
    )
    max_files: int = Field(
        default=50,
        ge=0,
        le=MAX_GITHUB_ADR_FILES,
    )
    max_file_bytes: int = Field(
        default=256 * 1024,
        ge=1,
        le=MAX_GITHUB_ADR_FILE_BYTES,
    )
    max_total_bytes: int = Field(
        default=2 * 1024 * 1024,
        ge=1,
        le=MAX_GITHUB_ADR_TOTAL_BYTES,
    )
    require_complete_tree: bool = True

    @field_validator("directories")
    @classmethod
    def normalize_directories(cls, directories: list[str]) -> list[str]:
        """Normalize and deduplicate repository-relative ADR directories."""

        normalized = [
            _normalize_repository_directory(directory)
            for directory in directories
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError("ADR directories must be unique after normalization.")

        return normalized


class GitHubHistoricalAdrSkippedFile(StrictGitHubModel):
    """An ADR candidate skipped under a documented local rule."""

    path: str = Field(min_length=1)
    blob_sha: str = Field(min_length=1)
    reported_size_bytes: int | None = Field(default=None, ge=0)
    reason: GitHubHistoricalAdrSkipReason

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        """Normalize and validate the skipped repository path."""

        return _normalize_repository_path(path)


class GitHubHistoricalAdrFileEvidence(StrictGitHubModel):
    """Source and timestamp evidence for one collected ADR."""

    item_key: str = Field(min_length=1)
    path: str = Field(min_length=1)
    blob_sha: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    html_url: str = Field(min_length=1)
    timestamp_basis: GitHubHistoricalAdrTimestampBasis = (
        GitHubHistoricalAdrTimestampBasis.SNAPSHOT_COMMIT
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, path: str) -> str:
        """Normalize and validate the evidence repository path."""

        return _normalize_repository_path(path)


class GitHubHistoricalAdrCollectionStats(StrictGitHubModel):
    """Bounded ADR discovery and collection counters."""

    tree_entries_seen: int = Field(ge=0)
    candidate_files_seen: int = Field(ge=0)
    selected_files: int = Field(ge=0)
    blobs_fetched: int = Field(ge=0)
    items_collected: int = Field(ge=0)
    skipped_files: int = Field(ge=0)
    decoded_bytes: int = Field(
        ge=0,
        le=MAX_GITHUB_ADR_TOTAL_BYTES,
    )
    tree_truncated: bool
    item_limit_reached: bool
    total_bytes_limit_reached: bool

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        """Reject internally inconsistent ADR collection counters."""

        if self.candidate_files_seen > self.tree_entries_seen:
            raise ValueError("candidate_files_seen must not exceed tree_entries_seen.")

        if self.selected_files > self.candidate_files_seen:
            raise ValueError("selected_files must not exceed candidate_files_seen.")

        if self.blobs_fetched > self.selected_files:
            raise ValueError("blobs_fetched must not exceed selected_files.")

        if self.items_collected + self.skipped_files > self.selected_files:
            raise ValueError(
                "Collected and skipped files must not exceed selected_files."
            )

        if self.items_collected > self.blobs_fetched:
            raise ValueError("items_collected must not exceed blobs_fetched.")

        expected_item_limit = (
            self.candidate_files_seen > self.selected_files
        )
        if self.item_limit_reached != expected_item_limit:
            raise ValueError(
                "item_limit_reached must match candidate and selected counts."
            )

        return self


class GitHubHistoricalAdrCollectionResult(StrictGitHubModel):
    """Repository ADR knowledge collected from one immutable snapshot."""

    repository: GitHubRepositoryRef
    knowledge_repository: KnowledgeRepositoryRef
    requested_ref: str = Field(min_length=1)
    resolved_commit_sha: str = Field(min_length=1)
    snapshot_commit_date: datetime
    tree_sha: str = Field(min_length=1)
    items: list[KnowledgeItem]
    file_evidence: list[GitHubHistoricalAdrFileEvidence]
    skipped_files: list[GitHubHistoricalAdrSkippedFile]
    stats: GitHubHistoricalAdrCollectionStats
    warnings: list[str]

    @field_validator("snapshot_commit_date")
    @classmethod
    def normalize_snapshot_commit_date(cls, value: datetime) -> datetime:
        """Require an aware snapshot timestamp and normalize it to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot_commit_date must be timezone-aware.")

        return value.astimezone(UTC)

    @field_validator("warnings")
    @classmethod
    def validate_warnings(cls, warnings: list[str]) -> list[str]:
        """Require deterministic non-empty unique warning strings."""

        if any(not warning for warning in warnings):
            raise ValueError("ADR collection warnings must not be empty.")

        if len(warnings) != len(set(warnings)):
            raise ValueError("ADR collection warnings must be unique.")

        return warnings

    @model_validator(mode="after")
    def validate_collection(self) -> Self:
        """Ensure ADR items, evidence, skips, and statistics agree."""

        expected_repository = KnowledgeRepositoryRef(
            provider="github",
            namespace=self.repository.owner,
            name=self.repository.name,
        )
        if self.knowledge_repository != expected_repository:
            raise ValueError(
                "knowledge_repository must match the GitHub repository identity."
            )

        if any(item.repository != self.knowledge_repository for item in self.items):
            raise ValueError("Every ADR item must belong to knowledge_repository.")

        if any(item.item_type != KnowledgeItemType.ADR for item in self.items):
            raise ValueError("Historical ADR results may contain only ADR items.")

        if any(
            item.source_kind != KnowledgeSourceKind.REPOSITORY_FILE
            for item in self.items
        ):
            raise ValueError(
                "Historical ADR items must use the repository-file source kind."
            )

        item_keys = [item.key for item in self.items]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Historical ADR item keys must be unique.")

        evidence_keys = [
            evidence.item_key
            for evidence in self.file_evidence
        ]
        if len(evidence_keys) != len(set(evidence_keys)):
            raise ValueError("Historical ADR evidence item keys must be unique.")

        evidence_paths = [
            evidence.path
            for evidence in self.file_evidence
        ]
        if len(evidence_paths) != len(set(evidence_paths)):
            raise ValueError("Historical ADR evidence paths must be unique.")

        items_by_key = {
            item.key: item
            for item in self.items
        }
        for evidence in self.file_evidence:
            item = items_by_key.get(evidence.item_key)
            if item is None:
                raise ValueError(
                    "Every ADR file evidence record must identify an item."
                )

            if item.external_id != evidence.path:
                raise ValueError("ADR item external_id must equal its evidence path.")

            if item.url != evidence.html_url:
                raise ValueError("ADR item URL must equal its evidence HTML URL.")

        if len(self.items) != self.stats.items_collected:
            raise ValueError("ADR item count must match collection statistics.")

        if len(self.file_evidence) != self.stats.items_collected:
            raise ValueError("ADR evidence count must match collection statistics.")

        if len(self.skipped_files) != self.stats.skipped_files:
            raise ValueError("Skipped ADR count must match collection statistics.")

        has_total_limit_skip = any(
            skipped.reason == GitHubHistoricalAdrSkipReason.TOTAL_BYTES_LIMIT
            for skipped in self.skipped_files
        )
        if self.stats.total_bytes_limit_reached != has_total_limit_skip:
            raise ValueError(
                "total_bytes_limit_reached must match skipped-file reasons."
            )

        item_paths = [item.external_id for item in self.items]
        if item_paths != sorted(item_paths):
            raise ValueError("ADR items must use path-ascending order.")

        if evidence_paths != sorted(evidence_paths):
            raise ValueError("ADR evidence must use path-ascending order.")

        skipped_paths = [
            skipped.path
            for skipped in self.skipped_files
        ]
        if skipped_paths != sorted(skipped_paths):
            raise ValueError("Skipped ADR files must use path-ascending order.")

        return self

    @computed_field
    @property
    def total_count(self) -> int:
        """Return the number of collected ADR items."""

        return len(self.items)

    @computed_field
    @property
    def complete(self) -> bool:
        """Return whether ADR discovery and collection were complete."""

        return not (
            self.stats.tree_truncated
            or self.stats.item_limit_reached
            or self.stats.total_bytes_limit_reached
            or self.skipped_files
        )


class GitHubJsonClient(Protocol):
    """REST-client behavior required by ADR collection."""

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


class _GitHubHistoricalAdrApiModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
    )


class _ApiCommitIdentity(_GitHubHistoricalAdrApiModel):
    name: str | None = None
    email: str | None = None
    date: datetime | None = None


class _ApiCommitDetails(_GitHubHistoricalAdrApiModel):
    author: _ApiCommitIdentity | None = None
    committer: _ApiCommitIdentity | None = None


class _ApiCommitResponse(_GitHubHistoricalAdrApiModel):
    sha: str = Field(min_length=1)
    html_url: str = Field(min_length=1)
    commit: _ApiCommitDetails


class _ApiTreeEntry(_GitHubHistoricalAdrApiModel):
    path: str = Field(min_length=1)
    mode: str = Field(min_length=1)
    type: str = Field(min_length=1)
    sha: str = Field(min_length=1)
    size: int | None = Field(default=None, ge=0)


class _ApiTreeResponse(_GitHubHistoricalAdrApiModel):
    sha: str = Field(min_length=1)
    truncated: bool
    tree: list[_ApiTreeEntry]


class _ApiBlobResponse(_GitHubHistoricalAdrApiModel):
    sha: str = Field(min_length=1)
    size: int = Field(ge=0)
    encoding: str = Field(min_length=1)
    content: str


def _build_repository_path(repository: GitHubRepositoryRef) -> str:
    owner = quote(repository.owner, safe="")
    name = quote(repository.name, safe="")
    return f"/repos/{owner}/{name}"


def _snapshot_date(commit: _ApiCommitResponse) -> datetime:
    committer_date = (
        commit.commit.committer.date
        if commit.commit.committer is not None
        else None
    )
    author_date = (
        commit.commit.author.date
        if commit.commit.author is not None
        else None
    )
    snapshot_date = committer_date or author_date
    if snapshot_date is None:
        raise GitHubHistoricalAdrCollectionError(
            "GitHub commit metadata did not include a snapshot date."
        )

    return snapshot_date


def _is_under_directory(path: str, directory: str) -> bool:
    return path.startswith(f"{directory}/")


def _discover_candidates(
    entries: list[_ApiTreeEntry],
    directories: list[str],
) -> list[_ApiTreeEntry]:
    candidates_by_path: dict[str, _ApiTreeEntry] = {}

    for entry in entries:
        if entry.type.casefold() != "blob":
            continue

        if entry.mode not in _REGULAR_FILE_MODES:
            continue

        try:
            normalized_path = _normalize_repository_path(entry.path)
        except ValueError:
            continue

        suffix = PurePosixPath(normalized_path).suffix.casefold()
        if suffix not in GITHUB_ADR_FILE_EXTENSIONS:
            continue

        if not any(
            _is_under_directory(normalized_path, directory)
            for directory in directories
        ):
            continue

        if normalized_path not in candidates_by_path:
            candidates_by_path[normalized_path] = entry.model_copy(
                update={
                    "path": normalized_path,
                }
            )

    return sorted(
        candidates_by_path.values(),
        key=lambda entry: entry.path,
    )


def _decode_blob(
    blob: _ApiBlobResponse,
    entry: _ApiTreeEntry,
) -> bytes:
    if blob.sha != entry.sha:
        raise GitHubHistoricalAdrCollectionError(
            f"GitHub blob SHA did not match the tree entry for {entry.path}."
        )

    if blob.encoding.casefold() != "base64":
        raise GitHubHistoricalAdrCollectionError(
            f"GitHub returned an unsupported blob encoding for {entry.path}."
        )

    compact_content = _ASCII_WHITESPACE_PATTERN.sub("", blob.content)
    try:
        decoded = base64.b64decode(
            compact_content,
            validate=True,
        )
    except (binascii.Error, ValueError) as exc:
        raise GitHubHistoricalAdrCollectionError(
            f"GitHub returned malformed Base64 for {entry.path}."
        ) from exc

    if len(decoded) != blob.size:
        raise GitHubHistoricalAdrCollectionError(
            f"GitHub blob size did not match decoded content for {entry.path}."
        )

    if entry.size is not None and entry.size != blob.size:
        raise GitHubHistoricalAdrCollectionError(
            f"GitHub tree and blob sizes did not match for {entry.path}."
        )

    return decoded


def _extract_markdown_title(text: str, path: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^ {0,3}#[ \t]+(.*)$", line)
        if match is None:
            continue

        title = match.group(1).strip()
        if re.fullmatch(r"#+", title):
            continue

        title = re.sub(r"[ \t]+#+$", "", title).strip()
        if title:
            return title

    filename = PurePosixPath(path).name
    return filename.rsplit(".", maxsplit=1)[0]


def _build_html_url(
    repository: GitHubRepositoryRef,
    commit_sha: str,
    path: str,
) -> str:
    owner = quote(repository.owner, safe="")
    name = quote(repository.name, safe="")
    encoded_sha = quote(commit_sha, safe="")
    encoded_path = "/".join(
        quote(segment, safe="")
        for segment in path.split("/")
    )
    return (
        f"https://github.com/{owner}/{name}"
        f"/blob/{encoded_sha}/{encoded_path}"
    )


def _skipped_file(
    entry: _ApiTreeEntry,
    *,
    reason: GitHubHistoricalAdrSkipReason,
    reported_size_bytes: int | None = None,
) -> GitHubHistoricalAdrSkippedFile:
    return GitHubHistoricalAdrSkippedFile(
        path=entry.path,
        blob_sha=entry.sha,
        reported_size_bytes=(
            entry.size
            if reported_size_bytes is None
            else reported_size_bytes
        ),
        reason=reason,
    )


class GitHubHistoricalAdrCollector:
    """Collect repository ADR knowledge from an immutable Git snapshot."""

    def __init__(self, *, rest_client: GitHubJsonClient) -> None:
        self._rest_client = rest_client

    async def collect(
        self,
        repository: GitHubRepositoryRef,
        *,
        git_ref: str,
        options: GitHubHistoricalAdrCollectionOptions | None = None,
    ) -> GitHubHistoricalAdrCollectionResult:
        """Discover and retrieve bounded ADR content at an exact ref."""

        requested_ref = git_ref.strip()
        if not requested_ref:
            raise ValueError("Git reference must not be empty.")

        selected_options = (
            options
            if options is not None
            else GitHubHistoricalAdrCollectionOptions()
        )
        repository_path = _build_repository_path(repository)
        encoded_ref = quote(requested_ref, safe="")

        commit_response = await self._rest_client.get_json(
            f"{repository_path}/commits/{encoded_ref}",
            response_type=_ApiCommitResponse,
        )
        commit = commit_response.data
        snapshot_commit_date = _snapshot_date(commit)

        encoded_commit_sha = quote(commit.sha, safe="")
        tree_response = await self._rest_client.get_json(
            f"{repository_path}/git/trees/{encoded_commit_sha}",
            params={
                "recursive": "1",
            },
            response_type=_ApiTreeResponse,
        )
        tree = tree_response.data

        if tree.truncated and selected_options.require_complete_tree:
            raise GitHubHistoricalAdrCollectionError(
                "GitHub returned a truncated recursive tree."
            )

        candidates = _discover_candidates(
            tree.tree,
            selected_options.directories,
        )
        selected_candidates = candidates[
            :selected_options.max_files
        ]
        item_limit_reached = (
            len(candidates) > len(selected_candidates)
        )

        items: list[KnowledgeItem] = []
        evidence_items: list[GitHubHistoricalAdrFileEvidence] = []
        skipped_files: list[GitHubHistoricalAdrSkippedFile] = []
        blobs_fetched = 0
        decoded_bytes = 0
        total_bytes_limit_reached = False
        knowledge_repository = knowledge_repository_from_github(repository)

        for index, entry in enumerate(selected_candidates):
            if (
                entry.size is not None
                and entry.size > selected_options.max_file_bytes
            ):
                skipped_files.append(
                    _skipped_file(
                        entry,
                        reason=GitHubHistoricalAdrSkipReason.FILE_TOO_LARGE,
                    )
                )
                continue

            if (
                entry.size is not None
                and decoded_bytes + entry.size
                > selected_options.max_total_bytes
            ):
                skipped_files.extend(
                    _skipped_file(
                        remaining,
                        reason=(
                            GitHubHistoricalAdrSkipReason.TOTAL_BYTES_LIMIT
                        ),
                    )
                    for remaining in selected_candidates[index:]
                )
                total_bytes_limit_reached = True
                break

            encoded_blob_sha = quote(entry.sha, safe="")
            blob_response = await self._rest_client.get_json(
                f"{repository_path}/git/blobs/{encoded_blob_sha}",
                response_type=_ApiBlobResponse,
            )
            blobs_fetched += 1
            blob = blob_response.data
            decoded = _decode_blob(blob, entry)

            if len(decoded) > selected_options.max_file_bytes:
                skipped_files.append(
                    _skipped_file(
                        entry,
                        reason=GitHubHistoricalAdrSkipReason.FILE_TOO_LARGE,
                        reported_size_bytes=blob.size,
                    )
                )
                continue

            if decoded_bytes + len(decoded) > selected_options.max_total_bytes:
                skipped_files.append(
                    _skipped_file(
                        entry,
                        reason=(
                            GitHubHistoricalAdrSkipReason.TOTAL_BYTES_LIMIT
                        ),
                        reported_size_bytes=blob.size,
                    )
                )
                skipped_files.extend(
                    _skipped_file(
                        remaining,
                        reason=(
                            GitHubHistoricalAdrSkipReason.TOTAL_BYTES_LIMIT
                        ),
                    )
                    for remaining in selected_candidates[index + 1:]
                )
                total_bytes_limit_reached = True
                break

            try:
                text = decoded.decode("utf-8")
            except UnicodeDecodeError:
                skipped_files.append(
                    _skipped_file(
                        entry,
                        reason=GitHubHistoricalAdrSkipReason.NON_UTF8_CONTENT,
                        reported_size_bytes=blob.size,
                    )
                )
                continue

            text = text.removeprefix("\ufeff")
            if not text.strip():
                skipped_files.append(
                    _skipped_file(
                        entry,
                        reason=GitHubHistoricalAdrSkipReason.EMPTY_DOCUMENT,
                        reported_size_bytes=blob.size,
                    )
                )
                continue

            html_url = _build_html_url(
                repository,
                commit.sha,
                entry.path,
            )
            item = KnowledgeItem(
                repository=knowledge_repository,
                item_type=KnowledgeItemType.ADR,
                external_id=entry.path,
                source_kind=KnowledgeSourceKind.REPOSITORY_FILE,
                state=KnowledgeItemState.UNKNOWN,
                title=_extract_markdown_title(text, entry.path),
                body=text,
                summary=None,
                url=html_url,
                author=None,
                created_at=snapshot_commit_date,
                updated_at=snapshot_commit_date,
                closed_at=None,
                labels=[],
                affected_paths=[],
                components=[],
                decision_significance=DecisionSignificance.NONE,
            )
            items.append(item)
            evidence_items.append(
                GitHubHistoricalAdrFileEvidence(
                    item_key=item.key,
                    path=entry.path,
                    blob_sha=entry.sha,
                    size_bytes=len(decoded),
                    html_url=html_url,
                )
            )
            decoded_bytes += len(decoded)

        warnings: list[str] = []
        if tree.truncated:
            warnings.append(_TREE_TRUNCATION_WARNING)
        if items:
            warnings.append(_TIMESTAMP_WARNING)

        return GitHubHistoricalAdrCollectionResult(
            repository=repository,
            knowledge_repository=knowledge_repository,
            requested_ref=requested_ref,
            resolved_commit_sha=commit.sha,
            snapshot_commit_date=snapshot_commit_date,
            tree_sha=tree.sha,
            items=items,
            file_evidence=evidence_items,
            skipped_files=skipped_files,
            stats=GitHubHistoricalAdrCollectionStats(
                tree_entries_seen=len(tree.tree),
                candidate_files_seen=len(candidates),
                selected_files=len(selected_candidates),
                blobs_fetched=blobs_fetched,
                items_collected=len(items),
                skipped_files=len(skipped_files),
                decoded_bytes=decoded_bytes,
                tree_truncated=tree.truncated,
                item_limit_reached=item_limit_reached,
                total_bytes_limit_reached=total_bytes_limit_reached,
            ),
            warnings=warnings,
        )
