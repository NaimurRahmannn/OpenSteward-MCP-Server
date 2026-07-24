"""Provider-independent lexical query and knowledge-document preparation."""

import re
import unicodedata
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Self

from pydantic import (
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from opensteward.knowledge.models import (
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemReference,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeRepositoryRef,
    StrictKnowledgeModel,
)

MIN_KNOWLEDGE_LEXICAL_TERM_LENGTH = 2
MAX_KNOWLEDGE_QUERY_TEXT_LENGTH = 20_000
MAX_KNOWLEDGE_EXACT_PHRASES = 50
MAX_KNOWLEDGE_IDENTIFIERS = 100
MAX_KNOWLEDGE_QUERY_LABELS = 100
MAX_KNOWLEDGE_QUERY_COMPONENTS = 100
MAX_KNOWLEDGE_QUERY_PATHS = 200
MAX_KNOWLEDGE_QUERY_REFERENCES = 100

_TERM_PATTERN = re.compile(r"[^\W_]+(?:[_-][^\W_]+)*")
_IDENTIFIER_EDGE_PUNCTUATION = ",.:;()[]{}\"'"
_MULTIPLE_UPPERCASE_PATTERN = re.compile(r"[A-Z].*[A-Z]")
_DOTTED_IDENTIFIER_PATTERN = re.compile(r"\S\.\S")
_ISSUE_REFERENCE_PATTERN = re.compile(
    r"\bissue\s+#?([0-9]+)\b(?!\.[0-9])",
    re.IGNORECASE,
)
_PULL_REQUEST_REFERENCE_PATTERN = re.compile(
    r"\b(?:pr|pull\s+request)\s+#?([0-9]+)\b(?!\.[0-9])",
    re.IGNORECASE,
)
_GENERIC_REFERENCE_PATTERN = re.compile(
    r"(?<![\w/])#([0-9]+)\b(?!\.[0-9])"
)
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)


class KnowledgeLexicalField(StrEnum):
    """Knowledge fields that can provide later lexical matches."""

    TITLE = "title"
    BODY = "body"
    SUMMARY = "summary"
    EXTERNAL_ID = "external_id"
    LABEL = "label"
    COMPONENT = "component"
    AFFECTED_PATH = "affected_path"


class KnowledgeLexicalReference(StrictKnowledgeModel):
    """An optional-type reference used by a lexical query."""

    item_type: KnowledgeItemType | None = None
    external_id: str = Field(min_length=1)

    @computed_field
    @property
    def key(self) -> str:
        """Return the type-qualified lexical reference key."""

        item_type = (
            self.item_type.value
            if self.item_type is not None
            else "*"
        )
        return f"{item_type}:{self.external_id}"


def _normalize_nfkc(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _canonical_text(value: str) -> str:
    return _collapse_whitespace(_normalize_nfkc(value).casefold())


def _comparison_key(value: str) -> str:
    return _normalize_nfkc(value).casefold()


def _stable_unique(
    values: list[str],
    *,
    comparison_key: Callable[[str], str] | None = None,
) -> list[str]:
    key_function = comparison_key or (lambda value: value)
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = key_function(value)
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def _normalize_repository_path(path: str) -> str:
    normalized = path.replace("\\", "/").strip()

    while normalized.startswith("./"):
        normalized = normalized[2:]

    if not normalized:
        raise ValueError("Repository paths must not be empty.")

    if normalized.startswith("/") or re.match(r"^[A-Za-z]:/", normalized):
        raise ValueError("Repository paths must be repository-relative.")

    parts = normalized.split("/")
    if any(part == "" for part in parts):
        raise ValueError("Repository paths must not contain empty segments.")

    if any(part in {".", ".."} for part in parts):
        raise ValueError("Repository paths must not contain '.' or '..' segments.")

    return normalized


def _validate_unique_strings(
    values: list[str],
    *,
    field_name: str,
    comparison_key: Callable[[str], str],
) -> list[str]:
    if any(not value for value in values):
        raise ValueError(f"{field_name} must contain non-empty strings.")

    keys = [comparison_key(value) for value in values]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{field_name} must contain unique values.")

    return values


def _extract_terms(value: str | None) -> list[str]:
    if value is None:
        return []

    terms = [
        match.group(0)
        for match in _TERM_PATTERN.finditer(value)
        if len(match.group(0)) >= MIN_KNOWLEDGE_LEXICAL_TERM_LENGTH
    ]
    return _stable_unique(terms)


def _is_identifier_candidate(value: str) -> bool:
    has_case_transition = any(
        previous.islower() and current.isupper()
        for previous, current in zip(value, value[1:], strict=False)
    )
    return (
        "_" in value
        or "::" in value
        or _DOTTED_IDENTIFIER_PATTERN.search(value) is not None
        or has_case_transition
        or _MULTIPLE_UPPERCASE_PATTERN.search(value) is not None
    )


def _extract_identifiers(value: str | None) -> list[str]:
    if value is None:
        return []

    candidates: list[str] = []
    for token in value.split():
        candidate = token.strip(_IDENTIFIER_EDGE_PUNCTUATION)
        if (
            len(candidate) < MIN_KNOWLEDGE_LEXICAL_TERM_LENGTH
            or not _is_identifier_candidate(candidate)
        ):
            continue
        candidates.append(candidate)

    return _stable_unique(
        candidates,
        comparison_key=_comparison_key,
    )


def _spans_overlap(
    first: tuple[int, int],
    second: tuple[int, int],
) -> bool:
    return first[0] < second[1] and second[0] < first[1]


def _extract_references(value: str | None) -> list[KnowledgeLexicalReference]:
    if value is None:
        return []

    url_spans = [
        match.span()
        for match in _URL_PATTERN.finditer(value)
    ]
    extracted: list[
        tuple[int, int, KnowledgeLexicalReference]
    ] = []
    typed_spans: list[tuple[int, int]] = []

    typed_patterns = (
        (_ISSUE_REFERENCE_PATTERN, KnowledgeItemType.ISSUE),
        (_PULL_REQUEST_REFERENCE_PATTERN, KnowledgeItemType.PULL_REQUEST),
    )
    for pattern, item_type in typed_patterns:
        for match in pattern.finditer(value):
            if any(_spans_overlap(match.span(), span) for span in url_spans):
                continue

            external_id = match.group(1)
            if int(external_id) == 0:
                continue

            typed_spans.append(match.span())
            extracted.append(
                (
                    match.start(),
                    0,
                    KnowledgeLexicalReference(
                        item_type=item_type,
                        external_id=external_id,
                    ),
                )
            )

    for match in _GENERIC_REFERENCE_PATTERN.finditer(value):
        if any(_spans_overlap(match.span(), span) for span in url_spans):
            continue
        if any(_spans_overlap(match.span(), span) for span in typed_spans):
            continue

        external_id = match.group(1)
        if int(external_id) == 0:
            continue

        extracted.append(
            (
                match.start(),
                1,
                KnowledgeLexicalReference(external_id=external_id),
            )
        )

    extracted.sort(key=lambda entry: (entry[0], entry[1]))
    references: list[KnowledgeLexicalReference] = []
    seen: set[str] = set()
    for _, _, reference in extracted:
        if reference.key in seen:
            continue
        seen.add(reference.key)
        references.append(reference)
    return references


class KnowledgeLexicalQuery(StrictKnowledgeModel):
    """Validated lexical evidence and optional item filters."""

    repository: KnowledgeRepositoryRef
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

    @field_validator("text")
    @classmethod
    def reject_empty_text(cls, text: str | None) -> str | None:
        """Reject a supplied text signal without content."""

        if text == "":
            raise ValueError("Query text must not be empty when supplied.")
        return text

    @field_validator("exact_phrases")
    @classmethod
    def normalize_exact_phrases(cls, phrases: list[str]) -> list[str]:
        """Collapse phrase whitespace and reject normalized duplicates."""

        normalized = [_collapse_whitespace(phrase) for phrase in phrases]
        return _validate_unique_strings(
            normalized,
            field_name="Exact phrases",
            comparison_key=_comparison_key,
        )

    @field_validator("identifiers")
    @classmethod
    def validate_identifiers(cls, identifiers: list[str]) -> list[str]:
        """Reject empty or normalized duplicate explicit identifiers."""

        return _validate_unique_strings(
            identifiers,
            field_name="Identifiers",
            comparison_key=_comparison_key,
        )

    @field_validator("labels", "components")
    @classmethod
    def validate_casefolded_unique_strings(
        cls,
        values: list[str],
    ) -> list[str]:
        """Reject empty or case-insensitive duplicate structured values."""

        return _validate_unique_strings(
            values,
            field_name="Structured query values",
            comparison_key=str.casefold,
        )

    @field_validator("affected_paths")
    @classmethod
    def normalize_affected_paths(cls, paths: list[str]) -> list[str]:
        """Normalize and deduplicate repository-relative query paths."""

        normalized = [_normalize_repository_path(path) for path in paths]
        if len(normalized) != len(set(normalized)):
            raise ValueError(
                "Affected paths must be unique after normalization."
            )
        return normalized

    @field_validator("references")
    @classmethod
    def validate_references(
        cls,
        references: list[KnowledgeLexicalReference],
    ) -> list[KnowledgeLexicalReference]:
        """Reject duplicate explicit reference keys."""

        keys = [reference.key for reference in references]
        if len(keys) != len(set(keys)):
            raise ValueError("Lexical references must have unique keys.")
        return references

    @field_validator("item_types", "states")
    @classmethod
    def validate_unique_filters(
        cls,
        values: list[KnowledgeItemType] | list[KnowledgeItemState],
    ) -> list[KnowledgeItemType] | list[KnowledgeItemState]:
        """Reject duplicate query filter values."""

        if len(values) != len(set(values)):
            raise ValueError("Lexical query filters must be unique.")
        return values

    @model_validator(mode="after")
    def require_search_signal(self) -> Self:
        """Require evidence beyond repository identity and filters."""

        if not (
            self.text
            or self.exact_phrases
            or self.identifiers
            or self.labels
            or self.components
            or self.affected_paths
            or self.references
        ):
            raise ValueError("A lexical query requires at least one search signal.")
        return self

    @computed_field
    @property
    def normalized_text(self) -> str | None:
        """Return canonical query text for lexical comparison."""

        if self.text is None:
            return None
        return _canonical_text(self.text)

    @computed_field
    @property
    def text_terms(self) -> list[str]:
        """Return deterministic general terms extracted from query text."""

        return _extract_terms(self.normalized_text)

    @computed_field
    @property
    def text_identifiers(self) -> list[str]:
        """Return deterministic identifiers extracted from original text."""

        return _extract_identifiers(self.text)

    @computed_field
    @property
    def text_references(self) -> list[KnowledgeLexicalReference]:
        """Return deterministic explicit item references from query text."""

        return _extract_references(self.text)


class KnowledgeLexicalDocument(StrictKnowledgeModel):
    """Deterministic lexical representation of one knowledge item."""

    reference: KnowledgeItemReference
    state: KnowledgeItemState
    decision_significance: DecisionSignificance
    created_at: datetime
    updated_at: datetime
    normalized_title: str = Field(min_length=1)
    normalized_body: str | None = None
    normalized_summary: str | None = None
    title_terms: list[str]
    body_terms: list[str]
    summary_terms: list[str]
    title_identifiers: list[str]
    body_identifiers: list[str]
    summary_identifiers: list[str]
    labels: list[str]
    components: list[str]
    affected_paths: list[str]

    @field_validator("created_at", "updated_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime) -> datetime:
        """Require aware document timestamps and normalize them to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Lexical document timestamps must be timezone-aware.")
        return value.astimezone(UTC)

    @field_validator(
        "normalized_title",
        "normalized_body",
        "normalized_summary",
    )
    @classmethod
    def validate_normalized_text(
        cls,
        value: str | None,
    ) -> str | None:
        """Require supplied normalized fields to be non-empty and canonical."""

        if value == "":
            raise ValueError("Normalized lexical text must not be empty.")
        if value is not None and value != _canonical_text(value):
            raise ValueError("Normalized lexical text must be canonical.")
        return value

    @field_validator("title_terms", "body_terms", "summary_terms")
    @classmethod
    def validate_terms(cls, terms: list[str]) -> list[str]:
        """Require unique canonical lexical terms."""

        if len(terms) != len(set(terms)):
            raise ValueError("Lexical terms must be unique.")

        for term in terms:
            if (
                len(term) < MIN_KNOWLEDGE_LEXICAL_TERM_LENGTH
                or term != _canonical_text(term)
                or _extract_terms(term) != [term]
            ):
                raise ValueError("Lexical terms must be non-empty canonical terms.")
        return terms

    @field_validator(
        "title_identifiers",
        "body_identifiers",
        "summary_identifiers",
    )
    @classmethod
    def validate_identifiers(cls, identifiers: list[str]) -> list[str]:
        """Require non-empty normalized-unique identifiers."""

        return _validate_unique_strings(
            identifiers,
            field_name="Document identifiers",
            comparison_key=_comparison_key,
        )

    @field_validator("labels", "components")
    @classmethod
    def validate_canonical_structured_values(
        cls,
        values: list[str],
    ) -> list[str]:
        """Require unique NFKC-casefold labels and components."""

        if any(not value or value != _comparison_key(value) for value in values):
            raise ValueError(
                "Document labels and components must be canonical."
            )
        if len(values) != len(set(values)):
            raise ValueError(
                "Document labels and components must be unique."
            )
        return values

    @field_validator("affected_paths")
    @classmethod
    def validate_affected_paths(cls, paths: list[str]) -> list[str]:
        """Require unique normalized repository-relative paths."""

        normalized = [_normalize_repository_path(path) for path in paths]
        if normalized != paths:
            raise ValueError("Document affected paths must already be normalized.")
        if len(paths) != len(set(paths)):
            raise ValueError("Document affected paths must be unique.")
        return paths

    @model_validator(mode="after")
    def validate_timestamp_order(self) -> Self:
        """Ensure the lexical document timestamp order is valid."""

        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at.")
        return self

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the source knowledge-item key."""

        return self.reference.key

    @computed_field
    @property
    def all_terms(self) -> list[str]:
        """Return unique terms in title, summary, then body order."""

        return _stable_unique([
            *self.title_terms,
            *self.summary_terms,
            *self.body_terms,
        ])

    @computed_field
    @property
    def all_identifiers(self) -> list[str]:
        """Return normalized-unique identifiers in field-priority order."""

        return _stable_unique(
            [
                *self.title_identifiers,
                *self.summary_identifiers,
                *self.body_identifiers,
            ],
            comparison_key=_comparison_key,
        )


def build_knowledge_lexical_document(
    item: KnowledgeItem,
) -> KnowledgeLexicalDocument:
    """Build one deterministic lexical document without mutating its item."""

    normalized_title = _canonical_text(item.title)
    normalized_body = (
        _canonical_text(item.body)
        if item.body is not None
        else None
    )
    normalized_summary = (
        _canonical_text(item.summary)
        if item.summary is not None
        else None
    )
    labels = _stable_unique(
        [_comparison_key(label) for label in item.labels]
    )
    components = _stable_unique(
        [_comparison_key(component) for component in item.components]
    )
    return KnowledgeLexicalDocument(
        reference=item.to_reference(),
        state=item.state,
        decision_significance=item.decision_significance,
        created_at=item.created_at,
        updated_at=item.updated_at,
        normalized_title=normalized_title,
        normalized_body=normalized_body,
        normalized_summary=normalized_summary,
        title_terms=_extract_terms(normalized_title),
        body_terms=_extract_terms(normalized_body),
        summary_terms=_extract_terms(normalized_summary),
        title_identifiers=_extract_identifiers(item.title),
        body_identifiers=_extract_identifiers(item.body),
        summary_identifiers=_extract_identifiers(item.summary),
        labels=labels,
        components=components,
        affected_paths=list(item.affected_paths),
    )


class KnowledgeLexicalCorpus(StrictKnowledgeModel):
    """Validated item-key-ordered lexical document corpus."""

    repository: KnowledgeRepositoryRef
    documents: list[KnowledgeLexicalDocument]

    @model_validator(mode="after")
    def validate_documents(self) -> Self:
        """Require repository consistency, unique keys, and stable order."""

        if any(
            document.reference.repository != self.repository
            for document in self.documents
        ):
            raise ValueError(
                "Every lexical document must belong to the corpus repository."
            )

        item_keys = [document.item_key for document in self.documents]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Lexical corpus document keys must be unique.")
        if item_keys != sorted(item_keys):
            raise ValueError(
                "Lexical corpus documents must use item-key ascending order."
            )
        return self

    @computed_field
    @property
    def total_count(self) -> int:
        """Return the total number of lexical documents."""

        return len(self.documents)

    def _count_type(self, item_type: KnowledgeItemType) -> int:
        return sum(
            document.reference.item_type == item_type
            for document in self.documents
        )

    @computed_field
    @property
    def issue_count(self) -> int:
        """Return the number of issue documents."""

        return self._count_type(KnowledgeItemType.ISSUE)

    @computed_field
    @property
    def pull_request_count(self) -> int:
        """Return the number of pull-request documents."""

        return self._count_type(KnowledgeItemType.PULL_REQUEST)

    @computed_field
    @property
    def discussion_count(self) -> int:
        """Return the number of discussion documents."""

        return self._count_type(KnowledgeItemType.DISCUSSION)

    @computed_field
    @property
    def adr_count(self) -> int:
        """Return the number of ADR documents."""

        return self._count_type(KnowledgeItemType.ADR)

    @computed_field
    @property
    def maintainer_decision_count(self) -> int:
        """Return the number of maintainer-decision documents."""

        return self._count_type(KnowledgeItemType.MAINTAINER_DECISION)

    @computed_field
    @property
    def release_note_count(self) -> int:
        """Return the number of release-note documents."""

        return self._count_type(KnowledgeItemType.RELEASE_NOTE)

    @computed_field
    @property
    def documentation_count(self) -> int:
        """Return the number of documentation documents."""

        return self._count_type(KnowledgeItemType.DOCUMENTATION)


def build_knowledge_lexical_corpus(
    repository: KnowledgeRepositoryRef,
    items: list[KnowledgeItem],
) -> KnowledgeLexicalCorpus:
    """Build a provider-independent item-key-ordered lexical corpus."""

    if any(item.repository != repository for item in items):
        raise ValueError("Every corpus item must belong to the requested repository.")

    item_keys = [item.key for item in items]
    if len(item_keys) != len(set(item_keys)):
        raise ValueError("Corpus source item keys must be unique.")

    documents = [
        build_knowledge_lexical_document(item)
        for item in items
    ]
    documents.sort(key=lambda document: document.item_key)
    return KnowledgeLexicalCorpus(
        repository=repository,
        documents=documents,
    )
