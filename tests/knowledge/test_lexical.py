"""Tests for provider-independent lexical knowledge preparation."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.knowledge import (
    MAX_KNOWLEDGE_EXACT_PHRASES,
    MAX_KNOWLEDGE_IDENTIFIERS,
    MAX_KNOWLEDGE_QUERY_COMPONENTS,
    MAX_KNOWLEDGE_QUERY_LABELS,
    MAX_KNOWLEDGE_QUERY_PATHS,
    MAX_KNOWLEDGE_QUERY_REFERENCES,
    MAX_KNOWLEDGE_QUERY_TEXT_LENGTH,
    MIN_KNOWLEDGE_LEXICAL_TERM_LENGTH,
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalCorpus,
    KnowledgeLexicalDocument,
    KnowledgeLexicalField,
    KnowledgeLexicalQuery,
    KnowledgeLexicalReference,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
    build_knowledge_lexical_corpus,
    build_knowledge_lexical_document,
)

CREATED_AT = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
UPDATED_AT = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)
NON_UTC_CREATED_AT = datetime(
    2026,
    5,
    1,
    15,
    0,
    tzinfo=timezone(timedelta(hours=6)),
)
NON_UTC_UPDATED_AT = datetime(
    2026,
    5,
    3,
    18,
    0,
    tzinfo=timezone(timedelta(hours=6)),
)
REPOSITORY = KnowledgeRepositoryRef(
    provider="github",
    namespace="acme",
    name="framework",
)
OTHER_REPOSITORY = KnowledgeRepositoryRef(
    provider="gitlab",
    namespace="other",
    name="project",
)


def make_item(
    *,
    item_type: KnowledgeItemType = KnowledgeItemType.ISSUE,
    external_id: str = "42",
    repository: KnowledgeRepositoryRef = REPOSITORY,
    title: str = "ParserRegistry Guide",
    body: str | None = "Use parser_registry and shared.",
    summary: str | None = "ParserRegistry shared.",
    labels: list[str] | None = None,
    components: list[str] | None = None,
    affected_paths: list[str] | None = None,
    created_at: datetime = CREATED_AT,
    updated_at: datetime = UPDATED_AT,
) -> KnowledgeItem:
    """Create a complete provider-independent knowledge item."""

    return KnowledgeItem(
        repository=repository,
        item_type=item_type,
        external_id=external_id,
        source_kind=KnowledgeSourceKind.GITHUB,
        state=KnowledgeItemState.CLOSED,
        title=title,
        body=body,
        summary=summary,
        url=f"https://example.test/items/{external_id}",
        created_at=created_at,
        updated_at=updated_at,
        labels=labels or ["Feature", "ＦＵＬＬ"],
        components=components or ["Core"],
        affected_paths=affected_paths or [
            "src/opensteward/parser.py",
            "docs/parser.md",
        ],
        decision_significance=DecisionSignificance.HIGH,
    )


def query(**updates: Any) -> KnowledgeLexicalQuery:
    """Create a lexical query with caller-supplied search evidence."""

    return KnowledgeLexicalQuery(
        repository=REPOSITORY,
        **updates,
    )


def document_payload(
    document: KnowledgeLexicalDocument,
) -> dict[str, object]:
    """Return only declared document fields for validation mutations."""

    return {
        field_name: getattr(document, field_name)
        for field_name in KnowledgeLexicalDocument.model_fields
    }


@pytest.mark.parametrize(
    "signal",
    [
        {"text": "parser"},
        {"exact_phrases": ["parser registry"]},
        {"identifiers": ["ParserRegistry"]},
        {"labels": ["bug"]},
        {"components": ["parser"]},
        {"affected_paths": ["src/parser.py"]},
        {"references": [{"external_id": "42"}]},
    ],
)
def test_each_query_search_signal_is_valid(signal: dict[str, object]) -> None:
    result = query(**signal)

    assert result.repository == REPOSITORY


def test_repository_and_filter_only_queries_are_rejected() -> None:
    with pytest.raises(ValidationError, match="search signal"):
        KnowledgeLexicalQuery(repository=REPOSITORY)

    with pytest.raises(ValidationError, match="search signal"):
        KnowledgeLexicalQuery(
            repository=REPOSITORY,
            item_types=[KnowledgeItemType.ISSUE],
            states=[KnowledgeItemState.CLOSED],
        )


def test_query_rejects_extra_fields_and_overlong_text() -> None:
    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeLexicalQuery.model_validate(
            {
                "repository": REPOSITORY,
                "text": "parser",
                "score": 1,
            }
        )

    with pytest.raises(ValidationError):
        query(text="x" * (MAX_KNOWLEDGE_QUERY_TEXT_LENGTH + 1))


def test_query_text_normalization_preserves_stored_spelling() -> None:
    result = query(text=" \tＦｏｏ  STRASSE\nParserRegistry ")

    assert result.text == "Ｆｏｏ  STRASSE\nParserRegistry"
    assert result.normalized_text == "foo strasse parserregistry"


def test_query_rejects_empty_supplied_text() -> None:
    with pytest.raises(ValidationError, match="must not be empty"):
        query(text=" \n\t ")


def test_exact_phrase_normalization_and_duplicate_detection() -> None:
    result = query(
        exact_phrases=[
            "  Parser\t Registry  ",
            "First Spelling",
        ]
    )

    assert result.exact_phrases == [
        "Parser Registry",
        "First Spelling",
    ]

    with pytest.raises(ValidationError, match="unique"):
        query(
            exact_phrases=[
                "Parser  Registry",
                "ｐａｒｓｅｒ registry",
            ]
        )


def test_explicit_identifier_normalization_and_duplicate_detection() -> None:
    result = query(identifiers=[" Parser.Registry ", "MAX_RETRIES"])

    assert result.identifiers == ["Parser.Registry", "MAX_RETRIES"]

    with pytest.raises(ValidationError, match="unique"):
        query(identifiers=["MAX_RETRIES", "ｍａｘ_ｒｅｔｒｉｅｓ"])


def test_labels_and_components_reject_case_insensitive_duplicates() -> None:
    with pytest.raises(ValidationError, match="unique"):
        query(labels=["Bug", "bug"])

    with pytest.raises(ValidationError, match="unique"):
        query(components=["Parser", "PARSER"])


def test_query_paths_normalize_and_reject_unsafe_or_duplicate_values() -> None:
    result = query(
        affected_paths=[
            " .\\src\\parser.py ",
            "././docs/parser.md",
        ]
    )

    assert result.affected_paths == [
        "src/parser.py",
        "docs/parser.md",
    ]

    for unsafe_path in (
        "/src/parser.py",
        "C:\\src\\parser.py",
        "../src/parser.py",
        "src/../parser.py",
        "src/./parser.py",
        "src//parser.py",
    ):
        with pytest.raises(ValidationError):
            query(affected_paths=[unsafe_path])

    with pytest.raises(ValidationError, match="unique after normalization"):
        query(affected_paths=["src/parser.py", ".\\src\\parser.py"])


def test_query_rejects_duplicate_references_and_filters() -> None:
    with pytest.raises(ValidationError, match="unique keys"):
        query(
            references=[
                KnowledgeLexicalReference(external_id="42"),
                KnowledgeLexicalReference(external_id="42"),
            ]
        )

    with pytest.raises(ValidationError, match="filters must be unique"):
        query(
            text="parser",
            item_types=[
                KnowledgeItemType.ISSUE,
                KnowledgeItemType.ISSUE,
            ],
        )

    with pytest.raises(ValidationError, match="filters must be unique"):
        query(
            text="parser",
            states=[
                KnowledgeItemState.CLOSED,
                KnowledgeItemState.CLOSED,
            ],
        )


@pytest.mark.parametrize(
    ("field_name", "maximum", "values"),
    [
        (
            "exact_phrases",
            MAX_KNOWLEDGE_EXACT_PHRASES,
            lambda count: [f"phrase {index}" for index in range(count)],
        ),
        (
            "identifiers",
            MAX_KNOWLEDGE_IDENTIFIERS,
            lambda count: [f"identifier_{index}" for index in range(count)],
        ),
        (
            "labels",
            MAX_KNOWLEDGE_QUERY_LABELS,
            lambda count: [f"label-{index}" for index in range(count)],
        ),
        (
            "components",
            MAX_KNOWLEDGE_QUERY_COMPONENTS,
            lambda count: [f"component-{index}" for index in range(count)],
        ),
        (
            "affected_paths",
            MAX_KNOWLEDGE_QUERY_PATHS,
            lambda count: [f"src/file-{index}.py" for index in range(count)],
        ),
        (
            "references",
            MAX_KNOWLEDGE_QUERY_REFERENCES,
            lambda count: [
                KnowledgeLexicalReference(external_id=str(index + 1))
                for index in range(count)
            ],
        ),
    ],
)
def test_query_list_safety_limits_are_enforced(
    field_name: str,
    maximum: int,
    values: Any,
) -> None:
    with pytest.raises(ValidationError):
        query(**{field_name: values(maximum + 1)})


def test_reference_model_builds_typed_and_untyped_keys() -> None:
    untyped = KnowledgeLexicalReference(external_id="42")
    typed = KnowledgeLexicalReference(
        item_type=KnowledgeItemType.ISSUE,
        external_id="42",
    )

    assert untyped.key == "*:42"
    assert typed.key == "issue:42"

    with pytest.raises(ValidationError):
        KnowledgeLexicalReference(external_id=" ")


def test_term_extraction_is_unicode_aware_and_deterministic() -> None:
    result = query(
        text=(
            "Élan parser_registry parser-registry !!! a 421 UTF8 "
            "the running runs ÉLAN"
        )
    )

    assert result.text_terms == [
        "élan",
        "parser_registry",
        "parser-registry",
        "421",
        "utf8",
        "the",
        "running",
        "runs",
    ]
    assert MIN_KNOWLEDGE_LEXICAL_TERM_LENGTH == 2


def test_identifier_extraction_covers_supported_shapes_and_order() -> None:
    result = query(
        text=(
            "(ParserRegistry), parser_registry parser.parse. "
            "[std::vector] HTTPServer éÉlan MAX_RETRIES ordinary 421 #421 "
            "error.code max_retries"
        )
    )

    assert result.text_identifiers == [
        "ParserRegistry",
        "parser_registry",
        "parser.parse",
        "std::vector",
        "HTTPServer",
        "éÉlan",
        "MAX_RETRIES",
        "error.code",
    ]


@pytest.mark.parametrize(
    ("text", "expected_type"),
    [
        ("#123", None),
        ("issue #123", KnowledgeItemType.ISSUE),
        ("issue 123", KnowledgeItemType.ISSUE),
        ("pr #123", KnowledgeItemType.PULL_REQUEST),
        ("pr 123", KnowledgeItemType.PULL_REQUEST),
        ("pull request #123", KnowledgeItemType.PULL_REQUEST),
        ("pull request 123", KnowledgeItemType.PULL_REQUEST),
        ("PULL REQUEST #123", KnowledgeItemType.PULL_REQUEST),
    ],
)
def test_reference_extraction_supports_documented_forms(
    text: str,
    expected_type: KnowledgeItemType | None,
) -> None:
    references = query(text=text).text_references

    assert references == [
        KnowledgeLexicalReference(
            item_type=expected_type,
            external_id="123",
        )
    ]


def test_reference_extraction_deduplicates_and_suppresses_typed_spans() -> None:
    result = query(
        text="issue #12, issue 12, then #12 and PR #7 and pr 7"
    )

    assert [reference.key for reference in result.text_references] == [
        "issue:12",
        "*:12",
        "pull_request:7",
    ]


def test_reference_extraction_preserves_encounter_order() -> None:
    result = query(text="#9 then issue #12, PR 7, and #12")

    assert [reference.key for reference in result.text_references] == [
        "*:9",
        "issue:12",
        "pull_request:7",
        "*:12",
    ]


def test_reference_extraction_rejects_zero_cross_repo_and_urls() -> None:
    result = query(
        text=(
            "#0 issue 0 pr #0 owner/repo#12 "
            "https://github.com/acme/framework/issues/13 "
            "https://example.test/#14"
        )
    )

    assert result.text_references == []


def test_complete_document_builder_preserves_and_normalizes_source_data() -> None:
    item = make_item(
        created_at=NON_UTC_CREATED_AT,
        updated_at=NON_UTC_UPDATED_AT,
    )
    before = item.model_dump(mode="json")

    document = build_knowledge_lexical_document(item)

    assert document.reference == item.to_reference()
    assert document.state == KnowledgeItemState.CLOSED
    assert document.decision_significance == DecisionSignificance.HIGH
    assert document.created_at == CREATED_AT
    assert document.updated_at == UPDATED_AT
    assert document.normalized_title == "parserregistry guide"
    assert document.normalized_body == "use parser_registry and shared."
    assert document.normalized_summary == "parserregistry shared."
    assert document.title_terms == ["parserregistry", "guide"]
    assert document.body_terms == ["use", "parser_registry", "and", "shared"]
    assert document.summary_terms == ["parserregistry", "shared"]
    assert document.title_identifiers == ["ParserRegistry"]
    assert document.body_identifiers == ["parser_registry"]
    assert document.summary_identifiers == ["ParserRegistry"]
    assert document.labels == ["feature", "full"]
    assert document.components == ["core"]
    assert document.affected_paths == [
        "src/opensteward/parser.py",
        "docs/parser.md",
    ]
    assert item.model_dump(mode="json") == before
    assert "body" not in KnowledgeLexicalDocument.model_fields


def test_document_computed_signal_order_and_deduplication() -> None:
    document = build_knowledge_lexical_document(make_item())

    assert document.item_key == document.reference.key
    assert document.all_terms == [
        "parserregistry",
        "guide",
        "shared",
        "use",
        "parser_registry",
        "and",
    ]
    assert document.all_identifiers == [
        "ParserRegistry",
        "parser_registry",
    ]


def test_document_rejects_naive_or_reversed_timestamps() -> None:
    document = build_knowledge_lexical_document(make_item())
    payload = document_payload(document)
    payload["created_at"] = datetime(2026, 5, 1, 9, 0)

    with pytest.raises(ValidationError, match="timezone-aware"):
        KnowledgeLexicalDocument.model_validate(payload)

    payload = document_payload(document)
    payload["created_at"] = UPDATED_AT
    payload["updated_at"] = CREATED_AT
    with pytest.raises(ValidationError, match="updated_at"):
        KnowledgeLexicalDocument.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("normalized_title", "Not Canonical"),
        ("normalized_body", ""),
        ("normalized_summary", " Mixed CASE "),
    ],
)
def test_document_rejects_noncanonical_normalized_text(
    field_name: str,
    invalid_value: str,
) -> None:
    document = build_knowledge_lexical_document(make_item())
    payload = document_payload(document)
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        KnowledgeLexicalDocument.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("title_terms", ["a"]),
        ("body_terms", ["parser", "parser"]),
        ("summary_terms", ["Parser"]),
        ("title_terms", ["parser!"]),
    ],
)
def test_document_rejects_invalid_or_duplicate_terms(
    field_name: str,
    invalid_value: list[str],
) -> None:
    document = build_knowledge_lexical_document(make_item())
    payload = document_payload(document)
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        KnowledgeLexicalDocument.model_validate(payload)


def test_document_rejects_duplicate_identifiers() -> None:
    document = build_knowledge_lexical_document(make_item())
    payload = document_payload(document)
    payload["title_identifiers"] = ["MAX_RETRIES", "ｍａｘ_ｒｅｔｒｉｅｓ"]

    with pytest.raises(ValidationError, match="unique"):
        KnowledgeLexicalDocument.model_validate(payload)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("labels", ["Feature"]),
        ("labels", ["feature", "feature"]),
        ("components", ["Ｃｏｒｅ"]),
    ],
)
def test_document_rejects_noncanonical_or_duplicate_structured_values(
    field_name: str,
    invalid_value: list[str],
) -> None:
    document = build_knowledge_lexical_document(make_item())
    payload = document_payload(document)
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        KnowledgeLexicalDocument.model_validate(payload)


@pytest.mark.parametrize(
    "invalid_paths",
    [
        ["../src/parser.py"],
        ["src\\parser.py"],
        ["src/parser.py", "src/parser.py"],
    ],
)
def test_document_rejects_invalid_or_duplicate_affected_paths(
    invalid_paths: list[str],
) -> None:
    document = build_knowledge_lexical_document(make_item())
    payload = document_payload(document)
    payload["affected_paths"] = invalid_paths

    with pytest.raises(ValidationError):
        KnowledgeLexicalDocument.model_validate(payload)


def test_document_serializes_nested_enums_times_and_computed_fields() -> None:
    document = build_knowledge_lexical_document(make_item())

    payload = document.model_dump(mode="json")

    assert payload["reference"]["item_type"] == "issue"
    assert payload["state"] == "closed"
    assert payload["decision_significance"] == "high"
    assert payload["created_at"] == "2026-05-01T09:00:00Z"
    assert payload["updated_at"] == "2026-05-03T12:00:00Z"
    assert payload["item_key"] == document.reference.key
    assert payload["all_terms"] == document.all_terms
    assert payload["all_identifiers"] == document.all_identifiers


def test_lexical_field_enum_has_exact_values() -> None:
    assert [field.value for field in KnowledgeLexicalField] == [
        "title",
        "body",
        "summary",
        "external_id",
        "label",
        "component",
        "affected_path",
    ]


def test_corpus_builder_sorts_mixed_types_and_preserves_inputs() -> None:
    item_types = list(KnowledgeItemType)
    items = [
        make_item(
            item_type=item_type,
            external_id=f"id-{index}",
            labels=["Label"],
            components=["Component"],
        )
        for index, item_type in enumerate(reversed(item_types), start=1)
    ]
    original_order = list(items)
    original_payloads = [item.model_dump(mode="json") for item in items]

    corpus = build_knowledge_lexical_corpus(REPOSITORY, items)

    assert [document.item_key for document in corpus.documents] == sorted(
        item.key for item in items
    )
    assert items == original_order
    assert [item.model_dump(mode="json") for item in items] == original_payloads
    assert corpus.total_count == len(item_types)
    assert corpus.issue_count == 1
    assert corpus.pull_request_count == 1
    assert corpus.discussion_count == 1
    assert corpus.adr_count == 1
    assert corpus.maintainer_decision_count == 1
    assert corpus.release_note_count == 1
    assert corpus.documentation_count == 1


def test_corpus_builder_accepts_empty_and_rejects_invalid_sources() -> None:
    corpus = build_knowledge_lexical_corpus(REPOSITORY, [])

    assert corpus.documents == []
    assert corpus.total_count == 0

    foreign = make_item(repository=OTHER_REPOSITORY)
    with pytest.raises(ValueError, match="requested repository"):
        build_knowledge_lexical_corpus(REPOSITORY, [foreign])

    duplicate = make_item()
    with pytest.raises(ValueError, match="keys must be unique"):
        build_knowledge_lexical_corpus(REPOSITORY, [duplicate, duplicate])


def test_corpus_model_rejects_wrong_order_duplicate_and_repository() -> None:
    first = build_knowledge_lexical_document(
        make_item(item_type=KnowledgeItemType.ISSUE, external_id="1")
    )
    second = build_knowledge_lexical_document(
        make_item(item_type=KnowledgeItemType.PULL_REQUEST, external_id="2")
    )
    ordered = sorted([first, second], key=lambda document: document.item_key)

    with pytest.raises(ValidationError, match="ascending order"):
        KnowledgeLexicalCorpus(
            repository=REPOSITORY,
            documents=list(reversed(ordered)),
        )

    with pytest.raises(ValidationError, match="keys must be unique"):
        KnowledgeLexicalCorpus(
            repository=REPOSITORY,
            documents=[first, first],
        )

    foreign = build_knowledge_lexical_document(
        make_item(repository=OTHER_REPOSITORY)
    )
    with pytest.raises(ValidationError, match="corpus repository"):
        KnowledgeLexicalCorpus(
            repository=REPOSITORY,
            documents=[foreign],
        )


def test_corpus_json_includes_document_keys_and_computed_counts() -> None:
    corpus = build_knowledge_lexical_corpus(
        REPOSITORY,
        [make_item()],
    )

    payload = corpus.model_dump(mode="json")

    assert payload["documents"][0]["item_key"] == corpus.documents[0].item_key
    assert payload["total_count"] == 1
    assert payload["issue_count"] == 1
    assert payload["pull_request_count"] == 0
    assert payload["discussion_count"] == 0
    assert payload["adr_count"] == 0
    assert payload["maintainer_decision_count"] == 0
    assert payload["release_note_count"] == 0
    assert payload["documentation_count"] == 0
