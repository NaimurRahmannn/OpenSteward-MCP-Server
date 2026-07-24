"""Tests for explainable deterministic lexical related-work search."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.knowledge import (
    KNOWLEDGE_BODY_EXACT_PHRASE_POINTS,
    KNOWLEDGE_BODY_IDENTIFIER_POINTS,
    KNOWLEDGE_BODY_TERM_POINTS,
    KNOWLEDGE_COMPONENT_POINTS,
    KNOWLEDGE_EXACT_PATH_POINTS,
    KNOWLEDGE_LABEL_POINTS,
    KNOWLEDGE_SHARED_DIRECTORY_POINTS,
    KNOWLEDGE_SUMMARY_EXACT_PHRASE_POINTS,
    KNOWLEDGE_SUMMARY_IDENTIFIER_POINTS,
    KNOWLEDGE_SUMMARY_TERM_POINTS,
    KNOWLEDGE_TITLE_EXACT_PHRASE_POINTS,
    KNOWLEDGE_TITLE_IDENTIFIER_POINTS,
    KNOWLEDGE_TITLE_TERM_POINTS,
    KNOWLEDGE_TYPED_REFERENCE_POINTS,
    KNOWLEDGE_UNTYPED_REFERENCE_POINTS,
    MAX_KNOWLEDGE_LEXICAL_EVIDENCE_PER_MATCH,
    MAX_KNOWLEDGE_LEXICAL_RESULTS,
    MAX_KNOWLEDGE_LEXICAL_SCORE,
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalCorpus,
    KnowledgeLexicalDocument,
    KnowledgeLexicalField,
    KnowledgeLexicalMatch,
    KnowledgeLexicalMatchEvidence,
    KnowledgeLexicalMatchKind,
    KnowledgeLexicalQuery,
    KnowledgeLexicalReference,
    KnowledgeLexicalSearchOptions,
    KnowledgeLexicalSearchResult,
    KnowledgeRepositoryRef,
    KnowledgeSourceKind,
    build_knowledge_lexical_corpus,
    build_knowledge_lexical_document,
    search_knowledge_lexical_corpus,
)

CREATED_AT = datetime(2026, 5, 1, 9, 0, tzinfo=UTC)
UPDATED_AT = datetime(2026, 5, 3, 12, 0, tzinfo=UTC)
LATER_AT = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
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
    external_id: str = "42",
    repository: KnowledgeRepositoryRef = REPOSITORY,
    item_type: KnowledgeItemType = KnowledgeItemType.ISSUE,
    state: KnowledgeItemState = KnowledgeItemState.CLOSED,
    title: str = "ParserRegistry guide",
    summary: str | None = "ParserRegistry summary",
    body: str | None = "Use parser_registry in the parser body",
    labels: list[str] | None = None,
    components: list[str] | None = None,
    affected_paths: list[str] | None = None,
    updated_at: datetime = UPDATED_AT,
    significance: DecisionSignificance = DecisionSignificance.HIGH,
) -> KnowledgeItem:
    """Build one complete provider-independent knowledge item."""

    return KnowledgeItem(
        repository=repository,
        item_type=item_type,
        external_id=external_id,
        source_kind=KnowledgeSourceKind.MANUAL,
        state=state,
        title=title,
        summary=summary,
        body=body,
        created_at=CREATED_AT,
        updated_at=updated_at,
        labels=["Feature"] if labels is None else labels,
        components=["Parser"] if components is None else components,
        affected_paths=(
            ["src/parser/registry.py"]
            if affected_paths is None
            else affected_paths
        ),
        decision_significance=significance,
    )


def make_query(**updates: Any) -> KnowledgeLexicalQuery:
    """Build a query for the default repository."""

    return KnowledgeLexicalQuery(repository=REPOSITORY, **updates)


def make_corpus(*items: KnowledgeItem) -> KnowledgeLexicalCorpus:
    """Build a corpus containing supplied items or one default item."""

    return build_knowledge_lexical_corpus(
        REPOSITORY,
        list(items) if items else [make_item()],
    )


def make_evidence(
    *,
    kind: KnowledgeLexicalMatchKind = KnowledgeLexicalMatchKind.TERM,
    field: KnowledgeLexicalField = KnowledgeLexicalField.TITLE,
    query_value: str = "parser",
    document_value: str = "parser",
    points: int = 5,
    explanation: str = "Term matched the historical item title.",
) -> KnowledgeLexicalMatchEvidence:
    """Build one valid evidence record."""

    return KnowledgeLexicalMatchEvidence(
        kind=kind,
        field=field,
        query_value=query_value,
        document_value=document_value,
        points=points,
        explanation=explanation,
    )


def make_match(
    *,
    item: KnowledgeItem | None = None,
    evidence: list[KnowledgeLexicalMatchEvidence] | None = None,
) -> KnowledgeLexicalMatch:
    """Build one valid lexical match."""

    source = item or make_item()
    return KnowledgeLexicalMatch(
        reference=source.to_reference(),
        state=source.state,
        decision_significance=source.decision_significance,
        updated_at=source.updated_at,
        evidence=evidence or [make_evidence()],
    )


def search(
    query: KnowledgeLexicalQuery,
    *items: KnowledgeItem,
    options: KnowledgeLexicalSearchOptions | None = None,
) -> KnowledgeLexicalSearchResult:
    """Search a corpus assembled from supplied test items."""

    return search_knowledge_lexical_corpus(
        query,
        make_corpus(*items),
        options=options,
    )


def result_payload(
    result: KnowledgeLexicalSearchResult,
) -> dict[str, object]:
    """Return declared result fields for validation mutations."""

    return {
        field_name: getattr(result, field_name)
        for field_name in KnowledgeLexicalSearchResult.model_fields
    }


def make_evidence_limit_case(
    *,
    label_count: int,
    include_phrase: bool = False,
) -> tuple[KnowledgeLexicalQuery, KnowledgeItem]:
    """Build one generated query/document evidence-limit case."""

    identifiers = [
        f"identifier_{index:03d}"
        for index in range(100)
    ]
    affected_paths = [
        f"src/generated/directory-{index:03d}/file.py"
        for index in range(200)
    ]
    components = [
        f"Component-{index:03d}"
        for index in range(100)
    ]
    labels = [
        f"Label-{index:03d}"
        for index in range(label_count)
    ]
    phrase = "Safety marker"
    title_parts = [
        *([phrase] if include_phrase else []),
        *identifiers,
    ]
    query = make_query(
        exact_phrases=[phrase] if include_phrase else [],
        identifiers=identifiers,
        affected_paths=affected_paths,
        components=components,
        labels=labels,
    )
    item = make_item(
        external_id="2",
        title=" ".join(title_parts),
        summary=None,
        body=None,
        affected_paths=affected_paths,
        components=components,
        labels=labels,
    )
    return query, item


def test_default_options_and_bounds() -> None:
    options = KnowledgeLexicalSearchOptions()

    assert options.max_results == 20
    assert options.minimum_score == 1

    for max_results in (0, MAX_KNOWLEDGE_LEXICAL_RESULTS + 1):
        with pytest.raises(ValidationError):
            KnowledgeLexicalSearchOptions(max_results=max_results)

    for minimum_score in (0, MAX_KNOWLEDGE_LEXICAL_SCORE + 1):
        with pytest.raises(ValidationError):
            KnowledgeLexicalSearchOptions(minimum_score=minimum_score)


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("query_value", " "),
        ("document_value", " "),
        ("explanation", " "),
    ],
)
def test_evidence_rejects_empty_values(field_name: str, value: str) -> None:
    payload = make_evidence().model_dump()
    payload[field_name] = value

    with pytest.raises(ValidationError):
        KnowledgeLexicalMatchEvidence.model_validate(payload)


@pytest.mark.parametrize("points", [0, MAX_KNOWLEDGE_LEXICAL_SCORE + 1])
def test_evidence_rejects_invalid_points(points: int) -> None:
    with pytest.raises(ValidationError):
        make_evidence(points=points)


def test_match_validation_and_computed_values() -> None:
    with pytest.raises(ValidationError):
        KnowledgeLexicalMatch(
            reference=make_item().to_reference(),
            state=KnowledgeItemState.CLOSED,
            decision_significance=DecisionSignificance.HIGH,
            updated_at=UPDATED_AT,
            evidence=[],
        )

    with pytest.raises(ValidationError, match="timezone-aware"):
        KnowledgeLexicalMatch(
            reference=make_item().to_reference(),
            state=KnowledgeItemState.CLOSED,
            decision_significance=DecisionSignificance.HIGH,
            updated_at=datetime(2026, 5, 3, 12, 0),
            evidence=[make_evidence()],
        )

    evidence = [
        make_evidence(points=60),
        make_evidence(
            kind=KnowledgeLexicalMatchKind.REFERENCE,
            field=KnowledgeLexicalField.EXTERNAL_ID,
            query_value="issue:42",
            document_value="issue:42",
            points=60,
            explanation="Explicit typed reference matched this historical item.",
        ),
        make_evidence(
            kind=KnowledgeLexicalMatchKind.TERM,
            field=KnowledgeLexicalField.BODY,
            query_value="body",
            document_value="body",
            points=1,
            explanation="Term matched the historical item body.",
        ),
    ]
    match = make_match(
        item=make_item(updated_at=NON_UTC_UPDATED_AT),
        evidence=evidence,
    )

    assert match.updated_at == UPDATED_AT
    assert match.raw_score == 121
    assert match.score == 100
    assert match.matched_fields == [
        KnowledgeLexicalField.TITLE,
        KnowledgeLexicalField.EXTERNAL_ID,
        KnowledgeLexicalField.BODY,
    ]
    assert match.matched_kinds == [
        KnowledgeLexicalMatchKind.TERM,
        KnowledgeLexicalMatchKind.REFERENCE,
    ]
    assert match.item_key == match.reference.key


def test_match_rejects_duplicate_evidence_identity_after_nfkc_casefold() -> None:
    duplicate = make_evidence(
        query_value="ＰＡＲＳＥＲ",
        document_value="ＰＡＲＳＥＲ",
        points=3,
    )

    with pytest.raises(ValidationError, match="duplicates"):
        make_match(evidence=[make_evidence(), duplicate])


def test_search_result_strictness_and_computed_counts() -> None:
    options = KnowledgeLexicalSearchOptions(max_results=1)
    result = search(
        make_query(text="parser"),
        make_item(external_id="1"),
        make_item(external_id="2"),
        options=options,
    )

    assert result.options == options
    assert result.returned_count == 1
    assert result.matched_document_count == 2
    assert result.truncated is True

    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeLexicalSearchResult.model_validate(
            {
                **result_payload(result),
                "complete": True,
            }
        )


def test_search_results_record_default_and_custom_options() -> None:
    default_result = search(make_query(text="parser"))
    custom_options = KnowledgeLexicalSearchOptions(
        max_results=7,
        minimum_score=3,
    )
    custom_result = search(
        make_query(text="parser"),
        options=custom_options,
    )

    assert default_result.options == KnowledgeLexicalSearchOptions(
        max_results=20,
        minimum_score=1,
    )
    assert custom_result.options == custom_options


def test_repository_mismatch_raises_before_scoring() -> None:
    query = KnowledgeLexicalQuery(repository=OTHER_REPOSITORY, text="parser")

    with pytest.raises(ValueError, match="repositories must match"):
        search_knowledge_lexical_corpus(query, make_corpus())


def test_item_type_and_state_filters_use_and_semantics() -> None:
    items = [
        make_item(
            external_id="1",
            item_type=KnowledgeItemType.ISSUE,
            state=KnowledgeItemState.OPEN,
        ),
        make_item(
            external_id="2",
            item_type=KnowledgeItemType.ISSUE,
            state=KnowledgeItemState.CLOSED,
        ),
        make_item(
            external_id="3",
            item_type=KnowledgeItemType.PULL_REQUEST,
            state=KnowledgeItemState.OPEN,
        ),
    ]

    type_result = search(
        make_query(text="parser", item_types=[KnowledgeItemType.ISSUE]),
        *items,
    )
    state_result = search(
        make_query(text="parser", states=[KnowledgeItemState.OPEN]),
        *items,
    )
    combined = search(
        make_query(
            text="parser",
            item_types=[KnowledgeItemType.ISSUE],
            states=[KnowledgeItemState.OPEN],
        ),
        *items,
    )
    unfiltered = search(make_query(text="parser"), *items)

    assert type_result.eligible_document_count == 2
    assert state_result.eligible_document_count == 2
    assert combined.eligible_document_count == 1
    assert [match.reference.external_id for match in combined.matches] == ["1"]
    assert unfiltered.eligible_document_count == 3
    assert unfiltered.corpus_total_count == 3


def test_typed_reference_matching_and_evidence() -> None:
    items = [
        make_item(external_id="42", item_type=KnowledgeItemType.ISSUE),
        make_item(
            external_id="42",
            item_type=KnowledgeItemType.PULL_REQUEST,
        ),
    ]
    result = search(
        make_query(
            references=[
                KnowledgeLexicalReference(
                    item_type=KnowledgeItemType.ISSUE,
                    external_id="42",
                )
            ]
        ),
        *items,
    )

    assert len(result.matches) == 1
    assert result.matches[0].reference.item_type == KnowledgeItemType.ISSUE
    evidence = result.matches[0].evidence[0]
    assert evidence.kind == KnowledgeLexicalMatchKind.REFERENCE
    assert evidence.field == KnowledgeLexicalField.EXTERNAL_ID
    assert evidence.points == KNOWLEDGE_TYPED_REFERENCE_POINTS == 60
    assert evidence.query_value == "issue:42"
    assert evidence.document_value == "issue:42"
    assert evidence.explanation == (
        "Explicit typed reference matched this historical item."
    )


def test_untyped_reference_matches_any_item_type() -> None:
    result = search(
        make_query(references=[KnowledgeLexicalReference(external_id="42")]),
        make_item(
            external_id="42",
            item_type=KnowledgeItemType.PULL_REQUEST,
        ),
    )
    evidence = result.matches[0].evidence[0]

    assert evidence.points == KNOWLEDGE_UNTYPED_REFERENCE_POINTS == 50
    assert evidence.document_value == "pull_request:42"
    assert evidence.explanation == (
        "Explicit item reference matched this historical item."
    )


def test_explicit_and_text_references_combine_and_deduplicate() -> None:
    duplicate_result = search(
        make_query(
            text="issue #42",
            references=[
                KnowledgeLexicalReference(
                    item_type=KnowledgeItemType.ISSUE,
                    external_id="42",
                )
            ],
        )
    )
    combined_result = search(
        make_query(
            text="#42",
            references=[
                KnowledgeLexicalReference(
                    item_type=KnowledgeItemType.ISSUE,
                    external_id="42",
                )
            ],
        )
    )
    duplicate_references = [
        item
        for item in duplicate_result.matches[0].evidence
        if item.kind == KnowledgeLexicalMatchKind.REFERENCE
    ]
    combined_references = [
        item
        for item in combined_result.matches[0].evidence
        if item.kind == KnowledgeLexicalMatchKind.REFERENCE
    ]

    assert [item.query_value for item in duplicate_references] == ["issue:42"]
    assert [item.query_value for item in combined_references] == [
        "issue:42",
        "*:42",
    ]
    assert sum(item.points for item in combined_references) == 110


@pytest.mark.parametrize(
    (
        "title",
        "summary",
        "body",
        "expected_field",
        "expected_points",
        "expected_explanation",
    ),
    [
        (
            "Adopt parser registry",
            "parser registry",
            "parser registry",
            KnowledgeLexicalField.TITLE,
            KNOWLEDGE_TITLE_EXACT_PHRASE_POINTS,
            "Exact phrase matched the historical item title.",
        ),
        (
            "Unrelated title",
            "Adopt parser registry",
            "parser registry",
            KnowledgeLexicalField.SUMMARY,
            KNOWLEDGE_SUMMARY_EXACT_PHRASE_POINTS,
            "Exact phrase matched the historical item summary.",
        ),
        (
            "Unrelated title",
            "Unrelated summary",
            "Adopt parser registry",
            KnowledgeLexicalField.BODY,
            KNOWLEDGE_BODY_EXACT_PHRASE_POINTS,
            "Exact phrase matched the historical item body.",
        ),
    ],
)
def test_exact_phrase_field_priority_and_points(
    title: str,
    summary: str,
    body: str,
    expected_field: KnowledgeLexicalField,
    expected_points: int,
    expected_explanation: str,
) -> None:
    result = search(
        make_query(exact_phrases=["parser registry"]),
        make_item(title=title, summary=summary, body=body),
    )
    evidence = result.matches[0].evidence

    assert len(evidence) == 1
    assert evidence[0].field == expected_field
    assert evidence[0].points == expected_points
    assert evidence[0].explanation == expected_explanation


def test_exact_phrase_is_nfkc_casefold_whitespace_collapsed_substring() -> None:
    result = search(
        make_query(
            exact_phrases=[
                "ＰＡＲＳＥＲ \t ＲＥＧＩＳＴＲＹ",
                "registry handles",
            ]
        ),
        make_item(
            title="The Parser   Registry handles requests",
            summary=None,
            body=None,
        ),
    )

    assert [item.document_value for item in result.matches[0].evidence] == [
        "parser registry",
        "registry handles",
    ]
    assert [item.query_value for item in result.matches[0].evidence] == [
        "ＰＡＲＳＥＲ ＲＥＧＩＳＴＲＹ",
        "registry handles",
    ]
    assert result.matches[0].raw_score == 60


def test_explicit_and_text_identifiers_combine_and_deduplicate() -> None:
    result = search(
        make_query(
            text="ＰａｒｓｅｒＲｅｇｉｓｔｒｙ OtherIdentifier",
            identifiers=["ParserRegistry"],
        ),
        make_item(
            title="ParserRegistry OtherIdentifier",
            summary=None,
            body=None,
        ),
    )
    identifier_evidence = [
        item
        for item in result.matches[0].evidence
        if item.kind == KnowledgeLexicalMatchKind.IDENTIFIER
    ]

    assert [item.query_value for item in identifier_evidence] == [
        "ParserRegistry",
        "OtherIdentifier",
    ]
    assert all(
        item.points == KNOWLEDGE_TITLE_IDENTIFIER_POINTS
        for item in identifier_evidence
    )


@pytest.mark.parametrize(
    (
        "title",
        "summary",
        "body",
        "expected_field",
        "expected_points",
        "expected_explanation",
    ),
    [
        (
            "ParserRegistry",
            "ParserRegistry",
            "ParserRegistry",
            KnowledgeLexicalField.TITLE,
            KNOWLEDGE_TITLE_IDENTIFIER_POINTS,
            "Identifier matched the historical item title.",
        ),
        (
            "Plain title",
            "ParserRegistry",
            "ParserRegistry",
            KnowledgeLexicalField.SUMMARY,
            KNOWLEDGE_SUMMARY_IDENTIFIER_POINTS,
            "Identifier matched the historical item summary.",
        ),
        (
            "Plain title",
            "Plain summary",
            "ParserRegistry",
            KnowledgeLexicalField.BODY,
            KNOWLEDGE_BODY_IDENTIFIER_POINTS,
            "Identifier matched the historical item body.",
        ),
    ],
)
def test_identifier_field_priority_points_and_spelling(
    title: str,
    summary: str,
    body: str,
    expected_field: KnowledgeLexicalField,
    expected_points: int,
    expected_explanation: str,
) -> None:
    result = search(
        make_query(identifiers=["parserregistry"]),
        make_item(title=title, summary=summary, body=body),
    )
    evidence = result.matches[0].evidence[0]

    assert evidence.field == expected_field
    assert evidence.points == expected_points
    assert evidence.query_value == "parserregistry"
    assert evidence.document_value == "ParserRegistry"
    assert evidence.explanation == expected_explanation


def test_identifier_matching_is_exact_not_substring() -> None:
    result = search(
        make_query(identifiers=["Parser"]),
        make_item(
            title="ParserRegistry",
            summary=None,
            body=None,
        ),
    )

    assert result.matches == []


def test_exact_path_precedes_shared_directory() -> None:
    result = search(
        make_query(affected_paths=["src/parser/registry.py"]),
        make_item(
            affected_paths=[
                "src/parser/cache.py",
                "src/parser/registry.py",
            ]
        ),
    )
    evidence = result.matches[0].evidence[0]

    assert evidence.kind == KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT
    assert evidence.field == KnowledgeLexicalField.AFFECTED_PATH
    assert evidence.points == KNOWLEDGE_EXACT_PATH_POINTS == 25
    assert evidence.document_value == "src/parser/registry.py"
    assert evidence.explanation == (
        "Repository path exactly matched historical changed-path evidence."
    )


def test_shared_directory_requires_two_directory_segments() -> None:
    matching = search(
        make_query(affected_paths=["src/parser/registry.py"]),
        make_item(affected_paths=["src/parser/cache.py"]),
    )
    nonmatching = search(
        make_query(affected_paths=["src/parser/registry.py"]),
        make_item(affected_paths=["src/runtime/cache.py"]),
    )
    filename_only = search(
        make_query(affected_paths=["src/parser/registry.py"]),
        make_item(affected_paths=["src/parser/registry.py/cache.py"]),
    )
    evidence = matching.matches[0].evidence[0]

    assert evidence.kind == (
        KnowledgeLexicalMatchKind.AFFECTED_PATH_SHARED_DIRECTORY
    )
    assert evidence.points == KNOWLEDGE_SHARED_DIRECTORY_POINTS == 8
    assert evidence.explanation == (
        "Repository paths share a specific directory hierarchy."
    )
    assert nonmatching.matches == []
    assert filename_only.matches[0].evidence[0].kind == (
        KnowledgeLexicalMatchKind.AFFECTED_PATH_SHARED_DIRECTORY
    )


def test_shared_directory_chooses_deepest_then_path_ascending() -> None:
    deepest = search(
        make_query(affected_paths=["tests/parser/unit/test_registry.py"]),
        make_item(
            affected_paths=[
                "tests/parser/cache.py",
                "tests/parser/unit/test_z.py",
                "tests/parser/unit/test_a.py",
            ]
        ),
    )

    assert deepest.matches[0].evidence[0].document_value == (
        "tests/parser/unit/test_a.py"
    )


def test_shared_directory_comparison_excludes_the_query_filename() -> None:
    result = search(
        make_query(affected_paths=["tests/parser/unit.py"]),
        make_item(
            affected_paths=[
                "tests/parser/unit.py/cache.py",
                "tests/parser/a.py",
            ]
        ),
    )

    assert result.matches[0].evidence[0].document_value == "tests/parser/a.py"


def test_each_query_path_contributes_at_most_once_and_paths_preserve_spelling() -> None:
    result = search(
        make_query(
            affected_paths=[
                "src/parser/registry.py",
                "tests/parser/test_registry.py",
            ]
        ),
        make_item(
            affected_paths=[
                "src/parser/registry.py",
                "src/parser/cache.py",
                "tests/parser/test_cache.py",
            ]
        ),
    )
    path_evidence = result.matches[0].evidence

    assert len(path_evidence) == 2
    assert [item.query_value for item in path_evidence] == [
        "src/parser/registry.py",
        "tests/parser/test_registry.py",
    ]
    assert [item.document_value for item in path_evidence] == [
        "src/parser/registry.py",
        "tests/parser/test_cache.py",
    ]


@pytest.mark.parametrize(
    (
        "query_field",
        "query_value",
        "item_field",
        "kind",
        "field",
        "points",
        "explanation",
    ),
    [
        (
            "components",
            "ＰＡＲＳＥＲ",
            "components",
            KnowledgeLexicalMatchKind.COMPONENT,
            KnowledgeLexicalField.COMPONENT,
            KNOWLEDGE_COMPONENT_POINTS,
            "Component matched historical component evidence.",
        ),
        (
            "labels",
            "ＦＥＡＴＵＲＥ",
            "labels",
            KnowledgeLexicalMatchKind.LABEL,
            KnowledgeLexicalField.LABEL,
            KNOWLEDGE_LABEL_POINTS,
            "Label matched historical label evidence.",
        ),
    ],
)
def test_component_and_label_matching(
    query_field: str,
    query_value: str,
    item_field: str,
    kind: KnowledgeLexicalMatchKind,
    field: KnowledgeLexicalField,
    points: int,
    explanation: str,
) -> None:
    item_updates = {item_field: ["Parser" if item_field == "components" else "Feature"]}
    result = search(
        make_query(**{query_field: [query_value]}),
        make_item(**item_updates),
    )
    evidence = result.matches[0].evidence[0]

    assert evidence.kind == kind
    assert evidence.field == field
    assert evidence.query_value == query_value
    assert evidence.document_value == (
        "parser" if item_field == "components" else "feature"
    )
    assert evidence.points == points
    assert evidence.explanation == explanation


def test_component_and_label_query_values_contribute_once() -> None:
    result = search(
        make_query(
            components=["Parser"],
            labels=["Feature"],
        ),
        make_item(
            components=["Parser"],
            labels=["Feature"],
        ),
    )

    assert [item.kind for item in result.matches[0].evidence] == [
        KnowledgeLexicalMatchKind.COMPONENT,
        KnowledgeLexicalMatchKind.LABEL,
    ]


@pytest.mark.parametrize(
    (
        "title",
        "summary",
        "body",
        "expected_field",
        "expected_points",
        "expected_explanation",
    ),
    [
        (
            "parser parser parser",
            "parser",
            "parser",
            KnowledgeLexicalField.TITLE,
            KNOWLEDGE_TITLE_TERM_POINTS,
            "Term matched the historical item title.",
        ),
        (
            "Unrelated",
            "parser parser",
            "parser",
            KnowledgeLexicalField.SUMMARY,
            KNOWLEDGE_SUMMARY_TERM_POINTS,
            "Term matched the historical item summary.",
        ),
        (
            "Unrelated",
            "Unrelated",
            "parser parser",
            KnowledgeLexicalField.BODY,
            KNOWLEDGE_BODY_TERM_POINTS,
            "Term matched the historical item body.",
        ),
    ],
)
def test_term_field_priority_points_and_no_frequency_scoring(
    title: str,
    summary: str,
    body: str,
    expected_field: KnowledgeLexicalField,
    expected_points: int,
    expected_explanation: str,
) -> None:
    result = search(
        make_query(text="parser"),
        make_item(title=title, summary=summary, body=body),
    )
    term_evidence = [
        item
        for item in result.matches[0].evidence
        if item.kind == KnowledgeLexicalMatchKind.TERM
    ]

    assert len(term_evidence) == 1
    assert term_evidence[0].field == expected_field
    assert term_evidence[0].points == expected_points
    assert term_evidence[0].explanation == expected_explanation


def test_multiple_query_terms_preserve_order() -> None:
    result = search(
        make_query(text="beta alpha"),
        make_item(title="alpha beta", summary=None, body=None),
    )

    assert [item.query_value for item in result.matches[0].evidence] == [
        "beta",
        "alpha",
    ]


def test_nontext_signals_do_not_create_term_evidence() -> None:
    phrase_result = search(
        make_query(exact_phrases=["parser registry"]),
        make_item(title="parser registry", summary=None, body=None),
    )
    identifier_result = search(
        make_query(identifiers=["ParserRegistry"]),
        make_item(title="ParserRegistry", summary=None, body=None),
    )

    assert [item.kind for item in phrase_result.matches[0].evidence] == [
        KnowledgeLexicalMatchKind.EXACT_PHRASE
    ]
    assert [item.kind for item in identifier_result.matches[0].evidence] == [
        KnowledgeLexicalMatchKind.IDENTIFIER
    ]


def test_evidence_uses_mandated_category_and_query_order() -> None:
    query = make_query(
        text="issue #42 TextIdentifier beta alpha",
        exact_phrases=["first phrase", "second phrase"],
        identifiers=["ExplicitIdentifier"],
        affected_paths=[
            "src/parser/registry.py",
            "tests/parser/test_registry.py",
        ],
        components=["Parser", "Core"],
        labels=["Feature", "Bug"],
    )
    item = make_item(
        title=(
            "first phrase second phrase ExplicitIdentifier TextIdentifier "
            "beta alpha"
        ),
        summary=None,
        body=None,
        components=["Parser", "Core"],
        labels=["Feature", "Bug"],
        affected_paths=[
            "src/parser/registry.py",
            "tests/parser/test_cache.py",
        ],
    )
    evidence = search(query, item).matches[0].evidence

    assert [item.kind for item in evidence] == [
        KnowledgeLexicalMatchKind.REFERENCE,
        KnowledgeLexicalMatchKind.EXACT_PHRASE,
        KnowledgeLexicalMatchKind.EXACT_PHRASE,
        KnowledgeLexicalMatchKind.IDENTIFIER,
        KnowledgeLexicalMatchKind.IDENTIFIER,
        KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT,
        KnowledgeLexicalMatchKind.AFFECTED_PATH_SHARED_DIRECTORY,
        KnowledgeLexicalMatchKind.COMPONENT,
        KnowledgeLexicalMatchKind.COMPONENT,
        KnowledgeLexicalMatchKind.LABEL,
        KnowledgeLexicalMatchKind.LABEL,
        KnowledgeLexicalMatchKind.TERM,
        KnowledgeLexicalMatchKind.TERM,
        KnowledgeLexicalMatchKind.TERM,
    ]
    assert [item.query_value for item in evidence[1:3]] == [
        "first phrase",
        "second phrase",
    ]
    assert [item.query_value for item in evidence[3:5]] == [
        "ExplicitIdentifier",
        "TextIdentifier",
    ]
    assert [item.query_value for item in evidence[5:7]] == [
        "src/parser/registry.py",
        "tests/parser/test_registry.py",
    ]
    assert [item.query_value for item in evidence[7:9]] == ["Parser", "Core"]
    assert [item.query_value for item in evidence[9:11]] == ["Feature", "Bug"]
    assert [item.query_value for item in evidence[11:]] == [
        "textidentifier",
        "beta",
        "alpha",
    ]
    assert evidence[0].points > evidence[1].points
    assert evidence[5].points > evidence[4].points


@pytest.mark.parametrize(
    ("label_count", "expected_count", "expected_raw_score"),
    [
        (99, 499, 9_290),
        (100, MAX_KNOWLEDGE_LEXICAL_EVIDENCE_PER_MATCH, 9_300),
    ],
)
def test_search_accepts_evidence_at_and_below_the_safety_limit(
    label_count: int,
    expected_count: int,
    expected_raw_score: int,
) -> None:
    query, item = make_evidence_limit_case(label_count=label_count)

    match = search(query, item).matches[0]

    assert len(match.evidence) == expected_count
    assert match.raw_score == expected_raw_score
    assert match.score == MAX_KNOWLEDGE_LEXICAL_SCORE
    assert [evidence.kind for evidence in match.evidence[:100]] == [
        KnowledgeLexicalMatchKind.IDENTIFIER
    ] * 100
    assert [evidence.kind for evidence in match.evidence[100:300]] == [
        KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT
    ] * 200
    assert [evidence.kind for evidence in match.evidence[300:400]] == [
        KnowledgeLexicalMatchKind.COMPONENT
    ] * 100
    assert [evidence.kind for evidence in match.evidence[400:]] == [
        KnowledgeLexicalMatchKind.LABEL
    ] * label_count


def test_search_raises_instead_of_returning_a_partial_over_limit_match() -> None:
    query, overflowing_item = make_evidence_limit_case(
        label_count=100,
        include_phrase=True,
    )
    earlier_matching_item = make_item(
        external_id="1",
        title="Unrelated",
        summary=None,
        body=None,
        labels=["Label-000"],
        components=[],
        affected_paths=[],
    )
    result: KnowledgeLexicalSearchResult | None = None

    with pytest.raises(
        ValueError,
        match=(
            "^Lexical match evidence exceeds the configured safety limit\\.$"
        ),
    ):
        result = search(query, earlier_matching_item, overflowing_item)

    assert result is None


def test_no_evidence_and_below_minimum_score_documents_are_excluded() -> None:
    items = [
        make_item(external_id="1", title="parser", summary=None, body=None),
        make_item(external_id="2", title="unrelated", summary=None, body=None),
    ]
    no_threshold = search(make_query(text="parser"), *items)
    threshold = search(
        make_query(text="parser"),
        *items,
        options=KnowledgeLexicalSearchOptions(minimum_score=6),
    )

    assert [match.reference.external_id for match in no_threshold.matches] == ["1"]
    assert threshold.matches == []
    assert threshold.matched_document_count == 0


def test_minimum_score_uses_capped_score_and_raw_score_can_exceed_100() -> None:
    result = search(
        make_query(
            references=[
                KnowledgeLexicalReference(
                    item_type=KnowledgeItemType.ISSUE,
                    external_id="42",
                ),
                KnowledgeLexicalReference(external_id="42"),
            ]
        ),
        options=KnowledgeLexicalSearchOptions(minimum_score=100),
    )
    match = result.matches[0]

    assert match.raw_score == 110
    assert match.score == 100


def test_ranking_uses_raw_score_then_item_key() -> None:
    items = [
        make_item(
            external_id="2",
            title="parser registry",
            summary=None,
            body=None,
        ),
        make_item(
            external_id="1",
            title="parser registry",
            summary=None,
            body=None,
        ),
        make_item(
            external_id="3",
            title="parser",
            summary=None,
            body=None,
        ),
    ]
    result = search(
        make_query(text="parser", exact_phrases=["parser registry"]),
        *items,
    )

    assert [match.reference.external_id for match in result.matches] == [
        "1",
        "2",
        "3",
    ]
    assert [match.raw_score for match in result.matches] == [35, 35, 5]


def test_ranking_ignores_time_significance_state_and_item_type() -> None:
    items = [
        make_item(
            external_id="2",
            item_type=KnowledgeItemType.PULL_REQUEST,
            state=KnowledgeItemState.OPEN,
            updated_at=CREATED_AT,
            significance=DecisionSignificance.NONE,
        ),
        make_item(
            external_id="1",
            item_type=KnowledgeItemType.ISSUE,
            state=KnowledgeItemState.CLOSED,
            updated_at=LATER_AT,
            significance=DecisionSignificance.CRITICAL,
        ),
    ]
    result = search(make_query(text="parser"), *items)

    assert [match.item_key for match in result.matches] == sorted(
        match.item_key for match in result.matches
    )
    assert {match.state for match in result.matches} == {
        KnowledgeItemState.OPEN,
        KnowledgeItemState.CLOSED,
    }


def test_max_results_applies_after_ranking_and_counts_all_matches() -> None:
    items = [
        make_item(external_id=str(index))
        for index in range(1, 4)
    ]
    result = search(
        make_query(text="parser"),
        *items,
        options=KnowledgeLexicalSearchOptions(max_results=2),
    )

    assert result.corpus_total_count == 3
    assert result.eligible_document_count == 3
    assert result.matched_document_count == 3
    assert result.returned_count == 2
    assert result.truncated is True
    assert [match.reference.external_id for match in result.matches] == ["1", "2"]


def test_empty_matches_are_valid() -> None:
    result = search(
        make_query(text="absent"),
        make_item(title="unrelated", summary=None, body=None),
    )

    assert result.matched_document_count == 0
    assert result.matches == []
    assert result.returned_count == 0
    assert result.truncated is False


def test_result_rejects_repository_and_duplicate_match_keys() -> None:
    valid = search(make_query(text="parser"))
    foreign_match = make_match(item=make_item(repository=OTHER_REPOSITORY))
    payload = result_payload(valid)
    payload["matches"] = [foreign_match]
    payload["matched_document_count"] = 1

    with pytest.raises(ValidationError, match="result repository"):
        KnowledgeLexicalSearchResult.model_validate(payload)

    match = valid.matches[0]
    payload = result_payload(valid)
    payload["matches"] = [match, match]
    payload["matched_document_count"] = 2
    payload["eligible_document_count"] = 2
    payload["corpus_total_count"] = 2
    with pytest.raises(ValidationError, match="unique"):
        KnowledgeLexicalSearchResult.model_validate(payload)


@pytest.mark.parametrize(
    ("corpus_count", "eligible_count", "matched_count", "match_count"),
    [
        (0, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 1, 0, 1),
    ],
)
def test_result_rejects_impossible_counts(
    corpus_count: int,
    eligible_count: int,
    matched_count: int,
    match_count: int,
) -> None:
    match = make_match()

    with pytest.raises(ValidationError):
        KnowledgeLexicalSearchResult(
            repository=REPOSITORY,
            query=make_query(text="parser"),
            options=KnowledgeLexicalSearchOptions(),
            corpus_total_count=corpus_count,
            eligible_document_count=eligible_count,
            matched_document_count=matched_count,
            matches=[match] * match_count,
        )


def test_result_rejects_incorrect_ranking_and_zero_score_match() -> None:
    first = make_match(item=make_item(external_id="1"))
    second = make_match(item=make_item(external_id="2"))

    with pytest.raises(ValidationError, match="ranking"):
        KnowledgeLexicalSearchResult(
            repository=REPOSITORY,
            query=make_query(text="parser"),
            options=KnowledgeLexicalSearchOptions(),
            corpus_total_count=2,
            eligible_document_count=2,
            matched_document_count=2,
            matches=[second, first],
        )

    zero_score = KnowledgeLexicalMatch.model_construct(
        reference=make_item().to_reference(),
        state=KnowledgeItemState.CLOSED,
        decision_significance=DecisionSignificance.HIGH,
        updated_at=UPDATED_AT,
        evidence=[],
    )
    with pytest.raises(ValidationError, match="minimum_score"):
        KnowledgeLexicalSearchResult(
            repository=REPOSITORY,
            query=make_query(text="parser"),
            options=KnowledgeLexicalSearchOptions(),
            corpus_total_count=1,
            eligible_document_count=1,
            matched_document_count=1,
            matches=[zero_score],
        )


def test_result_enforces_option_threshold_and_result_limit() -> None:
    first = make_match(item=make_item(external_id="1"))
    second = make_match(item=make_item(external_id="2"))

    with pytest.raises(ValidationError, match="minimum_score"):
        KnowledgeLexicalSearchResult(
            repository=REPOSITORY,
            query=make_query(text="parser"),
            options=KnowledgeLexicalSearchOptions(minimum_score=6),
            corpus_total_count=1,
            eligible_document_count=1,
            matched_document_count=1,
            matches=[first],
        )

    with pytest.raises(ValidationError, match="max_results"):
        KnowledgeLexicalSearchResult(
            repository=REPOSITORY,
            query=make_query(text="parser"),
            options=KnowledgeLexicalSearchOptions(max_results=1),
            corpus_total_count=2,
            eligible_document_count=2,
            matched_document_count=2,
            matches=[first, second],
        )


def test_result_json_includes_evidence_computed_fields_enums_and_utc() -> None:
    result = search(
        make_query(text="parser"),
        make_item(
            title="parser",
            summary=None,
            body=None,
            updated_at=NON_UTC_UPDATED_AT,
        ),
    )
    payload = result.model_dump(mode="json")
    match = payload["matches"][0]

    assert payload["options"] == {
        "max_results": 20,
        "minimum_score": 1,
    }
    assert match["state"] == "closed"
    assert match["decision_significance"] == "high"
    assert match["updated_at"] == "2026-05-03T12:00:00Z"
    assert match["evidence"][0]["kind"] == "term"
    assert match["evidence"][0]["field"] == "title"
    assert match["item_key"] == result.matches[0].item_key
    assert match["raw_score"] == result.matches[0].raw_score
    assert match["score"] == result.matches[0].score
    assert match["matched_fields"] == ["title"]
    assert match["matched_kinds"] == ["term"]
    assert payload["returned_count"] == 1
    assert payload["truncated"] is False


def test_search_does_not_mutate_query_corpus_documents_or_lists() -> None:
    query = make_query(
        text="issue #42 ParserRegistry parser",
        exact_phrases=["parser registry"],
        identifiers=["ParserRegistry"],
        labels=["Feature"],
        components=["Parser"],
        affected_paths=["src/parser/registry.py"],
    )
    corpus = make_corpus()
    query_before = query.model_dump(mode="json")
    corpus_before = corpus.model_dump(mode="json")
    documents_before = list(corpus.documents)
    evidence_lists_before = [
        (
            list(document.title_terms),
            list(document.title_identifiers),
            list(document.labels),
            list(document.components),
            list(document.affected_paths),
        )
        for document in corpus.documents
    ]

    search_knowledge_lexical_corpus(query, corpus)

    assert query.model_dump(mode="json") == query_before
    assert corpus.model_dump(mode="json") == corpus_before
    assert corpus.documents == documents_before
    assert [
        (
            document.title_terms,
            document.title_identifiers,
            document.labels,
            document.components,
            document.affected_paths,
        )
        for document in corpus.documents
    ] == evidence_lists_before


def test_empty_corpus_and_filters_with_zero_eligible_documents() -> None:
    query = make_query(text="parser")
    empty_result = search_knowledge_lexical_corpus(
        query,
        build_knowledge_lexical_corpus(REPOSITORY, []),
    )
    filtered_result = search(
        make_query(
            text="parser",
            item_types=[KnowledgeItemType.ADR],
        ),
    )

    assert empty_result.corpus_total_count == 0
    assert empty_result.eligible_document_count == 0
    assert empty_result.matched_document_count == 0
    assert empty_result.matches == []
    assert filtered_result.corpus_total_count == 1
    assert filtered_result.eligible_document_count == 0
    assert filtered_result.matches == []


def test_public_models_reject_extra_fields() -> None:
    evidence_payload = make_evidence().model_dump()
    evidence_payload["weight"] = 10
    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeLexicalMatchEvidence.model_validate(evidence_payload)

    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeLexicalSearchOptions.model_validate(
            {
                "max_results": 20,
                "minimum_score": 1,
                "semantic": True,
            }
        )


def test_match_kind_enum_has_exact_values() -> None:
    assert [kind.value for kind in KnowledgeLexicalMatchKind] == [
        "reference",
        "exact_phrase",
        "identifier",
        "label",
        "component",
        "affected_path_exact",
        "affected_path_shared_directory",
        "term",
    ]


def test_search_accepts_prepared_documents_only() -> None:
    document = build_knowledge_lexical_document(make_item())
    corpus = KnowledgeLexicalCorpus(
        repository=REPOSITORY,
        documents=[document],
    )

    result = search_knowledge_lexical_corpus(
        make_query(text="parser"),
        corpus,
    )

    assert isinstance(corpus.documents[0], KnowledgeLexicalDocument)
    assert result.returned_count == 1
