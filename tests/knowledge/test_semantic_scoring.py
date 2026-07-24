"""Tests for injectable provider-neutral semantic-scoring orchestration."""

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.knowledge import (
    MAX_KNOWLEDGE_SEMANTIC_DOCUMENT_CHARACTERS,
    MAX_KNOWLEDGE_SEMANTIC_DOCUMENTS,
    MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS,
    MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS,
    DecisionSignificance,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalCorpus,
    KnowledgeLexicalQuery,
    KnowledgeLexicalReference,
    KnowledgeRepositoryRef,
    KnowledgeSemanticScorerResponse,
    KnowledgeSemanticScorerScore,
    KnowledgeSemanticScoringDocument,
    KnowledgeSemanticScoringError,
    KnowledgeSemanticScoringOptions,
    KnowledgeSemanticScoringRequest,
    KnowledgeSemanticScoringResult,
    KnowledgeSemanticScoringService,
    KnowledgeSemanticScoringStatus,
    KnowledgeSemanticSimilarity,
    KnowledgeSourceKind,
    build_knowledge_lexical_corpus,
    build_knowledge_lexical_document,
)

CREATED_AT = datetime(2026, 1, 1, 9, 0, tzinfo=UTC)
UPDATED_AT = datetime(2026, 1, 2, 12, 0, tzinfo=UTC)
TRUNCATION_MARKER = "\n[truncated]"
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
    external_id: str = "1",
    repository: KnowledgeRepositoryRef = REPOSITORY,
    item_type: KnowledgeItemType = KnowledgeItemType.ISSUE,
    state: KnowledgeItemState = KnowledgeItemState.CLOSED,
    title: str = "Semantic Parser",
    summary: str | None = "Parser migration summary",
    body: str | None = "Adopt the parser registry.",
    labels: list[str] | None = None,
    components: list[str] | None = None,
    affected_paths: list[str] | None = None,
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
        url=f"https://example.test/items/{external_id}",
        created_at=CREATED_AT,
        updated_at=UPDATED_AT,
        labels=[] if labels is None else labels,
        components=[] if components is None else components,
        affected_paths=[] if affected_paths is None else affected_paths,
        decision_significance=significance,
    )


def make_query(**updates: Any) -> KnowledgeLexicalQuery:
    """Build a query for the default repository."""

    return KnowledgeLexicalQuery(repository=REPOSITORY, **updates)


def make_corpus(*items: KnowledgeItem) -> KnowledgeLexicalCorpus:
    """Build an ordered lexical corpus."""

    return build_knowledge_lexical_corpus(REPOSITORY, list(items))


def make_semantic_document(
    item: KnowledgeItem | None = None,
    *,
    text: str = "Title:\nsemantic parser",
    original_character_count: int | None = None,
    emitted_character_count: int | None = None,
    truncated: bool = False,
) -> KnowledgeSemanticScoringDocument:
    """Build one valid semantic scoring document."""

    source = item or make_item()
    return KnowledgeSemanticScoringDocument(
        reference=source.to_reference(),
        text=text,
        original_character_count=(
            len(text)
            if original_character_count is None
            else original_character_count
        ),
        emitted_character_count=(
            len(text)
            if emitted_character_count is None
            else emitted_character_count
        ),
        truncated=truncated,
    )


def make_request(
    *documents: KnowledgeSemanticScoringDocument,
    query_text: str = "Text:\nsemantic parser",
    query_original_character_count: int | None = None,
    query_emitted_character_count: int | None = None,
    query_truncated: bool = False,
) -> KnowledgeSemanticScoringRequest:
    """Build one complete semantic scorer request."""

    effective_documents = list(documents) or [make_semantic_document()]
    return KnowledgeSemanticScoringRequest(
        repository=REPOSITORY,
        query_text=query_text,
        query_original_character_count=(
            len(query_text)
            if query_original_character_count is None
            else query_original_character_count
        ),
        query_emitted_character_count=(
            len(query_text)
            if query_emitted_character_count is None
            else query_emitted_character_count
        ),
        query_truncated=query_truncated,
        documents=effective_documents,
    )


def response_for(
    request: KnowledgeSemanticScoringRequest,
    *,
    scores: list[int] | None = None,
    reverse: bool = True,
    provider: str = "test-provider",
    model: str = "test-model",
) -> KnowledgeSemanticScorerResponse:
    """Build exact response coverage, optionally in arbitrary order."""

    values = scores or [50] * len(request.documents)
    response_scores = [
        KnowledgeSemanticScorerScore(
            reference=document.reference,
            score=score,
        )
        for document, score in zip(request.documents, values, strict=True)
    ]
    if reverse:
        response_scores.reverse()
    return KnowledgeSemanticScorerResponse(
        provider=provider,
        model=model,
        scores=response_scores,
    )


class RecordingScorer:
    """Injected asynchronous scorer with controllable behavior."""

    def __init__(
        self,
        response_factory: (
            Callable[[KnowledgeSemanticScoringRequest], Any] | None
        ) = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self.requests: list[KnowledgeSemanticScoringRequest] = []
        self.response_factory = response_factory or response_for
        self.error = error
        self.responses: list[Any] = []

    async def score(
        self,
        request: KnowledgeSemanticScoringRequest,
    ) -> KnowledgeSemanticScorerResponse:
        """Record and score one request."""

        self.requests.append(request)
        if self.error is not None:
            raise self.error
        response = self.response_factory(request)
        self.responses.append(response)
        return response


def result_payload(
    result: KnowledgeSemanticScoringResult,
) -> dict[str, object]:
    """Return declared result fields for validation mutations."""

    return {
        field_name: getattr(result, field_name)
        for field_name in KnowledgeSemanticScoringResult.model_fields
    }


def make_scored_result(
    *,
    item: KnowledgeItem | None = None,
) -> KnowledgeSemanticScoringResult:
    """Build one complete successful result for model validation tests."""

    source = item or make_item()
    return KnowledgeSemanticScoringResult(
        repository=REPOSITORY,
        query=make_query(text="semantic parser"),
        status=KnowledgeSemanticScoringStatus.SCORED,
        eligible_document_count=1,
        scored_document_count=1,
        provider="test-provider",
        model="test-model",
        query_truncated=False,
        truncated_document_count=0,
        emitted_character_count=100,
        similarities=[
            KnowledgeSemanticSimilarity(
                reference=source.to_reference(),
                score=50,
                provider="test-provider",
                model="test-model",
            )
        ],
    )


def test_default_options_and_safety_constants() -> None:
    options = KnowledgeSemanticScoringOptions()

    assert options == KnowledgeSemanticScoringOptions(
        max_documents=250,
        max_query_characters=8_000,
        max_document_characters=12_000,
        max_total_characters=500_000,
    )
    assert MAX_KNOWLEDGE_SEMANTIC_DOCUMENTS == 500
    assert MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS == 20_000
    assert MAX_KNOWLEDGE_SEMANTIC_DOCUMENT_CHARACTERS == 20_000
    assert MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS == 1_000_000


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("max_documents", 0),
        ("max_query_characters", 99),
        ("max_document_characters", 99),
        ("max_total_characters", 99),
        ("max_documents", MAX_KNOWLEDGE_SEMANTIC_DOCUMENTS + 1),
        (
            "max_query_characters",
            MAX_KNOWLEDGE_SEMANTIC_QUERY_CHARACTERS + 1,
        ),
        (
            "max_document_characters",
            MAX_KNOWLEDGE_SEMANTIC_DOCUMENT_CHARACTERS + 1,
        ),
        (
            "max_total_characters",
            MAX_KNOWLEDGE_SEMANTIC_TOTAL_CHARACTERS + 1,
        ),
    ],
)
def test_options_enforce_every_bound(field_name: str, value: int) -> None:
    with pytest.raises(ValidationError):
        KnowledgeSemanticScoringOptions(**{field_name: value})


def test_semantic_document_item_key_and_character_counts() -> None:
    document = make_semantic_document()

    assert document.item_key == document.reference.key
    assert document.emitted_character_count == len(document.text)
    assert document.original_character_count == len(document.text)

    with pytest.raises(ValidationError, match="document text length"):
        make_semantic_document(emitted_character_count=1)

    with pytest.raises(ValidationError, match="must not be less"):
        make_semantic_document(original_character_count=1)


@pytest.mark.parametrize(
    "updates",
    [
        {"original_character_count": 30, "truncated": False},
        {"truncated": True},
    ],
)
def test_semantic_document_validates_truncation_state(
    updates: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="truncation"):
        make_semantic_document(**updates)


def test_semantic_document_requires_marker_when_truncated() -> None:
    text = "bounded text"
    with pytest.raises(ValidationError, match="truncation marker"):
        make_semantic_document(
            text=text,
            original_character_count=len(text) + 1,
            truncated=True,
        )

    marked = f"bounded{TRUNCATION_MARKER}"
    document = make_semantic_document(
        text=marked,
        original_character_count=len(marked) + 20,
        truncated=True,
    )
    assert document.truncated is True

    with pytest.raises(ValidationError, match="Untruncated"):
        make_semantic_document(text=marked)


def test_semantic_request_validates_query_counts_and_truncation() -> None:
    request = make_request()

    assert request.query_emitted_character_count == len(request.query_text)
    assert request.query_original_character_count == len(request.query_text)

    with pytest.raises(ValidationError, match="query text length"):
        make_request(query_emitted_character_count=1)
    with pytest.raises(ValidationError, match="must not be less"):
        make_request(query_original_character_count=1)
    with pytest.raises(ValidationError, match="Query truncation state"):
        make_request(
            query_original_character_count=len(request.query_text) + 1,
        )


def test_semantic_request_requires_exact_query_marker_state() -> None:
    text = f"Text:\nbounded{TRUNCATION_MARKER}"
    valid = make_request(
        query_text=text,
        query_original_character_count=len(text) + 5,
        query_truncated=True,
    )
    assert valid.query_truncated is True

    with pytest.raises(ValidationError, match="must end"):
        make_request(
            query_original_character_count=len("Text:\nsemantic parser") + 1,
            query_truncated=True,
        )
    with pytest.raises(ValidationError, match="Untruncated"):
        make_request(query_text=text)


def test_semantic_request_rejects_duplicate_cross_repository_and_wrong_order() -> None:
    first = make_semantic_document(make_item(external_id="1"))
    second = make_semantic_document(make_item(external_id="2"))
    foreign = make_semantic_document(
        make_item(external_id="3", repository=OTHER_REPOSITORY)
    )

    with pytest.raises(ValidationError, match="unique"):
        make_request(first, first)
    with pytest.raises(ValidationError, match="request repository"):
        make_request(foreign)
    with pytest.raises(ValidationError, match="ascending"):
        make_request(second, first)


def test_semantic_request_computed_counts_and_character_totals() -> None:
    first = make_semantic_document(
        make_item(external_id="1"),
        text="Title:\none",
    )
    second = make_semantic_document(
        make_item(external_id="2"),
        text="Title:\ntwo",
    )
    request = make_request(first, second)

    assert request.document_count == 2
    assert request.document_item_keys == [first.item_key, second.item_key]
    assert request.total_document_characters == len(first.text) + len(second.text)
    assert request.total_request_characters == (
        len(request.query_text) + request.total_document_characters
    )


def test_scorer_score_accepts_boundaries_and_rejects_out_of_range() -> None:
    reference = make_item().to_reference()

    assert KnowledgeSemanticScorerScore(reference=reference, score=0).score == 0
    assert KnowledgeSemanticScorerScore(reference=reference, score=100).score == 100
    for invalid_score in (-1, 101):
        with pytest.raises(ValidationError):
            KnowledgeSemanticScorerScore(
                reference=reference,
                score=invalid_score,
            )


def test_scorer_response_validates_provenance_and_unique_keys() -> None:
    score = KnowledgeSemanticScorerScore(
        reference=make_item().to_reference(),
        score=50,
    )
    for field_name in ("provider", "model"):
        payload = {
            "provider": "provider",
            "model": "model",
            "scores": [],
        }
        payload[field_name] = " "
        with pytest.raises(ValidationError):
            KnowledgeSemanticScorerResponse.model_validate(payload)

    with pytest.raises(ValidationError, match="unique"):
        KnowledgeSemanticScorerResponse(
            provider="provider",
            model="model",
            scores=[score, score],
        )

    empty = KnowledgeSemanticScorerResponse(
        provider="provider",
        model="model",
        scores=[],
    )
    assert empty.scores == []


def test_result_rejects_extra_fields() -> None:
    payload = result_payload(make_scored_result())
    payload["usage"] = {"tokens": 100}

    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeSemanticScoringResult.model_validate(payload)


@pytest.mark.asyncio
async def test_repository_mismatch_raises_before_scorer_invocation() -> None:
    scorer = RecordingScorer()
    service = KnowledgeSemanticScoringService(scorer=scorer)
    query = KnowledgeLexicalQuery(
        repository=OTHER_REPOSITORY,
        text="semantic",
    )

    with pytest.raises(KnowledgeSemanticScoringError, match="repositories"):
        await service.score(query, make_corpus(make_item()))
    assert scorer.requests == []


@pytest.mark.asyncio
async def test_item_type_and_state_filters_use_and_semantics() -> None:
    closed_issue = make_item(external_id="1")
    open_issue = make_item(
        external_id="2",
        state=KnowledgeItemState.OPEN,
    )
    closed_adr = make_item(
        external_id="3",
        item_type=KnowledgeItemType.ADR,
        state=KnowledgeItemState.ACTIVE,
    )
    corpus = make_corpus(closed_adr, open_issue, closed_issue)
    scorer = RecordingScorer()
    service = KnowledgeSemanticScoringService(scorer=scorer)

    result = await service.score(
        make_query(
            text="semantic",
            item_types=[KnowledgeItemType.ISSUE],
            states=[KnowledgeItemState.CLOSED],
        ),
        corpus,
    )

    assert result.eligible_document_count == 1
    assert scorer.requests[0].document_item_keys == [closed_issue.key]


@pytest.mark.asyncio
async def test_empty_filters_allow_all_and_eligible_documents_are_sorted() -> None:
    items = [
        make_item(external_id="3"),
        make_item(external_id="1"),
        make_item(external_id="2"),
    ]
    scorer = RecordingScorer()

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(*items),
    )

    assert result.eligible_document_count == 3
    assert scorer.requests[0].document_item_keys == sorted(item.key for item in items)


@pytest.mark.asyncio
async def test_duplicate_eligible_keys_are_rejected_defensively() -> None:
    document = build_knowledge_lexical_document(make_item())
    malformed_corpus = KnowledgeLexicalCorpus.model_construct(
        repository=REPOSITORY,
        documents=[document, document],
    )
    scorer = RecordingScorer()

    with pytest.raises(KnowledgeSemanticScoringError, match="keys must be unique"):
        await KnowledgeSemanticScoringService(scorer=scorer).score(
            make_query(text="semantic"),
            malformed_corpus,
        )
    assert scorer.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        make_query(text="semantic"),
        make_query(
            text="semantic",
            item_types=[KnowledgeItemType.ADR],
        ),
    ],
)
async def test_no_eligible_documents_skips_scorer(
    query: KnowledgeLexicalQuery,
) -> None:
    corpus = (
        make_corpus()
        if not query.item_types
        else make_corpus(make_item(item_type=KnowledgeItemType.ISSUE))
    )
    scorer = RecordingScorer()

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        query,
        corpus,
    )

    assert result.status == KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS
    assert result.eligible_document_count == 0
    assert result.scored_document_count == 0
    assert result.provider is None
    assert result.model is None
    assert result.similarities == []
    assert result.query_truncated is False
    assert result.truncated_document_count == 0
    assert result.emitted_character_count == 0
    assert result.performed is False
    assert result.complete_coverage is False
    assert scorer.requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        make_query(labels=["semantic-label"]),
        make_query(components=["semantic-component"]),
        make_query(affected_paths=["src/semantic.py"]),
        make_query(
            references=[KnowledgeLexicalReference(external_id="1")]
        ),
    ],
)
async def test_structured_only_query_skips_semantic_scoring(
    query: KnowledgeLexicalQuery,
) -> None:
    scorer = RecordingScorer()

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        query,
        make_corpus(make_item()),
    )

    assert result.status == KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY
    assert result.eligible_document_count == 1
    assert result.scored_document_count == 0
    assert result.provider is None
    assert result.model is None
    assert result.similarities == []
    assert result.query_truncated is False
    assert result.truncated_document_count == 0
    assert result.emitted_character_count == 0
    assert result.performed is False
    assert result.complete_coverage is False
    assert scorer.requests == []


@pytest.mark.asyncio
async def test_query_sections_are_canonical_deterministic_and_bounded_to_text() -> None:
    query = make_query(
        text="Use PARSERREGISTRY",
        exact_phrases=["  \uff26\uff4f\uff4f\t BAR  "],
        identifiers=["ParserRegistry", "ExplicitID"],
        labels=["label-secret"],
        components=["component-secret"],
        affected_paths=["path-secret/file.py"],
        references=[KnowledgeLexicalReference(external_id="reference-secret")],
        item_types=[KnowledgeItemType.ISSUE],
        states=[KnowledgeItemState.CLOSED],
    )
    scorer = RecordingScorer()

    await KnowledgeSemanticScoringService(scorer=scorer).score(
        query,
        make_corpus(make_item()),
    )
    request = scorer.requests[0]

    assert request.query_text == (
        "Text:\nuse parserregistry\n\n"
        "Exact phrases:\nfoo bar\n\n"
        "Identifiers:\nParserRegistry\nExplicitID"
    )
    assert request.query_text.count("ParserRegistry") == 1
    assert "PARSERREGISTRY" not in request.query_text.split("Identifiers:\n")[1]
    for excluded in (
        "label-secret",
        "component-secret",
        "path-secret",
        "reference-secret",
        "issue",
        "closed",
    ):
        assert excluded not in request.query_text
    assert "\n\n" in request.query_text
    assert not request.query_text.endswith("\n")
    assert request.query_truncated is False
    assert request.query_original_character_count == len(request.query_text)
    assert request.query_emitted_character_count == len(request.query_text)


@pytest.mark.asyncio
async def test_text_extracted_identifier_is_included() -> None:
    scorer = RecordingScorer()

    await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="Use ParserRegistry"),
        make_corpus(make_item()),
    )

    assert scorer.requests[0].query_text.endswith(
        "Identifiers:\nParserRegistry"
    )


@pytest.mark.asyncio
async def test_query_truncates_once_to_exact_configured_length() -> None:
    scorer = RecordingScorer()
    query = make_query(text="semantic " * 40)

    await KnowledgeSemanticScoringService(scorer=scorer).score(
        query,
        make_corpus(make_item()),
        options=KnowledgeSemanticScoringOptions(
            max_query_characters=100,
        ),
    )
    request = scorer.requests[0]

    assert request.query_truncated is True
    assert request.query_text.endswith(TRUNCATION_MARKER)
    assert len(request.query_text) == 100
    assert request.query_emitted_character_count == 100
    assert request.query_original_character_count > 100


@pytest.mark.asyncio
async def test_document_sections_include_only_natural_language_fields() -> None:
    item = make_item(
        external_id="external-secret",
        state=KnowledgeItemState.ACTIVE,
        title="Semantic Title",
        summary="Semantic Summary",
        body="Semantic Body",
        labels=["label-secret"],
        components=["component-secret"],
        affected_paths=["path-secret/file.py"],
        significance=DecisionSignificance.CRITICAL,
    )
    scorer = RecordingScorer()

    await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(item),
    )
    document = scorer.requests[0].documents[0]

    assert document.text == (
        "Title:\nsemantic title\n\n"
        "Summary:\nsemantic summary\n\n"
        "Body:\nsemantic body"
    )
    for excluded in (
        "external-secret",
        "label-secret",
        "component-secret",
        "path-secret",
        "active",
        "critical",
        "2026-01",
        "github",
        "acme",
        "framework",
    ):
        assert excluded not in document.text
    assert document.truncated is False
    assert document.original_character_count == len(document.text)
    assert document.emitted_character_count == len(document.text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("summary", "body", "expected"),
    [
        (None, "semantic body", "Title:\ntitle\n\nBody:\nsemantic body"),
        ("semantic summary", None, "Title:\ntitle\n\nSummary:\nsemantic summary"),
        (None, None, "Title:\ntitle"),
    ],
)
async def test_missing_document_sections_are_omitted(
    summary: str | None,
    body: str | None,
    expected: str,
) -> None:
    scorer = RecordingScorer()

    await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(make_item(title="Title", summary=summary, body=body)),
    )

    assert scorer.requests[0].documents[0].text == expected


@pytest.mark.asyncio
async def test_document_truncates_to_exact_configured_length() -> None:
    scorer = RecordingScorer()
    item = make_item(
        title="x" * 150,
        summary=None,
        body=None,
    )

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(item),
        options=KnowledgeSemanticScoringOptions(
            max_document_characters=100,
        ),
    )
    document = scorer.requests[0].documents[0]

    assert document.truncated is True
    assert document.text.endswith(TRUNCATION_MARKER)
    assert len(document.text) == 100
    assert document.emitted_character_count == 100
    assert document.original_character_count == len("Title:\n") + 150
    assert result.truncated_document_count == 1


@pytest.mark.asyncio
async def test_multiple_documents_preserve_item_key_order() -> None:
    items = [
        make_item(external_id="20"),
        make_item(external_id="10"),
        make_item(external_id="30"),
    ]
    scorer = RecordingScorer()

    await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(*items),
    )

    assert scorer.requests[0].document_item_keys == sorted(item.key for item in items)


@pytest.mark.asyncio
async def test_document_count_at_limit_is_accepted_without_drops_or_batches() -> None:
    items = [make_item(external_id=str(index)) for index in range(2)]
    scorer = RecordingScorer()

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(*items),
        options=KnowledgeSemanticScoringOptions(max_documents=2),
    )

    assert result.eligible_document_count == 2
    assert result.scored_document_count == 2
    assert len(scorer.requests) == 1
    assert scorer.requests[0].document_count == 2


@pytest.mark.asyncio
async def test_document_count_above_limit_raises_before_scorer() -> None:
    scorer = RecordingScorer()
    corpus = make_corpus(
        make_item(external_id="1"),
        make_item(external_id="2"),
    )

    with pytest.raises(
        KnowledgeSemanticScoringError,
        match=(
            "^Eligible document count exceeds the semantic-scoring safety "
            "limit\\.$"
        ),
    ):
        await KnowledgeSemanticScoringService(scorer=scorer).score(
            make_query(text="semantic"),
            corpus,
            options=KnowledgeSemanticScoringOptions(max_documents=1),
        )
    assert scorer.requests == []


@pytest.mark.asyncio
async def test_total_character_count_equal_to_limit_is_accepted() -> None:
    scorer = RecordingScorer()
    item = make_item(
        title="x" * 81,
        summary=None,
        body=None,
    )

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="anchor"),
        make_corpus(item),
        options=KnowledgeSemanticScoringOptions(max_total_characters=100),
    )

    assert scorer.requests[0].total_request_characters == 100
    assert result.emitted_character_count == 100


@pytest.mark.asyncio
async def test_total_character_count_above_limit_raises_before_scorer() -> None:
    scorer = RecordingScorer()
    item = make_item(
        title="x" * 82,
        summary=None,
        body=None,
    )

    with pytest.raises(
        KnowledgeSemanticScoringError,
        match=(
            "^Prepared semantic request exceeds the total-character safety "
            "limit\\.$"
        ),
    ):
        await KnowledgeSemanticScoringService(scorer=scorer).score(
            make_query(text="anchor"),
            make_corpus(item),
            options=KnowledgeSemanticScoringOptions(
                max_total_characters=100,
            ),
        )
    assert scorer.requests == []


@pytest.mark.asyncio
async def test_valid_request_invokes_scorer_exactly_once_with_complete_identity() -> None:
    items = [
        make_item(external_id="2"),
        make_item(external_id="1"),
    ]
    scorer = RecordingScorer()

    await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(*items),
    )

    assert len(scorer.requests) == 1
    request = scorer.requests[0]
    assert isinstance(request, KnowledgeSemanticScoringRequest)
    assert request.repository == REPOSITORY
    assert request.document_item_keys == sorted(item.key for item in items)


class SentinelScorerError(RuntimeError):
    """Sentinel provider failure used to verify unchanged propagation."""


@pytest.mark.asyncio
async def test_provider_exception_propagates_unchanged_without_partial_result() -> None:
    sentinel = SentinelScorerError("provider unavailable")
    scorer = RecordingScorer(error=sentinel)

    with pytest.raises(SentinelScorerError) as raised:
        await KnowledgeSemanticScoringService(scorer=scorer).score(
            make_query(text="semantic"),
            make_corpus(make_item()),
        )

    assert raised.value is sentinel
    assert len(scorer.requests) == 1
    assert scorer.responses == []


@pytest.mark.asyncio
async def test_exact_response_coverage_accepts_arbitrary_response_order() -> None:
    items = [
        make_item(external_id="1"),
        make_item(external_id="2"),
    ]
    scorer = RecordingScorer(
        lambda request: response_for(
            request,
            scores=[0, 100],
            reverse=True,
        )
    )

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(*items),
    )

    assert [similarity.item_key for similarity in result.similarities] == sorted(
        item.key for item in items
    )
    assert [similarity.score for similarity in result.similarities] == [0, 100]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["missing", "extra", "empty", "foreign"])
async def test_response_requires_exact_coverage(mode: str) -> None:
    def malformed(
        request: KnowledgeSemanticScoringRequest,
    ) -> KnowledgeSemanticScorerResponse:
        scores = response_for(request, reverse=False).scores
        if mode == "missing":
            scores = scores[:-1]
        elif mode == "extra":
            scores.append(
                KnowledgeSemanticScorerScore(
                    reference=make_item(external_id="extra").to_reference(),
                    score=50,
                )
            )
        elif mode == "empty":
            scores = []
        else:
            scores = [
                KnowledgeSemanticScorerScore(
                    reference=make_item(
                        external_id="foreign",
                        repository=OTHER_REPOSITORY,
                    ).to_reference(),
                    score=50,
                )
            ]
        return KnowledgeSemanticScorerResponse(
            provider="provider",
            model="model",
            scores=scores,
        )

    scorer = RecordingScorer(malformed)
    items = [make_item(external_id="1"), make_item(external_id="2")]

    with pytest.raises(
        KnowledgeSemanticScoringError,
        match=(
            "^Semantic scorer response must exactly cover every request "
            "document\\.$"
        ),
    ):
        await KnowledgeSemanticScoringService(scorer=scorer).score(
            make_query(text="semantic"),
            make_corpus(*items),
        )


@pytest.mark.asyncio
async def test_same_key_with_different_complete_reference_is_rejected() -> None:
    def mismatched(
        request: KnowledgeSemanticScoringRequest,
    ) -> KnowledgeSemanticScorerResponse:
        reference = request.documents[0].reference.model_copy(
            update={"title": "Different title"}
        )
        return KnowledgeSemanticScorerResponse(
            provider="provider",
            model="model",
            scores=[
                KnowledgeSemanticScorerScore(
                    reference=reference,
                    score=50,
                )
            ],
        )

    scorer = RecordingScorer(mismatched)

    with pytest.raises(
        KnowledgeSemanticScoringError,
        match=(
            "^Semantic scorer response reference does not match the request "
            "document\\.$"
        ),
    ):
        await KnowledgeSemanticScoringService(scorer=scorer).score(
            make_query(text="semantic"),
            make_corpus(make_item()),
        )


@pytest.mark.asyncio
async def test_invalid_scorer_response_model_is_not_silently_accepted() -> None:
    scorer = RecordingScorer(
        lambda request: KnowledgeSemanticScorerResponse.model_construct(
            provider="",
            model="model",
            scores=response_for(request).scores,
        )
    )

    with pytest.raises(ValidationError):
        await KnowledgeSemanticScoringService(scorer=scorer).score(
            make_query(text="semantic"),
            make_corpus(make_item()),
        )


@pytest.mark.asyncio
async def test_similarity_conversion_uses_request_identity_and_provenance() -> None:
    scorer = RecordingScorer(
        lambda request: response_for(
            request,
            scores=[73],
            provider="semantic-provider",
            model="semantic-model-v1",
        )
    )

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(make_item()),
    )
    request_reference = scorer.requests[0].documents[0].reference
    similarity = result.similarities[0]

    assert similarity.reference is request_reference
    assert similarity.score == 73
    assert similarity.provider == "semantic-provider"
    assert similarity.model == "semantic-model-v1"
    assert result.provider == "semantic-provider"
    assert result.model == "semantic-model-v1"


@pytest.mark.asyncio
async def test_success_result_reports_complete_metadata_and_computed_fields() -> None:
    scorer = RecordingScorer()
    items = [
        make_item(
            external_id="1",
            title="x" * 150,
            summary=None,
            body=None,
        ),
        make_item(external_id="2"),
    ]

    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic " * 40),
        make_corpus(*items),
        options=KnowledgeSemanticScoringOptions(
            max_query_characters=100,
            max_document_characters=100,
        ),
    )
    request = scorer.requests[0]

    assert result.status == KnowledgeSemanticScoringStatus.SCORED
    assert result.eligible_document_count == 2
    assert result.scored_document_count == 2
    assert result.provider == "test-provider"
    assert result.model == "test-model"
    assert result.query_truncated is True
    assert result.truncated_document_count == 1
    assert result.emitted_character_count == request.total_request_characters
    assert result.performed is True
    assert result.complete_coverage is True


def test_result_rejects_repository_key_order_and_provenance_inconsistencies() -> None:
    valid = make_scored_result()

    payload = result_payload(valid)
    payload["query"] = KnowledgeLexicalQuery(
        repository=OTHER_REPOSITORY,
        text="semantic",
    )
    with pytest.raises(ValidationError, match="query"):
        KnowledgeSemanticScoringResult.model_validate(payload)

    foreign_similarity = KnowledgeSemanticSimilarity(
        reference=make_item(repository=OTHER_REPOSITORY).to_reference(),
        score=50,
        provider="test-provider",
        model="test-model",
    )
    payload = result_payload(valid)
    payload["similarities"] = [foreign_similarity]
    with pytest.raises(ValidationError, match="result repository"):
        KnowledgeSemanticScoringResult.model_validate(payload)

    payload = result_payload(valid)
    payload["similarities"] = [valid.similarities[0], valid.similarities[0]]
    payload["eligible_document_count"] = 2
    payload["scored_document_count"] = 2
    with pytest.raises(ValidationError, match="unique"):
        KnowledgeSemanticScoringResult.model_validate(payload)

    first = make_scored_result(item=make_item(external_id="1")).similarities[0]
    second = make_scored_result(item=make_item(external_id="2")).similarities[0]
    payload = result_payload(valid)
    payload["similarities"] = [second, first]
    payload["eligible_document_count"] = 2
    payload["scored_document_count"] = 2
    with pytest.raises(ValidationError, match="ascending"):
        KnowledgeSemanticScoringResult.model_validate(payload)

    for field_name, value in (
        ("provider", "other-provider"),
        ("model", "other-model"),
    ):
        payload = result_payload(valid)
        payload[field_name] = value
        with pytest.raises(ValidationError, match=field_name):
            KnowledgeSemanticScoringResult.model_validate(payload)


@pytest.mark.parametrize(
    ("status", "updates"),
    [
        (
            KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS,
            {"eligible_document_count": 1},
        ),
        (
            KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY,
            {"eligible_document_count": 0},
        ),
        (
            KnowledgeSemanticScoringStatus.SCORED,
            {"scored_document_count": 0},
        ),
        (
            KnowledgeSemanticScoringStatus.SCORED,
            {"emitted_character_count": 0},
        ),
        (
            KnowledgeSemanticScoringStatus.SCORED,
            {"provider": None},
        ),
    ],
)
def test_result_rejects_inconsistent_status_fields(
    status: KnowledgeSemanticScoringStatus,
    updates: dict[str, object],
) -> None:
    valid = make_scored_result()
    payload = result_payload(valid)
    payload["status"] = status
    payload.update(updates)

    with pytest.raises(ValidationError):
        KnowledgeSemanticScoringResult.model_validate(payload)


def test_result_rejects_truncation_count_above_scored_count() -> None:
    payload = result_payload(make_scored_result())
    payload["truncated_document_count"] = 2

    with pytest.raises(ValidationError, match="must not exceed"):
        KnowledgeSemanticScoringResult.model_validate(payload)


@pytest.mark.asyncio
async def test_json_serialization_includes_status_provenance_and_computed_fields() -> None:
    scorer = RecordingScorer()
    result = await KnowledgeSemanticScoringService(scorer=scorer).score(
        make_query(text="semantic"),
        make_corpus(make_item()),
    )

    payload = result.model_dump(mode="json")

    assert payload["status"] == "scored"
    assert payload["provider"] == "test-provider"
    assert payload["model"] == "test-model"
    assert payload["similarities"][0]["provider"] == "test-provider"
    assert payload["similarities"][0]["model"] == "test-model"
    assert payload["similarities"][0]["score"] == 50
    assert payload["performed"] is True
    assert payload["complete_coverage"] is True


@pytest.mark.asyncio
async def test_service_does_not_mutate_inputs_or_scorer_response() -> None:
    query = make_query(
        text="Use ParserRegistry",
        exact_phrases=["Parser Registry"],
    )
    item = make_item()
    corpus = make_corpus(item)
    document = corpus.documents[0]
    scorer = RecordingScorer()
    query_before = query.model_dump()
    corpus_before = corpus.model_dump()
    document_before = document.model_dump()

    await KnowledgeSemanticScoringService(scorer=scorer).score(
        query,
        corpus,
    )
    response = scorer.responses[0]
    response_before = response.model_dump()

    assert query.model_dump() == query_before
    assert corpus.model_dump() == corpus_before
    assert document.model_dump() == document_before
    assert response.model_dump() == response_before
