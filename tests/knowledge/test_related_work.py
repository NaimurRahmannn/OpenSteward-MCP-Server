"""Tests for provider-independent related-work orchestration."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

import opensteward.knowledge.related_work as related_work_module
from opensteward.knowledge import (
    MAX_KNOWLEDGE_LEXICAL_RESULTS,
    MAX_KNOWLEDGE_RELATED_WORK_RESULTS,
    DecisionSignificance,
    KnowledgeActor,
    KnowledgeActorType,
    KnowledgeHybridRankingMode,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalCorpus,
    KnowledgeLexicalQuery,
    KnowledgeLexicalSearchOptions,
    KnowledgeRelatedWorkItemSummary,
    KnowledgeRelatedWorkMatch,
    KnowledgeRelatedWorkMode,
    KnowledgeRelatedWorkOptions,
    KnowledgeRelatedWorkResult,
    KnowledgeRelatedWorkService,
    KnowledgeRepositoryRef,
    KnowledgeSemanticScoringOptions,
    KnowledgeSemanticScoringResult,
    KnowledgeSemanticScoringStatus,
    KnowledgeSemanticSimilarity,
    KnowledgeSourceKind,
    rank_knowledge_hybrid_corpus,
    search_knowledge_lexical_corpus,
)

REPOSITORY = KnowledgeRepositoryRef(
    provider="test",
    namespace="acme",
    name="framework",
)
OTHER_REPOSITORY = KnowledgeRepositoryRef(
    provider="test",
    namespace="other",
    name="framework",
)
CREATED_AT = datetime(2025, 1, 1, 10, 0, tzinfo=UTC)
UPDATED_AT = datetime(2025, 6, 1, 10, 0, tzinfo=UTC)
AS_OF = datetime(2026, 1, 1, 10, 0, tzinfo=UTC)
NON_UTC_AS_OF = datetime(
    2026,
    1,
    1,
    16,
    0,
    tzinfo=timezone(timedelta(hours=6)),
)
FALLBACK_WARNING = (
    "Lexical candidate retrieval reached its safety limit; semantic and hybrid "
    "ranking were skipped to avoid incomplete score fusion."
)


def make_item(
    external_id: str = "001",
    *,
    repository: KnowledgeRepositoryRef = REPOSITORY,
    title: str = "Adopt parser registry",
    body: str | None = "Historical implementation details that must stay private.",
    summary: str | None = "Concise parser decision.",
    url: str | None = "https://example.test/items/001",
    state: KnowledgeItemState = KnowledgeItemState.CLOSED,
    significance: DecisionSignificance = DecisionSignificance.MEDIUM,
    labels: list[str] | None = None,
    affected_paths: list[str] | None = None,
    components: list[str] | None = None,
    updated_at: datetime = UPDATED_AT,
) -> KnowledgeItem:
    """Build one real normalized source item."""

    return KnowledgeItem(
        repository=repository,
        item_type=KnowledgeItemType.ISSUE,
        external_id=external_id,
        source_kind=KnowledgeSourceKind.MANUAL,
        state=state,
        title=title,
        body=body,
        summary=summary,
        url=url,
        author=KnowledgeActor(
            identifier="maintainer",
            actor_type=KnowledgeActorType.USER,
        ),
        created_at=CREATED_AT,
        updated_at=updated_at,
        closed_at=updated_at,
        labels=labels or ["parser", "architecture"],
        affected_paths=affected_paths or ["src/parser/registry.py"],
        components=components or ["Parser"],
        decision_significance=significance,
    )


def make_query(**updates: Any) -> KnowledgeLexicalQuery:
    """Build one valid related-work query."""

    payload: dict[str, Any] = {
        "repository": REPOSITORY,
        "text": "parser",
        "affected_paths": ["src/parser/service.py"],
        "labels": ["parser"],
    }
    payload.update(updates)
    return KnowledgeLexicalQuery(**payload)


def eligible_documents(
    query: KnowledgeLexicalQuery,
    corpus: KnowledgeLexicalCorpus,
) -> list[Any]:
    """Apply the public query filters for the recording semantic double."""

    return [
        document
        for document in corpus.documents
        if (
            not query.item_types
            or document.reference.item_type in query.item_types
        )
        and (not query.states or document.state in query.states)
    ]


class SentinelSemanticError(RuntimeError):
    """Distinct fake provider failure."""


class RecordingSemanticService:
    """Injected semantic service supporting every orchestration outcome."""

    def __init__(
        self,
        status: KnowledgeSemanticScoringStatus,
        *,
        scores: dict[str, int] | None = None,
        error: Exception | None = None,
        result: KnowledgeSemanticScoringResult | None = None,
    ) -> None:
        self.status = status
        self.scores = scores or {}
        self.error = error
        self.result = result
        self.calls: list[
            tuple[
                KnowledgeLexicalQuery,
                KnowledgeLexicalCorpus,
                KnowledgeSemanticScoringOptions | None,
            ]
        ] = []

    async def score(
        self,
        query: KnowledgeLexicalQuery,
        corpus: KnowledgeLexicalCorpus,
        *,
        options: KnowledgeSemanticScoringOptions | None = None,
    ) -> KnowledgeSemanticScoringResult:
        """Record and return one validated semantic result."""

        self.calls.append((query, corpus, options))
        if self.error is not None:
            raise self.error
        if self.result is not None:
            return self.result

        documents = eligible_documents(query, corpus)
        if self.status == KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS:
            return KnowledgeSemanticScoringResult(
                repository=query.repository,
                query=query,
                status=self.status,
                eligible_document_count=0,
                scored_document_count=0,
                query_truncated=False,
                truncated_document_count=0,
                emitted_character_count=0,
                similarities=[],
            )
        if self.status == KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY:
            return KnowledgeSemanticScoringResult(
                repository=query.repository,
                query=query,
                status=self.status,
                eligible_document_count=len(documents),
                scored_document_count=0,
                query_truncated=False,
                truncated_document_count=0,
                emitted_character_count=0,
                similarities=[],
            )

        similarities = [
            KnowledgeSemanticSimilarity(
                reference=document.reference,
                score=self.scores.get(document.item_key, 70),
                provider="recording-provider",
                model="recording-model",
            )
            for document in documents
        ]
        return KnowledgeSemanticScoringResult(
            repository=query.repository,
            query=query,
            status=self.status,
            eligible_document_count=len(documents),
            scored_document_count=len(documents),
            provider="recording-provider",
            model="recording-model",
            query_truncated=False,
            truncated_document_count=0,
            emitted_character_count=100,
            similarities=similarities,
        )


def result_payload(result: KnowledgeRelatedWorkResult) -> dict[str, Any]:
    """Return only declared model fields for defensive validation tests."""

    return result.model_dump(exclude_computed_fields=True)


async def deterministic_result(
    *items: KnowledgeItem,
    query: KnowledgeLexicalQuery | None = None,
    max_results: int = 20,
) -> KnowledgeRelatedWorkResult:
    """Run the service without semantic scoring."""

    return await KnowledgeRelatedWorkService().find(
        query or make_query(),
        list(items),
        as_of=AS_OF,
        options=KnowledgeRelatedWorkOptions(max_results=max_results),
    )


async def hybrid_result(
    *items: KnowledgeItem,
    query: KnowledgeLexicalQuery | None = None,
    max_results: int = 20,
    scores: dict[str, int] | None = None,
) -> tuple[KnowledgeRelatedWorkResult, RecordingSemanticService]:
    """Run the service with complete semantic scoring."""

    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED,
        scores=scores,
    )
    result = await KnowledgeRelatedWorkService(
        semantic_scoring_service=semantic_service
    ).find(
        query or make_query(),
        list(items),
        as_of=AS_OF,
        options=KnowledgeRelatedWorkOptions(max_results=max_results),
    )
    return result, semantic_service


async def semantic_result_for(
    query: KnowledgeLexicalQuery,
    items: list[KnowledgeItem],
    status: KnowledgeSemanticScoringStatus,
) -> KnowledgeSemanticScoringResult:
    """Build one structurally valid semantic result for stale-result tests."""

    corpus = related_work_module.build_knowledge_lexical_corpus(
        query.repository,
        items,
    )
    return await RecordingSemanticService(status).score(query, corpus)


def fallback_items(count: int = 101) -> list[KnowledgeItem]:
    """Build enough positive lexical matches to cross the retrieval bound."""

    return [
        make_item(
            f"{index:03d}",
            title=f"Parser historical item {index:03d}",
            summary=f"Parser summary {index:03d}",
            url=f"https://example.test/items/{index:03d}",
        )
        for index in range(count)
    ]


def test_public_modes_defaults_and_option_bounds() -> None:
    assert [mode.value for mode in KnowledgeRelatedWorkMode] == [
        "lexical_fallback",
        "deterministic",
        "hybrid",
    ]
    options = KnowledgeRelatedWorkOptions()
    assert options.max_results == 20
    assert options.semantic_scoring == KnowledgeSemanticScoringOptions()
    assert KnowledgeRelatedWorkOptions(max_results=1).max_results == 1
    assert (
        KnowledgeRelatedWorkOptions(
            max_results=MAX_KNOWLEDGE_RELATED_WORK_RESULTS
        ).max_results
        == MAX_KNOWLEDGE_RELATED_WORK_RESULTS
    )
    for value in (0, MAX_KNOWLEDGE_RELATED_WORK_RESULTS + 1):
        with pytest.raises(ValidationError):
            KnowledgeRelatedWorkOptions(max_results=value)


def test_options_preserve_nested_semantic_options() -> None:
    semantic_options = KnowledgeSemanticScoringOptions(
        max_documents=12,
        max_query_characters=500,
        max_document_characters=600,
        max_total_characters=5_000,
    )
    options = KnowledgeRelatedWorkOptions(
        max_results=4,
        semantic_scoring=semantic_options,
    )
    assert options.semantic_scoring == semantic_options


def test_item_summary_is_independently_strict_and_computes_key() -> None:
    item = make_item()
    payload = {
        "reference": item.to_reference(),
        "title": item.title,
        "summary": item.summary,
        "url": item.url,
        "state": item.state,
        "decision_significance": item.decision_significance,
        "updated_at": NON_UTC_AS_OF,
        "labels": ["Parser"],
        "affected_paths": [r".\src\parser.py"],
        "components": ["Runtime"],
    }
    summary = KnowledgeRelatedWorkItemSummary(**payload)
    assert summary.item_key == item.key
    assert summary.updated_at == NON_UTC_AS_OF.astimezone(UTC)
    assert summary.affected_paths == ["src/parser.py"]

    with pytest.raises(ValidationError, match="timezone-aware"):
        KnowledgeRelatedWorkItemSummary(
            **{**payload, "updated_at": datetime(2026, 1, 1)}
        )
    with pytest.raises(ValidationError, match="case-insensitively"):
        KnowledgeRelatedWorkItemSummary(
            **{**payload, "labels": ["Parser", "parser"]}
        )
    with pytest.raises(ValidationError, match="case-insensitively"):
        KnowledgeRelatedWorkItemSummary(
            **{**payload, "components": ["Runtime", "runtime"]}
        )
    with pytest.raises(ValidationError, match="segments"):
        KnowledgeRelatedWorkItemSummary(
            **{**payload, "affected_paths": ["../secret.txt"]}
        )


@pytest.mark.asyncio
async def test_match_models_validate_shapes_scores_and_counts() -> None:
    item = make_item()
    deterministic = await deterministic_result(item)
    deterministic_match = deterministic.matches[0]
    assert deterministic_match.mode == KnowledgeRelatedWorkMode.DETERMINISTIC
    assert deterministic_match.score == deterministic_match.hybrid_match.score
    assert deterministic_match.raw_score == (
        deterministic_match.hybrid_match.total_weighted_basis_points
    )
    assert deterministic_match.lexical_evidence_count > 0
    assert deterministic_match.contribution_count == 5

    hybrid, _ = await hybrid_result(item)
    hybrid_match = hybrid.matches[0]
    assert hybrid_match.score == hybrid_match.hybrid_match.score
    assert hybrid_match.raw_score == (
        hybrid_match.hybrid_match.total_weighted_basis_points
    )
    assert hybrid_match.contribution_count == 6

    fallback = await KnowledgeRelatedWorkService().find(
        make_query(),
        fallback_items(),
        as_of=AS_OF,
        options=KnowledgeRelatedWorkOptions(max_results=1),
    )
    fallback_match = fallback.matches[0]
    assert fallback_match.score == fallback_match.lexical_match.score
    assert fallback_match.raw_score == fallback_match.lexical_match.raw_score
    assert fallback_match.contribution_count == 0


@pytest.mark.asyncio
async def test_match_rejects_invalid_mode_shapes_and_source_identity() -> None:
    first = make_item("001")
    second = make_item("002", url="https://example.test/items/002")
    deterministic = await deterministic_result(first)
    match = deterministic.matches[0]
    payload = match.model_dump(exclude_computed_fields=True)

    with pytest.raises(ValidationError, match="requires a lexical match"):
        KnowledgeRelatedWorkMatch.model_validate(
            {
                **payload,
                "mode": KnowledgeRelatedWorkMode.LEXICAL_FALLBACK,
                "lexical_match": None,
                "hybrid_match": None,
            }
        )
    with pytest.raises(ValidationError, match="requires a hybrid match"):
        KnowledgeRelatedWorkMatch.model_validate(
            {**payload, "hybrid_match": None}
        )
    with pytest.raises(ValidationError, match="summarized item"):
        KnowledgeRelatedWorkMatch.model_validate(
            {
                **payload,
                "item": {
                    **payload["item"],
                    "reference": second.to_reference(),
                },
            }
        )


@pytest.mark.asyncio
async def test_semantic_only_hybrid_match_is_accepted() -> None:
    item = make_item(title="Unrelated historical title")
    query = make_query(
        text="semantic-only query",
        affected_paths=[],
        labels=[],
    )
    result, _ = await hybrid_result(item, query=query)
    match = result.matches[0]
    assert match.mode == KnowledgeRelatedWorkMode.HYBRID
    assert match.lexical_match is None
    assert match.hybrid_match.semantic_similarity is not None
    assert match.lexical_evidence_count == 0


@pytest.mark.asyncio
async def test_result_rejects_extra_fields() -> None:
    result = await deterministic_result(make_item())
    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeRelatedWorkResult.model_validate(
            {**result_payload(result), "unexpected": True}
        )


@pytest.mark.asyncio
async def test_input_validation_order_normalization_and_immutability() -> None:
    query = make_query()
    item = make_item()
    items = [item]
    query_before = query.model_dump()
    item_before = item.model_dump()
    list_before = list(items)

    with pytest.raises(ValueError, match="timezone-aware"):
        await KnowledgeRelatedWorkService().find(
            query,
            [make_item(repository=OTHER_REPOSITORY)],
            as_of=datetime(2026, 1, 1),
        )

    result = await KnowledgeRelatedWorkService().find(
        query,
        items,
        as_of=NON_UTC_AS_OF,
    )
    assert result.as_of == NON_UTC_AS_OF.astimezone(UTC)
    assert query.model_dump() == query_before
    assert item.model_dump() == item_before
    assert items == list_before


@pytest.mark.asyncio
async def test_cross_repository_and_duplicate_items_are_rejected() -> None:
    with pytest.raises(ValueError, match="query repository"):
        await KnowledgeRelatedWorkService().find(
            make_query(),
            [make_item(repository=OTHER_REPOSITORY)],
            as_of=AS_OF,
        )

    item = make_item()
    with pytest.raises(ValueError, match="keys must be unique"):
        await KnowledgeRelatedWorkService().find(
            make_query(),
            [item, item.model_copy(deep=True)],
            as_of=AS_OF,
        )


@pytest.mark.asyncio
async def test_summary_is_concise_and_does_not_share_nested_lists() -> None:
    item = make_item(
        labels=["Parser"],
        affected_paths=["src/parser.py"],
        components=["Runtime"],
    )
    result = await deterministic_result(item)
    summary = result.matches[0].item
    dumped = summary.model_dump()

    assert summary.title == item.title
    assert summary.summary == item.summary
    assert summary.url == item.url
    assert summary.state == item.state
    assert summary.decision_significance == item.decision_significance
    assert summary.updated_at == item.updated_at
    assert summary.labels == item.labels
    assert summary.affected_paths == item.affected_paths
    assert summary.components == item.components
    assert {"body", "author", "created_at", "closed_at"}.isdisjoint(dumped)

    summary.labels.append("output-only")
    summary.affected_paths.append("src/output.py")
    summary.components.append("Output")
    assert item.labels == ["Parser"]
    assert item.affected_paths == ["src/parser.py"]
    assert item.components == ["Runtime"]


@pytest.mark.asyncio
async def test_corpus_and_lexical_counts_are_carried_to_result() -> None:
    matching = make_item("001")
    unrelated = make_item(
        "002",
        title="Unrelated historical work",
        summary="No matching content.",
        labels=["other"],
        affected_paths=["src/other.py"],
        components=["Other"],
    )
    result = await deterministic_result(matching, unrelated)
    assert result.corpus_total_count == 2
    assert result.eligible_document_count == 2
    assert result.lexical_matched_document_count == 1


@pytest.mark.asyncio
async def test_service_uses_fixed_complete_lexical_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[KnowledgeLexicalSearchOptions] = []
    real_search = search_knowledge_lexical_corpus

    def recording_search(
        query: KnowledgeLexicalQuery,
        corpus: KnowledgeLexicalCorpus,
        *,
        options: KnowledgeLexicalSearchOptions | None = None,
    ) -> Any:
        assert options is not None
        captured.append(options)
        return real_search(query, corpus, options=options)

    monkeypatch.setattr(
        related_work_module,
        "search_knowledge_lexical_corpus",
        recording_search,
    )
    items = [make_item(f"{index:03d}") for index in range(30)]
    result = await KnowledgeRelatedWorkService().find(
        make_query(),
        items,
        as_of=AS_OF,
        options=KnowledgeRelatedWorkOptions(max_results=2),
    )

    assert captured == [
        KnowledgeLexicalSearchOptions(
            max_results=MAX_KNOWLEDGE_LEXICAL_RESULTS,
            minimum_score=1,
        )
    ]
    assert result.lexical_matched_document_count == 30
    assert result.returned_count == 2


@pytest.mark.asyncio
async def test_lexical_fallback_is_explicit_bounded_and_skips_other_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED
    )

    def forbidden_hybrid_ranking(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Hybrid ranking must not run during lexical fallback.")

    monkeypatch.setattr(
        related_work_module,
        "rank_knowledge_hybrid_corpus",
        forbidden_hybrid_ranking,
    )
    result = await KnowledgeRelatedWorkService(
        semantic_scoring_service=semantic_service
    ).find(
        make_query(),
        fallback_items(),
        as_of=AS_OF,
        options=KnowledgeRelatedWorkOptions(max_results=3),
    )

    assert result.mode == KnowledgeRelatedWorkMode.LEXICAL_FALLBACK
    assert semantic_service.calls == []
    assert result.warnings == [FALLBACK_WARNING]
    assert result.complete_ranking_coverage is False
    assert result.corpus_total_count == 101
    assert result.eligible_document_count == 101
    assert result.lexical_matched_document_count == 101
    assert result.candidate_count == 101
    assert result.returned_count == 3
    assert result.truncated is True
    assert result.semantic_status is None
    assert result.semantic_provider is None
    assert result.semantic_model is None
    assert result.semantic_scored_document_count == 0
    assert all(match.lexical_match is not None for match in result.matches)
    assert all(match.hybrid_match is None for match in result.matches)
    assert [match.raw_score for match in result.matches] == sorted(
        [match.raw_score for match in result.matches],
        reverse=True,
    )


@pytest.mark.asyncio
async def test_lexical_fallback_ties_use_item_key_order() -> None:
    result = await KnowledgeRelatedWorkService().find(
        make_query(),
        list(reversed(fallback_items())),
        as_of=AS_OF,
        options=KnowledgeRelatedWorkOptions(max_results=100),
    )
    equal_score_groups: dict[int, list[str]] = {}
    for match in result.matches:
        equal_score_groups.setdefault(match.raw_score, []).append(match.item_key)
    assert all(keys == sorted(keys) for keys in equal_score_groups.values())


@pytest.mark.asyncio
async def test_deterministic_path_has_complete_explainable_ranking() -> None:
    items = [
        make_item("001"),
        make_item(
            "002",
            title="Parser secondary work",
            labels=["other"],
            affected_paths=["src/other.py"],
        ),
    ]
    result = await deterministic_result(*items, max_results=1)
    assert result.mode == KnowledgeRelatedWorkMode.DETERMINISTIC
    assert result.semantic_status is None
    assert result.semantic_provider is None
    assert result.semantic_model is None
    assert result.semantic_scored_document_count == 0
    assert result.warnings == []
    assert result.complete_ranking_coverage is True
    assert result.returned_count == 1
    assert result.candidate_count == 2
    assert result.truncated is True
    assert result.matches[0].hybrid_match.mode == (
        KnowledgeHybridRankingMode.DETERMINISTIC
    )
    assert result.matches[0].hybrid_match.semantic_similarity is None
    assert result.matches[0].contribution_count == 5
    signals = [
        contribution.signal
        for contribution in result.matches[0].hybrid_match.contributions
    ]
    assert len(signals) == len(set(signals))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY,
        KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS,
    ],
)
async def test_skipped_semantic_results_use_deterministic_ranking(
    status: KnowledgeSemanticScoringStatus,
) -> None:
    items = [] if status == KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS else [
        make_item()
    ]
    query = (
        make_query()
        if not items
        else make_query(text=None, affected_paths=[], labels=["parser"])
    )
    semantic_service = RecordingSemanticService(status)
    options = KnowledgeRelatedWorkOptions(
        semantic_scoring=KnowledgeSemanticScoringOptions(max_documents=10)
    )
    result = await KnowledgeRelatedWorkService(
        semantic_scoring_service=semantic_service
    ).find(
        query,
        items,
        as_of=AS_OF,
        options=options,
    )

    assert len(semantic_service.calls) == 1
    assert semantic_service.calls[0][0] == query
    assert semantic_service.calls[0][2] == options.semantic_scoring
    assert result.mode == KnowledgeRelatedWorkMode.DETERMINISTIC
    assert result.semantic_status == status
    assert result.semantic_provider is None
    assert result.semantic_model is None
    assert result.semantic_scored_document_count == 0
    assert result.warnings == []
    assert result.semantic_performed is False


@pytest.mark.asyncio
async def test_scored_semantics_produce_hybrid_provenance_and_order() -> None:
    first = make_item("001", title="Parser first")
    second = make_item("002", title="Parser second")
    scores = {first.key: 20, second.key: 90}
    result, semantic_service = await hybrid_result(
        first,
        second,
        max_results=1,
        scores=scores,
    )

    assert len(semantic_service.calls) == 1
    assert result.mode == KnowledgeRelatedWorkMode.HYBRID
    assert result.semantic_status == KnowledgeSemanticScoringStatus.SCORED
    assert result.semantic_provider == "recording-provider"
    assert result.semantic_model == "recording-model"
    assert result.semantic_scored_document_count == 2
    assert result.semantic_performed is True
    assert result.complete_ranking_coverage is True
    assert result.candidate_count == 2
    assert result.returned_count == 1
    assert result.truncated is True
    assert result.matches[0].item_key == second.key
    assert result.matches[0].hybrid_match.semantic_similarity.score == 90
    assert result.matches[0].contribution_count == 6


@pytest.mark.asyncio
async def test_exact_semantic_result_provenance_remains_accepted() -> None:
    item = make_item()
    query = make_query()
    semantic_result = await semantic_result_for(
        query,
        [item],
        KnowledgeSemanticScoringStatus.SCORED,
    )
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED,
        result=semantic_result,
    )

    result = await KnowledgeRelatedWorkService(
        semantic_scoring_service=semantic_service
    ).find(query, [item], as_of=AS_OF)

    assert result.mode == KnowledgeRelatedWorkMode.HYBRID
    assert result.semantic_provider == semantic_result.provider
    assert result.semantic_model == semantic_result.model
    assert len(semantic_service.calls) == 1


@pytest.mark.asyncio
async def test_semantic_result_repository_is_validated_first() -> None:
    query = make_query()
    items = [make_item("001"), make_item("002")]
    other_query = KnowledgeLexicalQuery(
        repository=OTHER_REPOSITORY,
        text="different query",
    )
    stale_result = await semantic_result_for(
        other_query,
        [make_item(repository=OTHER_REPOSITORY)],
        KnowledgeSemanticScoringStatus.SCORED,
    )
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED,
        result=stale_result,
    )

    with pytest.raises(ValueError) as raised:
        await KnowledgeRelatedWorkService(
            semantic_scoring_service=semantic_service
        ).find(query, items, as_of=AS_OF)

    assert str(raised.value) == (
        "Semantic result repository must match the related-work repository."
    )
    assert len(semantic_service.calls) == 1


@pytest.mark.asyncio
async def test_semantic_result_query_is_validated_before_eligible_count() -> None:
    query = make_query()
    items = [make_item("001"), make_item("002")]
    stale_query = make_query(text="different query")
    stale_result = await semantic_result_for(
        stale_query,
        [items[0]],
        KnowledgeSemanticScoringStatus.SCORED,
    )
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED,
        result=stale_result,
    )

    with pytest.raises(ValueError) as raised:
        await KnowledgeRelatedWorkService(
            semantic_scoring_service=semantic_service
        ).find(query, items, as_of=AS_OF)

    assert str(raised.value) == (
        "Semantic result query must match the related-work query."
    )


@pytest.mark.asyncio
async def test_stale_scored_eligibility_is_rejected_before_hybrid_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = make_query()
    items = [make_item("001"), make_item("002")]
    stale_result = await semantic_result_for(
        query,
        [items[0]],
        KnowledgeSemanticScoringStatus.SCORED,
    )
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED,
        result=stale_result,
    )

    def forbidden_hybrid_ranking(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Stale semantic similarities must not be ranked.")

    monkeypatch.setattr(
        related_work_module,
        "rank_knowledge_hybrid_corpus",
        forbidden_hybrid_ranking,
    )
    with pytest.raises(ValueError) as raised:
        await KnowledgeRelatedWorkService(
            semantic_scoring_service=semantic_service
        ).find(query, items, as_of=AS_OF)

    assert str(raised.value) == (
        "Semantic result eligible count must match related-work eligibility."
    )
    assert len(semantic_service.calls) == 1


@pytest.mark.asyncio
async def test_stale_no_semantic_query_result_is_rejected() -> None:
    query = make_query()
    items = [make_item("001"), make_item("002")]
    stale_result = KnowledgeSemanticScoringResult(
        repository=query.repository,
        query=query,
        status=KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY,
        eligible_document_count=1,
        scored_document_count=0,
        query_truncated=False,
        truncated_document_count=0,
        emitted_character_count=0,
        similarities=[],
    )
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY,
        result=stale_result,
    )

    with pytest.raises(ValueError) as raised:
        await KnowledgeRelatedWorkService(
            semantic_scoring_service=semantic_service
        ).find(query, items, as_of=AS_OF)

    assert str(raised.value) == (
        "Semantic result eligible count must match related-work eligibility."
    )


@pytest.mark.asyncio
async def test_stale_no_eligible_documents_result_is_rejected() -> None:
    query = make_query()
    item = make_item()
    stale_result = KnowledgeSemanticScoringResult(
        repository=query.repository,
        query=query,
        status=KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS,
        eligible_document_count=0,
        scored_document_count=0,
        query_truncated=False,
        truncated_document_count=0,
        emitted_character_count=0,
        similarities=[],
    )
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS,
        result=stale_result,
    )

    with pytest.raises(ValueError) as raised:
        await KnowledgeRelatedWorkService(
            semantic_scoring_service=semantic_service
        ).find(query, [item], as_of=AS_OF)

    assert str(raised.value) == (
        "Semantic result eligible count must match related-work eligibility."
    )


@pytest.mark.asyncio
async def test_semantic_similarities_are_passed_unchanged_to_hybrid_ranking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    item = make_item()
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED
    )
    real_rank = rank_knowledge_hybrid_corpus
    captured: list[list[KnowledgeSemanticSimilarity] | None] = []

    def recording_rank(
        query: KnowledgeLexicalQuery,
        corpus: KnowledgeLexicalCorpus,
        lexical_result: Any,
        semantic_scores: list[KnowledgeSemanticSimilarity] | None = None,
        *,
        as_of: datetime,
        options: Any = None,
    ) -> Any:
        captured.append(semantic_scores)
        return real_rank(
            query,
            corpus,
            lexical_result,
            semantic_scores,
            as_of=as_of,
            options=options,
        )

    monkeypatch.setattr(
        related_work_module,
        "rank_knowledge_hybrid_corpus",
        recording_rank,
    )
    result = await KnowledgeRelatedWorkService(
        semantic_scoring_service=semantic_service
    ).find(make_query(), [item], as_of=AS_OF)

    assert captured[0] is not None
    assert captured[0] == [
        result.matches[0].hybrid_match.semantic_similarity
    ]


@pytest.mark.asyncio
async def test_semantic_only_candidate_can_surface_without_lexical_matches() -> None:
    item = make_item(title="Completely unrelated history")
    query = make_query(
        text="novel semantic request",
        affected_paths=[],
        labels=[],
    )
    result, _ = await hybrid_result(item, query=query)
    assert result.lexical_matched_document_count == 0
    assert result.candidate_count == 1
    assert result.returned_count == 1
    assert result.matches[0].lexical_match is None


@pytest.mark.asyncio
async def test_semantic_errors_propagate_without_retry_or_fallback() -> None:
    error = SentinelSemanticError("provider unavailable")
    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED,
        error=error,
    )
    with pytest.raises(SentinelSemanticError) as raised:
        await KnowledgeRelatedWorkService(
            semantic_scoring_service=semantic_service
        ).find(make_query(), [make_item()], as_of=AS_OF)
    assert raised.value is error
    assert len(semantic_service.calls) == 1

    try:
        KnowledgeSemanticScoringOptions(max_documents=0)
    except ValidationError as validation_error:
        captured_validation_error = validation_error
    validation_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.SCORED,
        error=captured_validation_error,
    )
    with pytest.raises(ValidationError) as validation_raised:
        await KnowledgeRelatedWorkService(
            semantic_scoring_service=validation_service
        ).find(make_query(), [make_item()], as_of=AS_OF)
    assert validation_raised.value is captured_validation_error
    assert len(validation_service.calls) == 1


@pytest.mark.asyncio
async def test_result_validates_repository_counts_keys_ranks_and_modes() -> None:
    result = await deterministic_result(make_item("001"), make_item("002"))
    base = result_payload(result)

    with pytest.raises(ValidationError, match="query"):
        KnowledgeRelatedWorkResult.model_validate(
            {
                **base,
                "repository": OTHER_REPOSITORY,
            }
        )
    for updates in (
        {"corpus_total_count": 0},
        {"eligible_document_count": 0},
        {"candidate_count": 0},
    ):
        with pytest.raises(ValidationError):
            KnowledgeRelatedWorkResult.model_validate({**base, **updates})

    duplicate_matches = [deepcopy(base["matches"][0]), deepcopy(base["matches"][0])]
    duplicate_matches[1]["rank"] = 2
    with pytest.raises(ValidationError, match="keys must be unique"):
        KnowledgeRelatedWorkResult.model_validate(
            {**base, "matches": duplicate_matches}
        )

    wrong_ranks = deepcopy(base["matches"])
    wrong_ranks[0]["rank"] = 2
    with pytest.raises(ValidationError, match="sequential"):
        KnowledgeRelatedWorkResult.model_validate(
            {**base, "matches": wrong_ranks}
        )

    mixed_modes = deepcopy(base["matches"])
    mixed_modes[0]["mode"] = KnowledgeRelatedWorkMode.HYBRID
    with pytest.raises(ValidationError):
        KnowledgeRelatedWorkResult.model_validate(
            {**base, "matches": mixed_modes}
        )


@pytest.mark.asyncio
async def test_result_validates_all_ranking_orders() -> None:
    first = make_item("001", significance=DecisionSignificance.CRITICAL)
    second = make_item("002", significance=DecisionSignificance.NONE)
    deterministic = await deterministic_result(first, second)
    deterministic_payload = result_payload(deterministic)
    reversed_deterministic = list(reversed(deterministic_payload["matches"]))
    for rank, match in enumerate(reversed_deterministic, start=1):
        match["rank"] = rank
    with pytest.raises(ValidationError, match="ranking order"):
        KnowledgeRelatedWorkResult.model_validate(
            {**deterministic_payload, "matches": reversed_deterministic}
        )

    hybrid, _ = await hybrid_result(
        first,
        second,
        scores={first.key: 90, second.key: 10},
    )
    hybrid_payload = result_payload(hybrid)
    reversed_hybrid = list(reversed(hybrid_payload["matches"]))
    for rank, match in enumerate(reversed_hybrid, start=1):
        match["rank"] = rank
    with pytest.raises(ValidationError, match="ranking order"):
        KnowledgeRelatedWorkResult.model_validate(
            {**hybrid_payload, "matches": reversed_hybrid}
        )

    fallback = await KnowledgeRelatedWorkService().find(
        make_query(),
        fallback_items(),
        as_of=AS_OF,
        options=KnowledgeRelatedWorkOptions(max_results=100),
    )
    fallback_payload = result_payload(fallback)
    reversed_fallback = list(reversed(fallback_payload["matches"]))
    for rank, match in enumerate(reversed_fallback, start=1):
        match["rank"] = rank
    with pytest.raises(ValidationError, match="ranking order"):
        KnowledgeRelatedWorkResult.model_validate(
            {**fallback_payload, "matches": reversed_fallback}
        )


@pytest.mark.asyncio
async def test_result_validates_mode_specific_provenance_and_warnings() -> None:
    deterministic = await deterministic_result(make_item())
    deterministic_payload = result_payload(deterministic)
    with pytest.raises(ValidationError, match="warnings"):
        KnowledgeRelatedWorkResult.model_validate(
            {**deterministic_payload, "warnings": ["unexpected"]}
        )
    with pytest.raises(ValidationError, match="skipped semantic"):
        KnowledgeRelatedWorkResult.model_validate(
            {
                **deterministic_payload,
                "semantic_status": KnowledgeSemanticScoringStatus.SCORED,
            }
        )
    with pytest.raises(ValidationError) as no_eligible:
        KnowledgeRelatedWorkResult.model_validate(
            {
                **deterministic_payload,
                "semantic_status": (
                    KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS
                ),
            }
        )
    assert (
        "NO_ELIGIBLE_DOCUMENTS requires zero eligible related-work documents."
        in str(no_eligible.value)
    )

    empty = await deterministic_result(query=make_query())
    with pytest.raises(ValidationError) as no_semantic_query:
        KnowledgeRelatedWorkResult.model_validate(
            {
                **result_payload(empty),
                "semantic_status": (
                    KnowledgeSemanticScoringStatus.NO_SEMANTIC_QUERY
                ),
            }
        )
    assert (
        "NO_SEMANTIC_QUERY requires eligible related-work documents."
        in str(no_semantic_query.value)
    )

    hybrid, _ = await hybrid_result(make_item())
    hybrid_payload = result_payload(hybrid)
    with pytest.raises(ValidationError, match="provider and model"):
        KnowledgeRelatedWorkResult.model_validate(
            {**hybrid_payload, "semantic_provider": None}
        )
    with pytest.raises(ValidationError, match="warnings"):
        KnowledgeRelatedWorkResult.model_validate(
            {**hybrid_payload, "warnings": ["unexpected"]}
        )
    with pytest.raises(ValidationError) as provider_mismatch:
        KnowledgeRelatedWorkResult.model_validate(
            {
                **hybrid_payload,
                "semantic_provider": "different-provider",
            }
        )
    assert (
        "Hybrid match semantic provider must match result provenance."
        in str(provider_mismatch.value)
    )
    with pytest.raises(ValidationError) as model_mismatch:
        KnowledgeRelatedWorkResult.model_validate(
            {
                **hybrid_payload,
                "semantic_model": "different-model",
            }
        )
    assert (
        "Hybrid match semantic model must match result provenance."
        in str(model_mismatch.value)
    )
    assert KnowledgeRelatedWorkResult.model_validate(hybrid_payload) == hybrid

    fallback = await KnowledgeRelatedWorkService().find(
        make_query(),
        fallback_items(),
        as_of=AS_OF,
        options=KnowledgeRelatedWorkOptions(max_results=1),
    )
    fallback_payload = result_payload(fallback)
    with pytest.raises(ValidationError, match="exact coverage warning"):
        KnowledgeRelatedWorkResult.model_validate(
            {**fallback_payload, "warnings": []}
        )
    with pytest.raises(ValidationError, match="semantic provenance"):
        KnowledgeRelatedWorkResult.model_validate(
            {
                **fallback_payload,
                "semantic_provider": "unexpected",
            }
        )


@pytest.mark.asyncio
async def test_empty_and_no_match_behavior() -> None:
    query = make_query()
    empty = await deterministic_result(query=query)
    assert empty.mode == KnowledgeRelatedWorkMode.DETERMINISTIC
    assert empty.corpus_total_count == 0
    assert empty.candidate_count == 0
    assert empty.matches == []
    assert empty.model_dump(mode="json")["matches"] == []

    semantic_service = RecordingSemanticService(
        KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS
    )
    semantic_empty = await KnowledgeRelatedWorkService(
        semantic_scoring_service=semantic_service
    ).find(query, [], as_of=AS_OF)
    assert semantic_empty.semantic_status == (
        KnowledgeSemanticScoringStatus.NO_ELIGIBLE_DOCUMENTS
    )
    assert semantic_empty.mode == KnowledgeRelatedWorkMode.DETERMINISTIC
    assert semantic_empty.matches == []

    no_match = await deterministic_result(
        make_item(title="Unrelated history"),
        query=make_query(text="nothing-matches", affected_paths=[], labels=[]),
    )
    assert no_match.eligible_document_count == 1
    assert no_match.lexical_matched_document_count == 0
    assert no_match.matches == []


@pytest.mark.asyncio
async def test_json_serialization_contains_concise_complete_provenance() -> None:
    item = make_item()
    result, _ = await hybrid_result(item)
    payload = result.model_dump(mode="json")

    assert payload["query"]["text"] == "parser"
    assert payload["options"]["max_results"] == 20
    assert payload["mode"] == "hybrid"
    assert payload["as_of"] == "2026-01-01T10:00:00Z"
    assert payload["semantic_status"] == "scored"
    assert payload["semantic_provider"] == "recording-provider"
    assert payload["returned_count"] == 1
    assert payload["semantic_performed"] is True
    assert payload["complete_ranking_coverage"] is True

    match = payload["matches"][0]
    assert match["item"]["title"] == item.title
    assert "body" not in match["item"]
    assert match["lexical_match"]["evidence"]
    assert len(match["hybrid_match"]["contributions"]) == 6
    assert match["hybrid_match"]["semantic_similarity"]["provider"] == (
        "recording-provider"
    )
    assert match["score"] == result.matches[0].score
    assert match["raw_score"] == result.matches[0].raw_score
    assert match["lexical_evidence_count"] > 0
    assert match["contribution_count"] == 6
