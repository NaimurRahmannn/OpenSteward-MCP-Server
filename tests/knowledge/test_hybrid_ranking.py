"""Tests for provider-independent hybrid related-work ranking."""

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from opensteward.knowledge import (
    KNOWLEDGE_DETERMINISTIC_LABEL_WEIGHT,
    KNOWLEDGE_DETERMINISTIC_LEXICAL_WEIGHT,
    KNOWLEDGE_DETERMINISTIC_PATH_WEIGHT,
    KNOWLEDGE_DETERMINISTIC_RECENCY_WEIGHT,
    KNOWLEDGE_DETERMINISTIC_SIGNIFICANCE_WEIGHT,
    KNOWLEDGE_HYBRID_LABEL_WEIGHT,
    KNOWLEDGE_HYBRID_LEXICAL_WEIGHT,
    KNOWLEDGE_HYBRID_PATH_WEIGHT,
    KNOWLEDGE_HYBRID_RECENCY_WEIGHT,
    KNOWLEDGE_HYBRID_SEMANTIC_WEIGHT,
    KNOWLEDGE_HYBRID_SIGNIFICANCE_WEIGHT,
    MAX_KNOWLEDGE_HYBRID_RESULTS,
    MAX_KNOWLEDGE_HYBRID_SCORE,
    DecisionSignificance,
    KnowledgeHybridRankedMatch,
    KnowledgeHybridRankingMode,
    KnowledgeHybridRankingOptions,
    KnowledgeHybridRankingResult,
    KnowledgeHybridSignal,
    KnowledgeHybridSignalContribution,
    KnowledgeItem,
    KnowledgeItemState,
    KnowledgeItemType,
    KnowledgeLexicalCorpus,
    KnowledgeLexicalField,
    KnowledgeLexicalMatch,
    KnowledgeLexicalMatchEvidence,
    KnowledgeLexicalMatchKind,
    KnowledgeLexicalQuery,
    KnowledgeLexicalReference,
    KnowledgeLexicalSearchOptions,
    KnowledgeLexicalSearchResult,
    KnowledgeRepositoryRef,
    KnowledgeSemanticSimilarity,
    KnowledgeSourceKind,
    build_knowledge_lexical_corpus,
    rank_knowledge_hybrid_corpus,
    search_knowledge_lexical_corpus,
)

CREATED_AT = datetime(2024, 1, 1, 9, 0, tzinfo=UTC)
AS_OF = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)
RECENT_AT = AS_OF - timedelta(days=10)
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

DETERMINISTIC_WEIGHTS = {
    KnowledgeHybridSignal.LEXICAL: KNOWLEDGE_DETERMINISTIC_LEXICAL_WEIGHT,
    KnowledgeHybridSignal.AFFECTED_PATH: KNOWLEDGE_DETERMINISTIC_PATH_WEIGHT,
    KnowledgeHybridSignal.LABEL: KNOWLEDGE_DETERMINISTIC_LABEL_WEIGHT,
    KnowledgeHybridSignal.DECISION_SIGNIFICANCE: (
        KNOWLEDGE_DETERMINISTIC_SIGNIFICANCE_WEIGHT
    ),
    KnowledgeHybridSignal.RECENCY: KNOWLEDGE_DETERMINISTIC_RECENCY_WEIGHT,
}
HYBRID_WEIGHTS = {
    KnowledgeHybridSignal.LEXICAL: KNOWLEDGE_HYBRID_LEXICAL_WEIGHT,
    KnowledgeHybridSignal.SEMANTIC: KNOWLEDGE_HYBRID_SEMANTIC_WEIGHT,
    KnowledgeHybridSignal.AFFECTED_PATH: KNOWLEDGE_HYBRID_PATH_WEIGHT,
    KnowledgeHybridSignal.LABEL: KNOWLEDGE_HYBRID_LABEL_WEIGHT,
    KnowledgeHybridSignal.DECISION_SIGNIFICANCE: (
        KNOWLEDGE_HYBRID_SIGNIFICANCE_WEIGHT
    ),
    KnowledgeHybridSignal.RECENCY: KNOWLEDGE_HYBRID_RECENCY_WEIGHT,
}

EXPLANATIONS = {
    KnowledgeHybridSignal.LEXICAL: (
        "Core lexical relevance excludes path and label evidence to avoid "
        "double-counting."
    ),
    KnowledgeHybridSignal.SEMANTIC: (
        "Semantic similarity was supplied by the configured semantic scorer."
    ),
    KnowledgeHybridSignal.AFFECTED_PATH: (
        "Changed-path score reflects exact and shared-directory query-path coverage."
    ),
    KnowledgeHybridSignal.LABEL: (
        "Label score reflects the proportion of query labels matched by the "
        "historical item."
    ),
    KnowledgeHybridSignal.DECISION_SIGNIFICANCE: (
        "Decision significance boosts historically important project decisions."
    ),
    KnowledgeHybridSignal.RECENCY: (
        "Recency score favors recently updated historical work."
    ),
}


def make_item(
    *,
    external_id: str = "1",
    repository: KnowledgeRepositoryRef = REPOSITORY,
    item_type: KnowledgeItemType = KnowledgeItemType.ISSUE,
    state: KnowledgeItemState = KnowledgeItemState.CLOSED,
    title: str = "anchor",
    summary: str | None = None,
    body: str | None = None,
    labels: list[str] | None = None,
    components: list[str] | None = None,
    affected_paths: list[str] | None = None,
    updated_at: datetime = RECENT_AT,
    significance: DecisionSignificance = DecisionSignificance.NONE,
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
        labels=[] if labels is None else labels,
        components=[] if components is None else components,
        affected_paths=[] if affected_paths is None else affected_paths,
        decision_significance=significance,
    )


def make_query(**updates: Any) -> KnowledgeLexicalQuery:
    """Build a lexical query for the default repository."""

    return KnowledgeLexicalQuery(repository=REPOSITORY, **updates)


def make_context(
    query: KnowledgeLexicalQuery,
    items: list[KnowledgeItem],
) -> tuple[KnowledgeLexicalCorpus, KnowledgeLexicalSearchResult]:
    """Build a corpus and its complete lexical result."""

    corpus = build_knowledge_lexical_corpus(REPOSITORY, items)
    lexical_result = search_knowledge_lexical_corpus(
        query,
        corpus,
        options=KnowledgeLexicalSearchOptions(max_results=100),
    )
    return corpus, lexical_result


def make_similarity(
    item: KnowledgeItem,
    score: int,
    *,
    provider: str = "test-provider",
    model: str = "test-model",
) -> KnowledgeSemanticSimilarity:
    """Build one validated semantic similarity."""

    return KnowledgeSemanticSimilarity(
        reference=item.to_reference(),
        score=score,
        provider=provider,
        model=model,
    )


def make_lexical_evidence(
    *,
    kind: KnowledgeLexicalMatchKind = KnowledgeLexicalMatchKind.TERM,
    field: KnowledgeLexicalField = KnowledgeLexicalField.TITLE,
    query_value: str = "anchor",
    document_value: str = "anchor",
    points: int = 5,
) -> KnowledgeLexicalMatchEvidence:
    """Build one valid lexical evidence record."""

    explanations = {
        KnowledgeLexicalMatchKind.REFERENCE: (
            "Explicit typed reference matched this historical item."
        ),
        KnowledgeLexicalMatchKind.EXACT_PHRASE: (
            "Exact phrase matched the historical item title."
        ),
        KnowledgeLexicalMatchKind.IDENTIFIER: (
            "Identifier matched the historical item title."
        ),
        KnowledgeLexicalMatchKind.COMPONENT: (
            "Component matched historical component evidence."
        ),
        KnowledgeLexicalMatchKind.LABEL: (
            "Label matched historical label evidence."
        ),
        KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT: (
            "Repository path exactly matched historical changed-path evidence."
        ),
        KnowledgeLexicalMatchKind.AFFECTED_PATH_SHARED_DIRECTORY: (
            "Repository paths share a specific directory hierarchy."
        ),
        KnowledgeLexicalMatchKind.TERM: (
            "Term matched the historical item title."
        ),
    }
    return KnowledgeLexicalMatchEvidence(
        kind=kind,
        field=field,
        query_value=query_value,
        document_value=document_value,
        points=points,
        explanation=explanations[kind],
    )


def make_lexical_match(
    item: KnowledgeItem,
    evidence: list[KnowledgeLexicalMatchEvidence] | None = None,
) -> KnowledgeLexicalMatch:
    """Build one valid lexical match."""

    return KnowledgeLexicalMatch(
        reference=item.to_reference(),
        state=item.state,
        decision_significance=item.decision_significance,
        updated_at=item.updated_at,
        evidence=evidence or [make_lexical_evidence()],
    )


def make_lexical_result(
    query: KnowledgeLexicalQuery,
    corpus: KnowledgeLexicalCorpus,
    matches: list[KnowledgeLexicalMatch],
) -> KnowledgeLexicalSearchResult:
    """Build one complete manually controlled lexical result."""

    ordered_matches = sorted(
        matches,
        key=lambda match: (-match.raw_score, match.item_key),
    )
    return KnowledgeLexicalSearchResult(
        repository=corpus.repository,
        query=query,
        options=KnowledgeLexicalSearchOptions(),
        corpus_total_count=corpus.total_count,
        eligible_document_count=corpus.total_count,
        matched_document_count=len(ordered_matches),
        matches=ordered_matches,
    )


def make_contributions(
    mode: KnowledgeHybridRankingMode,
    *,
    scores: dict[KnowledgeHybridSignal, int] | None = None,
) -> list[KnowledgeHybridSignalContribution]:
    """Build contributions in the required order for a ranking mode."""

    weights = (
        HYBRID_WEIGHTS
        if mode == KnowledgeHybridRankingMode.HYBRID
        else DETERMINISTIC_WEIGHTS
    )
    effective_scores = {
        KnowledgeHybridSignal.LEXICAL: 5,
        **(scores or {}),
    }
    return [
        KnowledgeHybridSignalContribution(
            signal=signal,
            signal_score=effective_scores.get(signal, 0),
            weight_percent=weight,
            weighted_basis_points=effective_scores.get(signal, 0) * weight,
            explanation=EXPLANATIONS[signal],
        )
        for signal, weight in weights.items()
    ]


def make_ranked_match(
    *,
    item: KnowledgeItem | None = None,
    mode: KnowledgeHybridRankingMode = KnowledgeHybridRankingMode.DETERMINISTIC,
    lexical_match: KnowledgeLexicalMatch | None = None,
    semantic_similarity: KnowledgeSemanticSimilarity | None = None,
    contributions: list[KnowledgeHybridSignalContribution] | None = None,
) -> KnowledgeHybridRankedMatch:
    """Build one valid ranked match."""

    source = item or make_item()
    effective_lexical_match = lexical_match or make_lexical_match(source)
    if mode == KnowledgeHybridRankingMode.HYBRID and semantic_similarity is None:
        semantic_similarity = make_similarity(source, 0)
    return KnowledgeHybridRankedMatch(
        reference=source.to_reference(),
        state=source.state,
        decision_significance=source.decision_significance,
        updated_at=source.updated_at,
        mode=mode,
        lexical_match=effective_lexical_match,
        semantic_similarity=semantic_similarity,
        contributions=contributions or make_contributions(mode),
    )


def contribution(
    match: KnowledgeHybridRankedMatch,
    signal: KnowledgeHybridSignal,
) -> KnowledgeHybridSignalContribution:
    """Return one named contribution from a ranked match."""

    return next(item for item in match.contributions if item.signal == signal)


def rank(
    query: KnowledgeLexicalQuery,
    items: list[KnowledgeItem],
    semantic_scores: list[KnowledgeSemanticSimilarity] | None = None,
    *,
    as_of: datetime = AS_OF,
    options: KnowledgeHybridRankingOptions | None = None,
) -> KnowledgeHybridRankingResult:
    """Build lexical inputs and hybrid-rank them."""

    corpus, lexical_result = make_context(query, items)
    return rank_knowledge_hybrid_corpus(
        query,
        corpus,
        lexical_result,
        semantic_scores,
        as_of=as_of,
        options=options,
    )


def test_weights_total_100_and_public_enums_are_exact() -> None:
    assert sum(DETERMINISTIC_WEIGHTS.values()) == 100
    assert sum(HYBRID_WEIGHTS.values()) == 100
    assert [mode.value for mode in KnowledgeHybridRankingMode] == [
        "deterministic",
        "hybrid",
    ]
    assert [signal.value for signal in KnowledgeHybridSignal] == [
        "lexical",
        "semantic",
        "affected_path",
        "label",
        "decision_significance",
        "recency",
    ]


def test_ranking_options_defaults_and_bounds() -> None:
    options = KnowledgeHybridRankingOptions()

    assert options.max_results == 20
    assert options.minimum_score == 1

    for value in (0, MAX_KNOWLEDGE_HYBRID_RESULTS + 1):
        with pytest.raises(ValidationError):
            KnowledgeHybridRankingOptions(max_results=value)
    for value in (0, MAX_KNOWLEDGE_HYBRID_SCORE + 1):
        with pytest.raises(ValidationError):
            KnowledgeHybridRankingOptions(minimum_score=value)


@pytest.mark.parametrize("score", [0, 100])
def test_semantic_similarity_accepts_bounds_and_computes_key(score: int) -> None:
    item = make_item()
    similarity = make_similarity(item, score)

    assert similarity.score == score
    assert similarity.item_key == item.key


@pytest.mark.parametrize("score", [-1, 101])
def test_semantic_similarity_rejects_out_of_range_scores(score: int) -> None:
    with pytest.raises(ValidationError):
        make_similarity(make_item(), score)


@pytest.mark.parametrize("field_name", ["provider", "model"])
def test_semantic_similarity_rejects_empty_provenance(field_name: str) -> None:
    payload = make_similarity(make_item(), 50).model_dump()
    payload[field_name] = " "

    with pytest.raises(ValidationError):
        KnowledgeSemanticSimilarity.model_validate(payload)


def test_contribution_validates_basis_points_and_weighted_score() -> None:
    contribution_model = KnowledgeHybridSignalContribution(
        signal=KnowledgeHybridSignal.LEXICAL,
        signal_score=75,
        weight_percent=40,
        weighted_basis_points=3_000,
        explanation="Core lexical relevance.",
    )

    assert contribution_model.weighted_score == 30.0

    payload = contribution_model.model_dump(exclude={"weighted_score"})
    payload["weighted_basis_points"] = 2_999
    with pytest.raises(ValidationError, match="must equal"):
        KnowledgeHybridSignalContribution.model_validate(payload)


def test_ranked_match_rejects_duplicate_signals_and_naive_time() -> None:
    duplicate = make_contributions(KnowledgeHybridRankingMode.DETERMINISTIC)
    duplicate[-1] = duplicate[0]

    with pytest.raises(ValidationError, match="unique"):
        make_ranked_match(contributions=duplicate)

    item = make_item()
    with pytest.raises(ValidationError, match="timezone-aware"):
        KnowledgeHybridRankedMatch(
            reference=item.to_reference(),
            state=item.state,
            decision_significance=item.decision_significance,
            updated_at=datetime(2026, 7, 1, 12, 0),
            mode=KnowledgeHybridRankingMode.DETERMINISTIC,
            lexical_match=make_lexical_match(item),
            semantic_similarity=None,
            contributions=make_contributions(
                KnowledgeHybridRankingMode.DETERMINISTIC
            ),
        )


def test_ranked_match_normalizes_time_and_rejects_source_mismatches() -> None:
    non_utc = datetime(
        2026,
        7,
        1,
        18,
        0,
        tzinfo=timezone(timedelta(hours=6)),
    )
    item = make_item(updated_at=non_utc)
    match = make_ranked_match(item=item)

    assert match.updated_at == AS_OF

    other = make_item(external_id="2")
    with pytest.raises(ValidationError, match="Lexical match"):
        make_ranked_match(item=item, lexical_match=make_lexical_match(other))
    with pytest.raises(ValidationError, match="Semantic similarity"):
        make_ranked_match(
            item=item,
            mode=KnowledgeHybridRankingMode.HYBRID,
            semantic_similarity=make_similarity(other, 50),
        )


def test_ranked_match_enforces_deterministic_mode_shape_order_and_weights() -> None:
    item = make_item()
    with pytest.raises(ValidationError, match="must not include semantic"):
        make_ranked_match(
            item=item,
            semantic_similarity=make_similarity(item, 50),
        )

    reversed_contributions = list(
        reversed(make_contributions(KnowledgeHybridRankingMode.DETERMINISTIC))
    )
    with pytest.raises(ValidationError, match="signal order"):
        make_ranked_match(contributions=reversed_contributions)

    wrong_weight = make_contributions(KnowledgeHybridRankingMode.DETERMINISTIC)
    wrong_weight[0] = KnowledgeHybridSignalContribution(
        signal=KnowledgeHybridSignal.LEXICAL,
        signal_score=5,
        weight_percent=59,
        weighted_basis_points=295,
        explanation=EXPLANATIONS[KnowledgeHybridSignal.LEXICAL],
    )
    with pytest.raises(ValidationError, match="weights"):
        make_ranked_match(contributions=wrong_weight)


def test_ranked_match_enforces_hybrid_mode_shape_order_and_weights() -> None:
    item = make_item()
    hybrid = make_contributions(KnowledgeHybridRankingMode.HYBRID)

    with pytest.raises(ValidationError, match="requires semantic"):
        KnowledgeHybridRankedMatch(
            reference=item.to_reference(),
            state=item.state,
            decision_significance=item.decision_significance,
            updated_at=item.updated_at,
            mode=KnowledgeHybridRankingMode.HYBRID,
            lexical_match=make_lexical_match(item),
            semantic_similarity=None,
            contributions=hybrid,
        )

    without_semantic = [
        item
        for item in hybrid
        if item.signal != KnowledgeHybridSignal.SEMANTIC
    ]
    with pytest.raises(ValidationError, match="signal order"):
        make_ranked_match(
            item=item,
            mode=KnowledgeHybridRankingMode.HYBRID,
            contributions=without_semantic,
        )

    reordered = [hybrid[1], hybrid[0], *hybrid[2:]]
    with pytest.raises(ValidationError, match="signal order"):
        make_ranked_match(
            item=item,
            mode=KnowledgeHybridRankingMode.HYBRID,
            contributions=reordered,
        )

    wrong_weight = list(hybrid)
    wrong_weight[1] = KnowledgeHybridSignalContribution(
        signal=KnowledgeHybridSignal.SEMANTIC,
        signal_score=0,
        weight_percent=29,
        weighted_basis_points=0,
        explanation=EXPLANATIONS[KnowledgeHybridSignal.SEMANTIC],
    )
    with pytest.raises(ValidationError, match="weights"):
        make_ranked_match(
            item=item,
            mode=KnowledgeHybridRankingMode.HYBRID,
            contributions=wrong_weight,
        )


def test_boosts_alone_cannot_create_match_and_score_rounds_half_up() -> None:
    zero_relevance = make_contributions(
        KnowledgeHybridRankingMode.DETERMINISTIC,
        scores={
            KnowledgeHybridSignal.LEXICAL: 0,
            KnowledgeHybridSignal.DECISION_SIGNIFICANCE: 100,
            KnowledgeHybridSignal.RECENCY: 100,
        },
    )
    with pytest.raises(ValidationError, match="positive lexical or semantic"):
        make_ranked_match(contributions=zero_relevance)

    rounded = make_contributions(
        KnowledgeHybridRankingMode.DETERMINISTIC,
        scores={
            KnowledgeHybridSignal.LEXICAL: 1,
            KnowledgeHybridSignal.RECENCY: 18,
        },
    )
    match = make_ranked_match(contributions=rounded)

    assert match.total_weighted_basis_points == 150
    assert match.score == 2


def test_core_lexical_raw_score_excludes_paths_and_labels_and_caps() -> None:
    item = make_item()
    evidence = [
        make_lexical_evidence(
            kind=KnowledgeLexicalMatchKind.REFERENCE,
            field=KnowledgeLexicalField.EXTERNAL_ID,
            query_value="issue:1",
            document_value="issue:1",
            points=60,
        ),
        make_lexical_evidence(
            kind=KnowledgeLexicalMatchKind.EXACT_PHRASE,
            query_value="anchor phrase",
            document_value="anchor phrase",
            points=50,
        ),
        make_lexical_evidence(
            kind=KnowledgeLexicalMatchKind.LABEL,
            field=KnowledgeLexicalField.LABEL,
            query_value="bug",
            document_value="bug",
            points=10,
        ),
        make_lexical_evidence(
            kind=KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT,
            field=KnowledgeLexicalField.AFFECTED_PATH,
            query_value="src/a.py",
            document_value="src/a.py",
            points=25,
        ),
        make_lexical_evidence(
            kind=KnowledgeLexicalMatchKind.AFFECTED_PATH_SHARED_DIRECTORY,
            field=KnowledgeLexicalField.AFFECTED_PATH,
            query_value="src/b.py",
            document_value="src/c.py",
            points=8,
        ),
    ]
    match = make_ranked_match(
        item=item,
        lexical_match=make_lexical_match(item, evidence),
    )

    assert match.core_lexical_raw_score == 110
    assert match.core_lexical_score == 100


def test_input_repository_and_query_consistency_validation() -> None:
    query = make_query(text="anchor")
    item = make_item()
    corpus, lexical_result = make_context(query, [item])
    foreign_query = KnowledgeLexicalQuery(
        repository=OTHER_REPOSITORY,
        text="anchor",
    )

    with pytest.raises(ValueError, match="query and corpus repositories"):
        rank_knowledge_hybrid_corpus(
            foreign_query,
            corpus,
            lexical_result,
            as_of=AS_OF,
        )

    foreign_result = lexical_result.model_copy(
        update={
            "repository": OTHER_REPOSITORY,
            "query": foreign_query,
        }
    )
    with pytest.raises(ValueError, match="result and corpus repositories"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            foreign_result,
            as_of=AS_OF,
        )

    other_query = make_query(text="different")
    with pytest.raises(ValueError, match="query must equal"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result.model_copy(update={"query": other_query}),
            as_of=AS_OF,
        )


def test_lexical_count_completeness_validation() -> None:
    query = make_query(text="anchor")
    items = [make_item(external_id="1"), make_item(external_id="2")]
    corpus, lexical_result = make_context(query, items)

    with pytest.raises(ValueError, match="corpus count"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result.model_copy(update={"corpus_total_count": 1}),
            as_of=AS_OF,
        )

    with pytest.raises(ValueError, match="eligible count"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result.model_copy(update={"eligible_document_count": 1}),
            as_of=AS_OF,
        )

    truncated = KnowledgeLexicalSearchResult(
        repository=REPOSITORY,
        query=query,
        options=KnowledgeLexicalSearchOptions(max_results=1),
        corpus_total_count=2,
        eligible_document_count=2,
        matched_document_count=2,
        matches=lexical_result.matches[:1],
    )
    with pytest.raises(ValueError, match="Truncated"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            truncated,
            as_of=AS_OF,
        )

    mismatched_count = lexical_result.model_copy(
        update={"matched_document_count": 0}
    )
    with pytest.raises(ValueError, match="matched count"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            mismatched_count,
            as_of=AS_OF,
        )


def test_hybrid_requires_complete_positive_lexical_coverage() -> None:
    query = make_query(text="anchor")
    item = make_item()
    corpus, complete_result = make_context(query, [item])

    accepted = rank_knowledge_hybrid_corpus(
        query,
        corpus,
        complete_result,
        as_of=AS_OF,
    )
    assert accepted.returned_count == 1

    thresholded_result = complete_result.model_copy(
        update={
            "options": KnowledgeLexicalSearchOptions(minimum_score=2),
        }
    )
    with pytest.raises(
        ValueError,
        match=(
            "^Hybrid ranking requires a lexical result produced with "
            "minimum_score=1\\.$"
        ),
    ):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            thresholded_result,
            as_of=AS_OF,
        )


def test_lexical_threshold_validation_precedes_completeness_checks() -> None:
    query = make_query(text="anchor")
    items = [make_item(external_id="1"), make_item(external_id="2")]
    corpus, lexical_result = make_context(query, items)
    threshold_options = KnowledgeLexicalSearchOptions(
        max_results=1,
        minimum_score=2,
    )
    threshold_message = (
        "Hybrid ranking requires a lexical result produced with minimum_score=1."
    )
    truncated = lexical_result.model_copy(
        update={
            "options": threshold_options,
            "matches": lexical_result.matches[:1],
        }
    )
    mismatched_count = lexical_result.model_copy(
        update={
            "options": threshold_options,
            "matched_document_count": 0,
        }
    )

    with pytest.raises(ValueError, match=f"^{threshold_message}$"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            truncated,
            as_of=AS_OF,
        )
    with pytest.raises(ValueError, match=f"^{threshold_message}$"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            mismatched_count,
            as_of=AS_OF,
        )


def test_hybrid_rejects_thresholded_body_term_result() -> None:
    query = make_query(text="needle")
    item = make_item(
        title="Unrelated",
        body="The body contains needle.",
    )
    corpus = build_knowledge_lexical_corpus(REPOSITORY, [item])
    lexical_result = search_knowledge_lexical_corpus(
        query,
        corpus,
        options=KnowledgeLexicalSearchOptions(minimum_score=5),
    )

    assert lexical_result.matches == []
    assert lexical_result.matched_document_count == 0
    assert lexical_result.truncated is False

    with pytest.raises(
        ValueError,
        match=(
            "^Hybrid ranking requires a lexical result produced with "
            "minimum_score=1\\.$"
        ),
    ):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result,
            as_of=AS_OF,
        )


def test_inputs_are_not_mutated() -> None:
    query = make_query(
        text="anchor",
        labels=["Bug"],
        affected_paths=["src/a.py"],
    )
    item = make_item(labels=["Bug"], affected_paths=["src/a.py"])
    corpus, lexical_result = make_context(query, [item])
    semantic_scores = [make_similarity(item, 80)]
    snapshots = [
        query.model_dump(mode="json"),
        corpus.model_dump(mode="json"),
        lexical_result.model_dump(mode="json"),
        [score.model_dump(mode="json") for score in semantic_scores],
    ]

    rank_knowledge_hybrid_corpus(
        query,
        corpus,
        lexical_result,
        semantic_scores,
        as_of=AS_OF,
    )

    assert query.model_dump(mode="json") == snapshots[0]
    assert corpus.model_dump(mode="json") == snapshots[1]
    assert lexical_result.model_dump(mode="json") == snapshots[2]
    assert [
        score.model_dump(mode="json")
        for score in semantic_scores
    ] == snapshots[3]


def test_item_type_and_state_filters_recalculate_eligibility() -> None:
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

    by_type = rank(
        make_query(text="anchor", item_types=[KnowledgeItemType.ISSUE]),
        items,
    )
    by_state = rank(
        make_query(text="anchor", states=[KnowledgeItemState.OPEN]),
        items,
    )
    combined = rank(
        make_query(
            text="anchor",
            item_types=[KnowledgeItemType.ISSUE],
            states=[KnowledgeItemState.OPEN],
        ),
        items,
    )
    unfiltered = rank(make_query(text="anchor"), items)

    assert by_type.eligible_document_count == 2
    assert by_state.eligible_document_count == 2
    assert combined.eligible_document_count == 1
    assert combined.matches[0].reference.external_id == "1"
    assert unfiltered.eligible_document_count == 3


def test_ineligible_semantic_score_and_future_document_are_rejected() -> None:
    eligible = make_item(
        external_id="1",
        item_type=KnowledgeItemType.ISSUE,
    )
    ineligible = make_item(
        external_id="2",
        item_type=KnowledgeItemType.PULL_REQUEST,
    )
    query = make_query(
        text="anchor",
        item_types=[KnowledgeItemType.ISSUE],
    )
    corpus, lexical_result = make_context(query, [eligible, ineligible])

    with pytest.raises(ValueError, match="exactly cover"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result,
            [make_similarity(eligible, 50), make_similarity(ineligible, 50)],
            as_of=AS_OF,
        )

    future = make_item(updated_at=AS_OF + timedelta(seconds=1))
    future_query = make_query(text="anchor")
    future_corpus, future_lexical = make_context(future_query, [future])
    with pytest.raises(ValueError, match="earlier"):
        rank_knowledge_hybrid_corpus(
            future_query,
            future_corpus,
            future_lexical,
            as_of=AS_OF,
        )


def test_none_and_empty_semantic_scores_use_deterministic_mode() -> None:
    query = make_query(text="anchor")
    item = make_item()

    none_result = rank(query, [item])
    empty_result = rank(query, [item], [])

    assert none_result.mode == KnowledgeHybridRankingMode.DETERMINISTIC
    assert empty_result.mode == KnowledgeHybridRankingMode.DETERMINISTIC
    assert all(
        contribution.signal != KnowledgeHybridSignal.SEMANTIC
        for contribution in none_result.matches[0].contributions
    )


def test_complete_semantic_coverage_uses_hybrid_mode_including_zero() -> None:
    items = [make_item(external_id="1"), make_item(external_id="2")]
    result = rank(
        make_query(text="anchor"),
        items,
        [make_similarity(items[0], 0), make_similarity(items[1], 100)],
    )

    assert result.mode == KnowledgeHybridRankingMode.HYBRID
    assert result.eligible_document_count == 2
    assert {
        match.semantic_similarity.score
        for match in result.matches
        if match.semantic_similarity is not None
    } == {0, 100}


def test_partial_extra_duplicate_and_cross_repository_semantics_are_rejected() -> None:
    items = [make_item(external_id="1"), make_item(external_id="2")]
    query = make_query(text="anchor")
    corpus, lexical_result = make_context(query, items)
    first = make_similarity(items[0], 50)

    with pytest.raises(ValueError, match="exactly cover"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result,
            [first],
            as_of=AS_OF,
        )
    with pytest.raises(ValueError, match="unique"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result,
            [first, first],
            as_of=AS_OF,
        )

    foreign = make_similarity(
        make_item(repository=OTHER_REPOSITORY),
        50,
    )
    with pytest.raises(ValueError, match="corpus repository"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result,
            [first, foreign],
            as_of=AS_OF,
        )


def test_semantic_only_candidate_and_zero_relevance_exclusion() -> None:
    semantic_item = make_item(external_id="1", title="unrelated")
    zero_item = make_item(external_id="2", title="also unrelated")
    result = rank(
        make_query(text="absent"),
        [semantic_item, zero_item],
        [
            make_similarity(semantic_item, 80),
            make_similarity(zero_item, 0),
        ],
    )
    match = result.matches[0]

    assert result.mode == KnowledgeHybridRankingMode.HYBRID
    assert result.candidate_count == 1
    assert match.reference.external_id == "1"
    assert match.lexical_match is None
    assert match.core_lexical_score == 0
    assert contribution(match, KnowledgeHybridSignal.LEXICAL).signal_score == 0
    semantic = contribution(match, KnowledgeHybridSignal.SEMANTIC)
    assert semantic.signal_score == 80
    assert semantic.explanation == EXPLANATIONS[KnowledgeHybridSignal.SEMANTIC]


@pytest.mark.parametrize(
    ("kind", "field", "points"),
    [
        (
            KnowledgeLexicalMatchKind.REFERENCE,
            KnowledgeLexicalField.EXTERNAL_ID,
            60,
        ),
        (
            KnowledgeLexicalMatchKind.EXACT_PHRASE,
            KnowledgeLexicalField.TITLE,
            30,
        ),
        (
            KnowledgeLexicalMatchKind.IDENTIFIER,
            KnowledgeLexicalField.TITLE,
            18,
        ),
        (
            KnowledgeLexicalMatchKind.COMPONENT,
            KnowledgeLexicalField.COMPONENT,
            15,
        ),
        (
            KnowledgeLexicalMatchKind.TERM,
            KnowledgeLexicalField.TITLE,
            5,
        ),
    ],
)
def test_supported_core_lexical_evidence_contributes(
    kind: KnowledgeLexicalMatchKind,
    field: KnowledgeLexicalField,
    points: int,
) -> None:
    item = make_item()
    query = make_query(text="anchor")
    corpus = build_knowledge_lexical_corpus(REPOSITORY, [item])
    lexical_match = make_lexical_match(
        item,
        [
            make_lexical_evidence(
                kind=kind,
                field=field,
                points=points,
            )
        ],
    )
    lexical_result = make_lexical_result(query, corpus, [lexical_match])
    result = rank_knowledge_hybrid_corpus(
        query,
        corpus,
        lexical_result,
        as_of=AS_OF,
    )
    match = result.matches[0]

    assert match.core_lexical_raw_score == points
    assert contribution(
        match,
        KnowledgeHybridSignal.LEXICAL,
    ).signal_score == points
    assert contribution(
        match,
        KnowledgeHybridSignal.LEXICAL,
    ).explanation == EXPLANATIONS[KnowledgeHybridSignal.LEXICAL]


def test_path_and_label_evidence_are_excluded_from_core_lexical_signal() -> None:
    item = make_item(labels=["Bug"], affected_paths=["src/parser/a.py"])
    query = make_query(
        text="anchor",
        labels=["Bug"],
        affected_paths=["src/parser/a.py"],
    )
    result = rank(query, [item])
    match = result.matches[0]

    assert match.lexical_match is not None
    assert match.lexical_match.raw_score == 40
    assert match.core_lexical_raw_score == 5
    assert contribution(match, KnowledgeHybridSignal.LEXICAL).signal_score == 5


@pytest.mark.parametrize(
    ("document_path", "expected_score"),
    [
        ("src/parser/registry.py", 100),
        ("src/parser/cache.py", 40),
        ("src/runtime/cache.py", 0),
    ],
)
def test_path_signal_exact_shared_and_missing(
    document_path: str,
    expected_score: int,
) -> None:
    result = rank(
        make_query(
            text="anchor",
            affected_paths=["src/parser/registry.py"],
        ),
        [make_item(affected_paths=[document_path])],
    )
    path = contribution(result.matches[0], KnowledgeHybridSignal.AFFECTED_PATH)

    assert path.signal_score == expected_score
    assert path.explanation == EXPLANATIONS[KnowledgeHybridSignal.AFFECTED_PATH]


def test_path_signal_averages_multiple_paths_with_half_up_rounding() -> None:
    query = make_query(
        text="anchor",
        affected_paths=[
            "src/parser/a.py",
            "tests/parser/a.py",
            "docs/parser/a.md",
        ],
    )
    item = make_item(
        affected_paths=[
            "src/parser/a.py",
            "tests/parser/b.py",
        ]
    )
    result = rank(query, [item])

    assert contribution(
        result.matches[0],
        KnowledgeHybridSignal.AFFECTED_PATH,
    ).signal_score == 47


def test_empty_paths_and_duplicate_path_evidence() -> None:
    empty = rank(make_query(text="anchor"), [make_item()])
    assert contribution(
        empty.matches[0],
        KnowledgeHybridSignal.AFFECTED_PATH,
    ).signal_score == 0

    item = make_item()
    query = make_query(text="anchor", affected_paths=["src/a.py"])
    corpus = build_knowledge_lexical_corpus(REPOSITORY, [item])
    lexical_match = make_lexical_match(
        item,
        [
            make_lexical_evidence(),
            make_lexical_evidence(
                kind=KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT,
                field=KnowledgeLexicalField.AFFECTED_PATH,
                query_value="src/a.py",
                document_value="src/a.py",
                points=25,
            ),
            make_lexical_evidence(
                kind=KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT,
                field=KnowledgeLexicalField.AFFECTED_PATH,
                query_value="src/a.py",
                document_value="src/other.py",
                points=25,
            ),
        ],
    )
    result = rank_knowledge_hybrid_corpus(
        query,
        corpus,
        make_lexical_result(query, corpus, [lexical_match]),
        as_of=AS_OF,
    )

    assert contribution(
        result.matches[0],
        KnowledgeHybridSignal.AFFECTED_PATH,
    ).signal_score == 100


def test_path_evidence_uses_normalized_query_identity() -> None:
    query = make_query(
        text="anchor",
        affected_paths=[".\\src\\parser\\a.py"],
    )
    result = rank(
        query,
        [make_item(affected_paths=["src/parser/a.py"])],
    )

    assert query.affected_paths == ["src/parser/a.py"]
    assert contribution(
        result.matches[0],
        KnowledgeHybridSignal.AFFECTED_PATH,
    ).signal_score == 100


def test_label_signal_full_partial_nfkc_and_empty() -> None:
    all_matched = rank(
        make_query(text="anchor", labels=["ＢＵＧ", "Feature"]),
        [make_item(labels=["bug", "FEATURE"])],
    )
    partial = rank(
        make_query(text="anchor", labels=["One", "Two", "Three"]),
        [make_item(labels=["one"])],
    )
    empty = rank(make_query(text="anchor"), [make_item()])

    assert contribution(
        all_matched.matches[0],
        KnowledgeHybridSignal.LABEL,
    ).signal_score == 100
    assert contribution(
        partial.matches[0],
        KnowledgeHybridSignal.LABEL,
    ).signal_score == 33
    empty_label = contribution(empty.matches[0], KnowledgeHybridSignal.LABEL)
    assert empty_label.signal_score == 0
    assert empty_label.explanation == EXPLANATIONS[KnowledgeHybridSignal.LABEL]


def test_repeated_label_evidence_counts_once() -> None:
    item = make_item()
    query = make_query(text="anchor", labels=["Bug", "Feature"])
    corpus = build_knowledge_lexical_corpus(REPOSITORY, [item])
    lexical_match = make_lexical_match(
        item,
        [
            make_lexical_evidence(),
            make_lexical_evidence(
                kind=KnowledgeLexicalMatchKind.LABEL,
                field=KnowledgeLexicalField.LABEL,
                query_value="ＢＵＧ",
                document_value="bug",
                points=10,
            ),
            make_lexical_evidence(
                kind=KnowledgeLexicalMatchKind.LABEL,
                field=KnowledgeLexicalField.LABEL,
                query_value="bug",
                document_value="bug-alias",
                points=10,
            ),
        ],
    )
    result = rank_knowledge_hybrid_corpus(
        query,
        corpus,
        make_lexical_result(query, corpus, [lexical_match]),
        as_of=AS_OF,
    )

    assert contribution(
        result.matches[0],
        KnowledgeHybridSignal.LABEL,
    ).signal_score == 50


@pytest.mark.parametrize(
    ("significance", "expected_score"),
    [
        (DecisionSignificance.NONE, 0),
        (DecisionSignificance.LOW, 25),
        (DecisionSignificance.MEDIUM, 50),
        (DecisionSignificance.HIGH, 75),
        (DecisionSignificance.CRITICAL, 100),
    ],
)
def test_significance_mapping_and_explanation(
    significance: DecisionSignificance,
    expected_score: int,
) -> None:
    result = rank(
        make_query(text="anchor"),
        [make_item(significance=significance)],
    )
    significance_contribution = contribution(
        result.matches[0],
        KnowledgeHybridSignal.DECISION_SIGNIFICANCE,
    )

    assert significance_contribution.signal_score == expected_score
    assert significance_contribution.explanation == EXPLANATIONS[
        KnowledgeHybridSignal.DECISION_SIGNIFICANCE
    ]


def test_lexical_and_document_significance_must_match() -> None:
    item = make_item(significance=DecisionSignificance.HIGH)
    query = make_query(text="anchor")
    corpus, lexical_result = make_context(query, [item])
    inconsistent = lexical_result.matches[0].model_copy(
        update={"decision_significance": DecisionSignificance.LOW}
    )
    invalid_result = lexical_result.model_copy(update={"matches": [inconsistent]})

    with pytest.raises(ValueError, match="significance"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            invalid_result,
            as_of=AS_OF,
        )


@pytest.mark.parametrize(
    ("age", "expected_score"),
    [
        (timedelta(days=30), 100),
        (timedelta(days=30, seconds=1), 80),
        (timedelta(days=90), 80),
        (timedelta(days=180), 60),
        (timedelta(days=365), 40),
        (timedelta(days=730), 20),
        (timedelta(days=730, seconds=1), 0),
    ],
)
def test_recency_buckets_and_explanation(
    age: timedelta,
    expected_score: int,
) -> None:
    result = rank(
        make_query(text="anchor"),
        [make_item(updated_at=AS_OF - age)],
    )
    recency = contribution(result.matches[0], KnowledgeHybridSignal.RECENCY)

    assert recency.signal_score == expected_score
    assert recency.explanation == EXPLANATIONS[KnowledgeHybridSignal.RECENCY]


def test_as_of_requires_awareness_and_normalizes_to_utc() -> None:
    query = make_query(text="anchor")
    item = make_item()
    corpus, lexical_result = make_context(query, [item])

    with pytest.raises(ValueError, match="timezone-aware"):
        rank_knowledge_hybrid_corpus(
            query,
            corpus,
            lexical_result,
            as_of=datetime(2026, 7, 1, 12, 0),
        )

    non_utc = datetime(
        2026,
        7,
        1,
        18,
        0,
        tzinfo=timezone(timedelta(hours=6)),
    )
    result = rank_knowledge_hybrid_corpus(
        query,
        corpus,
        lexical_result,
        as_of=non_utc,
    )
    assert result.as_of == AS_OF


def test_deterministic_fusion_order_weights_zeros_and_score() -> None:
    query = make_query(
        text="anchor",
        affected_paths=["src/a.py"],
        labels=["Bug"],
    )
    item = make_item(
        affected_paths=["src/a.py"],
        labels=["Bug"],
        significance=DecisionSignificance.HIGH,
    )
    match = rank(query, [item]).matches[0]

    assert len(match.contributions) == 5
    assert [item.signal for item in match.contributions] == list(
        DETERMINISTIC_WEIGHTS
    )
    assert [item.weight_percent for item in match.contributions] == list(
        DETERMINISTIC_WEIGHTS.values()
    )
    assert [item.signal_score for item in match.contributions] == [
        5,
        100,
        100,
        75,
        100,
    ]
    assert match.total_weighted_basis_points == 4_050
    assert match.score == 41

    text_only = rank(make_query(text="anchor"), [make_item()]).matches[0]
    assert contribution(
        text_only,
        KnowledgeHybridSignal.AFFECTED_PATH,
    ).signal_score == 0
    assert contribution(
        text_only,
        KnowledgeHybridSignal.LABEL,
    ).signal_score == 0


def test_deterministic_minimum_score_excludes_and_does_not_count() -> None:
    result = rank(
        make_query(text="anchor"),
        [make_item()],
        options=KnowledgeHybridRankingOptions(minimum_score=9),
    )

    assert result.candidate_count == 0
    assert result.matches == []


def test_hybrid_fusion_order_weights_semantic_and_score() -> None:
    query = make_query(
        text="anchor",
        affected_paths=["src/a.py"],
        labels=["Bug"],
    )
    item = make_item(
        affected_paths=["src/a.py"],
        labels=["Bug"],
        significance=DecisionSignificance.HIGH,
    )
    match = rank(
        query,
        [item],
        [make_similarity(item, 80)],
    ).matches[0]

    assert len(match.contributions) == 6
    assert [item.signal for item in match.contributions] == list(HYBRID_WEIGHTS)
    assert [item.weight_percent for item in match.contributions] == list(
        HYBRID_WEIGHTS.values()
    )
    assert [item.signal_score for item in match.contributions] == [
        5,
        80,
        100,
        100,
        75,
        100,
    ]
    assert match.total_weighted_basis_points == 5_350
    assert match.score == 54


def test_lexical_candidate_with_zero_semantic_score_is_retained() -> None:
    item = make_item()
    result = rank(
        make_query(text="anchor"),
        [item],
        [make_similarity(item, 0)],
    )

    assert result.candidate_count == 1
    assert contribution(
        result.matches[0],
        KnowledgeHybridSignal.SEMANTIC,
    ).signal_score == 0


def test_ranking_uses_total_basis_points_descending() -> None:
    first = make_item(external_id="1")
    second = make_item(external_id="2")
    result = rank(
        make_query(text="anchor"),
        [first, second],
        [make_similarity(first, 10), make_similarity(second, 90)],
    )

    assert [match.reference.external_id for match in result.matches] == ["2", "1"]


def test_total_tie_uses_core_lexical_raw_score() -> None:
    high_core = make_item(external_id="2", title="unrelated")
    semantic_offset = make_item(external_id="1", title="target phrase")
    query = make_query(
        exact_phrases=["target phrase"],
        references=[
            KnowledgeLexicalReference(
                item_type=KnowledgeItemType.ISSUE,
                external_id="2",
            )
        ],
    )
    result = rank(
        query,
        [high_core, semantic_offset],
        [
            make_similarity(high_core, 0),
            make_similarity(semantic_offset, 40),
        ],
    )

    assert [
        match.total_weighted_basis_points
        for match in result.matches
    ] == [2_900, 2_900]
    assert [match.reference.external_id for match in result.matches] == ["2", "1"]
    assert [match.core_lexical_raw_score for match in result.matches] == [60, 30]


def test_remaining_tie_uses_semantic_score_before_item_key() -> None:
    semantic_winner = make_item(
        external_id="2",
        updated_at=AS_OF - timedelta(days=365),
    )
    recent_offset = make_item(
        external_id="1",
        updated_at=AS_OF - timedelta(days=10),
    )
    result = rank(
        make_query(text="anchor"),
        [semantic_winner, recent_offset],
        [
            make_similarity(semantic_winner, 10),
            make_similarity(recent_offset, 0),
        ],
    )

    assert [
        match.total_weighted_basis_points
        for match in result.matches
    ] == [700, 700]
    assert [match.core_lexical_raw_score for match in result.matches] == [5, 5]
    assert [match.reference.external_id for match in result.matches] == ["2", "1"]


def test_final_tie_uses_item_key_and_not_input_order() -> None:
    first = make_item(external_id="1")
    second = make_item(external_id="2")
    result = rank(
        make_query(text="anchor"),
        [second, first],
    )

    assert [match.item_key for match in result.matches] == sorted(
        match.item_key for match in result.matches
    )


def test_significance_and_updated_at_only_affect_weighted_contributions() -> None:
    significant_old = make_item(
        external_id="2",
        significance=DecisionSignificance.HIGH,
        updated_at=AS_OF - timedelta(days=365),
    )
    recent_plain = make_item(
        external_id="1",
        significance=DecisionSignificance.NONE,
        updated_at=AS_OF - timedelta(days=10),
    )
    result = rank(
        make_query(text="anchor"),
        [significant_old, recent_plain],
    )

    assert result.matches[0].total_weighted_basis_points == 1_250
    assert result.matches[1].total_weighted_basis_points == 800


def test_max_results_counts_before_truncation() -> None:
    items = [
        make_item(external_id=str(index))
        for index in range(1, 4)
    ]
    result = rank(
        make_query(text="anchor"),
        items,
        options=KnowledgeHybridRankingOptions(max_results=2),
    )

    assert result.candidate_count == 3
    assert result.returned_count == 2
    assert result.truncated is True


def test_empty_deterministic_and_all_zero_hybrid_results_are_valid() -> None:
    deterministic = rank(
        make_query(text="absent"),
        [make_item(title="unrelated")],
    )
    item = make_item(title="unrelated")
    hybrid = rank(
        make_query(text="absent"),
        [item],
        [make_similarity(item, 0)],
    )

    assert deterministic.mode == KnowledgeHybridRankingMode.DETERMINISTIC
    assert deterministic.matches == []
    assert deterministic.candidate_count == 0
    assert hybrid.mode == KnowledgeHybridRankingMode.HYBRID
    assert hybrid.matches == []
    assert hybrid.candidate_count == 0


def result_payload(
    result: KnowledgeHybridRankingResult,
) -> dict[str, object]:
    """Return declared result fields for validation mutations."""

    return {
        field_name: getattr(result, field_name)
        for field_name in KnowledgeHybridRankingResult.model_fields
    }


def test_result_rejects_query_match_mode_and_key_inconsistencies() -> None:
    valid = rank(make_query(text="anchor"), [make_item()])
    payload = result_payload(valid)
    payload["query"] = KnowledgeLexicalQuery(
        repository=OTHER_REPOSITORY,
        text="anchor",
    )
    with pytest.raises(ValidationError, match="result repository"):
        KnowledgeHybridRankingResult.model_validate(payload)

    foreign_match = make_ranked_match(item=make_item(repository=OTHER_REPOSITORY))
    payload = result_payload(valid)
    payload["matches"] = [foreign_match]
    with pytest.raises(ValidationError, match="result repository"):
        KnowledgeHybridRankingResult.model_validate(payload)

    hybrid_match = make_ranked_match(
        mode=KnowledgeHybridRankingMode.HYBRID,
    )
    payload = result_payload(valid)
    payload["matches"] = [hybrid_match]
    with pytest.raises(ValidationError, match="ranking mode"):
        KnowledgeHybridRankingResult.model_validate(payload)

    payload = result_payload(valid)
    payload["matches"] = [valid.matches[0], valid.matches[0]]
    payload["candidate_count"] = 2
    payload["eligible_document_count"] = 2
    payload["corpus_total_count"] = 2
    with pytest.raises(ValidationError, match="unique"):
        KnowledgeHybridRankingResult.model_validate(payload)


@pytest.mark.parametrize(
    ("corpus_count", "eligible_count", "candidate_count", "match_count"),
    [
        (0, 1, 0, 0),
        (1, 0, 1, 0),
        (1, 1, 0, 1),
    ],
)
def test_result_rejects_impossible_counts(
    corpus_count: int,
    eligible_count: int,
    candidate_count: int,
    match_count: int,
) -> None:
    match = make_ranked_match()

    with pytest.raises(ValidationError):
        KnowledgeHybridRankingResult(
            repository=REPOSITORY,
            query=make_query(text="anchor"),
            mode=KnowledgeHybridRankingMode.DETERMINISTIC,
            as_of=AS_OF,
            corpus_total_count=corpus_count,
            eligible_document_count=eligible_count,
            candidate_count=candidate_count,
            matches=[match] * match_count,
        )


def test_result_rejects_wrong_order_and_zero_score() -> None:
    first = make_ranked_match(item=make_item(external_id="1"))
    second = make_ranked_match(item=make_item(external_id="2"))
    with pytest.raises(ValidationError, match="ranking order"):
        KnowledgeHybridRankingResult(
            repository=REPOSITORY,
            query=make_query(text="anchor"),
            mode=KnowledgeHybridRankingMode.DETERMINISTIC,
            as_of=AS_OF,
            corpus_total_count=2,
            eligible_document_count=2,
            candidate_count=2,
            matches=[second, first],
        )

    zero_score = KnowledgeHybridRankedMatch.model_construct(
        reference=make_item().to_reference(),
        state=KnowledgeItemState.CLOSED,
        decision_significance=DecisionSignificance.NONE,
        updated_at=RECENT_AT,
        mode=KnowledgeHybridRankingMode.DETERMINISTIC,
        lexical_match=None,
        semantic_similarity=None,
        contributions=[],
    )
    invalid_result = KnowledgeHybridRankingResult.model_construct(
        repository=REPOSITORY,
        query=make_query(text="anchor"),
        mode=KnowledgeHybridRankingMode.DETERMINISTIC,
        as_of=AS_OF,
        corpus_total_count=1,
        eligible_document_count=1,
        candidate_count=1,
        matches=[zero_score],
    )
    with pytest.raises(ValueError, match="positive score"):
        invalid_result.validate_result()


def test_result_json_serializes_provenance_scores_counts_enums_and_utc() -> None:
    non_utc_as_of = datetime(
        2026,
        7,
        1,
        18,
        0,
        tzinfo=timezone(timedelta(hours=6)),
    )
    item = make_item()
    result = rank(
        make_query(text="anchor"),
        [item],
        [make_similarity(item, 80, provider="provider", model="model-v1")],
        as_of=non_utc_as_of,
    )
    payload = result.model_dump(mode="json")
    match = payload["matches"][0]

    assert payload["mode"] == "hybrid"
    assert payload["as_of"] == "2026-07-01T12:00:00Z"
    assert payload["returned_count"] == 1
    assert payload["truncated"] is False
    assert match["mode"] == "hybrid"
    assert match["state"] == "closed"
    assert match["decision_significance"] == "none"
    assert match["updated_at"].endswith("Z")
    assert match["semantic_similarity"]["provider"] == "provider"
    assert match["semantic_similarity"]["model"] == "model-v1"
    assert match["semantic_similarity"]["score"] == 80
    assert match["contributions"][0]["signal"] == "lexical"
    assert match["contributions"][0]["weighted_score"] == 2.0
    assert match["total_weighted_basis_points"] == (
        result.matches[0].total_weighted_basis_points
    )
    assert match["score"] == result.matches[0].score
    assert match["core_lexical_raw_score"] == 5
    assert match["core_lexical_score"] == 5


def test_public_models_reject_extra_fields() -> None:
    payload = make_similarity(make_item(), 50).model_dump()
    payload["embedding"] = [0.1]
    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeSemanticSimilarity.model_validate(payload)

    contribution_payload = make_contributions(
        KnowledgeHybridRankingMode.DETERMINISTIC
    )[0].model_dump()
    contribution_payload["metadata"] = {}
    with pytest.raises(ValidationError, match="Extra inputs"):
        KnowledgeHybridSignalContribution.model_validate(contribution_payload)
