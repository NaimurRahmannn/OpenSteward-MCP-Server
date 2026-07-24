"""Provider-independent hybrid related-work ranking and signal fusion."""

import unicodedata
from collections.abc import Callable, Iterable
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Self

from pydantic import Field, computed_field, field_validator, model_validator

from opensteward.knowledge.lexical import (
    KnowledgeLexicalCorpus,
    KnowledgeLexicalDocument,
    KnowledgeLexicalQuery,
)
from opensteward.knowledge.lexical_search import (
    KnowledgeLexicalMatch,
    KnowledgeLexicalMatchKind,
    KnowledgeLexicalSearchResult,
)
from opensteward.knowledge.models import (
    DecisionSignificance,
    KnowledgeItemReference,
    KnowledgeItemState,
    KnowledgeRepositoryRef,
    StrictKnowledgeModel,
)

MAX_KNOWLEDGE_HYBRID_RESULTS = 100
MAX_KNOWLEDGE_HYBRID_SCORE = 100
MAX_KNOWLEDGE_HYBRID_BASIS_POINTS = 10_000

KNOWLEDGE_DETERMINISTIC_LEXICAL_WEIGHT = 60
KNOWLEDGE_DETERMINISTIC_PATH_WEIGHT = 15
KNOWLEDGE_DETERMINISTIC_LABEL_WEIGHT = 10
KNOWLEDGE_DETERMINISTIC_SIGNIFICANCE_WEIGHT = 10
KNOWLEDGE_DETERMINISTIC_RECENCY_WEIGHT = 5

KNOWLEDGE_HYBRID_LEXICAL_WEIGHT = 40
KNOWLEDGE_HYBRID_SEMANTIC_WEIGHT = 30
KNOWLEDGE_HYBRID_PATH_WEIGHT = 10
KNOWLEDGE_HYBRID_LABEL_WEIGHT = 5
KNOWLEDGE_HYBRID_SIGNIFICANCE_WEIGHT = 10
KNOWLEDGE_HYBRID_RECENCY_WEIGHT = 5

_CORE_LEXICAL_EXCLUDED_KINDS = {
    KnowledgeLexicalMatchKind.LABEL,
    KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT,
    KnowledgeLexicalMatchKind.AFFECTED_PATH_SHARED_DIRECTORY,
}
_SIGNIFICANCE_SCORES = {
    DecisionSignificance.NONE: 0,
    DecisionSignificance.LOW: 25,
    DecisionSignificance.MEDIUM: 50,
    DecisionSignificance.HIGH: 75,
    DecisionSignificance.CRITICAL: 100,
}


class KnowledgeHybridRankingMode(StrEnum):
    """Available hybrid ranking modes."""

    DETERMINISTIC = "deterministic"
    HYBRID = "hybrid"


class KnowledgeHybridSignal(StrEnum):
    """Signals that contribute to a hybrid related-work score."""

    LEXICAL = "lexical"
    SEMANTIC = "semantic"
    AFFECTED_PATH = "affected_path"
    LABEL = "label"
    DECISION_SIGNIFICANCE = "decision_significance"
    RECENCY = "recency"


_DETERMINISTIC_SIGNAL_WEIGHTS = {
    KnowledgeHybridSignal.LEXICAL: KNOWLEDGE_DETERMINISTIC_LEXICAL_WEIGHT,
    KnowledgeHybridSignal.AFFECTED_PATH: KNOWLEDGE_DETERMINISTIC_PATH_WEIGHT,
    KnowledgeHybridSignal.LABEL: KNOWLEDGE_DETERMINISTIC_LABEL_WEIGHT,
    KnowledgeHybridSignal.DECISION_SIGNIFICANCE: (
        KNOWLEDGE_DETERMINISTIC_SIGNIFICANCE_WEIGHT
    ),
    KnowledgeHybridSignal.RECENCY: KNOWLEDGE_DETERMINISTIC_RECENCY_WEIGHT,
}
_HYBRID_SIGNAL_WEIGHTS = {
    KnowledgeHybridSignal.LEXICAL: KNOWLEDGE_HYBRID_LEXICAL_WEIGHT,
    KnowledgeHybridSignal.SEMANTIC: KNOWLEDGE_HYBRID_SEMANTIC_WEIGHT,
    KnowledgeHybridSignal.AFFECTED_PATH: KNOWLEDGE_HYBRID_PATH_WEIGHT,
    KnowledgeHybridSignal.LABEL: KNOWLEDGE_HYBRID_LABEL_WEIGHT,
    KnowledgeHybridSignal.DECISION_SIGNIFICANCE: (
        KNOWLEDGE_HYBRID_SIGNIFICANCE_WEIGHT
    ),
    KnowledgeHybridSignal.RECENCY: KNOWLEDGE_HYBRID_RECENCY_WEIGHT,
}


class KnowledgeSemanticSimilarity(StrictKnowledgeModel):
    """Validated externally calculated semantic similarity."""

    reference: KnowledgeItemReference
    score: int = Field(ge=0, le=MAX_KNOWLEDGE_HYBRID_SCORE)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the semantically scored knowledge-item key."""

        return self.reference.key


class KnowledgeHybridRankingOptions(StrictKnowledgeModel):
    """Bounds and threshold for one hybrid ranking operation."""

    max_results: int = Field(
        default=20,
        ge=1,
        le=MAX_KNOWLEDGE_HYBRID_RESULTS,
    )
    minimum_score: int = Field(
        default=1,
        ge=1,
        le=MAX_KNOWLEDGE_HYBRID_SCORE,
    )


class KnowledgeHybridSignalContribution(StrictKnowledgeModel):
    """One weighted and explainable hybrid ranking signal."""

    signal: KnowledgeHybridSignal
    signal_score: int = Field(ge=0, le=MAX_KNOWLEDGE_HYBRID_SCORE)
    weight_percent: int = Field(ge=1, le=100)
    weighted_basis_points: int = Field(
        ge=0,
        le=MAX_KNOWLEDGE_HYBRID_BASIS_POINTS,
    )
    explanation: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_weighted_basis_points(self) -> Self:
        """Require weighted basis points to match score times weight."""

        if self.weighted_basis_points != self.signal_score * self.weight_percent:
            raise ValueError(
                "weighted_basis_points must equal signal_score times weight_percent."
            )
        return self

    @computed_field
    @property
    def weighted_score(self) -> float:
        """Return the explanatory weighted score."""

        return self.weighted_basis_points / 100


class KnowledgeHybridRankedMatch(StrictKnowledgeModel):
    """One historical item with fused ranking contributions."""

    reference: KnowledgeItemReference
    state: KnowledgeItemState
    decision_significance: DecisionSignificance
    updated_at: datetime
    mode: KnowledgeHybridRankingMode
    lexical_match: KnowledgeLexicalMatch | None
    semantic_similarity: KnowledgeSemanticSimilarity | None
    contributions: list[KnowledgeHybridSignalContribution] = Field(min_length=1)

    @field_validator("updated_at")
    @classmethod
    def normalize_updated_at(cls, updated_at: datetime) -> datetime:
        """Require an aware update timestamp and normalize it to UTC."""

        return _normalize_aware_datetime(updated_at, "updated_at")

    @model_validator(mode="after")
    def validate_ranking_inputs(self) -> Self:
        """Validate source identities, mode shape, signal order, and weights."""

        signals = [contribution.signal for contribution in self.contributions]
        if len(signals) != len(set(signals)):
            raise ValueError("Hybrid contribution signals must be unique.")

        if (
            self.lexical_match is not None
            and self.lexical_match.item_key != self.reference.key
        ):
            raise ValueError("Lexical match must identify the ranked reference.")
        if (
            self.semantic_similarity is not None
            and self.semantic_similarity.item_key != self.reference.key
        ):
            raise ValueError("Semantic similarity must identify the ranked reference.")

        expected_weights = _weights_for_mode(self.mode)
        expected_signals = list(expected_weights)
        if self.mode == KnowledgeHybridRankingMode.DETERMINISTIC:
            if self.semantic_similarity is not None:
                raise ValueError("Deterministic ranking must not include semantic data.")
        elif self.semantic_similarity is None:
            raise ValueError("Hybrid ranking requires semantic similarity.")

        if signals != expected_signals:
            raise ValueError("Hybrid contributions use an invalid signal order.")
        if any(
            contribution.weight_percent != expected_weights[contribution.signal]
            for contribution in self.contributions
        ):
            raise ValueError("Hybrid contributions use invalid signal weights.")

        if self.total_weighted_basis_points > MAX_KNOWLEDGE_HYBRID_BASIS_POINTS:
            raise ValueError("Hybrid weighted basis points exceed the safety limit.")

        contribution_scores = {
            contribution.signal: contribution.signal_score
            for contribution in self.contributions
        }
        if (
            contribution_scores[KnowledgeHybridSignal.LEXICAL] == 0
            and contribution_scores.get(KnowledgeHybridSignal.SEMANTIC, 0) == 0
        ):
            raise ValueError(
                "A ranked match requires positive lexical or semantic relevance."
            )
        return self

    @computed_field
    @property
    def item_key(self) -> str:
        """Return the ranked knowledge-item key."""

        return self.reference.key

    @computed_field
    @property
    def total_weighted_basis_points(self) -> int:
        """Return the sum of integer weighted basis points."""

        return sum(
            contribution.weighted_basis_points
            for contribution in self.contributions
        )

    @computed_field
    @property
    def score(self) -> int:
        """Return the nearest whole hybrid score using half-up rounding."""

        return (self.total_weighted_basis_points + 50) // 100

    @computed_field
    @property
    def core_lexical_raw_score(self) -> int:
        """Return lexical evidence points excluding path and label evidence."""

        return _core_lexical_raw_score(self.lexical_match)

    @computed_field
    @property
    def core_lexical_score(self) -> int:
        """Return the capped core lexical relevance score."""

        return min(MAX_KNOWLEDGE_HYBRID_SCORE, self.core_lexical_raw_score)


class KnowledgeHybridRankingResult(StrictKnowledgeModel):
    """Validated deterministic hybrid ranking output."""

    repository: KnowledgeRepositoryRef
    query: KnowledgeLexicalQuery
    mode: KnowledgeHybridRankingMode
    as_of: datetime
    corpus_total_count: int = Field(ge=0)
    eligible_document_count: int = Field(ge=0)
    candidate_count: int = Field(ge=0)
    matches: list[KnowledgeHybridRankedMatch]

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, as_of: datetime) -> datetime:
        """Require an aware ranking time and normalize it to UTC."""

        return _normalize_aware_datetime(as_of, "as_of")

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        """Validate repository identity, counts, uniqueness, and ranking."""

        if self.query.repository != self.repository:
            raise ValueError("Hybrid query must belong to the result repository.")
        if any(
            match.reference.repository != self.repository
            for match in self.matches
        ):
            raise ValueError("Every hybrid match must belong to the result repository.")
        if any(match.mode != self.mode for match in self.matches):
            raise ValueError("Every hybrid match must use the result ranking mode.")

        item_keys = [match.item_key for match in self.matches]
        if len(item_keys) != len(set(item_keys)):
            raise ValueError("Hybrid ranking match keys must be unique.")

        if self.corpus_total_count < self.eligible_document_count:
            raise ValueError("Corpus count must be at least the eligible count.")
        if self.eligible_document_count < self.candidate_count:
            raise ValueError("Eligible count must be at least the candidate count.")
        if self.candidate_count < len(self.matches):
            raise ValueError("Candidate count must be at least the returned count.")
        if any(match.score < 1 for match in self.matches):
            raise ValueError("Returned hybrid matches must have a positive score.")

        ranking = [_ranking_key(match) for match in self.matches]
        if ranking != sorted(ranking):
            raise ValueError("Hybrid matches must use deterministic ranking order.")
        return self

    @computed_field
    @property
    def returned_count(self) -> int:
        """Return the number of ranked matches included in this result."""

        return len(self.matches)

    @computed_field
    @property
    def truncated(self) -> bool:
        """Return whether the result bound omitted ranked candidates."""

        return self.candidate_count > self.returned_count


def _normalize_nfkc(value: str) -> str:
    return unicodedata.normalize("NFKC", value)


def _comparison_key(value: str) -> str:
    return _normalize_nfkc(value).casefold()


def _stable_unique[Value](
    values: Iterable[Value],
    *,
    key: Callable[[Value], str],
) -> list[Value]:
    unique: list[Value] = []
    seen: set[str] = set()
    for value in values:
        comparison = key(value)
        if comparison in seen:
            continue
        seen.add(comparison)
        unique.append(value)
    return unique


def _normalize_aware_datetime(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
    return value.astimezone(UTC)


def _weights_for_mode(
    mode: KnowledgeHybridRankingMode,
) -> dict[KnowledgeHybridSignal, int]:
    if mode == KnowledgeHybridRankingMode.HYBRID:
        return _HYBRID_SIGNAL_WEIGHTS
    return _DETERMINISTIC_SIGNAL_WEIGHTS


def _core_lexical_raw_score(
    lexical_match: KnowledgeLexicalMatch | None,
) -> int:
    if lexical_match is None:
        return 0
    return sum(
        evidence.points
        for evidence in lexical_match.evidence
        if evidence.kind not in _CORE_LEXICAL_EXCLUDED_KINDS
    )


def _path_signal_score(
    query: KnowledgeLexicalQuery,
    lexical_match: KnowledgeLexicalMatch | None,
) -> int:
    if not query.affected_paths:
        return 0

    evidence_by_path: dict[str, int] = {}
    if lexical_match is not None:
        for evidence in lexical_match.evidence:
            if evidence.kind == KnowledgeLexicalMatchKind.AFFECTED_PATH_EXACT:
                evidence_by_path[evidence.query_value] = 100
            elif (
                evidence.kind
                == KnowledgeLexicalMatchKind.AFFECTED_PATH_SHARED_DIRECTORY
            ):
                evidence_by_path.setdefault(evidence.query_value, 40)

    per_path_scores = [
        evidence_by_path.get(path, 0)
        for path in query.affected_paths
    ]
    count = len(per_path_scores)
    return (sum(per_path_scores) + count // 2) // count


def _label_signal_score(
    query: KnowledgeLexicalQuery,
    lexical_match: KnowledgeLexicalMatch | None,
) -> int:
    query_labels = _stable_unique(
        (_comparison_key(label) for label in query.labels),
        key=lambda label: label,
    )
    if not query_labels:
        return 0

    matched_labels: set[str] = set()
    if lexical_match is not None:
        matched_labels = {
            _comparison_key(evidence.query_value)
            for evidence in lexical_match.evidence
            if evidence.kind == KnowledgeLexicalMatchKind.LABEL
        }
    matched_count = sum(label in matched_labels for label in query_labels)
    total_labels = len(query_labels)
    return (matched_count * 100 + total_labels // 2) // total_labels


def _recency_signal_score(as_of: datetime, updated_at: datetime) -> int:
    age = as_of - updated_at
    buckets = (
        (30, 100),
        (90, 80),
        (180, 60),
        (365, 40),
        (730, 20),
    )
    for maximum_days, score in buckets:
        if age <= timedelta(days=maximum_days):
            return score
    return 0


def _make_contribution(
    signal: KnowledgeHybridSignal,
    signal_score: int,
    weight_percent: int,
    explanation: str,
) -> KnowledgeHybridSignalContribution:
    return KnowledgeHybridSignalContribution(
        signal=signal,
        signal_score=signal_score,
        weight_percent=weight_percent,
        weighted_basis_points=signal_score * weight_percent,
        explanation=explanation,
    )


def _build_contributions(
    mode: KnowledgeHybridRankingMode,
    query: KnowledgeLexicalQuery,
    document: KnowledgeLexicalDocument,
    lexical_match: KnowledgeLexicalMatch | None,
    semantic_similarity: KnowledgeSemanticSimilarity | None,
    as_of: datetime,
) -> list[KnowledgeHybridSignalContribution]:
    weights = _weights_for_mode(mode)
    signal_scores = {
        KnowledgeHybridSignal.LEXICAL: min(
            MAX_KNOWLEDGE_HYBRID_SCORE,
            _core_lexical_raw_score(lexical_match),
        ),
        KnowledgeHybridSignal.AFFECTED_PATH: _path_signal_score(
            query,
            lexical_match,
        ),
        KnowledgeHybridSignal.LABEL: _label_signal_score(query, lexical_match),
        KnowledgeHybridSignal.DECISION_SIGNIFICANCE: _SIGNIFICANCE_SCORES[
            document.decision_significance
        ],
        KnowledgeHybridSignal.RECENCY: _recency_signal_score(
            as_of,
            document.updated_at,
        ),
    }
    if semantic_similarity is not None:
        signal_scores[KnowledgeHybridSignal.SEMANTIC] = semantic_similarity.score

    explanations = {
        KnowledgeHybridSignal.LEXICAL: (
            "Core lexical relevance excludes path and label evidence to avoid "
            "double-counting."
        ),
        KnowledgeHybridSignal.SEMANTIC: (
            "Semantic similarity was supplied by the configured semantic scorer."
        ),
        KnowledgeHybridSignal.AFFECTED_PATH: (
            "Changed-path score reflects exact and shared-directory query-path "
            "coverage."
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
    return [
        _make_contribution(
            signal,
            signal_scores[signal],
            weight,
            explanations[signal],
        )
        for signal, weight in weights.items()
    ]


def _is_eligible(
    query: KnowledgeLexicalQuery,
    document: KnowledgeLexicalDocument,
) -> bool:
    return (
        document.reference.repository == query.repository
        and (
            not query.item_types
            or document.reference.item_type in query.item_types
        )
        and (
            not query.states
            or document.state in query.states
        )
    )


def _unique_lookup[Value](
    values: Iterable[Value],
    *,
    key: Callable[[Value], str],
    description: str,
) -> dict[str, Value]:
    lookup: dict[str, Value] = {}
    for value in values:
        item_key = key(value)
        if item_key in lookup:
            raise ValueError(f"{description} keys must be unique.")
        lookup[item_key] = value
    return lookup


def _validate_lexical_inputs(
    query: KnowledgeLexicalQuery,
    corpus: KnowledgeLexicalCorpus,
    lexical_result: KnowledgeLexicalSearchResult,
    eligible_documents: list[KnowledgeLexicalDocument],
) -> None:
    if query.repository != corpus.repository:
        raise ValueError("Hybrid query and corpus repositories must match.")
    if lexical_result.repository != corpus.repository:
        raise ValueError("Lexical result and corpus repositories must match.")
    if lexical_result.query != query:
        raise ValueError("Lexical result query must equal the hybrid query.")
    if lexical_result.corpus_total_count != corpus.total_count:
        raise ValueError("Lexical result corpus count does not match the corpus.")
    if lexical_result.eligible_document_count != len(eligible_documents):
        raise ValueError("Lexical result eligible count does not match local eligibility.")
    if lexical_result.options.minimum_score != 1:
        raise ValueError(
            "Hybrid ranking requires a lexical result produced with minimum_score=1."
        )
    if lexical_result.truncated:
        raise ValueError("Truncated lexical results cannot be hybrid ranked.")
    if lexical_result.matched_document_count != len(lexical_result.matches):
        raise ValueError("Lexical result matched count must equal returned matches.")


def _validate_semantic_scores(
    semantic_scores: list[KnowledgeSemanticSimilarity],
    repository: KnowledgeRepositoryRef,
    eligible_lookup: dict[str, KnowledgeLexicalDocument],
) -> dict[str, KnowledgeSemanticSimilarity]:
    semantic_lookup = _unique_lookup(
        semantic_scores,
        key=lambda similarity: similarity.item_key,
        description="Semantic similarity",
    )
    if any(
        similarity.reference.repository != repository
        for similarity in semantic_scores
    ):
        raise ValueError("Every semantic similarity must belong to the corpus repository.")

    semantic_keys = set(semantic_lookup)
    eligible_keys = set(eligible_lookup)
    if semantic_keys != eligible_keys:
        raise ValueError(
            "Semantic scores must exactly cover all eligible corpus documents."
        )
    return semantic_lookup


def _ranking_key(
    match: KnowledgeHybridRankedMatch,
) -> tuple[int, int, int, str]:
    semantic_score = (
        match.semantic_similarity.score
        if match.semantic_similarity is not None
        else 0
    )
    return (
        -match.total_weighted_basis_points,
        -match.core_lexical_raw_score,
        -semantic_score,
        match.item_key,
    )


def rank_knowledge_hybrid_corpus(
    query: KnowledgeLexicalQuery,
    corpus: KnowledgeLexicalCorpus,
    lexical_result: KnowledgeLexicalSearchResult,
    semantic_scores: list[KnowledgeSemanticSimilarity] | None = None,
    *,
    as_of: datetime,
    options: KnowledgeHybridRankingOptions | None = None,
) -> KnowledgeHybridRankingResult:
    """Fuse lexical and optional semantic relevance into hybrid rankings."""

    normalized_as_of = _normalize_aware_datetime(as_of, "as_of")
    eligible_documents = [
        document
        for document in corpus.documents
        if _is_eligible(query, document)
    ]
    _validate_lexical_inputs(
        query,
        corpus,
        lexical_result,
        eligible_documents,
    )
    if any(document.updated_at > normalized_as_of for document in eligible_documents):
        raise ValueError("as_of must not be earlier than an eligible document update.")

    eligible_lookup = _unique_lookup(
        eligible_documents,
        key=lambda document: document.item_key,
        description="Eligible document",
    )
    lexical_lookup = _unique_lookup(
        lexical_result.matches,
        key=lambda match: match.item_key,
        description="Lexical match",
    )
    if not set(lexical_lookup).issubset(eligible_lookup):
        raise ValueError("Every lexical match must identify an eligible corpus document.")

    if semantic_scores:
        mode = KnowledgeHybridRankingMode.HYBRID
        semantic_lookup = _validate_semantic_scores(
            semantic_scores,
            corpus.repository,
            eligible_lookup,
        )
        candidate_keys = list(eligible_lookup)
    else:
        mode = KnowledgeHybridRankingMode.DETERMINISTIC
        semantic_lookup = {}
        candidate_keys = list(lexical_lookup)

    effective_options = options or KnowledgeHybridRankingOptions()
    matches: list[KnowledgeHybridRankedMatch] = []
    for item_key in candidate_keys:
        document = eligible_lookup[item_key]
        lexical_match = lexical_lookup.get(item_key)
        semantic_similarity = semantic_lookup.get(item_key)
        if (
            lexical_match is not None
            and lexical_match.decision_significance
            != document.decision_significance
        ):
            raise ValueError(
                "Lexical match significance must equal document significance."
            )

        contributions = _build_contributions(
            mode,
            query,
            document,
            lexical_match,
            semantic_similarity,
            normalized_as_of,
        )
        scores = {
            contribution.signal: contribution.signal_score
            for contribution in contributions
        }
        if (
            scores[KnowledgeHybridSignal.LEXICAL] == 0
            and scores.get(KnowledgeHybridSignal.SEMANTIC, 0) == 0
        ):
            continue

        match = KnowledgeHybridRankedMatch(
            reference=document.reference,
            state=document.state,
            decision_significance=document.decision_significance,
            updated_at=document.updated_at,
            mode=mode,
            lexical_match=lexical_match,
            semantic_similarity=semantic_similarity,
            contributions=contributions,
        )
        if match.score >= effective_options.minimum_score:
            matches.append(match)

    matches.sort(key=_ranking_key)
    candidate_count = len(matches)
    return KnowledgeHybridRankingResult(
        repository=corpus.repository,
        query=query,
        mode=mode,
        as_of=normalized_as_of,
        corpus_total_count=corpus.total_count,
        eligible_document_count=len(eligible_documents),
        candidate_count=candidate_count,
        matches=matches[:effective_options.max_results],
    )
